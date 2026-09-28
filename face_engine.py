"""Face detection + face recognition engine (YuNet detector + SFace recognizer).

Why this replaced the Haar-cascade + LBP-histogram engine
---------------------------------------------------------
The previous engine (Haar cascade detection, 8x8-grid LBP histograms, chi-square
similarity, provisional threshold 0.08) was measured against real photos and
against this project's own audit log before being replaced:

  * Only 2 of 72 face-verification attempts in the shipped database succeeded
    (27 "does not match", 21 "no face detected", 14 "more than one face").
  * LBP texture histograms change strongly with lighting, head pose and camera
    exposure, so genuine and impostor scores overlap -- a single fixed
    threshold either rejects real voters or accepts other people.
  * The Haar cascade with minNeighbors=6 on a full 1280x720 frame both misses
    real faces (dim light, slight turn) and fires on background texture, which
    the old code reported as "more than one face".
  * The duplicate-face check used a normalised L1 distance < 0.02 on those
    histograms, which two photos of the same person essentially never reach,
    so one person could register several times.

This version uses two models from the OpenCV Model Zoo, both run through
OpenCV's own DNN module (no PyTorch/TensorFlow, CPU-only, fully offline once
the two .onnx files are in ./models):

  1. YuNet (Wu et al., 2023, "YuNet: A Tiny Millisecond-level Face Detector",
     Machine Intelligence Research) -- a ~230 KB CNN face detector that also
     returns 5 facial landmarks.
  2. SFace (Zhong et al., 2021, "SFace: Sigmoid-Constrained Hypersphere Loss
     for Robust Face Recognition", IEEE Transactions on Image Processing) --
     produces a 128-d embedding from a landmark-aligned 112x112 face crop.
     Identity is compared with cosine similarity. OpenCV's published
     operating point for SFace is cosine >= 0.363 (see the OpenCV
     FaceRecognizerSF tutorial); that is the default here and MUST be
     re-calibrated on your own volunteers with evaluate_accuracy.py before
     you report FAR/FRR (report the threshold you actually used and how).

Public API kept compatible with app.py:
  FaceEngine().extract(pil_image) -> sample with .embedding, .bbox, .frame_shape
  build_registration_template(samples) -> (template, template, consistency)
  cosine_similarity(a, b) -> float in [-1, 1] (true cosine similarity now)
  SFACE_MATCH_THRESHOLD
"""

import base64
import hashlib
import io
import os
import ssl
import threading
import time
import urllib.request

import certifi
import cv2
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
MODEL_SHA256 = {
    YUNET_FILE: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    SFACE_FILE: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}
MODEL_URLS = {
    YUNET_FILE: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + YUNET_FILE,
    SFACE_FILE: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/" + SFACE_FILE,
}

EMBEDDING_DIM = 128

# Cosine-similarity threshold for "same person". 0.363 is OpenCV's published
# operating point for SFace. Override with the EVOTING_FACE_THRESHOLD env var
# after calibrating on your own data (evaluate_accuracy.py prints the EER
# threshold and FAR/FRR at several operating points).
SFACE_MATCH_THRESHOLD = float(os.environ.get("EVOTING_FACE_THRESHOLD", "0.363"))

# A new registration is treated as a duplicate of an existing voter when the
# two templates are at least this similar. It is deliberately the same value
# as the verification threshold: if the face would pass verification as that
# voter, it must not be allowed to register a second identity.
DUPLICATE_FACE_THRESHOLD = float(os.environ.get("EVOTING_DUPLICATE_THRESHOLD", str(SFACE_MATCH_THRESHOLD)))

# Detection settings.
DETECT_MAX_SIDE = 640          # frames are downscaled to this for detection (speed)
DETECT_SCORE_THRESHOLD = 0.6   # YuNet confidence
MIN_FACE_FRACTION = 0.07       # face width must be >= 7% of the frame width ...
MIN_FACE_PIXELS = 64           # ... and >= 64 px (SFace aligns to 112x112)
SECOND_FACE_RATIO = 0.35       # a 2nd face >= 35% of the main face's area = a real 2nd person
MIN_BRIGHTNESS = 45            # mean grey level of the face crop (0-255)
MAX_BRIGHTNESS = 225
MIN_SHARPNESS = 12.0           # variance of Laplacian on the 112x112 aligned crop


class FaceError(RuntimeError):
    pass


class FaceModelsMissing(FaceError):
    pass


class LowQualityImage(FaceError):
    pass


class MultipleFacesDetected(FaceError):
    pass


class NoFaceDetected(FaceError):
    pass


class LegacyTemplate(FaceError):
    """The stored template was created by the old LBP engine and cannot be
    compared with an SFace embedding. The voter must re-register."""


# ------------------------------------------------------------------ models --

def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(download=True):
    """Checks both ONNX files exist and match their published SHA-256; downloads
    them from the OpenCV Model Zoo if missing and download=True. Also warms the
    engine up so the first voter does not wait for model loading."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, expected in MODEL_SHA256.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path) and download:
            print(f"Downloading {name} ...")
            tmp = path + ".part"
            context = ssl.create_default_context(cafile=certifi.where())
            try:
                with urllib.request.urlopen(MODEL_URLS[name], context=context, timeout=60) as response:
                    with open(tmp, "wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                os.replace(tmp, path)
            except Exception:
                try:
                    os.remove(tmp)
                except FileNotFoundError:
                    pass
                raise
        if not os.path.exists(path):
            raise FaceModelsMissing(f"missing model file models/{name}")
        actual = _sha256(path)
        if actual != expected:
            raise FaceModelsMissing(
                f"models/{name} is corrupted (sha256 {actual[:12]}..., expected {expected[:12]}...). "
                "Delete it and run `python app.py setup` again."
            )
    get_engine()
    return True


# ------------------------------------------------------------------ images --

def decode_camera_image(value):
    if value is None or not str(value).strip():
        raise ValueError("no image was captured - press Capture before submitting")

    encoded = str(value)
    if encoded.startswith("data:image"):
        encoded = encoded.split(",", 1)[1]

    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError("camera image was not valid base64") from exc

    try:
        image = Image.open(io.BytesIO(payload))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError("camera image could not be decoded") from exc

    if image.width < 32 or image.height < 32:
        raise ValueError("camera image was too small")
    return image


def _to_bgr(pil_image):
    return cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------ engine --

class _Sample:
    __slots__ = ("embedding", "bbox", "frame_shape", "det_score", "timings_ms")

    def __init__(self, embedding, bbox=None, frame_shape=None, det_score=None, timings_ms=None):
        self.embedding = embedding
        self.bbox = bbox
        self.frame_shape = frame_shape
        self.det_score = det_score
        self.timings_ms = timings_ms or {}


class FaceEngine:
    """Thread-safe wrapper. OpenCV DNN nets are not safe to call concurrently,
    and waitress serves requests on several threads, so inference is
    serialised with a lock (each call takes only tens of milliseconds)."""

    def __init__(self):
        yunet_path = os.path.join(MODELS_DIR, YUNET_FILE)
        sface_path = os.path.join(MODELS_DIR, SFACE_FILE)
        if not (os.path.exists(yunet_path) and os.path.exists(sface_path)):
            raise FaceModelsMissing("face models not found in ./models - run `python app.py setup`")
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise FaceModelsMissing(
                f"OpenCV {cv2.__version__} lacks FaceDetectorYN/FaceRecognizerSF - install opencv-contrib-python>=4.8"
            )
        self._detector = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320), DETECT_SCORE_THRESHOLD, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
        self._lock = threading.Lock()

    def _detect(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / float(max(h, w)))
        small = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 15), dtype=np.float32)
        faces = faces.copy()
        faces[:, :14] /= scale  # back to full-resolution coordinates
        return faces

    def extract(self, image):
        if image is None:
            raise NoFaceDetected("no face image was captured")
        t0 = time.perf_counter()
        bgr = _to_bgr(image)
        frame_h, frame_w = bgr.shape[:2]

        with self._lock:
            faces = self._detect(bgr)
            t1 = time.perf_counter()
            if len(faces) == 0:
                raise NoFaceDetected(
                    "no face was detected - face the camera directly, move closer and make sure your face is well lit"
                )

            areas = faces[:, 2] * faces[:, 3]
            order = np.argsort(-areas)
            faces = faces[order]
            areas = areas[order]
            if len(faces) > 1 and areas[1] >= SECOND_FACE_RATIO * areas[0]:
                raise MultipleFacesDetected(
                    "more than one person is visible - only the voter should be in front of the camera"
                )
            face = faces[0]
            x, y, w, h = [float(v) for v in face[:4]]
            if w < max(MIN_FACE_PIXELS, MIN_FACE_FRACTION * frame_w):
                raise LowQualityImage("your face is too far from the camera - please move closer")

            aligned = self._recognizer.alignCrop(bgr, face)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            if brightness < MIN_BRIGHTNESS:
                raise LowQualityImage("the image is too dark - add light in front of your face")
            if brightness > MAX_BRIGHTNESS:
                raise LowQualityImage("the image is over-exposed - avoid strong light directly on the camera")
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < MIN_SHARPNESS:
                raise LowQualityImage("the image is blurry - hold still and capture again")

            feature = self._recognizer.feature(aligned).flatten().astype(np.float64)
            t2 = time.perf_counter()

        norm = np.linalg.norm(feature)
        if norm == 0:
            raise LowQualityImage("could not compute a face descriptor - please capture again")
        embedding = feature / norm
        bbox = (int(round(x)), int(round(y)), int(round(w)), int(round(h)))
        return _Sample(
            embedding=embedding,
            bbox=bbox,
            frame_shape=(frame_h, frame_w),
            det_score=float(face[14]),
            timings_ms={"detect": (t1 - t0) * 1000.0, "embed": (t2 - t1) * 1000.0},
        )


_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = FaceEngine()
    return _ENGINE


# -------------------------------------------------------------- templates --

def _as_unit_vector(value):
    vector = np.asarray(value, dtype=np.float64).ravel()
    if vector.shape[0] != EMBEDDING_DIM:
        raise LegacyTemplate(
            "this voter was registered with the old face engine - please register again"
        )
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise LegacyTemplate("stored face template is empty - please register again")
    return vector / norm


def cosine_similarity(a, b):
    """True cosine similarity between two SFace embeddings, in [-1, 1].
    Higher = more similar. Compare against SFACE_MATCH_THRESHOLD."""
    return float(np.dot(_as_unit_vector(a), _as_unit_vector(b)))


def build_registration_template(samples):
    """Averages the enrollment embeddings (then re-normalises) -- a standard
    multi-shot enrollment template. Returns (template, template, consistency)
    where consistency is the MINIMUM pairwise cosine similarity between the
    enrollment captures. If it is below the match threshold the captures do
    not look like the same person (or one capture was bad) and the caller
    should reject the enrollment."""
    if not samples:
        raise NoFaceDetected("no samples were provided for registration")
    stacked = np.stack([_as_unit_vector(sample.embedding) for sample in samples])
    mean = stacked.mean(axis=0)
    template = (mean / np.linalg.norm(mean)).tolist()
    if len(stacked) > 1:
        sims = [
            float(np.dot(stacked[i], stacked[j]))
            for i in range(len(stacked))
            for j in range(i + 1, len(stacked))
        ]
        consistency = min(sims)
    else:
        consistency = 1.0
    return template, template, consistency


def verify_liveness(center, challenge, action):
    """Motion-based liveness check (kept for completeness; NOT wired into the
    current single-capture verification flow -- if you describe liveness in the
    paper, re-enable a two-capture challenge in app.py first).

    Honest scope: this checks that the face bounding box moved sideways
    ("turn") or grew ("closer") between two captures. It is a presentation-
    attack DETERRENT, not certified anti-spoofing (ISO/IEC 30107-3)."""
    if not (center.bbox and challenge.bbox and center.frame_shape and challenge.frame_shape):
        return False
    cx, _cy, cw, ch = center.bbox
    xx, _xy, xw, xh = challenge.bbox
    _frame_h, frame_w = center.frame_shape
    horizontal_shift = abs((xx + xw / 2.0) - (cx + cw / 2.0)) / frame_w
    growth_ratio = (xw * xh) / float(cw * ch) if cw * ch else 0.0
    if action == "turn":
        return horizontal_shift >= 0.04
    return growth_ratio >= 1.15


if __name__ == "__main__":
    ensure_models(download=True)
    print(f"face_engine ready: YuNet + SFace, OpenCV {cv2.__version__}, threshold {SFACE_MATCH_THRESHOLD}")
