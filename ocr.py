"""Nepal voter card number scanning via OCR (Tesseract).

Scope and honesty note: this is a RESEARCH SIMULATION. There is no live
government national-ID database to check the extracted number against, so
this module cannot and does not "verify" an ID card is genuine -- it only
extracts a printed ID number from a captured photo of a card (real or a
mock card printed for the study, per the consent form) well enough to use
as a second uniqueness key alongside face matching. Say this plainly in the
paper: OCR extraction accuracy is being evaluated, not government identity
verification.

What changed vs the previous version (and why scans kept failing):
  * Devanagari digits (०-९) can only be read if Tesseract's Nepali ("nep")
    or Hindi ("hin") language data is installed. A default `brew install
    tesseract` / `apt install tesseract-ocr` ships English only, so every
    card with Devanagari numerals failed with "no ID-number-shaped text".
    `language_status()` now reports this so the admin page can warn.
  * The old code took the FIRST 6-12 digit run in the text when the
    "मतदाता" label was not recognised -- which is often the date of birth
    (e.g. २०५१/०६/२४ -> 20510624). Dates are now rejected explicitly and
    candidates are ranked by proximity to the voter-number label.
  * A single Otsu-threshold pass was used. Photos with uneven light or a
    slight blur now get further passes (CLAHE greyscale, adaptive threshold,
    90/180/270 degree rotations) until a labelled number is found.
  * `pytesseract.get_languages()` spawned a Tesseract process on every scan;
    it is now cached.
"""

import os
import re
import shutil
import time
import unicodedata
from functools import lru_cache

import cv2
import numpy as np
import pytesseract

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_COMMON_TESSERACT_PATHS = (
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

# Voter numbers are accepted within this length range (digits only).
MIN_DIGITS = int(os.environ.get("EVOTING_ID_MIN_DIGITS", "6"))
MAX_DIGITS = int(os.environ.get("EVOTING_ID_MAX_DIGITS", "12"))

# Labels that precede the voter number on the card. Matched after Unicode
# NFC normalisation and with whitespace removed, so OCR spacing does not matter.
_LABELS = ("मतदातानं", "मतदातानम्बर", "मतदाताक्रमसंख्या", "मतदाता", "voterno", "voterid", "votersno", "voter")
# Labels of fields that also contain long digit runs and must NOT be taken.
_NEGATIVE_LABELS = ("जन्म", "मिति", "जारी", "नागरिकता", "date", "dob", "birth", "issue", "phone", "मोबाइल", "फोन")

_NUMBER_RE = re.compile(r"(?<!\d)\d[\d \t./:-]{3,26}\d(?!\d)")
_DATE_RE = re.compile(
    r"^\s*(\d{4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2}|\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{4})\s*$"
)


class OcrError(RuntimeError):
    pass


class TesseractUnavailable(OcrError):
    pass


class NoIdNumberFound(OcrError):
    pass


# ------------------------------------------------------------- tesseract --

def _find_tesseract():
    configured_path = os.environ.get("TESSERACT_CMD")
    if configured_path:
        if os.path.isfile(configured_path) and os.access(configured_path, os.X_OK):
            return configured_path
        raise TesseractUnavailable(
            f"TESSERACT_CMD points to an unavailable executable: {configured_path}"
        )
    path_candidate = shutil.which("tesseract")
    if path_candidate:
        return path_candidate
    for candidate in _COMMON_TESSERACT_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


@lru_cache(maxsize=1)
def _configure_tesseract():
    executable = _find_tesseract()
    if not executable:
        raise TesseractUnavailable(
            "Tesseract OCR is not installed or is not in PATH. Install it "
            "(macOS: `brew install tesseract tesseract-lang`; Ubuntu: "
            "`sudo apt install tesseract-ocr tesseract-ocr-nep`), then restart the application."
        )
    pytesseract.pytesseract.tesseract_cmd = executable
    return executable


@lru_cache(maxsize=1)
def _available_languages():
    _configure_tesseract()
    try:
        return frozenset(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractUnavailable("Tesseract OCR could not be started.") from exc


def _language_string():
    langs = _available_languages()
    chosen = [name for name in ("nep", "eng") if name in langs]
    if "nep" not in langs and "hin" in langs:
        chosen.insert(0, "hin")
    return "+".join(chosen) or "eng"


def language_status():
    """Returns (ok, message) for display on the admin dashboard / startup log."""
    try:
        langs = _available_languages()
    except TesseractUnavailable as exc:
        return False, str(exc)
    if "nep" in langs or "hin" in langs:
        return True, f"Tesseract ready ({_language_string()})"
    return False, (
        "Tesseract has no Nepali/Hindi language data, so Devanagari digits (०-९) on voter "
        "cards cannot be read. Install it: macOS `brew install tesseract-lang`, Ubuntu "
        "`sudo apt install tesseract-ocr-nep`, Windows: tick 'Nepali' in the installer."
    )


# --------------------------------------------------------- preprocessing --

def _base_gray(pil_image):
    gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    # Normalise size: Tesseract works best with ~30-40 px tall text. Cards
    # photographed at 1280 px wide need upscaling; huge uploads need shrinking.
    h, w = gray.shape[:2]
    target = 1800.0
    scale = target / float(max(h, w))
    if abs(scale - 1.0) > 0.1:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    return gray


def _variants(gray):
    """Yields (name, image) preprocessing variants, cheapest/most likely first."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield "clahe", clahe
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu", otsu
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15
    )
    yield "adaptive", adaptive


_ROTATIONS = (
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    (180, cv2.ROTATE_180),
)


# ------------------------------------------------------- candidate logic --

def _squash(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text)).lower()


def _candidates_from_text(raw_text):
    """Returns a list of (score, digits, line) candidates from one OCR result."""
    text = unicodedata.normalize("NFC", raw_text).translate(_DEVANAGARI_DIGITS)
    lines = [line for line in text.splitlines() if line.strip()]
    candidates = []
    for index, line in enumerate(lines):
        squashed = _squash(line)
        has_label = any(label in squashed for label in _LABELS)
        has_negative = any(label in squashed for label in _NEGATIVE_LABELS)
        prev_label = index > 0 and any(label in _squash(lines[index - 1]) for label in _LABELS)
        for match in _NUMBER_RE.finditer(line):
            chunk = match.group(0)
            if _DATE_RE.match(chunk):
                continue  # 2051/06/24, 24-06-1994 ...
            if chunk.count("/") >= 2 or (chunk.count(".") >= 2 and len(re.sub(r"\D", "", chunk)) <= 8):
                continue  # other date-like formats
            digits = re.sub(r"\D", "", chunk)
            if not (MIN_DIGITS <= len(digits) <= MAX_DIGITS):
                continue
            score = 0.0
            if has_label:
                score += 10
            elif prev_label:
                score += 6
            if has_negative and not has_label:
                score -= 8
            # a clean run of digits is more trustworthy than one stitched together
            if re.fullmatch(r"\d+", chunk.strip()):
                score += 2
            if 8 <= len(digits) <= 10:
                score += 1
            candidates.append((score, digits, line.strip()))
    return candidates


def _ocr(image, lang):
    try:
        return pytesseract.image_to_string(image, lang=lang, config="--oem 1 --psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractUnavailable(
            "Tesseract OCR could not be started. Check that `tesseract` is installed "
            "and available in PATH, then restart the application."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise OcrError(f"Tesseract failed: {exc}") from exc


def extract_id_number_detailed(pil_image):
    """Runs the multi-pass OCR and returns a dict:
        number, raw_text, score, labelled (bool), passes, variant, rotation, ms
    Raises NoIdNumberFound when no plausible number is found."""
    _configure_tesseract()
    lang = _language_string()
    started = time.perf_counter()
    base = _base_gray(pil_image)

    best = None
    best_text = ""
    passes = 0
    all_text = []
    for angle, rotate_code in _ROTATIONS:
        gray = base if rotate_code is None else cv2.rotate(base, rotate_code)
        # Upright: try every variant. Rotated: only the first (CLAHE) variant,
        # which keeps the worst case at 6 Tesseract passes instead of 12.
        variants = _variants(gray) if angle == 0 else [next(_variants(gray))]
        for variant_name, image in variants:
            raw = _ocr(image, lang)
            passes += 1
            all_text.append(raw)
            for score, digits, line in _candidates_from_text(raw):
                if best is None or score > best[0]:
                    best = (score, digits, line, variant_name, angle)
                    best_text = raw
            if best is not None and best[0] >= 10:
                break  # a number on the labelled line: good enough
        if best is not None and best[0] >= 6:
            break  # found something labelled in this orientation; don't try others

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if best is None or best[0] < 0:
        raise NoIdNumberFound(
            "no voter number was found on the card - hold the card flat, fill the frame, avoid glare"
        )
    score, digits, _line, variant_name, angle = best
    return {
        "number": digits,
        "raw_text": best_text.strip(),
        "score": score,
        "labelled": score >= 6,
        "passes": passes,
        "variant": variant_name,
        "rotation": angle,
        "ms": elapsed_ms,
        "lang": lang,
    }


def extract_id_number(pil_image):
    """Backward-compatible API: returns (number, raw_text)."""
    result = extract_id_number_detailed(pil_image)
    return result["number"], result["raw_text"]


if __name__ == "__main__":
    ok, message = language_status()
    print(("OK: " if ok else "WARNING: ") + message)
