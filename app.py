"""Local-only e-voting research simulation with camera enrollment, ID-card
OCR, deep(-ish, see face_engine.py) face verification, a tamper-evident
anonymous vote ledger, and a password-protected admin panel."""

import argparse
import base64
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta

from flask import Flask, abort, flash, g, redirect, render_template_string, request, session, url_for
from markupsafe import Markup, escape
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

import database
from database import AlreadyVoted, DuplicateNationalId, ElectionClosed
from face_engine import (
    SFACE_MATCH_THRESHOLD,
    FaceError,
    FaceModelsMissing,
    LowQualityImage,
    MultipleFacesDetected,
    NoFaceDetected,
    build_registration_template,
    cosine_similarity,
    decode_camera_image,
    ensure_models,
)
from ocr import NoIdNumberFound, OcrError, TesseractUnavailable, extract_id_number
from security import SESSION_KEY, format_registration_number, normalize_registration_number

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SYMBOLS_DIR = os.path.join(STATIC_DIR, "symbols")
RESULTS_VISIBLE = os.environ.get("EVOTING_SHOW_RESULTS", "0") == "1"
CHALLENGE_TTL_SECONDS = 120

os.umask(0o077)
os.makedirs(SYMBOLS_DIR, exist_ok=True)

DEMO_PARTIES = [
    ("Party A", "Candidate 1", "sun.png"),
    ("Party B", "Candidate 2", "tree.png"),
    ("Party C", "Candidate 3", "umbrella.png"),
    ("Party D", "Candidate 4", "star.png"),
    ("Party E", "Candidate 5", "plough.png"),
    ("Independent", "Candidate 6", "hand.png"),
]

_FACE_ENGINE = None
_FACE_ENGINE_LOCK = threading.Lock()


def get_face_engine():
    global _FACE_ENGINE
    if _FACE_ENGINE is None:
        with _FACE_ENGINE_LOCK:
            if _FACE_ENGINE is None:
                from face_engine import FaceEngine
                _FACE_ENGINE = FaceEngine()
    return _FACE_ENGINE


def generate_symbols():
    size = 200

    def canvas():
        return Image.new("RGBA", (size, size), (255, 255, 255, 0))

    def save(image, name):
        path = os.path.join(SYMBOLS_DIR, name)
        if not os.path.exists(path):
            image.save(path)
            os.chmod(path, 0o644)

    import math

    image = canvas(); draw = ImageDraw.Draw(image)
    cx, cy, radius = size // 2, size // 2, 45
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 165, 0))
    for index in range(12):
        angle = math.radians(index * 30)
        start = (cx + (radius + 10) * math.cos(angle), cy + (radius + 10) * math.sin(angle))
        end = (cx + (radius + 35) * math.cos(angle), cy + (radius + 35) * math.sin(angle))
        draw.line([start, end], fill=(255, 165, 0), width=8)
    save(image, "sun.png")

    image = canvas(); draw = ImageDraw.Draw(image)
    draw.rectangle([size/2-8, size/2+20, size/2+8, size/2+70], fill=(101, 67, 33))
    draw.ellipse([size/2-55, size/2-70, size/2+55, size/2+40], fill=(34, 139, 34))
    save(image, "tree.png")

    image = canvas(); draw = ImageDraw.Draw(image)
    draw.pieslice([size/2-60, size/2-60, size/2+60, size/2+60], 180, 360, fill=(178, 34, 34))
    draw.line([size/2, size/2, size/2, size/2+70], fill=(80, 80, 80), width=6)
    save(image, "umbrella.png")

    image = canvas(); draw = ImageDraw.Draw(image)
    points = []
    for index in range(10):
        radius = 65 if index % 2 == 0 else 28
        angle = math.radians(index * 36 - 90)
        points.append((size/2 + radius * math.cos(angle), size/2 + radius * math.sin(angle)))
    draw.polygon(points, fill=(0, 56, 147))
    save(image, "star.png")

    image = canvas(); draw = ImageDraw.Draw(image)
    draw.line([size/2-60, size/2+50, size/2+60, size/2-50], fill=(90, 60, 20), width=14)
    draw.polygon([(size/2+50, size/2-60), (size/2+75, size/2-35), (size/2+45, size/2-25)], fill=(120, 120, 120))
    save(image, "plough.png")

    image = canvas(); draw = ImageDraw.Draw(image)
    draw.ellipse([size/2-45, size/2-20, size/2+45, size/2+60], fill=(230, 184, 138))
    for offset in (-35, -15, 5, 25):
        draw.rounded_rectangle([size/2+offset-8, size/2-70, size/2+offset+8, size/2-15], radius=8, fill=(230, 184, 138))
    save(image, "hand.png")


def seed_demo_parties():
    if database.count_parties():
        return
    for order, (name, candidate, symbol) in enumerate(DEMO_PARTIES, start=1):
        database.create_party(name, candidate, symbol, order)


def initialize_admin_password():
    """Set the admin password from EVOTING_ADMIN_PASSWORD, defaulting to ADMIN."""
    password = os.environ.get("EVOTING_ADMIN_PASSWORD", "ADMIN")
    database.set_setting("admin_password_hash", generate_password_hash(password))


app = Flask(__name__, static_folder=STATIC_DIR)
app.config.update(
    SECRET_KEY=base64.urlsafe_b64encode(SESSION_KEY).decode("ascii"),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_NAME="evoting_session",
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=10),
    TRUSTED_HOSTS=["127.0.0.1", "localhost"],
)

RATE_BUCKETS = defaultdict(deque)
RATE_LOCK = threading.Lock()

STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f5f6f8;margin:0;color:#202124}.top{background:#b5121b;color:#fff;padding:14px 22px;display:flex;justify-content:space-between}.top a{color:#fff}.wrap{max-width:760px;margin:28px auto;padding:0 16px}.card{background:#fff;border:1px solid #e1e4e8;border-radius:14px;padding:24px;box-shadow:0 4px 18px rgba(0,0,0,.04)}.center{text-align:center}.btn{display:inline-block;padding:11px 18px;border-radius:8px;border:0;font-weight:650;cursor:pointer;text-decoration:none;margin:5px;background:#b5121b;color:#fff}.btn.alt{background:#eceff1;color:#222}.btn.danger{background:#7a1114}.btn:disabled{opacity:.5;cursor:not-allowed}.flash{background:#fff3cd;border:1px solid #ffe58f;color:#765b00;padding:10px 14px;border-radius:8px;margin-bottom:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-top:14px}.opt{background:#fff;border:2px solid #e2e4e8;border-radius:10px;padding:14px;cursor:pointer;text-align:center}.opt:hover{border-color:#b5121b}.opt img{width:60px;height:60px}.row{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}.camera{margin:18px 0;padding:14px;border:1px solid #dfe3e7;border-radius:12px;background:#fafbfc}.camera video{display:block;width:100%;max-height:420px;border-radius:10px;background:#111;object-fit:cover}.preview{margin-top:10px;width:120px;border-radius:8px}input[type=text],input[type=password]{width:100%;padding:11px;border:1px solid #ccd1d5;border-radius:7px;margin:6px 0 14px;box-sizing:border-box}.code{font:700 1.6rem ui-monospace,SFMono-Regular,monospace;letter-spacing:.08em;background:#f4f5f7;padding:12px;border-radius:8px}.muted{color:#666;font-size:.93rem}.good{background:#e9f8ee;border:1px solid #b9e6c6;padding:12px;border-radius:8px}.bad{background:#fdecea;border:1px solid #f3b3ac;padding:12px;border-radius:8px}.bar{width:100%;height:9px;background:#eee;border-radius:99px;overflow:hidden}.bar span{display:block;height:100%;background:#003893}table{width:100%;border-collapse:collapse;margin-top:10px}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid #eee;font-size:.92rem}.pill{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.82rem;font-weight:650}.pill.open{background:#e9f8ee;color:#1a7a3c}.pill.closed{background:#fdecea;color:#a3231b}
"""

BASE_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }}</title><style nonce="{{ nonce }}">{{ style|safe }}</style></head><body><div class="top"><b>Nepal E-Voting Research Simulation</b><a href="{{ results_url }}">Results</a></div><div class="wrap">{% with messages=get_flashed_messages() %}{% if messages %}<div class="flash">{% for message in messages %}<p>{{ message }}</p>{% endfor %}</div>{% endif %}{% endwith %}{{ body|safe }}</div></body></html>"""


def csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def require_csrf():
    expected = session.get("csrf_token", "")
    supplied = request.form.get("csrf_token", "")
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        abort(400)


def rate_limited(bucket, limit, window_seconds):
    key = f"{bucket}:{request.remote_addr or 'local'}"
    now = time.monotonic()
    with RATE_LOCK:
        entries = RATE_BUCKETS[key]
        while entries and now - entries[0] > window_seconds:
            entries.popleft()
        if len(entries) >= limit:
            return True
        entries.append(now)
    return False


def page(title, body):
    return render_template_string(
        BASE_TEMPLATE,
        title=title,
        body=Markup(body),
        style=STYLE,
        nonce=g.csp_nonce,
        results_url=url_for("results"),
    )


def camera_box(fields, prompts):
    hidden = "".join(f'<input type="hidden" name="{field}" data-camera-field>' for field in fields)
    prompt_text = "|".join(prompts)
    return f"""
    <div class="camera" data-camera-box data-prompts="{escape(prompt_text)}">
      <video autoplay playsinline muted></video><canvas hidden></canvas>
      {hidden}
      <div class="row"><button type="button" class="btn alt" data-start-camera>Open Camera</button><button type="button" class="btn" data-capture-frame disabled>Capture</button></div>
      <p class="muted" data-camera-status>Open the camera, keep one face visible, and follow each capture instruction.</p>
      <img class="preview" data-camera-preview hidden alt="Latest captured frame">
    </div><script src="{url_for('static', filename='camera.js')}" defer></script>"""


def parse_party_id(value):
    try:
        party_id = int(value)
    except (TypeError, ValueError):
        return None
    return party_id if party_id > 0 else None


@app.before_request
def prepare_request():
    g.csp_nonce = secrets.token_urlsafe(18)


@app.after_request
def security_headers(response):
    nonce = g.get("csp_nonce", "")
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; "
        f"style-src 'nonce-{nonce}'; "
        "script-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
        "connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.route("/")
def index():
    token = csrf_token()
    election_open = database.is_election_open()
    registered_voters = database.count_voters()
    status_pill = (
        '<span class="pill open">Election Open</span>' if election_open
        else '<span class="pill closed">Election Closed</span>'
    )
    action_buttons = (
        f"""<div class="row"><a class="btn" href="{url_for('scan_id')}">Register Simulation Voter</a>
        <form method="post" action="{url_for('start_voting')}"><input type="hidden" name="csrf_token" value="{token}"><button class="btn alt" type="submit">Start Voting</button></form></div>"""
        if election_open else
        '<p class="muted">Registration and voting are currently closed by the election administrator.</p>'
    )
    return page("E-Voting Simulation", f"""
    <div class="card center"><h1>E-Voting Simulation</h1>{status_pill}
    <p>Registered voters: <b>{registered_voters}</b></p>
    <p>Scan a voter card, register with one face capture, receive a generated registration number, then use it to test the voting workflow.</p>
    {action_buttons}
    <p class="muted">Research prototype only. It is not a production public-election system. <a href="{url_for('admin_login')}">Admin</a></p></div>""")


# ------------------------------------------------------------- registration --

@app.route("/scan-id", methods=["GET", "POST"])
def scan_id():
    if not database.is_election_open():
        flash("Registration is currently closed.")
        return redirect(url_for("index"))

    if request.method == "GET":
        token = csrf_token()
        camera = camera_box(["id_card_image"], ["Hold the voter card flat and steady, then capture"])
        return page("Scan ID Card", f"""
        <div class="card"><h2>Step 0 - Scan Voter Card</h2>
        <p>Capture the voter card clearly. The system reads the मतदाता नं (voter number), including Devanagari digits, and stores it with the voter record.</p>
        <form method="post" enctype="multipart/form-data"><input type="hidden" name="csrf_token" value="{token}">{camera}
        <div class="card"><label for="voter_card_file"><b>Or upload a voter card image</b></label>
        <p class="muted">Use a clear JPG, PNG, or WEBP photo if the camera image is blurry.</p>
        <input id="voter_card_file" type="file" name="voter_card_file" accept="image/jpeg,image/png,image/webp"></div>
        <button class="btn" type="submit">Scan Card</button></form></div>""")

    require_csrf()
    if rate_limited("scan-id", 15, 300):
        abort(429)

    try:
        upload = request.files.get("voter_card_file")
        if upload and upload.filename:
            with Image.open(upload.stream) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Choose a JPG, PNG, or WEBP voter card image.")
                if max(source.size) > 8192:
                    raise ValueError("The uploaded image must be no larger than 8192 pixels per side.")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            image = decode_camera_image(request.form.get("id_card_image"))
        voter_number, _raw_text = extract_id_number(image)
    except TesseractUnavailable as exc:
        database.log_event("id_scan_failed", str(exc))
        flash(f"ID scanning is unavailable: {exc}")
        return redirect(url_for("scan_id"))
    except (ValueError, OcrError, NoIdNumberFound, Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        database.log_event("id_scan_failed", str(exc))
        flash(f"Could not read the voter card image: {exc}. Please use a clear JPG, PNG, or WEBP image and retry.")
        return redirect(url_for("scan_id"))

    session["pending_voter_number"] = voter_number
    session["pending_voter_number_masked"] = f"***{voter_number[-4:]}"
    database.log_event("id_scan_ok", None)
    return redirect(url_for("register"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if not database.is_election_open():
        flash("Registration is currently closed.")
        return redirect(url_for("index"))
    if not session.get("pending_voter_number"):
        flash("Please scan your voter card first.")
        return redirect(url_for("scan_id"))

    if request.method == "GET":
        token = csrf_token()
        camera = camera_box(["face_image"], ["Look straight at the camera and capture once"])
        masked = escape(session.get("pending_voter_number_masked", ""))
        return page("Register", f"""
        <div class="card"><h2>Step 1 - Simulation Registration</h2>
        <p class="muted">Voter number scanned: ending in {masked}</p>
        <p>Enter a display name and capture your face once. The system will generate the registration number automatically.</p>
        <form method="post"><input type="hidden" name="csrf_token" value="{token}">
        <label>Display name</label><input type="text" name="name" maxlength="100" autocomplete="name" required>
        {camera}<button class="btn" type="submit">Create Registration</button></form></div>""")

    require_csrf()
    if rate_limited("register", 10, 300):
        abort(429)

    voter_number = session.get("pending_voter_number")
    if not voter_number:
        flash("Please scan your voter card first.")
        return redirect(url_for("scan_id"))

    try:
        engine = get_face_engine()
        image = decode_camera_image(request.form.get("face_image"))
        samples = [engine.extract(image)]
        sface_template, lbp_template, _consistency = build_registration_template(samples)
        duplicate = database.find_duplicate_face(sface_template)
        if duplicate:
            database.log_event("registration_duplicate_face", None)
            flash("This face appears to be already registered in the simulation.")
            return redirect(url_for("register"))
        number = database.create_voter(request.form.get("name"), sface_template, lbp_template, voter_number)
    except FaceModelsMissing:
        flash("Face models are not installed. Run: python app.py setup")
        return redirect(url_for("register"))
    except DuplicateNationalId:
        database.log_event("registration_duplicate_id", None)
        flash("This voter card number has already been registered.")
        return redirect(url_for("scan_id"))
    except ElectionClosed:
        flash("Registration closed while you were completing this form.")
        return redirect(url_for("index"))
    except (ValueError, FaceError, NoFaceDetected, MultipleFacesDetected, LowQualityImage) as exc:
        flash(f"Registration could not be completed: {exc}")
        return redirect(url_for("register"))

    session.pop("pending_voter_number", None)
    session.pop("pending_voter_number_masked", None)
    session["new_registration_number"] = number
    database.log_event("voter_registered", None)
    return redirect(url_for("registered"))


@app.route("/registered")
def registered():
    number = session.pop("new_registration_number", None)
    if not number:
        return redirect(url_for("index"))
    formatted = format_registration_number(number)
    return page("Registered", f"""
    <div class="card center"><h2>Registration Created</h2><p>Use this simulation registration number when voting:</p>
    <div class="code">{escape(formatted)}</div><p class="muted">Write it down. Only its protected lookup hash is stored.</p>
    <a class="btn" href="{url_for('identify')}">Continue to Voting</a></div>""")


# ------------------------------------------------------------------- voting --

@app.route("/start", methods=["POST"])
def start_voting():
    require_csrf()
    if not database.is_election_open():
        flash("Voting is currently closed.")
        return redirect(url_for("index"))
    session.clear()
    session.permanent = True
    csrf_token()
    return redirect(url_for("identify"))


@app.route("/identify", methods=["GET", "POST"])
def identify():
    if not database.is_election_open():
        flash("Voting is currently closed.")
        return redirect(url_for("index"))

    if request.method == "GET":
        token = csrf_token()
        return page("Enter Registration", f"""
        <div class="card"><h2>Step 1 - Registration Number</h2>
        <p>Enter the 12-digit simulation number generated during registration.</p>
        <form method="post"><input type="hidden" name="csrf_token" value="{token}">
        <input type="text" name="registration_number" inputmode="numeric" autocomplete="off" placeholder="1234-5678-9012" required>
        <button class="btn" type="submit">Continue</button></form></div>""")

    require_csrf()
    if rate_limited("identify", 20, 300):
        abort(429)
    try:
        number = normalize_registration_number(request.form.get("registration_number"))
        voter = database.get_voter_by_registration(number)
    except ValueError:
        voter = None

    if not voter or voter["has_voted"]:
        database.log_event("identify_failed", None)
        flash("Registration could not be verified for voting.")
        return redirect(url_for("identify"))

    session["voter_id"] = voter["id"]
    session.pop("face_verified", None)
    session.pop("party_id", None)
    return redirect(url_for("verify_face_route"))


@app.route("/verify-face", methods=["GET", "POST"])
def verify_face_route():
    voter_id = session.get("voter_id")
    voter = database.get_voter_by_id(voter_id) if voter_id else None
    if not voter or voter["has_voted"]:
        return redirect(url_for("index"))

    if request.method == "GET":
        token = csrf_token()
        camera = camera_box(["face_image"], ["Look straight at the camera and capture once"])
        return page("Verify Face", f"""
        <div class="card"><h2>Step 2 - Face Verification</h2>
        <p>Look straight at the camera and capture your face once to verify your identity.</p>
        <form method="post"><input type="hidden" name="csrf_token" value="{token}">{camera}
        <button class="btn" type="submit">Verify &amp; Continue</button></form></div>""")

    require_csrf()
    if rate_limited("verify-face", 10, 300):
        abort(429)

    try:
        engine = get_face_engine()
        center = engine.extract(decode_camera_image(request.form.get("face_image")))
        center_score = cosine_similarity(center.embedding, voter["sface_template"])
        if center_score < SFACE_MATCH_THRESHOLD:
            raise FaceError("face does not match the registered template")
    except (ValueError, FaceError, NoFaceDetected, MultipleFacesDetected, LowQualityImage) as exc:
        database.log_event("face_verification_failed", str(exc))
        flash(f"Face verification failed: {exc}. Please try again in good light.")
        return redirect(url_for("verify_face_route"))

    session["face_verified"] = True
    database.log_event("face_verification_ok", None)
    return redirect(url_for("ballot"))


@app.route("/ballot")
def ballot():
    voter_id = session.get("voter_id")
    if not voter_id or not session.get("face_verified") or database.has_voted(voter_id):
        return redirect(url_for("index"))

    token = csrf_token()
    options = []
    for party in database.list_parties():
        name = escape(party["name"])
        candidate = escape(party["candidate_name"] or "")
        symbol_name = os.path.basename(party["symbol_filename"])
        symbol_url = escape(url_for("static", filename=f"symbols/{symbol_name}"))
        options.append(
            f'<button name="party_id" value="{party["id"]}" class="opt" type="submit">'
            f'<img src="{symbol_url}" alt=""><br><b>{name}</b><br><span class="muted">{candidate}</span></button>'
        )

    return page("Ballot", f"""
    <div class="card"><h2>Step 3 - Select an Option</h2><form method="post" action="{url_for('select_party')}">
    <input type="hidden" name="csrf_token" value="{token}"><div class="grid">{''.join(options)}</div></form></div>""")


@app.route("/select-party", methods=["POST"])
def select_party():
    require_csrf()
    voter_id = session.get("voter_id")
    if not voter_id or not session.get("face_verified") or database.has_voted(voter_id):
        return redirect(url_for("index"))
    party_id = parse_party_id(request.form.get("party_id"))
    party = database.get_party(party_id) if party_id else None
    if not party:
        flash("Invalid ballot selection.")
        return redirect(url_for("ballot"))
    session["party_id"] = party_id
    return redirect(url_for("confirm_vote"))


@app.route("/confirm", methods=["GET", "POST"])
def confirm_vote():
    voter_id = session.get("voter_id")
    party_id = session.get("party_id")
    if not voter_id or not party_id or not session.get("face_verified") or database.has_voted(voter_id):
        return redirect(url_for("index"))
    party = database.get_party(party_id)
    if not party:
        return redirect(url_for("ballot"))

    if request.method == "GET":
        token = csrf_token()
        party_name = escape(party["name"])
        symbol_name = os.path.basename(party["symbol_filename"])
        symbol_url = escape(url_for("static", filename=f"symbols/{symbol_name}"))
        return page("Confirm", f"""
        <div class="card center"><h2>Step 4 - Confirm</h2><img src="{symbol_url}" width="90" alt=""><h3>{party_name}</h3>
        <form method="post"><input type="hidden" name="csrf_token" value="{token}">
        <button class="btn" name="decision" value="yes">Cast Vote</button>
        <button class="btn alt" name="decision" value="no">Go Back</button></form></div>""")

    require_csrf()
    if request.form.get("decision") == "no":
        session.pop("party_id", None)
        return redirect(url_for("ballot"))
    if request.form.get("decision") != "yes":
        abort(400)
    try:
        database.cast_vote(voter_id, party_id)
    except AlreadyVoted:
        flash("This voting session is no longer eligible to cast a ballot.")
        return redirect(url_for("index"))
    except ElectionClosed:
        flash("Voting closed before this ballot could be recorded.")
        return redirect(url_for("index"))

    database.log_event("vote_cast", None)
    session.clear()
    session["vote_complete"] = True
    return redirect(url_for("voted"))


@app.route("/voted")
def voted():
    if not session.pop("vote_complete", False):
        return redirect(url_for("index"))
    return page("Vote Recorded", """
    <div class="card center"><h1>Vote Recorded</h1><p>The anonymous ballot was committed and this voter is marked as having voted. The ballot record itself carries no link back to your identity.</p><a class="btn" href="/">Finish</a></div>""")


@app.route("/results")
def results():
    if not RESULTS_VISIBLE:
        return page("Results", """
        <div class="card center"><h2>Results Hidden</h2><p>Live tallies are disabled by default. Set EVOTING_SHOW_RESULTS=1 only for a completed simulation.</p></div>""")

    total = database.total_votes()
    rows = []
    for party in database.list_parties():
        count = database.party_vote_count(party["id"])
        percentage = round(count / total * 100, 1) if total else 0.0
        rows.append(
            f'<p>{escape(party["name"])}: {count} ({percentage}%)</p>'
            f'<progress value="{percentage}" max="100">{percentage}%</progress>'
        )
    chain_ok, broken_at = database.verify_chain()
    audit_text = "OK" if chain_ok else f"FAILED at ballot {broken_at}"
    return page("Results", f"""
    <div class="card"><h2>Simulation Results</h2><p>Total votes: <b>{total}</b></p>{''.join(rows)}
    <p>Ledger integrity: <b>{escape(audit_text)}</b></p></div>""")


# -------------------------------------------------------------------- admin --

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        token = csrf_token()
        return page("Admin Login", f"""
        <div class="card"><h2>Admin Login</h2>
        <form method="post"><input type="hidden" name="csrf_token" value="{token}">
        <label>Admin password</label><input type="password" name="password" required autofocus>
        <button class="btn" type="submit">Log In</button></form></div>""")

    require_csrf()
    if rate_limited("admin-login", 8, 300):
        abort(429)
    password = request.form.get("password", "")
    stored_hash = database.get_setting("admin_password_hash", "")
    if not stored_hash or not check_password_hash(stored_hash, password):
        database.log_event("admin_login_failed", None)
        flash("Incorrect admin password.")
        return redirect(url_for("admin_login"))

    session["is_admin"] = True
    database.log_event("admin_login_ok", None)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    require_csrf()
    session.pop("is_admin", None)
    return redirect(url_for("index"))


@app.route("/admin")
def admin_dashboard():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))

    token = csrf_token()
    election_open = database.is_election_open()
    total_votes = database.total_votes()
    chain_ok, broken_at = database.verify_chain()
    chain_text = (
        '<span class="good" style="padding:4px 10px;border-radius:6px">OK</span>' if chain_ok
        else f'<span class="bad" style="padding:4px 10px;border-radius:6px">FAILED at ballot {broken_at}</span>'
    )

    toggle_label = "Close Election" if election_open else "Open Election"
    parties_rows = []
    for party in database.list_parties():
        count = database.party_vote_count(party["id"])
        delete_disabled = "disabled" if count else ""
        symbol_name = os.path.basename(party["symbol_filename"])
        symbol_url = escape(url_for("static", filename=f"symbols/{symbol_name}"))
        parties_rows.append(f"""<tr><td><img src="{symbol_url}" width="36" height="36" alt=""></td>
        <td>{escape(party['name'])}</td><td>{escape(party['candidate_name'] or '')}</td>
        <td>{count}</td><td><form method="post" action="{url_for('admin_delete_party')}" style="margin:0">
        <input type="hidden" name="csrf_token" value="{token}"><input type="hidden" name="party_id" value="{party['id']}">
        <button class="btn danger" type="submit" {delete_disabled}>Remove</button></form></td></tr>""")

    log_rows = "".join(
        f"<tr><td>{escape(row['created_at'])}</td><td>{escape(row['event'])}</td><td>{escape(row['detail'] or '')}</td></tr>"
        for row in database.recent_audit_log(50)
    )

    return page("Admin Dashboard", f"""
    <div class="card"><h2>Election Control</h2>
    <p>Status: {'<span class="pill open">Open</span>' if election_open else '<span class="pill closed">Closed</span>'}
    &nbsp; Total votes cast: <b>{total_votes}</b> &nbsp; Ledger integrity: {chain_text}</p>
    <form method="post" action="{url_for('admin_toggle_election')}">
    <input type="hidden" name="csrf_token" value="{token}">
    <button class="btn" type="submit">{toggle_label}</button></form>
    <form method="post" action="{url_for('admin_logout')}" style="display:inline">
    <input type="hidden" name="csrf_token" value="{token}"><button class="btn alt" type="submit">Log Out</button></form>
    </div>

    <div class="card"><h2>Parties</h2><table><tr><th>Symbol</th><th>Name</th><th>Candidate</th><th>Votes</th><th></th></tr>{''.join(parties_rows)}</table>
    <h3>Add Party</h3>
    <form method="post" action="{url_for('admin_add_party')}" enctype="multipart/form-data">
    <input type="hidden" name="csrf_token" value="{token}">
    <label>Party name</label><input type="text" name="name" maxlength="100" required>
    <label>Candidate name</label><input type="text" name="candidate_name" maxlength="100">
    <label>Symbol picture (PNG, JPG, or WEBP)</label><input type="file" name="symbol_image" accept="image/png,image/jpeg,image/webp" required>
    <button class="btn" type="submit">Add Party</button></form></div>

    <div class="card"><h2>Recent Audit Log</h2><table><tr><th>Time (UTC)</th><th>Event</th><th>Detail</th></tr>{log_rows}</table></div>
    """)


@app.route("/admin/election/toggle", methods=["POST"])
def admin_toggle_election():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    require_csrf()
    new_state = not database.is_election_open()
    database.set_election_open(new_state)
    database.log_event("election_opened" if new_state else "election_closed", None)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/parties/add", methods=["POST"])
def admin_add_party():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    require_csrf()
    name = (request.form.get("name") or "").strip()
    candidate_name = (request.form.get("candidate_name") or "").strip()
    upload = request.files.get("symbol_image")
    if not name or not upload or not upload.filename:
        flash("Invalid party details.")
        return redirect(url_for("admin_dashboard"))

    try:
        with Image.open(upload.stream) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Choose a PNG, JPG, or WEBP image.")
            if max(source.size) > 4096:
                raise ValueError("The symbol image must be no larger than 4096 pixels per side.")
            source.verify()
        upload.stream.seek(0)
        with Image.open(upload.stream) as source:
            symbol = ImageOps.exif_transpose(source).convert("RGBA")
            symbol.thumbnail((512, 512), Image.Resampling.LANCZOS)
            symbol_filename = f"party_{secrets.token_hex(12)}.png"
            symbol.save(os.path.join(SYMBOLS_DIR, symbol_filename), format="PNG", optimize=True)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        flash(f"Could not use that symbol image: {exc}")
        return redirect(url_for("admin_dashboard"))
    ordering = database.count_parties() + 1
    database.create_party(name, candidate_name, symbol_filename, ordering)
    database.log_event("party_added", name)
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/parties/delete", methods=["POST"])
def admin_delete_party():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    require_csrf()
    party_id = parse_party_id(request.form.get("party_id"))
    if party_id:
        try:
            database.delete_party(party_id)
            database.log_event("party_removed", str(party_id))
        except ValueError as exc:
            flash(str(exc))
    return redirect(url_for("admin_dashboard"))


@app.errorhandler(400)
def bad_request(_error):
    return page("Bad Request", '<div class="card"><h2>Bad request</h2><p>The form was invalid or expired.</p></div>'), 400


@app.errorhandler(413)
def too_large(_error):
    return page("Request Too Large", '<div class="card"><h2>Request too large</h2><p>The captured images exceeded the request limit.</p></div>'), 413


@app.errorhandler(429)
def too_many(_error):
    return page("Too Many Attempts", '<div class="card"><h2>Too many attempts</h2><p>Please retry after a short pause.</p></div>'), 429


def initialize_project(download_models=True):
    if download_models:
        print("Checking face models...")
        ensure_models(download=True)
    database.init_db()
    generate_symbols()
    seed_demo_parties()
    initialize_admin_password()


def cmd_setup(_args):
    initialize_project(download_models=True)
    print("Setup complete. Run: python app.py serve")


def cmd_reset_demo(_args):
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(database.DB_PATH + suffix)
        except FileNotFoundError:
            pass
    database.init_db()
    generate_symbols()
    seed_demo_parties()
    initialize_admin_password()
    print("Simulation database reset. Face models and the local master key were kept.")


def cmd_serve(args):
    try:
        initialize_project(download_models=True)
    except Exception as exc:
        raise SystemExit(f"Setup failed: {exc}") from exc
    if not 1024 <= args.port <= 65535:
        raise SystemExit("Port must be between 1024 and 65535")
    from waitress import serve
    print(f"Serving local-only on http://127.0.0.1:{args.port}")
    serve(app, host="127.0.0.1", port=args.port, threads=4)


def main():
    parser = argparse.ArgumentParser(description="Local-only e-voting research simulation")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("setup", help="download verified DNN models and initialize the simulation").set_defaults(func=cmd_setup)
    commands.add_parser("reset-demo", help="clear voters and ballots for a fresh simulation").set_defaults(func=cmd_reset_demo)
    serve = commands.add_parser("serve", help="run the localhost web app")
    serve.add_argument("--port", type=int, default=5001)
    serve.set_defaults(func=cmd_serve)
    args = parser.parse_args()
    if args.command is None:
        cmd_serve(argparse.Namespace(port=5001))
    else:
        args.func(args)


if __name__ == "__main__":
    main()
