# Nepal E-Voting Research Simulation

A small Flask web application for demonstrating a voter registration and voting workflow. It combines voter-card number OCR, camera-based face matching, an administrator dashboard, and an anonymous hash-chained ballot ledger.

> **Research/demo software only. Do not use this project to run a real election or collect real voter identity or biometric data.** The OCR does not check a government database, the face matcher is a lightweight experimental method, and the motion check is not certified anti-spoofing. The public Render configuration also uses a single SQLite file and is not an election-grade hosting design.

## What the project demonstrates

- A browser captures a voter-card image and reads a number labelled `मतदाता नं` (voter number) using OCR.
- The voter supplies a display name and a face image. The system detects one face and creates a Local Binary Pattern (LBP) texture histogram.
- A random 12-digit simulation registration number is issued. The voter uses it to start the vote flow, then verifies their face again.
- The voter selects a party/candidate and confirms the choice.
- A ballot is recorded without a voter ID on the ballot row. The voter record is marked as having voted in the same database transaction.
- An admin can close/open the simulation, manage parties, review registered-voter details and voting status, see totals and recent audit events, and inspect the ballot-chain integrity check.

The app seeds six demo parties and symbols on first startup. The `/results` page hides tallies unless `EVOTING_SHOW_RESULTS=1` is set.

## Technology

| Component | Role |
| --- | --- |
| Python 3.11 | Application runtime |
| Flask | HTTP routes, request handling, sessions, and server-rendered pages |
| SQLite | Local persistence for voters, parties, ballots, settings, and audit events |
| Waitress | Production-style WSGI server used by the local and Render commands |
| OpenCV (`opencv-contrib-python`) | Face detection, image preprocessing, and LBP operations |
| NumPy | Face histogram calculations |
| Pillow | Image decoding, validation, resizing, and generated party symbols |
| Tesseract OCR + `pytesseract` | Reading printed voter numbers from voter-card images |
| HTML, CSS, browser JavaScript | User interface and webcam capture (`static/camera.js`) |
| Docker + Render | Optional containerized deployment setup |

The app pages are currently built with HTML strings and a shared inline CSS template in `app.py`; there is no separate frontend framework or build step.

## How the workflow works

### Registration

1. **Scan card:** `/scan-id` accepts a webcam capture or JPG/PNG/WEBP upload. Tesseract preprocesses the image and looks for a voter-number-shaped digit sequence, including Devanagari digits. This is text extraction only; there is no check that the card is genuine or that the number belongs to a real person.
2. **Register face:** `/register` asks for a display name and one camera capture. OpenCV's bundled Haar cascade looks for exactly one face and rejects a very blurry crop. The system computes an LBP histogram for that face and stores the template for later comparison. Similar templates are checked to discourage duplicate registrations.
3. **Issue number:** A random 12-digit registration number is generated and shown once. The voter needs to keep it for the demo voting flow.

### Voting

1. The voter starts a voting session and enters the 12-digit registration number.
2. The face from the new camera capture is compared with the stored LBP histogram using a chi-square distance converted to a similarity score. The current threshold (`SFACE_MATCH_THRESHOLD` in `face_engine.py`) is provisional and has not been calibrated for a real voter population.
3. A short liveness prompt asks the voter to move. The code checks a change in detected face position or size. This is a basic motion deterrent; it cannot reliably stop photo or video replay attacks.
4. The voter selects a party and confirms. `database.cast_vote()` writes the choice to `ballots` and marks that voter as having voted in the same transaction. The ballot row has no voter ID field.

### Data and integrity

`database.py` creates `evoting.db` beside the source code on startup. Its main tables are:

- `voters`: display name, generated registration number, OCR voter number, face templates, and `has_voted` flag.
- `parties`: party/candidate names and symbol image filenames.
- `ballots`: party choice, timestamp, random nonce, previous hash, and current SHA-256 hash. No voter ID is included.
- `audit_log`: event labels and short details for actions such as registrations, failed checks, votes, and admin activity.
- `settings`: election-open state and a password hash for the admin.

The ballot hash is calculated from the prior ballot hash plus the choice, nonce, and timestamp. `verify_chain()` recomputes the values and detects edits to stored ballot records. This is a tamper-evidence check inside one SQLite database, not a distributed blockchain: someone able to rewrite the whole database can also recompute the chain. Ballots are unlinkable to voter IDs at the schema level, but totals and timing may still reveal information in a small demo.

**Data handling caveat:** despite a comment in the registration page suggesting otherwise, the current code stores the generated registration number and OCR voter number as values in the `voters` table. Treat `evoting.db` as sensitive; do not commit or share it. The `.gitignore` excludes the database and session key.

## Requirements

- Python 3.11 (the Docker image is based on Python 3.11).
- Tesseract OCR installed on the machine, with language data appropriate for the card. English is sufficient for ASCII digits; Nepali text recognition improves when Nepali trained data is installed.
- A webcam, or a clear JPG/PNG/WEBP voter-card image for registration. Browser camera access generally requires `localhost` or HTTPS.

## Run locally

### macOS

```bash
brew install tesseract
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py serve
```

On Debian/Ubuntu, install the system OCR program first with `sudo apt install tesseract-ocr tesseract-ocr-nep`. On Windows, install Tesseract and make sure its executable is on `PATH`; then use the equivalent venv and pip commands.

Open [http://127.0.0.1:5001](http://127.0.0.1:5001). The camera may require permission in the browser. You can use `python app.py serve --port 8000` to select a different local port.

The first startup creates the database, default party symbols and initial demo parties. Locally the admin password defaults to `ADMIN`, so set a private password before using the admin page:

```bash
export EVOTING_ADMIN_PASSWORD='choose-a-local-demo-password'
export EVOTING_SECRET_KEY='choose-a-long-random-secret-value'
python app.py serve
```

On PowerShell, set these with `$env:EVOTING_ADMIN_PASSWORD="..."` and `$env:EVOTING_SECRET_KEY="..."`. If `EVOTING_SECRET_KEY` is omitted locally, the app creates a random `.session_key` file. Keep that file private and stable while using the same session database.

Go to `/admin/login` to manage the demo. If you do not set a local admin password, the code uses the weak default `ADMIN`; that default is unsuitable for any network-accessible deployment.

## Run with Docker

The Dockerfile installs Tesseract, Nepali OCR data, Python dependencies, then starts `render_start.py` on port 10000. Supply both required secrets:

```bash
docker build -t nepal-evoting-demo .
docker run --rm -p 10000:10000 \
  -e EVOTING_ADMIN_PASSWORD='choose-a-private-password' \
  -e EVOTING_SECRET_KEY='choose-a-long-random-secret-value' \
  nepal-evoting-demo
```

Open [http://localhost:10000](http://localhost:10000). The database is `/app/evoting.db` inside the container. The current Render configuration does not declare persistent storage, so database contents can be lost when the instance is replaced or redeployed. A persistent volume needs to be configured for the database path without masking the application files. The checked-in Render service definition uses the same Dockerfile and expects the two environment values to be configured in Render.

## Admin and configuration

- `/admin/login`: password-protected admin sign-in.
- `/admin`: election open/closed state, vote count, ledger check, party management, and recent audit log.
- The admin dashboard's **Registered Voters** table shows each display name, generated registration number, scanned voter-card number, whether the account has voted, and registration time. Access is restricted to an authenticated admin; biometric templates are not shown.
- A party with recorded votes cannot be removed.
- `EVOTING_ADMIN_PASSWORD`: admin password used on initialization. The stored value is a Werkzeug password hash.
- `EVOTING_SECRET_KEY`: stable secret used to sign Flask session cookies. Keep it secret; changing it logs out existing sessions.
- `EVOTING_SHOW_RESULTS=1`: show the simulated results page; results are hidden by default.
- `TESSERACT_CMD`: optional full path to the Tesseract executable if it is not in `PATH`.

The local server binds only to `127.0.0.1`. The Render entry point binds to `0.0.0.0` because the hosting service needs an externally reachable port.

## Reset demo records

This removes the local voter/ballot database and recreates its demo parties/settings; it does not remove the session key or face detector files:

```bash
python app.py reset-demo
```

Do not run this against data you need to keep. To back up or move a local simulation, stop the app and keep `evoting.db` private.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Flask application, pages, registration/voting routes, admin controls, startup commands |
| `database.py` | SQLite schema, voter/party operations, vote transaction, hash-chain check, audit log |
| `face_engine.py` | Face detection, image quality check, LBP templates, face comparison, motion challenge |
| `ocr.py` | Tesseract setup, voter-card image preprocessing, voter-number extraction |
| `security.py` | Session signing key and registration-number formatting/validation |
| `static/camera.js` | Browser camera capture and frame transfer |
| `static/symbols/` | Default party symbols; uploaded admin symbols are also saved here |
| `render_start.py` | Render/host startup and required environment checks |
| `Dockerfile` | Container image and runtime command |
| `render.yaml` | Render web-service configuration |
| `requirements.txt` | Python packages |

## Limitations and responsible use

- This is not suitable for public elections, production identity verification, or real ballots.
- OCR can misread a card and does not establish authenticity or voter eligibility.
- Haar detection and LBP are lightweight demonstrations, not modern high-assurance face recognition. The similarity threshold needs evaluation on an appropriately consented dataset before making accuracy claims.
- The movement liveness prompt can be spoofed and is not certified anti-spoofing.
- SQLite, one server process, and the demo's operational setup are not designed for election-scale concurrency, independent audits, disaster recovery, or a public election threat model.
- The audit log and hash chain are held in the same database and are not independently signed or externally anchored.
- Camera images are processed in memory by the application; face templates and the scanned voter number persist in the local database. Protect and delete demo data responsibly.

## License

No license file is currently included. Ask the project owner before redistributing or reusing this code if no license has been added.
