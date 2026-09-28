# Nepal E-Voting Research Simulation

The voter flow is: scan the voter card (OCR), confirm the number, register with 3 face photos, get a registration number, verify your face, vote on an anonymous hash-chained ballot, then check results and the audit log.

## 1. Install

```bash
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip uninstall -y opencv-python opencv-python-headless  # avoid OpenCV package conflicts
pip install -r requirements.txt
```

Tesseract **with Nepali**. Without it, Devanagari digits can't be read:

| OS | command |
|---|---|
| macOS | `brew install tesseract tesseract-lang` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr tesseract-ocr-nep` |
| Windows | UB-Mannheim installer, tick **Nepali** |

Check: `tesseract --list-langs` must list `nep`.

The face models are in `models/` (YuNet 230 KB, SFace 37 MB). They are checked against their SHA-256 on every start. If they are missing, `python app.py setup` downloads them.

## 2. Run locally

```bash
python app.py reset-demo     # fresh, empty study database (do this once before data collection)
python app.py serve          # http://127.0.0.1:5001
```

On first run a random admin password is printed **once**. Change it with `python app.py set-admin-password`, or set `EVOTING_ADMIN_PASSWORD`.

## 3. Deploy (public URL)

Browsers only allow the camera on **HTTPS** (or localhost). Run the app behind an HTTPS reverse proxy (nginx, Caddy, or a host that terminates TLS) and set:

```bash
export EVOTING_HOST=0.0.0.0
export EVOTING_PORT=5001
export EVOTING_TRUSTED_HOSTS=vote.yourdomain.com    # the exact host name users type
export EVOTING_HTTPS=1
export EVOTING_BEHIND_PROXY=1
export EVOTING_ADMIN_PASSWORD='a-long-password'
export EVOTING_SECRET_KEY='a-long-random-string'    # keep sessions valid across restarts
python app.py serve
```

If `EVOTING_TRUSTED_HOSTS` doesn't match your domain, every request gets "Bad request: this host name is not in EVOTING_TRUSTED_HOSTS".

## 4. Instructions to give volunteers

* Use even, frontal light (a window or lamp **in front**, not behind you). Only one person should be in the frame, with the face filling roughly a third of the picture.
* The 3 registration photos: straight, then head **slightly** left, then **slightly** right.
* For the card: hold it flat, fill the frame, and avoid glare. Check the number on the confirmation screen and correct it if it's wrong. Corrections are logged anonymously, which gives you real OCR accuracy.

## 5. Getting the numbers for the paper

| What | How |
|---|---|
| Face FAR / FRR / EER / ROC | Collect ≥5 photos per consenting volunteer in `dataset/<person>/`, then run `python evaluate_accuracy.py dataset --enroll 3` |
| OCR accuracy | Put card photos and `labels.csv` (`file,number`) in a folder, then run `python evaluate_ocr.py cards/` |
| Live study (success rates, failure reasons, timings, OCR corrections) | Use Admin, then **Download audit log (CSV)**, then run `python analyze_audit_log.py evoting_audit_log.csv` |
| Ledger integrity | Shown on the admin dashboard (`verify_chain()`) |

Calibrate the face threshold on your own volunteers and set it with `EVOTING_FACE_THRESHOLD`. Report the threshold and how you chose it.

## 6. Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `EVOTING_FACE_THRESHOLD` | 0.363 | SFace cosine threshold (OpenCV's published operating point) |
| `EVOTING_DUPLICATE_THRESHOLD` | same | registration duplicate-face threshold |
| `EVOTING_ID_MIN_DIGITS` / `MAX` | 6 / 12 | accepted voter-number length |
| `EVOTING_SHOW_RESULTS` | 0 | show live tallies on /results |
| `EVOTING_THREADS` | 8 | waitress worker threads |
| `TESSERACT_CMD` | auto | path to the tesseract binary |

## 7. Scope (state this in the paper)

* This is a research prototype, not a production election system.
* OCR reads a printed number. It does **not** verify a government ID; there is no NID database.
* There is no liveness or anti-spoofing check in the current flow. A printed photo or screen replay of a registered voter could pass face verification. Report this as a limitation, or add and evaluate presentation-attack detection before claiming it.
* Ballots carry no voter ID. However, a ballot's timestamp and the web-server access log could in principle be correlated by someone with server access. Mention this as a limitation.
