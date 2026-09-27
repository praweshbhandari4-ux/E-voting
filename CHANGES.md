# Changelog

## Pass 2 — feature completion for publication readiness

### 1. Real tamper-evident, anonymous vote ledger (`database.py`)
The previous `verify_chain()` only checked that ballot row IDs were
sequential — it would not detect an edited vote, only a deleted row. Also,
every ballot stored `voter_id`, so anyone with database access could map a
specific vote back to a specific voter (no ballot secrecy at all, despite
the UI implying anonymity).

Fixed:
- Ballots now carry a real SHA-256 hash chain: each row stores
  `prev_hash`, a random `nonce`, a `timestamp`, and
  `hash = SHA256(prev_hash | party_id | nonce | timestamp)`. `verify_chain()`
  recomputes every hash from scratch and checks both the stored hash and the
  link to the next row — editing any past field on any row breaks the chain
  from that point forward, and this is now demonstrated (see Verified below).
- The `ballots` table has **no `voter_id` column at all**. The link between
  a voter and their vote is dropped the instant the vote is committed (same
  transaction that marks `has_voted = 1`) — confirmed by inspecting the
  actual table schema after a vote was cast.

### 2. National ID card scan + OCR (`ocr.py`, new `/scan-id` route)
Your original spec called for scanning an ID card before biometric
verification; the previous build skipped straight to a self-generated
registration number. Added:
- `ocr.py`: extracts a printed ID-number-shaped string from a captured card
  photo using Tesseract OCR (regex-based candidate matching + digit
  cleanup). Tested against a synthetic card image — extracted correctly.
- `/scan-id` route, now the first step of registration: captures the card
  photo, extracts the ID number, stores only a SHA-256 hash of it (never
  the raw number) as a second uniqueness key alongside face matching.
  A second registration attempt with the same card is now correctly
  rejected (`DuplicateNationalId`) even with a different face.
- **Scope stated honestly in the module docstring**: this is OCR extraction
  accuracy, not government ID verification — there is no live NID database
  to check against in a research prototype. Say this in your paper.

### 3. Admin panel (`/admin/login`, `/admin`)
Previously there was no way to open/close the election or manage parties
without editing the database directly. Added a password-protected admin
panel:
- Password set via `EVOTING_ADMIN_PASSWORD` env var, or auto-generated and
  printed once on first run (never hardcoded), hashed with
  `werkzeug.security.generate_password_hash`.
- Open/close the election (blocks new registrations and votes while
  closed — enforced in `database.py` via `ElectionClosed`, not just hidden
  in the UI).
- Add/remove parties (a party with recorded votes cannot be removed, to
  protect the ledger).
- Live view of total votes, ledger integrity status, and the audit log.

### 4. Audit logging (`audit_log` table)
Key events are now recorded with a timestamp: ID scan success/failure,
registration success/duplicate (face or ID), identify success/failure, face
verification success/failure, vote cast, admin login success/failure,
election opened/closed, party added/removed. No raw biometric data or ID
numbers are logged — event names and short reasons only. Viewable from the
admin dashboard; useful both operationally and as data for your paper's
Results section (e.g. failed-verification rates during testing).

### Verified end-to-end (Flask test client)
Full flow: scan ID card → OCR extraction → register with face capture →
registration number issued → duplicate ID card correctly rejected on a
second attempt (different face, same card) → identify → face verification
with liveness challenge → ballot → confirm → vote recorded anonymously →
admin login (wrong password rejected, correct password accepted) → election
closed from admin panel → registration/voting correctly blocked while
closed → **tampering with a stored ballot's party_id was correctly detected
by `verify_chain()`**, pinpointing the exact broken ballot.

---

# Pass 1 changes (previous delivery)
See the original fixes: real face detection/recognition replacing the
RGB-average stub, motion-based liveness check, the registration-number bug
in `create_voter()`, and the hardcoded session key. All of that remains in
place and unchanged by this pass.

---

## What's still on you (cannot be done without real people / your own judgment)

1. **Recalibrate `SFACE_MATCH_THRESHOLD`** (currently a provisional `0.08`)
   using `evaluate_accuracy.py` against real volunteer photos, per your
   consent form and ethics statement.
2. **Run the usability study** (SUS questionnaire already drafted) and
   collect real timing/error data.
3. **Write the paper sections** using the real data from (1) and (2), plus
   the feature-comparison table against the two reference papers discussed
   earlier.
4. **Before sharing the code anywhere:** add a `.gitignore` excluding
   `evoting.db*`, `.session_key`, and any captured photos — none of these
   should end up in a public repo.
5. **State plainly in Limitations:** the OCR step reads a printed number, it
   does not verify a real government ID; the liveness check is a motion
   deterrent, not certified anti-spoofing; LBP is a deliberate lightweight
   choice, not a claim of deep-learning-level accuracy.
