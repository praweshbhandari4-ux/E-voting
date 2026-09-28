"""Summarise the audit log exported from /admin (Download audit log CSV) into
the numbers a results section needs: registration success, face-verification
success rate and failure reasons, genuine-attempt score distribution,
OCR accept-vs-corrected rate, and timings.

Usage:
    python analyze_audit_log.py evoting_audit_log.csv
    python analyze_audit_log.py --db evoting.db        # read the database directly

Caveat to state in the paper: verification attempts in the live log are
(almost always) GENUINE attempts by the registered voter, so this gives the
in-the-wild False Reject Rate and usability numbers. False Accept Rate must be
measured separately with evaluate_accuracy.py (impostor comparisons).
"""

import argparse
import csv
import re
import sqlite3
import statistics
from collections import Counter


def load_rows(args):
    if args.db:
        conn = sqlite3.connect(args.db)
        return [
            {"event": e, "detail": d or "", "created_at_utc": c}
            for e, d, c in conn.execute("SELECT event, detail, created_at FROM audit_log ORDER BY id")
        ]
    with open(args.csv, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(rows, event, key):
    out = []
    for row in rows:
        if row["event"] == event:
            match = re.search(rf"{key}=([-\d.]+)", row["detail"] or "")
            if match:
                out.append(float(match.group(1)))
    return out


def describe(name, data):
    if not data:
        return f"{name}: n=0"
    data = sorted(data)
    return (f"{name}: n={len(data)} mean={statistics.mean(data):.3f} median={statistics.median(data):.3f} "
            f"min={data[0]:.3f} max={data[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--db")
    args = parser.parse_args()
    if not args.csv and not args.db:
        parser.error("give the exported CSV or --db evoting.db")
    rows = load_rows(args)
    events = Counter(row["event"] for row in rows)

    ok = events["face_verification_ok"]
    failed = events["face_verification_failed"]
    reasons = Counter()
    for row in rows:
        if row["event"] == "face_verification_failed":
            detail = row["detail"] or ""
            reasons["no_match (below threshold)" if "reason=no_match" in detail else detail.split(":")[0] or "unknown"] += 1

    confirmed, corrected = events["id_confirmed"], events["id_corrected"]
    print("=== Registration ===")
    print(f"card scans ok: {events['id_scan_ok']}  scan failures: {events['id_scan_failed']}")
    if confirmed + corrected:
        print(f"OCR number accepted unchanged: {confirmed}/{confirmed + corrected} "
              f"({confirmed / (confirmed + corrected) * 100:.1f}%)  corrected by voter: {corrected}")
    print(f"voters registered: {events['voter_registered']}  failed attempts: {events['registration_failed']}  "
          f"duplicate face blocked: {events['registration_duplicate_face']}  duplicate card blocked: {events['registration_duplicate_id']}")
    print(describe("enrollment consistency (min pairwise cosine)", values(rows, "voter_registered", "consistency")))
    print(describe("OCR time ms", values(rows, "id_scan_ok", "ms")))

    print("\n=== Face verification (live attempts) ===")
    if ok + failed:
        print(f"attempts: {ok + failed}  accepted: {ok} ({ok / (ok + failed) * 100:.1f}%)  rejected: {failed}")
    for reason, count in reasons.most_common():
        print(f"  rejected - {reason}: {count}")
    print(describe("accepted scores", values(rows, "face_verification_ok", "score")))
    print(describe("below-threshold scores", values(rows, "face_verification_failed", "score")))
    print(describe("verification compute ms", values(rows, "face_verification_ok", "ms")))

    print("\n=== Voting ===")
    print(f"votes cast: {events['vote_cast']}  identify failures: {events['identify_failed']}")


if __name__ == "__main__":
    main()
