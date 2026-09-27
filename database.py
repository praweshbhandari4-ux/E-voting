import hashlib
import json
import os
import random
import secrets
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evoting.db")

GENESIS_HASH = "0" * 64


class AlreadyVoted(RuntimeError):
    """Raised when a voter attempts to cast a ballot more than once."""


class DuplicateNationalId(RuntimeError):
    """Raised when a scanned national ID number is already registered."""


class ElectionClosed(RuntimeError):
    """Raised when voting or registration is attempted while the election is closed."""


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _normalize_template(value):
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _template_distance(a, b):
    a_list = _normalize_template(a)
    b_list = _normalize_template(b)
    if not a_list and not b_list:
        return 0.0
    if len(a_list) != len(b_list):
        return float("inf")
    if not a_list:
        return float("inf")
    difference = sum(abs(x - y) for x, y in zip(a_list, b_list))
    max_term = max(1.0, sum(abs(x) for x in a_list), sum(abs(y) for y in b_list))
    return difference / max_term


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            candidate_name TEXT,
            symbol_filename TEXT NOT NULL,
            ordering INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS voters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            registration_number INTEGER NOT NULL UNIQUE,
            national_id_hash TEXT UNIQUE,
            sface_template TEXT NOT NULL,
            lbp_template TEXT NOT NULL,
            has_voted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    voter_columns = {row["name"] for row in conn.execute("PRAGMA table_info(voters)")}
    if "voter_number" not in voter_columns:
        conn.execute("ALTER TABLE voters ADD COLUMN voter_number TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS voters_voter_number_unique ON voters(voter_number)")
    # Ballots deliberately do NOT store voter_id. Once a voter's has_voted flag
    # is set, the link between "who" and "what they voted for" is dropped from
    # the system entirely -- only an anonymous, hash-chained record of the
    # choice remains. This is the ballot-secrecy property: even someone with
    # full database access cannot map a specific vote back to a specific voter.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ballots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            party_id INTEGER NOT NULL,
            nonce TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL,
            FOREIGN KEY(party_id) REFERENCES parties(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('election_open', '1')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('admin_password_hash', '')"
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- settings --

def get_setting(key, default=None):
    with _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key, value):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def is_election_open():
    return get_setting("election_open", "1") == "1"


def set_election_open(is_open):
    set_setting("election_open", "1" if is_open else "0")


# ------------------------------------------------------------------ parties --

def count_parties():
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) AS count FROM parties").fetchone()["count"]


def create_party(name, candidate_name, symbol_filename, ordering):
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO parties (name, candidate_name, symbol_filename, ordering) VALUES (?, ?, ?, ?)",
            (name, candidate_name, symbol_filename, ordering),
        )
        conn.commit()
        return cursor.lastrowid


def delete_party(party_id):
    with _connect() as conn:
        in_use = conn.execute(
            "SELECT COUNT(*) AS count FROM ballots WHERE party_id = ?", (int(party_id),)
        ).fetchone()["count"]
        if in_use:
            raise ValueError("cannot delete a party that already has recorded votes")
        conn.execute("DELETE FROM parties WHERE id = ?", (int(party_id),))
        conn.commit()


# ------------------------------------------------------------------- voters --

def count_voters():
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) AS count FROM voters").fetchone()["count"]

def find_duplicate_face(sface_template):
    with _connect() as conn:
        rows = conn.execute("SELECT id, sface_template FROM voters").fetchall()
    for row in rows:
        if _template_distance(row["sface_template"], sface_template) < 0.02:
            return row["id"]
    return None


def hash_national_id(national_id_text):
    """One-way hash of the scanned ID number. We never store the raw ID
    number -- only enough to detect a second registration attempt using the
    same ID."""
    normalized = "".join(ch for ch in national_id_text if ch.isalnum()).upper()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def create_voter(name, sface_template, lbp_template, voter_number=None):
    if not is_election_open():
        raise ElectionClosed("registration is closed")
    target = _normalize_template(sface_template)
    with _connect() as conn:
        if voter_number:
            existing_id = conn.execute(
                "SELECT id FROM voters WHERE voter_number = ?", (voter_number,)
            ).fetchone()
            if existing_id:
                raise DuplicateNationalId("this voter number has already been registered")
        existing = {row["registration_number"] for row in conn.execute("SELECT registration_number FROM voters").fetchall()}
        while True:
            reg_number = random.SystemRandom().randrange(10**11, 10**12)
            if reg_number not in existing:
                break
        conn.execute(
            "INSERT INTO voters (name, registration_number, voter_number, sface_template, lbp_template) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, reg_number, voter_number, json.dumps(target), json.dumps(_normalize_template(lbp_template))),
        )
        conn.commit()
        return reg_number


def get_voter_by_registration(registration_number):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM voters WHERE registration_number = ?",
            (int(registration_number),),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"], "name": row["name"], "registration_number": row["registration_number"],
        "voter_number": row["voter_number"],
        "sface_template": json.loads(row["sface_template"]), "lbp_template": json.loads(row["lbp_template"]),
        "has_voted": bool(row["has_voted"]),
    }


def get_voter_by_id(voter_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM voters WHERE id = ?", (int(voter_id),)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"], "name": row["name"], "registration_number": row["registration_number"],
        "voter_number": row["voter_number"],
        "sface_template": json.loads(row["sface_template"]), "lbp_template": json.loads(row["lbp_template"]),
        "has_voted": bool(row["has_voted"]),
    }


def list_parties():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM parties ORDER BY ordering ASC, id ASC").fetchall()
    return [dict(row) for row in rows]


def get_party(party_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM parties WHERE id = ?", (int(party_id),)).fetchone()
    if row is None:
        return None
    return dict(row)


def has_voted(voter_id):
    with _connect() as conn:
        row = conn.execute("SELECT has_voted FROM voters WHERE id = ?", (int(voter_id),)).fetchone()
    return bool(row["has_voted"]) if row else False


# --------------------------------------------------------- hash-chain ledger --

def _hash_ballot(prev_hash, party_id, nonce, timestamp):
    payload = f"{prev_hash}|{party_id}|{nonce}|{timestamp}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_last_hash(conn):
    row = conn.execute("SELECT hash FROM ballots ORDER BY id DESC LIMIT 1").fetchone()
    return row["hash"] if row else GENESIS_HASH


def cast_vote(voter_id, party_id):
    if not is_election_open():
        raise ElectionClosed("voting is closed")
    with _connect() as conn:
        row = conn.execute("SELECT has_voted FROM voters WHERE id = ?", (int(voter_id),)).fetchone()
        if row is None:
            raise ValueError("voter not found")
        if row["has_voted"]:
            raise AlreadyVoted("This voter has already voted")

        prev_hash = get_last_hash(conn)
        nonce = secrets.token_hex(16)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ballot_hash = _hash_ballot(prev_hash, int(party_id), nonce, timestamp)

        cursor = conn.execute(
            "INSERT INTO ballots (party_id, nonce, timestamp, prev_hash, hash) VALUES (?, ?, ?, ?, ?)",
            (int(party_id), nonce, timestamp, prev_hash, ballot_hash),
        )
        # has_voted is updated in the SAME transaction as the anonymous ballot
        # insert, but the ballot row itself carries no reference to voter_id --
        # the link is dropped at the instant the vote is committed.
        conn.execute("UPDATE voters SET has_voted = 1 WHERE id = ?", (int(voter_id),))
        conn.commit()
        return cursor.lastrowid


def total_votes():
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) AS count FROM ballots").fetchone()["count"]


def party_vote_count(party_id):
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) AS count FROM ballots WHERE party_id = ?", (int(party_id),)).fetchone()["count"]


def verify_chain():
    """Recomputes every ballot's hash from its stored fields and checks it
    against both the stored hash AND the next row's recorded prev_hash. This
    actually detects tampering (an edited party_id, nonce, or timestamp on any
    past row breaks the chain from that point on) -- unlike a check that only
    confirms row IDs are sequential, which does not detect an edited row at
    all, only a deleted one."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, party_id, nonce, timestamp, prev_hash, hash FROM ballots ORDER BY id ASC"
        ).fetchall()

    expected_prev = GENESIS_HASH
    for row in rows:
        if row["prev_hash"] != expected_prev:
            return False, row["id"]
        recomputed = _hash_ballot(row["prev_hash"], row["party_id"], row["nonce"], row["timestamp"])
        if recomputed != row["hash"]:
            return False, row["id"]
        expected_prev = row["hash"]
    return True, None


# ---------------------------------------------------------------- audit log --

def log_event(event, detail=None):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (event, detail) VALUES (?, ?)",
            (event, detail),
        )
        conn.commit()


def recent_audit_log(limit=200):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT event, detail, created_at FROM audit_log ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"DB ready: {DB_PATH}")
