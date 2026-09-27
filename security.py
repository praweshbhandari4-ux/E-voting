import os
import re
import secrets
import stat

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_KEY_PATH = os.path.join(_BASE_DIR, ".session_key")


def _load_or_create_session_key():
    """Generates a random 32-byte session-signing key on first run and persists
    it to a local, owner-only-readable file. Never hardcode this key in source:
    a key committed to a file (and therefore to any git history, backup, or copy
    of the repo) lets anyone who has read the code forge session cookies."""
    env_key = os.environ.get("EVOTING_SECRET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    if os.path.exists(_KEY_PATH):
        with open(_KEY_PATH, "rb") as handle:
            key = handle.read()
        if len(key) >= 32:
            return key

    key = secrets.token_bytes(32)
    fd = os.open(_KEY_PATH, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    os.chmod(_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    return key


SESSION_KEY = _load_or_create_session_key()


def format_registration_number(number):
    value = str(int(number)).zfill(12)
    return f"{value[:4]}-{value[4:8]}-{value[8:12]}"


def normalize_registration_number(value):
    if value is None:
        raise ValueError("registration number required")
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 12 or not digits.isdigit():
        raise ValueError("registration number must be 12 digits")
    return int(digits)
