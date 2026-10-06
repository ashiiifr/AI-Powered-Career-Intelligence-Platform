"""
auth.py
-------
Authentication layer for Milestone 4.

- Passwords hashed with bcrypt via passlib (never stored plaintext)
- Session stored in st.session_state["user"] as {id, username, email}
- All functions are pure (no Streamlit coupling) except the session helpers
"""

from __future__ import annotations

import re
import sqlite3

import database as db

# ── Password hashing via argon2-cffi ─────────────────────────────────────────
# bcrypt has a Python 3.14 compatibility issue with passlib's version detection.
# argon2 is the modern, recommended password hashing algorithm (OWASP).
try:
    from argon2 import PasswordHasher as _PH
    from argon2.exceptions import VerifyMismatchError as _VME
    _ph = _PH()

    def hash_password(password: str) -> str:
        return _ph.hash(password)

    def verify_password(plain: str, hashed: str) -> bool:
        try:
            return _ph.verify(hashed, plain)
        except _VME:
            return False
        except Exception:
            return False

except ImportError:
    # Fallback: sha256_crypt via passlib (no native binary needed)
    from passlib.context import CryptContext as _CC
    _pwd_ctx = _CC(schemes=["sha256_crypt"], deprecated="auto")

    def hash_password(password: str) -> str:
        return _pwd_ctx.hash(password)

    def verify_password(plain: str, hashed: str) -> bool:
        return _pwd_ctx.verify(plain, hashed)


# ── Validation ────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def validate_registration(username: str, email: str,
                           password: str, confirm: str) -> list[str]:
    """Return a list of human-readable error strings (empty = valid)."""
    errors: list[str] = []
    u = username.strip()
    e = email.strip()
    p = password

    if len(u) < 3:
        errors.append("Username must be at least 3 characters.")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", u):
        errors.append("Username may only contain letters, digits, _ . -")
    if not _EMAIL_RE.match(e):
        errors.append("Enter a valid email address.")
    if len(p) < 8:
        errors.append("Password must be at least 8 characters.")
    if p != confirm:
        errors.append("Passwords do not match.")
    return errors


# ── Registration / login ──────────────────────────────────────────────────────

def register(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Create a new account.
    Returns (True, "") on success or (False, error_message).
    """
    try:
        db.create_user(
            username=username.strip(),
            email=email.strip().lower(),
            password_hash=hash_password(password),
        )
        return True, ""
    except sqlite3.IntegrityError as exc:
        msg = str(exc).lower()
        if "username" in msg:
            return False, "Username already taken."
        if "email" in msg:
            return False, "An account with that email already exists."
        return False, "Registration failed — please try again."


def login(username: str, password: str) -> tuple[bool, dict | str]:
    """
    Authenticate a user.
    Returns (True, user_dict) or (False, error_message).
    Never reveals whether the username or password was wrong individually.
    """
    user = db.get_user_by_username(username.strip())
    if user is None or not verify_password(password, user["password_hash"]):
        return False, "Invalid username or password."
    return True, {
        "id":       user["id"],
        "username": user["username"],
        "email":    user["email"],
    }


# ── Streamlit session helpers ─────────────────────────────────────────────────

def get_session_user(st_session) -> dict | None:
    """Return the logged-in user dict from st.session_state, or None."""
    return st_session.get("user")


def set_session_user(st_session, user: dict) -> None:
    st_session["user"] = user


def clear_session(st_session) -> None:
    """Log out: wipe all session state."""
    for key in list(st_session.keys()):
        del st_session[key]


def require_auth(st_module) -> dict:
    """
    Call at the top of any page that needs a logged-in user.
    If not authenticated, renders a warning and stops page execution.
    Returns the user dict if authenticated.
    """
    user = get_session_user(st_module.session_state)
    if not user:
        st_module.warning("🔒 Please log in to access this page.")
        st_module.stop()
    return user
