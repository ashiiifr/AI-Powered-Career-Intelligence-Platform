"""
database.py
-----------
SQLite persistence layer for Milestone 4.

Schema
------
users           — registered accounts (hashed passwords)
meetings        — one row per uploaded/imported recording
meeting_chunks  — text chunks + embeddings for RAG search

All user-facing queries are scoped by user_id so no cross-user
data leakage is possible at the database level.

The database file defaults to  meetings.db  next to this file.
Set  DATABASE_PATH  env var to override (useful for tests).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path(__file__).parent / "meetings.db"
_DB_PATH    = Path(os.environ.get("DATABASE_PATH", str(_DEFAULT_DB)))

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    created_at    REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS meetings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title            TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    file_size_bytes  INTEGER,
    duration_seconds REAL,
    language         TEXT,
    whisper_model    TEXT,
    status           TEXT    NOT NULL DEFAULT 'pending',
    -- status: pending | processing | done | failed
    error_message    TEXT,
    transcript       TEXT,
    summary          TEXT,
    key_points       TEXT,   -- JSON list
    decisions        TEXT,   -- JSON list
    participants     TEXT,   -- JSON list
    topics           TEXT,   -- JSON list
    action_items     TEXT,   -- JSON list of objects
    all_deadlines    TEXT,   -- JSON list
    speaker_count    INTEGER,
    -- provider fields (Zoom / Google Meet)
    provider         TEXT,   -- 'upload' | 'zoom' | 'google_meet'
    provider_id      TEXT,   -- stable recording ID from provider
    provider_meta    TEXT,   -- JSON blob of raw provider metadata
    created_at       REAL    NOT NULL DEFAULT (unixepoch('now')),
    processed_at     REAL
);

CREATE INDEX IF NOT EXISTS idx_meetings_user    ON meetings(user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_status  ON meetings(status);
CREATE INDEX IF NOT EXISTS idx_meetings_provider ON meetings(provider, provider_id);

CREATE TABLE IF NOT EXISTS meeting_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    embedding   BLOB,   -- numpy float32 array serialised as bytes
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON meeting_chunks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user    ON meeting_chunks(user_id);
"""

# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _conn():
    """Yield a thread-safe SQLite connection with row_factory set."""
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(path: Path | None = None) -> None:
    """Create all tables if they don't exist.  Safe to call on every startup."""
    global _DB_PATH
    if path is not None:
        _DB_PATH = path
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript(_DDL)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str) -> int:
    """Insert a new user. Returns new user id. Raises sqlite3.IntegrityError on duplicate."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
            (username.strip(), email.strip().lower(), password_hash),
        )
        return cur.lastrowid


def get_user_by_username(username: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


# ── Meetings — create / update ────────────────────────────────────────────────

def create_meeting(
    user_id: int,
    title: str,
    filename: str,
    file_size_bytes: int | None = None,
    provider: str = "upload",
    provider_id: str | None = None,
    provider_meta: dict | None = None,
) -> int:
    """Insert a new meeting record in 'pending' state. Returns meeting id."""
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO meetings
               (user_id, title, filename, file_size_bytes, provider, provider_id, provider_meta, status)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                user_id, title, filename, file_size_bytes,
                provider, provider_id,
                json.dumps(provider_meta) if provider_meta else None,
                "pending",
            ),
        )
        return cur.lastrowid


def update_meeting_status(meeting_id: int, status: str, error: str | None = None) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE meetings SET status=?, error_message=? WHERE id=?",
            (status, error, meeting_id),
        )


def save_transcript(meeting_id: int, transcript: str, language: str,
                    whisper_model: str, duration: float | None,
                    speaker_count: int | None) -> None:
    with _conn() as con:
        con.execute(
            """UPDATE meetings SET transcript=?, language=?, whisper_model=?,
               duration_seconds=?, speaker_count=?, status='processing'
               WHERE id=?""",
            (transcript, language, whisper_model, duration, speaker_count, meeting_id),
        )


def save_analysis(meeting_id: int, analysis: dict, gemini: dict | None) -> None:
    """Persist Python NLP + optional Gemini results. Sets status='done'."""
    kp  = json.dumps(analysis.get("key_points", []))
    ai  = json.dumps(analysis.get("action_items", []))
    dl  = json.dumps(analysis.get("all_deadlines", []))
    top = json.dumps(analysis.get("topics", []))

    summary      = gemini.get("summary", "")       if gemini else ""
    decisions    = json.dumps(gemini.get("decisions", []))    if gemini else "[]"
    participants = json.dumps(gemini.get("participants", [])) if gemini else "[]"
    g_topics     = json.dumps(gemini.get("topics", []))       if gemini else top
    g_kp         = json.dumps(gemini.get("key_points", []))   if gemini else kp
    g_ai         = json.dumps(gemini.get("action_items", [])) if gemini else ai

    with _conn() as con:
        con.execute(
            """UPDATE meetings SET
               summary=?, key_points=?, decisions=?, participants=?,
               topics=?, action_items=?, all_deadlines=?,
               speaker_count=?, status='done', processed_at=?
               WHERE id=?""",
            (
                summary,
                g_kp if gemini else kp,
                decisions, participants, g_topics,
                g_ai if gemini else ai,
                dl,
                analysis.get("speaker_count"),
                time.time(),
                meeting_id,
            ),
        )


# ── Meetings — read ───────────────────────────────────────────────────────────

def get_meeting(meeting_id: int, user_id: int) -> dict | None:
    """Fetch a single meeting. Returns None if not found OR wrong user — no info leak."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM meetings WHERE id=? AND user_id=?",
            (meeting_id, user_id),
        ).fetchone()
    return _hydrate(dict(row)) if row else None


def list_meetings(user_id: int, status: str | None = None,
                  search: str | None = None, limit: int = 100) -> list[dict]:
    sql  = "SELECT * FROM meetings WHERE user_id=?"
    args: list[Any] = [user_id]
    if status:
        sql += " AND status=?"
        args.append(status)
    if search:
        sql += " AND (title LIKE ? OR transcript LIKE ?)"
        args += [f"%{search}%", f"%{search}%"]
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with _conn() as con:
        rows = con.execute(sql, args).fetchall()
    return [_hydrate(dict(r)) for r in rows]


def provider_meeting_exists(user_id: int, provider: str, provider_id: str) -> bool:
    """Duplicate detection — has this provider recording already been imported?"""
    with _conn() as con:
        row = con.execute(
            "SELECT id FROM meetings WHERE user_id=? AND provider=? AND provider_id=?",
            (user_id, provider, provider_id),
        ).fetchone()
    return row is not None


def delete_meeting(meeting_id: int, user_id: int) -> bool:
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM meetings WHERE id=? AND user_id=?", (meeting_id, user_id)
        )
    return cur.rowcount > 0


# ── Chunks (RAG) ──────────────────────────────────────────────────────────────

def save_chunks(meeting_id: int, user_id: int,
                chunks: list[tuple[int, str, bytes]]) -> None:
    """chunks: list of (index, text, embedding_bytes)."""
    # Delete old chunks first (re-index on re-process)
    with _conn() as con:
        con.execute("DELETE FROM meeting_chunks WHERE meeting_id=?", (meeting_id,))
        con.executemany(
            """INSERT INTO meeting_chunks
               (meeting_id, user_id, chunk_index, chunk_text, embedding)
               VALUES (?,?,?,?,?)""",
            [(meeting_id, user_id, idx, text, emb) for idx, text, emb in chunks],
        )


def get_all_chunks_for_user(user_id: int) -> list[dict]:
    """Return all chunks belonging to this user (for vector search)."""
    with _conn() as con:
        rows = con.execute(
            """SELECT mc.id, mc.meeting_id, mc.chunk_index, mc.chunk_text, mc.embedding,
                      m.title, m.created_at as meeting_date
               FROM meeting_chunks mc
               JOIN meetings m ON m.id = mc.meeting_id
               WHERE mc.user_id=? AND m.status='done'
               ORDER BY mc.meeting_id, mc.chunk_index""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hydrate(row: dict) -> dict:
    """Deserialise JSON columns back to Python objects."""
    for col in ("key_points", "decisions", "participants", "topics",
                "action_items", "all_deadlines"):
        if isinstance(row.get(col), str):
            try:
                row[col] = json.loads(row[col])
            except (json.JSONDecodeError, TypeError):
                row[col] = []
    if isinstance(row.get("provider_meta"), str):
        try:
            row["provider_meta"] = json.loads(row["provider_meta"])
        except (json.JSONDecodeError, TypeError):
            row["provider_meta"] = {}
    return row
