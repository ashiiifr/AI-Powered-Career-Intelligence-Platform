"""
zoom_adapter.py
---------------
Zoom Server-to-Server OAuth adapter for Milestone 4.

Credentials (from .env):
    ZOOM_ACCOUNT_ID    — your Zoom account ID
    ZOOM_CLIENT_ID     — Server-to-Server OAuth app client ID
    ZOOM_CLIENT_SECRET — Server-to-Server OAuth app client secret

When credentials are absent the adapter runs in MOCK mode and returns
sample recording data so the UI and processing pipeline can be demonstrated
without a live Zoom account. Mock mode is clearly labelled in all return values.

Live mode is activated automatically when all three env vars are present.

Security
--------
- Credentials read from environment only, never hardcoded.
- Downloaded content validated (MIME type + file size) before processing.
- Provider recording ID used as stable idempotency key.
- Expired/revoked tokens trigger a clear RuntimeError (not a silent failure).
- No credentials or token values are logged.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

_ACCOUNT_ID    = os.environ.get("ZOOM_ACCOUNT_ID", "").strip()
_CLIENT_ID     = os.environ.get("ZOOM_CLIENT_ID", "").strip()
_CLIENT_SECRET = os.environ.get("ZOOM_CLIENT_SECRET", "").strip()

LIVE_MODE = bool(_ACCOUNT_ID and _CLIENT_ID and _CLIENT_SECRET)

# Allowed audio/video MIME types (same as audio_processor.py)
_ALLOWED_MIME = {
    "audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/flac", "video/mp4", "audio/webm",
    "application/octet-stream",   # Zoom sometimes sends generic binary
}
_MAX_FILE_BYTES = 500 * 1024 * 1024   # 500 MB


# ── Token management ──────────────────────────────────────────────────────────

_cached_token: str | None = None


def _get_access_token() -> str:
    """Fetch a Server-to-Server OAuth access token from Zoom."""
    global _cached_token
    if not LIVE_MODE:
        raise RuntimeError("Zoom credentials not configured.")

    import base64

    credentials = base64.b64encode(
        f"{_CLIENT_ID}:{_CLIENT_SECRET}".encode()
    ).decode()

    resp = httpx.post(
        "https://zoom.us/oauth/token",
        params={"grant_type": "account_credentials", "account_id": _ACCOUNT_ID},
        headers={"Authorization": f"Basic {credentials}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )

    if resp.status_code == 401:
        raise RuntimeError(
            "Zoom authentication failed — check ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET."
        )
    resp.raise_for_status()
    _cached_token = resp.json()["access_token"]
    return _cached_token


def _zoom_get(path: str, params: dict | None = None) -> Any:
    """Authenticated GET against the Zoom API v2."""
    token = _get_access_token()
    resp = httpx.get(
        f"https://api.zoom.us/v2{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params or {},
        timeout=20,
    )
    if resp.status_code == 401:
        raise RuntimeError("Zoom token expired or revoked. Re-authentication required.")
    if resp.status_code == 404:
        raise FileNotFoundError(f"Zoom resource not found: {path}")
    resp.raise_for_status()
    return resp.json()


# ── Public API ────────────────────────────────────────────────────────────────

def list_recordings(page_size: int = 30) -> list[dict]:
    """
    Return a list of cloud recordings for the authenticated account.

    Each item:
    {
        "provider_id": str,      # stable Zoom recording file UUID
        "meeting_id":  str,      # Zoom meeting ID
        "topic":       str,      # meeting title
        "start_time":  str,      # ISO 8601
        "duration":    int,      # minutes
        "file_type":   str,      # "MP4" | "M4A" | etc.
        "file_size":   int,      # bytes
        "download_url": str,
        "mock":        bool,
    }
    """
    if not LIVE_MODE:
        return _mock_recordings()

    try:
        data = _zoom_get("/users/me/recordings", {"page_size": page_size})
    except Exception as exc:
        raise RuntimeError(f"Failed to list Zoom recordings: {exc}") from exc

    results = []
    for meeting in data.get("meetings", []):
        for rf in meeting.get("recording_files", []):
            if rf.get("status") != "completed":
                continue
            results.append({
                "provider_id":  rf["id"],
                "meeting_id":   str(meeting["id"]),
                "topic":        meeting.get("topic", "Untitled"),
                "start_time":   meeting.get("start_time", ""),
                "duration":     meeting.get("duration", 0),
                "file_type":    rf.get("file_type", ""),
                "file_size":    rf.get("file_size", 0),
                "download_url": rf.get("download_url", ""),
                "mock":         False,
            })
    return results


def download_recording(recording: dict) -> tuple[str, str]:
    """
    Download a Zoom recording to a temporary file.

    Returns (tmp_file_path, original_filename).
    Validates MIME type and file size before returning.
    Caller is responsible for deleting the temp file.
    """
    if not LIVE_MODE:
        raise RuntimeError("Cannot download in mock mode — Zoom credentials not configured.")

    import httpx

    url       = recording["download_url"]
    file_type = recording.get("file_type", "mp4").lower()
    ext       = f".{file_type}" if file_type else ".mp4"
    filename  = f"{recording['topic']}{ext}".replace(" ", "_")

    token = _get_access_token()

    with httpx.stream(
        "GET", url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
        follow_redirects=True,
    ) as resp:
        if resp.status_code == 401:
            raise RuntimeError("Zoom token expired during download.")
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type not in _ALLOWED_MIME:
            raise ValueError(f"Unexpected content type from Zoom: {content_type}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        downloaded = 0
        for chunk in resp.iter_bytes(chunk_size=65536):
            downloaded += len(chunk)
            if downloaded > _MAX_FILE_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise ValueError(f"Zoom recording exceeds 500 MB limit.")
            tmp.write(chunk)
        tmp.close()

    return tmp.name, filename


def provider_id_checksum(recording: dict) -> str:
    """Stable idempotency key = SHA256 of provider_id."""
    return hashlib.sha256(recording["provider_id"].encode()).hexdigest()[:16]


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_recordings() -> list[dict]:
    """Return realistic fake recordings for demo/testing when credentials absent."""
    return [
        {
            "provider_id":  "mock-zoom-rec-001",
            "meeting_id":   "123456789",
            "topic":        "[DEMO] Weekly Standup — Mock Recording",
            "start_time":   "2026-09-01T09:00:00Z",
            "duration":     30,
            "file_type":    "MP4",
            "file_size":    15_000_000,
            "download_url": "",
            "mock":         True,
        },
        {
            "provider_id":  "mock-zoom-rec-002",
            "meeting_id":   "987654321",
            "topic":        "[DEMO] Product Review — Mock Recording",
            "start_time":   "2026-09-02T14:00:00Z",
            "duration":     60,
            "file_type":    "M4A",
            "file_size":    22_000_000,
            "download_url": "",
            "mock":         True,
        },
    ]
