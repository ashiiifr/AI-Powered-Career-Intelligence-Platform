"""
google_meet_adapter.py
----------------------
Google Meet recording adapter for Milestone 4.

Google Meet recordings are stored in the user's Google Drive. This adapter
uses the Google Drive API (v3) to discover and download them.

Credentials (from .env):
    GOOGLE_CLIENT_ID      — OAuth 2.0 client ID
    GOOGLE_CLIENT_SECRET  — OAuth 2.0 client secret
    GOOGLE_REDIRECT_URI   — e.g. http://localhost:8501
    GOOGLE_REFRESH_TOKEN  — obtained via OAuth consent flow (see docs)

When credentials are absent, MOCK mode returns sample data so the dashboard
and processing pipeline can be demonstrated without a Google account.

The OAuth consent flow is a one-time browser step:
    python google_meet_adapter.py --auth
This prints an auth URL, you approve, paste the code back, and the refresh
token is saved to your .env. After that the adapter works headlessly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
_REDIRECT_URI  = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501").strip()
_REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN", "").strip()

LIVE_MODE = bool(_CLIENT_ID and _CLIENT_SECRET and _REFRESH_TOKEN)

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_MAX_FILE_BYTES = 500 * 1024 * 1024


# ── Auth helpers ──────────────────────────────────────────────────────────────

def get_auth_url() -> str:
    """Return the Google OAuth2 consent URL. Used in the one-time setup flow."""
    import urllib.parse
    params = {
        "client_id":     _CLIENT_ID,
        "redirect_uri":  _REDIRECT_URI,
        "response_type": "code",
        "scope":         " ".join(_SCOPES),
        "access_type":   "offline",
        "prompt":        "consent",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_code(code: str) -> dict:
    """Exchange an auth code for tokens. Returns token dict."""
    import httpx
    resp = httpx.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          code,
            "client_id":     _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "redirect_uri":  _REDIRECT_URI,
            "grant_type":    "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _get_access_token() -> str:
    """Use the refresh token to get a short-lived access token."""
    if not LIVE_MODE:
        raise RuntimeError("Google credentials not configured.")
    import httpx
    resp = httpx.post(
        "https://oauth2.googleapis.com/token",
        data={
            "refresh_token": _REFRESH_TOKEN,
            "client_id":     _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "grant_type":    "refresh_token",
        },
        timeout=15,
    )
    if resp.status_code == 401:
        raise RuntimeError(
            "Google token refresh failed — re-run the OAuth consent flow."
        )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── Drive search ──────────────────────────────────────────────────────────────

def list_recordings(page_size: int = 30) -> list[dict]:
    """
    List Google Meet recordings from Google Drive.

    Google Meet recordings have MIME type video/mp4 and are created by
    the Google Meet service. We search for files with 'Meet' in the name
    or in the 'Meet Recordings' folder.

    Each item:
    {
        "provider_id":  str,   # Drive file ID (stable)
        "topic":        str,   # filename
        "start_time":   str,   # createdTime
        "file_size":    int,   # bytes
        "mime_type":    str,
        "download_url": str,
        "mock":         bool,
    }
    """
    if not LIVE_MODE:
        return _mock_recordings()

    import httpx
    token = _get_access_token()

    query = (
        "mimeType='video/mp4' and "
        "(name contains 'Meet' or name contains 'Recording') and "
        "trashed=false"
    )
    resp = httpx.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q":        query,
            "fields":   "files(id,name,createdTime,size,mimeType)",
            "pageSize": page_size,
            "orderBy":  "createdTime desc",
        },
        timeout=20,
    )
    if resp.status_code == 401:
        raise RuntimeError("Google access token expired.")
    resp.raise_for_status()

    results = []
    for f in resp.json().get("files", []):
        results.append({
            "provider_id":  f["id"],
            "topic":        f.get("name", "Untitled"),
            "start_time":   f.get("createdTime", ""),
            "file_size":    int(f.get("size", 0)),
            "mime_type":    f.get("mimeType", "video/mp4"),
            "download_url": f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media",
            "mock":         False,
        })
    return results


def download_recording(recording: dict) -> tuple[str, str]:
    """
    Download a Google Drive file to a temp file.
    Returns (tmp_path, filename).
    """
    if not LIVE_MODE:
        raise RuntimeError("Cannot download in mock mode — Google credentials not configured.")

    import httpx
    token    = _get_access_token()
    url      = recording["download_url"]
    filename = recording["topic"]
    if not filename.endswith(".mp4"):
        filename += ".mp4"

    with httpx.stream(
        "GET", url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
        follow_redirects=True,
    ) as resp:
        if resp.status_code == 401:
            raise RuntimeError("Google token expired during download.")
        resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        downloaded = 0
        for chunk in resp.iter_bytes(chunk_size=65536):
            downloaded += len(chunk)
            if downloaded > _MAX_FILE_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise ValueError("Google Meet recording exceeds 500 MB limit.")
            tmp.write(chunk)
        tmp.close()

    return tmp.name, filename


# ── Mock data ─────────────────────────────────────────────────────────────────

def _mock_recordings() -> list[dict]:
    return [
        {
            "provider_id":  "mock-gmeet-rec-001",
            "topic":        "[DEMO] Team Sync — Mock Google Meet Recording",
            "start_time":   "2026-09-01T10:00:00Z",
            "file_size":    18_000_000,
            "mime_type":    "video/mp4",
            "download_url": "",
            "mock":         True,
        },
        {
            "provider_id":  "mock-gmeet-rec-002",
            "topic":        "[DEMO] Design Review — Mock Google Meet Recording",
            "start_time":   "2026-09-03T15:30:00Z",
            "file_size":    25_000_000,
            "mime_type":    "video/mp4",
            "download_url": "",
            "mock":         True,
        },
    ]


# ── One-time OAuth setup CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--auth" in sys.argv:
        if not (_CLIENT_ID and _CLIENT_SECRET):
            print("Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env first.")
            sys.exit(1)
        print("Open this URL in your browser and approve access:")
        print()
        print(get_auth_url())
        print()
        code = input("Paste the authorization code here: ").strip()
        tokens = exchange_code(code)
        print()
        print("Add this to your .env file:")
        print(f"GOOGLE_REFRESH_TOKEN={tokens.get('refresh_token', '')}")
