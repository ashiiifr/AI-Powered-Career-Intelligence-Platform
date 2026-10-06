"""
tests/test_all.py
-----------------
Full test suite for Milestone 4.
All external services (Gemini, Zoom, Google) are mocked.
A temporary SQLite database is used — no production data is touched.

Run:  cd "d:\\Projects\\Milestone 01"
      .\\myenv\\Scripts\\python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Point at a temp DB before any module import touches the default path
_TMP_DB = Path(tempfile.mkdtemp()) / "test_meetings.db"
os.environ["DATABASE_PATH"] = str(_TMP_DB)

# Now safe to import our modules
import database as db
import auth
import export as exp
import rag_engine
from zoom_adapter import _mock_recordings as zoom_mock_recs
from google_meet_adapter import _mock_recordings as gmeet_mock_recs


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Each test gets its own isolated database."""
    test_db = tmp_path / "test.db"
    db._DB_PATH = test_db
    db.init_db(test_db)
    yield
    if test_db.exists():
        test_db.unlink()


@pytest.fixture()
def user_id(fresh_db):
    uid = db.create_user("alice", "alice@example.com",
                          auth.hash_password("password123"))
    return uid


@pytest.fixture()
def meeting_id(user_id):
    mid = db.create_meeting(user_id, "Test Meeting", "test.mp3",
                             file_size_bytes=1000, provider="upload")
    return mid


# ─────────────────────────────────────────────────────────────────────────────
# database.py
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:

    def test_create_and_fetch_user(self, fresh_db):
        uid = db.create_user("bob", "bob@example.com", "hash")
        user = db.get_user_by_username("bob")
        assert user is not None
        assert user["id"] == uid
        assert user["email"] == "bob@example.com"

    def test_username_case_insensitive(self, fresh_db):
        db.create_user("Carol", "carol@example.com", "hash")
        user = db.get_user_by_username("carol")
        assert user is not None

    def test_duplicate_username_raises(self, fresh_db):
        db.create_user("dave", "dave@example.com", "hash")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_user("dave", "dave2@example.com", "hash")

    def test_duplicate_email_raises(self, fresh_db):
        db.create_user("eve", "eve@example.com", "hash")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_user("eve2", "eve@example.com", "hash")

    def test_get_user_by_id(self, user_id):
        user = db.get_user_by_id(user_id)
        assert user["username"] == "alice"

    def test_create_meeting(self, user_id):
        mid = db.create_meeting(user_id, "Standup", "audio.mp3")
        assert isinstance(mid, int) and mid > 0

    def test_meeting_scoped_to_user(self, user_id):
        other_uid = db.create_user("mallory", "m@example.com", "hash")
        mid = db.create_meeting(user_id, "Private", "priv.mp3")
        # Mallory cannot access alice's meeting
        assert db.get_meeting(mid, other_uid) is None
        # Alice can
        assert db.get_meeting(mid, user_id) is not None

    def test_update_meeting_status(self, user_id, meeting_id):
        db.update_meeting_status(meeting_id, "done")
        m = db.get_meeting(meeting_id, user_id)
        assert m["status"] == "done"

    def test_update_meeting_status_failed(self, user_id, meeting_id):
        db.update_meeting_status(meeting_id, "failed", "Whisper crashed")
        m = db.get_meeting(meeting_id, user_id)
        assert m["status"] == "failed"
        assert m["error_message"] == "Whisper crashed"

    def test_save_transcript(self, user_id, meeting_id):
        db.save_transcript(meeting_id, "Hello world", "en", "base", 120.0, 2)
        m = db.get_meeting(meeting_id, user_id)
        assert m["transcript"] == "Hello world"
        assert m["language"] == "en"
        assert m["status"] == "processing"

    def test_save_analysis(self, user_id, meeting_id):
        analysis = {
            "key_points": ["Point A"],
            "action_items": [{"task": "Do it", "assigned_to": "Bob",
                               "deadline": "Friday", "priority": "high",
                               "status": "pending"}],
            "all_deadlines": ["Friday"],
            "topics": ["AI"],
            "speaker_count": 2,
        }
        gemini = {
            "summary": "Good meeting.",
            "decisions": ["Launch it"],
            "participants": ["Alice", "Bob"],
            "topics": ["AI", "Launch"],
            "key_points": ["Point A"],
            "action_items": analysis["action_items"],
        }
        db.save_analysis(meeting_id, analysis, gemini)
        m = db.get_meeting(meeting_id, user_id)
        assert m["status"] == "done"
        assert m["summary"] == "Good meeting."
        assert m["decisions"] == ["Launch it"]
        assert m["participants"] == ["Alice", "Bob"]
        assert len(m["action_items"]) == 1

    def test_list_meetings_empty(self, user_id):
        assert db.list_meetings(user_id) == []

    def test_list_meetings_filter_status(self, user_id):
        mid = db.create_meeting(user_id, "Done Meeting", "d.mp3")
        db.update_meeting_status(mid, "done")
        db.create_meeting(user_id, "Pending Meeting", "p.mp3")
        done = db.list_meetings(user_id, status="done")
        assert len(done) == 1
        assert done[0]["title"] == "Done Meeting"

    def test_list_meetings_search(self, user_id):
        db.create_meeting(user_id, "Alpha Meeting", "a.mp3")
        db.create_meeting(user_id, "Beta Meeting", "b.mp3")
        results = db.list_meetings(user_id, search="Alpha")
        assert len(results) == 1
        assert results[0]["title"] == "Alpha Meeting"

    def test_delete_meeting(self, user_id, meeting_id):
        assert db.delete_meeting(meeting_id, user_id) is True
        assert db.get_meeting(meeting_id, user_id) is None

    def test_delete_wrong_user(self, user_id, meeting_id):
        other_uid = db.create_user("zara", "z@example.com", "hash")
        assert db.delete_meeting(meeting_id, other_uid) is False

    def test_provider_duplicate_detection(self, user_id):
        db.create_meeting(user_id, "Zoom Rec", "z.mp4",
                           provider="zoom", provider_id="abc123")
        assert db.provider_meeting_exists(user_id, "zoom", "abc123") is True
        assert db.provider_meeting_exists(user_id, "zoom", "different") is False

    def test_json_columns_hydrated(self, user_id, meeting_id):
        analysis = {"key_points": ["A", "B"], "action_items": [],
                    "all_deadlines": [], "topics": [], "speaker_count": 1}
        db.save_analysis(meeting_id, analysis, None)
        m = db.get_meeting(meeting_id, user_id)
        assert isinstance(m["key_points"], list)
        assert isinstance(m["action_items"], list)

    def test_save_and_get_chunks(self, user_id, meeting_id):
        import numpy as np
        chunks = [
            (0, "First chunk text", np.zeros(384, dtype=np.float32).tobytes()),
            (1, "Second chunk text", np.ones(384, dtype=np.float32).tobytes()),
        ]
        db.save_chunks(meeting_id, user_id, chunks)
        all_c = db.get_all_chunks_for_user(user_id)
        # Meeting is in 'pending' status so chunks won't appear (status must be 'done')
        # Mark it done first
        db.update_meeting_status(meeting_id, "done")
        all_c = db.get_all_chunks_for_user(user_id)
        assert len(all_c) == 2


# ─────────────────────────────────────────────────────────────────────────────
# auth.py
# ─────────────────────────────────────────────────────────────────────────────

class TestAuth:

    def test_hash_and_verify(self):
        h = auth.hash_password("secret123")
        assert auth.verify_password("secret123", h) is True
        assert auth.verify_password("wrong", h) is False

    def test_validate_registration_valid(self):
        errs = auth.validate_registration("alice", "alice@x.com", "Password1!", "Password1!")
        assert errs == []

    def test_validate_registration_short_username(self):
        errs = auth.validate_registration("ab", "a@x.com", "Password1!", "Password1!")
        assert any("Username" in e for e in errs)

    def test_validate_registration_bad_email(self):
        errs = auth.validate_registration("alice", "notanemail", "Password1!", "Password1!")
        assert any("email" in e.lower() for e in errs)

    def test_validate_registration_short_password(self):
        errs = auth.validate_registration("alice", "a@x.com", "short", "short")
        assert any("Password" in e for e in errs)

    def test_validate_registration_password_mismatch(self):
        errs = auth.validate_registration("alice", "a@x.com", "Password1!", "Different!")
        assert any("match" in e.lower() for e in errs)

    def test_register_and_login(self, fresh_db):
        ok, _ = auth.register("bob", "bob@x.com", "MyPassword1!")
        assert ok is True
        ok, user = auth.login("bob", "MyPassword1!")
        assert ok is True
        assert user["username"] == "bob"

    def test_login_wrong_password(self, fresh_db):
        auth.register("carol", "carol@x.com", "CorrectPassword!")
        ok, msg = auth.login("carol", "WrongPassword!")
        assert ok is False
        assert "Invalid" in msg

    def test_login_nonexistent_user(self, fresh_db):
        ok, msg = auth.login("nobody", "anything")
        assert ok is False
        assert "Invalid" in msg

    def test_duplicate_registration(self, fresh_db):
        auth.register("dave", "dave@x.com", "Password1!")
        ok, msg = auth.register("dave", "dave2@x.com", "Password1!")
        assert ok is False
        assert "taken" in msg.lower()

    def test_session_helpers(self, fresh_db):
        session = {}
        assert auth.get_session_user(session) is None
        auth.set_session_user(session, {"id": 1, "username": "alice"})
        assert auth.get_session_user(session)["username"] == "alice"
        auth.clear_session(session)
        assert auth.get_session_user(session) is None

    def test_password_not_stored_plaintext(self, fresh_db):
        auth.register("eve", "eve@x.com", "MySecret!")
        user = db.get_user_by_username("eve")
        assert user["password_hash"] != "MySecret!"
        assert len(user["password_hash"]) > 20


# ─────────────────────────────────────────────────────────────────────────────
# export.py
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_MEETING = {
    "id": 1,
    "title": "Q3 Planning",
    "filename": "q3.mp3",
    "created_at": 1756800000.0,
    "processed_at": 1756803600.0,
    "language": "en",
    "whisper_model": "base",
    "speaker_count": 3,
    "provider": "upload",
    "status": "done",
    "summary": "We discussed the Q3 roadmap and budget.",
    "decisions": ["Increase marketing spend", "Launch v2 in October"],
    "participants": ["Alice", "Bob", "Carol"],
    "key_points": ["Marketing budget approved", "v2 release date confirmed"],
    "topics": ["Roadmap", "Budget", "Launch"],
    "action_items": [
        {"task": "Prepare budget report", "assigned_to": "Alice",
         "deadline": "next Friday", "priority": "high", "status": "pending"},
        {"task": "Write release notes", "assigned_to": "Bob",
         "deadline": None, "priority": None, "status": "pending"},
    ],
    "all_deadlines": ["next Friday"],
    "transcript": "Alice: Good morning everyone. Bob: Let's start with the roadmap.",
}


class TestExport:

    def test_csv_returns_bytes(self):
        data = exp.export_csv(SAMPLE_MEETING)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_csv_contains_title(self):
        data = exp.export_csv(SAMPLE_MEETING).decode("utf-8-sig")
        assert "Q3 Planning" in data

    def test_csv_contains_summary(self):
        data = exp.export_csv(SAMPLE_MEETING).decode("utf-8-sig")
        assert "Q3 roadmap" in data

    def test_csv_contains_decisions(self):
        data = exp.export_csv(SAMPLE_MEETING).decode("utf-8-sig")
        assert "Increase marketing spend" in data

    def test_csv_contains_participants(self):
        data = exp.export_csv(SAMPLE_MEETING).decode("utf-8-sig")
        assert "Alice" in data

    def test_csv_action_items(self):
        data = exp.export_csv(SAMPLE_MEETING).decode("utf-8-sig")
        assert "Prepare budget report" in data
        assert "high" in data.lower()

    def test_csv_handles_none_values(self):
        m = dict(SAMPLE_MEETING)
        m["summary"] = None
        m["decisions"] = None
        data = exp.export_csv(m)
        assert isinstance(data, bytes)

    def test_csv_escapes_commas(self):
        m = dict(SAMPLE_MEETING)
        m["title"] = 'Meeting, "quoted", newline\ntest'
        data = exp.export_csv(m).decode("utf-8-sig")
        assert "Meeting" in data   # must not crash

    def test_pdf_returns_bytes(self):
        data = exp.export_pdf(SAMPLE_MEETING)
        assert isinstance(data, bytes)
        assert data[:4] == b"%PDF"   # valid PDF magic bytes

    def test_pdf_nonzero_size(self):
        data = exp.export_pdf(SAMPLE_MEETING)
        assert len(data) > 1000   # a real PDF with content is never trivially small

    def test_pdf_missing_fields(self):
        m = {"id": 1, "title": "Sparse", "status": "done",
             "filename": "s.mp3", "created_at": None}
        data = exp.export_pdf(m)
        assert data[:4] == b"%PDF"

    def test_safe_filename(self):
        assert exp.safe_filename("Q3 Planning!", "pdf") == "Q3_Planning_.pdf"
        assert exp.safe_filename("", "csv") == "meeting.csv"


# ─────────────────────────────────────────────────────────────────────────────
# rag_engine.py  (Gemini call mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAG:

    def test_index_meeting_stores_chunks(self, user_id, meeting_id):
        db.update_meeting_status(meeting_id, "done")
        n = rag_engine.index_meeting(
            meeting_id, user_id,
            "This is a test transcript. " * 30
        )
        assert n > 0
        chunks = db.get_all_chunks_for_user(user_id)
        assert len(chunks) == n

    def test_index_empty_transcript(self, user_id, meeting_id):
        n = rag_engine.index_meeting(meeting_id, user_id, "")
        assert n == 0

    def test_search_returns_results(self, user_id, meeting_id):
        db.update_meeting_status(meeting_id, "done")
        transcript = (
            "We decided to launch the product in October. "
            "Alice will handle marketing. Bob will write documentation. " * 10
        )
        rag_engine.index_meeting(meeting_id, user_id, transcript)
        results = rag_engine.search(user_id, "product launch October")
        assert len(results) > 0
        assert results[0]["meeting_id"] == meeting_id

    def test_search_empty_db(self, user_id):
        results = rag_engine.search(user_id, "anything")
        assert results == []

    def test_search_scoped_to_user(self, user_id, meeting_id):
        other_uid = db.create_user("mallory", "m@x.com", "hash")
        db.update_meeting_status(meeting_id, "done")
        rag_engine.index_meeting(
            meeting_id, user_id,
            "Secret company roadmap. " * 20
        )
        # Mallory has no meetings indexed
        results = rag_engine.search(other_uid, "roadmap")
        assert results == []

    def test_answer_no_sources(self, user_id):
        # No chunks indexed → should return no_sources=True without calling Gemini
        result = rag_engine.answer(user_id, "What is the capital of France?")
        assert result["no_sources"] is True
        assert result["error"] is None

    def test_answer_with_mocked_gemini(self, user_id, meeting_id):
        db.update_meeting_status(meeting_id, "done")
        rag_engine.index_meeting(
            meeting_id, user_id,
            "We decided to launch the API in November. Alice owns this task. " * 15
        )

        mock_interaction = MagicMock()
        mock_interaction.output_text = "The API will launch in November. [SOURCE 1]"
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}), \
             patch("google.genai.Client", return_value=mock_client):
            result = rag_engine.answer(user_id, "When is the API launching?")

        assert result["no_sources"] is False
        assert "November" in result["answer"]
        assert len(result["sources"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# zoom_adapter.py
# ─────────────────────────────────────────────────────────────────────────────

class TestZoomAdapter:

    def test_mock_recordings_returned_when_no_creds(self, monkeypatch):
        monkeypatch.setattr("zoom_adapter.LIVE_MODE", False)
        import zoom_adapter
        recs = zoom_adapter.list_recordings()
        assert len(recs) > 0
        assert all(r["mock"] is True for r in recs)

    def test_mock_recordings_have_required_fields(self):
        recs = zoom_mock_recs()
        for r in recs:
            for field in ("provider_id", "topic", "start_time", "file_size",
                          "file_type", "download_url"):
                assert field in r, f"Missing field: {field}"

    def test_download_raises_in_mock_mode(self, monkeypatch):
        monkeypatch.setattr("zoom_adapter.LIVE_MODE", False)
        import zoom_adapter
        with pytest.raises(RuntimeError, match="mock mode"):
            zoom_adapter.download_recording(zoom_mock_recs()[0])

    @patch("zoom_adapter.httpx")
    def test_list_recordings_live_parses_response(self, mock_httpx):
        """Test the response parsing logic with a mocked HTTP call."""
        import zoom_adapter

        token_resp = MagicMock()
        token_resp.status_code = 200
        token_resp.json.return_value = {"access_token": "fake-token"}

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {
            "meetings": [{
                "id": "111",
                "topic": "Test Meeting",
                "start_time": "2026-09-01T09:00:00Z",
                "duration": 45,
                "recording_files": [{
                    "id": "rec-001",
                    "status": "completed",
                    "file_type": "MP4",
                    "file_size": 10_000_000,
                    "download_url": "https://zoom.us/fake/download",
                }],
            }]
        }

        mock_httpx.post.return_value = token_resp
        mock_httpx.get.return_value = list_resp

        with patch.object(zoom_adapter, "LIVE_MODE", True), \
             patch.object(zoom_adapter, "_ACCOUNT_ID", "acc"), \
             patch.object(zoom_adapter, "_CLIENT_ID", "cid"), \
             patch.object(zoom_adapter, "_CLIENT_SECRET", "sec"), \
             patch.object(zoom_adapter, "_cached_token", None):
            recs = zoom_adapter.list_recordings()

        assert len(recs) == 1
        assert recs[0]["topic"] == "Test Meeting"
        assert recs[0]["provider_id"] == "rec-001"

    def test_provider_id_checksum_stable(self):
        import zoom_adapter
        rec = {"provider_id": "abc-123"}
        c1 = zoom_adapter.provider_id_checksum(rec)
        c2 = zoom_adapter.provider_id_checksum(rec)
        assert c1 == c2
        assert len(c1) == 16


# ─────────────────────────────────────────────────────────────────────────────
# google_meet_adapter.py
# ─────────────────────────────────────────────────────────────────────────────

class TestGoogleMeetAdapter:

    def test_mock_recordings_when_no_creds(self, monkeypatch):
        monkeypatch.setattr("google_meet_adapter.LIVE_MODE", False)
        import google_meet_adapter
        recs = google_meet_adapter.list_recordings()
        assert len(recs) > 0
        assert all(r["mock"] is True for r in recs)

    def test_mock_recordings_have_required_fields(self):
        recs = gmeet_mock_recs()
        for r in recs:
            for field in ("provider_id", "topic", "start_time", "file_size", "mime_type"):
                assert field in r

    def test_download_raises_in_mock_mode(self, monkeypatch):
        monkeypatch.setattr("google_meet_adapter.LIVE_MODE", False)
        import google_meet_adapter
        with pytest.raises(RuntimeError, match="mock mode"):
            google_meet_adapter.download_recording(gmeet_mock_recs()[0])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-user isolation integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossUserIsolation:

    def test_user_cannot_see_other_users_meetings(self, fresh_db):
        uid_a = db.create_user("userA", "a@x.com", auth.hash_password("pass"))
        uid_b = db.create_user("userB", "b@x.com", auth.hash_password("pass"))
        mid_a = db.create_meeting(uid_a, "Alice's Secret Meeting", "a.mp3")

        # userB lists meetings → should be empty
        b_meetings = db.list_meetings(uid_b)
        assert len(b_meetings) == 0

        # userB tries to get Alice's meeting by ID → None
        assert db.get_meeting(mid_a, uid_b) is None

        # userB tries to delete Alice's meeting → False
        assert db.delete_meeting(mid_a, uid_b) is False

    def test_rag_scoped_to_user(self, fresh_db):
        uid_a = db.create_user("searchA", "sa@x.com", auth.hash_password("pass"))
        uid_b = db.create_user("searchB", "sb@x.com", auth.hash_password("pass"))
        mid_a = db.create_meeting(uid_a, "A Meeting", "a.mp3")
        db.update_meeting_status(mid_a, "done")
        rag_engine.index_meeting(mid_a, uid_a, "Classified project delta information " * 20)

        # userB searches → gets nothing
        results = rag_engine.search(uid_b, "classified project delta")
        assert results == []

    def test_provider_duplicate_per_user(self, fresh_db):
        uid_a = db.create_user("pa", "pa@x.com", auth.hash_password("pass"))
        uid_b = db.create_user("pb", "pb@x.com", auth.hash_password("pass"))
        db.create_meeting(uid_a, "Zoom Rec", "z.mp4",
                           provider="zoom", provider_id="zoom-001")
        # Same provider_id for userB should NOT be flagged as duplicate
        assert db.provider_meeting_exists(uid_a, "zoom", "zoom-001") is True
        assert db.provider_meeting_exists(uid_b, "zoom", "zoom-001") is False


# ─────────────────────────────────────────────────────────────────────────────
# meeting_analyzer.py (existing, quick smoke-test)
# ─────────────────────────────────────────────────────────────────────────────

class TestMeetingAnalyzer:

    def test_analyze_returns_required_keys(self):
        from meeting_analyzer import analyze_transcript
        text = ("Alice will submit the budget report by next Friday. "
                "Bob should review the code urgently. "
                "We decided to proceed with the launch. " * 5)
        result = analyze_transcript(text, [])
        assert "speaker_count" in result
        assert "topics" in result
        assert "key_points" in result
        assert "action_items" in result
        assert "all_deadlines" in result

    def test_action_items_have_priority_status(self):
        from meeting_analyzer import extract_action_items
        items = extract_action_items(
            "Alice will submit the report urgently. "
            "Bob should review by next Monday."
        )
        assert all("priority" in i for i in items)
        assert all("status" in i for i in items)
        assert all(i["status"] == "pending" for i in items)


# ─────────────────────────────────────────────────────────────────────────────
# gemini_summary.py  (API call mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiSummary:

    def test_validate_transcript_rejects_non_string(self):
        from gemini_summary import validate_transcript
        with pytest.raises(TypeError):
            validate_transcript(123)

    def test_validate_transcript_rejects_short(self):
        from gemini_summary import validate_transcript
        with pytest.raises(ValueError):
            validate_transcript("hi")

    def test_validate_transcript_truncates(self):
        from gemini_summary import validate_transcript, _MAX_TRANSCRIPT_CHARS
        long_text = "word " * 200_000
        cleaned, warning = validate_transcript(long_text)
        assert len(cleaned) <= _MAX_TRANSCRIPT_CHARS + 1   # allow one char for sentence boundary
        assert warning is not None

    def test_normalise_response_coerces_invalid_priority(self):
        from gemini_summary import _normalise_response
        raw = {
            "summary": "ok",
            "key_points": [],
            "decisions": [],
            "participants": [],
            "topics": [],
            "action_items": [{
                "task": "Do it",
                "assigned_to": None,
                "deadline": None,
                "priority": "INVALID",
                "status": "INVALID",
            }],
        }
        result = _normalise_response(raw)
        assert result["action_items"][0]["priority"] is None
        assert result["action_items"][0]["status"] == "pending"

    def test_generate_meeting_summary_mocked(self):
        from gemini_summary import generate_meeting_summary

        fake_response = {
            "summary": "A great meeting about AI.",
            "key_points": ["Point 1"],
            "decisions": ["Go ahead"],
            "participants": ["Alice"],
            "topics": ["AI"],
            "action_items": [{
                "task": "Write report",
                "assigned_to": "Alice",
                "deadline": "Friday",
                "priority": "high",
                "status": "pending",
            }],
        }

        mock_interaction = MagicMock()
        mock_interaction.output_text = json.dumps(fake_response)
        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}), \
             patch("gemini_summary._get_client", return_value=mock_client):
            result = generate_meeting_summary(
                "Alice and Bob discussed AI development priorities for the next quarter. " * 5
            )

        assert result["summary"] == "A great meeting about AI."
        assert result["decisions"] == ["Go ahead"]
        assert result["participants"] == ["Alice"]
        assert result["action_items"][0]["priority"] == "high"
