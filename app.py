"""
app.py
------
Milestone 4 — Full Dashboard
Multi-page Streamlit application.

Pages (via sidebar navigation):
  🔐 Login / Register
  🏠 Dashboard        — meeting list, search, filters
  📋 Meeting Details  — transcript, summary, analysis, exports
  🤖 AI Assistant     — RAG search over all user meetings
  🔗 Import           — Zoom + Google Meet recording import

All pages are user-scoped. Unauthenticated requests cannot reach private data.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Bootstrap DB on every startup ─────────────────────────────────────────────
import database as db
db.init_db()

import auth
import export as exp
from audio_processor import validate_audio_file, save_uploaded_file, cleanup_temp_file
from transcriber import transcribe_audio, WHISPER_MODELS
from meeting_analyzer import analyze_transcript
from gemini_summary import generate_meeting_summary
import rag_engine
import zoom_adapter
import google_meet_adapter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.meeting-card {
    background: #f8faff;
    border: 1px solid #dde3f0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-done     { background:#2ca02c; color:white; }
.badge-pending  { background:#aaa;    color:white; }
.badge-proc     { background:#1f77b4; color:white; }
.badge-failed   { background:#d62728; color:white; }
.badge-high     { background:#d62728; color:white; }
.badge-medium   { background:#ff7f0e; color:white; }
.badge-low      { background:#2ca02c; color:white; }
.section-head {
    font-size: 1.05rem;
    font-weight: 700;
    color: #2c3e70;
    margin-top: 18px;
    margin-bottom: 6px;
    border-bottom: 2px solid #dde3f0;
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "—"


def _status_badge(status: str) -> str:
    cls = {"done": "badge-done", "pending": "badge-pending",
           "processing": "badge-proc", "failed": "badge-failed"}.get(status, "badge-pending")
    return f"<span class='badge {cls}'>{status.upper()}</span>"


def _priority_badge(p: str | None) -> str:
    if not p:
        return ""
    cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(p.lower(), "")
    return f"<span class='badge {cls}'>{p.upper()}</span>"


def _render_action_items(items: list[dict]) -> None:
    if not items:
        st.info("No action items.")
        return
    for item in items:
        with st.container(border=True):
            badges = _priority_badge(item.get("priority"))
            status_cls = "badge-done" if item.get("status") == "completed" else "badge-pending"
            badges += f"<span class='badge {status_cls}'>{(item.get('status','pending')).upper()}</span>"
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown(f"**Task:** {item.get('task','')}")
            c1, c2 = st.columns(2)
            c1.markdown(f"👤 **Assigned:** {item.get('assigned_to') or '—'}")
            c2.markdown(f"📅 **Deadline:** {item.get('deadline') or '—'}")


def _process_meeting(meeting_id: int, user_id: int,
                     tmp_path: str, whisper_model: str,
                     run_gemini: bool = True) -> None:
    """
    Full processing pipeline: transcribe → analyze → gemini → index.
    Updates the meeting record at each stage.
    Cleans up tmp_path on completion or failure.
    """
    try:
        db.update_meeting_status(meeting_id, "processing")

        # 1. Transcribe
        result = transcribe_audio(tmp_path, model_size=whisper_model)
        transcript    = result["text"]
        language      = result["language"]
        segments      = result["segments"]
        duration      = segments[-1]["end"] if segments else None

        db.save_transcript(
            meeting_id, transcript, language, whisper_model,
            duration, None,   # speaker_count filled by analysis
        )

        # 2. Python analysis
        analysis = analyze_transcript(transcript, segments)

        # 3. Gemini (optional)
        gemini_result = None
        if run_gemini:
            try:
                gemini_result = generate_meeting_summary(transcript)
            except Exception:
                gemini_result = None   # Gemini failure must not fail the meeting

        db.save_analysis(meeting_id, analysis, gemini_result)

        # 4. Index for RAG
        try:
            rag_engine.index_meeting(meeting_id, user_id, transcript)
        except Exception:
            pass   # RAG indexing failure doesn't block the meeting

    except Exception as exc:
        db.update_meeting_status(meeting_id, "failed", str(exc))
        raise
    finally:
        from audio_processor import cleanup_temp_file
        cleanup_temp_file(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar() -> str:
    with st.sidebar:
        st.title("🎙️ Meeting Intelligence")
        st.markdown("---")

        user = auth.get_session_user(st.session_state)
        if user:
            st.markdown(f"👤 **{user['username']}**")
            st.markdown(f"<small>{user['email']}</small>", unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True):
                auth.clear_session(st.session_state)
                st.rerun()
            st.markdown("---")

        pages = {
            "🔐 Login / Register": "auth",
            "🏠 Dashboard":         "dashboard",
            "📋 Meeting Details":   "detail",
            "🤖 AI Assistant":      "rag",
            "🔗 Import Recordings": "import",
        }
        if user:
            pages.pop("🔐 Login / Register")
        else:
            pages = {"🔐 Login / Register": "auth"}

        choice = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption("Milestone 4 · Whisper + Gemini 3.5 Flash")
        return pages[choice]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Auth
# ══════════════════════════════════════════════════════════════════════════════

def page_auth() -> None:
    st.title("🔐 Welcome to Meeting Intelligence")
    tab_login, tab_reg = st.tabs(["Login", "Register"])

    with tab_login:
        st.subheader("Sign in")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
        if submitted:
            if not username or not password:
                st.error("Please enter your username and password.")
            else:
                ok, result = auth.login(username, password)
                if ok:
                    auth.set_session_user(st.session_state, result)
                    st.success(f"Welcome back, {result['username']}!")
                    st.rerun()
                else:
                    st.error(result)

    with tab_reg:
        st.subheader("Create account")
        with st.form("reg_form"):
            r_user  = st.text_input("Username")
            r_email = st.text_input("Email")
            r_pass  = st.text_input("Password", type="password")
            r_conf  = st.text_input("Confirm password", type="password")
            r_sub   = st.form_submit_button("Register", use_container_width=True, type="primary")
        if r_sub:
            errors = auth.validate_registration(r_user, r_email, r_pass, r_conf)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                ok, msg = auth.register(r_user, r_email, r_pass)
                if ok:
                    st.success("Account created! Please log in.")
                else:
                    st.error(msg)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard(user: dict) -> None:
    st.title("🏠 Dashboard")

    # ── Upload new meeting ─────────────────────────────────────────────────────
    with st.expander("➕ Upload new meeting recording", expanded=False):
        with st.form("upload_form"):
            title       = st.text_input("Meeting title", placeholder="e.g. Weekly Standup — Sep 2")
            up_file     = st.file_uploader(
                "Audio/video file",
                type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
            )
            col_m, col_g = st.columns(2)
            w_model     = col_m.selectbox("Whisper model", WHISPER_MODELS, index=1)
            use_gemini  = col_g.checkbox("Run Gemini AI summary", value=True)
            up_sub      = st.form_submit_button("🚀 Upload & Process", type="primary",
                                                use_container_width=True)

        if up_sub:
            if not title.strip():
                st.error("Please enter a meeting title.")
            elif up_file is None:
                st.error("Please select a file.")
            else:
                is_valid, msg = validate_audio_file(up_file)
                if not is_valid:
                    st.error(f"❌ {msg}")
                else:
                    up_file.seek(0)
                    tmp = save_uploaded_file(up_file)
                    mid = db.create_meeting(
                        user_id=user["id"],
                        title=title.strip(),
                        filename=up_file.name,
                        file_size_bytes=up_file.size,
                        provider="upload",
                    )
                    with st.spinner("Processing… (this may take a minute)"):
                        try:
                            _process_meeting(mid, user["id"], tmp, w_model, use_gemini)
                            st.success("✅ Meeting processed successfully!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"❌ Processing failed: {exc}")

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_s, col_f = st.columns([3, 1])
    search_q  = col_s.text_input("🔍 Search meetings", placeholder="Search title or transcript…",
                                  label_visibility="collapsed")
    status_f  = col_f.selectbox("Filter", ["All", "done", "processing", "pending", "failed"],
                                 label_visibility="collapsed")

    # ── Meeting list ──────────────────────────────────────────────────────────
    meetings = db.list_meetings(
        user_id=user["id"],
        status=None if status_f == "All" else status_f,
        search=search_q.strip() or None,
    )

    if not meetings:
        st.info("No meetings found. Upload your first recording above.")
        return

    st.markdown(f"**{len(meetings)} meeting(s)**")

    for m in meetings:
        with st.container():
            st.markdown(f"""
            <div class='meeting-card'>
                <b>{m['title']}</b>
                {_status_badge(m['status'])}
                <span style='color:#888;font-size:0.85rem;float:right'>{_fmt_ts(m['created_at'])}</span><br>
                <small>📁 {m['filename']}
                {"&nbsp;&nbsp;🌐 " + m['language'].upper() if m.get('language') else ""}
                {"&nbsp;&nbsp;👥 " + str(m['speaker_count']) + " speakers" if m.get('speaker_count') else ""}
                {"&nbsp;&nbsp;🔗 " + m['provider'].title() if m.get('provider') else ""}
                </small>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 1, 1])
            if c1.button("📋 View Details", key=f"view_{m['id']}", use_container_width=True):
                st.session_state["selected_meeting_id"] = m["id"]
                st.session_state["nav_override"] = "detail"
                st.rerun()
            if c2.button("🗑️ Delete", key=f"del_{m['id']}", use_container_width=True):
                db.delete_meeting(m["id"], user["id"])
                st.rerun()
            if m["status"] == "failed" and c3.button("🔄 Retry", key=f"retry_{m['id']}", use_container_width=True):
                # Re-create a fresh meeting record for retry
                st.info("Re-upload the file to retry processing.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Meeting Detail
# ══════════════════════════════════════════════════════════════════════════════

def page_detail(user: dict) -> None:
    mid = st.session_state.get("selected_meeting_id")
    if not mid:
        st.warning("No meeting selected. Go to the Dashboard and click 'View Details'.")
        return

    meeting = db.get_meeting(mid, user["id"])
    if not meeting:
        st.error("Meeting not found or access denied.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.title(f"📋 {meeting['title']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status",   meeting["status"].upper())
    col2.metric("Language", (meeting.get("language") or "—").upper())
    col3.metric("Speakers", meeting.get("speaker_count") or "—")
    col4.metric("Provider", (meeting.get("provider") or "upload").title())

    st.markdown(f"**Created:** {_fmt_ts(meeting.get('created_at'))}  |  "
                f"**Processed:** {_fmt_ts(meeting.get('processed_at'))}  |  "
                f"**File:** {meeting.get('filename','—')}")

    if meeting["status"] == "failed":
        st.error(f"Processing failed: {meeting.get('error_message','unknown error')}")
        return
    if meeting["status"] in ("pending", "processing"):
        st.info("⏳ This meeting is still being processed. Refresh in a moment.")
        return

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("---")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        pdf_bytes = exp.export_pdf(meeting)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=exp.safe_filename(meeting["title"], "pdf"),
            mime="application/pdf",
            use_container_width=True,
        )
    with exp_col2:
        csv_bytes = exp.export_csv(meeting)
        st.download_button(
            "📊 Download CSV Report",
            data=csv_bytes,
            file_name=exp.safe_filename(meeting["title"], "csv"),
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # ── Summary ───────────────────────────────────────────────────────────────
    if meeting.get("summary"):
        st.markdown("<div class='section-head'>📝 Summary</div>", unsafe_allow_html=True)
        st.markdown(meeting["summary"])

    # ── Decisions ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>🎯 Key Decisions</div>", unsafe_allow_html=True)
    decisions = meeting.get("decisions") or []
    if decisions:
        for d in decisions:
            st.markdown(f"- {d}")
    else:
        st.info("No decisions recorded.")

    # ── Participants ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>🙋 Participants</div>", unsafe_allow_html=True)
    participants = meeting.get("participants") or []
    if participants:
        st.markdown("  ".join(
            f"<span style='background:#e377c2;color:white;padding:3px 9px;"
            f"border-radius:10px;font-size:0.85rem;'>{p}</span>"
            for p in participants
        ), unsafe_allow_html=True)
        st.markdown(" ")
    else:
        st.info("No participants identified.")

    # ── Key Points ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>💡 Key Discussion Points</div>", unsafe_allow_html=True)
    kp = meeting.get("key_points") or []
    if kp:
        for i, p in enumerate(kp, 1):
            st.markdown(f"**{i}.** {p}")
    else:
        st.info("No key points recorded.")

    # ── Topics ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>📌 Topics</div>", unsafe_allow_html=True)
    topics = meeting.get("topics") or []
    if topics:
        cols = st.columns(min(len(topics), 5))
        for i, t in enumerate(topics):
            cols[i % 5].markdown(
                f"<span style='background:#1f77b4;color:white;padding:4px 10px;"
                f"border-radius:12px;font-size:0.85rem;'>{t}</span>",
                unsafe_allow_html=True,
            )
        st.markdown(" ")
    else:
        st.info("No topics extracted.")

    # ── Action Items ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>✅ Action Items</div>", unsafe_allow_html=True)
    _render_action_items(meeting.get("action_items") or [])

    # ── All Deadlines ─────────────────────────────────────────────────────────
    deadlines = meeting.get("all_deadlines") or []
    if deadlines:
        with st.expander(f"📅 All dates/deadlines found ({len(deadlines)})"):
            for d in deadlines:
                st.markdown(f"• {d}")

    # ── Transcript ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>📜 Full Transcript</div>", unsafe_allow_html=True)
    if meeting.get("transcript"):
        st.text_area("Transcript", value=meeting["transcript"], height=300,
                     label_visibility="collapsed")
    else:
        st.info("Transcript not available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI Assistant (RAG)
# ══════════════════════════════════════════════════════════════════════════════

def page_rag(user: dict) -> None:
    st.title("🤖 AI Meeting Assistant")
    st.caption("Ask questions about any of your meetings. "
               "Answers are grounded only in your recorded meetings.")

    query = st.text_input("Ask a question about your meetings",
                          placeholder="e.g. What were the decisions about the API integration?")

    if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your meetings and generating answer…"):
                result = rag_engine.answer(user["id"], query.strip())

            st.markdown("---")

            if result.get("error"):
                st.error(f"Error: {result['error']}")

            if result.get("no_sources"):
                st.warning("🔍 No relevant meeting content found for this question.")
            else:
                st.markdown("### 💬 Answer")
                st.markdown(result["answer"])

                sources = result.get("sources", [])
                if sources:
                    st.markdown("### 📚 Sources")
                    for s in sources:
                        with st.container(border=True):
                            date_str = _fmt_ts(s.get("meeting_date"))
                            score_pct = f"{s['score']*100:.0f}% match"
                            st.markdown(
                                f"**{s['meeting_title']}** — {date_str} "
                                f"<span style='color:#888;font-size:0.8rem;'>({score_pct})</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"> *{s['excerpt'][:400]}…*")

    # ── Re-index button ────────────────────────────────────────────────────────
    with st.expander("⚙️ Re-index all meetings"):
        st.caption("Run this if search results seem stale after re-processing a meeting.")
        if st.button("🔄 Re-index now"):
            meetings = db.list_meetings(user["id"], status="done")
            count = 0
            for m in meetings:
                if m.get("transcript"):
                    rag_engine.index_meeting(m["id"], user["id"], m["transcript"])
                    count += 1
            st.success(f"Re-indexed {count} meeting(s).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Import (Zoom + Google Meet)
# ══════════════════════════════════════════════════════════════════════════════

def page_import(user: dict) -> None:
    st.title("🔗 Import Recordings")

    tab_zoom, tab_gmeet = st.tabs(["Zoom", "Google Meet"])

    # ── ZOOM ──────────────────────────────────────────────────────────────────
    with tab_zoom:
        if zoom_adapter.LIVE_MODE:
            st.success("✅ Zoom credentials configured — live mode active.")
        else:
            st.warning(
                "⚠️ Zoom credentials not configured. Showing demo recordings.\n\n"
                "Add `ZOOM_ACCOUNT_ID`, `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET` to your `.env` "
                "to connect your real Zoom account."
            )

        if st.button("🔄 Refresh Zoom recordings", use_container_width=True):
            st.session_state.pop("zoom_recs", None)

        if "zoom_recs" not in st.session_state:
            with st.spinner("Fetching Zoom recordings…"):
                try:
                    st.session_state["zoom_recs"] = zoom_adapter.list_recordings()
                except Exception as exc:
                    st.error(f"Failed to fetch Zoom recordings: {exc}")
                    st.session_state["zoom_recs"] = []

        recs = st.session_state.get("zoom_recs", [])
        if not recs:
            st.info("No Zoom recordings found.")
        else:
            col_m, col_g = st.columns(2)
            w_model    = col_m.selectbox("Whisper model", WHISPER_MODELS, index=1, key="zoom_model")
            use_gemini = col_g.checkbox("Run Gemini AI summary", value=True, key="zoom_gemini")

            for rec in recs:
                with st.container(border=True):
                    mock_label = " 🎭 DEMO" if rec.get("mock") else ""
                    st.markdown(f"**{rec['topic']}**{mock_label}")
                    st.markdown(
                        f"📅 {rec['start_time'][:10]}  |  "
                        f"⏱ {rec['duration']} min  |  "
                        f"📦 {rec['file_size']//1024//1024} MB  |  "
                        f"🎞 {rec['file_type']}"
                    )

                    already = db.provider_meeting_exists(user["id"], "zoom", rec["provider_id"])
                    if already:
                        st.success("✅ Already imported")
                    elif rec.get("mock"):
                        st.info("Demo recording — cannot download without real Zoom credentials.")
                    else:
                        if st.button("⬇️ Import", key=f"zoom_{rec['provider_id']}",
                                     use_container_width=True):
                            mid = db.create_meeting(
                                user_id=user["id"],
                                title=rec["topic"],
                                filename=f"{rec['topic']}.{rec['file_type'].lower()}",
                                file_size_bytes=rec["file_size"],
                                provider="zoom",
                                provider_id=rec["provider_id"],
                                provider_meta=rec,
                            )
                            with st.spinner("Downloading and processing…"):
                                try:
                                    tmp, fname = zoom_adapter.download_recording(rec)
                                    _process_meeting(mid, user["id"], tmp, w_model, use_gemini)
                                    st.success("✅ Imported successfully!")
                                    st.rerun()
                                except Exception as exc:
                                    db.update_meeting_status(mid, "failed", str(exc))
                                    st.error(f"Import failed: {exc}")

    # ── GOOGLE MEET ───────────────────────────────────────────────────────────
    with tab_gmeet:
        if google_meet_adapter.LIVE_MODE:
            st.success("✅ Google credentials configured — live mode active.")
        else:
            st.warning(
                "⚠️ Google credentials not configured. Showing demo recordings.\n\n"
                "To connect Google Meet, run:\n"
                "```\npython google_meet_adapter.py --auth\n```\n"
                "Then add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, "
                "`GOOGLE_REFRESH_TOKEN` to your `.env`."
            )

        if st.button("🔄 Refresh Google Meet recordings", use_container_width=True):
            st.session_state.pop("gmeet_recs", None)

        if "gmeet_recs" not in st.session_state:
            with st.spinner("Fetching Google Meet recordings…"):
                try:
                    st.session_state["gmeet_recs"] = google_meet_adapter.list_recordings()
                except Exception as exc:
                    st.error(f"Failed to fetch Google Meet recordings: {exc}")
                    st.session_state["gmeet_recs"] = []

        g_recs = st.session_state.get("gmeet_recs", [])
        if not g_recs:
            st.info("No Google Meet recordings found.")
        else:
            gcol_m, gcol_g = st.columns(2)
            gw_model    = gcol_m.selectbox("Whisper model", WHISPER_MODELS, index=1, key="gm_model")
            guse_gemini = gcol_g.checkbox("Run Gemini AI summary", value=True, key="gm_gemini")

            for rec in g_recs:
                with st.container(border=True):
                    mock_label = " 🎭 DEMO" if rec.get("mock") else ""
                    st.markdown(f"**{rec['topic']}**{mock_label}")
                    st.markdown(
                        f"📅 {rec['start_time'][:10]}  |  "
                        f"📦 {rec['file_size']//1024//1024} MB"
                    )

                    already = db.provider_meeting_exists(user["id"], "google_meet", rec["provider_id"])
                    if already:
                        st.success("✅ Already imported")
                    elif rec.get("mock"):
                        st.info("Demo recording — cannot download without Google credentials.")
                    else:
                        if st.button("⬇️ Import", key=f"gmeet_{rec['provider_id']}",
                                     use_container_width=True):
                            mid = db.create_meeting(
                                user_id=user["id"],
                                title=rec["topic"],
                                filename=f"{rec['topic']}.mp4",
                                file_size_bytes=rec["file_size"],
                                provider="google_meet",
                                provider_id=rec["provider_id"],
                                provider_meta=rec,
                            )
                            with st.spinner("Downloading and processing…"):
                                try:
                                    tmp, fname = google_meet_adapter.download_recording(rec)
                                    _process_meeting(mid, user["id"], tmp, gw_model, guse_gemini)
                                    st.success("✅ Imported successfully!")
                                    st.rerun()
                                except Exception as exc:
                                    db.update_meeting_status(mid, "failed", str(exc))
                                    st.error(f"Import failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Main router
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    page = _sidebar()

    # Allow detail page to be triggered from dashboard
    override = st.session_state.pop("nav_override", None)
    if override:
        page = override

    user = auth.get_session_user(st.session_state)

    if page == "auth" or not user:
        page_auth()
        return

    if page == "dashboard":
        page_dashboard(user)
    elif page == "detail":
        page_detail(user)
    elif page == "rag":
        page_rag(user)
    elif page == "import":
        page_import(user)


if __name__ == "__main__":
    main()
