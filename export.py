"""
export.py
---------
PDF and CSV export for a single meeting (Milestone 4).

Both exports:
- Use only the authorized meeting dict (already user-scoped by the caller)
- Handle missing/null values gracefully
- Encode all user content safely (no injection in CSV, no crashes in PDF)
- Return bytes that can be handed directly to st.download_button()
"""

from __future__ import annotations

import csv
import io
import time
from datetime import datetime
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return "-"


def _safe(val: Any, fallback: str = "-") -> str:
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s else fallback


def _list_lines(items: list | None, field: str | None = None) -> list[str]:
    """Flatten a list of strings or dicts to a list of display strings."""
    if not items:
        return []
    result = []
    for item in items:
        if isinstance(item, dict):
            if field:
                result.append(_safe(item.get(field, "")))
            else:
                # Action item: task | assigned | deadline | priority | status
                parts = [_safe(item.get("task", ""))]
                if item.get("assigned_to"):
                    parts.append(f"→ {item['assigned_to']}")
                if item.get("deadline"):
                    parts.append(f"by {item['deadline']}")
                if item.get("priority"):
                    parts.append(f"[{item['priority'].upper()}]")
                result.append("  ".join(parts))
        else:
            result.append(_safe(item))
    return result


def safe_filename(title: str, ext: str) -> str:
    """Turn a meeting title into a safe filename."""
    clean = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
    clean = clean.strip().replace(" ", "_")[:60] or "meeting"
    return f"{clean}.{ext}"


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(meeting: dict) -> bytes:
    """
    Return UTF-8 CSV bytes for the meeting.
    Fields with commas, quotes, or newlines are correctly encoded by the
    csv module (quoting=csv.QUOTE_ALL).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")

    # Header row
    writer.writerow(["Field", "Value"])

    # ── Metadata ──────────────────────────────────────────────────────────────
    writer.writerow(["Meeting Title",    _safe(meeting.get("title"))])
    writer.writerow(["Date Created",     _fmt_ts(meeting.get("created_at"))])
    writer.writerow(["Date Processed",   _fmt_ts(meeting.get("processed_at"))])
    writer.writerow(["Language",         _safe(meeting.get("language"))])
    writer.writerow(["Whisper Model",    _safe(meeting.get("whisper_model"))])
    writer.writerow(["Estimated Speakers", _safe(meeting.get("speaker_count"))])
    writer.writerow(["Provider",         _safe(meeting.get("provider"))])
    writer.writerow(["Status",           _safe(meeting.get("status"))])
    writer.writerow([])

    # ── Summary ───────────────────────────────────────────────────────────────
    writer.writerow(["Summary", _safe(meeting.get("summary"), fallback="")])
    writer.writerow([])

    # ── Decisions ─────────────────────────────────────────────────────────────
    decisions = _list_lines(meeting.get("decisions"))
    writer.writerow(["Key Decisions", ""])
    for d in decisions:
        writer.writerow(["", d])
    if not decisions:
        writer.writerow(["", "—"])
    writer.writerow([])

    # ── Participants ──────────────────────────────────────────────────────────
    participants = _list_lines(meeting.get("participants"))
    writer.writerow(["Participants", ""])
    for p in participants:
        writer.writerow(["", p])
    if not participants:
        writer.writerow(["", "—"])
    writer.writerow([])

    # ── Key Points ────────────────────────────────────────────────────────────
    kp = _list_lines(meeting.get("key_points"))
    writer.writerow(["Key Points", ""])
    for k in kp:
        writer.writerow(["", k])
    if not kp:
        writer.writerow(["", "—"])
    writer.writerow([])

    # ── Topics ────────────────────────────────────────────────────────────────
    topics = _list_lines(meeting.get("topics"))
    writer.writerow(["Topics", ""])
    for t in topics:
        writer.writerow(["", t])
    if not topics:
        writer.writerow(["", "—"])
    writer.writerow([])

    # ── Action Items (one row per item) ───────────────────────────────────────
    writer.writerow(["Action Items", "Task", "Assigned To", "Deadline", "Priority", "Status"])
    action_items = meeting.get("action_items") or []
    if action_items:
        for item in action_items:
            writer.writerow([
                "",
                _safe(item.get("task")),
                _safe(item.get("assigned_to")),
                _safe(item.get("deadline")),
                _safe(item.get("priority")),
                _safe(item.get("status", "pending")),
            ])
    else:
        writer.writerow(["", "—", "—", "—", "—", "—"])
    writer.writerow([])

    # ── Transcript (last — can be long) ───────────────────────────────────────
    writer.writerow(["Full Transcript", _safe(meeting.get("transcript"), fallback="")])

    return buf.getvalue().encode("utf-8-sig")   # BOM for Excel compatibility


# ── PDF export ────────────────────────────────────────────────────────────────

def export_pdf(meeting: dict) -> bytes:
    """
    Return PDF bytes for the meeting using fpdf2.
    Layout: title page info → summary → decisions → participants →
            key points → action items table → transcript.
    """
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Meeting Intelligence Report", align="R")
            self.ln(2)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(14, 16, 14)

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    title = _safe(meeting.get("title"), "Meeting Report")
    pdf.multi_cell(0, 10, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    meta_lines = [
        f"Date: {_fmt_ts(meeting.get('created_at'))}",
        f"Language: {_safe(meeting.get('language'))}  |  "
        f"Model: {_safe(meeting.get('whisper_model'))}  |  "
        f"Provider: {_safe(meeting.get('provider'))}",
        f"Speakers (estimated): {_safe(meeting.get('speaker_count'))}  |  "
        f"Status: {_safe(meeting.get('status'))}",
    ]
    for line in meta_lines:
        pdf.cell(0, 5, line)
        pdf.ln(5)
    pdf.ln(3)
    pdf.set_draw_color(60, 120, 200)
    pdf.set_line_width(0.5)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(6)

    def section(heading: str):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(40, 80, 160)
        pdf.cell(0, 7, heading)
        pdf.ln(7)
        pdf.set_text_color(30, 30, 30)

    def body(text: str, size: int = 10):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(40, 40, 40)
        # Replace characters fpdf2 can't encode in latin-1
        safe = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe)
        pdf.ln(2)

    def bullet_list(items: list[str]):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for item in items:
            safe = item.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, f"*  {safe}")   # * instead of bullet (latin-1 safe)
            pdf.ln(1)
        pdf.ln(2)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _safe(meeting.get("summary"), fallback="")
    if summary and summary != "—":
        section("Summary")
        body(summary)

    # ── Decisions ─────────────────────────────────────────────────────────────
    decisions = _list_lines(meeting.get("decisions"))
    section("Key Decisions")
    if decisions:
        bullet_list(decisions)
    else:
        body("No decisions recorded.")

    # ── Participants ──────────────────────────────────────────────────────────
    participants = _list_lines(meeting.get("participants"))
    section("Participants")
    if participants:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 6, "  |  ".join(participants))
        pdf.ln(8)
    else:
        body("No participants identified.")

    # ── Key Points ────────────────────────────────────────────────────────────
    kp = _list_lines(meeting.get("key_points"))
    section("Key Discussion Points")
    if kp:
        for i, point in enumerate(kp, 1):
            safe = point.encode("latin-1", errors="replace").decode("latin-1")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5, f"{i}.  {safe}")
            pdf.ln(1)
        pdf.ln(2)
    else:
        body("No key points recorded.")

    # ── Action Items table ────────────────────────────────────────────────────
    section("Action Items")
    action_items = meeting.get("action_items") or []
    if action_items:
        col_w = [75, 33, 30, 20, 22]
        headers = ["Task", "Assigned To", "Deadline", "Priority", "Status"]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(230, 238, 255)
        pdf.set_text_color(30, 30, 90)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, border=1, fill=True)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        for item in action_items:
            row = [
                _safe(item.get("task"))[:80],
                _safe(item.get("assigned_to"))[:25],
                _safe(item.get("deadline"))[:20],
                _safe(item.get("priority"))[:10],
                _safe(item.get("status", "pending"))[:12],
            ]
            row_h = 6
            pdf.set_fill_color(248, 250, 255)
            for i, cell in enumerate(row):
                safe = cell.encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(col_w[i], row_h, safe, border=1)
            pdf.ln(row_h)
        pdf.ln(4)
    else:
        body("No action items recorded.")

    # ── Topics ────────────────────────────────────────────────────────────────
    topics = _list_lines(meeting.get("topics"))
    if topics:
        section("Topics")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 6, "  |  ".join(topics))
        pdf.ln(8)

    # ── Transcript ────────────────────────────────────────────────────────────
    transcript = _safe(meeting.get("transcript"), fallback="")
    if transcript and transcript != "—":
        pdf.add_page()
        section("Full Transcript")
        body(transcript, size=9)

    return bytes(pdf.output())
