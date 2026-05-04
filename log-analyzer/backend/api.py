"""
FastAPI backend — upload, process, query log files.
Single AppState singleton; processing runs in a background thread.
WebSocket /ws/chat streams agent or RAG responses.
"""
import asyncio
import json
import os
import shutil
import threading
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.analyzer import build_summary
from backend.agent import run_agent_stream, run_rag_stream
from backend.parser import parse_log_file
from backend.rag import RAGStore, build_chunks, build_index, load_index
from backend.providers import (
    make_async_client, make_sync_client, get_models_list,
    DEFAULT_PROVIDER, DEFAULT_MODEL, MODELS, Provider,
)
from backend.report import generate_report_sections, SECTION_TITLES

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR.mkdir(exist_ok=True)

UPLOAD_PATH = DATA_DIR / "current.log"

# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.events:        list[dict]      = []
        self.summary:       Optional[dict]  = None
        self.rag_store:     Optional[RAGStore] = None
        self.log_filename:  str             = ""
        self.is_processing: bool            = False
        self.processing_step: str           = ""
        self.processing_detail: list[str]   = []  # live sub-step log lines
        self.processing_error: str          = ""
        self.conversation_history: list[dict] = []

    def log(self, msg: str):
        """Append a detail line (thread-safe for simple appends in CPython)."""
        self.processing_detail.append(msg)

    def reset(self):
        self.__init__()


_state = AppState()

# ── Client helpers ────────────────────────────────────────────────────────────

def _resolve_model(model_id: str | None) -> tuple[str, "Provider", bool]:
    """Return (model_id, provider, supports_tools) for the given model_id."""
    mid = model_id or DEFAULT_MODEL
    for m in MODELS:
        if m.id == mid:
            return m.id, m.provider, m.supports_tools
    # Fallback: treat as groq model
    return mid, DEFAULT_PROVIDER, True

# ── Background processing ─────────────────────────────────────────────────────

def _process_log(path: Path):
    """Parse → analyse → build RAG. Runs in a daemon thread."""
    state = _state
    client = make_sync_client(DEFAULT_PROVIDER)
    try:
        state.processing_step = "Parsing log file…"
        state.log("Reading and parsing log file…")
        events, profile = parse_log_file(path)
        state.events = events
        state.log(f"Parsed {len(events):,} events  ·  format: {profile.get('name', 'unknown')}")

        state.processing_step = "Characterising log with AI…"
        state.log("Running AI characterisation (detecting log type, entities, prompts)…")
        summary = build_summary(events, profile, client)
        state.summary = summary
        char = summary.get("characterization", {})
        state.log(f"Detected: {char.get('log_type', 'unknown log type')}")

        state.processing_step = "Building RAG index… (chunking)"
        state.log("Splitting log into semantic chunks…")
        chunks = build_chunks(events, char)
        type_counts: dict = {}
        for c in chunks:
            t = c.get("chunk_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        state.log(f"Created {len(chunks)} chunks  ·  " + "  ".join(f"{t}: {n}" for t, n in type_counts.items()))

        state.processing_step = "Building RAG index… (loading model)"
        state.log("Loading sentence embedding model (all-MiniLM-L6-v2)…")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        state.log("Embedding model ready.")

        state.processing_step = "Building RAG index… (encoding)"
        state.log(f"Encoding {len(chunks)} chunks into vectors…")

        def _on_batch(done: int, total: int):
            pct = int(done / total * 100)
            state.log(f"Embedding batches: {done}/{total}  ({pct}%)")
            state.processing_step = f"Building RAG index… ({pct}% encoded)"

        index = build_index(chunks, embed_model, progress_cb=_on_batch)
        state.log(f"FAISS index built  ·  {index.ntotal} vectors  ·  dim={index.d}")

        state.rag_store = RAGStore(index, chunks, embed_model)
        state.log("RAG store ready — you can start chatting!")

        state.processing_step = "Ready"
        state.is_processing = False

    except Exception:
        state.processing_error = traceback.format_exc()
        state.log("Error: " + state.processing_error.splitlines()[-1])
        state.processing_step  = "Error"
        state.is_processing    = False


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Log Analyzer", version="1.0")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_log(file: UploadFile = File(...)):
    """Accept a log file, kick off background processing."""
    if _state.is_processing:
        return JSONResponse({"error": "Processing in progress"}, status_code=409)

    _state.reset()
    _state.is_processing = True
    _state.log_filename  = file.filename or "log"
    _state.processing_step = "Uploading…"

    # Save to disk
    with open(UPLOAD_PATH, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Start background thread
    t = threading.Thread(target=_process_log, args=(UPLOAD_PATH,), daemon=True)
    t.start()

    return {"status": "processing", "filename": _state.log_filename}


@app.get("/api/status")
async def get_status():
    return {
        "is_processing":    _state.is_processing,
        "step":             _state.processing_step,
        "detail":           list(_state.processing_detail),
        "error":            _state.processing_error,
        "has_summary":      _state.summary is not None,
        "has_rag":          _state.rag_store is not None,
        "filename":         _state.log_filename,
        "total_events":     len(_state.events),
        "rag_chunk_count":  _state.rag_store.chunk_count if _state.rag_store else 0,
    }


@app.get("/api/summary")
async def get_summary():
    if not _state.summary:
        return JSONResponse({"error": "No log loaded"}, status_code=404)
    return _state.summary


@app.get("/api/events")
async def get_events(
    page:    int = Query(1, ge=1),
    limit:   int = Query(50, ge=1, le=500),
    level:   str = Query(""),
    keyword: str = Query(""),
    ts_from: str = Query(""),
    ts_to:   str = Query(""),
):
    if not _state.events:
        return JSONResponse({"error": "No log loaded"}, status_code=404)

    from backend.agent import _ts_parse
    level_f   = level.upper()
    keyword_f = keyword.lower()
    ts_from_e = _ts_parse(ts_from) if ts_from else 0.0
    ts_to_e   = _ts_parse(ts_to)   if ts_to   else 0.0

    filtered = []
    for e in _state.events:
        if level_f and e["level"] != level_f:
            continue
        if keyword_f and keyword_f not in e["msg"].lower():
            continue
        if ts_from_e and e["ts_epoch"] > 0 and e["ts_epoch"] < ts_from_e:
            continue
        if ts_to_e and e["ts_epoch"] > 0 and e["ts_epoch"] > ts_to_e:
            continue
        filtered.append(e)

    total = len(filtered)
    start = (page - 1) * limit
    page_events = filtered[start:start + limit]

    return {
        "total": total,
        "page": page,
        "pages": -(-total // limit),
        "events": [
            {"ts": e["ts"], "level": e["level"], "logger": e["logger"], "msg": e["msg"][:400]}
            for e in page_events
        ],
    }


@app.get("/api/search")
async def search_logs(
    q:     str = Query(...),
    top_k: int = Query(8, ge=1, le=20),
    chunk_type: str = Query(""),
):
    if not _state.rag_store:
        return JSONResponse({"error": "RAG index not built"}, status_code=404)
    results = _state.rag_store.query(q, top_k=top_k, type_filter=chunk_type or None)
    return {"results": results}


@app.get("/api/loggers")
async def get_loggers():
    if not _state.events:
        return JSONResponse({"error": "No log loaded"}, status_code=404)
    from collections import Counter
    counts = Counter(e["logger"] for e in _state.events)
    return {"loggers": [{"name": k, "count": v} for k, v in counts.most_common(50)]}


@app.get("/api/levels")
async def get_levels():
    if not _state.events:
        return JSONResponse({"error": "No log loaded"}, status_code=404)
    from collections import Counter
    counts = Counter(e["level"] for e in _state.events)
    return {"levels": dict(counts)}


@app.post("/api/reset")
async def reset():
    if _state.is_processing:
        return JSONResponse({"error": "Processing in progress"}, status_code=409)
    _state.reset()
    return {"status": "reset"}


@app.delete("/api/history")
async def clear_history():
    _state.conversation_history.clear()
    return {"status": "cleared"}


@app.get("/api/models")
async def list_models():
    return {"models": get_models_list()}


# ── Session persistence ───────────────────────────────────────────────────────

SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text())
            sessions.append({
                "id":       p.stem,
                "filename": meta.get("filename", ""),
                "saved_at": meta.get("saved_at", ""),
                "messages": len(meta.get("messages", [])),
                "summary_title": meta.get("summary_title", ""),
            })
        except Exception:
            pass
    return {"sessions": sessions}


@app.post("/api/sessions/save")
async def save_session(req: dict):
    import time, hashlib
    sid = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:10]
    path = SESSIONS_DIR / f"{sid}.json"
    payload = {
        "id":            sid,
        "filename":      _state.log_filename,
        "saved_at":      __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "summary_title": req.get("title", _state.log_filename),
        "messages":      req.get("messages", []),
        "level_counts":  _state.summary.get("level_counts", {}) if _state.summary else {},
        "total_events":  len(_state.events),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return {"id": sid, "saved_at": payload["saved_at"]}


@app.get("/api/sessions/{sid}")
async def load_session(sid: str):
    path = SESSIONS_DIR / f"{sid}.json"
    if not path.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return json.loads(path.read_text())


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    path = SESSIONS_DIR / f"{sid}.json"
    if path.exists():
        path.unlink()
    return {"status": "deleted"}


# ── Report generation ─────────────────────────────────────────────────────────

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


@app.get("/api/report/generate")
async def generate_report(model: str = Query(default="")):
    """SSE endpoint that streams report sections as they are generated."""
    if not _state.summary:
        return JSONResponse({"error": "No log loaded"}, status_code=404)

    model_id = model or DEFAULT_MODEL
    model, provider, _ = _resolve_model(model_id)

    try:
        client = make_async_client(provider)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Collect all sections so we can save the report
    report_sections: dict[str, str] = {}

    async def event_stream():
        import hashlib, time as _time
        report_id = hashlib.md5(f"{_time.time()}".encode()).hexdigest()[:10]
        nonlocal report_sections

        async for msg in generate_report_sections(_state, client, model):
            yield f"data: {json.dumps(msg)}\n\n"

            # Accumulate section content
            if msg["type"] == "section_content":
                sec = msg["section"]
                report_sections[sec] = report_sections.get(sec, "") + msg["content"]

            # On complete, persist report to disk
            if msg["type"] == "complete":
                report_data = {
                    "id": report_id,
                    "filename": _state.log_filename,
                    "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                    "model": model,
                    "sections": report_sections,
                    "summary": _state.summary,
                }
                report_path = REPORTS_DIR / f"{report_id}.json"
                report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
                yield f"data: {json.dumps({'type': 'saved', 'report_id': report_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/report/list")
async def list_reports():
    reports = []
    for p in sorted(REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text())
            reports.append({
                "id":           p.stem,
                "filename":     meta.get("filename", ""),
                "generated_at": meta.get("generated_at", ""),
                "model":        meta.get("model", ""),
            })
        except Exception:
            pass
    return {"reports": reports}


@app.get("/api/report/{report_id}")
async def get_report(report_id: str):
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        return JSONResponse({"error": "Report not found"}, status_code=404)
    return json.loads(path.read_text())


@app.get("/api/report/{report_id}/html")
async def download_report_html(report_id: str):
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        return JSONResponse({"error": "Report not found"}, status_code=404)

    report = json.loads(path.read_text())
    html = _render_report_html(report)
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f'attachment; filename="report-{report_id}.html"'},
    )


def _render_report_html(report: dict) -> str:
    """Render the saved report JSON as a self-contained HTML file."""
    import re as _re

    sections = report.get("sections", {})
    summary = report.get("summary", {})
    filename = report.get("filename", "log")
    generated_at = report.get("generated_at", "")
    char = summary.get("characterization", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)

    # Cover data
    cover_raw = sections.get("cover", "{}")
    try:
        cover = json.loads(cover_raw)
    except Exception:
        cover = {}

    # Stat overview data
    stat_raw = sections.get("statistical_overview", "{}")
    try:
        stat = json.loads(stat_raw)
    except Exception:
        stat = {}

    # Appendix data
    app_raw = sections.get("appendix", "{}")
    try:
        appendix = json.loads(app_raw)
    except Exception:
        appendix = {}

    level_labels = list(lc.keys())
    level_values = list(lc.values())
    level_colors = {
        "DEBUG": "#6B7280", "INFO": "#3B82F6", "WARNING": "#F59E0B",
        "ERROR": "#EF4444", "CRITICAL": "#7C3AED", "RAW": "#9CA3AF",
    }
    level_color_list = [level_colors.get(l, "#9CA3AF") for l in level_labels]

    # Events per hour chart data
    eph = stat.get("events_per_hour", {})
    erph = stat.get("errors_per_hour", {})
    hour_labels = sorted(set(list(eph.keys()) + list(erph.keys())))
    hour_events = [eph.get(h, 0) for h in hour_labels]
    hour_errors = [erph.get(h, 0) for h in hour_labels]

    # Top loggers chart
    top_loggers = stat.get("top_loggers", [])[:12]
    logger_labels = [l["logger"] for l in top_loggers]
    logger_counts = [l["count"] for l in top_loggers]
    logger_errors_data = [l.get("error_rate", 0) * l["count"] for l in top_loggers]

    def md_to_html(text: str) -> str:
        # very basic markdown → html for standalone report
        # Use marked.js in the HTML output instead
        return text

    llm_sections = [
        ("abstract", "Abstract"),
        ("executive_summary", "Executive Summary"),
        ("system_overview", "System Overview"),
        ("error_analysis", "Error Analysis"),
        ("error_bursts", "Error Burst Analysis"),
        ("performance_throughput", "Performance & Throughput"),
        ("component_health", "Component Health Report"),
        ("pattern_analysis", "Pattern Analysis"),
        ("entity_analysis", "Entity Analysis"),
        ("pain_points", "Pain Points & Where to Look"),
        ("recommendations", "Recommendations"),
    ]

    # Build section HTML for LLM sections
    llm_sections_html = ""
    for i, (sec_id, sec_title) in enumerate(llm_sections, start=3):
        content = sections.get(sec_id, "*Not generated.*")
        pain_class = ' class="pain-section"' if sec_id == "pain_points" else ""
        llm_sections_html += f"""
        <section id="sec-{sec_id}"{pain_class}>
          <h2>{i}. {sec_title}</h2>
          <div class="md-content" data-md="{sec_id}"></div>
          <script>
            document.querySelector('[data-md="{sec_id}"]').innerHTML = marked.parse({json.dumps(content)});
          </script>
        </section>
        """

    # Appendix tables
    error_samples_rows = ""
    for s in appendix.get("error_samples", [])[:50]:
        ts = s.get("ts", "")
        logger = s.get("logger", "")
        msg = s.get("msg", "")[:200]
        error_samples_rows += f"<tr><td>{ts}</td><td>{logger}</td><td>{msg}</td></tr>"

    burst_rows = ""
    for b in appendix.get("error_bursts", []):
        burst_rows += f"<tr><td>{b.get('hour','')}</td><td>{b.get('count','')}</td></tr>"

    pattern_rows = ""
    for p in appendix.get("top_patterns", [])[:30]:
        pattern_rows += f"<tr><td>{p.get('pattern','')[:120]}</td><td>{p.get('count','')}</td></tr>"

    logger_table_rows = ""
    for l in appendix.get("all_loggers", [])[:30]:
        rate_pct = round(l.get("error_rate", 0) * 100, 1)
        status = "🔴" if rate_pct > 20 else ("⚠️" if rate_pct > 5 else "✅")
        logger_table_rows += f"<tr><td>{l.get('logger','')}</td><td>{l.get('count',''):,}</td><td>{int(l.get('count',0)*l.get('error_rate',0)):,}</td><td>{rate_pct}%</td><td>{status}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Log Analysis Report — {filename}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --primary: #818cf8;
    --error: #f87171; --warning: #fbbf24; --success: #34d399;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
  .cover {{ min-height: 100vh; display: flex; align-items: center; justify-content: center; text-align: center; padding: 3rem; background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }}
  .cover h1 {{ font-size: 2.5rem; font-weight: 700; color: var(--primary); margin-bottom: 1rem; }}
  .cover .subtitle {{ font-size: 1.1rem; color: var(--muted); margin-bottom: 2rem; }}
  .cover .meta-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; max-width: 700px; margin: 0 auto 2rem; }}
  .cover .meta-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1rem; }}
  .cover .meta-val {{ font-size: 1.5rem; font-weight: 700; color: var(--primary); }}
  .cover .meta-lbl {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .cover .err-val {{ color: var(--error); }}
  .toc {{ padding: 3rem 4rem; max-width: 900px; margin: 0 auto; }}
  .toc h2 {{ font-size: 1.5rem; margin-bottom: 1.5rem; color: var(--primary); }}
  .toc ol {{ list-style: decimal; padding-left: 1.5rem; }}
  .toc li {{ padding: 0.35rem 0; }}
  .toc a {{ color: var(--text); text-decoration: none; }}
  .toc a:hover {{ color: var(--primary); }}
  section {{ max-width: 900px; margin: 0 auto; padding: 3rem 4rem; border-top: 1px solid var(--border); }}
  section h2 {{ font-size: 1.6rem; font-weight: 700; color: var(--primary); margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--border); }}
  section h3 {{ font-size: 1.15rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: var(--text); }}
  section h4 {{ font-size: 1rem; font-weight: 600; margin: 1rem 0 0.5rem; color: var(--muted); }}
  .md-content p {{ margin-bottom: 0.75rem; color: var(--text); }}
  .md-content ul, .md-content ol {{ margin: 0.5rem 0 0.75rem 1.5rem; }}
  .md-content li {{ margin-bottom: 0.25rem; }}
  .md-content strong {{ color: var(--primary); }}
  .md-content code {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.25rem; padding: 0.1rem 0.35rem; font-family: monospace; font-size: 0.85em; color: #a5f3fc; }}
  .md-content pre {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1rem; overflow-x: auto; margin: 0.75rem 0; }}
  .md-content pre code {{ background: none; border: none; padding: 0; }}
  .md-content blockquote {{ border-left: 3px solid var(--primary); padding-left: 1rem; color: var(--muted); margin: 0.75rem 0; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
  .chart-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1rem; }}
  .chart-card h3 {{ font-size: 0.875rem; color: var(--muted); margin-bottom: 0.75rem; }}
  .chart-wide {{ grid-column: 1/-1; }}
  .stat-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  .stat-table th {{ background: var(--surface); padding: 0.5rem 0.75rem; text-align: left; font-size: 0.75rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }}
  .stat-table td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; word-break: break-word; }}
  .stat-table tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .pain-section .md-content h3 {{ background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--error); border-radius: 0.5rem; padding: 0.75rem 1rem; margin: 1.5rem 0 0.75rem; }}
  .pain-section .md-content strong {{ color: var(--error); }}
  .footer {{ text-align: center; padding: 2rem; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); }}
  @media print {{
    body {{ background: white; color: black; }}
    .cover {{ background: white; }}
    section {{ border-top: 1px solid #ddd; }}
  }}
</style>
</head>
<body>

<!-- Cover -->
<div class="cover">
  <div>
    <p style="font-size:0.85rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Log Analysis Report</p>
    <h1>{filename}</h1>
    <p class="subtitle">{char.get('log_type', 'Unknown System')} &mdash; {char.get('system_description', '')[:120]}</p>
    <div class="meta-grid">
      <div class="meta-card">
        <div class="meta-val">{total:,}</div>
        <div class="meta-lbl">Total Events</div>
      </div>
      <div class="meta-card">
        <div class="meta-val err-val">{errors:,}</div>
        <div class="meta-lbl">Errors</div>
      </div>
      <div class="meta-card">
        <div class="meta-val err-val">{errors/max(total,1)*100:.1f}%</div>
        <div class="meta-lbl">Error Rate</div>
      </div>
      <div class="meta-card" style="grid-column:1/-1">
        <div class="meta-val" style="font-size:1rem">{cover.get('time_range_first','N/A')} &rarr; {cover.get('time_range_last','N/A')}</div>
        <div class="meta-lbl">Time Range ({cover.get('span_hours',0):.1f} hours)</div>
      </div>
    </div>
    <p style="color:var(--muted);font-size:0.8rem">Generated {generated_at} &bull; Format: {summary.get('format','Unknown')}</p>
  </div>
</div>

<!-- Table of Contents -->
<div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#sec-abstract">Abstract</a></li>
    <li><a href="#sec-executive_summary">Executive Summary</a></li>
    <li><a href="#sec-system_overview">System Overview</a></li>
    <li><a href="#sec-statistical_overview">Statistical Overview</a></li>
    <li><a href="#sec-error_analysis">Error Analysis</a></li>
    <li><a href="#sec-error_bursts">Error Burst Analysis</a></li>
    <li><a href="#sec-performance_throughput">Performance &amp; Throughput</a></li>
    <li><a href="#sec-component_health">Component Health Report</a></li>
    <li><a href="#sec-pattern_analysis">Pattern Analysis</a></li>
    <li><a href="#sec-entity_analysis">Entity Analysis</a></li>
    <li><a href="#sec-pain_points">Pain Points &amp; Where to Look</a></li>
    <li><a href="#sec-recommendations">Recommendations</a></li>
    <li><a href="#sec-appendix">Appendix</a></li>
  </ol>
</div>

{llm_sections_html}

<!-- Statistical Overview (charts) -->
<section id="sec-statistical_overview">
  <h2>4. Statistical Overview</h2>
  <div class="charts-grid">
    <div class="chart-card">
      <h3>Level Distribution</h3>
      <canvas id="chart-levels" height="200"></canvas>
    </div>
    <div class="chart-card">
      <h3>Top Loggers</h3>
      <canvas id="chart-loggers" height="200"></canvas>
    </div>
    <div class="chart-card chart-wide">
      <h3>Events Per Hour</h3>
      <canvas id="chart-hours" height="120"></canvas>
    </div>
    <div class="chart-card chart-wide">
      <h3>Errors Per Hour</h3>
      <canvas id="chart-errors" height="120"></canvas>
    </div>
  </div>
  <h3>Level Breakdown</h3>
  <table class="stat-table">
    <thead><tr><th>Level</th><th>Count</th><th>% of Total</th></tr></thead>
    <tbody>
      {"".join(f"<tr><td>{l}</td><td>{c:,}</td><td>{c/max(total,1)*100:.2f}%</td></tr>" for l,c in lc.items())}
    </tbody>
  </table>
</section>

<!-- Appendix -->
<section id="sec-appendix">
  <h2>13. Appendix</h2>
  <h3>A. Logger Summary</h3>
  <table class="stat-table">
    <thead><tr><th>Logger</th><th>Total Events</th><th>Errors</th><th>Error Rate</th><th>Status</th></tr></thead>
    <tbody>{logger_table_rows}</tbody>
  </table>
  <h3 style="margin-top:2rem">B. Error Samples</h3>
  <table class="stat-table">
    <thead><tr><th>Timestamp</th><th>Logger</th><th>Message</th></tr></thead>
    <tbody>{error_samples_rows}</tbody>
  </table>
  <h3 style="margin-top:2rem">C. Error Bursts</h3>
  <table class="stat-table">
    <thead><tr><th>Hour</th><th>Error Count</th></tr></thead>
    <tbody>{burst_rows if burst_rows else "<tr><td colspan='2'>No error bursts detected</td></tr>"}</tbody>
  </table>
  <h3 style="margin-top:2rem">D. Top Message Patterns</h3>
  <table class="stat-table">
    <thead><tr><th>Pattern</th><th>Count</th></tr></thead>
    <tbody>{pattern_rows}</tbody>
  </table>
</section>

<div class="footer">
  Log Analysis Report &bull; {filename} &bull; Generated {generated_at}
</div>

<script>
// Chart.js charts
const levelCtx = document.getElementById('chart-levels').getContext('2d');
new Chart(levelCtx, {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(level_labels)},
    datasets: [{{ data: {json.dumps(level_values)}, backgroundColor: {json.dumps(level_color_list)}, borderWidth: 0 }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e2e8f0', font: {{ size: 11 }} }} }} }}, cutout: '60%' }}
}});

const logCtx = document.getElementById('chart-loggers').getContext('2d');
new Chart(logCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(logger_labels)},
    datasets: [
      {{ label: 'Total', data: {json.dumps(logger_counts)}, backgroundColor: '#818cf8' }},
      {{ label: 'Errors', data: {json.dumps([int(x) for x in logger_errors_data])}, backgroundColor: '#f87171' }}
    ]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45 }}, grid: {{ color: '#334155' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
    }}
  }}
}});

const hourCtx = document.getElementById('chart-hours').getContext('2d');
new Chart(hourCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(hour_labels)},
    datasets: [{{ label: 'Events/Hour', data: {json.dumps(hour_events)}, backgroundColor: '#818cf8' }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxRotation: 60 }}, grid: {{ color: '#334155' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
    }}
  }}
}});

const errCtx = document.getElementById('chart-errors').getContext('2d');
new Chart(errCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(hour_labels)},
    datasets: [{{ label: 'Errors/Hour', data: {json.dumps(hour_errors)}, borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.15)', fill: true, tension: 0.3 }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxRotation: 60 }}, grid: {{ color: '#334155' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


# ── WebSocket chat ────────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    # Cache clients per provider to avoid re-creating on every message
    _clients: dict = {}

    def _get_client(provider: "Provider"):
        if provider not in _clients:
            _clients[provider] = make_async_client(provider)
        return _clients[provider]

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            question  = payload.get("question", "").strip()
            mode      = payload.get("mode", "agent")   # "agent" | "rag"
            model_id  = payload.get("model") or None

            if not question:
                await websocket.send_json({"type": "error", "content": "Empty question"})
                continue

            if not _state.summary:
                await websocket.send_json({"type": "error", "content": "No log loaded. Upload a log file first."})
                continue

            model, provider, supports_tools = _resolve_model(model_id)
            try:
                client = _get_client(provider)
            except RuntimeError as e:
                await websocket.send_json({"type": "error", "content": str(e)})
                continue

            if mode == "rag":
                if not _state.rag_store:
                    await websocket.send_json({"type": "error", "content": "RAG index not ready yet."})
                    continue
                async for chunk in run_rag_stream(question, _state, client, model=model):
                    await websocket.send_json(chunk)
            else:
                assistant_tokens = []
                async for chunk in run_agent_stream(
                    question, _state.conversation_history, _state, client,
                    model=model, supports_tools=supports_tools,
                ):
                    await websocket.send_json(chunk)
                    if chunk["type"] == "token":
                        assistant_tokens.append(chunk["content"])
                    if chunk["type"] == "done":
                        _state.conversation_history.append({"role": "user", "content": question})
                        if assistant_tokens:
                            _state.conversation_history.append({"role": "assistant", "content": "".join(assistant_tokens)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


# ── Static files (must be last) ───────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
