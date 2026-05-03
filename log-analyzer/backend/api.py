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

from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.analyzer import build_summary
from backend.agent import run_agent_stream, run_rag_stream
from backend.parser import parse_log_file
from backend.rag import RAGStore, build_chunks, build_index, load_index
from backend.providers import (
    make_async_client, make_sync_client, get_models_list,
    DEFAULT_PROVIDER, DEFAULT_MODEL, MODELS, Provider,
)

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
                async for chunk in run_agent_stream(
                    question, _state.conversation_history, _state, client,
                    model=model, supports_tools=supports_tools,
                ):
                    await websocket.send_json(chunk)
                    if chunk["type"] == "done":
                        _state.conversation_history.append({"role": "user", "content": question})

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
