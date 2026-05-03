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

from groq import Groq, AsyncGroq
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.analyzer import build_summary
from backend.agent import run_agent_stream, run_rag_stream
from backend.parser import parse_log_file
from backend.rag import RAGStore, build_chunks, build_index, load_index

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
        self.processing_step: str           = ""   # human-readable progress
        self.processing_error: str          = ""
        self.conversation_history: list[dict] = []

    def reset(self):
        self.__init__()


_state = AppState()

# ── Groq client helpers ──────────────────────────────────────────────────────

def _get_api_key() -> str:
    api_key = os.environ.get("GROQ") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ or GROQ_API_KEY environment variable must be set.")
    return api_key


def _sync_client() -> Groq:
    return Groq(api_key=_get_api_key())


def _async_client() -> AsyncGroq:
    return AsyncGroq(api_key=_get_api_key())

# ── Background processing ─────────────────────────────────────────────────────

def _process_log(path: Path):
    """Parse → analyse → build RAG. Runs in a daemon thread."""
    state = _state
    client = _sync_client()
    try:
        state.processing_step = "Parsing log file…"
        events, profile = parse_log_file(path)
        state.events = events

        state.processing_step = "Computing statistics and characterising log with AI…"
        summary = build_summary(events, profile, client)
        state.summary = summary

        state.processing_step = "Building RAG index…"
        char = summary.get("characterization", {})
        chunks = build_chunks(events, char)
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        index = build_index(chunks, embed_model)
        state.rag_store = RAGStore(index, chunks, embed_model)

        state.processing_step = "Ready"
        state.is_processing = False

    except Exception:
        state.processing_error = traceback.format_exc()
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


# ── WebSocket chat ────────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    client = _async_client()

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            question = payload.get("question", "").strip()
            mode     = payload.get("mode", "agent")  # "agent" | "rag"

            if not question:
                await websocket.send_json({"type": "error", "content": "Empty question"})
                continue

            if not _state.summary:
                await websocket.send_json({"type": "error", "content": "No log loaded. Upload a log file first."})
                continue

            if mode == "rag":
                if not _state.rag_store:
                    await websocket.send_json({"type": "error", "content": "RAG index not ready yet."})
                    continue
                async for chunk in run_rag_stream(question, _state, client):
                    await websocket.send_json(chunk)
            else:
                # Agent mode — persist conversation
                async for chunk in run_agent_stream(question, _state.conversation_history, _state, client):
                    await websocket.send_json(chunk)
                    if chunk["type"] == "done":
                        # Append to history for context continuity
                        _state.conversation_history.append({"role": "user",    "content": question})
                        # (assistant content appended inside run_agent_stream via messages list,
                        #  but we track user turns here for the history display)

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
