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
        self.processing_step: str           = ""   # human-readable progress
        self.processing_error: str          = ""
        self.conversation_history: list[dict] = []

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
        events, profile = parse_log_file(path)
        state.events = events

        state.processing_step = "Computing statistics and characterising log with AI…"
        summary = build_summary(events, profile, client)
        state.summary = summary

        state.processing_step = "Building RAG index… (chunking log events)"
        print(f"[RAG] Step 1: Building chunks from {len(events)} events…")
        char = summary.get("characterization", {})
        chunks = build_chunks(events, char)
        type_counts = {}
        for c in chunks:
            t = c.get("chunk_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"[RAG] Built {len(chunks)} chunks: {type_counts}")

        state.processing_step = "Building RAG index… (loading embedding model)"
        print("[RAG] Step 2: Loading SentenceTransformer model all-MiniLM-L6-v2…")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[RAG] Embedding model loaded.")

        state.processing_step = "Building RAG index… (encoding chunks & building FAISS index)"
        print(f"[RAG] Step 3: Encoding {len(chunks)} chunks and building FAISS index…")
        index = build_index(chunks, embed_model)
        print(f"[RAG] FAISS index built. Total vectors: {index.ntotal}")

        state.rag_store = RAGStore(index, chunks, embed_model)
        print("[RAG] RAGStore ready.")

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


@app.get("/api/models")
async def list_models():
    return {"models": get_models_list()}


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
