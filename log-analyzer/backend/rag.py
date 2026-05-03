"""
RAG layer — FAISS vector store over 4 types of log chunks.
Covers: time windows, error bursts, per-logger threads, init sequences.
This ensures ALL log information (dates, model names, config, IPs, etc.) is indexed.
"""
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import numpy as np

from backend.parser import LogEvent

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_NPY = DATA_DIR / "rag.index.npy"
META_JSON  = DATA_DIR / "rag.meta.json"

CHUNK_MINUTES   = 2
MAX_CHUNK_CHARS = 1600
MAX_CHUNK_LINES = 50


# ── Chunk builders ────────────────────────────────────────────────────────────

def _fmt(e: LogEvent, max_msg: int = 320) -> str:
    return f"[{e['ts']}] {e['level']} {e['logger']}: {e['msg'][:max_msg]}"


def _time_key(epoch: float, minutes: int) -> str:
    if epoch == 0.0:
        return "1970-01-01T00:00"
    dt = datetime.fromtimestamp(epoch)
    m = (dt.minute // minutes) * minutes
    return dt.strftime(f"%Y-%m-%dT%H:{m:02d}")


def _build_time_window_chunks(
    events: list[LogEvent],
    noisy_loggers: set[str],
    chunk_minutes: int = CHUNK_MINUTES,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    chunks = []
    bucket_key = None
    bucket_lines: list[str] = []
    bucket_ts_start = ""
    bucket_ts_end   = ""

    def flush():
        if bucket_lines:
            chunks.append({
                "chunk_type": "time_window",
                "ts_start": bucket_ts_start,
                "ts_end": bucket_ts_end,
                "text": f"TIME WINDOW [{bucket_ts_start} → {bucket_ts_end}]:\n" + "\n".join(bucket_lines),
                "metadata": {"line_count": len(bucket_lines)},
            })

    for e in events:
        if e["logger"] in noisy_loggers:
            continue
        key = _time_key(e["ts_epoch"], chunk_minutes)
        line = _fmt(e)

        if bucket_key is None:
            bucket_key = key
            bucket_ts_start = e["ts"]

        roll = key != bucket_key or len(bucket_lines) >= MAX_CHUNK_LINES or sum(len(l) for l in bucket_lines) > max_chars
        if roll:
            flush()
            bucket_lines = [line]
            bucket_key = key
            bucket_ts_start = e["ts"]
        else:
            bucket_lines.append(line)
        bucket_ts_end = e["ts"]

    flush()
    return chunks


def _build_error_burst_chunks(events: list[LogEvent], window: int = 15) -> list[dict]:
    errors = [e for e in events if e["level"] in ("ERROR", "CRITICAL")]
    if not errors:
        return []
    chunks = []
    i = 0
    while i < len(errors):
        batch = errors[i:i + window]
        text_lines = [_fmt(e, 400) for e in batch]
        chunks.append({
            "chunk_type": "error_burst",
            "ts_start": batch[0]["ts"],
            "ts_end": batch[-1]["ts"],
            "text": f"ERROR BURST [{batch[0]['ts']} → {batch[-1]['ts']}] ({len(batch)} errors):\n" + "\n".join(text_lines),
            "metadata": {
                "count": len(batch),
                "loggers": list({e["logger"] for e in batch}),
            },
        })
        i += window
    return chunks


def _build_logger_thread_chunks(
    events: list[LogEvent],
    noisy_loggers: set[str],
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    by_logger: dict[str, list[LogEvent]] = defaultdict(list)
    for e in events:
        if e["logger"] not in noisy_loggers and e["level"] != "RAW":
            by_logger[e["logger"]].append(e)

    chunks = []
    for logger, evts in sorted(by_logger.items(), key=lambda x: -len(x[1])):
        # Split into pages
        page_lines: list[str] = []
        page_chars = 0
        page_ts_start = evts[0]["ts"] if evts else ""
        page_ts_end   = ""
        page_num = 1

        def flush_page(lines, ts_s, ts_e, pg):
            if lines:
                chunks.append({
                    "chunk_type": "logger_thread",
                    "ts_start": ts_s,
                    "ts_end": ts_e,
                    "text": f"ALL MESSAGES FROM [{logger}] (page {pg}):\n" + "\n".join(lines),
                    "metadata": {"logger": logger, "count": len(lines), "page": pg},
                })

        for e in evts:
            line = _fmt(e)
            if page_chars + len(line) > max_chars and page_lines:
                flush_page(page_lines, page_ts_start, page_ts_end, page_num)
                page_lines = []
                page_chars = 0
                page_ts_start = e["ts"]
                page_num += 1
            page_lines.append(line)
            page_chars += len(line)
            page_ts_end = e["ts"]

        flush_page(page_lines, page_ts_start, page_ts_end, page_num)

    return chunks


def _build_init_sequence_chunks(events: list[LogEvent], window_seconds: int = 300) -> list[dict]:
    """Collect startup/config/initialization events."""
    _INIT_KW = re.compile(
        r'\b(init|initializ|start|load|version|model|config|connect|ready|boot|launch|setup)\b',
        re.I
    )
    init_events = [e for e in events if _INIT_KW.search(e["msg"]) or e["logger"] == "__main__"]
    if not init_events:
        return []

    # Find "system start" markers (lines with system-level init phrases)
    _START_KW = re.compile(r'(starting|initialized system|system is running|application started)', re.I)
    start_epochs = [e["ts_epoch"] for e in events if _START_KW.search(e["msg"])]
    if not start_epochs:
        start_epochs = [events[0]["ts_epoch"]] if events else []

    chunks = []
    for start_epoch in start_epochs[:10]:  # cap at 10 restarts
        window_evts = [
            e for e in init_events
            if start_epoch <= e["ts_epoch"] <= start_epoch + window_seconds
        ]
        if len(window_evts) < 3:
            continue
        ts_s = datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S")
        ts_e = window_evts[-1]["ts"]
        chunks.append({
            "chunk_type": "init_sequence",
            "ts_start": ts_s,
            "ts_end": ts_e,
            "text": f"INITIALIZATION SEQUENCE starting at {ts_s}:\n" + "\n".join(_fmt(e, 400) for e in window_evts[:60]),
            "metadata": {"event_count": len(window_evts)},
        })
    return chunks


# ── Index construction ────────────────────────────────────────────────────────

def build_chunks(events: list[LogEvent], characterization: dict) -> list[dict]:
    """Build all 4 chunk types and assign sequential IDs."""
    noisy = set(characterization.get("noisy_loggers", ["httpx", "aiohttp.access"]))

    all_chunks = (
        _build_time_window_chunks(events, noisy)
        + _build_error_burst_chunks(events)
        + _build_logger_thread_chunks(events, noisy)
        + _build_init_sequence_chunks(events)
    )

    for i, c in enumerate(all_chunks):
        c["id"] = i

    return all_chunks


def build_index(chunks: list[dict], embed_model, progress_cb=None) -> Any:
    """Embed chunks and build FAISS IndexFlatIP. Saves to DATA_DIR."""
    import faiss
    texts = [c["text"] for c in chunks]
    batch_size = 32
    all_embs = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = embed_model.encode(batch, show_progress_bar=False).astype(np.float32)
        all_embs.append(emb)
        done = min(i + batch_size, len(texts))
        if progress_cb:
            progress_cb(done, len(texts))
    import numpy as _np
    embs = _np.concatenate(all_embs, axis=0)
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print(f"[RAG] FAISS index built with {index.ntotal} vectors (dim={embs.shape[1]}).")

    DATA_DIR.mkdir(exist_ok=True)
    print(f"[RAG] Saving index to {INDEX_NPY} and metadata to {META_JSON}…")
    np.save(INDEX_NPY, embs)
    with open(META_JSON, "w") as f:
        json.dump(chunks, f)
    print("[RAG] Index saved.")

    return index


def load_index() -> tuple[Any, list[dict], Any]:
    """Load saved FAISS index. Returns (index, chunks, embed_model)."""
    import faiss
    from sentence_transformers import SentenceTransformer

    embs = np.load(INDEX_NPY).astype(np.float32)
    with open(META_JSON) as f:
        chunks = json.load(f)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, chunks, model


# ── RAGStore ──────────────────────────────────────────────────────────────────

class RAGStore:
    """Stateful container: holds FAISS index + chunks + embed model."""

    def __init__(self, index, chunks: list[dict], embed_model):
        self._index  = index
        self._chunks = chunks
        self._model  = embed_model

    def query(
        self,
        q: str,
        top_k: int = 8,
        type_filter: Optional[str] = None,
    ) -> list[dict]:
        import faiss
        q_emb = self._model.encode([q]).astype(np.float32)
        faiss.normalize_L2(q_emb)

        # Search more than needed if filtering
        k = top_k * 3 if type_filter else top_k
        k = min(k, len(self._chunks))
        scores, idxs = self._index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            if type_filter and c.get("chunk_type") != type_filter:
                continue
            results.append({**c, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def get_chunk(self, chunk_id: int) -> Optional[dict]:
        for c in self._chunks:
            if c.get("id") == chunk_id:
                return c
        return None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunk_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for c in self._chunks:
            counts[c.get("chunk_type", "unknown")] += 1
        return dict(counts)
