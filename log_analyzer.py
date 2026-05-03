#!/usr/bin/env python3
"""
CV System Log Analyzer
Two modes:
  1. Agent mode  -- structured pre-analysis + Claude conversational Q&A (fast, no embedding)
  2. RAG mode    -- FAISS vector store over chunked log + Claude Q&A (deep search)
"""

import re
import sys
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np

LOG_FILE = Path(__file__).parent / "headless_system.log"

# ──────────────────────────────────────────────────────────────────────────────
# PART 1 – LOG PARSING  (shared by both modes)
# ──────────────────────────────────────────────────────────────────────────────

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r" - (?P<logger>\S+)"
    r" - (?P<level>INFO|WARNING|ERROR|CRITICAL|DEBUG)"
    r" - (?P<msg>.+)$"
)


def parse_log(path: Path):
    events = []
    with open(path, "rb") as f:
        for raw in f:
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            events.append(
                {
                    "ts": m.group("ts"),
                    "logger": m.group("logger"),
                    "level": m.group("level"),
                    "msg": m.group("msg"),
                }
            )
    return events


def build_summary(events: list[dict]) -> dict:
    """Build a rich structured summary from parsed events."""
    summary = {
        "total_lines": len(events),
        "date_range": {"first": events[0]["ts"][:10] if events else "?", "last": events[-1]["ts"][:10] if events else "?"},
        "days": set(),
        "cameras": set(),
        "detections": defaultdict(int),          # type -> count
        "detections_per_camera": defaultdict(lambda: defaultdict(int)),
        "detections_per_hour": defaultdict(int),  # YYYY-MM-DD HH -> count
        "errors": defaultdict(int),              # logger -> count
        "warnings": [],
        "presence": [],        # {person, camera, duration_s}
        "person_counts": defaultdict(int),
        "reconnects": defaultdict(int),  # camera_id -> count
        "face_reinits": 0,
        "ws_disconnects": 0,
        "cpu_fallbacks": 0,
        "restricted_violations": 0,
        "yolo_errors": 0,
        "system_restarts": [],
        "error_bursts": [],
    }

    # Patterns
    det_pat = re.compile(r"Camera (\d+): (SMOKING|FIREARM) DETECTED")
    ppe_pat = re.compile(r"camera.*PPE|PPE.*camera", re.I)
    presence_enter = re.compile(r"Person '(.+?)' entered view on camera (\d+)")
    presence_exit = re.compile(r"Logged presence for '(.+?)': ([\d.]+)s on camera (\d+)")
    reconnect_pat = re.compile(r"Camera (\d+) reconnect")
    hour_pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2})")

    prev_error_count = 0
    hour_error_map = defaultdict(int)

    for e in events:
        ts, level, msg, logger = e["ts"], e["level"], e["msg"], e["logger"]
        day = ts[:10]
        hour = ts[:13]
        summary["days"].add(day)

        # Detections
        m = det_pat.search(msg)
        if m:
            cam, dtype = m.group(1), m.group(2)
            summary["detections"][dtype] += 1
            summary["detections_per_camera"][cam][dtype] += 1
            summary["detections_per_hour"][hour] += 1
            summary["cameras"].add(f"Camera {cam}")

        # PPE (fires as workflow trigger)
        if "ppe_detection" in msg or ("PPE" in msg and "triggered" in msg):
            summary["detections"]["PPE"] += 1

        # Presence
        m = presence_exit.search(msg)
        if m:
            summary["presence"].append(
                {"person": m.group(1), "duration_s": float(m.group(2)), "camera": m.group(3), "ts": ts}
            )
            summary["person_counts"][m.group(1)] += 1

        # Camera from init lines
        if "CameraThread initialized for camera" in msg:
            cam_m = re.search(r"camera (\d+)", msg)
            if cam_m:
                summary["cameras"].add(f"Camera {cam_m.group(1)}")

        # Errors
        if level == "ERROR":
            summary["errors"][logger] += 1
            hour_error_map[hour] += 1
            if "Error in update" in msg:
                summary["yolo_errors"] += 1

        if level == "WARNING":
            if "HF_TOKEN" not in msg:  # skip noisy HF warnings
                summary["warnings"].append({"ts": ts, "msg": msg[:120]})

        # Reconnects
        m = reconnect_pat.search(msg)
        if m and "Attempting" not in msg:
            summary["reconnects"][f"Camera {m.group(1)}"] += 1

        # Misc
        if "Reinitializing face" in msg:
            summary["face_reinits"] += 1
        if "WebSocket disconnect" in msg:
            summary["ws_disconnects"] += 1
        if "CPU fallback" in msg.lower():
            summary["cpu_fallbacks"] += 1
        if "restricted area violation" in msg.lower():
            summary["restricted_violations"] += 1
        if "Initializing Headless" in msg:
            summary["system_restarts"].append(ts)

    # Find error bursts (hours with >20 errors)
    for h, cnt in sorted(hour_error_map.items()):
        if cnt > 20:
            summary["error_bursts"].append({"hour": h, "errors": cnt})

    # Convert sets to sorted lists for JSON serialisation
    summary["days"] = sorted(summary["days"])
    summary["cameras"] = sorted(summary["cameras"])
    summary["detections"] = dict(summary["detections"])
    summary["detections_per_camera"] = {
        cam: dict(dtypes) for cam, dtypes in summary["detections_per_camera"].items()
    }
    # Top 20 busiest hours for detections
    top_hours = sorted(summary["detections_per_hour"].items(), key=lambda x: x[1], reverse=True)[:20]
    summary["detections_per_hour"] = dict(top_hours)
    summary["errors"] = dict(summary["errors"])
    summary["person_counts"] = dict(summary["person_counts"])
    summary["reconnects"] = dict(summary["reconnects"])

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# PART 2 – AGENT MODE  (structured summary → Claude Q&A)
# ──────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM = """You are a computer vision system log analyst. You have been given a structured
summary of a real-time surveillance system log (headless_system.log). The system runs:
- YOLO person detection & tracking
- FaceNet face recognition
- Firearm detection (TensorRT FP16)
- Smoking detection
- PPE detection
- Restricted area monitoring
- 2-8 cameras on NVIDIA Jetson Orin (CUDA 12.6)

Use the provided structured summary to answer questions accurately. Be specific with numbers,
camera IDs, timestamps, and persons. When identifying issues, suggest actionable fixes.
Format responses in clean markdown."""


def agent_mode(summary: dict, client: anthropic.Anthropic):
    summary_json = json.dumps(summary, indent=2, default=str)

    print("\n" + "="*60)
    print("AGENT MODE — Claude Q&A over structured log summary")
    print("="*60)
    print(f"Log span: {summary['date_range']['first']} → {summary['date_range']['last']}")
    print(f"Days: {', '.join(summary['days'])}")
    print(f"Total log entries parsed: {summary['total_lines']:,}")
    print("\nType your question, or 'quit' to exit.\n")

    messages = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        messages.append({"role": "user", "content": question})

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=AGENT_SYSTEM + f"\n\n<log_summary>\n{summary_json}\n</log_summary>",
            messages=messages,
        ) as stream:
            print("\nClaude: ", end="", flush=True)
            full_text = ""
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    print(event.delta.text, end="", flush=True)
                    full_text += event.delta.text
            print("\n")

        messages.append({"role": "assistant", "content": full_text})


# ──────────────────────────────────────────────────────────────────────────────
# PART 3 – RAG MODE  (FAISS + sentence-transformers + Claude)
# ──────────────────────────────────────────────────────────────────────────────

CHUNK_MINUTES = 2     # group log lines into N-minute windows
MAX_CHUNK_CHARS = 1500
INDEX_FILE = Path(__file__).parent / "log_rag.index.npy"
META_FILE  = Path(__file__).parent / "log_rag.meta.json"
TOP_K = 8


def chunk_events(events: list[dict]) -> list[dict]:
    """Group events into time-window chunks, skipping noisy HTTP lines."""
    chunks = []
    bucket: list[str] = []
    bucket_start = None
    SKIP = {"aiohttp.access", "httpx"}

    def flush(start, lines):
        if lines:
            chunks.append({"ts": start, "text": "\n".join(lines)})

    for e in events:
        if e["logger"] in SKIP:
            continue
        # skip pure HF_TOKEN warnings
        if "HF_TOKEN" in e["msg"]:
            continue

        ts_str = e["ts"][:16]  # YYYY-MM-DD HH:MM
        line = f"[{e['ts']}] {e['level']} {e['logger']}: {e['msg'][:300]}"

        if bucket_start is None:
            bucket_start = ts_str

        # New chunk if minute window exceeded or chunk too large
        cur_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
        start_dt = datetime.strptime(bucket_start, "%Y-%m-%d %H:%M")
        delta_min = (cur_dt - start_dt).total_seconds() / 60

        if delta_min >= CHUNK_MINUTES or sum(len(l) for l in bucket) > MAX_CHUNK_CHARS:
            flush(bucket_start, bucket)
            bucket = [line]
            bucket_start = ts_str
        else:
            bucket.append(line)

    flush(bucket_start, bucket)
    return chunks


def build_rag_index(chunks: list[dict], embed_model):
    import faiss

    print(f"  Embedding {len(chunks):,} chunks …", end="", flush=True)
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine after L2-norm
    index.add(embeddings)
    print(f" done (dim={dim})")

    # Save
    np.save(INDEX_FILE, embeddings)
    with open(META_FILE, "w") as f:
        json.dump([{"ts": c["ts"], "text": c["text"]} for c in chunks], f)

    return index, chunks


def load_rag_index():
    import faiss

    embeddings = np.load(INDEX_FILE).astype(np.float32)
    with open(META_FILE) as f:
        chunks = json.load(f)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, chunks


RAG_SYSTEM = """You are a computer vision security system analyst. You have been given
relevant excerpts from a real surveillance system log (headless_system.log).
The system uses YOLO, FaceNet, TensorRT-based firearm/smoking/PPE detectors on NVIDIA Jetson Orin.

Answer questions using ONLY the provided log excerpts. Be specific with timestamps, camera IDs,
person names, and error counts. If the excerpts don't contain enough information to answer,
say so clearly. Format responses in clean markdown."""


def rag_mode(events: list[dict], client: anthropic.Anthropic):
    from sentence_transformers import SentenceTransformer
    import faiss

    print("\n" + "="*60)
    print("RAG MODE — FAISS vector search + Claude")
    print("="*60)

    # Load or build index
    if INDEX_FILE.exists() and META_FILE.exists():
        print("  Loading existing FAISS index …", end="", flush=True)
        index, chunks = load_rag_index()
        print(f" done ({len(chunks):,} chunks)")
    else:
        print("  Building FAISS index (first run, may take ~1 min) …")
        print("  Loading embedding model …", end="", flush=True)
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(" done")
        chunks = chunk_events(events)
        print(f"  Created {len(chunks):,} chunks from log")
        index, chunks = build_rag_index(chunks, embed_model)
        embed_model_ref = embed_model

    # Load embed model (may already be loaded above)
    if "embed_model_ref" not in dir():
        print("  Loading embedding model …", end="", flush=True)
        embed_model_ref = SentenceTransformer("all-MiniLM-L6-v2")
        print(" done")

    print("\nType your question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        # Retrieve
        q_emb = embed_model_ref.encode([question]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, TOP_K)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(chunks):
                retrieved.append(
                    f"[Relevance: {score:.2f} | Time: {chunks[idx]['ts']}]\n{chunks[idx]['text']}"
                )

        context = "\n\n---\n\n".join(retrieved)

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=RAG_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"<log_excerpts>\n{context}\n</log_excerpts>\n\nQuestion: {question}",
                }
            ],
        ) as stream:
            print("\nClaude: ", end="", flush=True)
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    print(event.delta.text, end="", flush=True)
            print("\n")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GROQ") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ environment variable first.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("Parsing log file …", end="", flush=True)
    events = parse_log(LOG_FILE)
    print(f" {len(events):,} lines parsed")

    print("Building structured summary …", end="", flush=True)
    summary = build_summary(events)
    print(" done")

    # Quick stats printout
    print(f"\n{'─'*50}")
    print(f"  Days covered    : {', '.join(summary['days'])}")
    print(f"  Cameras active  : {', '.join(summary['cameras'])}")
    print(f"  Total detections: {sum(summary['detections'].values()):,}")
    for dtype, cnt in sorted(summary['detections'].items(), key=lambda x: -x[1]):
        print(f"    {dtype:<12}: {cnt:,}")
    print(f"  Presence logs   : {len(summary['presence'])}")
    print(f"  YOLO errors     : {summary['yolo_errors']}")
    print(f"  WS disconnects  : {summary['ws_disconnects']}")
    print(f"  Face reinits    : {summary['face_reinits']}")
    print(f"  System starts   : {len(summary['system_restarts'])}")
    print(f"{'─'*50}\n")

    # Mode selection
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Select mode:")
        print("  1. Agent  – fast Q&A over structured summary (recommended)")
        print("  2. RAG    – semantic search over full log chunks")
        choice = input("Enter 1 or 2: ").strip()
        mode = "rag" if choice == "2" else "agent"

    if mode in ("rag", "2"):
        rag_mode(events, client)
    else:
        agent_mode(summary, client)


if __name__ == "__main__":
    main()
