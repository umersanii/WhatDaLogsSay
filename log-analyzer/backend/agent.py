"""
AI agent with tool use + streaming for both Agent and RAG modes.
Tools query AppState (pre-built summary + events + RAGStore).
The agent/RAG system prompts come from the AI's own log characterization.
"""
import json
from datetime import datetime
from typing import AsyncIterator, TYPE_CHECKING

from groq import AsyncGroq

if TYPE_CHECKING:
    from backend.api import AppState

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_logs",
            "description": (
                "Semantic vector search over the full log using FAISS. "
                "Returns the most relevant log excerpts for a natural-language query. "
                "Use this to find specific events, error messages, configuration values, "
                "model names, IP addresses, or any content that may not be in the summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "default": 6, "description": "Number of results (1–20)"},
                    "chunk_type": {
                        "type": "string",
                        "enum": ["time_window", "error_burst", "logger_thread", "init_sequence"],
                        "description": "Optional: restrict to one chunk type",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": (
                "Return structured statistics from the pre-computed log summary. "
                "Covers: level counts, top loggers, date range, events-per-hour, "
                "errors-per-hour, error samples, top patterns, entities, and "
                "the system characterization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Which summary fields to return. Empty = return all. "
                            "Options: level_counts, top_loggers, date_range, events_per_hour, "
                            "errors_per_hour, top_patterns, error_samples, error_bursts, "
                            "entities, characterization, format"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline",
            "description": (
                "Return paginated log events, optionally filtered by level, keyword, or time range. "
                "Use to trace event sequences around a specific time or incident."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page":    {"type": "integer", "default": 1, "description": "Page number (1-based)"},
                    "limit":   {"type": "integer", "default": 50, "description": "Events per page (max 200)"},
                    "level":   {"type": "string", "description": "Filter by level (INFO/WARNING/ERROR/CRITICAL)"},
                    "keyword": {"type": "string", "description": "Case-insensitive substring in msg"},
                    "ts_from": {"type": "string", "description": "ISO datetime lower bound"},
                    "ts_to":   {"type": "string", "description": "ISO datetime upper bound"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_samples",
            "description": (
                "Return sample log lines from a specific logger or matching a pattern. "
                "Useful for understanding what a particular component logs, finding config values, "
                "or checking a specific subsystem's behaviour."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "logger":  {"type": "string", "description": "Logger name (partial match OK)"},
                    "level":   {"type": "string", "description": "Filter by log level"},
                    "keyword": {"type": "string", "description": "Substring filter on message"},
                    "limit":   {"type": "integer", "default": 30, "description": "Max results (max 150)"},
                },
                "required": [],
            },
        },
    },
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _ts_parse(s: str) -> float:
    """Parse ISO-ish string to epoch, 0 on failure."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def _tool_search_logs(inp: dict, state: "AppState") -> str:
    if not state.rag_store:
        return json.dumps({"error": "RAG index not built yet. Use get_stats or get_timeline instead."})
    results = state.rag_store.query(
        inp["query"],
        top_k=min(int(inp.get("top_k") or 6), 20),
        type_filter=inp.get("chunk_type"),
    )
    if not results:
        return json.dumps({"results": [], "message": "No relevant chunks found."})
    formatted = [
        {
            "score": round(r["score"], 3),
            "type": r["chunk_type"],
            "ts_start": r["ts_start"],
            "ts_end":   r["ts_end"],
            "text":     r["text"][:800],
        }
        for r in results
    ]
    return json.dumps({"results": formatted}, default=str)


def _tool_get_stats(inp: dict, state: "AppState") -> str:
    if not state.summary:
        return json.dumps({"error": "No log loaded."})
    fields = inp.get("fields") or []
    if not fields:
        return json.dumps(state.summary, default=str)
    return json.dumps({k: state.summary.get(k) for k in fields if k in state.summary}, default=str)


def _tool_get_timeline(inp: dict, state: "AppState") -> str:
    if not state.events:
        return json.dumps({"error": "No log loaded."})

    page    = max(1, int(inp.get("page") or 1))
    limit   = min(200, max(1, int(inp.get("limit") or 50)))
    level   = (inp.get("level") or "").upper()
    keyword = (inp.get("keyword") or "").lower()
    ts_from = _ts_parse(inp.get("ts_from") or "")
    ts_to   = _ts_parse(inp.get("ts_to")   or "")

    filtered = []
    for e in state.events:
        if level and e["level"] != level:
            continue
        if keyword and keyword not in e["msg"].lower():
            continue
        if ts_from and e["ts_epoch"] > 0 and e["ts_epoch"] < ts_from:
            continue
        if ts_to and e["ts_epoch"] > 0 and e["ts_epoch"] > ts_to:
            continue
        filtered.append(e)

    total = len(filtered)
    start = (page - 1) * limit
    page_events = filtered[start:start + limit]

    return json.dumps({
        "total": total,
        "page": page,
        "pages": -(-total // limit),
        "events": [
            {"ts": e["ts"], "level": e["level"], "logger": e["logger"], "msg": e["msg"][:300]}
            for e in page_events
        ],
    }, default=str)


def _tool_get_samples(inp: dict, state: "AppState") -> str:
    if not state.events:
        return json.dumps({"error": "No log loaded."})

    logger_q = (inp.get("logger") or "").lower()
    level_q  = (inp.get("level")  or "").upper()
    keyword  = (inp.get("keyword") or "").lower()
    limit    = min(150, max(1, int(inp.get("limit") or 30)))

    results = []
    for e in state.events:
        if logger_q and logger_q not in e["logger"].lower():
            continue
        if level_q and e["level"] != level_q:
            continue
        if keyword and keyword not in e["msg"].lower():
            continue
        results.append({"ts": e["ts"], "level": e["level"], "logger": e["logger"], "msg": e["msg"][:400]})
        if len(results) >= limit:
            break

    return json.dumps({"count": len(results), "samples": results}, default=str)


def execute_tool(name: str, inp: dict, state: "AppState") -> str:
    try:
        if name == "search_logs":    return _tool_search_logs(inp, state)
        if name == "get_stats":      return _tool_get_stats(inp, state)
        if name == "get_timeline":   return _tool_get_timeline(inp, state)
        if name == "get_samples":    return _tool_get_samples(inp, state)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agent streaming loop ──────────────────────────────────────────────────────

async def run_agent_stream(
    question: str,
    history: list[dict],
    state: "AppState",
    client: AsyncGroq,
) -> AsyncIterator[dict]:
    """
    Yields dicts: {type: "token"|"tool_call"|"tool_result"|"done"|"error"}
    Agent loop: call Groq → stream text → execute tools → repeat.
    """
    char = state.summary.get("characterization", {}) if state.summary else {}
    system = char.get("agent_system_prompt") or (
        "You are a log file analyst with access to tools. "
        "Answer questions accurately using your tools. Format responses in markdown."
    )

    messages = history[-30:] + [{"role": "user", "content": question}]
    max_iters = 10

    for _ in range(max_iters):
        full_text = []
        tool_use_blocks = []

        async with client.chat.completions.stream(
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.choices and event.choices[0].delta.content:
                    yield {"type": "token", "content": event.choices[0].delta.content}
                    full_text.append(event.choices[0].delta.content)

            final = await stream.get_final_message()

        # Tool calls in Groq/OpenAI format come in the message content
        if final.choices[0].message.tool_calls:
            tool_use_blocks = final.choices[0].message.tool_calls

        messages.append({"role": "assistant", "content": final.choices[0].message.content or ""})
        if final.choices[0].message.tool_calls:
            messages[-1]["tool_calls"] = final.choices[0].message.tool_calls

        if not tool_use_blocks:
            break

        # Execute tools
        tool_results = []
        for tb in tool_use_blocks:
            yield {"type": "tool_call", "name": tb.function.name, "input": tb.function.arguments}
            try:
                args = json.loads(tb.function.arguments) if isinstance(tb.function.arguments, str) else tb.function.arguments
            except:
                args = {}
            result_str = execute_tool(tb.function.name, args, state)
            yield {"type": "tool_result", "name": tb.function.name, "content": result_str[:300]}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_str,
            })
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "done"}


async def run_rag_stream(
    question: str,
    state: "AppState",
    client: AsyncGroq,
    top_k: int = 8,
) -> AsyncIterator[dict]:
    """
    Single-turn RAG: retrieve → stream response.
    Yields same dict protocol as run_agent_stream.
    """
    char = state.summary.get("characterization", {}) if state.summary else {}
    system = char.get("rag_system_prompt") or (
        "You are a log analyst. Answer questions using ONLY the provided log excerpts. "
        "Be specific with timestamps and values."
    )

    results = state.rag_store.query(question, top_k=top_k)
    context = "\n\n---\n\n".join(
        f"[{r['chunk_type'].upper()} | {r['ts_start']} → {r['ts_end']} | score:{r['score']:.2f}]\n{r['text']}"
        for r in results
    )

    yield {"type": "status", "content": f"Retrieved {len(results)} relevant log chunks…"}

    async with client.chat.completions.stream(
        model="llama-3.3-70b-versatile",
        max_tokens=3000,
        system=system,
        messages=[{
            "role": "user",
            "content": f"<log_excerpts>\n{context}\n</log_excerpts>\n\nQuestion: {question}",
        }],
    ) as stream:
        async for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield {"type": "token", "content": event.choices[0].delta.content}

    yield {"type": "done"}
