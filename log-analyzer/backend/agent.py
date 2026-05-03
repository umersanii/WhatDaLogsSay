"""
AI agent with tool use + streaming for both Agent and RAG modes.
Tools query AppState (pre-built summary + events + RAGStore).
The agent/RAG system prompts come from the AI's own log characterization.
"""
import json
from datetime import datetime
from typing import AsyncIterator, TYPE_CHECKING, Any

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
    client: Any,
    model: str = "llama-3.3-70b-versatile",
    supports_tools: bool = True,
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

    # Inject pre-computed summary stats so the model always has real numbers
    if state.summary:
        level_counts  = state.summary.get("level_counts", {})
        top_loggers   = state.summary.get("top_loggers", [])[:8]
        error_bursts  = state.summary.get("error_bursts", [])[:5]
        top_patterns  = state.summary.get("top_patterns", [])[:5]
        date_range    = state.summary.get("date_range", {})
        total_events  = sum(level_counts.values()) if level_counts else 0
        loggers_json  = json.dumps([{"logger": l["logger"], "count": l["count"], "error_rate": round(l.get("error_rate", 0) * 100, 1)} for l in top_loggers])
        patterns_json = json.dumps([{"pattern": p["pattern"][:80], "count": p["count"]} for p in top_patterns])
        system += f"""

CURRENT LOG SUMMARY (use these exact numbers in your components — do not fabricate):
- Total events: {total_events}
- Level counts: {json.dumps(level_counts)}
- Top loggers (name, count, error_rate): {loggers_json}
- Error bursts: {json.dumps(error_bursts)}
- Top patterns: {patterns_json}
- Date range: {json.dumps(date_range)}
"""

    # Augment system prompt with rich UI instructions
    system += """

RENDERING RULES:

Your responses are rendered as interactive UI widgets. Use these components ONLY when they add clarity — do NOT inject them mechanically into every response.

When to use each component:
- :::metric::: — only for a specific stat the user asked about (not a dump of all stats)
- :::chart::: — only when comparing multiple values (e.g. error distribution across loggers)
- :::log-ref::: — only when quoting an actual log line from the data
- :::timeline::: — only for ordered event sequences
- :::quiz::: — only for root-cause diagnostic questions

COMPONENT SYNTAX (each block on its own line, valid JSON inside):

:::metric{"label":"Total Errors","value":"457","trend":"up","color":"error","note":"of 608 total events"}:::

:::chart{"type":"bar","title":"Log Levels","labels":["ERROR","INFO","WARNING"],"datasets":[{"label":"Count","data":[457,121,14],"color":"error"}]}:::

:::chart{"type":"bar","title":"Logger Activity","labels":["recognition","__unparsed__"],"datasets":[{"label":"Total","data":[592,16],"color":"primary"},{"label":"Error %","data":[77.2,0],"color":"error"}]}:::

:::log-ref{"ts":"2024-01-01 12:00:01","level":"ERROR","logger":"recognition","msg":"Batched face recognition failed"}:::

:::timeline{"title":"Event Sequence","events":[{"ts":"12:01:05","level":"WARNING","msg":"First warning"},{"ts":"12:01:12","level":"ERROR","msg":"First error"}]}:::

:::quiz{"question":"What is the root cause?","options":["A","B","C"],"answer":0,"explanation":"Because..."}:::

BE DIRECT AND CONCISE. Answer only what was asked. Do not repeat general stats on every response. If the user asks for the first 10 log events, show them with :::log-ref::: blocks — nothing else. If asked for top errors, list them. If asked a yes/no question, answer it directly in one sentence.
"""

    messages = history[-30:] + [{"role": "user", "content": question}]
    max_iters = 10

    for _ in range(max_iters):
        full_text = []
        accumulated_tool_calls: dict = {}

        create_kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": True,
        }
        if supports_tools:
            create_kwargs["tools"] = TOOLS
        stream = await client.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield {"type": "token", "content": delta.content}
                full_text.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    if tc.id:
                        accumulated_tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated_tool_calls[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

        tool_use_blocks = list(accumulated_tool_calls.values())
        assistant_msg: dict = {"role": "assistant", "content": "".join(full_text)}

        if tool_use_blocks:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_use_blocks
            ]

        messages.append(assistant_msg)

        if not tool_use_blocks:
            break

        # Execute tools
        tool_results = []
        for tb in tool_use_blocks:
            yield {"type": "tool_call", "name": tb["name"], "input": tb["arguments"]}
            try:
                args = json.loads(tb["arguments"]) if isinstance(tb["arguments"], str) else tb["arguments"]
            except Exception:
                args = {}
            result_str = execute_tool(tb["name"], args, state)
            yield {"type": "tool_result", "name": tb["name"], "content": result_str[:300]}
            tool_results.append({
                "role": "tool",
                "tool_call_id": tb["id"],
                "content": result_str,
            })
        messages.extend(tool_results)

    yield {"type": "done"}


async def run_rag_stream(
    question: str,
    state: "AppState",
    client: Any,
    model: str = "llama-3.3-70b-versatile",
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
    system += """

RENDERING RULES: Use UI components only when they help — not on every response.
- :::metric::: for specific stats the user asked about
- :::chart::: for multi-value comparisons
- :::log-ref::: when quoting an actual log line
- :::timeline::: for ordered event sequences

BE CONCISE. Answer only what was asked. Do not dump all stats on every response.
"""

    results = state.rag_store.query(question, top_k=top_k)
    context = "\n\n---\n\n".join(
        f"[{r['chunk_type'].upper()} | {r['ts_start']} → {r['ts_end']} | score:{r['score']:.2f}]\n{r['text']}"
        for r in results
    )

    yield {"type": "status", "content": f"Retrieved {len(results)} relevant log chunks…"}

    stream = await client.chat.completions.create(
        model=model,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"<log_excerpts>\n{context}\n</log_excerpts>\n\nQuestion: {question}",
            },
        ],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield {"type": "token", "content": chunk.choices[0].delta.content}

    yield {"type": "done"}
