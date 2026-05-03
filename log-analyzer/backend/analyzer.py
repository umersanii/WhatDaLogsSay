"""
Generic log analyzer — builds statistical summaries and uses Claude
to characterize any log type without domain-specific hard-coding.
"""
import re
import json
from collections import defaultdict, Counter
from datetime import datetime
from typing import Optional
import anthropic

from backend.parser import LogEvent, FormatProfile


# ── Message pattern normalizer ────────────────────────────────────────────────

_NUM    = re.compile(r'\b\d+\b')
_HEX    = re.compile(r'\b[0-9a-fA-F]{8,}\b')
_IP     = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b')
_PATH   = re.compile(r'[/\\][\w./\\-]{4,}')
_URL    = re.compile(r'https?://\S+')
_QUOTED = re.compile(r"['\"]([^'\"]{3,60})['\"]")
_UUID   = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b')


def _normalise(msg: str) -> str:
    """Replace variable parts with tokens to cluster similar messages."""
    s = _URL.sub('<URL>', msg)
    s = _UUID.sub('<UUID>', s)
    s = _IP.sub('<IP>', s)
    s = _HEX.sub('<HEX>', s)
    s = _PATH.sub('<PATH>', s)
    s = _NUM.sub('<N>', s)
    return s[:140].strip()


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(events: list[LogEvent], profile: FormatProfile) -> dict:
    """
    Pure Python stats — no Claude calls.
    Returns the full statistical summary dict.
    """
    if not events:
        return {}

    timed = [e for e in events if e["ts_epoch"] > 0]

    # Date range
    if timed:
        first_dt = datetime.fromtimestamp(timed[0]["ts_epoch"])
        last_dt  = datetime.fromtimestamp(timed[-1]["ts_epoch"])
        span_h   = (last_dt - first_dt).total_seconds() / 3600
        date_range = {
            "first": first_dt.isoformat(sep=" ")[:19],
            "last":  last_dt.isoformat(sep=" ")[:19],
            "span_hours": round(span_h, 1),
            "days": sorted(set(
                datetime.fromtimestamp(e["ts_epoch"]).strftime("%Y-%m-%d")
                for e in timed
            )),
        }
    else:
        date_range = {"first": None, "last": None, "span_hours": 0, "days": []}

    # Level counts
    level_counts = dict(Counter(e["level"] for e in events).most_common())

    # Logger counts + error rate
    logger_totals: dict[str, int] = defaultdict(int)
    logger_errors: dict[str, int] = defaultdict(int)
    for e in events:
        logger_totals[e["logger"]] += 1
        if e["level"] in ("ERROR", "CRITICAL"):
            logger_errors[e["logger"]] += 1

    top_loggers = sorted(
        [
            {
                "logger": lg,
                "count": cnt,
                "error_rate": round(logger_errors[lg] / cnt, 3) if cnt else 0,
            }
            for lg, cnt in logger_totals.items()
        ],
        key=lambda x: -x["count"],
    )[:30]

    # Hourly buckets
    events_per_hour: dict[str, int] = defaultdict(int)
    errors_per_hour: dict[str, int] = defaultdict(int)
    for e in timed:
        h = datetime.fromtimestamp(e["ts_epoch"]).strftime("%Y-%m-%d %H:00")
        events_per_hour[h] += 1
        if e["level"] in ("ERROR", "CRITICAL"):
            errors_per_hour[h] += 1

    # Top message patterns
    pattern_counts = Counter(_normalise(e["msg"]) for e in events if e["msg"] and e["level"] != "RAW")
    top_patterns = [{"pattern": p, "count": c} for p, c in pattern_counts.most_common(30)]

    # Error samples (up to 5 per logger, max 60 total)
    error_by_logger: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e["level"] in ("ERROR", "CRITICAL") and len(error_by_logger[e["logger"]]) < 5:
            error_by_logger[e["logger"]].append({"ts": e["ts"], "msg": e["msg"][:300]})
    all_error_samples = []
    for lg, samples in error_by_logger.items():
        for s in samples:
            all_error_samples.append({"logger": lg, **s})
    all_error_samples = all_error_samples[:60]

    # Error bursts (hours where errors > 5× average)
    if errors_per_hour:
        avg_err = sum(errors_per_hour.values()) / len(errors_per_hour)
        threshold = max(5, avg_err * 5)
        error_bursts = [
            {"hour": h, "count": c}
            for h, c in sorted(errors_per_hour.items())
            if c > threshold
        ]
    else:
        error_bursts = []

    # Entities: quoted strings, IPs, file paths
    quoted_counter: Counter = Counter()
    ip_counter:     Counter = Counter()
    path_counter:   Counter = Counter()
    for e in events:
        for q in _QUOTED.findall(e["msg"]):
            quoted_counter[q] += 1
        for ip in _IP.findall(e["msg"]):
            ip_counter[ip] += 1
        for p in _PATH.findall(e["msg"]):
            path_counter[p.split("/")[-1] or p] += 1

    entities = {
        "quoted_values": dict(quoted_counter.most_common(20)),
        "ip_addresses":  dict(ip_counter.most_common(10)),
        "file_paths":    dict(path_counter.most_common(20)),
    }

    return {
        "format": profile["name"],
        "total_events": len(events),
        "parsed_lines": sum(1 for e in events if e["level"] != "RAW"),
        "skipped_lines": sum(1 for e in events if e["level"] == "RAW"),
        "date_range": date_range,
        "level_counts": level_counts,
        "top_loggers": top_loggers,
        "events_per_hour": dict(sorted(events_per_hour.items())),
        "errors_per_hour": dict(sorted(errors_per_hour.items())),
        "top_patterns": top_patterns,
        "error_samples": all_error_samples,
        "error_bursts": error_bursts,
        "entities": entities,
    }


# ── Claude characterization ───────────────────────────────────────────────────

_CHARACTERIZE_PROMPT = """You are given a sample from an unknown log file plus basic statistics.

Analyze the sample and return a JSON object with EXACTLY these keys (no markdown fences):

{{
  "log_type": "one-line label, e.g. NVIDIA Jetson computer-vision pipeline",
  "system_description": "2-3 sentences describing the system that produced this log",
  "key_entities": ["list of important identifiers: IDs, hostnames, model names, version strings"],
  "key_event_types": ["important recurring event verbs/nouns, e.g. detection, reconnect, init"],
  "important_loggers": ["logger names that carry the most signal (skip noisy HTTP/access loggers)"],
  "noisy_loggers": ["logger names that are high-volume but low-signal, e.g. aiohttp.access, httpx"],
  "agent_system_prompt": "A complete system prompt (300-500 words) for a Claude agent that answers questions about THIS specific log. Include inferred domain knowledge. The agent has tools: search_logs, get_stats, get_timeline, get_samples.",
  "rag_system_prompt": "A complete system prompt (200-300 words) for a Claude RAG assistant that receives raw log excerpts as context. Be specific about the system domain."
}}

Stats:
{stats}

<log_sample>
{sample}
</log_sample>"""


def _build_sample(events: list[LogEvent], n: int = 180) -> str:
    """Take n events spread across beginning, middle, and end of the log."""
    if not events:
        return ""
    third = n // 3
    indices = (
        list(range(min(third, len(events))))
        + list(range(max(0, len(events) // 2 - third // 2), min(len(events), len(events) // 2 + third // 2)))
        + list(range(max(0, len(events) - third), len(events)))
    )
    seen = set()
    lines = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            e = events[i]
            lines.append(f"[{e['ts']}] {e['level']} {e['logger']}: {e['msg'][:200]}")
    return "\n".join(lines)


def characterize_log(
    events: list[LogEvent],
    stats: dict,
    client: anthropic.Anthropic,
) -> dict:
    """
    Ask Claude to characterize the log. Returns the parsed JSON dict.
    Falls back to a safe default on any error.
    """
    sample = _build_sample(events)
    stats_brief = {
        "total": stats.get("total_events"),
        "format": stats.get("format"),
        "date_range": stats.get("date_range"),
        "level_counts": stats.get("level_counts"),
        "top_loggers": [l["logger"] for l in (stats.get("top_loggers") or [])[:10]],
        "top_patterns": [(p["pattern"][:80], p["count"]) for p in (stats.get("top_patterns") or [])[:5]],
    }

    prompt = _CHARACTERIZE_PROMPT.format(
        stats=json.dumps(stats_brief, default=str, indent=2),
        sample=sample,
    )

    try:
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Strip markdown fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception as e:
        return {
            "log_type": f"Unknown log ({stats.get('format', 'raw')})",
            "system_description": "Unable to characterize this log automatically.",
            "key_entities": [],
            "key_event_types": [],
            "important_loggers": [],
            "noisy_loggers": ["httpx", "aiohttp.access"],
            "agent_system_prompt": (
                "You are a log file analyst. You have access to tools to search and query "
                "the log data. Answer questions accurately based on what you find. "
                "Format responses in clean markdown."
            ),
            "rag_system_prompt": (
                "You are a log analyst. Answer questions using ONLY the provided log excerpts. "
                "Be specific with timestamps and values from the excerpts."
            ),
        }


def build_summary(
    events: list[LogEvent],
    profile: FormatProfile,
    client: anthropic.Anthropic,
) -> dict:
    """
    Full pipeline: compute stats then Claude characterization.
    Returns the complete summary dict including characterization.
    """
    stats = compute_stats(events, profile)
    characterization = characterize_log(events, stats, client)
    stats["characterization"] = characterization
    return stats
