"""
Full log analysis report generator.
Calls the LLM sequentially — one call per section — to produce a comprehensive,
long-form report without hitting token limits.

Rate-limit friendly: adds a configurable delay between LLM calls and retries
with exponential backoff on 429 / transient errors.
"""
import json
import asyncio
import traceback
from datetime import datetime
from typing import AsyncIterator, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.api import AppState

# Delay (seconds) between consecutive LLM calls — keeps free-plan rate limits happy.
INTER_CALL_DELAY = 3.0
# Retry settings for rate-limit errors
MAX_RETRIES = 3
RETRY_BASE_DELAY = 8.0  # doubles each retry

# ── Section definitions ───────────────────────────────────────────────────────

SECTIONS = [
    "cover",
    "table_of_contents",
    "abstract",
    "executive_summary",
    "system_overview",
    "statistical_overview",
    "error_analysis",
    "error_bursts",
    "performance_throughput",
    "component_health",
    "pattern_analysis",
    "entity_analysis",
    "pain_points",
    "recommendations",
    "appendix",
]

SECTION_TITLES = {
    "cover":                 "Cover",
    "table_of_contents":     "Table of Contents",
    "abstract":              "Abstract",
    "executive_summary":     "Executive Summary",
    "system_overview":       "System Overview",
    "statistical_overview":  "Statistical Overview",
    "error_analysis":        "Error Analysis",
    "error_bursts":          "Error Burst Analysis",
    "performance_throughput":"Performance & Throughput",
    "component_health":      "Component Health Report",
    "pattern_analysis":      "Pattern Analysis",
    "entity_analysis":       "Entity Analysis",
    "pain_points":           "Pain Points & Where to Look",
    "recommendations":       "Recommendations",
    "appendix":              "Appendix",
}

# ── Prompt builders ────────────────────────────────────────────────────────────

def _fmt_json(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)


def _build_abstract_prompt(summary: dict) -> str:
    char = summary.get("characterization", {})
    dr = summary.get("date_range", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    return f"""You are writing the ABSTRACT section of a formal technical log analysis report.

Write 250-350 words as a single cohesive paragraph (no headers). Cover:
- What system produced this log and what it does
- Time period and scale of data
- Most important findings (critical errors, key patterns)
- Overall system health verdict

Log context:
- System: {char.get('log_type', 'Unknown')} — {char.get('system_description', '')}
- Time range: {dr.get('first', 'N/A')} → {dr.get('last', 'N/A')} ({dr.get('span_hours', 0):.1f} hours)
- Total events: {total:,} | Error rate: {errors/max(total,1)*100:.2f}% ({errors:,} errors)
- Level distribution: {_fmt_json(lc)}
- Error bursts: {_fmt_json(summary.get('error_bursts', [])[:5])}
- Key entities: {char.get('key_entities', [])}

Write ONLY the abstract text. No headers, no bullet points. Formal technical language."""


def _build_executive_summary_prompt(summary: dict) -> str:
    char = summary.get("characterization", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    bursts = summary.get("error_bursts", [])
    top_loggers = summary.get("top_loggers", [])[:10]
    error_samples = summary.get("error_samples", [])[:10]
    return f"""Write the EXECUTIVE SUMMARY section of a log analysis report (400-550 words).

Use these markdown headers exactly:

## Key Findings
List the 5 most important findings with specific timestamps and counts.

## System Health Verdict
Overall health: Healthy / Degraded / Critical. One paragraph justifying the verdict.

## Top 3 Critical Pain Points
For each: what is happening, when (specific timestamps), which logger, severity.

## Immediate Actions Required
3-5 concrete numbered actions ordered by priority.

Log data:
- System: {char.get('log_type', 'Unknown')} — {char.get('system_description', '')}
- Time: {summary.get('date_range', {}).get('first')} → {summary.get('date_range', {}).get('last')}
- Events: {total:,} | Errors: {errors:,} ({errors/max(total,1)*100:.1f}%)
- Level counts: {_fmt_json(lc)}
- Error bursts: {_fmt_json(bursts)}
- Top loggers by error rate: {_fmt_json(sorted(top_loggers, key=lambda x: -x.get('error_rate',0))[:6])}
- Error samples: {_fmt_json(error_samples)}

Be specific — use real timestamps and numbers."""


def _build_system_overview_prompt(summary: dict) -> str:
    char = summary.get("characterization", {})
    top_loggers = summary.get("top_loggers", [])[:20]
    dr = summary.get("date_range", {})
    return f"""Write the SYSTEM OVERVIEW section of a log analysis report (400-500 words).

Use these markdown headers:

## System Description
What system/application produced this log, its purpose and technology stack.

## Component Inventory
Table or list: each significant logger → what it does → event count → role.

## Operational Context
When captured, how long, what the system was doing, any notable init/shutdown events.

## Key Identifiers
Important model names, IDs, hostnames, version strings from the logs.

Log data:
- Log type: {char.get('log_type', 'Unknown')}
- Description: {char.get('system_description', '')}
- Key entities: {char.get('key_entities', [])}
- Key event types: {char.get('key_event_types', [])}
- Important loggers: {char.get('important_loggers', [])}
- Noisy loggers: {char.get('noisy_loggers', [])}
- Time range: {dr.get('first')} → {dr.get('last')} ({dr.get('span_hours', 0):.1f} hours)
- All loggers: {_fmt_json(top_loggers[:15])}"""


def _build_error_analysis_prompt(summary: dict, rag_context: str) -> str:
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    error_samples = summary.get("error_samples", [])
    top_loggers = summary.get("top_loggers", [])
    high_error_loggers = [l for l in top_loggers if l.get("error_rate", 0) > 0.01][:8]
    return f"""Write the ERROR ANALYSIS section of a log analysis report (550-700 words).

Use these markdown headers:

## Error Overview
Total errors, percentage, severity breakdown by level.

## Error Categories
Group errors by root cause. For each: frequency, affected components, example log line.

## Root Cause Analysis
For the top 3 most frequent errors: likely root cause, correlation with other events.

## Most Problematic Loggers
Loggers with highest error count and highest error rate. What their patterns tell us.

## Critical Events
The 3-5 most severe individual events with exact timestamps and why they matter.

Log data:
- Total: {total:,} | Errors: {errors:,} | Critical: {lc.get('CRITICAL', 0):,}
- Level counts: {_fmt_json(lc)}
- High-error-rate loggers: {_fmt_json(high_error_loggers)}
- Error samples: {_fmt_json(error_samples[:20])}

Log excerpts:
{rag_context}

Include exact timestamps and actual log messages. Be technical."""


def _build_error_bursts_prompt(summary: dict, rag_context: str) -> str:
    bursts = summary.get("error_bursts", [])
    errors_per_hour = summary.get("errors_per_hour", {})
    error_samples = summary.get("error_samples", [])[:15]
    if not bursts:
        return """Write the ERROR BURST ANALYSIS section (150 words).
State that no significant error bursts were detected in this log.
Describe the normal error distribution pattern instead.
Use markdown."""
    return f"""Write the ERROR BURST ANALYSIS section of a log analysis report (500-650 words).

For EACH burst write a subsection:
### Burst at [TIME]
- Exact window, peak error count, affected loggers
- Sequence of events leading in and out of the burst
- Likely trigger based on the error messages
- Recovery time and method

Then add:
## Burst Patterns
Common triggers, leading indicators, correlation with restarts/load.

## Impact
What was disrupted during bursts.

Log data:
- Error bursts detected: {_fmt_json(bursts)}
- Errors per hour context: {_fmt_json(dict(list(errors_per_hour.items())[:20]))}
- Error samples: {_fmt_json(error_samples)}

Log excerpts:
{rag_context}

Use exact timestamps. Be very specific about what happened in each burst."""


def _build_performance_prompt(summary: dict, rag_context: str) -> str:
    events_per_hour = summary.get("events_per_hour", {})
    errors_per_hour = summary.get("errors_per_hour", {})
    dr = summary.get("date_range", {})
    total = summary.get("total_events", 0)
    hours = max(dr.get("span_hours", 1) or 1, 0.01)
    return f"""Write the PERFORMANCE & THROUGHPUT section of a log analysis report (400-500 words).

## Throughput Analysis
Average events/hour, peak periods (what was happening), low periods.

## Performance Patterns
Diurnal patterns, throughput drops, correlation between volume and errors.

## Bottleneck Indicators
Periods of stress (high errors + high volume), signs of system being overwhelmed.

## System Availability
Uptime estimate, gaps in logging that might indicate crashes.

Log data:
- Time range: {dr.get('first')} → {dr.get('last')} ({hours:.1f} hours)
- Total events: {total:,} | Avg/hour: {total/hours:.0f}
- Events per hour (sample): {_fmt_json(dict(list(events_per_hour.items())[:15]))}
- Errors per hour (sample): {_fmt_json(dict(list(errors_per_hour.items())[:15]))}

Log excerpts:
{rag_context}

Reference specific timestamps when discussing patterns."""


def _build_component_health_prompt(summary: dict, rag_context: str) -> str:
    top_loggers = summary.get("top_loggers", [])[:20]
    char = summary.get("characterization", {})
    error_samples = summary.get("error_samples", [])[:15]
    return f"""Write the COMPONENT HEALTH REPORT section of a log analysis report (500-650 words).

## Health Summary Table
For each significant logger: | Logger | Events | Error Rate | Status |
Status: ✅ Healthy / ⚠️ Warning / 🔴 Critical

## Detailed Analysis (Warning/Critical only)
For each problematic component:
### [Logger Name] — ⚠️/🔴 STATUS
- What it does, event count, error rate
- Specific problematic behaviors with example log lines
- Recommended action

## Healthy Components
Brief one-liner for loggers operating normally.

## Cascading Failures
Any components where one failure causes others to fail.

Log data:
- Important loggers: {char.get('important_loggers', [])}
- Noisy loggers: {char.get('noisy_loggers', [])}
- All loggers: {_fmt_json(top_loggers)}
- Error samples: {_fmt_json(error_samples)}

Log excerpts:
{rag_context}"""


def _build_pattern_analysis_prompt(summary: dict) -> str:
    top_patterns = summary.get("top_patterns", [])[:25]
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    return f"""Write the PATTERN ANALYSIS section of a log analysis report (400-500 words).

## Most Frequent Patterns
Top recurring messages: what they indicate about normal vs. abnormal operation.

## Anomalous Patterns
Patterns at unexpected frequency. Any that suggest bugs or misconfigurations.

## Pattern Categories
Group into: Initialization / Normal operation / Warning-error / Retry-recovery.

## Actionable Insights
Which patterns would make good alert triggers. Any inefficiency indicators.

Log data:
- Total events: {total:,} | Levels: {_fmt_json(lc)}
- Top patterns (normalized, with count): {_fmt_json(top_patterns)}

Quote specific pattern strings in your analysis."""


def _build_entity_analysis_prompt(summary: dict) -> str:
    entities = summary.get("entities", {})
    char = summary.get("characterization", {})
    return f"""Write the ENTITY ANALYSIS section of a log analysis report (300-400 words).

## Network Entities
IP addresses/hostnames: which are most active, their roles, anything suspicious.

## File System Paths
Referenced paths: what they reveal about system layout, any in error contexts.

## Key Identifiers
Most common quoted values/IDs: what they represent, any appearing in errors.

Log data:
- Key entities (AI-identified): {char.get('key_entities', [])}
- IPs: {_fmt_json(entities.get('ip_addresses', {}))}
- File paths: {_fmt_json(entities.get('file_paths', {}))}
- Quoted values: {_fmt_json(entities.get('quoted_values', {}))}"""


def _build_pain_points_prompt(summary: dict, rag_context: str) -> str:
    bursts = summary.get("error_bursts", [])
    error_samples = summary.get("error_samples", [])
    top_loggers = summary.get("top_loggers", [])
    high_error = [l for l in top_loggers if l.get("error_rate", 0) > 0.05][:6]
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    return f"""Write PAIN POINTS & WHERE TO LOOK — the most actionable section of this report (600-800 words).

This is a debugging guide for engineers. For each pain point use this exact format:

### Pain Point N: [Clear Title]
**Severity:** Critical / High / Medium / Low
**When:** [exact timestamps or frequency]
**Logger(s):** `logger_name`
**Search for:** `keyword or pattern`
**What's happening:** one sentence description
**Sample log line:** quote an actual log line with timestamp
**Likely cause:** technical hypothesis
**Impact:** what breaks or degrades

Identify 5-7 distinct pain points ordered by severity (Critical first).

Log data:
- Events: {total:,} | Errors: {lc.get('ERROR',0)+lc.get('CRITICAL',0):,}
- High error-rate loggers: {_fmt_json(high_error)}
- Error bursts: {_fmt_json(bursts)}
- Error samples: {_fmt_json(error_samples[:20])}

Log excerpts:
{rag_context}

Include exact timestamps and actual quoted log lines. Engineers will use this to debug."""


def _build_recommendations_prompt(summary: dict, rag_context: str) -> str:
    char = summary.get("characterization", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    bursts = summary.get("error_bursts", [])
    top_loggers = summary.get("top_loggers", [])[:10]
    return f"""Write the RECOMMENDATIONS section of a log analysis report (450-550 words).

## Immediate Actions (within 24 hours)
Numbered list. Fix critical issues. Each cites specific log evidence.

## Short-term (1-2 weeks)
Root cause fixes, reliability improvements, alerting improvements.

## Long-term
Architecture changes, logging hygiene (what to add/remove), observability.

## Monitoring & Alerting
3-5 specific alert thresholds based on this log's patterns.

Log data:
- System: {char.get('log_type')} — {char.get('system_description', '')}
- Error rate: {errors/max(total,1)*100:.1f}% ({errors:,}/{total:,})
- Error bursts: {_fmt_json(bursts)}
- Logger error rates: {_fmt_json(sorted(top_loggers, key=lambda x: -x.get('error_rate',0))[:8])}

Log excerpts:
{rag_context}

Each recommendation must cite specific log evidence."""


# ── RAG helper ─────────────────────────────────────────────────────────────────

def _rag_search(state: "AppState", query: str, top_k: int = 5) -> str:
    if not state.rag_store:
        return "(RAG index not available)"
    try:
        results = state.rag_store.query(query, top_k=top_k)
    except Exception:
        return "(RAG search failed)"
    if not results:
        return "(No relevant excerpts found)"
    parts = []
    for r in results:
        parts.append(
            f"[{r['chunk_type'].upper()} | {r['ts_start']} → {r['ts_end']}]\n{r['text'][:500]}"
        )
    return "\n\n---\n\n".join(parts)


# ── LLM call with retry ────────────────────────────────────────────────────────

async def _call_llm_with_retry(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int = 1500,
) -> str:
    """Call LLM with exponential-backoff retry on rate-limit errors."""
    delay = RETRY_BASE_DELAY
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Retry on rate-limit or transient errors
            if "rate" in err_str or "429" in err_str or "timeout" in err_str or "502" in err_str or "503" in err_str:
                if attempt < MAX_RETRIES - 1:
                    wait = delay * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue
            # Non-retriable error — raise immediately
            raise
    raise last_err or RuntimeError("LLM call failed after retries")


# ── Main generator ─────────────────────────────────────────────────────────────

async def generate_report_sections(
    state: "AppState",
    client: Any,
    model: str,
) -> AsyncIterator[dict]:
    """
    Yields SSE-ready dicts for each report section.
    Sections are generated sequentially with INTER_CALL_DELAY between LLM calls.
    All exceptions are caught and surfaced as section error content so the SSE
    stream never closes unexpectedly.
    """
    summary = state.summary or {}
    char    = summary.get("characterization", {})
    dr      = summary.get("date_range", {})
    lc      = summary.get("level_counts", {})
    total   = summary.get("total_events", len(state.events))
    errors  = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)

    yield {"type": "start", "total_sections": len(SECTIONS)}

    # Track how many LLM calls we've made so we can insert delays
    llm_call_count = 0

    async def llm_section(section_id: str, index: int, prompt_fn, max_tokens: int = 1500, rag_query: str = ""):
        """Helper: yield start → content → done for one LLM section."""
        nonlocal llm_call_count
        yield {"type": "section_start", "section": section_id, "index": index}

        # Delay between LLM calls to respect rate limits
        if llm_call_count > 0:
            yield {"type": "status", "content": f"Waiting {INTER_CALL_DELAY}s to respect rate limits…"}
            await asyncio.sleep(INTER_CALL_DELAY)

        try:
            rag_ctx = _rag_search(state, rag_query, top_k=5) if rag_query else ""
            prompt  = prompt_fn(summary, rag_ctx) if rag_query else prompt_fn(summary)
            text    = await _call_llm_with_retry(client, model, prompt, max_tokens=max_tokens)
            llm_call_count += 1
            yield {"type": "section_content", "section": section_id, "content": text}
        except Exception as e:
            tb = traceback.format_exc()
            yield {"type": "section_content", "section": section_id,
                   "content": f"*Error generating this section: {e}*\n\n```\n{tb[-500:]}\n```"}
        yield {"type": "section_done", "section": section_id}

    # ── 1. Cover (static) ──────────────────────────────────────────────────────
    yield {"type": "section_start", "section": "cover", "index": 0}
    yield {"type": "section_content", "section": "cover", "content": json.dumps({
        "filename":        state.log_filename,
        "generated_at":    datetime.now().isoformat(timespec="seconds"),
        "log_type":        char.get("log_type", "Unknown"),
        "format":          summary.get("format", "Unknown"),
        "time_range_first": dr.get("first", "N/A"),
        "time_range_last":  dr.get("last", "N/A"),
        "span_hours":      dr.get("span_hours", 0),
        "total_events":    total,
        "error_count":     errors,
        "error_rate_pct":  round(errors / max(total, 1) * 100, 2),
        "system_description": char.get("system_description", ""),
    })}
    yield {"type": "section_done", "section": "cover"}

    # ── 2. Table of Contents (static) ─────────────────────────────────────────
    yield {"type": "section_start", "section": "table_of_contents", "index": 1}
    yield {"type": "section_content", "section": "table_of_contents", "content": json.dumps([
        {"num": i+1, "title": SECTION_TITLES[s], "section": s}
        for i, s in enumerate(SECTIONS[2:], start=1)
    ])}
    yield {"type": "section_done", "section": "table_of_contents"}

    # ── 3. Abstract (LLM) ─────────────────────────────────────────────────────
    async for msg in llm_section("abstract", 2, _build_abstract_prompt, max_tokens=500):
        yield msg

    # ── 4. Executive Summary (LLM) ────────────────────────────────────────────
    async for msg in llm_section("executive_summary", 3, _build_executive_summary_prompt, max_tokens=800):
        yield msg

    # ── 5. System Overview (LLM) ──────────────────────────────────────────────
    async for msg in llm_section("system_overview", 4, _build_system_overview_prompt, max_tokens=700):
        yield msg

    # ── 6. Statistical Overview (static — chart data, no LLM) ────────────────
    yield {"type": "section_start", "section": "statistical_overview", "index": 5}
    yield {"type": "section_content", "section": "statistical_overview", "content": json.dumps({
        "level_counts":    lc,
        "total_events":    total,
        "events_per_hour": summary.get("events_per_hour", {}),
        "errors_per_hour": summary.get("errors_per_hour", {}),
        "top_loggers":     summary.get("top_loggers", [])[:20],
        "error_bursts":    summary.get("error_bursts", []),
        "date_range":      dr,
    })}
    yield {"type": "section_done", "section": "statistical_overview"}

    # ── 7. Error Analysis (LLM + RAG) ─────────────────────────────────────────
    async for msg in llm_section("error_analysis", 6,
                                  _build_error_analysis_prompt, max_tokens=900,
                                  rag_query="error critical failure exception traceback"):
        yield msg

    # ── 8. Error Bursts (LLM + RAG) ───────────────────────────────────────────
    async for msg in llm_section("error_bursts", 7,
                                  _build_error_bursts_prompt, max_tokens=800,
                                  rag_query="error burst spike surge multiple failures"):
        yield msg

    # ── 9. Performance & Throughput (LLM + RAG) ───────────────────────────────
    async for msg in llm_section("performance_throughput", 8,
                                  _build_performance_prompt, max_tokens=700,
                                  rag_query="slow timeout latency throughput performance"):
        yield msg

    # ── 10. Component Health (LLM + RAG) ──────────────────────────────────────
    async for msg in llm_section("component_health", 9,
                                  _build_component_health_prompt, max_tokens=800,
                                  rag_query="component service module initialization startup failure"):
        yield msg

    # ── 11. Pattern Analysis (LLM, no RAG) ────────────────────────────────────
    async for msg in llm_section("pattern_analysis", 10, _build_pattern_analysis_prompt, max_tokens=600):
        yield msg

    # ── 12. Entity Analysis (LLM, no RAG) ─────────────────────────────────────
    async for msg in llm_section("entity_analysis", 11, _build_entity_analysis_prompt, max_tokens=500):
        yield msg

    # ── 13. Pain Points (LLM + RAG) ───────────────────────────────────────────
    async for msg in llm_section("pain_points", 12,
                                  _build_pain_points_prompt, max_tokens=1200,
                                  rag_query="error critical warning failure problem issue"):
        yield msg

    # ── 14. Recommendations (LLM + RAG) ──────────────────────────────────────
    async for msg in llm_section("recommendations", 13,
                                  _build_recommendations_prompt, max_tokens=700,
                                  rag_query="retry reconnect recovery workaround fix mitigation"):
        yield msg

    # ── 15. Appendix (static) ─────────────────────────────────────────────────
    yield {"type": "section_start", "section": "appendix", "index": 14}
    yield {"type": "section_content", "section": "appendix", "content": json.dumps({
        "error_samples": summary.get("error_samples", []),
        "error_bursts":  summary.get("error_bursts", []),
        "top_patterns":  summary.get("top_patterns", [])[:30],
        "all_loggers":   summary.get("top_loggers", []),
        "entities":      summary.get("entities", {}),
    })}
    yield {"type": "section_done", "section": "appendix"}

    yield {"type": "complete"}
