"""
Full log analysis report generator.
Calls the LLM sequentially — one call per section — to produce a comprehensive,
long-form report without hitting token limits.
Each section streams back via an async generator yielding SSE-ready dicts.
"""
import json
import asyncio
from datetime import datetime
from typing import AsyncIterator, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.api import AppState

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
    return f"""You are writing a formal technical report on a log file analysis.

Write the ABSTRACT section (300-400 words). This should be a standalone paragraph that concisely covers:
- What system produced this log and what it does
- The time period covered and scale of data
- The most important findings (critical errors, performance issues, patterns)
- The overall system health verdict
- What the reader will find in the full report

Log context:
- System type: {char.get('log_type', 'Unknown')}
- System description: {char.get('system_description', 'N/A')}
- Time range: {dr.get('first', 'N/A')} → {dr.get('last', 'N/A')} ({dr.get('span_hours', 0):.1f} hours)
- Total events: {total:,}
- Level distribution: {_fmt_json(lc)}
- Error rate: {errors/total*100:.2f}% ({errors:,} errors/criticals out of {total:,})
- Key entities: {char.get('key_entities', [])}
- Key event types: {char.get('key_event_types', [])}
- Error bursts: {_fmt_json(summary.get('error_bursts', [])[:5])}

Write ONLY the abstract text — no headers, no markdown fences. Use formal technical language."""


def _build_executive_summary_prompt(summary: dict) -> str:
    char = summary.get("characterization", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    bursts = summary.get("error_bursts", [])
    top_loggers = summary.get("top_loggers", [])[:10]
    error_samples = summary.get("error_samples", [])[:15]
    return f"""You are writing the EXECUTIVE SUMMARY section of a formal log analysis report.

This section is for technical managers and senior engineers. Write 500-700 words covering:

## Key Findings
- Top 5 most important findings from the log, with specific timestamps and counts

## System Health Verdict
- Overall health score (Healthy / Degraded / Critical) with justification
- Which components are healthy vs problematic

## Top 3 Critical Pain Points
For each pain point:
- What is happening (describe the problem)
- When it happens (specific timestamps or time ranges)
- Which logger/component is responsible
- How severe is it (quantify with real numbers)

## Immediate Actions Required
- 3-5 concrete actions the team should take, ordered by priority

Log data:
- System: {char.get('log_type', 'Unknown')} — {char.get('system_description', '')}
- Time range: {summary.get('date_range', {}).get('first')} → {summary.get('date_range', {}).get('last')}
- Total events: {total:,} | Errors: {errors:,} ({errors/max(total,1)*100:.1f}%)
- Level counts: {_fmt_json(lc)}
- Error bursts: {_fmt_json(bursts)}
- Top loggers by error rate: {_fmt_json(sorted(top_loggers, key=lambda x: -x.get('error_rate', 0))[:8])}
- Error samples: {_fmt_json(error_samples)}

Write in markdown with headers (##, ###). Be specific — use real timestamps and numbers from the data."""


def _build_system_overview_prompt(summary: dict) -> str:
    char = summary.get("characterization", {})
    top_loggers = summary.get("top_loggers", [])[:20]
    dr = summary.get("date_range", {})
    return f"""You are writing the SYSTEM OVERVIEW section of a formal log analysis report.

Write 500-600 words covering:

## System Description
- What system/application produced this log
- Inferred architecture and purpose
- Technology stack clues from the log data

## Component Inventory
- List every significant logger as a component
- For each component: what it does, how active it is, its role in the system
- Identify the system topology (what calls what)

## Operational Context
- When was this log captured and for how long
- What was the system doing during this period
- Any notable initialization, shutdown, or maintenance events

## Key Identifiers
- Important model names, IDs, hostnames, version strings found in the logs

Log data:
- Log type: {char.get('log_type', 'Unknown')}
- System description: {char.get('system_description', '')}
- Key entities: {char.get('key_entities', [])}
- Key event types: {char.get('key_event_types', [])}
- Important loggers: {char.get('important_loggers', [])}
- Noisy loggers: {char.get('noisy_loggers', [])}
- Time range: {dr.get('first')} → {dr.get('last')} ({dr.get('span_hours', 0):.1f} hours, days: {dr.get('days', [])})
- All loggers: {_fmt_json(top_loggers)}

Write in markdown with headers. Be specific and technical."""


def _build_error_analysis_prompt(summary: dict, rag_context: str) -> str:
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    error_samples = summary.get("error_samples", [])
    top_loggers = summary.get("top_loggers", [])
    high_error_loggers = [l for l in top_loggers if l.get("error_rate", 0) > 0.01]
    return f"""You are writing the ERROR ANALYSIS section of a formal log analysis report.

Write 700-900 words with deep technical analysis covering:

## Error Overview
- Total error count, percentage, distribution by severity
- Timeline of when errors occur (which hours/periods are worst)

## Error Categories
- Group errors into categories by root cause or pattern
- For each category: frequency, affected components, example log lines (use actual lines from the samples)

## Root Cause Analysis
- For the most frequent errors: what is the likely root cause?
- Are errors correlated (does A cause B)?
- Are errors clustered in time (bursts) or spread evenly?

## Most Problematic Loggers
- Which loggers have the highest absolute error counts?
- Which have the highest error RATE (errors/total)?
- What does each logger's error pattern tell us?

## Critical Events
- List the most severe individual events with their exact timestamps
- Explain why each is significant

Log data:
- Total events: {total:,} | Errors: {errors:,} | Critical: {lc.get('CRITICAL', 0):,}
- Level counts: {_fmt_json(lc)}
- High-error-rate loggers: {_fmt_json(high_error_loggers)}
- Error samples (up to 60): {_fmt_json(error_samples)}

Relevant log excerpts from semantic search:
{rag_context}

Write in markdown with headers. Include exact timestamps and log messages in your analysis. Be technical and precise."""


def _build_error_bursts_prompt(summary: dict, rag_context: str) -> str:
    bursts = summary.get("error_bursts", [])
    errors_per_hour = summary.get("errors_per_hour", {})
    events_per_hour = summary.get("events_per_hour", {})
    error_samples = summary.get("error_samples", [])[:20]
    return f"""You are writing the ERROR BURST ANALYSIS section of a formal log analysis report.

Write 600-800 words. An "error burst" is a period where errors spike far above the normal rate. For each burst:

## Burst Identification
- List each burst with exact time, duration, and peak error rate
- Compare to normal (baseline) error rate

## Burst Deep Dives
For EACH burst, write a dedicated subsection:
### Burst at [TIMESTAMP]
- Exact time window
- Number of errors in this burst
- Which loggers were affected
- The sequence of events leading into and out of the burst
- What the error messages reveal about the trigger
- Was the system able to recover? How long did recovery take?

## Burst Patterns
- Do bursts follow a predictable pattern?
- Are there leading indicators (warnings before bursts)?
- Are bursts correlated with specific events (restarts, load spikes, config changes)?

## Impact Assessment
- What was the user/system impact during bursts?
- What operations were disrupted?

Log data:
- Error bursts (hours where errors > 5× average): {_fmt_json(bursts)}
- Errors per hour (all): {_fmt_json(errors_per_hour)}
- Events per hour (all): {_fmt_json(events_per_hour)}
- Error samples: {_fmt_json(error_samples)}

Relevant log excerpts from semantic search:
{rag_context}

Write in markdown. Use exact timestamps. Be very specific about what happened during each burst."""


def _build_performance_prompt(summary: dict, rag_context: str) -> str:
    events_per_hour = summary.get("events_per_hour", {})
    errors_per_hour = summary.get("errors_per_hour", {})
    dr = summary.get("date_range", {})
    total = summary.get("total_events", 0)
    hours = dr.get("span_hours", 1) or 1
    avg_rate = total / hours
    return f"""You are writing the PERFORMANCE & THROUGHPUT section of a formal log analysis report.

Write 500-700 words covering:

## Throughput Analysis
- Average events per hour across the log period
- Peak throughput periods (highest events/hour) — what was happening?
- Low throughput periods — idle time, maintenance windows, or problems?
- Throughput variability (stable vs. erratic)

## Performance Patterns
- Is there a diurnal pattern (busy/quiet at certain hours)?
- Are there sudden throughput drops that might indicate failures?
- Correlation between high throughput and high error rates

## Bottleneck Indicators
- Which periods show stress indicators (high errors + high volume)?
- Are there periods where the system appears overwhelmed?

## System Availability
- Estimate uptime vs. downtime periods
- Any gaps in logging that might indicate crashes or restarts?

Log data:
- Time range: {dr.get('first')} → {dr.get('last')} ({hours:.1f} hours)
- Total events: {total:,}
- Average events/hour: {avg_rate:.0f}
- Events per hour: {_fmt_json(events_per_hour)}
- Errors per hour: {_fmt_json(errors_per_hour)}

Relevant log excerpts from semantic search:
{rag_context}

Write in markdown with headers. Reference specific timestamps when discussing patterns."""


def _build_component_health_prompt(summary: dict, rag_context: str) -> str:
    top_loggers = summary.get("top_loggers", [])[:25]
    char = summary.get("characterization", {})
    important = char.get("important_loggers", [])
    noisy = char.get("noisy_loggers", [])
    error_samples = summary.get("error_samples", [])[:20]
    return f"""You are writing the COMPONENT HEALTH REPORT section of a formal log analysis report.

Write 700-900 words. Assess the health of EVERY significant component (logger) in the system.

## Health Assessment Methodology
- Explain how health is determined (error rate, activity, patterns)

## Component Health Table Summary
- List each component with: event count, error count, error rate %, health status (Healthy/Warning/Critical)

## Detailed Component Analysis
For each component with Warning or Critical status, write a subsection:
### [Component Name] — STATUS
- What this component does
- How many events it logged
- Error rate and types of errors
- Specific problematic behaviors observed
- Recommended actions

## Healthy Components
- Brief mention of components operating normally
- Note any components that are suspiciously silent (possible dead code or down services)

## Cross-Component Dependencies
- Which components interact based on log patterns?
- Are there cascading failures (one component failing causes others to fail)?

Log data:
- Important loggers: {important}
- Noisy/low-signal loggers: {noisy}
- All loggers with counts and error rates: {_fmt_json(top_loggers)}
- Error samples by logger: {_fmt_json(error_samples)}

Relevant log excerpts:
{rag_context}

Write in markdown. For each Warning/Critical component give a health score like: ⚠️ WARNING or 🔴 CRITICAL."""


def _build_pattern_analysis_prompt(summary: dict) -> str:
    top_patterns = summary.get("top_patterns", [])[:30]
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    return f"""You are writing the PATTERN ANALYSIS section of a formal log analysis report.

Write 500-700 words covering:

## Most Frequent Message Patterns
- Analyze the top recurring message patterns
- What do high-frequency patterns tell us about normal system operation?
- Which patterns indicate healthy activity vs. problems?

## Anomalous Patterns
- Which patterns appear at abnormal frequencies?
- Are there patterns that should NOT be this common?
- Any patterns that suggest misconfigurations or bugs?

## Pattern Categories
Group patterns into categories:
- Initialization/startup patterns
- Normal operational patterns
- Warning/error patterns
- Retry/recovery patterns

## Pattern Insights
- What do the patterns reveal about the system's architecture?
- Are there any patterns that suggest inefficiencies?
- Any patterns that would be useful for alerting?

Log data:
- Total events: {total:,}
- Level counts: {_fmt_json(lc)}
- Top 30 message patterns (normalized, with count): {_fmt_json(top_patterns)}

Write in markdown. Quote specific pattern strings when discussing them."""


def _build_entity_analysis_prompt(summary: dict) -> str:
    entities = summary.get("entities", {})
    char = summary.get("characterization", {})
    return f"""You are writing the ENTITY ANALYSIS section of a formal log analysis report.

Write 400-600 words covering:

## IP Addresses & Network Entities
- What IP addresses/hostnames appear in the log?
- Which are most active? What role do they play?
- Any suspicious or unexpected IPs?

## File System Paths
- What file paths are referenced?
- What do they reveal about the system's file organization?
- Any paths that appear frequently in error contexts?

## Quoted Values & Identifiers
- What named entities (IDs, names, labels) appear frequently?
- What do the most common quoted values represent?
- Any values that appear in error contexts specifically?

## Security Observations
- Are there any authentication-related entities?
- Any access control or permission-related patterns?
- Sensitive data patterns (if any) in log messages?

Log data:
- Key entities from AI characterization: {char.get('key_entities', [])}
- Extracted IPs: {_fmt_json(entities.get('ip_addresses', {}))}
- Extracted file paths: {_fmt_json(entities.get('file_paths', {}))}
- Extracted quoted values: {_fmt_json(entities.get('quoted_values', {}))}

Write in markdown. Be specific about what each entity type means in the context of this system."""


def _build_pain_points_prompt(summary: dict, rag_context: str) -> str:
    bursts = summary.get("error_bursts", [])
    error_samples = summary.get("error_samples", [])
    top_loggers = summary.get("top_loggers", [])
    high_error = [l for l in top_loggers if l.get("error_rate", 0) > 0.05]
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    return f"""You are writing the most important section of this log analysis report: PAIN POINTS & WHERE TO LOOK.

Write 800-1000 words. This section is the actionable guide for engineers who need to debug this system.

For each pain point, provide ALL of this information:
1. **Pain Point Title** — clear name for the issue
2. **Severity** — Critical / High / Medium / Low
3. **Description** — what is happening, why it's a problem
4. **When it occurs** — exact timestamp ranges, hours of the day, frequency
5. **Where to look** — EXACT logger names to filter on
6. **What to search for** — exact keywords or patterns to search in the log
7. **Sample log lines** — quote 2-3 actual log lines that show this problem (with timestamps)
8. **Likely cause** — technical root cause hypothesis
9. **Impact** — what systems/users are affected

Identify AT LEAST 5 distinct pain points. Order them by severity (Critical first).

Format each as:
### Pain Point N: [Title]
**Severity:** [level]
**When:** [timestamp range or frequency]
**Logger(s):** `[logger names]`
**Search for:** `[keywords]`
...etc

Log data:
- Total events: {total:,} | Errors: {lc.get('ERROR',0)+lc.get('CRITICAL',0):,}
- High error-rate loggers: {_fmt_json(high_error)}
- Error bursts: {_fmt_json(bursts)}
- Error samples: {_fmt_json(error_samples)}

Relevant log excerpts from semantic search:
{rag_context}

Be EXTREMELY specific — engineers will use this section as a debugging guide. Include exact timestamps, logger names, and quoted log messages."""


def _build_recommendations_prompt(summary: dict, rag_context: str) -> str:
    char = summary.get("characterization", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", 0)
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)
    bursts = summary.get("error_bursts", [])
    top_loggers = summary.get("top_loggers", [])[:15]
    return f"""You are writing the RECOMMENDATIONS section of a formal log analysis report.

Write 600-800 words. Provide concrete, prioritized recommendations for the engineering team.

## Immediate Actions (Do within 24 hours)
- Fix or mitigate critical issues found in the logs
- Each item must reference the specific log evidence that justifies it

## Short-term Improvements (1-2 weeks)
- Address the underlying root causes of recurring errors
- Performance and reliability improvements
- Monitoring/alerting improvements based on identified pain points

## Long-term Architectural Improvements
- Systemic changes to improve reliability
- Logging hygiene improvements (what to add/remove)
- Architecture changes suggested by the error patterns

## Logging Improvements
- What additional logging would help diagnose future issues?
- What log levels should be changed for specific loggers?
- Any unnecessary verbosity that creates noise?

## Monitoring & Alerting
- What specific thresholds should be set for alerts?
- Which metrics matter most based on this log?
- What dashboards should be created?

For each recommendation, cite the specific evidence from the log that justifies it.

Log data:
- System: {char.get('log_type')} — {char.get('system_description', '')}
- Error rate: {errors/max(total,1)*100:.1f}% ({errors:,} of {total:,})
- Error bursts: {_fmt_json(bursts)}
- Logger error rates: {_fmt_json(sorted(top_loggers, key=lambda x: -x.get('error_rate',0))[:10])}

Relevant log excerpts:
{rag_context}

Write in markdown. Use numbered lists for priorities. Be specific and actionable."""


# ── RAG helper ─────────────────────────────────────────────────────────────────

def _rag_search(state: "AppState", query: str, top_k: int = 6) -> str:
    if not state.rag_store:
        return "(RAG index not available)"
    results = state.rag_store.query(query, top_k=top_k)
    if not results:
        return "(No relevant excerpts found)"
    parts = []
    for r in results:
        parts.append(
            f"[{r['chunk_type'].upper()} | {r['ts_start']} → {r['ts_end']}]\n{r['text'][:600]}"
        )
    return "\n\n---\n\n".join(parts)


# ── LLM call helper ────────────────────────────────────────────────────────────

async def _call_llm(client: Any, model: str, prompt: str, max_tokens: int = 2000) -> str:
    """Non-streaming LLM call. Returns full text."""
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return resp.choices[0].message.content or ""


# ── Main generator ─────────────────────────────────────────────────────────────

async def generate_report_sections(
    state: "AppState",
    client: Any,
    model: str,
) -> AsyncIterator[dict]:
    """
    Yields SSE-ready dicts for each report section.
    Sections are generated sequentially — one LLM call each.
    """
    summary = state.summary or {}
    char = summary.get("characterization", {})
    dr = summary.get("date_range", {})
    lc = summary.get("level_counts", {})
    total = summary.get("total_events", len(state.events))
    errors = lc.get("ERROR", 0) + lc.get("CRITICAL", 0)

    yield {"type": "start", "total_sections": len(SECTIONS)}

    # ── 1. Cover (static) ──────────────────────────────────────────────────────
    yield {"type": "section_start", "section": "cover", "index": 0}
    cover_content = {
        "filename": state.log_filename,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "log_type": char.get("log_type", "Unknown"),
        "format": summary.get("format", "Unknown"),
        "time_range_first": dr.get("first", "N/A"),
        "time_range_last": dr.get("last", "N/A"),
        "span_hours": dr.get("span_hours", 0),
        "total_events": total,
        "error_count": errors,
        "error_rate_pct": round(errors / max(total, 1) * 100, 2),
        "system_description": char.get("system_description", ""),
    }
    yield {"type": "section_content", "section": "cover", "content": json.dumps(cover_content)}
    yield {"type": "section_done", "section": "cover"}

    # ── 2. Table of Contents (static) ─────────────────────────────────────────
    yield {"type": "section_start", "section": "table_of_contents", "index": 1}
    toc_entries = [
        {"num": i+1, "title": SECTION_TITLES[s], "section": s}
        for i, s in enumerate(SECTIONS[2:], start=1)  # skip cover and TOC itself
    ]
    yield {"type": "section_content", "section": "table_of_contents", "content": json.dumps(toc_entries)}
    yield {"type": "section_done", "section": "table_of_contents"}

    # ── 3. Abstract (LLM) ─────────────────────────────────────────────────────
    yield {"type": "section_start", "section": "abstract", "index": 2}
    try:
        text = await _call_llm(client, model, _build_abstract_prompt(summary), max_tokens=600)
        yield {"type": "section_content", "section": "abstract", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "abstract", "content": f"*Error generating abstract: {e}*"}
    yield {"type": "section_done", "section": "abstract"}

    # ── 4. Executive Summary (LLM) ────────────────────────────────────────────
    yield {"type": "section_start", "section": "executive_summary", "index": 3}
    try:
        text = await _call_llm(client, model, _build_executive_summary_prompt(summary), max_tokens=1500)
        yield {"type": "section_content", "section": "executive_summary", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "executive_summary", "content": f"*Error generating executive summary: {e}*"}
    yield {"type": "section_done", "section": "executive_summary"}

    # ── 5. System Overview (LLM) ──────────────────────────────────────────────
    yield {"type": "section_start", "section": "system_overview", "index": 4}
    try:
        text = await _call_llm(client, model, _build_system_overview_prompt(summary), max_tokens=1200)
        yield {"type": "section_content", "section": "system_overview", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "system_overview", "content": f"*Error generating system overview: {e}*"}
    yield {"type": "section_done", "section": "system_overview"}

    # ── 6. Statistical Overview (static — chart data) ─────────────────────────
    yield {"type": "section_start", "section": "statistical_overview", "index": 5}
    stat_content = {
        "level_counts": lc,
        "total_events": total,
        "events_per_hour": summary.get("events_per_hour", {}),
        "errors_per_hour": summary.get("errors_per_hour", {}),
        "top_loggers": summary.get("top_loggers", [])[:20],
        "error_bursts": summary.get("error_bursts", []),
        "date_range": dr,
    }
    yield {"type": "section_content", "section": "statistical_overview", "content": json.dumps(stat_content)}
    yield {"type": "section_done", "section": "statistical_overview"}

    # ── 7. Error Analysis (LLM + RAG) ─────────────────────────────────────────
    yield {"type": "section_start", "section": "error_analysis", "index": 6}
    try:
        rag_ctx = _rag_search(state, "error critical failure exception traceback", top_k=8)
        text = await _call_llm(client, model, _build_error_analysis_prompt(summary, rag_ctx), max_tokens=2000)
        yield {"type": "section_content", "section": "error_analysis", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "error_analysis", "content": f"*Error generating error analysis: {e}*"}
    yield {"type": "section_done", "section": "error_analysis"}

    # ── 8. Error Bursts (LLM + RAG) ───────────────────────────────────────────
    yield {"type": "section_start", "section": "error_bursts", "index": 7}
    try:
        rag_ctx = _rag_search(state, "error burst spike surge multiple failures", top_k=8)
        text = await _call_llm(client, model, _build_error_bursts_prompt(summary, rag_ctx), max_tokens=1800)
        yield {"type": "section_content", "section": "error_bursts", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "error_bursts", "content": f"*Error generating burst analysis: {e}*"}
    yield {"type": "section_done", "section": "error_bursts"}

    # ── 9. Performance & Throughput (LLM + RAG) ───────────────────────────────
    yield {"type": "section_start", "section": "performance_throughput", "index": 8}
    try:
        rag_ctx = _rag_search(state, "slow timeout latency throughput performance degradation", top_k=6)
        text = await _call_llm(client, model, _build_performance_prompt(summary, rag_ctx), max_tokens=1500)
        yield {"type": "section_content", "section": "performance_throughput", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "performance_throughput", "content": f"*Error generating performance analysis: {e}*"}
    yield {"type": "section_done", "section": "performance_throughput"}

    # ── 10. Component Health (LLM + RAG) ──────────────────────────────────────
    yield {"type": "section_start", "section": "component_health", "index": 9}
    try:
        rag_ctx = _rag_search(state, "component service module initialization startup failure", top_k=8)
        text = await _call_llm(client, model, _build_component_health_prompt(summary, rag_ctx), max_tokens=2000)
        yield {"type": "section_content", "section": "component_health", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "component_health", "content": f"*Error generating component health: {e}*"}
    yield {"type": "section_done", "section": "component_health"}

    # ── 11. Pattern Analysis (LLM) ────────────────────────────────────────────
    yield {"type": "section_start", "section": "pattern_analysis", "index": 10}
    try:
        text = await _call_llm(client, model, _build_pattern_analysis_prompt(summary), max_tokens=1500)
        yield {"type": "section_content", "section": "pattern_analysis", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "pattern_analysis", "content": f"*Error generating pattern analysis: {e}*"}
    yield {"type": "section_done", "section": "pattern_analysis"}

    # ── 12. Entity Analysis (LLM) ─────────────────────────────────────────────
    yield {"type": "section_start", "section": "entity_analysis", "index": 11}
    try:
        text = await _call_llm(client, model, _build_entity_analysis_prompt(summary), max_tokens=1200)
        yield {"type": "section_content", "section": "entity_analysis", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "entity_analysis", "content": f"*Error generating entity analysis: {e}*"}
    yield {"type": "section_done", "section": "entity_analysis"}

    # ── 13. Pain Points (LLM + RAG, most important) ───────────────────────────
    yield {"type": "section_start", "section": "pain_points", "index": 12}
    try:
        rag_ctx = _rag_search(state, "error critical warning failure problem issue", top_k=10)
        text = await _call_llm(client, model, _build_pain_points_prompt(summary, rag_ctx), max_tokens=2500)
        yield {"type": "section_content", "section": "pain_points", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "pain_points", "content": f"*Error generating pain points: {e}*"}
    yield {"type": "section_done", "section": "pain_points"}

    # ── 14. Recommendations (LLM + RAG) ──────────────────────────────────────
    yield {"type": "section_start", "section": "recommendations", "index": 13}
    try:
        rag_ctx = _rag_search(state, "retry reconnect recovery workaround fix mitigation", top_k=6)
        text = await _call_llm(client, model, _build_recommendations_prompt(summary, rag_ctx), max_tokens=1800)
        yield {"type": "section_content", "section": "recommendations", "content": text}
    except Exception as e:
        yield {"type": "section_content", "section": "recommendations", "content": f"*Error generating recommendations: {e}*"}
    yield {"type": "section_done", "section": "recommendations"}

    # ── 15. Appendix (static) ─────────────────────────────────────────────────
    yield {"type": "section_start", "section": "appendix", "index": 14}
    appendix_content = {
        "error_samples": summary.get("error_samples", []),
        "error_bursts": summary.get("error_bursts", []),
        "top_patterns": summary.get("top_patterns", [])[:30],
        "all_loggers": summary.get("top_loggers", []),
        "entities": summary.get("entities", {}),
    }
    yield {"type": "section_content", "section": "appendix", "content": json.dumps(appendix_content)}
    yield {"type": "section_done", "section": "appendix"}

    yield {"type": "complete"}
