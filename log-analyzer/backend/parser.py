"""
Generic log parser — auto-detects format from a sample, then parses every line.
Supports: Python logging, Log4j, syslog, ISO8601, nginx/apache, raw fallback.
"""
import re
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Optional


class LogEvent(TypedDict):
    ts: str           # raw timestamp string
    ts_epoch: float   # unix epoch, 0.0 if unparseable
    logger: str
    level: str        # DEBUG | INFO | WARNING | ERROR | CRITICAL | RAW
    msg: str
    raw: str          # original line


class FormatProfile(TypedDict):
    name: str
    regex_str: str
    ts_format: Optional[str]
    ts_field: Optional[str]
    level_field: Optional[str]
    logger_field: Optional[str]
    msg_field: str


# ── Candidate formats (tried in order, most specific first) ────────────────────

_CANDIDATES: list[FormatProfile] = [
    {
        "name": "Python logging (comma-ms)",
        "regex_str": (
            r"^(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[,\.]\d+)"
            r"\s*[-–]\s*(?P<logger>\S+)"
            r"\s*[-–]\s*(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)"
            r"\s*[-–]\s*(?P<msg>.+)$"
        ),
        "ts_format": "%Y-%m-%d %H:%M:%S,%f",
        "ts_field": "ts", "level_field": "level",
        "logger_field": "logger", "msg_field": "msg",
    },
    {
        "name": "Log4j bracket",
        "regex_str": (
            r"^(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[,\.]?\d*)"
            r"\s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)"
            r"\s+\[(?P<logger>[^\]]+)\]"
            r"\s+[-–]?\s*(?P<msg>.+)$"
        ),
        "ts_format": "%Y-%m-%d %H:%M:%S,%f",
        "ts_field": "ts", "level_field": "level",
        "logger_field": "logger", "msg_field": "msg",
    },
    {
        "name": "ISO8601 with level",
        "regex_str": (
            r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
            r"\s+(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)"
            r"(?:\s+(?P<logger>\S+))?"
            r"\s+[-–:]?\s*(?P<msg>.+)$"
        ),
        "ts_format": "%Y-%m-%dT%H:%M:%S",
        "ts_field": "ts", "level_field": "level",
        "logger_field": "logger", "msg_field": "msg",
    },
    {
        "name": "Syslog RFC3164",
        "regex_str": (
            r"^(?P<ts>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
            r"\s+(?P<host>\S+)\s+(?P<logger>\S+?)(?:\[\d+\])?:\s+(?P<msg>.+)$"
        ),
        "ts_format": "%b %d %H:%M:%S",
        "ts_field": "ts", "level_field": None,
        "logger_field": "logger", "msg_field": "msg",
    },
    {
        "name": "Nginx/Apache access",
        "regex_str": (
            r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]'
            r'\s+"(?P<msg>[^"]+)"\s+(?P<level>\d{3})\s+(?P<bytes>\S+)'
        ),
        "ts_format": "%d/%b/%Y:%H:%M:%S %z",
        "ts_field": "ts", "level_field": "level",
        "logger_field": "host", "msg_field": "msg",
    },
    {
        "name": "Bracketed timestamp",
        "regex_str": (
            r"^\[(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[,\.]?\d*)\]"
            r"(?:\s*\[?(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\]?)?"
            r"(?:\s+(?P<logger>\S+))?"
            r"\s*[-:]\s*(?P<msg>.+)$"
        ),
        "ts_format": "%Y-%m-%d %H:%M:%S",
        "ts_field": "ts", "level_field": "level",
        "logger_field": "logger", "msg_field": "msg",
    },
    {
        "name": "Raw (no structure)",
        "regex_str": r"^(?P<msg>.+)$",
        "ts_format": None,
        "ts_field": None, "level_field": None,
        "logger_field": None, "msg_field": "msg",
    },
]

# Pre-compile regexes
_COMPILED = [(p, re.compile(p["regex_str"])) for p in _CANDIDATES]


# ── Timestamp parsing ─────────────────────────────────────────────────────────

_TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%b/%Y:%H:%M:%S",
    "%b %d %H:%M:%S",
    "%b  %d %H:%M:%S",
]


def _parse_epoch(ts_raw: str) -> float:
    """Parse a raw timestamp string to Unix epoch. Returns 0.0 on failure."""
    if not ts_raw:
        return 0.0
    # Normalise: strip timezone, replace T with space
    s = ts_raw.strip()
    s = re.sub(r'[Zz]$', '', s)
    s = re.sub(r'[+-]\d{2}:?\d{2}$', '', s)
    s = s.replace('T', ' ')
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s[:26], fmt[:26] if len(fmt) > 26 else fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def _epoch_to_iso(epoch: float) -> str:
    if epoch == 0.0:
        return ""
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(path: Path, sample_size: int = 300) -> FormatProfile:
    """
    Read up to sample_size non-empty lines, score each candidate format
    by match rate, return the best (or Raw fallback).
    """
    lines: list[str] = []
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.strip():
                lines.append(line)
            if len(lines) >= sample_size:
                break

    best_profile, best_score = _CANDIDATES[-1], 0
    for profile, rx in _COMPILED[:-1]:  # skip Raw fallback in scoring
        hits = sum(1 for l in lines if rx.match(l))
        score = hits / max(len(lines), 1)
        if score > best_score:
            best_score, best_profile = score, profile

    # Accept if > 20% match; otherwise fall back to Raw
    return best_profile if best_score > 0.20 else _CANDIDATES[-1]


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_log(path: Path, profile: FormatProfile) -> list[LogEvent]:
    """
    Parse entire file using the given FormatProfile.
    Lines that don't match are attached to the previous event's timestamp
    (continuation lines / stack traces).
    """
    rx = re.compile(profile["regex_str"])
    events: list[LogEvent] = []
    last_epoch = 0.0
    last_ts = ""

    level_map = {
        "WARN": "WARNING", "FATAL": "CRITICAL",
        "DEBUG": "DEBUG", "INFO": "INFO",
        "WARNING": "WARNING", "ERROR": "ERROR", "CRITICAL": "CRITICAL",
    }

    with open(path, "rb") as f:
        for raw_bytes in f:
            line = raw_bytes.decode("utf-8", errors="replace").rstrip()
            if not line.strip():
                continue

            m = rx.match(line)
            if not m:
                # Continuation / unparsed line
                events.append(LogEvent(
                    ts=last_ts, ts_epoch=last_epoch,
                    logger="__unparsed__", level="RAW",
                    msg=line[:500], raw=line,
                ))
                continue

            g = m.groupdict()
            ts_raw = g.get(profile["ts_field"] or "", "") if profile["ts_field"] else ""
            epoch = _parse_epoch(ts_raw)
            if epoch > 0:
                last_epoch, last_ts = epoch, ts_raw

            raw_level = g.get(profile["level_field"] or "", "INFO") if profile["level_field"] else "INFO"
            level = level_map.get((raw_level or "INFO").upper(), "INFO")

            events.append(LogEvent(
                ts=ts_raw or last_ts,
                ts_epoch=epoch or last_epoch,
                logger=(g.get(profile["logger_field"] or "") or "root").strip() if profile["logger_field"] else "root",
                level=level,
                msg=(g.get(profile["msg_field"]) or "").strip()[:600],
                raw=line,
            ))

    return events


def parse_log_file(path: Path) -> tuple[list[LogEvent], FormatProfile]:
    """Top-level entry: auto-detect format, then parse."""
    profile = detect_format(path)
    events = parse_log(path, profile)
    return events, profile
