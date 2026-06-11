# SPDX-License-Identifier: Apache-2.0
"""Map step: classify each chunk with a stateless model call.

Two engines:
  llm        — OpenAI-compatible /v1/chat/completions (the served 0.5B), with
               vLLM guided_json so verdicts cannot be malformed.
  heuristic  — regex rules. Bootstrap mode: lets the harness run end-to-end
               before a model is trained, and doubles as the weak labeler for
               the initial training dataset.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import requests

from .chunk import iter_chunks
from .schema import (
    ACTIONABLE,
    SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    chunk_sha,
    fingerprint,
)

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    label: str
    reason: str
    evidence: str
    log_path: str
    first_line: int
    chunk_sha: str
    fingerprint: str
    model_version: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# Ordered: first match wins. Conservative on purpose — heuristics only catch
# what they were written for; the model exists to generalize past them.
_RULES = [
    ("resource_exhaustion", re.compile(
        r"out of memory|OOM|cannot allocate|allocation failed|No space left"
        r"|trace region.*(?:exceed|full|too small)|L1 buffer.*(?:exceed|overflow)",
        re.I)),
    ("hardware_fault", re.compile(
        r"device (?:hang|reset|lost)|link (?:down|error)|tt-smi.*(?:fault|fail)"
        r"|ethernet training failed|ARC (?:fw|firmware) (?:hang|timeout)", re.I)),
    ("known_failure", re.compile(
        r"trace capture.*fail|skip.*system.sw.validation"
        r"|Connection refused|Connection reset by peer", re.I)),
    ("new_error", re.compile(
        r"Traceback \(most recent call last\)|CRITICAL|FATAL"
        r"|exited with (?:non-zero|code [1-9])|RuntimeError|AssertionError", re.I)),
    ("noise", re.compile(r"WARNING|deprecat", re.I)),
]

_EVIDENCE = re.compile(r"ERROR|CRITICAL|FATAL|Traceback|fail|hang|OOM", re.I)


def _heuristic_verdict(text: str) -> tuple[str, str, str]:
    for label, pattern in _RULES:
        match = pattern.search(text)
        if match:
            line = next(
                (ln for ln in text.splitlines() if match.group(0) in ln),
                match.group(0),
            )
            return label, f"matched rule: {pattern.pattern[:60]}", line.strip()
    return "benign", "no rule matched", ""


def _llm_verdict(text: str, base_url: str, model: str) -> tuple[str, str, str]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "max_tokens": 150,
            "temperature": 0,
            "guided_json": VERDICT_SCHEMA,
        },
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(raw)
        return data["label"], data.get("reason", ""), data.get("evidence", "")
    except (json.JSONDecodeError, KeyError):
        logger.warning("unparseable verdict, marking benign: %.80s", raw)
        return "benign", "unparseable model output", ""


def triage_paths(paths: list[Path], engine: str = "heuristic") -> Iterator[Verdict]:
    """Sweep log files and yield one Verdict per chunk."""
    base_url = os.environ.get("LOGWATCH_MODEL_URL", "http://localhost:8000")
    model = os.environ.get("LOGWATCH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    model_version = (
        os.environ.get("LOGWATCH_MODEL_VERSION", model)
        if engine == "llm"
        else "heuristic-v0"
    )

    for path in paths:
        for first_line, text in iter_chunks(path):
            if engine == "llm":
                label, reason, evidence = _llm_verdict(text, base_url, model)
            else:
                label, reason, evidence = _heuristic_verdict(text)
            yield Verdict(
                label=label,
                reason=reason,
                evidence=evidence,
                log_path=str(path),
                first_line=first_line,
                chunk_sha=chunk_sha(text),
                fingerprint=fingerprint(label, evidence) if evidence else "",
                model_version=model_version,
            )


def reduce_findings(verdicts: list[Verdict]) -> list[dict]:
    """Reduce step: dedupe actionable verdicts by fingerprint, count repeats."""
    findings: dict[str, dict] = {}
    for v in verdicts:
        if v.label not in ACTIONABLE or not v.fingerprint:
            continue
        entry = findings.setdefault(
            v.fingerprint,
            {**asdict(v), "count": 0, "locations": []},
        )
        entry["count"] += 1
        if len(entry["locations"]) < 10:
            entry["locations"].append(f"{v.log_path}:{v.first_line}")
    return sorted(findings.values(), key=lambda f: -f["count"])
