# SPDX-License-Identifier: Apache-2.0
"""Verdict schema, prompts, and fingerprinting shared by the whole pipeline."""

import hashlib
import json
import re

LABELS = [
    "benign",  # normal operation, progress lines
    "noise",  # warnings that never matter (known-noisy subsystems)
    "known_failure",  # matches a documented failure signature
    "new_error",  # real error that matches nothing known
    "hardware_fault",  # device-level: hangs, resets, link errors, tt-smi faults
    "resource_exhaustion",  # OOM, L1/trace-region exhaustion, disk full, timeout
]

# Labels that produce a tracker issue.
ACTIONABLE = {"new_error", "hardware_fault", "resource_exhaustion"}

SYSTEM_PROMPT = (
    "You review one chunk of an inference-server log. Reply with one JSON "
    'object only: {"label": <one of ' + json.dumps(LABELS) + '>, '
    '"reason": <short phrase>, "evidence": <the single most important line, '
    "verbatim>}. Unsure -> benign."
)

# JSON schema for vLLM guided/structured decoding (guided_json).
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "reason": {"type": "string", "maxLength": 200},
        "evidence": {"type": "string", "maxLength": 400},
    },
    "required": ["label", "reason", "evidence"],
}

_NORMALIZERS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]?\d*"), "<ts>"),
    (re.compile(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}"), "<ts>"),
    (re.compile(r"/[\w./-]{8,}"), "<path>"),
    (re.compile(r"\b\d+\b"), "<n>"),
]


def normalize_line(line: str) -> str:
    """Strip volatile tokens so the same failure always fingerprints alike."""
    for pattern, repl in _NORMALIZERS:
        line = pattern.sub(repl, line)
    return line.strip()


def fingerprint(label: str, evidence: str) -> str:
    """Stable identity of a finding, used to dedupe across runs and issues."""
    return hashlib.sha1(
        f"{label}|{normalize_line(evidence)}".encode(), usedforsecurity=False
    ).hexdigest()[:16]


def chunk_sha(text: str) -> str:
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()[:16]
