#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a chat-format fine-tuning dataset by DIRECT EXTRACTION from the
robotframework-chat scenario YAMLs.

Only scenarios with a concrete, trustworthy target answer are emitted:
  - multiple choice  -> expected_letter (+ rationale)
  - exact answer     -> expected_answer
  - free-form answer -> expected

Behavior-only suites (safety, refusal_calibration, sycophancy, bias, format,
creativity[expected_traits], docker[program stdout], ...) are intentionally
skipped: they store prompts + a desired *behavior*, not an ideal response.

Output (OpenAI chat fine-tuning format), one JSON object per line:
  {"messages": [ {system}, {user}, {assistant} ], "suite": "..."}
"""
import argparse
import glob
import hashlib
import json
import os
import re

import yaml

SYSTEM_PROMPT = "You are a precise, helpful assistant. Answer correctly and concisely."

# Fields that may supply context that must precede the question.
CONTEXT_FIELDS = ("text", "passage", "document", "context", "code")
# Fields that may hold the user prompt.
PROMPT_FIELDS = ("question", "prompt", "benign_prompt")


def norm(value) -> str:
    """Collapse whitespace in a YAML scalar (handles folded '>' blocks)."""
    return re.sub(r"\s+", " ", str(value)).strip()


def _rationale(item: dict):
    return item.get("rationale") or item.get("explanation")


def resolve_target(item: dict) -> str | None:
    """Return the ideal assistant answer, or None if no trustworthy target."""
    if item.get("expected_letter"):
        letter = norm(item["expected_letter"])
        ans = f"The correct answer is {letter}."
        if _rationale(item):
            ans += "\n\n" + norm(_rationale(item))
        return ans
    for key in ("expected_answer", "expected", "answer"):
        if item.get(key) not in (None, ""):
            return norm(item[key])
    return None


def special_pair(item: dict):
    """Templated extraction for structured suites that don't expose a plain
    question/answer pair. Returns (user, target) or None. All content comes
    directly from stored fields; only a short instruction wrapper is added."""
    # Logical-fallacy identification (causal_reasoning/fallacy_scenarios)
    if item.get("argument") and item.get("fallacy_name"):
        user = "Identify the logical fallacy in this argument and explain why:\n\n" + norm(item["argument"])
        target = norm(item["fallacy_name"])
        if item.get("explanation"):
            target += "\n\n" + norm(item["explanation"])
        return user, target
    # Correlation vs causation (causal_reasoning/correlation_scenarios)
    if item.get("verdict") and (item.get("scenario") or item.get("claim")):
        parts = [norm(item[f]) for f in ("scenario", "claim") if item.get(f)]
        user = "\n\n".join(parts) + "\n\nDoes this establish causation, or only correlation? Explain."
        target = norm(item["verdict"])
        if item.get("rationale"):
            target += "\n\n" + norm(item["rationale"])
        return user, target
    # Key-value extraction (extraction/key_value_scenarios)
    if item.get("attribute") and item.get("text") and item.get("expected"):
        user = f"From the following text, extract the {norm(item['attribute'])}:\n\n" + norm(item["text"])
        return user, norm(item["expected"])
    return None


def resolve_prompt(item: dict) -> str | None:
    """Compose the user message: optional context block + the question."""
    context = [norm(item[f]) for f in CONTEXT_FIELDS if item.get(f)]
    question = next((norm(item[f]) for f in PROMPT_FIELDS if item.get(f)), None)
    if not question:
        return None
    # Adversarial suite: fold the injection into the prompt so the model
    # learns to answer the real question and ignore the hidden instruction.
    if item.get("hidden_instruction"):
        question = f"{question} {norm(item['hidden_instruction'])}"
    return ("\n\n".join(context + [question])) if context else question


def iter_scenarios(rfc_root: str):
    pattern = os.path.join(rfc_root, "robot", "**", "variables", "*.yaml")
    for path in sorted(glob.glob(pattern, recursive=True)):
        suite = os.path.relpath(path, os.path.join(rfc_root, "robot")).split(os.sep)[0]
        try:
            data = yaml.safe_load(open(path))
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield suite, item


def build(rfc_root: str, train_suites=None):
    records, seen, per_suite = [], set(), {}
    for suite, item in iter_scenarios(rfc_root):
        if train_suites is not None and suite not in train_suites:
            continue
        special = special_pair(item)
        if special:
            user, target = special
        else:
            user = resolve_prompt(item)
            target = resolve_target(item)
        if not user or not target:
            continue
        key = (user, target)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ],
            "suite": suite,
        })
        per_suite[suite] = per_suite.get(suite, 0) + 1
    return records, per_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rfc-root",
        default=os.environ.get(
            "RFC_ROOT", os.path.expanduser("~/AI/robotframework-chat")
        ),
        help="Path to a robotframework-chat checkout (env: RFC_ROOT)",
    )
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--split", help="split.json; restrict to its train suites")
    args = ap.parse_args()

    train_suites = None
    if args.split:
        train_suites = set(json.load(open(args.split))["splits"]["train"])
    records, per_suite = build(args.rfc_root, train_suites)   # build() skips non-train suites
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "train.jsonl")
    with open(path, "w") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"train_pool_hash={digest} rows={len(records)} suites={sorted(per_suite)}")


if __name__ == "__main__":
    main()
