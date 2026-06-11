# SPDX-License-Identifier: Apache-2.0
"""Feedback loop: closed issues -> corrections (training data) and grades.

Outcome semantics (see trackers.py for the issue conventions):
  rejected  -> the model was wrong. Emit a chat-format correction example
               (the original chunk -> the corrected verdict) ready to merge
               into the next fine-tune. Human can supply the right label via
               a `correction: <label>` comment; default is "benign".
  addressed -> the model was right. Emit a grade record; aggregate precision
               per model version goes to metrics.json.

All outputs land in the metrics submodule (logwatch/metrics) so the model's
track record is versioned independently of this repo.
"""

import json
import time
from pathlib import Path

from .chunk import iter_chunks
from .schema import LABELS, SYSTEM_PROMPT, chunk_sha

METRICS_DIR = Path(__file__).parent / "metrics"


def _find_chunk(log_path: str, sha: str) -> str | None:
    path = Path(log_path)
    if not path.exists():
        return None
    for _, text in iter_chunks(path):
        if chunk_sha(text) == sha:
            return text
    return None


def _load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as fh:
        return {json.loads(line)["ref"] for line in fh if line.strip()}


def process_closed_issues(closed: list[dict], metrics_dir: Path = METRICS_DIR) -> dict:
    """Idempotently fold closed issues into corrections/grades/metrics."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    corrections_path = metrics_dir / "corrections.jsonl"
    grades_path = metrics_dir / "grades.jsonl"
    seen = _load_seen(corrections_path) | _load_seen(grades_path)

    new_corrections, new_grades = 0, 0
    for issue in closed:
        if issue["ref"] in seen:
            continue
        meta = issue["metadata"]
        record = {
            "ref": issue["ref"],
            "fingerprint": meta["fingerprint"],
            "model_version": meta.get("model_version", "unknown"),
            "predicted_label": meta["label"],
            "outcome": issue["outcome"],
            "closed_at": issue.get("closed_at"),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if issue["outcome"] == "rejected":
            corrected = issue.get("correction") or "benign"
            if corrected not in LABELS:
                corrected = "benign"
            chunk = _find_chunk(meta.get("log_path", ""), meta["chunk_sha"])
            record["corrected_label"] = corrected
            record["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk or meta["evidence"]},
                {"role": "assistant", "content": json.dumps({
                    "label": corrected,
                    "reason": "human-corrected via issue review",
                    "evidence": meta["evidence"] if corrected != "benign" else "",
                })},
            ]
            with open(corrections_path, "a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            new_corrections += 1
        else:
            with open(grades_path, "a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            new_grades += 1

    summary = _summarize(metrics_dir)
    with open(metrics_dir / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    summary["new_corrections"] = new_corrections
    summary["new_grades"] = new_grades
    return summary


def _summarize(metrics_dir: Path) -> dict:
    """Per-model-version precision: addressed / (addressed + rejected)."""
    by_version: dict[str, dict] = {}
    for name, key in (("grades.jsonl", "addressed"),
                      ("corrections.jsonl", "rejected")):
        path = metrics_dir / name
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                version = json.loads(line).get("model_version", "unknown")
                stats = by_version.setdefault(
                    version, {"addressed": 0, "rejected": 0})
                stats[key] += 1
    for stats in by_version.values():
        total = stats["addressed"] + stats["rejected"]
        stats["precision"] = round(stats["addressed"] / total, 3) if total else None
    return {"by_model_version": by_version}
