#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the log-triage fine-tuning dataset for the logwatch model.

Sources, in priority order:
  1. Human corrections from the metrics submodule (logwatch/metrics/
     corrections.jsonl) — gold labels from rejected issues, oversampled.
  2. Weak labels from the logwatch heuristic engine over real logs in
     workflow_logs/ — bootstrap supervision until a teacher-model pass
     replaces it (see --teacher note below).

Output: OpenAI chat-format JSONL (same contract as build_dataset.py /
train_lora.py), one verdict per chunk.

Teacher upgrade path: re-label the same chunks with a frontier model and feed
that file in via --extra; corrections still win on conflict.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logwatch.chunk import iter_chunks  # noqa: E402
from logwatch.schema import SYSTEM_PROMPT  # noqa: E402
from logwatch.triage import _heuristic_verdict  # noqa: E402

CORRECTION_OVERSAMPLE = 5  # human corrections are rare and precious


def weak_label_records(log_dirs: list[Path]) -> list[dict]:
    records = []
    for log_dir in log_dirs:
        for path in sorted(log_dir.rglob("*.log")):
            for _, text in iter_chunks(path):
                label, _, evidence = _heuristic_verdict(text)
                records.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": json.dumps({
                            "label": label,
                            "reason": "weak label (heuristic-v0)",
                            "evidence": evidence,
                        })},
                    ],
                    "source": "heuristic",
                })
    return records


def correction_records(corrections_path: Path) -> list[dict]:
    if not corrections_path.exists():
        return []
    records = []
    with open(corrections_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" in row:
                records.append({"messages": row["messages"],
                                "source": "correction"})
    return records


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="*",
                    default=[str(repo_root / "workflow_logs")])
    ap.add_argument("--corrections",
                    default=str(repo_root / "logwatch" / "metrics"
                                / "corrections.jsonl"))
    ap.add_argument("--extra", help="extra labeled JSONL (e.g. teacher pass)")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__),
                                         "out", "logwatch"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    weak = weak_label_records([Path(d) for d in args.log_dirs])
    corrections = correction_records(Path(args.corrections))
    extra = []
    if args.extra:
        with open(args.extra) as fh:
            extra = [json.loads(line) for line in fh if line.strip()]

    records = weak + extra + corrections * CORRECTION_OVERSAMPLE
    random.seed(args.seed)
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_frac)) if records else 0
    val, train = records[:n_val], records[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"weak={len(weak)} teacher={len(extra)} "
          f"corrections={len(corrections)} (x{CORRECTION_OVERSAMPLE})")
    print(f"train={len(train)} val={len(val)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
