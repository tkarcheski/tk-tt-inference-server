# logwatch — small-LLM log triage with a human feedback loop

A harness around a small (0.5B) model that reviews logs and files tracker
issues, designed so the model's context stays tiny: the harness chunks and
aggregates in plain Python (map/reduce); the model only ever judges one
~80-line window per stateless call.

```
logs ──chunk──► model verdict per chunk ──reduce──► findings ──► GitHub/GitLab issues
                                                                      │
   retrain ◄── corrections.jsonl ◄── rejected (not planned)  ◄────────┤ human review
      ▲                                                               │
      └────── grades.jsonl / metrics.json ◄── addressed (completed) ◄─┘
```

## Usage

```bash
# Sweep logs with the bootstrap heuristic engine (no model needed)
python3 -m logwatch triage workflow_logs

# Same, with the served fine-tuned model (vLLM, guided JSON decoding)
LOGWATCH_MODEL_URL=http://localhost:8000 LOGWATCH_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
python3 -m logwatch triage workflow_logs --engine llm

# File one issue per new actionable finding (deduped by fingerprint)
GITHUB_TOKEN=... LOGWATCH_GITHUB_REPO=owner/repo \
python3 -m logwatch file-issues workflow_logs --tracker github
# GitLab: GITLAB_TOKEN=... LOGWATCH_GITLAB_PROJECT=group/project --tracker gitlab

# Fold closed-issue outcomes into training data + metrics (run on a schedule)
python3 -m logwatch feedback --tracker github
```

Integration surface is deliberately minimal: one token + one repo/project env
var per tracker, plain REST, no SDKs, no webhooks (feedback polls).

## The human review contract

Each issue carries a `logwatch-metadata` block. Reviewers communicate back by
how they close the issue:

| Reviewer action | Meaning | Effect |
|---|---|---|
| Close as **completed** | finding was real | grade recorded → `metrics.json` precision |
| Close as **not planned** (or label `logwatch-rejected`) | false positive | correction example emitted for retraining |
| Comment `correction: <label>` before closing rejected | "it's real but mislabeled" | correction uses that label |

## Training loop

```bash
python3 finetune/build_log_dataset.py   # weak labels + oversampled corrections
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct DATASET=finetune/out/logwatch \
  python3 finetune/train_lora.py        # (load_dataset accepts a local dir)
```

Initial supervision is the heuristic engine (`heuristic-v0`) as weak labeler;
upgrade by re-labeling chunks with a frontier model and passing `--extra`.
Human corrections from the metrics submodule always win and are oversampled.

`logwatch/metrics` is a git submodule (private repo `tkarcheski/logwatch-metrics`)
so the model's track record — corrections, grades, per-version precision —
is versioned independently. After running `feedback`, commit and push inside
the submodule.

## Labels

`benign`, `noise`, `known_failure`, `new_error`, `hardware_fault`,
`resource_exhaustion` — only the last three file issues. See `schema.py`.
