# finetune/ — LoRA fine-tuning on robotframework-chat skills

Pipeline: extract Q→A pairs from robotframework-chat scenario YAMLs, LoRA-tune
a Qwen instruct model on them, then (eventually) evaluate the adapter against
the rfc suites. See `docs/p100_demo_branch.md` for the bigger picture.

**The full self-improving MODEL_TUNER loop — architecture, the McNemar gate,
shadow-only safety, and the 24/7 service — is documented in
[`docs/rsi-loop.md`](../docs/rsi-loop.md).**

## Fast loop

```bash
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -r requirements-train.txt

# 1. Build the dataset (seconds; reads $RFC_ROOT, default ~/AI/robotframework-chat)
python3 build_dataset.py

# 2. Smoke test (minutes, CPU, tiny base model) — always run before a real run
./smoke_test.sh

# 3. Real run (GPU node; install the CUDA torch build first)
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct EPOCHS=3 MERGE=1 python3 train_lora.py
```

All training knobs are env vars — see the docstring in `train_lora.py`.

## Notes

- `out/` (datasets, adapters) is gitignored: the JSONL files contain rfc eval
  answer keys and must not be published.
- Base model is Qwen by design — never Llama.
- trl is pinned to 0.12.2 → full-sequence SFT; upgrade to trl ≥ 0.13 for
  `assistant_only_loss=True`.

## MODEL_TUNER loop

Shadow-only fine-tune ⇄ eval loop: `split_suites.py` (frozen firewall) →
`build_dataset.py` (train-pool only) → `train_lora.py` (CPU LoRA) →
`serve_ollama.py` (Ollama tag) → `eval_rfc.py` (robot, both arms) →
`import_results.py` (→ Postgres) → `gate.py` (McNemar). See
`docs/CONTRACT.md` for the full observed contract (schema, split salt,
measured baseline, deviations from plan).

### Prereqs

- Glue venv `/home/tyler/.rsi-loop-venv` (psycopg2-binary, scipy, lxml,
  pyyaml, tiktoken, datasets, pytest — runs everything except training).
- Training venv `finetune/.venv-train` (Python 3.10 + torch/datasets/peft/
  transformers/trl).
- Postgres `rfc` reachable on `localhost:5434` (container
  `rfc-monorepo-postgres-1`), schema `rsi` (apply `sql/rsi_bootstrap.sql`).
- Ollama reachable on `localhost:11434`.
- An rfc checkout at `/home/tyler/AI/rfc/wt-rfc` with a `.env`.

### Regenerate the frozen split

```bash
cd finetune && PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/python split_suites.py /home/tyler/AI/rfc/wt-rfc split.json
```

Raises `LeakageError` (and refuses to write) if any holdout/canary answer
fingerprint would land in the train pool, or if any pool is empty.

### Run a shadow smoke round

```bash
RSI_MODE=shadow MAX_STEPS=5 BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  PYTHONPATH=finetune /home/tyler/.rsi-loop-venv/bin/python finetune/run_loop.py --once --smoke
```

Note: under `--smoke`, `run_loop.py` overrides the training base to
`Qwen/Qwen2.5-0.5B-Instruct`, so `BASE_MODEL` in the smoke command is inert —
it only takes effect on non-smoke rounds.

### Env knobs

| Var | Effect |
|---|---|
| `RSI_MODE` | `shadow` (default) or `live` — only `live` (+ `RSI_AGENTS_ENABLED != "false"`) allows `can_promote()` to return `True`. |
| `RSI_OPEN_PR=1` | Opens a draft PR when a round proposes promotion. |
| `RSI_KILL=1` | Aborts the round before any work happens. |
| `RSI_BASE_TAG` | Base Ollama tag to compare against (default `qwen2.5:3b`). |
| `RSI_REPEATS` | Repeats per suite/pool during eval (default `1`). |
| `BASE_MODEL` | HF base model id for training (default `Qwen/Qwen2.5-3B-Instruct`). |
| `MAX_STEPS` | Cap training steps (useful for smoke rounds). |

Promotion is **human-gated and shadow-only** — `run_loop.py` never swaps the
`:8011` serving endpoint and never merges a PR; it can only open a draft PR
proposal. The `split_suites.py` firewall is what prevents train/test
contamination — every round re-derives the split and fails loudly
(`LeakageError`) rather than silently training on eval answers.
