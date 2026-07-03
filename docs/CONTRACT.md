# MODEL_TUNER loop — observed contract

This document captures the **observed reality** of the fine-tune ⇄ eval
("MODEL_TUNER") loop under `finetune/` as it actually exists on this branch —
not the aspirational design. Every fact below is verified against the code
and schema in this repo (or cited to the doc that measured it). If a claim
here and the code disagree, the code wins and this file is stale.

## Warehouse

- Postgres database `rfc` on `localhost:5434`, served by container
  `rfc-monorepo-postgres-1`.
- Schema `rsi`, role `rfc` (`finetune/rsi_common.py` default DSN:
  `postgresql://rfc:changeme@localhost:5434/rfc`).
- Tables (`finetune/sql/rsi_bootstrap.sql`):
  - `rsi.experiments` — one row per evaluated arm (base or tuned): intent,
    provider_variant, serving_runtime, base_model_sha, lora_adapter_hash,
    train_pool_hash, leakage_score, train_hardware, parent_experiment_id, etc.
  - `rsi.test_results` — one row per (experiment, suite, test, pool, repeat).
    **`PRIMARY KEY (experiment_id, suite_id, test_id, pool, repeat_idx)`** —
    `pool` is part of the key specifically so a train/holdout/canary row for
    the same test id can never collide with or silently overwrite another
    pool's row for that same experiment.

## Provider variants

- **`vllm-tt`** — the P100 Llama-3.1-8B server on `:8011`. This is a
  **reference line only**: the loop's code (`run_loop.py`, `eval_rfc.py`)
  never invokes it. It exists in the schema/design for provenance comparison,
  not as an active arm.
- **`ollama`** (`:11434`) — the actual paired A/B control used by the loop:
  base `qwen2.5:3b` vs tuned `rsi-qwen:round`, both evaluated on the **same**
  Ollama endpoint (`finetune/eval_rfc.py::arm_env`, `finetune/run_loop.py`).
  Comparing arms only within one endpoint avoids confounding the gate with
  cross-hardware/cross-runtime differences.

## Measured P100 baseline (reference arm only)

Source: `docs/p100_capacity_decision.md` (Llama-3.1-8B, base model, 64K-context
config, driven directly — not via `run.py`, which is broken on this branch):

| Concurrency | Aggregate tok/s | Per-user tok/s |
|---|---|---|
| 1  | 29.2 | 29.2 |
| 16 | 361.3 | 22.6 |

Summarized as "~29 tok/s/user single-user, 361 tok/s aggregate @16
concurrency, warm TTFT ~90ms." These numbers describe the `vllm-tt` reference
line only; they are not produced by, and do not gate, the MODEL_TUNER loop
itself.

## P100 hard constraints (bound the `vllm-tt` reference arm only)

- Prompts must be **<2048 tokens**: at the measured 64K-context config, a
  ≥2048-token prefill crashes the server (L1 circular-buffer clash,
  `program.cpp:916`). This constraint applies only to the vLLM/Llama
  reference arm — see "Known deviations" below for its current
  (non-)implementation status in this loop's code.
- **gpt-oss-20b cannot run on one P100.** The MoE expert matmul needs 13
  core-blocks; a single P100 exposes a 12-core grid
  (`TT_FATAL … num_blocks_total <= num_cores_available`, "13 blocks > 12
  cores"). This is a compute-placement limit, not a memory limit — see
  `docs/p100_capacity_decision.md` and `CLAUDE.md`.

## Gold-answer suites and frozen split

10 gold-answer suites participate in the loop (`finetune/rsi_common.py::GOLD_SUITES`):
`adversarial, c_interview, causal_reasoning, code_review, context_window,
extraction, hallucination, legal, quantization, temporal_reasoning`.

The top-level `variables` suite is **excluded** from the gold set because it
covers the same underlying facts as the `hallucination` canary suite
(reworded, not identical). Excluding it is a conservative firewall measure:
training on `variables` could teach a fact that also appears (differently
worded) in the `hallucination` canary, a fact-level overlap the
`(user, target)` fingerprint check would NOT catch. Excluding it keeps the
canary an honest held-out measurement.

**Split mechanics** (`finetune/split_suites.py`):
- `SALT = "rfc-split-v816"`.
- `bucket(suite_id) = int(sha256(SALT + suite_id).hexdigest(), 16) % 10`.
- `train` = buckets 0–6, `holdout` = buckets 7–8, `canary` = bucket 9.
- Concrete assignment (as committed in `finetune/split.json`):
  - `train` = {adversarial, c_interview, causal_reasoning, code_review,
    quantization} — 136 answer fingerprints.
  - `holdout` = {context_window, extraction, temporal_reasoning} — 29
    fingerprints.
  - `canary` = {hallucination, legal} — 20 fingerprints.
- A scenario's **fingerprint** is `sha256(user + "\x00" + target)`, computed
  over the exact `(user, target)` pair that `build_dataset.build()` emits —
  i.e. the fingerprint is derived from the same extraction path used for
  training, not a re-derivation, so "same fingerprint" means "same training
  example."
- **Leakage is fatal:** `split_suites.write_split` raises `LeakageError` if
  any holdout/canary fingerprint also appears in the train pool, or if any of
  the three pools comes out empty (which would otherwise silently produce a
  vacuous/always-passing split).

## Dataset build

`build_dataset.py --split <split.json>` (the `--split` flag is **required**)
emits `train.jsonl` containing **only** records from the split's `train`
pool. This fixed a contamination bug in an earlier version of the script that
randomly split **all** graded scenarios (train + holdout + canary) into
train/val — i.e. it was training on the eval suite's own answers.

## Training

- CPU LoRA fine-tune of a Qwen instruct base. The MODEL_TUNER round
  (`run_loop.py`) defaults to a `Qwen/Qwen2.5-3B-Instruct` base (env
  `BASE_MODEL`) for fast CPU-only LoRA cadence. The underlying trainer
  `train_lora.py` is base-size-agnostic — invoked standalone it defaults to
  `Qwen/Qwen2.5-7B-Instruct`; it only requires a non-Llama (Qwen) instruct
  model.
- `MERGE=1` produces a merged fp16 model, which `serve_ollama.py` turns into
  an Ollama tag via `ollama create --experimental -q q4_K_M`.
- Environment split: training runs under `finetune/.venv-train` (Python
  3.10 + torch/datasets/peft/transformers/trl). Every other loop module
  (`split_suites`, `build_dataset`, `eval_rfc`, `import_results`, `gate`,
  `run_loop`, and the test suite) runs under the glue venv,
  `/home/tyler/.rsi-loop-venv` (psycopg2-binary, scipy, lxml, pyyaml,
  tiktoken, datasets, pytest — no torch).

## Gate

`finetune/gate.py::mcnemar_from_pairs` runs an exact McNemar test — SciPy
`binomtest(c, b + c, 0.5, alternative="greater")` on paired per-test PASS/FAIL
outcomes (`b` = tuned broke a test the base passed, `c` = tuned fixed a test
the base failed). A round **passes** iff `delta_pp >= 5` (percentage-point
pass-rate improvement) **and** `p_value < 0.05`. **Canary is the promotion
metric**; **holdout is the sanity read** — `run_loop.py` requires both to
pass before it will even consider proposing (`report["proposed"] =
canary["passes"] and holdout["passes"]`).

**`RSI_REPEATS > 1` caveat:** each repeat of a test is treated as an
independent McNemar pair, but repeats of the same test are correlated, not
independent — so `RSI_REPEATS > 1` inflates `n` and understates `p_value`
versus what the exact test assumes. Default is `RSI_REPEATS=1`.

## Shadow-only / promotion

- `run_loop.py` **never** swaps the `:8011` serving endpoint and **never**
  merges a PR, under any condition in this code path.
- It opens a **draft** PR proposal only when all three hold:
  `report["proposed"]` (both gates passed) **and** `not can_promote()`
  **and** `RSI_OPEN_PR == "1"`.
- `can_promote()` returns `True` only when `RSI_MODE == "live"` **and**
  `RSI_AGENTS_ENABLED != "false"` — i.e. promotion requires an explicit,
  human-set mode change; nothing in the loop code flips this on its own.
- `RSI_KILL=1` aborts a round (`SystemExit`) before any work (split, train,
  serve, eval) happens.

## Known deviations from the original plan (documented honestly)

- **(a) `load_local_or_hub` was decoupled into `finetune/dataset_loader.py`**
  rather than living inside `train_lora.py`, so the dataset-loading logic is
  importable and unit-testable without pulling in `torch` (which only exists
  in the training venv).
- **(b) `pool` was added to `rsi.test_results`'s primary key**
  (`(experiment_id, suite_id, test_id, pool, repeat_idx)`), beyond what an
  earlier design implied, specifically to prevent a train/holdout/canary row
  for the same test id from silently overwriting (via `ON CONFLICT DO
  NOTHING`/upsert semantics) a different pool's row for that same experiment.
- **(c) The `<2048`-token prompt filter is NOT yet implemented** in
  `eval_rfc.run_suite` — it currently just shells out to `robot` against a
  suite directory with no token-length pre-filtering. This is a known gap,
  not a silent regression: the filter only matters for the `vllm-tt`
  reference arm (the 64K P100 config's prefill crash boundary), and the loop
  as built never invokes that arm — the Ollama-vs-Ollama paired control has
  no such constraint. If the `vllm-tt` arm is ever wired into the loop for
  real, this filter must be added first.
