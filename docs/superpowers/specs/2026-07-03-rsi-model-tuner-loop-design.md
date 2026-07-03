# MODEL_TUNER: a closed fine-tune ⇄ rfc self-improvement loop — design

**Date:** 2026-07-03
**Status:** Design (approved direction; pending spec review)
**Owner:** Tyler
**Home repo for the loop code:** `tk-tt-inference-server` (`finetune/`). **Roadmap home:** a new
first-class Phase-3 issue in `tkarcheski/RSI` (working name `MODEL_TUNER`).

## Context

Tyler asked for "a loop where we try to optimize the model, and run robotframework-chat
(rfc) tests to verify model performance" — an open-ended, 24/7 self-improving loop with an
LLM-as-optimizer that reads rfc failures and improves the model, verified by rfc pass rates.

This sits inside an existing, larger program — `tkarcheski/RSI` (53 issues, Phases 0–4). An
adversarial 5-lens review of that roadmap against today's reality produced three findings that
shape this design:

1. **The roadmap orphans this exact ask.** No RSI issue owns a *weight* fine-tune loop. TUNER
   (#44–46) proposes config/flag deltas; REGRESSION_CURATOR (#47–49) grows the eval set and
   explicitly does not fine-tune; rsi_selfsolve (#50–52) measures *agent* solve-rate, not model
   quality. The ask needs a new home and manifest vocabulary.
2. **The existing fine-tune path leaks (verified in code).** `finetune/build_dataset.py:98`
   globs the same `robot/**/variables/*.yaml` files rfc grades against, then `:156-159` does a
   `random.seed(42)` shuffle-split of *all* scenarios into train/val. Training targets are the
   graded answer keys; fine-tuning on this and evaluating on the same suites is training on the
   test set. **This is a correctness bug and the prerequisite fix.**
3. **Today's reality is smaller than the roadmap assumes.** One Tenstorrent P100 serving base
   `Llama-3.1-8B` via the direct container (`scripts/p100/direct_server_p100.sh`; `run.py` is
   broken), OpenAI-compatible on `:8011`, measured ~29 tok/s/user. No GPU — LoRA training is
   CPU-only. The tuned artifact is a LoRA-**Qwen** served via **Ollama** (rfc supports
   `LLM_PROVIDER=ollama|vllm`). rfc emits per-suite pass rates (`output.xml` carries per-test).

The goal of this document is a minimal, **honest** first loop runnable this week on that
machine, plus its place in the RSI roadmap.

## Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Optimizer engine | **Extraction-first.** v1 trains on real gold answers from the train pool only. LLM-synthesis of examples for failed suites is a later, separately-gated experiment requiring label-validation. |
| 2 | Base model | **Small Qwen (≤3B)** — `Qwen2.5-1.5B/3B-Instruct` — for fast CPU rounds. 7B reserved for periodic runs. |
| 3 | Roadmap placement | **New first-class `MODEL_TUNER` Phase-3 issue** in `tkarcheski/RSI`, bootstrapped as the operator-run side-track but with a real manifest slot + promotion gate. |
| 4 | Deliverable now | **Design spec → implementation plan.** No code or GitHub issues yet. |
| 5 | Result granularity | **Per-test rows** parsed from `output.xml` (nearly free; unblocks the gate and leakage checks). |
| 6 | Holdout policy | **Fixed, frozen** holdout for the solo phase. Quarterly rotation (#53 §22) deferred. |
| 7 | CONTRACT.md timing | **Evidence-first** — write `docs/CONTRACT.md` (RSI #24) *last*, from observed reality, logging each binding decision as the spike reveals it. |

## Non-goals (explicitly deferred)

Not built in v1 — they are unattended-multi-daemon controls that buy nothing until the loop
runs green once: MinIO artifact store (RSI #27; use local filesystem `artifact_uri` + inline
JSON <64KB), the LISTEN/NOTIFY event queue (#26; a single script needs no wake-loop), the
gateway *as a service* (#28; keep it a **lint rule** banning direct `subprocess`/`docker`/`gh`/
`psql` outside one module), the migration *runner* (#5; hand-apply one `.sql`), the Bayesian
sweep (#19), reverse RFC→TT triggers (#38), and the `$50/day` + `≤5 PR/day` ceilings. Also not
in v1: fine-tuning on the P100 (CPU/Ollama only), and any auto-promotion of a tuned adapter.

## Architecture — the MODEL_TUNER loop

One long-running script (`finetune/run_loop.py`, replacing the doc-only `run_loop.sh`) drives a
round; a thin systemd/`while` wrapper reruns it 24/7. A round is:

```
                ┌─────────────────────────────────────────────────────────────┐
                │  frozen references (never in the A/B control):               │
                │   • P100 base Llama-3.1-8B  @ :8011  (context line only)      │
                │   • Qwen-base (adapter-off) @ Ollama (the paired control)     │
                └─────────────────────────────────────────────────────────────┘
 rfc suites ──► [train/ | holdout/ | canary/]  (frozen, disjoint split; §Firewall)
     │                     │            │  └── never seen by optimizer or training
     │            build_dataset(train/ only) ─► train.jsonl  (+ train_pool_hash)
     │                     │
     │            smoke_test.sh (MAX_STEPS=5 preflight)
     │                     │
     │            train_lora.py (Qwen ≤3B, CPU) ─► adapter (lora_adapter_hash)
     │                     │
     │            export Modelfile ─► `ollama create qwen-tuned-vN`
     │                     ▼
     └──► rfc eval BOTH arms on holdout (k=3–5, temp=0, prompts <2048 tokens)
                     │   arm A: qwen-base     arm B: qwen-tuned-vN
                     ▼
          importer: output.xml ─► rsi.test_results (per-test rows)
                     ▼
          gate: McNemar paired test on matched per-test outcomes (holdout);
                effect ≥5pp, p<0.05; secondaries within Ollama lane only;
                canary suite (optimizer never saw) is the promotion metric
                     ▼
          leakage_score check ─► if 0 and gate passes:
                open DRAFT PR with adapter_hash + scores  ──► HUMAN promotes
          else: log rsi.experiments row status='proposed', continue
```

The LLM-as-optimizer (local `qwen3.6:35b` via Ollama) enters at round planning: it reads the
previous round's per-test failures (wrapped in `<untrusted>`) and proposes the next change from
a **constrained menu** — which train-pool suites to weight up, LoRA hyperparams (r/alpha/LR/
epochs), and system-prompt edits. In v1 it never invents training targets (Decision 1).

### Components (each independently testable)

- **`finetune/split_suites.py`** (new) — deterministic suite-level partition of rfc `robot/*`
  into `train/`, `holdout/`, `canary/` by hashing suite IDs. Emits a frozen `split.json` +
  `holdout_answer_hashes.txt`. The safety primitive that has no owning issue today.
- **`finetune/build_dataset.py`** (fix) — read **only** `train/` suites (accept a `--split`
  file); drop the random val-split of graded scenarios; emit `train_pool_hash`. Keep the
  existing extraction logic and behavior-only-suite exclusions.
- **`finetune/train_lora.py`** (reuse) — Qwen ≤3B base, CPU, LoRA; already correct. Add nothing
  but a `--dataset-hash` passthrough for provenance.
- **`finetune/serve_ollama.py`** (new, thin) — merge/export the adapter to an Ollama Modelfile
  and `ollama create qwen-tuned-vN`; return the Ollama tag + `lora_adapter_hash`.
- **`finetune/eval_rfc.py`** (new, thin) — run selected rfc `make robot-<suite>` against a given
  endpoint/profile, k repeats at temp=0, enforcing a **<2048-token prompt pre-flight filter**
  (≥2048 crashes the P100 64K config). Returns the `output.xml` paths.
- **`finetune/import_results.py`** (new) — parse `output.xml` → per-test rows → `rsi.test_results`
  keyed by `experiment_id`. (The rfc-eval importer of RSI #34/#35, per-test only.)
- **`finetune/gate.py`** (new) — McNemar paired test on holdout + canary; secondaries within the
  Ollama lane; returns accept/reject + a report.
- **`finetune/run_loop.py`** (new) — orchestrates the round; writes `rsi.experiments`; on pass,
  opens a **draft** PR (never auto-serves).

### Data flow / warehouse (minimal, hand-applied)

One hand-applied `rsi_bootstrap.sql` (skip the #5 migration runner), extending RSI #25's
manifest with the fields the frozen schema lacks:

```sql
CREATE SCHEMA IF NOT EXISTS rsi;
CREATE TABLE rsi.experiments (
  experiment_id        uuid PRIMARY KEY,
  intent               text,                 -- 'baseline' | 'model_tuner_round'
  provider_variant     text NOT NULL,        -- 'vllm-tt' | 'ollama' | 'vllm'
  serving_runtime      text NOT NULL,
  base_model_sha       text NOT NULL,
  lora_adapter_hash    text,                 -- NULL for base/reference
  train_pool_hash      text,                 -- provenance of the training data
  train_split_hash     text,                 -- the frozen split.json hash
  leakage_score        numeric,              -- 0 required to promote
  endpoint_url         text,
  tt_container_digest  text,                 -- baseline reproducibility
  rfc_sha              text,
  seed                 int,
  train_hardware       text,                 -- 'cpu' | 'cuda' | 'p100'
  parent_experiment_id uuid,
  created_at           timestamptz DEFAULT now()
);
CREATE TABLE rsi.test_results (
  experiment_id   uuid REFERENCES rsi.experiments(experiment_id),
  suite_id        text NOT NULL,
  test_id         text NOT NULL,            -- per-test, from output.xml
  pool            text NOT NULL,            -- 'train' | 'holdout' | 'canary'
  status          text NOT NULL,           -- 'PASS' | 'FAIL'
  grader_rationale text,
  repeat_idx      int,
  PRIMARY KEY (experiment_id, suite_id, test_id, repeat_idx)
);
```

Postgres is owned by the rfc host; TT/loop writes via a scoped `rsi_writer` role (#53 §4).
`artifact_uri` points at a local path in v1; `output.xml`/adapter live on disk.

## Metric & promotion gate

- **Primary:** per-test correctness pass-rate delta on the **frozen holdout**, computed with
  **McNemar's paired test** over matched per-test outcomes (not Welch — the outcomes are binary,
  paired, and per-suite percentages are underpowered). Bar (spirit of #53 §20): paired,
  pre-registered, `p<0.05`, effect ≥ **5 percentage points**, k=3–5 repeats at temperature 0.
- **Promotion metric = a held-out canary suite the optimizer never sees.** Gains on mined
  suites are not evidence.
- **Paired control = Qwen-base (adapter-off) vs Qwen-tuned, same Ollama endpoint.** The P100
  Llama-8B is a **plotted reference line, never the A/B control** (different model + silicon).
- **Secondaries (TTFT-P95 / tok-s / cost) measured within the Ollama lane only** (Qwen_vN vs
  vN+1). The cross-hardware −5% gate against the P100 is invalid and dropped.
- **Baseline frozen:** fixed `seed`, `tt_container_digest` `0.7.0-55fd115-aa4ae1e`, `rfc_sha`,
  via `scripts/p100/direct_server_p100.sh` (reproducible since `run.py` is broken).

## Safety guardrails (day one, non-negotiable)

1. **Train/holdout/canary firewall + CI leakage check.** Split at suite level, deterministic,
   *before* extraction. `build_dataset.py` reads `train/` only. **CI fails if any holdout/canary
   suite ID or answer-hash appears in the emitted JSONL.** Log `leakage_score` (exact-answer /
   n-gram overlap between training targets and holdout expected outputs); nonzero blocks
   promotion.
2. **Shadow-only, human-gated promotion.** The loop proposes an adapter and opens a **draft PR**
   carrying `lora_adapter_hash`, `leakage_score`, canary delta, and intra-lane secondaries. It
   **never swaps the canonical serving endpoint.** A LoRA adapter and any system-prompt edit are
   in the human-approval-forever set (#53 §7; this is #16 shadow→live extended to a weight
   artifact — the roadmap never wrote that issue).
3. **Untrusted-input handling.** rfc transcripts/rationales feed an agent that writes training
   data — wrap in `<untrusted>` (#53 §9), treat proposals as data never executed, schema-validate
   JSONL before training (reject rows whose target didn't originate from a `train/` file).
4. **Kill-switch on from day one** (`RSI_AGENTS_ENABLED=false` → shadow in 60s) even while
   operator-run — cheap. Budgets/PR-ceilings deferred until unattended.
5. **Gateway as a lint rule** now (ban direct `subprocess`/`docker`/`gh`/`psql` outside one
   module); the gateway-as-service (#28) is deferred.

## Week-1 build slice (implementation order)

0. rfc `.env` from `.env.example` at `/home/tyler/AI/rfc/wt-rfc` — two profiles: `vllm` → `:8011`
   (Llama reference), `ollama` → tuned-Qwen tag. *(No owning issue — real day-1 blocker.)*
1. Day-1 spike: 2–3 rfc suites vs each arm; confirm both emit `output.xml`; enforce <2048-token
   pre-flight. *(RSI #1/#2 partial; Ollama arm = gap.)*
2. `split_suites.py` — frozen train/holdout/canary. *(No owner — the missing safety primitive.)*
3. Fix `build_dataset.py` leak (train-only, drop shuffle-split, hash). *(No owner — prerequisite.)*
4. Hand-apply `rsi_bootstrap.sql` (§warehouse). *(Extends #25; skip #5 runner.)*
5. `import_results.py` — per-test `output.xml` → `rsi.test_results`. *(rfc-eval importer of #34/#35.)*
6. `run_loop.py` — the round orchestrator. *(No owner — this is the ask; new MODEL_TUNER issue.)*
7. `gate.py` — McNemar on holdout + canary; draft-PR on pass. *(No owner — §Metric.)*
8. `docs/CONTRACT.md` last, from observed reality. *(RSI #24, inverted to last.)*

## Roadmap changes (Decision 4 = spec only; these are proposed, not yet created)

- **New:** `MODEL_TUNER` (Phase 3) — owns the weight fine-tune loop; children for the firewall,
  the leak fix, `run_loop.py`, and the McNemar gate.
- **New (gap issues):** rfc `.env` two-profile wiring; Ollama `provider_variant` + `lora_adapter_hash`
  manifest extension; per-test result granularity (tie to #24).
- **Amend #24 (CONTRACT.md):** invert to evidence-last; add the Ollama variant, per-test shape,
  the P100 direct-launch reality, the <2048 constraint, and the measured baseline.

## Verification (how we know it works)

- **Firewall:** a unit test asserts `train/ ∩ (holdout ∪ canary) = ∅` and that
  `build_dataset.py` output contains zero holdout answer-hashes; `leakage_score == 0`.
- **End-to-end dry run:** `run_loop.py --once --smoke` runs a MAX_STEPS=5 CPU adapter, serves it
  via Ollama, evals 2 suites × k=2 on holdout for both arms, imports rows, and prints a McNemar
  report — completing without touching the P100 serving endpoint.
- **Baseline reproducibility:** re-running the frozen baseline twice yields identical
  `rsi.experiments` provenance (same hashes) and pass rates within noise.
- **Honesty check:** deliberately point `build_dataset.py` at a holdout suite and confirm CI
  fails (leakage guard works).
- **No-regression on the P100:** the loop never restarts or reconfigures the `:8011` container;
  the baseline row's `tt_container_digest` is unchanged across rounds.

## Risks / open items

- CPU LoRA cadence on ≤3B is minutes-to-tens-of-minutes/round; if too slow, drop to 1.5B or
  fewer steps. Confirm the exact Qwen base `train_lora.py` + the Ollama tag point at.
- rfc suite flakiness may swamp a 5pp effect at k=3–5; may need more repeats or suite selection.
- Extraction-first means the loop can only improve on suites that *have* gold answers; behavior-
  only suites (safety/refusal/etc.) are out of scope until the gated synthesis experiment.
- The LLM-optimizer's constrained menu may plateau quickly; that is an acceptable v1 outcome —
  the point is an honest, leak-free loop that *can* show real holdout gains, not maximal gains.
