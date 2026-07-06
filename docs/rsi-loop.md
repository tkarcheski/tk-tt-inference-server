# The RSI MODEL_TUNER Loop

A closed, self-improving loop that LoRA-fine-tunes a small Qwen on
robotframework-chat (rfc) suites, serves the tuned model alongside an untuned
baseline through Ollama, evaluates both on a frozen held-out set, and runs a
statistical promotion gate — **proposing** improvements but never promoting them
on its own. It is **shadow-only and human-gated** by construction.

Code lives in [`finetune/`](../finetune); the 24/7 service tooling lives in
[`scripts/rsi/`](../scripts/rsi). This document is the big-picture map; the
per-module docstrings are the detail.

---

## One round, end to end

```
 split_suites ──► build_dataset ──► train_lora ──► (merge) ──► serve_ollama
  firewall +       train-pool        LoRA SFT       fp16       GGUF → ollama
  leak check       extraction        (CPU, Qwen)    merge      create tag
      │                                                             │
      ▼                                                             ▼
  split.json                                          rsi-qwen:round (tuned arm)
                                                      qwen2.5:3b     (base arm)
                                                              │
                             ┌────────────────────────────────┘
                             ▼
   eval_rfc ──► import_results ──► gate (McNemar) ──► propose?  (SHADOW: never promotes)
   run rfc      output.xml →       holdout + canary   draft-PR only, triple-gated
   suites       rsi.test_results   pass/fail
```

`run_loop.run_round()` orchestrates exactly this. `run_loop.run_forever()`
repeats it back-to-back as the 24/7 supervisor.

---

## Modules (`finetune/`)

| Module | Responsibility |
|---|---|
| `rsi_common.py` | Shared config: `DSN`/`connect()` (Postgres warehouse), frozen `SALT`, `GOLD_SUITES`, `token_len()`. |
| `split_suites.py` | Deterministic **train/holdout/canary firewall** + leakage check. Writes `split.json`. |
| `build_dataset.py` | **Extraction-first** dataset build from rfc scenarios, filtered to the *train* pool. Prints `train_pool_hash`. |
| `train_lora.py` | LoRA SFT of a Qwen instruct model (CPU-friendly). Env-driven knobs incl. `SEED`, `MAX_STEPS`, `LORA_R`. Optional fp16 `merge`. |
| `dataset_loader.py` | Load a local `.jsonl` or a HF-hub dataset id (decoupled so it's testable without torch). |
| `serve_ollama.py` | Merged model dir → GGUF → `ollama create` (server-side import). `adapter_hash()` for provenance. |
| `eval_rfc.py` | Run one rfc suite against a model "arm" via `uv run robot`; returns the `output.xml` path. |
| `import_results.py` | Parse `output.xml` → per-test rows in `rsi.test_results`. |
| `gate.py` | Paired **McNemar** promotion gate over matched test outcomes. |
| `run_loop.py` | The round orchestrator + `--forever` supervisor + per-round hyperparameter variation. |

---

## The leakage firewall (why this loop is honest)

The load-bearing risk in any "train on X, test on X" loop is contamination.
`split_suites.py` prevents it:

- Each `GOLD_SUITES` suite is assigned to a pool by a **frozen** hash:
  `bucket = sha256(SALT + suite) % 10`, with `SALT = "rfc-split-v816"`. Buckets
  `0–6 → train`, `7–8 → holdout`, `9 → canary`. The split is therefore stable
  across runs and independent of any training outcome.
- A **leakage check** fingerprints every emitted `(prompt, answer)` pair as
  `sha256(user + "\x00" + target)` and refuses (raises `LeakageError`) if any
  *training* fingerprint also appears in the holdout or canary pool (i.e.
  train↔eval contamination), or if any pool is empty. (It guards the train↔eval
  boundary specifically; a holdout/canary-only overlap is not flagged.)
- `build_dataset.py` requires `--split` and trains **only** on the train pool.

The current split (deterministic): **train** = adversarial, c_interview,
causal_reasoning, code_review, quantization · **holdout** = context_window,
extraction, temporal_reasoning · **canary** = hallucination, legal.

---

## The two arms and the Ollama model name

Every round compares two models on the *same* Ollama endpoint (a paired A/B, so
serving conditions cancel out):

| Ollama tag | What it is | Role |
|---|---|---|
| `qwen2.5:3b` (`RSI_BASE_TAG`) | Stock untuned Qwen2.5-3B from the Ollama library | **base arm** (control) |
| `rsi-qwen:round` | This round's LoRA-tuned model, merged → GGUF → imported | **tuned arm** |

`rsi-qwen` is our chosen model name and `:round` its tag (Ollama tags are
`name:tag`, like Docker). Each round, `serve_ollama.create_tag(merged,
"rsi-qwen:round")` converts the merged fp16 weights to a GGUF (llama.cpp
`convert_hf_to_gguf.py --outtype f16`) and runs `ollama create rsi-qwen:round -f
Modelfile` so the **daemon imports it server-side** — the only path that works on
this host (a direct `--experimental` safetensors import writes to the caller's
store, which the daemon doesn't serve; `-q` quantization needs Apple-Silicon MLX).

**Important:** `rsi-qwen:round` is a **fixed tag, overwritten every round.** There
is only ever one, holding the *latest* round's tuned model. Which round/adapter it
came from is *not* in the tag — that provenance lives in the warehouse
(`lora_adapter_hash`, `seed`, `train_pool_hash`). See [Known limitations](#known-limitations).

---

## The gate (`gate.py`)

A paired **McNemar** test over matched per-test outcomes (same suite+test on both
arms). For discordant pairs — `b` = base-right/tuned-wrong, `c` = tuned-right/base-wrong —
it computes a one-sided exact binomial `binomtest(c, b + c, 0.5, alternative="greater")`.

A pool **passes** iff **both**: `delta_pp >= 5.0` (tuned pass-rate at least 5
points above base) **and** `p_value < 0.05`. The gate is evaluated on **both**
pools; a round is `proposed` only when **holdout AND canary both pass**. The
canary is the strict promotion guard (it catches memorization that inflates
holdout); holdout is the corroborating read. A `degenerate` flag fires if either
pool has zero matched pairs (a broken harness can otherwise masquerade as "no
improvement").

---

## Shadow-only safety

The loop **never** swaps a live serving endpoint and **never** auto-merges:

- `run_round` only ever records results and, at most, opens a **draft PR**.
- `can_promote()` returns True only when the loop is flipped out of shadow mode
  (`RSI_MODE=live` **and** `RSI_AGENTS_ENABLED != "false"`), and **nothing acts on
  that** beyond the check.
- `maybe_open_pr()` is triple-gated: it opens a draft PR only if the round
  `proposed` **and** `not can_promote()` **and** `RSI_OPEN_PR == "1"`. The 24/7
  service ships `RSI_MODE=shadow` and leaves `RSI_OPEN_PR` unset, so unattended
  rounds write to the warehouse and journal but open **no** PRs.

A human reviews the warehouse and decides every promotion.

---

## Publishing (`publish.py`, opt-in)

Separate from serving-promotion (which the loop never does), the loop can
**publish** a winning model. `publish.maybe_publish(report, merged)` fires from
`run_round` only when the strict gate passed (`report["proposed"]`) **and**
`RSI_PUBLISH == "1"` (master switch, default off). On a pass it:

1. **git-submodule model registry** — copies the GGUF + a `Modelfile` + a
   provenance `card.md` into the LFS-backed submodule at `ollama-models/`
   (repo `tkarcheski/rsi-ollama-models`), path `rsi-qwen/v{seed}-{hash8}/`, and
   pushes a tagged commit (`rsi-qwen-v{seed}-{hash8}`). Consumers
   `git clone` + `git checkout <tag>` + `ollama create`.
2. **GitHub release** — `gh release create` on the **fork**
   (`tkarcheski/tk-tt-inference-server`, never upstream) as the human record:
   McNemar deltas, provenance, and the clone-and-create instructions.
3. **`rsi.publications`** — records the publication for audit and idempotency
   (`UNIQUE(tuned_id)` → a tuned model is published at most once).

Each step is isolated (one failing step is logged and never crashes the round or
blocks the others). Version `v{seed}-{adapter_hash[:8]}` never overwrites, so
every publish is immutably pinned in the registry by tag. Rollback:
`gh release delete` + delete the submodule tag/commit + `ollama rm`.

---

## Public status dashboard (`publish_status.py`, opt-in)

The loop's results live in the local warehouse + journal. `publish_status.py`
makes them **world-visible** on GitHub Pages so anyone can watch the loop:

**Live:** <https://tkarcheski.github.io/tk-tt-inference-server/>

It **recomputes** every round's gate outcome from the stored `test_results` using
the same statistic the live gate uses (`gate.mcnemar_from_pairs`), so the page is
authoritative and covers all history — not a separate log that can drift. It
renders three artifacts and commits them to the **`gh-pages`** branch of the
public fork (Pages-only; **never main**):

- **`data.json`** — full round history + summary (machine-readable).
- **`index.html`** — a self-contained dashboard (no external assets) that
  re-fetches `data.json` every 5 min, so an open tab tracks the loop live.
- **`README.md`** — a Markdown leaderboard that renders on GitHub directly.

Only aggregate, non-sensitive fields are published (seed, timestamps, short
hashes, per-pool pass rates + McNemar Δ/p, gate decision). No DB credentials,
tokens, paths, prompts, or grader rationales leave the box. An idle refresh
(same rounds, newer clock) is a **no-op** — the timestamp alone never produces a
commit — so a frequent refresher does not spam `gh-pages`.

Two ways to keep it fresh (choose one; both opt-in, default off):

1. **Per-round hook** — `publish_status.maybe_publish_status(report)` fires from
   `run_round` when `RSI_PUBLISH_STATUS == "1"`; the dashboard refreshes once per
   round (its natural cadence).
2. **Standalone / timer** — `python publish_status.py` always publishes; drive it
   from a `systemd --user` timer / cron for a tighter refresh independent of the
   round. Needs a checkout of the fork on `gh-pages` at `RSI_STATUS_DIR` whose
   `origin` is the fork.

Rollback: the branch is disposable — disable Pages (or delete `gh-pages`) to take
it down; no model or endpoint is ever touched.

---

## The 24/7 supervisor (`run_loop.py --forever` + `scripts/rsi/`)

`run_forever()` runs rounds continuously:

- **Continue-on-error** — a failing round is logged; the loop proceeds (one bad
  round can't take the loop down).
- **Kill switch** — `RSI_KILL=1` or a stop-file (`RSI_STOP_FILE`) halts the loop
  gracefully after the current round.
- **Per-round variation** — `round_config(idx)` cycles a small LoRA grid
  (`LORA_R`/`LORA_ALPHA`/`LR`/`MAX_STEPS`) and advances `SEED`, so each round is a
  distinct experiment (seed recorded in `rsi.experiments.seed`). `MAX_STEPS` is
  deliberately small (≈1–3 epochs on the ~136-example train pool); large step
  counts memorize the pool and hurt generalization.

It runs as a **systemd `--user` service** via `scripts/rsi/rsictl.sh`
(`install`/`start`/`pause`/`resume`/`stop`/`status`/`logs`). The unit bakes the
live `PATH` (so `ollama`/`uv` resolve), uses `Restart=on-failure` (a crash
restarts; a graceful stop-file exit stays down), and enables linger so it
survives logout/reboot. See [`scripts/rsi/README.md`](../scripts/rsi/README.md)
for operations.

---

## Warehouse (`rsi.*`, Postgres)

`rsi_common.connect()` → `RSI_DSN` (default
`postgresql://rfc:changeme@localhost:5434/rfc`). Bootstrap schema:
[`finetune/sql/rsi_bootstrap.sql`](../finetune/sql/rsi_bootstrap.sql).

- **`rsi.experiments`** — one row per arm per round: `experiment_id`, `intent`,
  `provider_variant`, `serving_runtime`, `base_model_sha`, `lora_adapter_hash`,
  `train_pool_hash`, `train_split_hash`, `leakage_score`, `endpoint_url`,
  `tt_container_digest`, `rfc_sha`, `seed`, `train_hardware`,
  `parent_experiment_id`, `created_at`. (The tuned arm points at its base arm via
  `parent_experiment_id`.)
- **`rsi.test_results`** — one row per test per arm:
  `PRIMARY KEY (experiment_id, suite_id, test_id, pool, repeat_idx)`, plus
  `status` (PASS/FAIL/SKIP text) and `grader_rationale`. `pool` is in the PK so a
  cross-pool suite/test-name collision can't silently drop a gate-feeding row.
- **`rsi.publications`** — one row per auto-published model: `version` (PK),
  `tuned_id` (`UNIQUE` → idempotent), `submodule_commit`, `release_url`,
  `holdout_delta_pp`, `canary_delta_pp`, `created_at`.

---

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `RSI_DSN` | `postgresql://rfc:changeme@localhost:5434/rfc` | Warehouse connection. |
| `RFC_ROOT` | `/home/tyler/AI/rfc/wt-rfc` | robotframework-chat checkout. |
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | HF base for the tuned arm (real rounds). |
| `RSI_BASE_TAG` | `qwen2.5:3b` | Ollama tag for the untuned base arm. |
| `RSI_SKIP_SUITES` | *(unset)* | Comma-list of eval suites to skip (latency); firewall split untouched. |
| `RSI_SMOKE` | `0` | `1` = tiny/fast rounds (0.5B, `RSI_SMOKE_STEPS`). |
| `RSI_SMOKE_STEPS` | `3` | Train steps in smoke mode. |
| `RSI_LOOP_SLEEP` | `300` | Seconds between rounds (`--forever`). |
| `RSI_MAX_ROUNDS` | `0` | `>0` caps total rounds (mostly for tests). |
| `RSI_STOP_FILE` | `finetune/.rsi-stop` | Kill-switch file. |
| `RSI_KILL` | `0` | `1` halts the loop. |
| `RSI_MODE` | `shadow` | `live` only flips the (inert) promote gate; keep `shadow`. |
| `RSI_AGENTS_ENABLED` | `true` | Second promote-gate condition; `false` forces `can_promote()` False even in `live`. |
| `RSI_OPEN_PR` | *(unset)* | `1` allows draft-PR proposals (still triple-gated). |
| `RSI_PUBLISH` | *(unset)* | `1` arms the publish pipeline (registry push + GitHub release on a passing gate). |
| `RSI_MODELS_SUBMODULE` | `ollama-models` | Path to the git-LFS model-registry submodule. |
| `RSI_PUBLISH_STATUS` | *(unset)* | `1` arms the public status dashboard (refresh `gh-pages` each round). |
| `RSI_STATUS_DIR` | `~/AI/rsi-status-pages` | Fork checkout on `gh-pages` that `publish_status.py` commits to. |
| `RSI_STATUS_BRANCH` | `gh-pages` | Pages branch the dashboard is pushed to (never main). |
| `RSI_REPEATS` | `1` | Eval repeats per suite. |
| `MAX_STEPS` / `SEED` / `LORA_R` / `LORA_ALPHA` / `LR` | grid-driven | Per-round `train_lora` knobs (set by `round_config`). |

---

## Running it

```bash
scripts/rsi/rsictl.sh install     # write systemd unit + enable linger
scripts/rsi/rsictl.sh start       # start now + on boot
scripts/rsi/rsictl.sh follow      # tail the journal
scripts/rsi/rsictl.sh pause       # graceful stop after the current round
```

A single round for testing (no service):

```bash
cd finetune
RFC_ROOT=... RSI_BASE_TAG=qwen2.5:3b RSI_SKIP_SUITES=context_window,legal \
  /home/tyler/.rsi-loop-venv/bin/python run_loop.py --once --smoke
```

Two venvs: the **glue** venv (`/home/tyler/.rsi-loop-venv`, loop + tests, no
torch) runs `run_loop.py`; the **training** venv
(`finetune/.venv-train`, python3.10 + torch/trl/peft) runs `train_lora.py` and
the GGUF converter. `run_loop.py` invokes each by absolute path, so the loop can
run from any checkout (venvs are git-ignored and live only in the main checkout).

---

## Known limitations

- **Fixed serving tag.** `rsi-qwen:round` is overwritten each round; a round is
  not independently re-servable from Ollama afterward. Per-round tags
  (`rsi-qwen:{seed}` / `:{adapter_hash[:8]}`) would fix this at a disk cost (~6 GB
  each).
- **Restart resets the round counter.** `run_forever` restarts at `idx = 0`, so
  the `round_config`/`SEED` sequence repeats after a service restart (experiments
  are still distinct DB rows, just not globally unique seeds). Persisting the
  counter (e.g. from the warehouse experiment count) is a natural improvement.
- **Eval is the cadence bottleneck.** On a CPU box, 3B inference across the suites
  dominates round time (hours), even after `MAX_STEPS` is kept small. A smaller
  base (e.g. `Qwen/Qwen2.5-1.5B-Instruct` + matching `RSI_BASE_TAG`) is the lever
  for many-rounds-per-day cadence.
- **Latency-bound suites are skipped.** Needle-in-haystack stress tests
  (`context_window`, and `legal`'s `test_needle_in_haystack`, 5-min timeouts) are
  excluded via `RSI_SKIP_SUITES` because they measure latency tolerance, not
  answer quality. `legal` mixes quality + stress tests with no distinguishing
  tag, so it's skipped whole; tagging the stress tests in rfc would let
  `robot --exclude` keep legal's quality signal in canary.
