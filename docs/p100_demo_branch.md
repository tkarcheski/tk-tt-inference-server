# P100 Demo Branch — Full Write-Up

**Branch:** `tk/claude-p100-demo-chat-dashboard` (fork: `tkarcheski/tk-tt-inference-server`)
**Status:** Experimental / personal demo branch — not officially supported
**Base:** upstream `main` as of 2025-12-24 (`87da99f9`, rc-v0.6.0 era; ~1,100 commits behind current upstream)
**Last updated:** 2026-06-10

## What this branch is

End-to-end experimental support for running **GPT-OSS-20b on a Tenstorrent P100
(Blackhole)** card, with a chat dashboard on top and a LoRA fine-tuning pipeline
alongside it. Roughly 4,000 added lines across five components:

### 1. GPT-OSS model support (`workflows/model_spec.py`, `evals/eval_config.py`)
- Model specs for `openai/gpt-oss-20b` and `openai/gpt-oss-120b` (first pass).
- P100 device configuration for the 20b: `max_concurrency=16`,
  `max_context=32768`, `trace_region_size=30000000`, `VLLM_USE_V1=1`,
  `default_impl=False` (experimental).
- Chat-endpoint usage and eval config entries.

### 2. P100 helper scripts (`scripts/p100/`)
- `gpt-oss-20b-server.sh` — start the inference server in Docker (auto-setup,
  Docker image override supported).
- `gpt-oss-20b-benchmark.sh` — start server, benchmark, tear down.
- `gpt-oss-20b-chat.sh` — quick chat smoke test.
- `start-dashboard.sh` / `test-dashboard.sh` — launch / verify the dashboard.
- `README.md` — prerequisites, usage, troubleshooting.

### 3. Chat dashboard (`frontend/`)
Dash (Plotly) app with tabs for chat (streaming), server control,
hardware status (`tt-smi`), env config, prompt library, and logs.
- Direct-control path: `utils/server_manager.py` manages the Docker container;
  `utils/api_client.py` (`VLLMClient`) streams chat completions with JWT auth.
- A second, FastAPI-based backend (`api_server.py` + `app_api_client.py`)
  exists but is **not launched** by `start-dashboard.sh` — see Algorithm §2.
- pytest suite under `frontend/tests/`.

### 4. LoRA fine-tuning pipeline (`finetune/`)
- `build_dataset.py` — direct extraction of trustworthy Q→A pairs from
  [robotframework-chat](https://github.com/nutinspace/robotframework-chat)
  scenario YAMLs into OpenAI chat-format JSONL (train/val/all). Behavior-only
  suites (safety, refusals, sycophancy, …) are intentionally skipped.
- `train_lora.py` — LoRA SFT of a Qwen instruct model (never Llama) on the
  `nutinspace/robotframework-chat-skills` HF dataset. Env-var configured,
  runs on CPU for smoke tests or GPU for real runs; optional fp16 merge.
- Generated datasets (`out/`) and the training venv are gitignored — the
  JSONL files contain eval answer keys and must not be published.

### 5. Eval additions (`evals/eval_config.py`)
GPT-OSS eval wiring on top of the upstream eval framework.

## Current focus

**Fine-tuning ⇄ robotframework-chat integration.** The near-term goal is a
closed loop: extract dataset from rfc scenarios → LoRA-tune → run rfc eval
suites against the tuned model (served from this stack) → compare against the
base model. Longer-term (per the P100 fine-tune project): move training itself
onto Blackhole via tt-train, targeting gpt-oss-20b/Qwen bases.

---

## Elon's Algorithm pass (2026-06-10)

> 1. Make requirements less dumb. 2. Delete the part or process. 3. Simplify or
> optimize. 4. Accelerate cycle time. 5. Automate. — in that order.

### 1. Question the requirements
- **Does this branch need gpt-oss-120b specs?** A single P100 cannot serve
  120b. It rode in with the gpt-oss foundation commits; it is upstream's
  problem, not this demo's. → candidate to drop from this branch's scope.
- **Does the dashboard need its own FastAPI backend?** The requirement was
  "control the server and chat from a browser" — Dash already does both
  in-process. A separate REST backend was speculative generality.
- **Does fine-tuning need to run here at all yet?** Yes — but only the
  dataset-builder truly depends on rfc; the trainer is generic. Keep both,
  but don't grow them prematurely.

### 2. Delete the part or process
- **Delete `frontend/api_server.py` + `frontend/app_api_client.py`** (~350
  lines): `start-dashboard.sh` never launches the FastAPI backend; `app.py`
  already drives `ServerManager`/`VLLMClient` directly. Two half-wired
  architectures is one too many. (If a headless REST API is wanted later,
  re-add it as the *only* path.)
- **Squash WIP history**: `3590db13 wip set env vars` and
  `148b5a84 ... 1st pass` fold naturally into `59186fc8` (P100 support).
- **`test-dashboard.sh` vs `frontend/tests/`**: keep one entry point (the
  shell script can just run pytest).

### 3. Simplify / optimize
- One API client module, one config path: env handling exists in
  `components/env_config.py`, `utils/env_loader.py`, and the shell scripts —
  converge on a single `.env` contract.
- `build_dataset.py`: parameterize the hardcoded `--rfc-root` default
  (`/home/tyler/AI/robotframework-chat`) via env var for portability.
- `train_lora.py`: `ds.get("validation")` silently yields no eval set if the
  HF split is named `val` — resolve the split name explicitly.

### 4. Accelerate cycle time
- Fine-tune smoke test: document the `MAX_STEPS=5` + small-base-model CPU
  path as the standard pre-flight before any long run.
- Dashboard: Dash hot-reload in dev mode; keep the Docker image override
  (already done in `e065e21a`) so server restarts skip rebuilds.
- Dataset rebuilds are seconds — wire `build_dataset.py` output freshness
  check into the training script so stale data can't be trained on silently.

### 5. Automate
- CI (fork-level GitHub Actions): run `frontend/tests/` + ruff on push.
- One-command loop: `finetune/run_loop.sh` = build dataset → push to HF →
  train → serve adapter → run rfc eval suite → emit comparison report.
  This is the rfc-integration milestone in executable form.

---

## Known gaps / honest caveats
- P100 support is experimental; `default_impl=False` everywhere on purpose.
- trl is pinned to 0.12.2 → full-sequence SFT only (no assistant-only loss
  masking until trl ≥ 0.13).
- Branch base is ~6 months behind upstream `main`; a catch-up rebase will be
  a large effort (1,100+ commits) and should be scheduled deliberately.
- See `docs/p100_demo_upstreaming.md` for the keep-vs-upstream component
  discussion.
