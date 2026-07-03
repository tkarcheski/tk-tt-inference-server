# CLAUDE.md — tk-tt-inference-server (P100 demo fork)

Guidance for Claude Code working in this repository.

## What this repo is

Tenstorrent `tt-inference-server`: orchestration (`run.py`) that launches vLLM +
tt-metal inference in Docker containers on Tenstorrent accelerators, plus
benchmarks, evals, and a chat dashboard (`frontend/`). This is a **personal demo
fork** on branch `tk/claude-p100-demo-chat-dashboard`, adding experimental support
for a single **Blackhole P100** card (see `docs/p100_demo_branch.md`,
`scripts/p100/`).

Model specs are the source of truth for what runs where. Note the repo is
**mid-migration** from YAML specs (`workflows/model_specs/{dev,prod}/llm.yaml`) to
Python specs (`workflows/model_spec.py`, `spec_templates` → `MODEL_SPECS`). At the
current branch tip `MODEL_SPECS` is built from the Python list, **not** the YAML.
The migration is incomplete — see the caveat below.

## Objectives

### Primary: host/system memory alongside the Tenstorrent card
**Goal:** use host (system) RAM together with the Blackhole card so larger models —
especially the 20B-parameter `gpt-oss-20b` MoE — can be served on a single P100,
via weight/KV offload or host-DRAM paging.

**Status: research, not a config toggle. Frame honestly before promising it:**
- tt-metal keeps model weights in the card's on-board GDDR6; there is no known
  first-class "spill to host RAM" switch in vLLM-tt / tt-metal today. Confirm what
  (if anything) the current tt-metal / vLLM-tt stack exposes before designing around
  it — do not assume it exists.
- **Host RAM would not fix the gpt-oss-20b failure on one P100.** That failure is a
  *compute-placement* limit, not a memory-capacity limit: the MoE expert matmul
  needs 13 core-blocks but a single P100 exposes 12
  (`TT_FATAL ... num_blocks_total <= num_cores_available`, "13 blocks > 12 cores").
  Adding system memory does not add cores, so the matmul still cannot be placed.
  Host-memory offload could help fit larger *dense* weights / longer KV cache, but
  not this MoE core-grid wall.
- Tie any progress here back to `docs/p100_capacity_decision.md` (the buy /
  more-cards / wait-for-P300 analysis).

### Secondary
- Get a real, measured P100 tokens/sec number (currently blocked — see below).
- Fine-tuning ⇄ robotframework-chat loop (`finetune/`), per `docs/p100_demo_branch.md`.

## Known caveat: the spec migration left run.py broken

The June-2026 "Python spec" migration (commit `d3ee2fc2d`) rewrote
`workflows/model_spec.py` but did **not** update `run.py` to the new API, and was
apparently never run afterward. At the branch tip `run.py` cannot start a workflow:
it calls `get_runtime_model_spec(model=, device=, impl=)` expecting a 3-tuple and
calls `model_spec.apply_overrides(...)`, but the migrated module provides
`get_runtime_model_spec(args)` → single spec with `apply_runtime_args`, and
`apply_overrides` was deleted. Partial fixes have been applied on this branch
(restored `export_model_specs_json`, `max_tokens_all_users`, and a
`tensor_cache_timeout` field), but finishing the port of `run.py` and its helpers
(`RuntimeConfig`, `populate_model_spec_cli_args`) to the new API remains **open
work**. Until then, `run.py` on this branch is not end-to-end runnable.

## Conventions

- Run `run.py` with the project venv: `~/.tenstorrent-venv/bin/python` (Python 3.10;
  the bare host `python3` is 3.14 and lacks deps). CLI uses `--tt-device` (e.g.
  `p100`) and hyphenated `--impl` values (e.g. `tt-transformers`); the older
  `--device` form in `scripts/p100/*.sh` is stale.
- Check the card with `tt-smi -ls` (single Blackhole `p100a` on this host).
- Docker images are tagged `{version}-{tt_metal_commit}-{vllm_commit}`; run.py
  refuses images below v0.11.0 (the entrypoint contract changed). Locally-pulled
  P100 images are older (`0.7.0-55fd115-aa4ae1e`), which is why live benchmarking is
  currently blocked on a version/image mismatch.
