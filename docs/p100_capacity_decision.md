# P100: keep it, buy more cards, or wait for P300?

**Question (Tyler):** what tokens/sec can I get on the P100 — should I keep the one
card, buy more, or wait for the P300? Decide by seeing how the P100 performs.

**Short answer:** the decision does **not** hinge on a fresh throughput number,
because the more important signal is a *hard* one: the model this whole branch is
built around — `gpt-oss-20b` — **cannot run on a single P100 at all**, for a
structural reason no configuration or extra system memory can fix. What you should
buy depends entirely on whether your real workload is dense small models (P100 is
plausibly enough) or MoE / large / high-concurrency models (it is not — that needs
P300-class or multi-card hardware).

---

## 1. State of the evidence

There is **no measured P100 tokens/sec anywhere** in this repo, and none could be
produced this session. `workflow_logs/benchmarks_output/` is empty. Every recorded
attempt failed before serving a token:

| Attempt | Model | Outcome |
|---|---|---|
| 2025-12-30 | Llama-3.1-8B-Instruct | Server never became healthy — first-run tensor-cache generation for a **64K-context / 32-concurrency** config was still running at the 90-min timeout. Slow, not crashed. |
| 2026-02-12 | gpt-oss-20b | **Hard crash** during MoE warmup: `TT_FATAL … num_blocks_total <= num_cores_available` — "13 blocks > 12 cores". |

The tok/s figures that exist in the specs (Llama-8B P100: **3.3 / 16.5 / 33**
tok/s/user for functional / complete / target tiers; gpt-oss-20b: **2.6 / 13 / 26**)
are hardcoded *target bars*, not measurements.

**Why no live number this session:** the branch's own tooling is currently
un-runnable. The June-2026 spec migration broke `run.py` (see `CLAUDE.md`), and the
only P100 docker images available locally (`0.7.0-55fd115-aa4ae1e`) are refused by
every runnable `run.py` here as "pre-0.11" (a Docker entrypoint-contract change).
Getting a live number requires resolving that version/image mismatch or finishing
the migration — see §4.

## 2. The decisive fact: one P100 cannot place the gpt-oss-20b MoE

The gpt-oss-20b crash is not "too slow" or "out of memory" — it is a **compute
placement** limit. The mixture-of-experts expert matmul is sharded into 13 blocks,
but a single P100 exposes a 12-core grid, so the kernel cannot be laid out. This is
asserted in tt-metal itself (`matmul_op_multi_core_reuse_mcast_1d_program_factory`),
not in configurable server settings.

Consequences:
- **No config fixes it** on one card (concurrency, context, batch size, trace region
  are all irrelevant to the core count).
- **More system/host RAM does not fix it** — host memory adds capacity, not cores.
- **More P100s only help if the model is sharded across their combined cores** —
  i.e. a multi-card *mesh*, not just "two cards in a box."

## 3. What the repo's own hardware roadmap already tells you

The spec catalog encodes the vendor's intended hardware-per-model mapping. It is a
strong hint about where each workload is meant to live:

- **gpt-oss-120b** is specced **only** on `P300X2` and `GALAXY` — never a single
  small card. The bigger MoE models are *designed* for P300-class / multi-card.
- **gpt-oss-20b** is specced for `T3K` (8-chip), `GALAXY`, and — experimentally,
  by this fork — a single `P100` (which we now know fails).
- **Llama-3.1-8B** (dense) has a **`P150X4`** 4-card data-parallel spec
  (`data_parallel: 4`) for scale-out, alongside single-card N150/N300/P100 entries.
- **`P300` / `P300X2` already exist as first-class device types** in `run.py`
  (`--tt-device` choices) and in specs — the next-gen card is real and supported in
  tree, not vaporware.

Read together: single small cards serve **dense ≤ ~8B** models; MoE / 20B+ / long
context / high concurrency are routed to **P300-class or multi-card meshes**.

## 4. Recommendation

Pick the row that matches your actual workload:

**A. Your workload is dense models ≤ ~8B (Llama-3.1-8B, Qwen3-8B, etc.), modest
context and concurrency.**
→ **Keep the single P100 — but get the real number before spending anything.** The
dense path has no core-grid wall; the only prior failure was a first-run cache-gen
timeout, which the shrunk P100 spec on this branch (4K context, concurrency 1,
`tensor_cache_timeout=10800`) is designed to clear. Validate against the 33
tok/s/user single-user target. Do **not** buy more cards for this workload until a
measured number says the P100 is too slow for your latency/throughput needs.

**B. Your workload is gpt-oss-20b (or any MoE / 20B+ / long-context / high-
concurrency serving).**
→ **A single P100 is structurally insufficient — do not buy more P100s expecting to
run 20B on one.** The lever is cores+memory across a mesh. Two realistic paths:
- **Wait for / buy P300-class hardware** (`P300`, `P300X2`) — the repo already specs
  the big MoE models there. If 20B/120B is the goal, "wait for P300" is well-founded.
- **Multi-card now** (T3K, Galaxy, or `P150X4`-style data parallel) if you need it
  before P300 is available to you — but confirm the specific model has a working
  multi-card spec (gpt-oss-20b → T3K/Galaxy; 120b → P300X2/Galaxy).

**On "use system memory with the card":** worth pursuing for fitting larger *dense*
weights / longer KV, but it will **not** make gpt-oss-20b run on one P100 (§2). Keep
it a research objective, not a purchase justification. See `CLAUDE.md` → Objectives.

## 5. To actually get a P100 number (unblock the live benchmark)

The hardware is attached (`tt-smi -ls` → single Blackhole `p100a`). What's missing is
a mutually-consistent **run.py era + spec version + docker image** triple. Options,
cheapest first:
1. **Use the local image + matching-era run.py.** The image
   `0.7.0-55fd115-aa4ae1e` is already pulled and (per the Feb run) starts the
   container. Find the branch commit whose `run.py` accepts a 0.7.0 image (before the
   `#3543` "refuse pre-0.11 images" gate) and still has the P100 Llama spec, then run
   with `--override-docker-image` and the shrunk spec. No download.
2. **Pull a v0.11.0+ P100/Blackhole Llama image** (if one exists) and run from the
   pre-migration commit `fee61924c` with the shrunk spec. Requires a large download
   and confirmation such an image exists for Blackhole.
3. **Finish the spec migration** so the branch tip runs (port `run.py` /
   `RuntimeConfig` to the new `model_spec` API) — the most work, but fixes the branch.

Once any of these serves tokens, the shrunk P100 Llama-8B benchmark (`--workflow
benchmarks`) writes tok/s / TTFT / TPOT to `workflow_logs/benchmarks_output/`, and
the dashboard shows live tokens/sec.
