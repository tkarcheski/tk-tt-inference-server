# Discussion: Component Ownership — Keep on Fork vs Upstream to Tenstorrent

**Status:** OPEN — discussion framing, no decisions made yet
**Branch:** `tk/claude-p100-demo-chat-dashboard`
**Related:** `docs/p100_demo_branch.md`

This repo has no git submodules; "submodules" here means the branch's
self-contained components (`finetune/`, `frontend/`, `scripts/p100/`, the
model-spec/eval changes) and whether each should live on the fork, be split
into its own repo, or be offered upstream to `tenstorrent/tt-inference-server`.

## Component-by-component starting positions

| Component | Proposal | Reasoning |
|---|---|---|
| GPT-OSS-20b P100 model spec (`workflows/model_spec.py`) | **Upstream** | Smallest diff, highest general value. Tenstorrent already carries gpt-oss branches (`origin/gpt-oss`, `handrews/gpt-oss`, `tstesco/feat-gpt-oss-120b-support`); an experimental P100 device entry (`default_impl=False`) fits their pattern. Needs rebase onto current main first. |
| gpt-oss eval config (`evals/eval_config.py`) | **Upstream with the spec** | Travels with the model spec; meaningless alone. |
| `scripts/p100/` server/benchmark/chat scripts | **Upstream candidate, second wave** | Useful to any P100 owner, but upstream may prefer this folded into their existing workflow tooling rather than standalone scripts. Propose after the spec lands. |
| `frontend/` chat dashboard | **Keep on fork** (possibly split to own repo) | Demo-quality, personal tooling, overlaps with whatever official UI exists. Splitting to its own repo would decouple it from inference-server rebases — it only talks to the server over HTTP anyway. |
| `finetune/build_dataset.py` | **Keep on fork / move toward rfc repo** | Tightly coupled to robotframework-chat's scenario YAML schema — it arguably belongs *in* the rfc repo next to the data it extracts, not in an inference server. |
| `finetune/train_lora.py` + requirements | **Keep on fork for now** | Generic LoRA SFT trainer; nothing tt-inference-server-specific. If tt-train on Blackhole becomes the path, this gets replaced rather than upstreamed. |
| Dashboard/finetune docs (`docs/p100_demo_*.md`) | **Keep on fork** | Branch-specific. |

## Open questions to resolve

1. **Rebase cost vs upstream value:** upstreaming anything requires rebasing
   onto current `main` (~1,100 commits ahead). Is the model-spec diff small
   enough to cherry-pick onto a fresh branch instead of rebasing everything?
   (Likely yes — that's the recommended route.)
2. **Does upstream want experimental P100 entries at all?** Check whether
   recent upstream gpt-oss work already covers P100/Blackhole, which would
   reduce our upstream contribution to a config tweak or nothing.
3. **Where does dataset extraction live long-term?** If
   `build_dataset.py` moves to the rfc repo, the fine-tune loop spans two
   repos — acceptable, or argument for a small dedicated `finetune` repo?
4. **Dashboard's future:** retire it once an official UI covers the use case,
   or invest and split it out? Depends on how much it's actually used.
5. **120b specs:** drop from this branch (can't run on one P100), or keep as a
   cherry-pick gift for upstream's 120b efforts?

## Proposed next actions (pending discussion)

- [ ] Diff our gpt-oss-20b P100 spec against current upstream `main` gpt-oss
      support to size the real upstream contribution.
- [ ] Open a draft PR (or issue) on `tenstorrent/tt-inference-server` for the
      P100 model-spec entry only, cherry-picked onto current main.
- [ ] Decide fork-vs-rfc-repo home for `build_dataset.py`.
- [ ] Decide whether `frontend/` graduates to its own repo.
