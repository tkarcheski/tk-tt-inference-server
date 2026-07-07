# RSI 24/7 loop — `scripts/rsi/`

Runs the shadow-only MODEL_TUNER loop continuously as a **systemd `--user`**
service. Each round: split → build dataset → LoRA-train a small Qwen (CPU) →
convert to GGUF → serve via Ollama → eval base-vs-tuned on holdout+canary →
McNemar gate → propose. Rounds run back-to-back with a configurable sleep, and
per-round hyperparameters (`LORA_R`/`LORA_ALPHA`/`LR`/`MAX_STEPS` + advancing
`SEED`) are varied so each round is a distinct experiment, not a re-roll.

## Safety — this is shadow-only

`run_loop.run_round` **never** swaps the served endpoint, **never** auto-merges,
and only ever proposes. `can_promote()` returns True solely if someone flips the
service out of shadow mode (`RSI_MODE=live`), and even then nothing acts on it
beyond the draft-PR gate. The service ships `RSI_MODE=shadow` and leaves
`RSI_OPEN_PR` unset, so unattended rounds record results to the `rsi.*`
warehouse and the journal but do **not** open PRs. A human reviews the warehouse
and decides promotions.

## Quick start

```bash
scripts/rsi/rsictl.sh install     # write ~/.config/systemd/user/rsi-loop.service + enable linger
scripts/rsi/rsictl.sh start       # enable on boot + start now
scripts/rsi/rsictl.sh follow      # tail the journal
```

## Control

| Command | Effect |
|---|---|
| `install` | Generate the unit (bakes live `PATH`, deploy dir, config) + `daemon-reload` + enable linger (survives logout/reboot). Re-run to change config. |
| `start` | Clear the stop-file, `enable --now` (starts now and on boot). |
| `pause` | `touch` the stop-file → the loop exits **gracefully after the current round**. `Restart=on-failure` means a clean exit stays down. |
| `resume` | Remove the stop-file and restart. |
| `stop` | `systemctl stop` — immediate SIGTERM; may interrupt a round mid-flight. Prefer `pause`. |
| `restart` | Clear stop-file + restart. |
| `status` / `logs [N]` / `follow` | Service state / last N journal lines / live tail. |
| `uninstall` | Stop, disable, remove the unit. |

## Configuration (env at `install` time, baked into the unit)

| Var | Default | Meaning |
|---|---|---|
| `RSI_SMOKE` | `0` | `1` = tiny 0.5B / few-step rounds for validation; `0` = real rounds (`BASE_MODEL` default Qwen2.5-3B). |
| `RSI_LOOP_SLEEP` | `300` | Seconds between rounds. |
| `RSI_BASE_TAG` | `qwen2.5:3b` | Untuned base arm (paired A/B control); match it to the tuned base model. |
| `RSI_SKIP_SUITES` | `context_window,legal` | Eval suites skipped for latency (frozen firewall split untouched). |
| `RSI_MODE` | `shadow` | Keep `shadow`. `live` only enables the (unimplemented) promote path's gate. |
| `RSI_MAX_ROUNDS` | `0` | `>0` caps total rounds (mostly for testing). |
| `RSI_PUBLISH` | *(unset)* | `1` arms the publish pipeline: on a passing gate, push the GGUF to the git-LFS registry submodule + cut a GitHub release on the fork. Off by default. See `docs/rsi-loop.md` → Publishing. Rollback: `gh release delete` + delete the submodule tag/commit + `ollama rm`. |
| `RSI_PUBLISH_STATUS` | *(unset)* | `1` refreshes the **public status dashboard** each round (commits to the `gh-pages` branch; **never main**). Live: <https://tkarcheski.github.io/tk-tt-inference-server/>. Needs a fork checkout on `gh-pages` at `RSI_STATUS_DIR`. Off by default. See `docs/rsi-loop.md` → Public status dashboard. Rollback: disable Pages / delete `gh-pages`. |
| `RSI_PUSH_OLLAMA` | *(unset)* | `1` pushes a gate-passing model to `tkarcheski/rsi-qwen:3b-latest` (Ollama registry) and rolls it in as the new baseline (self-improving ratchet). The larger RFC-chat cluster validates it. Needs one-time Ollama-registry auth — see [issue #11](https://github.com/tkarcheski/tk-tt-inference-server/issues/11). Off by default; champion only rolls if the push succeeds. Rollback: `ollama rm` + delete the `rsi.baselines` row. |

Example — validate fast, then go real:

```bash
RSI_SMOKE=1 scripts/rsi/rsictl.sh install && scripts/rsi/rsictl.sh start   # confirm rounds cycle
scripts/rsi/rsictl.sh pause                                                # let the current round finish
RSI_SMOKE=0 RSI_LOOP_SLEEP=300 scripts/rsi/rsictl.sh install               # real rounds
scripts/rsi/rsictl.sh resume
```

## Notes

- **Cadence:** real 3B rounds on this CPU box are heavy (train + full eval can be
  a couple of hours each). Lower `MAX_STEPS`, set a smaller `BASE_MODEL`
  (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) + matching `RSI_BASE_TAG`, or raise
  `RSI_LOOP_SLEEP` to throttle.
- **Deploy dir:** the unit points at the checkout that holds this script. Deploy
  from a durable checkout (not an ephemeral `.claude/worktrees/*` copy). The
  training/glue venvs are referenced by absolute path, so the deploy dir only
  needs the source.
- **Kill switch:** `pause` (stop-file) or `RSI_KILL=1` in the environment.
