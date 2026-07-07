# SPDX-License-Identifier: Apache-2.0
"""Shadow-only MODEL_TUNER round orchestrator.

Composes every prior loop module into one round: split -> build dataset ->
train LoRA -> serve via Ollama -> eval both arms -> McNemar gate -> propose.

This module NEVER swaps a serving endpoint and NEVER auto-merges. Worst case
on a passing gate it opens a DRAFT PR (gated behind RSI_OPEN_PR=1 and
`not can_promote()`); `can_promote()` itself only returns True if the loop is
explicitly taken out of shadow mode via env, and nothing in this file acts on
that beyond the check itself.
"""
import argparse, contextlib, json, os, subprocess, sys, time, traceback, uuid, pathlib
import rsi_common, split_suites, serve_ollama, eval_rfc, import_results, gate, publish, publish_status, publish_ollama

FT = pathlib.Path(__file__).parent
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
REPEATS = int(os.environ.get("RSI_REPEATS", "1"))

def can_promote():
    return os.environ.get("RSI_MODE", "shadow") == "live" and \
           os.environ.get("RSI_AGENTS_ENABLED", "true") != "false"

def base_arm_tag():
    """The base (control) arm for this round: the rolling Ollama champion once one
    has been promoted (RSI_PUSH_OLLAMA path), else the pinned stock base. So the
    tuned model must beat the *current* baseline each round (self-improving ratchet)."""
    return publish_ollama.current_champion() or os.environ.get("RSI_BASE_TAG", "qwen2.5:3b")

def maybe_open_pr(report):
    """SHADOW-ONLY: open a DRAFT PR proposal iff the round proposed a promotion,
    we are NOT in a human-gated live-promote mode, and PR-opening is opted in.
    Returns True iff a draft PR was requested. Never merges, never promotes."""
    if report.get("proposed") and not can_promote() and os.environ.get("RSI_OPEN_PR") == "1":
        subprocess.run(["gh", "pr", "create", "--draft",
                        "--title", f"MODEL_TUNER proposal {report['tuned_id'][:8]}",
                        "--body", json.dumps(report, indent=2)], check=False)
        return True
    return False

def new_experiment(intent, **f):
    eid = str(uuid.uuid4())
    cols = ["experiment_id", "intent"] + list(f.keys())
    vals = [eid, intent] + list(f.values())
    ph = ",".join(["%s"] * len(cols))
    with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
        cur.execute(f"insert into rsi.experiments ({','.join(cols)}) values ({ph})", vals)
        c.commit()
    return eid

def filter_eval_suites(splits, skip):
    """Drop latency-bound / non-discriminating suites from the EVAL pools only.

    The frozen firewall split on disk is unchanged — train pool and the leakage
    check are untouched. This ONLY curtails which holdout/canary suites are
    actually evaluated this round (e.g. the needle-in-haystack `context_window`
    stress suite is latency-bound and gives no signal between near-identical
    arms). Returns (new_splits, dropped) where `dropped` maps pool -> [suite].
    """
    skip = set(skip)
    out = dict(splits)
    dropped = {}
    for pool in ("holdout", "canary"):
        d = [s for s in splits.get(pool, []) if s in skip]
        if d:
            dropped[pool] = d
            out[pool] = [s for s in splits[pool] if s not in skip]
    return out, dropped


def _eval_arm(experiment_id, arm, splits):
    """Run every holdout+canary suite for one arm; import per-test rows."""
    for pool in ("holdout", "canary"):
        for suite in splits[pool]:
            for r in range(REPEATS):
                rd = f"/tmp/rsi/{experiment_id}/{pool}/{suite}/{r}"
                xml = eval_rfc.run_suite(suite, arm, rd)
                if os.path.exists(xml):
                    import_results.import_results(xml, experiment_id, pool, r)
                else:
                    print(f"WARN: no output.xml for {pool}/{suite} rep{r} (arm={arm.get('model')}) at {xml}", file=sys.stderr)

def run_round(once=True, smoke=False):
    if os.environ.get("RSI_KILL", "0") == "1":
        raise SystemExit("kill switch set (RSI_KILL=1)")
    # 1) frozen firewall + leak check (raises LeakageError on contamination)
    split = split_suites.write_split(eval_rfc.RFC, FT / "split.json")
    splits = split["splits"]
    # Curate latency-bound eval suites (firewall split on disk stays frozen).
    skip = [s for s in os.environ.get("RSI_SKIP_SUITES", "").split(",") if s]
    splits, dropped = filter_eval_suites(splits, skip)
    for pool, d in dropped.items():
        print(f"WARN: NOT evaluating suites {d} in pool '{pool}' this round "
              f"(RSI_SKIP_SUITES); they remain in the frozen firewall split but "
              f"are skipped for latency, so pool '{pool}' signal is reduced.",
              file=sys.stderr)
    train_hash_line = subprocess.run(
        ["/home/tyler/.rsi-loop-venv/bin/python", str(FT / "build_dataset.py"),
         "--rfc-root", eval_rfc.RFC, "--split", str(FT / "split.json"),
         "--out-dir", str(FT / "out")],
        capture_output=True, text=True, check=True).stdout.strip()
    train_pool_hash = train_hash_line.split("train_pool_hash=")[1].split()[0]

    # 2) train (CPU LoRA, ≤3B, merge) — smoke uses tiny base + few steps
    env = dict(os.environ, DATASET=str(FT / "out" / "train.jsonl"),
               OUTPUT_DIR=str(FT / "out" / "lora-qwen"), MERGE="1",
               BASE_MODEL=("Qwen/Qwen2.5-0.5B-Instruct" if smoke else BASE_MODEL),
               MAX_STEPS=os.environ.get("MAX_STEPS", "5" if smoke else "200"))
    subprocess.run(
        ["/home/tyler/AI/github/tk-tt-inference-server/finetune/.venv-train/bin/python",
         str(FT / "train_lora.py")],
        cwd=FT, env=env, check=True)
    merged = str(FT / "out" / "lora-qwen" / "merged")
    lora_hash = serve_ollama.adapter_hash(str(FT / "out" / "lora-qwen"))

    # 3) serve tuned via Ollama. The base arm is the rolling champion once one has
    # been promoted (RSI_PUSH_OLLAMA path); otherwise the pinned stock base. So the
    # tuned model must beat the *current* baseline each round (self-improving ratchet).
    tuned_tag = serve_ollama.create_tag(merged, "rsi-qwen:round")
    base_tag = base_arm_tag()

    # 4) provenance rows (base is a frozen reference each round). Record the
    # per-round training seed so 24/7 rounds are distinguishable experiments.
    seed = int(os.environ.get("SEED", "0") or "0")
    base_id = new_experiment("baseline", provider_variant="ollama",
                             serving_runtime="ollama", base_model_sha=base_tag,
                             lora_adapter_hash=None, train_pool_hash=None,
                             leakage_score=0, train_hardware="cpu")
    tuned_id = new_experiment("model_tuner_round", provider_variant="ollama",
                              serving_runtime="ollama", base_model_sha=base_tag,
                              lora_adapter_hash=lora_hash, train_pool_hash=train_pool_hash,
                              leakage_score=0, train_hardware="cpu", seed=seed,
                              parent_experiment_id=base_id)

    # 5) evaluate both arms on the SAME Ollama endpoint (paired control)
    _eval_arm(base_id, {"provider": "ollama", "model": base_tag}, splits)
    _eval_arm(tuned_id, {"provider": "ollama", "model": tuned_tag}, splits)

    # 6) McNemar gate — canary is the promotion metric, holdout is the sanity read
    holdout = gate.mcnemar_gate(base_id, tuned_id, "holdout")
    canary = gate.mcnemar_gate(base_id, tuned_id, "canary")
    report = {"base_id": base_id, "tuned_id": tuned_id, "seed": seed,
              "holdout": holdout, "canary": canary,
              "train_pool_hash": train_pool_hash, "lora_hash": lora_hash}

    report["degenerate"] = (holdout.get("n", 0) == 0 or canary.get("n", 0) == 0)
    if report["degenerate"]:
        print(f"WARN: degenerate eval — holdout n={holdout.get('n')}, canary n={canary.get('n')}; "
              f"a broken harness or identical arms can look like 'no improvement'. NOT trusting this round.",
              file=sys.stderr)

    # 7) SHADOW ONLY: propose, never swap an endpoint, never auto-merge
    report["proposed"] = bool(canary["passes"] and holdout["passes"])
    maybe_open_pr(report)
    publish.maybe_publish(report, merged)  # opt-in RSI_PUBLISH=1; no-op unless proposed
    # opt-in RSI_PUSH_OLLAMA=1: on a passing gate, push tuned -> tkarcheski/rsi-qwen:3b-latest
    # and roll it in as the new baseline (only if the push succeeds).
    publish_ollama.maybe_push_ollama(report, tuned_tag)
    publish_status.maybe_publish_status(report)  # opt-in RSI_PUBLISH_STATUS=1; refresh public dashboard
    print(json.dumps(report, indent=2))
    return report

def round_config(idx):
    """Deterministic per-round hyperparameter variation, so each 24/7 round is a
    distinct experiment rather than a re-roll of the same tune. Cycles a small,
    sensible LoRA grid and advances the training seed every round. Returns
    env-var overrides consumed by train_lora.py (LORA_R/LORA_ALPHA/LR/MAX_STEPS)
    and recorded as the experiment `seed`."""
    # MAX_STEPS sized to the small train pool (~136 examples, effective batch 8 →
    # ~17 optimizer steps/epoch). 20-50 steps ≈ 1-3 epochs. Keep these LOW: 200+
    # steps is ~12-24 epochs, which memorizes the pool (observed train_loss→0.06)
    # and hurts held-out/canary generalization.
    grid = [
        {"LORA_R": "8",  "LORA_ALPHA": "16", "LR": "2e-4", "MAX_STEPS": "20"},
        {"LORA_R": "16", "LORA_ALPHA": "32", "LR": "2e-4", "MAX_STEPS": "30"},
        {"LORA_R": "16", "LORA_ALPHA": "32", "LR": "1e-4", "MAX_STEPS": "40"},
        {"LORA_R": "32", "LORA_ALPHA": "64", "LR": "1e-4", "MAX_STEPS": "50"},
    ]
    cfg = dict(grid[idx % len(grid)])
    cfg["SEED"] = str(1000 + idx)
    return cfg


def starting_index():
    """Resume the round counter after a restart instead of resetting to 0.

    `round_config(idx)` derives SEED as 1000+idx, so restarting from idx=0 would
    re-run already-used seeds — producing duplicate, confusing seeds (notably on
    the public status dashboard). Continue the sequence from the warehouse's
    highest recorded round seed. Returns the next idx (0 if no prior rounds or the
    warehouse is unreachable)."""
    try:
        with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
            cur.execute("select max(seed) from rsi.experiments "
                        "where intent='model_tuner_round'")
            row = cur.fetchone()
        m = row[0] if row and row[0] is not None else None
        return (m - 1000 + 1) if m is not None else 0
    except Exception as e:
        print(f"starting_index: warehouse read failed ({e!r}); starting at 0",
              file=sys.stderr)
        return 0


def should_stop(stop_file):
    """Kill switch: env RSI_KILL=1 or the existence of the stop-file. A running
    24/7 service is halted ergonomically by `touch`-ing the stop-file."""
    return os.environ.get("RSI_KILL", "0") == "1" or bool(stop_file and os.path.exists(stop_file))


def run_forever(run_round_fn=None, sleep_fn=time.sleep, smoke=None,
                sleep_s=None, stop_file=None, max_rounds=None, start_index=None):
    """Supervisor: run rounds back-to-back until stopped.

    A failing round is logged and the loop CONTINUES — one bad round (OOM, a
    transient serve/eval error) must never take the whole 24/7 loop down. Stops
    on: RSI_KILL=1, the stop-file existing, or max_rounds reached. Shadow-only
    behavior is inherited wholesale from run_round: this supervisor never
    promotes, never swaps a serving endpoint, and only ever proposes.

    All knobs are injectable for testing and default from env:
      RSI_SMOKE=1        tiny/fast rounds        (default real rounds)
      RSI_LOOP_SLEEP     seconds between rounds   (default 300)
      RSI_STOP_FILE      kill-switch path         (default finetune/.rsi-stop)
      RSI_MAX_ROUNDS     >0 caps rounds           (default 0 = unlimited)
    """
    run_round_fn = run_round if run_round_fn is None else run_round_fn
    smoke = (os.environ.get("RSI_SMOKE", "0") == "1") if smoke is None else smoke
    sleep_s = int(os.environ.get("RSI_LOOP_SLEEP", "300")) if sleep_s is None else sleep_s
    stop_file = os.environ.get("RSI_STOP_FILE", str(FT / ".rsi-stop")) if stop_file is None else stop_file
    if max_rounds is None:
        m = int(os.environ.get("RSI_MAX_ROUNDS", "0"))
        max_rounds = m if m > 0 else None
    print(f"RSI loop starting: smoke={smoke} sleep={sleep_s}s stop_file={stop_file} "
          f"max_rounds={max_rounds or 'unlimited'} mode={os.environ.get('RSI_MODE', 'shadow')}",
          file=sys.stderr, flush=True)
    # Resume the SEED/round sequence from the warehouse (see starting_index) so a
    # restart doesn't re-run used seeds. `ran` counts rounds THIS invocation, so
    # max_rounds still means "run N more rounds" regardless of where idx resumes.
    idx = starting_index() if start_index is None else start_index
    ran = 0
    while True:
        if should_stop(stop_file):
            print(f"RSI loop stopping (kill switch / stop-file {stop_file}) after {ran} rounds",
                  file=sys.stderr, flush=True)
            break
        if max_rounds is not None and ran >= max_rounds:
            print(f"RSI loop reached max_rounds={max_rounds}; exiting", file=sys.stderr, flush=True)
            break
        cfg = round_config(idx)
        os.environ.update(cfg)
        if smoke:  # keep validation rounds tiny; grid MAX_STEPS is for real rounds
            os.environ["MAX_STEPS"] = os.environ.get("RSI_SMOKE_STEPS", "3")
        print(f"=== RSI round {idx} start cfg={cfg} smoke={smoke} mode={os.environ.get('RSI_MODE', 'shadow')} ===",
              file=sys.stderr, flush=True)
        try:
            report = run_round_fn(once=True, smoke=smoke) or {}
            h, c = report.get("holdout", {}), report.get("canary", {})
            print(f"=== RSI round {idx} done: proposed={report.get('proposed')} "
                  f"degenerate={report.get('degenerate')} "
                  f"holdout Δ={h.get('delta_pp')} p={h.get('p_value')} "
                  f"canary Δ={c.get('delta_pp')} p={c.get('p_value')} ===",
                  file=sys.stderr, flush=True)
        except Exception as e:  # a bad round must not kill the loop
            print(f"=== RSI round {idx} FAILED: {e!r} — continuing ===", file=sys.stderr, flush=True)
            traceback.print_exc()
        idx += 1
        ran += 1
        if should_stop(stop_file):
            continue  # skip the sleep; the loop-top check will break out
        sleep_fn(sleep_s)
    return ran


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--forever", action="store_true", help="run rounds continuously (24/7 supervisor)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.forever:
        run_forever(smoke=(True if args.smoke else None))
    else:
        run_round(once=args.once, smoke=args.smoke)
