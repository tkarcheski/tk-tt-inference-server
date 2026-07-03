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
import argparse, json, os, subprocess, uuid, pathlib
import rsi_common, split_suites, serve_ollama, eval_rfc, import_results, gate

FT = pathlib.Path(__file__).parent
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
REPEATS = int(os.environ.get("RSI_REPEATS", "1"))

def can_promote():
    return os.environ.get("RSI_MODE", "shadow") == "live" and \
           os.environ.get("RSI_AGENTS_ENABLED", "true") != "false"

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
    with rsi_common.connect() as c, c.cursor() as cur:
        cur.execute(f"insert into rsi.experiments ({','.join(cols)}) values ({ph})", vals)
        c.commit()
    return eid

def _eval_arm(experiment_id, arm, splits):
    """Run every holdout+canary suite for one arm; import per-test rows."""
    for pool in ("holdout", "canary"):
        for suite in splits[pool]:
            for r in range(REPEATS):
                rd = f"/tmp/rsi/{experiment_id}/{pool}/{suite}/{r}"
                xml = eval_rfc.run_suite(suite, arm, rd)
                if os.path.exists(xml):
                    import_results.import_results(xml, experiment_id, pool, r)

def run_round(once=True, smoke=False):
    if os.environ.get("RSI_KILL", "0") == "1":
        raise SystemExit("kill switch set (RSI_KILL=1)")
    # 1) frozen firewall + leak check (raises LeakageError on contamination)
    split = split_suites.write_split(eval_rfc.RFC, FT / "split.json")
    splits = split["splits"]
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

    # 3) serve tuned via Ollama (base arm reuses a pinned base tag)
    tuned_tag = serve_ollama.create_tag(merged, "rsi-qwen:round")
    base_tag = os.environ.get("RSI_BASE_TAG", "qwen2.5:3b")

    # 4) provenance rows (base is a frozen reference each round)
    base_id = new_experiment("baseline", provider_variant="ollama",
                             serving_runtime="ollama", base_model_sha=base_tag,
                             lora_adapter_hash=None, train_pool_hash=None,
                             leakage_score=0, train_hardware="cpu")
    tuned_id = new_experiment("model_tuner_round", provider_variant="ollama",
                              serving_runtime="ollama", base_model_sha=base_tag,
                              lora_adapter_hash=lora_hash, train_pool_hash=train_pool_hash,
                              leakage_score=0, train_hardware="cpu", parent_experiment_id=base_id)

    # 5) evaluate both arms on the SAME Ollama endpoint (paired control)
    _eval_arm(base_id, {"provider": "ollama", "model": base_tag}, splits)
    _eval_arm(tuned_id, {"provider": "ollama", "model": tuned_tag}, splits)

    # 6) McNemar gate — canary is the promotion metric, holdout is the sanity read
    holdout = gate.mcnemar_gate(base_id, tuned_id, "holdout")
    canary = gate.mcnemar_gate(base_id, tuned_id, "canary")
    report = {"base_id": base_id, "tuned_id": tuned_id,
              "holdout": holdout, "canary": canary,
              "train_pool_hash": train_pool_hash, "lora_hash": lora_hash}

    # 7) SHADOW ONLY: propose, never swap an endpoint, never auto-merge
    report["proposed"] = bool(canary["passes"] and holdout["passes"])
    maybe_open_pr(report)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_round(once=args.once, smoke=args.smoke)
