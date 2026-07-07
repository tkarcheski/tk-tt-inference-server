# SPDX-License-Identifier: Apache-2.0
"""Push a gate-passing tuned model to the Ollama registry as the rolling baseline.

When a round clears the promotion gate (`report["proposed"]`) AND
`RSI_PUSH_OLLAMA=1`, this tags the round's tuned model as
`tkarcheski/rsi-qwen:3b-latest` and `ollama push`es it, then records it in
`rsi.baselines` as the new current champion. The loop's next round evaluates the
tuned arm against this champion (`current_champion()` feeds `run_round`'s base
arm), so each promotion raises the bar it must clear — a self-improving ratchet.
The larger RFC-chat cluster pulls `3b-latest` for full validation.

Opt-in (default off) and shadow-safe: every step is isolated so a failure never
crashes a round, and **the champion only rolls if the push actually succeeds** —
if `ollama push` fails (e.g. registry auth not set up), the baseline is left
unchanged, so the loop never advances onto a model the cluster never received.

Ollama-registry auth: pushing to the `tkarcheski/*` namespace needs this box's
Ollama public key (`~/.ollama/id_ed25519.pub`) registered on the `tkarcheski`
ollama.com account (Settings -> Keys). Until then `ollama push` 401s and this
is a logged no-op.
"""
import contextlib
import os
import subprocess
import sys

import rsi_common

# The rolling-baseline tag the larger RFC-chat cluster pulls.
OLLAMA_TARGET = os.environ.get("RSI_OLLAMA_TARGET", "tkarcheski/rsi-qwen:3b-latest")
DEFAULT_BASE = "qwen2.5:3b"


def baseline_version(report):
    """Immutable version for this promotion: v{seed}-{adapter_hash[:8]}."""
    return f"v{report.get('seed', 0)}-{(report.get('lora_hash') or '')[:8]}"


def current_champion():
    """Ollama ref of the current rolling baseline, or None if nothing promoted yet.
    Read at the top of each round to set the base arm (the model to beat)."""
    try:
        with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
            cur.execute("select ollama_ref from rsi.baselines where is_current limit 1")
            row = cur.fetchone()
        return row[0] if row else None
    except Exception as e:  # a warehouse hiccup must not break the round's base arm
        print(f"champion: warehouse read failed ({e!r}); using default base",
              file=sys.stderr)
        return None


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed ({r.returncode}): "
                           f"{r.stderr.strip() or r.stdout.strip()}")
    return (r.stdout or "").strip()


def push_to_ollama(tuned_tag, target=None):
    """Local-copy the tuned tag under the registry name, then push it. Raises on
    failure (caller isolates). Requires ollama.com auth for the target namespace."""
    target = target or OLLAMA_TARGET
    _run(["ollama", "cp", tuned_tag, target])
    _run(["ollama", "push", target])
    return target


def record_baseline(report, version, ollama_ref):
    """Make this the sole current champion (demote the previous) and record it.
    Returns True iff a new row was written."""
    h, c = report.get("holdout", {}), report.get("canary", {})
    with contextlib.closing(rsi_common.connect()) as conn, conn.cursor() as cur:
        cur.execute("update rsi.baselines set is_current=false where is_current")
        cur.execute(
            "insert into rsi.baselines "
            "(version, tuned_id, ollama_ref, holdout_delta_pp, canary_delta_pp, "
            " is_current, cluster_status) "
            "values (%s,%s,%s,%s,%s,true,'pending') on conflict (tuned_id) do nothing",
            (version, report.get("tuned_id"), ollama_ref,
             h.get("delta_pp"), c.get("delta_pp")))
        conn.commit()
        return cur.rowcount > 0


def maybe_push_ollama(report, tuned_tag="rsi-qwen:round"):
    """Promote the tuned model to the rolling baseline iff the gate passed AND
    RSI_PUSH_OLLAMA=1. The champion rolls only on a successful push. Isolated so a
    failure never crashes the round. Returns a result dict, or None."""
    if not (report.get("proposed") and os.environ.get("RSI_PUSH_OLLAMA") == "1"):
        return None
    version = baseline_version(report)
    try:
        ollama_ref = push_to_ollama(tuned_tag)
    except Exception as e:  # do NOT roll the baseline if the cluster never got it
        ns = OLLAMA_TARGET.split("/")[0]
        print(f"champion: ollama push failed ({e!r}); NOT rolling the baseline "
              f"(is ollama.com auth set up for the '{ns}' namespace?)", file=sys.stderr)
        return None
    try:
        record_baseline(report, version, ollama_ref)
    except Exception as e:
        print(f"champion: pushed {version} but warehouse record failed ({e!r})",
              file=sys.stderr)
    print(f"champion: promoted {version} -> {ollama_ref} (new rolling baseline)",
          file=sys.stderr)
    return {"version": version, "ollama_ref": ollama_ref}
