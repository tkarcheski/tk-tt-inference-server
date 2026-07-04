# SPDX-License-Identifier: Apache-2.0
"""Publish a tuned model when a round clears the promotion gate.

Fully automatic but OPT-IN via RSI_PUBLISH=1 (default off — shadow-safe). On a
passing gate (`report["proposed"]`), `maybe_publish`:
  1. commits the GGUF + a `Modelfile` + a provenance `card.md` into the
     git-submodule model registry (git-LFS), tagged `rsi-qwen-{version}`;
  2. cuts a GitHub RELEASE on the FORK (never upstream) as the human record with
     clone-and-`ollama create` instructions;
  3. records the publication in `rsi.publications` for audit + idempotency.

Each step is isolated: a failure in one is logged and never crashes the round,
and a step never blocks the others. A given tuned_id is published at most once.
"""
import contextlib
import os
import shutil
import subprocess
import sys

import rsi_common

# Releases MUST target the fork — `gh`'s default here resolves to upstream.
FORK_REPO = "tkarcheski/tk-tt-inference-server"
# The git-LFS model-registry repo backing the "Ollama" publish target.
REGISTRY_REMOTE = "tkarcheski/rsi-ollama-models"
_FT = os.path.dirname(os.path.abspath(__file__))


def publish_version(report):
    """Unique, non-overwriting version: v{seed}-{adapter_hash[:8]}."""
    return f"v{report.get('seed', 0)}-{(report.get('lora_hash') or '')[:8]}"


def _submodule_dir():
    return os.environ.get("RSI_MODELS_SUBMODULE",
                          os.path.join(os.path.dirname(_FT), "ollama-models"))


def _run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed ({r.returncode}): "
                           f"{r.stderr.strip() or r.stdout.strip()}")
    return (r.stdout or "").strip()


def _card(report, version):
    h, c = report.get("holdout", {}), report.get("canary", {})
    return (
        f"# rsi-qwen {version}\n\n"
        "Auto-published by the RSI MODEL_TUNER loop on a passing promotion gate "
        "(≥5pp AND p<0.05 on both holdout and canary).\n\n"
        f"- tuned_experiment: `{report.get('tuned_id')}`\n"
        f"- lora_adapter_hash: `{report.get('lora_hash')}`\n"
        f"- train_pool_hash: `{report.get('train_pool_hash')}`\n"
        f"- holdout: Δ{h.get('delta_pp')}pp p={h.get('p_value')}\n"
        f"- canary:  Δ{c.get('delta_pp')}pp p={c.get('p_value')}\n\n"
        f"## Use\n\n    ollama create rsi-qwen:{version} -f Modelfile\n"
    )


def commit_to_submodule(gguf_path, report, version):
    """Copy the GGUF + Modelfile + card into the registry submodule and push a
    tagged commit. Raises if the submodule isn't initialized (caller isolates)."""
    sub = _submodule_dir()
    dot_git = os.path.join(sub, ".git")            # dir in a clone, file in a submodule
    if not (os.path.isdir(dot_git) or os.path.isfile(dot_git)):
        raise RuntimeError(f"model registry submodule not initialized at {sub} "
                           f"(run `git submodule update --init {os.path.basename(sub)}`)")
    rel = os.path.join("rsi-qwen", version)
    dest = os.path.join(sub, rel)
    os.makedirs(dest, exist_ok=True)
    shutil.copy(gguf_path, os.path.join(dest, "model.gguf"))
    with open(os.path.join(dest, "Modelfile"), "w") as fh:
        fh.write("FROM ./model.gguf\n")
    with open(os.path.join(dest, "card.md"), "w") as fh:
        fh.write(_card(report, version))
    tag = f"rsi-qwen-{version}"
    _run(["git", "add", "-A", rel], cwd=sub)
    _run(["git", "commit", "-m", f"publish rsi-qwen {version}"], cwd=sub)
    # Annotated tag + explicit tag push: `--follow-tags` skips lightweight tags,
    # and the version tag is how consumers `git checkout` the model. Version is
    # unique per round (seed+hash) so no force is needed.
    _run(["git", "tag", "-a", tag, "-m", f"rsi-qwen {version}"], cwd=sub)
    _run(["git", "push", "origin", "HEAD"], cwd=sub)
    _run(["git", "push", "origin", tag], cwd=sub)
    return {"commit": _run(["git", "rev-parse", "HEAD"], cwd=sub), "tag": tag}


def _release_notes(report, version, submodule_ref):
    h, c = report.get("holdout", {}), report.get("canary", {})
    tag = f"rsi-qwen-{version}"
    lines = [
        f"Auto-published tuned model **rsi-qwen {version}** by the RSI MODEL_TUNER "
        "loop — the promotion gate passed (≥5pp AND p<0.05 on **both** pools).",
        "",
        "**Gate:**",
        f"- holdout: Δ {h.get('delta_pp')}pp, p={h.get('p_value')}",
        f"- canary:  Δ {c.get('delta_pp')}pp, p={c.get('p_value')}",
        "",
        f"**Provenance:** tuned_id `{report.get('tuned_id')}`, "
        f"adapter `{report.get('lora_hash')}`, train_pool `{report.get('train_pool_hash')}`.",
        "",
    ]
    if submodule_ref:
        lines += [
            "**Get the model** (weights in the git-LFS registry submodule):",
            "```",
            f"git clone https://github.com/{REGISTRY_REMOTE}",
            f"cd rsi-ollama-models && git checkout {tag}",
            f"cd rsi-qwen/{version} && ollama create rsi-qwen:{version} -f Modelfile",
            "```",
        ]
    else:
        lines.append("_Note: registry push did not complete for this release; "
                     "weights are not yet available in the submodule._")
    return "\n".join(lines)


def create_release(report, version, submodule_ref):
    tag = f"rsi-qwen-{version}"
    _run(["gh", "release", "create", tag, "--repo", FORK_REPO,
          "--title", f"rsi-qwen {version}",
          "--notes", _release_notes(report, version, submodule_ref)])
    return f"https://github.com/{FORK_REPO}/releases/tag/{tag}"


def already_published(tuned_id):
    with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM rsi.publications WHERE tuned_id=%s", (tuned_id,))
        return cur.fetchone() is not None


def record_publication(report, version, submodule_ref, release_url):
    h, c = report.get("holdout", {}), report.get("canary", {})
    with contextlib.closing(rsi_common.connect()) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rsi.publications "
            "(version, tuned_id, submodule_commit, release_url, holdout_delta_pp, canary_delta_pp) "
            "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (tuned_id) DO NOTHING",
            (version, report.get("tuned_id"), (submodule_ref or {}).get("commit"),
             release_url, h.get("delta_pp"), c.get("delta_pp")))
        conn.commit()
        return cur.rowcount > 0


def maybe_publish(report, merged_dir):
    """Publish iff the gate passed AND RSI_PUBLISH=1. Idempotent per tuned_id.
    Returns a result dict, or None when it did not publish."""
    if not (report.get("proposed") and os.environ.get("RSI_PUBLISH") == "1"):
        return None
    tuned_id = report.get("tuned_id")
    try:
        if already_published(tuned_id):
            print(f"publish: {tuned_id} already published — skip", file=sys.stderr)
            return None
    except Exception as e:  # DB down should not block a publish
        print(f"publish: idempotency check failed ({e!r}); continuing", file=sys.stderr)

    version = publish_version(report)
    gguf = os.path.join(merged_dir, "model.gguf")
    result = {"version": version}
    submodule_ref = release_url = None

    try:
        submodule_ref = commit_to_submodule(gguf, report, version)
        result["submodule"] = submodule_ref
    except Exception as e:
        print(f"publish: registry (submodule) step failed: {e!r}", file=sys.stderr)
    try:
        release_url = create_release(report, version, submodule_ref)
        result["release_url"] = release_url
    except Exception as e:
        print(f"publish: GitHub release step failed: {e!r}", file=sys.stderr)

    if submodule_ref or release_url:                      # only record real publishes
        try:
            record_publication(report, version, submodule_ref, release_url)
        except Exception as e:
            print(f"publish: warehouse record failed: {e!r}", file=sys.stderr)
    print(f"publish: rsi-qwen {version} -> release={release_url} "
          f"registry={'ok' if submodule_ref else 'skipped'}", file=sys.stderr)
    return result
