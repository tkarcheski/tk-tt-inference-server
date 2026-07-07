# SPDX-License-Identifier: Apache-2.0
"""Publish a live, WORLD-VISIBLE status dashboard of the RSI MODEL_TUNER loop.

The shadow loop's results live only in the local Postgres warehouse + the systemd
journal. This module makes them public: it recomputes every round's McNemar gate
outcome from the stored `test_results` (reusing `gate.mcnemar_from_pairs` — the
exact statistic the live gate uses), renders three artifacts, and commits them to
the `gh-pages` branch of the PUBLIC fork, which GitHub Pages serves as a live page:

  * data.json  — machine-readable full round history + summary
  * index.html — a self-contained dashboard (no external assets) that re-fetches
                 data.json every few minutes, so an open tab updates in near-real-time
  * README.md  — a Markdown leaderboard that renders on GitHub directly

Read-only over the warehouse and SHADOW-SAFE: it only reports what the gate
already decided, never promotes, and never pushes to main (the `gh-pages` branch
is Pages-only). Opt-in via RSI_PUBLISH_STATUS=1 for the per-round hook; the
standalone entrypoint (`python publish_status.py`) always publishes so a timer/cron
can drive near-real-time refreshes independent of the ~hours-long round cadence.

Only aggregate, non-sensitive fields are published (seed, timestamps, short
hashes, per-pool pass rates + McNemar deltas/p-values, gate decision). No DB
credentials, tokens, file paths, prompts, or grader rationales ever leave the box.
"""
import argparse
import contextlib
import datetime
import json
import os
import shutil
import subprocess
import sys

import rsi_common
import gate

POOLS = ("holdout", "canary")
FORK_REPO = "tkarcheski/tk-tt-inference-server"
# Where the models live (surfaced on the dashboard).
BASE_HF = "Qwen/Qwen2.5-3B-Instruct"
BASE_HF_URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"
TUNED_REGISTRY = "tkarcheski/rsi-ollama-models"          # git-LFS registry (private)
TUNED_REGISTRY_URL = "https://github.com/tkarcheski/rsi-ollama-models"
STATUS_BRANCH = os.environ.get("RSI_STATUS_BRANCH", "gh-pages")
STATUS_REMOTE = os.environ.get("RSI_STATUS_REMOTE", "origin")
PAGES_URL = os.environ.get("RSI_STATUS_PAGES_URL",
                           "https://tkarcheski.github.io/tk-tt-inference-server/")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _status_dir():
    return os.environ.get("RSI_STATUS_DIR",
                          os.path.join(os.path.expanduser("~"), "AI", "rsi-status-pages"))


def collect_rounds():
    """Every model_tuner_round paired with its baseline (parent_experiment_id),
    each pool's outcome recomputed from stored test_results. Two batched queries
    (metadata + matched per-test outcomes) instead of a gate call per round."""
    with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
        cur.execute(
            "select experiment_id, parent_experiment_id, seed, lora_adapter_hash, "
            "       train_pool_hash, to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS') "
            "from rsi.experiments "
            "where intent='model_tuner_round' and parent_experiment_id is not null "
            "order by created_at")
        meta = cur.fetchall()
        # All matched (base,tuned) per-test outcomes for every round, one query.
        cur.execute(
            "select e.experiment_id, tr.pool, "
            "  bool_or(tr.status='PASS') filter (where tr.experiment_id=e.parent_experiment_id), "
            "  bool_or(tr.status='PASS') filter (where tr.experiment_id=e.experiment_id) "
            "from rsi.experiments e "
            "join rsi.test_results tr "
            "  on tr.experiment_id in (e.experiment_id, e.parent_experiment_id) "
            "where e.intent='model_tuner_round' and e.parent_experiment_id is not null "
            "group by e.experiment_id, tr.pool, tr.suite_id, tr.test_id, tr.repeat_idx "
            "having count(distinct tr.experiment_id)=2")
        pairs = {}
        for tuned_id, pool, base_pass, tuned_pass in cur.fetchall():
            pairs.setdefault((tuned_id, pool), []).append((base_pass, tuned_pass))

    rounds = []
    for tuned_id, base_id, seed, lora_hash, pool_hash, started in meta:
        r = {"seed": seed, "started": started, "tuned_id": tuned_id, "base_id": base_id,
             "lora_hash": (lora_hash or "")[:8], "train_pool_hash": (pool_hash or "")[:8]}
        for pool in POOLS:
            g = gate.mcnemar_from_pairs(pairs.get((tuned_id, pool), []))
            r[pool] = {"n": g["n"],
                       "base_pass": round(g["base_pass"], 1),
                       "tuned_pass": round(g["tuned_pass"], 1),
                       "delta_pp": round(g["delta_pp"], 1),
                       "p_value": round(g["p_value"], 4),
                       "passes": g["passes"]}
        r["degenerate"] = r["holdout"]["n"] == 0 or r["canary"]["n"] == 0
        r["proposed"] = bool(r["holdout"]["passes"] and r["canary"]["passes"]
                             and not r["degenerate"])
        rounds.append(r)
    return rounds


def build_data(rounds, generated_at, base_model):
    graded = [r for r in rounds if not r["degenerate"]]
    best = max((r["canary"]["delta_pp"] for r in graded), default=None)
    return {
        "generated_at": generated_at,
        "pages_url": PAGES_URL,
        "repo_url": f"https://github.com/{FORK_REPO}",
        # rsi-loop.md is bundled onto this (gh-pages) branch, so the link resolves
        # regardless of what has (or hasn't) been merged to main.
        "doc_url": f"https://github.com/{FORK_REPO}/blob/{STATUS_BRANCH}/rsi-loop.md",
        "base_model": base_model,
        "models": {
            "base_ollama": base_model,
            "base_hf": BASE_HF, "base_hf_url": BASE_HF_URL,
            "tuned_registry": TUNED_REGISTRY, "tuned_registry_url": TUNED_REGISTRY_URL,
            "registry_private": True,
        },
        "shadow_only": True,
        "summary": {
            "total_rounds": len(rounds),
            "graded_rounds": len(graded),
            "proposed": sum(1 for r in rounds if r["proposed"]),
            "best_canary_delta_pp": best,
            "latest_seed": rounds[-1]["seed"] if rounds else None,
            "latest_started": rounds[-1]["started"] if rounds else None,
        },
        "rounds": rounds,
    }


def render_readme(data):
    s, m = data["summary"], data["models"]
    lines = [
        "# RSI MODEL_TUNER — live shadow-loop status",
        "",
        "> Auto-generated every round by the 24/7 [RSI MODEL_TUNER loop]"
        f"({data['doc_url']}). **Shadow-only**: the loop "
        "LoRA-fine-tunes a small Qwen on robotframework-chat suites, evaluates the "
        "tuned model against the untuned base on a frozen holdout+canary split, and "
        "runs a McNemar promotion gate. Passing rounds are *proposed*, never "
        "auto-promoted — a human decides.",
        "",
        f"**Live dashboard:** {data['pages_url']}",
        "",
        "## Models",
        "",
        f"- **Base (control):** [`{m['base_hf']}`]({m['base_hf_url']}) on Hugging Face, "
        f"served as Ollama `{m['base_ollama']}`.",
        f"- **Tuned (per round):** a LoRA fine-tune of the base, merged + served as "
        "Ollama `rsi-qwen:round`.",
        f"- **Published tuned models:** [`{m['tuned_registry']}`]({m['tuned_registry_url']}) "
        f"— git-LFS registry{' (private)' if m['registry_private'] else ''}; a round is "
        "pushed there only when it passes the gate.",
        "",
        "## Summary",
        "",
        f"- Rounds run: **{s['total_rounds']}** ({s['graded_rounds']} graded)",
        f"- Rounds that passed the gate (proposed): **{s['proposed']}**",
        f"- Best canary Δ so far: **{s['best_canary_delta_pp']} pp**",
        f"- Latest round: seed `{s['latest_seed']}` at {s['latest_started']}",
        f"- Generated: {data['generated_at']}",
        "",
        "## Rounds (most recent first)",
        "",
        "| Seed | Started | Holdout Δpp (p) | Canary Δpp (p) | Gate |",
        "|---:|:--|:--|:--|:--|",
    ]
    for r in reversed(data["rounds"]):
        h, c = r["holdout"], r["canary"]
        if r["degenerate"]:
            gatecell = "⚪ degenerate"
        elif r["proposed"]:
            gatecell = "🟢 proposed"
        else:
            gatecell = "🔴 held"
        lines.append(
            f"| {r['seed']} | {r['started']} | "
            f"{h['delta_pp']:+} (p={h['p_value']}) | "
            f"{c['delta_pp']:+} (p={c['p_value']}) | {gatecell} |")
    lines += [
        "",
        "Gate rule: a round is **proposed** only when the tuned model beats base by "
        "**≥5pp AND p<0.05** (exact McNemar) on **both** holdout and canary.",
        "",
    ]
    return "\n".join(lines)


# Self-contained dashboard: no external assets, re-fetches data.json on a timer so
# an open tab tracks the loop in near-real-time. Kept static (all dynamic content
# comes from data.json) so only data.json + README churn each round.
_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RSI MODEL_TUNER — live status</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1117; --card:#1a1d27; --fg:#e7e9ee;
          --mut:#9aa0ac; --line:#2a2e3a; --go:#3fb950; --hold:#f85149; --deg:#6e7681; }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f6f8fa; --card:#fff; --fg:#1f2328; --mut:#656d76; --line:#d0d7de; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:960px; margin:0 auto; padding:24px 18px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--mut); margin:0 0 20px; font-size:13.5px; }
  .sub a { color:inherit; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; }
  .card .k { color:var(--mut); font-size:12px; text-transform:uppercase;
             letter-spacing:.04em; }
  .card .v { font-size:26px; font-weight:650; margin-top:4px; }
  .chart { display:flex; align-items:flex-end; gap:3px; height:120px; padding:10px 0;
           border-bottom:1px solid var(--line); margin-bottom:18px; overflow-x:auto; }
  .bar { flex:0 0 auto; width:14px; border-radius:3px 3px 0 0; background:var(--deg);
         min-height:2px; }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13.5px; }
  th,td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
          white-space:nowrap; }
  th { color:var(--mut); font-weight:600; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .pill { display:inline-block; padding:2px 9px; border-radius:20px; font-size:12px;
          font-weight:600; }
  .go   { background:rgba(63,185,80,.15);  color:var(--go); }
  .hold { background:rgba(248,81,73,.15);   color:var(--hold); }
  .deg  { background:rgba(110,118,129,.18); color:var(--deg); }
  .foot { color:var(--mut); font-size:12.5px; margin-top:26px; }
  .live { display:inline-block; width:8px; height:8px; border-radius:50%;
          background:var(--go); margin-right:6px; animation:pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
</style></head>
<body><div class="wrap">
  <h1><span class="live"></span>RSI MODEL_TUNER — live status</h1>
  <p class="sub" id="sub">Loading…</p>
  <div class="cards" id="cards"></div>
  <p class="foot" id="models" style="margin-top:-8px;margin-bottom:18px"></p>
  <div class="chart" id="chart" title="Canary Δpp per round"></div>
  <div class="tablewrap"><table id="tbl">
    <thead><tr><th>Seed</th><th>Started</th><th>Holdout Δpp</th><th>p</th>
      <th>Canary Δpp</th><th>p</th><th>Gate</th></tr></thead>
    <tbody id="rows"></tbody></table></div>
  <p class="foot" id="foot"></p>
</div>
<script>
const fmt = v => (v==null?'—':(v>0?'+':'')+v);
function pill(r){ if(r.degenerate) return '<span class="pill deg">degenerate</span>';
  return r.proposed ? '<span class="pill go">proposed</span>'
                    : '<span class="pill hold">held</span>'; }
async function load(){
  try{
    const d = await (await fetch('./data.json?t='+Date.now())).json();
    const s = d.summary;
    document.getElementById('sub').innerHTML =
      'Shadow-only self-improvement loop. Tuned Qwen vs. untuned base ('+
      '<code>'+d.base_model+'</code>) on a frozen holdout+canary split, McNemar gate. '+
      'Passing rounds are <b>proposed, never auto-promoted</b>. '+
      '<a href="'+d.doc_url+'">How it works ↗</a>';
    const m = d.models;
    document.getElementById('models').innerHTML =
      '<b>Models:</b> base <a href="'+m.base_hf_url+'"><code>'+m.base_hf+'</code></a> '+
      '(Ollama <code>'+m.base_ollama+'</code>) · tuned <code>rsi-qwen:round</code> · '+
      'published to <a href="'+m.tuned_registry_url+'"><code>'+m.tuned_registry+'</code></a>'+
      (m.registry_private ? ' (private)' : '');
    const cards = [
      ['Rounds', s.total_rounds],
      ['Proposed', s.proposed],
      ['Best canary Δ', fmt(s.best_canary_delta_pp)+' pp'],
      ['Latest seed', s.latest_seed ?? '—'],
    ];
    document.getElementById('cards').innerHTML = cards.map(c=>
      '<div class="card"><div class="k">'+c[0]+'</div><div class="v">'+c[1]+'</div></div>').join('');
    // canary Δ bars (chronological), green if that round proposed
    const graded = d.rounds.filter(r=>!r.degenerate);
    const mx = Math.max(6, ...graded.map(r=>Math.abs(r.canary.delta_pp)));
    document.getElementById('chart').innerHTML = graded.map(r=>{
      const h = Math.round(Math.abs(r.canary.delta_pp)/mx*100);
      const col = r.proposed ? 'var(--go)' : (r.canary.delta_pp<0?'var(--hold)':'var(--deg)');
      return '<div class="bar" style="height:'+h+'%;background:'+col+'" title="seed '+
             r.seed+': '+fmt(r.canary.delta_pp)+'pp"></div>'; }).join('');
    document.getElementById('rows').innerHTML = d.rounds.slice().reverse().map(r=>
      '<tr><td>'+r.seed+'</td><td>'+r.started+'</td>'+
      '<td class="num">'+fmt(r.holdout.delta_pp)+'</td><td class="num">'+r.holdout.p_value+'</td>'+
      '<td class="num">'+fmt(r.canary.delta_pp)+'</td><td class="num">'+r.canary.p_value+'</td>'+
      '<td>'+pill(r)+'</td></tr>').join('');
    document.getElementById('foot').textContent =
      'Generated '+d.generated_at+' · auto-refreshes every 5 min · '+
      s.graded_rounds+' graded of '+s.total_rounds+' rounds';
  }catch(e){ document.getElementById('sub').textContent = 'Could not load data.json: '+e; }
}
load(); setInterval(load, 300000);
</script>
</body></html>
"""


_TEMPLATE_VERSION = "2"  # bump when index.html / README layout changes


def _round_signature(data):
    """Everything that should trigger a republish — the full payload EXCEPT
    generated_at (so an idle tick with a newer clock is a no-op and does not spam
    gh-pages), plus a template version so layout/link changes also republish."""
    payload = {k: v for k, v in data.items() if k != "generated_at"}
    return json.dumps({"v": _TEMPLATE_VERSION, "data": payload}, sort_keys=True)


def _unchanged(status_dir, data):
    path = os.path.join(status_dir, "data.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as fh:
            return _round_signature(json.load(fh)) == _round_signature(data)
    except Exception:
        return False  # unreadable / old schema -> rewrite


def _doc_source():
    """The how-it-works doc to bundle onto gh-pages. RSI_STATUS_DOC wins; else the
    repo's docs/rsi-loop.md if this module runs from the checkout. None -> skip
    (the dashboard link still points at the gh-pages copy from a prior publish)."""
    env = os.environ.get("RSI_STATUS_DOC")
    if env:
        return env if os.path.exists(env) else None
    cand = os.path.join(os.path.dirname(_HERE), "docs", "rsi-loop.md")
    return cand if os.path.exists(cand) else None


def write_artifacts(status_dir, data):
    with open(os.path.join(status_dir, "data.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    with open(os.path.join(status_dir, "index.html"), "w") as fh:
        fh.write(_HTML)
    with open(os.path.join(status_dir, "README.md"), "w") as fh:
        fh.write(render_readme(data))
    # Bundle the explainer so the "How it works" link resolves off this branch,
    # independent of what's merged to main.
    doc = _doc_source()
    if doc:
        shutil.copy(doc, os.path.join(status_dir, "rsi-loop.md"))
    # Pages must not run Jekyll (it would hide files/dirs starting with _ or .).
    open(os.path.join(status_dir, ".nojekyll"), "a").close()


def _run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed ({r.returncode}): "
                           f"{r.stderr.strip() or r.stdout.strip()}")
    return (r.stdout or "").strip()


def commit_and_push(status_dir):
    """Commit the artifacts and push to the Pages branch. No-op (returns None) when
    nothing changed, so the timer can run often without spamming empty commits."""
    dot_git = os.path.join(status_dir, ".git")
    if not (os.path.isdir(dot_git) or os.path.isfile(dot_git)):
        raise RuntimeError(f"status dir not a git checkout: {status_dir} "
                           f"(clone the fork on branch {STATUS_BRANCH})")
    _run(["git", "add", "-A"], cwd=status_dir)
    # Nothing staged to commit -> up to date, skip.
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=status_dir).returncode == 0:
        return None
    _run(["git", "commit", "-m", "status: refresh RSI MODEL_TUNER dashboard"], cwd=status_dir)
    _run(["git", "push", STATUS_REMOTE, f"HEAD:{STATUS_BRANCH}"], cwd=status_dir)
    return _run(["git", "rev-parse", "HEAD"], cwd=status_dir)


def publish_now(status_dir=None, generated_at=None,
                base_model=None, push=True):
    """Collect -> render -> write -> (optionally) push. Returns a result dict."""
    status_dir = status_dir or _status_dir()
    generated_at = generated_at or datetime.datetime.now(datetime.timezone.utc)\
        .strftime("%Y-%m-%d %H:%M UTC")
    base_model = base_model or os.environ.get("RSI_BASE_TAG", "qwen2.5:3b")
    os.makedirs(status_dir, exist_ok=True)
    rounds = collect_rounds()
    data = build_data(rounds, generated_at, base_model)
    result = {"rounds": len(rounds), "status_dir": status_dir,
              "proposed": data["summary"]["proposed"]}
    # Idle-tick fast path: identical round data -> leave files (and their clock)
    # untouched so a frequent timer never produces an empty commit.
    if _unchanged(status_dir, data):
        result["unchanged"] = True
        result["commit"] = None
        return result
    write_artifacts(status_dir, data)
    if push:
        result["commit"] = commit_and_push(status_dir)
    return result


def maybe_publish_status(report=None):
    """Per-round hook: refresh the public dashboard iff RSI_PUBLISH_STATUS=1.
    Isolated so a publish failure never crashes a round. `report` is unused (the
    dashboard is rebuilt from the warehouse, which already has this round)."""
    if os.environ.get("RSI_PUBLISH_STATUS") != "1":
        return None
    try:
        res = publish_now()
        print(f"status: dashboard refreshed ({res['rounds']} rounds, "
              f"commit={res.get('commit') or 'no-change'})", file=sys.stderr)
        return res
    except Exception as e:  # visibility must never take down a round
        print(f"status: publish failed: {e!r}", file=sys.stderr)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Publish the RSI status dashboard.")
    ap.add_argument("--no-push", action="store_true",
                    help="render artifacts into the status dir but do not commit/push")
    ap.add_argument("--status-dir", default=None)
    args = ap.parse_args(argv)
    res = publish_now(status_dir=args.status_dir, push=not args.no_push)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
