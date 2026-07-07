import json

import publish_status as ps


# ---- fixtures ---------------------------------------------------------------

def _round(seed, h_delta, h_p, c_delta, c_p, h_n=10, c_n=10):
    def pool(delta, p, n):
        return {"n": n, "base_pass": 50.0, "tuned_pass": 50.0 + delta,
                "delta_pp": delta, "p_value": p,
                "passes": bool(delta >= 5.0 and p < 0.05)}
    r = {"seed": seed, "started": f"2026-07-05T0{seed%9}:00:00",
         "tuned_id": f"t{seed}", "base_id": f"b{seed}",
         "lora_hash": "abcd1234", "train_pool_hash": "poolhash",
         "holdout": pool(h_delta, h_p, h_n), "canary": pool(c_delta, c_p, c_n)}
    r["degenerate"] = r["holdout"]["n"] == 0 or r["canary"]["n"] == 0
    r["proposed"] = bool(r["holdout"]["passes"] and r["canary"]["passes"]
                         and not r["degenerate"])
    return r


def _rounds():
    return [
        _round(1001, -3.0, 1.0, -15.0, 0.99),        # held
        _round(1002, 22.0, 0.03, 7.0, 0.04),         # proposed (both pass)
        _round(1003, 12.0, 0.2, -5.0, 0.9),          # held
        _round(1004, 0.0, 1.0, 0.0, 1.0, c_n=0),     # degenerate (canary n=0)
    ]


class _Cur:
    """Fake cursor: returns canned rows for the two collect_rounds queries."""
    def __init__(self, meta, outcomes):
        self._results = [meta, outcomes]
        self._i = -1
    def execute(self, *a, **k):
        self._i += 1
    def fetchall(self):
        return self._results[self._i]
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Conn:
    def __init__(self, cur): self._cur = cur
    def cursor(self): return self._cur
    def close(self): pass


# ---- collect_rounds ---------------------------------------------------------

def test_collect_rounds_pairs_and_derives(monkeypatch):
    meta = [("t1", "b1", 1001, "adapterhashXXXX", "poolhashYYYY", "2026-07-05T05:00:00")]
    # base_pass, tuned_pass per matched test, per pool
    outcomes = [
        ("t1", "holdout", False, True), ("t1", "holdout", False, True),
        ("t1", "holdout", True, True),  ("t1", "holdout", True, True),
        ("t1", "canary",  True, False),                       # tuned regressed
    ]
    monkeypatch.setattr(ps.rsi_common, "connect", lambda: _Conn(_Cur(meta, outcomes)))
    rounds = ps.collect_rounds()
    assert len(rounds) == 1
    r = rounds[0]
    assert r["seed"] == 1001
    assert r["lora_hash"] == "adapterh"           # truncated to 8
    assert r["holdout"]["n"] == 4 and r["holdout"]["tuned_pass"] == 100.0
    assert r["canary"]["n"] == 1
    assert r["degenerate"] is False
    assert r["proposed"] is False                  # canary did not improve


def test_collect_rounds_marks_degenerate_when_a_pool_empty(monkeypatch):
    meta = [("t1", "b1", 1001, "h", "p", "2026-07-05T05:00:00")]
    outcomes = [("t1", "holdout", True, True)]     # canary has no matched rows
    monkeypatch.setattr(ps.rsi_common, "connect", lambda: _Conn(_Cur(meta, outcomes)))
    r = ps.collect_rounds()[0]
    assert r["canary"]["n"] == 0
    assert r["degenerate"] is True
    assert r["proposed"] is False


# ---- build_data summary -----------------------------------------------------

def test_build_data_summary_counts():
    data = ps.build_data(_rounds(), "2026-07-05 12:00 UTC", "qwen2.5:3b")
    s = data["summary"]
    assert s["total_rounds"] == 4
    assert s["graded_rounds"] == 3                 # one degenerate excluded
    assert s["proposed"] == 1                      # only seed 1002
    assert s["best_canary_delta_pp"] == 7.0        # best among graded
    assert s["latest_seed"] == 1004
    assert data["shadow_only"] is True


# ---- signature ignores the clock (idle-tick no-op) --------------------------

def test_round_signature_ignores_generated_at():
    a = ps.build_data(_rounds(), "2026-07-05 12:00 UTC", "qwen2.5:3b")
    b = ps.build_data(_rounds(), "2099-01-01 00:00 UTC", "qwen2.5:3b")  # different clock
    assert ps._round_signature(a) == ps._round_signature(b)


def test_round_signature_changes_when_a_round_lands():
    a = ps.build_data(_rounds(), "t", "qwen2.5:3b")
    more = _rounds() + [_round(1005, 9.0, 0.01, 9.0, 0.01)]
    b = ps.build_data(more, "t", "qwen2.5:3b")
    assert ps._round_signature(a) != ps._round_signature(b)


# ---- README leaderboard -----------------------------------------------------

def test_readme_has_pills_gate_rule_and_no_secrets():
    data = ps.build_data(_rounds(), "t", "qwen2.5:3b")
    md = ps.render_readme(data)
    assert "🟢 proposed" in md and "🔴 held" in md and "⚪ degenerate" in md
    assert "≥5pp AND p<0.05" in md
    assert "never auto-promoted" in md
    # never leak connection secrets
    for secret in ("changeme", "password", "DATABASE_URL", "5434"):
        assert secret not in md


def test_readme_links_resolve_and_document_the_models():
    data = ps.build_data(_rounds(), "t", "qwen2.5:3b")
    md = ps.render_readme(data)
    # the how-it-works link must NOT point at main (rsi-loop.md isn't there -> 404)
    assert "blob/main/docs/rsi-loop.md" not in md
    assert f"blob/{ps.STATUS_BRANCH}/rsi-loop.md" in md    # bundled on this branch
    # models are documented: base (HF + ollama) and the tuned registry
    assert "## Models" in md
    assert ps.BASE_HF in md and ps.BASE_HF_URL in md
    assert ps.TUNED_REGISTRY in md and ps.TUNED_REGISTRY_URL in md


def test_build_data_exposes_doc_url_and_models():
    data = ps.build_data(_rounds(), "t", "qwen2.5:3b")
    assert data["doc_url"].endswith(f"blob/{ps.STATUS_BRANCH}/rsi-loop.md")
    m = data["models"]
    assert m["base_hf"] == ps.BASE_HF
    assert m["tuned_registry"] == ps.TUNED_REGISTRY
    assert m["registry_private"] is True


def test_template_version_change_forces_republish(monkeypatch):
    # A layout-only change (same rounds) must still republish, else fixes never ship.
    data = ps.build_data(_rounds(), "t", "qwen2.5:3b")
    sig_before = ps._round_signature(data)
    monkeypatch.setattr(ps, "_TEMPLATE_VERSION", ps._TEMPLATE_VERSION + "-next")
    assert ps._round_signature(data) != sig_before


# ---- dashboard is fully self-contained --------------------------------------

def test_html_has_no_external_assets():
    html = ps._HTML
    # no CDN scripts, external stylesheets, remote fonts, or absolute-URL fetches
    for bad in ("src=\"http", "src='http", "href=\"http", "cdn.", "googleapis",
                "unpkg", "jsdelivr", "fetch('http", 'fetch("http'):
        assert bad not in html
    assert "fetch('./data.json" in html          # same-origin relative fetch only
    assert "setInterval(load" in html            # auto-refresh present


# ---- publish_now write + idle-tick no-op ------------------------------------

def test_publish_now_writes_then_skips_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "collect_rounds", _rounds)
    d = str(tmp_path)
    r1 = ps.publish_now(status_dir=d, generated_at="2026-07-05 12:00 UTC",
                        base_model="qwen2.5:3b", push=False)
    assert r1.get("unchanged") is not True
    for f in ("data.json", "index.html", "README.md", ".nojekyll"):
        assert (tmp_path / f).exists()
    # second run, same rounds, later clock -> no rewrite, no push
    r2 = ps.publish_now(status_dir=d, generated_at="2099-01-01 00:00 UTC",
                        base_model="qwen2.5:3b", push=False)
    assert r2["unchanged"] is True
    # data.json still shows the ORIGINAL generated_at (not rewritten)
    assert json.load(open(tmp_path / "data.json"))["generated_at"] == "2026-07-05 12:00 UTC"


# ---- commit_and_push git sequence -------------------------------------------

def _mock_run(calls, diff_rc=1):
    def run(cmd, cwd=None, capture_output=True, text=True):
        calls.append(cmd)
        rc = diff_rc if cmd[:3] == ["git", "diff", "--cached"] else 0
        return type("R", (), {"returncode": rc, "stdout": "sha999", "stderr": ""})()
    return run


def test_write_artifacts_bundles_doc(tmp_path, monkeypatch):
    doc = tmp_path / "src.md"; doc.write_text("# how it works\n")
    monkeypatch.setenv("RSI_STATUS_DOC", str(doc))
    out = tmp_path / "out"; out.mkdir()
    ps.write_artifacts(str(out), ps.build_data(_rounds(), "t", "qwen2.5:3b"))
    assert (out / "rsi-loop.md").read_text() == "# how it works\n"   # link target exists


def test_commit_and_push_pushes_to_pages_branch(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    calls = []
    monkeypatch.setattr(ps.subprocess, "run", _mock_run(calls, diff_rc=1))  # 1 = has changes
    sha = ps.commit_and_push(str(tmp_path))
    assert sha == "sha999"
    assert ["git", "add", "-A"] in calls
    assert any(c[:2] == ["git", "commit"] for c in calls)
    push = next(c for c in calls if c[:2] == ["git", "push"])
    assert push[-1] == f"HEAD:{ps.STATUS_BRANCH}"     # Pages branch, never main
    assert "main" not in push and "master" not in push


def test_commit_and_push_noop_when_clean(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    calls = []
    monkeypatch.setattr(ps.subprocess, "run", _mock_run(calls, diff_rc=0))  # 0 = no changes
    assert ps.commit_and_push(str(tmp_path)) is None
    assert not any(c[:2] == ["git", "push"] for c in calls)   # nothing pushed


def test_commit_and_push_requires_git_checkout(tmp_path):
    try:
        ps.commit_and_push(str(tmp_path))       # no .git
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "not a git checkout" in str(e)


# ---- per-round hook is opt-in ----------------------------------------------

def test_maybe_publish_status_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("RSI_PUBLISH_STATUS", raising=False)
    called = []
    monkeypatch.setattr(ps, "publish_now", lambda *a, **k: called.append(1))
    assert ps.maybe_publish_status() is None
    assert called == []


def test_maybe_publish_status_runs_when_enabled(monkeypatch):
    monkeypatch.setenv("RSI_PUBLISH_STATUS", "1")
    monkeypatch.setattr(ps, "publish_now",
                        lambda *a, **k: {"rounds": 3, "commit": "abc"})
    out = ps.maybe_publish_status()
    assert out["rounds"] == 3


def test_maybe_publish_status_swallows_errors(monkeypatch):
    monkeypatch.setenv("RSI_PUBLISH_STATUS", "1")
    def boom(*a, **k): raise RuntimeError("db down")
    monkeypatch.setattr(ps, "publish_now", boom)
    assert ps.maybe_publish_status() is None      # never propagates
