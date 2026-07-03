import run_loop as rl

def test_new_experiment_inserts_and_returns_uuid():
    eid = rl.new_experiment("model_tuner_round", provider_variant="ollama",
                            serving_runtime="ollama", base_model_sha="deadbeef")
    assert len(eid) == 36
    with rl.rsi_common.connect() as c, c.cursor() as cur:
        cur.execute("select intent from rsi.experiments where experiment_id=%s", (eid,))
        assert cur.fetchone()[0] == "model_tuner_round"

def test_shadow_mode_never_promotes(monkeypatch):
    monkeypatch.setenv("RSI_MODE", "shadow")
    assert rl.can_promote() is False

def _pr_calls(monkeypatch, proposed, mode, open_pr, agents="true"):
    calls = []
    monkeypatch.setattr(rl.subprocess, "run",
                        lambda *a, **k: calls.append(a[0]) or type("R", (), {"returncode": 0})())
    monkeypatch.setenv("RSI_MODE", mode)
    monkeypatch.setenv("RSI_AGENTS_ENABLED", agents)
    if open_pr is None:
        monkeypatch.delenv("RSI_OPEN_PR", raising=False)
    else:
        monkeypatch.setenv("RSI_OPEN_PR", open_pr)
    rl.maybe_open_pr({"proposed": proposed, "tuned_id": "abcd1234deadbeef"})
    return calls

def test_pr_opens_only_when_proposed_shadow_and_opted_in(monkeypatch):
    calls = _pr_calls(monkeypatch, proposed=True, mode="shadow", open_pr="1")
    assert len(calls) == 1 and calls[0][:3] == ["gh", "pr", "create"] and "--draft" in calls[0]

def test_no_pr_when_not_proposed(monkeypatch):
    assert _pr_calls(monkeypatch, proposed=False, mode="shadow", open_pr="1") == []

def test_no_pr_without_opt_in(monkeypatch):
    assert _pr_calls(monkeypatch, proposed=True, mode="shadow", open_pr=None) == []

def test_no_pr_in_live_mode(monkeypatch):
    # can_promote() True in live → draft-PR proposal path suppressed
    assert _pr_calls(monkeypatch, proposed=True, mode="live", open_pr="1") == []
