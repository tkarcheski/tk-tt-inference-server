import run_loop as rl

def test_round_config_is_deterministic_and_varies_per_round():
    # Same index → same config; seed advances every round; grid cycles.
    assert rl.round_config(0) == rl.round_config(0)
    assert rl.round_config(0)["SEED"] == "1000"
    assert rl.round_config(1)["SEED"] == "1001"
    assert rl.round_config(0) != rl.round_config(1)          # distinct experiments
    assert rl.round_config(0)["LORA_R"] == rl.round_config(4)["LORA_R"]  # grid cycles at len 4
    assert rl.round_config(4)["SEED"] == "1004"              # but seed keeps advancing
    for i in range(4):
        c = rl.round_config(i)
        assert {"LORA_R", "LORA_ALPHA", "LR", "MAX_STEPS", "SEED"} <= set(c)
        # Guard: MAX_STEPS stays sized to the ~136-example pool (~17 steps/epoch).
        # >~60 steps overfits (memorizes → hurts generalization); keep it low.
        assert int(c["MAX_STEPS"]) <= 60

def test_should_stop_kill_switch(monkeypatch, tmp_path):
    monkeypatch.delenv("RSI_KILL", raising=False)
    sf = tmp_path / "stop"
    assert rl.should_stop(str(sf)) is False
    sf.write_text("")                       # stop-file present
    assert rl.should_stop(str(sf)) is True
    sf.unlink()
    monkeypatch.setenv("RSI_KILL", "1")      # env kill switch
    assert rl.should_stop(str(sf)) is True

def test_run_forever_stops_on_stop_file(monkeypatch, tmp_path):
    monkeypatch.delenv("RSI_KILL", raising=False)
    sf = tmp_path / "stop"; sf.write_text("")   # already stopped
    calls = []
    n = rl.run_forever(run_round_fn=lambda **k: calls.append(1) or {},
                       sleep_fn=lambda s: None, smoke=True, sleep_s=0, stop_file=str(sf),
                       start_index=0)
    assert n == 0 and calls == []               # never ran a round

def test_run_forever_runs_max_rounds_then_exits(monkeypatch, tmp_path):
    monkeypatch.delenv("RSI_KILL", raising=False)
    calls = []
    n = rl.run_forever(run_round_fn=lambda **k: calls.append(k) or {"proposed": False},
                       sleep_fn=lambda s: None, smoke=True, sleep_s=0,
                       stop_file=str(tmp_path / "nope"), max_rounds=3, start_index=0)
    assert n == 3 and len(calls) == 3

def test_run_forever_continues_after_a_failed_round(monkeypatch, tmp_path):
    # A round that raises must NOT kill the loop; later rounds still run.
    monkeypatch.delenv("RSI_KILL", raising=False)
    seen = []
    def flaky(**k):
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("boom")   # first round explodes
        return {"proposed": False}
    n = rl.run_forever(run_round_fn=flaky, sleep_fn=lambda s: None, smoke=True,
                       sleep_s=0, stop_file=str(tmp_path / "nope"), max_rounds=3, start_index=0)
    assert n == 3 and len(seen) == 3       # survived the failure, ran all 3

def test_run_forever_smoke_caps_max_steps(monkeypatch, tmp_path):
    import os
    monkeypatch.delenv("RSI_KILL", raising=False)
    seen = []
    rl.run_forever(run_round_fn=lambda **k: seen.append(os.environ.get("MAX_STEPS")) or {},
                   sleep_fn=lambda s: None, smoke=True, sleep_s=0,
                   stop_file=str(tmp_path / "nope"), max_rounds=2, start_index=0)
    assert seen == ["3", "3"]                     # tiny, not the grid's 200/300

def test_run_forever_real_uses_grid_max_steps(monkeypatch, tmp_path):
    import os
    monkeypatch.delenv("RSI_KILL", raising=False)
    seen = []
    rl.run_forever(run_round_fn=lambda **k: seen.append(os.environ.get("MAX_STEPS")) or {},
                   sleep_fn=lambda s: None, smoke=False, sleep_s=0,
                   stop_file=str(tmp_path / "nope"), max_rounds=2, start_index=0)
    assert seen == [rl.round_config(0)["MAX_STEPS"], rl.round_config(1)["MAX_STEPS"]]

def test_filter_eval_suites_drops_only_named_suites_from_eval():
    splits = {"train": ["a", "b"], "holdout": ["context_window", "extraction"],
              "canary": ["hallucination", "legal"]}
    out, dropped = rl.filter_eval_suites(splits, ["context_window"])
    assert out["holdout"] == ["extraction"]           # skipped one dropped
    assert out["canary"] == ["hallucination", "legal"]  # untouched
    assert out["train"] == ["a", "b"]                  # firewall/train untouched
    assert dropped == {"holdout": ["context_window"]}

def test_filter_eval_suites_noop_when_skip_empty():
    splits = {"holdout": ["extraction"], "canary": ["legal"]}
    out, dropped = rl.filter_eval_suites(splits, [])
    assert out == splits and dropped == {}

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


class _FakeCur:
    def __init__(self, val): self._val = val
    def execute(self, *a, **k): pass
    def fetchone(self): return (self._val,)
    def __enter__(self): return self
    def __exit__(self, *a): return False

class _FakeConn:
    def __init__(self, val): self._val = val
    def cursor(self): return _FakeCur(self._val)
    def close(self): pass

def test_starting_index_resumes_from_warehouse_max_seed(monkeypatch):
    monkeypatch.setattr(rl.rsi_common, "connect", lambda: _FakeConn(1013))
    assert rl.starting_index() == 14        # 1013 - 1000 + 1 (no duplicate seeds)

def test_starting_index_zero_when_no_rounds(monkeypatch):
    monkeypatch.setattr(rl.rsi_common, "connect", lambda: _FakeConn(None))
    assert rl.starting_index() == 0

def test_starting_index_degrades_gracefully_on_db_error(monkeypatch):
    def boom(): raise RuntimeError("db down")
    monkeypatch.setattr(rl.rsi_common, "connect", boom)
    assert rl.starting_index() == 0         # never crashes the supervisor

def test_run_forever_resumes_seed_sequence_from_starting_index(monkeypatch, tmp_path):
    # With no start_index override, the loop resumes from the warehouse: the first
    # round after a restart must use the NEXT seed, not re-run seed 1000.
    monkeypatch.delenv("RSI_KILL", raising=False)
    monkeypatch.setattr(rl, "starting_index", lambda: 14)
    seen = []
    import os
    rl.run_forever(run_round_fn=lambda **k: seen.append(os.environ.get("SEED")) or {},
                   sleep_fn=lambda s: None, smoke=False, sleep_s=0,
                   stop_file=str(tmp_path / "nope"), max_rounds=2)
    assert seen == ["1014", "1015"]         # resumed, not reset to 1000

def test_base_arm_uses_rolling_champion_when_promoted(monkeypatch):
    # Once a champion is promoted, the base arm is it (self-improving ratchet).
    monkeypatch.setattr(rl.publish_ollama, "current_champion",
                        lambda: "tkarcheski/rsi-qwen:3b-latest")
    assert rl.base_arm_tag() == "tkarcheski/rsi-qwen:3b-latest"

def test_base_arm_falls_back_to_stock_base(monkeypatch):
    monkeypatch.setattr(rl.publish_ollama, "current_champion", lambda: None)
    monkeypatch.setenv("RSI_BASE_TAG", "qwen2.5:3b")
    assert rl.base_arm_tag() == "qwen2.5:3b"
