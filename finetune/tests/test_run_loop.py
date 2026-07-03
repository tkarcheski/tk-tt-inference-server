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
