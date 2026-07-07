import publish_ollama as po


def _report(proposed=True):
    return {"tuned_id": "abcd1234-0000-0000-0000-00000000abcd",
            "lora_hash": "deadbeefcafebabe" + "0" * 48, "seed": 1042,
            "holdout": {"delta_pp": 12.0}, "canary": {"delta_pp": 7.0},
            "proposed": proposed}


def _mock_run(calls, rc=0):
    def run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        return type("R", (), {"returncode": rc, "stdout": "ok", "stderr": "boom"})()
    return run


class _Cur:
    def __init__(self, fetch=None):
        self.fetch = fetch
        self.executed = []
        self.rowcount = 1
    def execute(self, sql, params=None):
        self.executed.append((sql, params))
    def fetchone(self):
        return self.fetch
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Conn:
    def __init__(self, cur):
        self._cur = cur
    def cursor(self): return self._cur
    def commit(self): pass
    def close(self): pass


def test_baseline_version_format():
    assert po.baseline_version(_report()) == "v1042-deadbeef"


def test_target_is_the_tkarcheski_registry():
    assert po.OLLAMA_TARGET == "tkarcheski/rsi-qwen:3b-latest"


def test_current_champion_returns_ref_or_none(monkeypatch):
    monkeypatch.setattr(po.rsi_common, "connect",
                        lambda: _Conn(_Cur(fetch=("tkarcheski/rsi-qwen:3b-latest",))))
    assert po.current_champion() == "tkarcheski/rsi-qwen:3b-latest"
    monkeypatch.setattr(po.rsi_common, "connect", lambda: _Conn(_Cur(fetch=None)))
    assert po.current_champion() is None


def test_current_champion_degrades_on_db_error(monkeypatch):
    def boom(): raise RuntimeError("db down")
    monkeypatch.setattr(po.rsi_common, "connect", boom)
    assert po.current_champion() is None       # never breaks the round's base arm


def test_push_to_ollama_cp_then_push(monkeypatch):
    calls = []
    monkeypatch.setattr(po.subprocess, "run", _mock_run(calls))
    ref = po.push_to_ollama("rsi-qwen:round")
    assert ref == "tkarcheski/rsi-qwen:3b-latest"
    assert calls[0] == ["ollama", "cp", "rsi-qwen:round", "tkarcheski/rsi-qwen:3b-latest"]
    assert calls[1] == ["ollama", "push", "tkarcheski/rsi-qwen:3b-latest"]


def test_push_to_ollama_raises_on_failure(monkeypatch):
    monkeypatch.setattr(po.subprocess, "run", _mock_run([], rc=1))
    try:
        po.push_to_ollama("rsi-qwen:round")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "failed" in str(e)


def test_record_baseline_demotes_old_then_inserts(monkeypatch):
    cur = _Cur()
    monkeypatch.setattr(po.rsi_common, "connect", lambda: _Conn(cur))
    assert po.record_baseline(_report(), "v1042-deadbeef", "tkarcheski/rsi-qwen:3b-latest") is True
    sqls = " ".join(s for s, _ in cur.executed).lower()
    assert "update rsi.baselines set is_current=false" in sqls   # demote previous
    assert "insert into rsi.baselines" in sqls                    # add new current
    # the insert carries the ollama ref + tuned id
    ins = next(p for s, p in cur.executed if "insert" in s.lower())
    assert "tkarcheski/rsi-qwen:3b-latest" in ins


def test_maybe_push_noop_when_not_proposed(monkeypatch):
    monkeypatch.setenv("RSI_PUSH_OLLAMA", "1")
    calls = []
    monkeypatch.setattr(po.subprocess, "run", _mock_run(calls))
    assert po.maybe_push_ollama(_report(proposed=False)) is None
    assert calls == []                                            # nothing pushed


def test_maybe_push_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("RSI_PUSH_OLLAMA", raising=False)
    calls = []
    monkeypatch.setattr(po.subprocess, "run", _mock_run(calls))
    assert po.maybe_push_ollama(_report(proposed=True)) is None
    assert calls == []


def test_maybe_push_pushes_and_records_on_pass(monkeypatch):
    monkeypatch.setenv("RSI_PUSH_OLLAMA", "1")
    calls = []
    monkeypatch.setattr(po.subprocess, "run", _mock_run(calls))
    recorded = {}
    monkeypatch.setattr(po, "record_baseline",
                        lambda r, v, ref: recorded.update(v=v, ref=ref) or True)
    out = po.maybe_push_ollama(_report(), tuned_tag="rsi-qwen:round")
    assert out == {"version": "v1042-deadbeef", "ollama_ref": "tkarcheski/rsi-qwen:3b-latest"}
    assert ["ollama", "push", "tkarcheski/rsi-qwen:3b-latest"] in calls
    assert recorded["v"] == "v1042-deadbeef"


def test_champion_does_not_roll_when_push_fails(monkeypatch):
    # If the ollama push fails, the baseline must NOT roll — the cluster never got it.
    monkeypatch.setenv("RSI_PUSH_OLLAMA", "1")
    monkeypatch.setattr(po.subprocess, "run", _mock_run([], rc=1))   # push fails
    recorded = []
    monkeypatch.setattr(po, "record_baseline", lambda *a: recorded.append(1))
    assert po.maybe_push_ollama(_report()) is None
    assert recorded == []                                            # baseline unchanged
