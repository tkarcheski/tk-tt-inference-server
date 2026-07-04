import publish as pub


def _report(proposed=True):
    return {"tuned_id": "abcd1234-0000-0000-0000-00000000abcd",
            "lora_hash": "deadbeefcafebabe" + "0" * 48, "seed": 1007,
            "train_pool_hash": "poolhash", "base_id": "b",
            "holdout": {"delta_pp": 22.5, "p_value": 0.03, "passes": True},
            "canary": {"delta_pp": 7.0, "p_value": 0.04, "passes": True},
            "proposed": proposed}


def _mock_run(calls, rc=0):
    def run(cmd, cwd=None, capture_output=True, text=True):
        calls.append({"cmd": cmd, "cwd": cwd})
        return type("R", (), {"returncode": rc, "stdout": "sha123abc", "stderr": ""})()
    return run


def test_publish_version_format():
    assert pub.publish_version(_report()) == "v1007-deadbeef"


def test_release_repo_is_the_fork_never_upstream():
    assert pub.FORK_REPO == "tkarcheski/tk-tt-inference-server"
    assert "tenstorrent" not in pub.FORK_REPO


def test_maybe_publish_noop_when_not_proposed(monkeypatch):
    monkeypatch.setenv("RSI_PUBLISH", "1")
    calls = []
    monkeypatch.setattr(pub.subprocess, "run", _mock_run(calls))
    assert pub.maybe_publish(_report(proposed=False), "/tmp/x") is None
    assert calls == []


def test_maybe_publish_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("RSI_PUBLISH", raising=False)
    calls = []
    monkeypatch.setattr(pub.subprocess, "run", _mock_run(calls))
    assert pub.maybe_publish(_report(proposed=True), "/tmp/x") is None
    assert calls == []


def test_maybe_publish_builds_submodule_and_release(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PUBLISH", "1")
    sub = tmp_path / "ollama-models"
    (sub / ".git").mkdir(parents=True)              # looks initialized
    monkeypatch.setenv("RSI_MODELS_SUBMODULE", str(sub))
    merged = tmp_path / "merged"; merged.mkdir()
    (merged / "model.gguf").write_bytes(b"GGUFDATA")
    calls = []
    monkeypatch.setattr(pub.subprocess, "run", _mock_run(calls))
    monkeypatch.setattr(pub, "already_published", lambda tid: False)
    recorded = {}
    monkeypatch.setattr(pub, "record_publication",
                        lambda r, v, sref, url: recorded.update(v=v, url=url) or True)

    out = pub.maybe_publish(_report(), str(merged))

    assert out["version"] == "v1007-deadbeef"
    dest = sub / "rsi-qwen" / "v1007-deadbeef"
    assert (dest / "model.gguf").read_bytes() == b"GGUFDATA"
    assert (dest / "Modelfile").read_text().startswith("FROM ./model.gguf")
    assert (dest / "card.md").exists()
    cmds = [c["cmd"] for c in calls]
    assert any(c[:2] == ["git", "add"] for c in cmds)
    assert any(c[:2] == ["git", "commit"] for c in cmds)
    assert any(c[:3] == ["git", "tag", "-a"] for c in cmds)          # annotated (pushable)
    assert any(c[:2] == ["git", "push"] and "HEAD" in c for c in cmds)
    assert any(c[:2] == ["git", "push"] and "rsi-qwen-v1007-deadbeef" in c for c in cmds)  # tag pushed
    rel = next(c for c in cmds if c[:3] == ["gh", "release", "create"])
    assert "--repo" in rel and pub.FORK_REPO in rel
    assert "tenstorrent" not in " ".join(rel)          # never upstream
    assert "rsi-qwen-v1007-deadbeef" in rel            # versioned tag
    assert recorded["v"] == "v1007-deadbeef"


def test_submodule_uninitialized_skips_but_release_still_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PUBLISH", "1")
    monkeypatch.setenv("RSI_MODELS_SUBMODULE", str(tmp_path / "missing"))  # no .git
    merged = tmp_path / "merged"; merged.mkdir()
    (merged / "model.gguf").write_bytes(b"G")
    calls = []
    monkeypatch.setattr(pub.subprocess, "run", _mock_run(calls))
    monkeypatch.setattr(pub, "already_published", lambda tid: False)
    monkeypatch.setattr(pub, "record_publication", lambda *a: True)

    out = pub.maybe_publish(_report(), str(merged))

    cmds = [c["cmd"] for c in calls]
    assert not any(c[0] == "git" for c in cmds)         # submodule step bailed cleanly
    assert any(c[:3] == ["gh", "release", "create"] for c in cmds)  # release still fired
    assert out["version"] == "v1007-deadbeef"


def test_already_published_is_idempotent_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_PUBLISH", "1")
    sub = tmp_path / "ollama-models"; (sub / ".git").mkdir(parents=True)
    monkeypatch.setenv("RSI_MODELS_SUBMODULE", str(sub))
    merged = tmp_path / "merged"; merged.mkdir(); (merged / "model.gguf").write_bytes(b"G")
    calls = []
    monkeypatch.setattr(pub.subprocess, "run", _mock_run(calls))
    monkeypatch.setattr(pub, "already_published", lambda tid: True)   # seen before
    assert pub.maybe_publish(_report(), str(merged)) is None
    assert calls == []                                  # nothing published twice
