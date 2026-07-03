import serve_ollama as so

def test_writes_modelfile_and_calls_ollama(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(so.subprocess, "run", lambda *a, **k: calls.append(a[0]) or type("R", (), {"returncode": 0})())
    tag = so.create_tag(str(merged), "rsi-qwen:test")
    assert tag == "rsi-qwen:test"
    assert (merged / "Modelfile").read_text().startswith("FROM ")
    assert any("ollama" in c and "create" in c for c in calls)

def test_adapter_hash_stable(tmp_path):
    a = tmp_path / "ad"; a.mkdir(); (a / "adapter_config.json").write_text("{}")
    (a / "adapter_model.safetensors").write_bytes(b"weights")
    assert len(so.adapter_hash(str(a))) == 64
