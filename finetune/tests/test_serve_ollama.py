import pytest

import serve_ollama as so


def _mock_run(calls):
    return lambda *a, **k: calls.append(a[0]) or type(
        "R", (), {"returncode": 0, "stderr": "", "stdout": ""}
    )()


def test_writes_modelfile_and_calls_ollama(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(so.subprocess, "run", _mock_run(calls))
    tag = so.create_tag(str(merged), "rsi-qwen:test")
    assert tag == "rsi-qwen:test"
    assert (merged / "Modelfile").read_text().startswith("FROM ")
    cmd = calls[0]
    assert cmd[:3] == ["ollama", "create", "rsi-qwen:test"] and "--experimental" in cmd
    # no MLX-only -q quantization by default (fails on Linux/x86)
    assert "-q" not in cmd


def test_quant_opt_in_adds_flag(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    calls = []
    monkeypatch.setattr(so.subprocess, "run", _mock_run(calls))
    so.create_tag(str(merged), "t", quant="q4_K_M")
    cmd = calls[0]
    assert "-q" in cmd and "q4_K_M" in cmd


def test_create_tag_raises_with_ollama_stderr(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    monkeypatch.setattr(
        so.subprocess, "run",
        lambda *a, **k: type("R", (), {
            "returncode": 1, "stderr": "quantization requires MLX support", "stdout": ""
        })(),
    )
    with pytest.raises(RuntimeError, match="MLX"):
        so.create_tag(str(merged), "t", quant="q4_K_M")


def test_adapter_hash_stable(tmp_path):
    a = tmp_path / "ad"; a.mkdir(); (a / "adapter_config.json").write_text("{}")
    (a / "adapter_model.safetensors").write_bytes(b"weights")
    assert len(so.adapter_hash(str(a))) == 64
