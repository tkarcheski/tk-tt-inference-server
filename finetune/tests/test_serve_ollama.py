import pytest

import serve_ollama as so


def _mock_run(calls, returncode=0, stderr="", stdout=""):
    return lambda *a, **k: calls.append(a[0]) or type(
        "R", (), {"returncode": returncode, "stderr": stderr, "stdout": stdout}
    )()


def test_gguf_path_is_default(tmp_path, monkeypatch):
    # Default use_gguf=True: convert to GGUF, Modelfile FROM the .gguf, and
    # `ollama create` WITHOUT --experimental / -q (those are MLX-only).
    merged = tmp_path / "merged"; merged.mkdir()
    monkeypatch.setattr(so, "merged_to_gguf", lambda d, **k: str(merged / "model.gguf"))
    calls = []
    monkeypatch.setattr(so.subprocess, "run", _mock_run(calls))
    tag = so.create_tag(str(merged), "rsi-qwen:test")
    assert tag == "rsi-qwen:test"
    assert (merged / "Modelfile").read_text() == f"FROM {merged / 'model.gguf'}\n"
    cmd = calls[0]
    assert cmd == ["ollama", "create", "rsi-qwen:test", "-f", str(merged / "Modelfile")]
    assert "--experimental" not in cmd and "-q" not in cmd


def test_experimental_fallback_and_quant(tmp_path, monkeypatch):
    # use_gguf=False keeps the legacy safetensors path; quant only applies here.
    merged = tmp_path / "merged"; merged.mkdir()
    calls = []
    monkeypatch.setattr(so.subprocess, "run", _mock_run(calls))
    so.create_tag(str(merged), "t", quant="q4_K_M", use_gguf=False)
    cmd = calls[0]
    assert cmd[:3] == ["ollama", "create", "t"]
    assert "--experimental" in cmd and "-q" in cmd and "q4_K_M" in cmd
    assert (merged / "Modelfile").read_text() == f"FROM {merged}\n"


def test_merged_to_gguf_builds_cmd(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    calls = []
    monkeypatch.setattr(so.subprocess, "run", _mock_run(calls))
    out = so.merged_to_gguf(str(merged))
    assert out == str(merged / "model.gguf")
    cmd = calls[0]
    assert cmd[0] == so.CONVERT_PY and cmd[1] == so.CONVERT_SCRIPT
    assert cmd[2] == str(merged)
    assert "--outfile" in cmd and str(merged / "model.gguf") in cmd
    assert "--outtype" in cmd and "f16" in cmd


def test_merged_to_gguf_raises_with_stderr(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    monkeypatch.setattr(
        so.subprocess, "run",
        _mock_run([], returncode=1, stderr="boom: torch missing"),
    )
    with pytest.raises(RuntimeError, match="boom: torch missing"):
        so.merged_to_gguf(str(merged))


def test_create_tag_raises_with_ollama_stderr(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    monkeypatch.setattr(so, "merged_to_gguf", lambda d, **k: str(merged / "model.gguf"))
    monkeypatch.setattr(
        so.subprocess, "run",
        _mock_run([], returncode=1, stderr="something failed"),
    )
    with pytest.raises(RuntimeError, match="something failed"):
        so.create_tag(str(merged), "t")


def test_adapter_hash_stable(tmp_path):
    a = tmp_path / "ad"; a.mkdir(); (a / "adapter_config.json").write_text("{}")
    (a / "adapter_model.safetensors").write_bytes(b"weights")
    assert len(so.adapter_hash(str(a))) == 64
