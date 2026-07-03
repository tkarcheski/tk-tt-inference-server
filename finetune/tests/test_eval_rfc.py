import eval_rfc as e


def test_arm_env_ollama():
    env = e.arm_env({"provider": "ollama", "model": "rsi-qwen:v1"})
    assert env["LLM_PROVIDER"] == "ollama" and env["DEFAULT_MODEL"] == "rsi-qwen:v1"


def test_arm_env_vllm():
    env = e.arm_env({"provider": "vllm", "base_url": "http://localhost:8011/v1"})
    assert env["LLM_PROVIDER"] == "vllm" and env["VLLM_BASE_URL"] == "http://localhost:8011/v1"
