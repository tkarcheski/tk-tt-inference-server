# SPDX-License-Identifier: Apache-2.0
"""Run an robotframework-chat (rfc) suite against a chosen model "arm".

An arm is `{"provider": "ollama"|"vllm", "model": tag, "base_url": url}`.
`arm_env` maps an arm to the env-var overrides rfc's `.env` expects so the
suite talks to the right provider/model; `run_suite` mirrors the Makefile's
`robot` recipe (minus rfc's own DB listeners, since the RSI loop owns its
`rsi.*` result tables) and returns the path to the resulting `output.xml`.
"""
import os
import subprocess

RFC = os.environ.get("RFC_ROOT", "/home/tyler/AI/rfc/wt-rfc")


def arm_env(arm):
    env = dict(os.environ)
    env["LLM_PROVIDER"] = arm["provider"]
    if arm["provider"] == "ollama":
        env["DEFAULT_MODEL"] = arm["model"]
        env["OLLAMA_ENDPOINT"] = arm.get("base_url", "http://localhost:11434")
    elif arm["provider"] == "vllm":
        env["VLLM_BASE_URL"] = arm.get("base_url", "http://localhost:8011/v1")
    return env


def run_suite(suite_id, arm, results_dir):
    # Mirror the Makefile robot recipe WITHOUT rfc's DB listeners (we own rsi.*).
    os.makedirs(results_dir, exist_ok=True)
    cmd = ["uv", "run", "robot", "-d", results_dir,
           "--metadata", f"test_suite:{suite_id}",
           f"robot/{suite_id}/"]
    subprocess.run(cmd, cwd=RFC, env=arm_env(arm), check=False)  # robot exits nonzero on test failures
    return os.path.join(results_dir, "output.xml")
