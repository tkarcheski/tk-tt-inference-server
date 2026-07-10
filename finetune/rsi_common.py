import os
import psycopg2

DSN = os.environ.get("RSI_DSN", "postgresql://rfc:changeme@localhost:5434/rfc")
SALT = "rfc-split-v816"
# 11 gold-answer suites minus the top-level `variables` duplicate of hallucination.
# Gold-answer suites carry a concrete target answer: they are fingerprinted for the
# train/holdout/canary firewall and extracted into the fine-tuning dataset.
GOLD_SUITES = [
    "adversarial", "c_interview", "causal_reasoning", "code_review",
    "context_window", "extraction", "hallucination", "legal",
    "quantization", "temporal_reasoning",
]

# Execution-graded suites: real per-test pass/fail comes from RUNNING code in a
# sandboxed Docker container (container exit_code / stdout), not from a stored
# answer. They are eval-only — never fingerprinted, never trained on
# (build_dataset intentionally skips behavior-only suites), and pinned to an eval
# pool (see split_suites.split_of) so they are actually run yet can never leak
# through the training firewall. The rfc suite enforces its own sandbox bounds
# (network_mode=none, cpu/mem caps, per-test timeout) via its container profile.
EXEC_SUITES = [
    "docker/python",
]

# Every suite the loop evaluates each round (gold-answer + execution-graded).
EVAL_SUITES = GOLD_SUITES + EXEC_SUITES

def connect():
    return psycopg2.connect(DSN)

_ENC = None
def token_len(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text or ""))
