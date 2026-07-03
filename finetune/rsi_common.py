import os
import psycopg2

DSN = os.environ.get("RSI_DSN", "postgresql://rfc:changeme@localhost:5434/rfc")
SALT = "rfc-split-v816"
# 11 gold-answer suites minus the top-level `variables` duplicate of hallucination.
GOLD_SUITES = [
    "adversarial", "c_interview", "causal_reasoning", "code_review",
    "context_window", "extraction", "hallucination", "legal",
    "quantization", "temporal_reasoning",
]

def connect():
    return psycopg2.connect(DSN)

_ENC = None
def token_len(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text or ""))
