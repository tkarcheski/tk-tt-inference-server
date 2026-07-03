# SPDX-License-Identifier: Apache-2.0
"""Turn a merged fine-tuned model directory into a served Ollama model tag.

Consumes a merged safetensors dir (`train_lora.py` MERGE=1 output at
`OUTPUT_DIR/merged`). Writes a Modelfile pointing `FROM` the merged dir and
runs `ollama create` to register the tag. Also provides `adapter_hash` for a
stable content hash of a LoRA adapter directory, used for provenance.
"""
import hashlib
import os
import subprocess


def adapter_hash(adapter_dir):
    h = hashlib.sha256()
    for name in sorted(os.listdir(adapter_dir)):
        p = os.path.join(adapter_dir, name)
        if os.path.isfile(p):
            h.update(name.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


def create_tag(merged_dir, tag, quant=None):
    """Register the merged model dir as an Ollama tag via `ollama create`.

    quant: optional Ollama quantization (e.g. "q4_K_M"). Left None by default:
    `-q` quantization during an --experimental safetensors import requires MLX
    (Apple Silicon) and fails on Linux/x86 ("quantization requires MLX
    support"), so we import the merged fp16 weights as-is unless a caller on a
    supported platform opts in.

    DEPLOYMENT NOTE: `ollama create` writes the tag into the CALLER's
    OLLAMA_MODELS store. For a running daemon to serve the tag, that store must
    match the daemon's OLLAMA_MODELS and be writable by the calling user — on a
    host where the daemon owns its store (e.g. OLLAMA_MODELS=/mnt/ai/ollama
    owned by user `ollama`), create the tag as that user, or convert the merged
    model to GGUF and `ollama create` from the .gguf so the daemon imports it
    server-side.
    """
    modelfile = os.path.join(merged_dir, "Modelfile")
    with open(modelfile, "w") as fh:
        fh.write(f"FROM {os.path.abspath(merged_dir)}\n")
    cmd = ["ollama", "create", tag, "--experimental"]
    if quant:
        cmd += ["-q", quant]
    cmd += ["-f", modelfile]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return tag
