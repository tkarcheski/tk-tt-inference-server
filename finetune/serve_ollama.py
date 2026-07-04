# SPDX-License-Identifier: Apache-2.0
"""Turn a merged fine-tuned model directory into a served Ollama model tag.

Consumes a merged safetensors dir (`train_lora.py` MERGE=1 output at
`OUTPUT_DIR/merged`). By default it converts that dir to a GGUF file and runs
`ollama create` from the `.gguf`, so the running daemon imports the model
*server-side* — this is the path that actually works on a Linux/x86 host where
the daemon owns its model store (see DEPLOYMENT NOTE below). Also provides
`adapter_hash` for a stable content hash of a LoRA adapter directory, used for
provenance.

GGUF conversion uses llama.cpp's `convert_hf_to_gguf.py`. Its location and the
Python used to run it (needs torch/transformers, i.e. the training venv) are
configurable via env vars:

  RSI_LLAMACPP_CONVERT  path to convert_hf_to_gguf.py
                        (default /home/tyler/AI/tools/llama.cpp/convert_hf_to_gguf.py)
  RSI_CONVERT_PY        python interpreter to run it
                        (default <this dir>/.venv-train/bin/python)
"""
import hashlib
import os
import subprocess

CONVERT_SCRIPT = os.environ.get(
    "RSI_LLAMACPP_CONVERT",
    "/home/tyler/AI/tools/llama.cpp/convert_hf_to_gguf.py",
)
# Absolute path to the training venv's interpreter (has torch/transformers,
# which the converter imports). It is NOT resolved relative to this module: the
# venv is git-ignored, so it exists only in the main checkout and is absent from
# git worktree copies. Mirrors the same hardcoded path run_loop.py uses for the
# train subprocess; override with RSI_CONVERT_PY on other hosts.
CONVERT_PY = os.environ.get(
    "RSI_CONVERT_PY",
    "/home/tyler/AI/github/tk-tt-inference-server/finetune/.venv-train/bin/python",
)


def adapter_hash(adapter_dir):
    h = hashlib.sha256()
    for name in sorted(os.listdir(adapter_dir)):
        p = os.path.join(adapter_dir, name)
        if os.path.isfile(p):
            h.update(name.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


def merged_to_gguf(merged_dir, outtype="f16"):
    """Convert a merged HF safetensors dir to a single GGUF file.

    Returns the path to the written `.gguf`. Raises RuntimeError with the
    converter's stderr on failure. `outtype` "f16" keeps full fp16 weights (no
    quantization) — quantization is a separate, optional step and is NOT done
    here (the `-q` path of `ollama create` needs MLX/Apple-Silicon).
    """
    out = os.path.join(merged_dir, "model.gguf")
    cmd = [CONVERT_PY, CONVERT_SCRIPT, merged_dir, "--outfile", out,
           "--outtype", outtype]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed (`{' '.join(cmd)}`, exit "
            f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )
    return out


def create_tag(merged_dir, tag, quant=None, use_gguf=True):
    """Register the merged model dir as an Ollama tag via `ollama create`.

    use_gguf (default True): convert the merged dir to a GGUF first and point
    the Modelfile `FROM` the `.gguf`. `ollama create` then imports the model
    server-side through the daemon — the ONLY path proven to work on this host,
    where the daemon serves from a store it owns (see DEPLOYMENT NOTE).

    use_gguf=False: legacy `--experimental` path that imports the safetensors
    dir directly. `quant` (e.g. "q4_K_M") only applies here and requires MLX
    (Apple Silicon) — it fails on Linux/x86 with "quantization requires MLX
    support". Kept for callers on a supported platform.

    DEPLOYMENT NOTE: `ollama create --experimental` from a safetensors dir
    writes into the CALLER's OLLAMA_MODELS store; if the daemon serves from a
    different, root/`ollama`-owned store, the tag never appears in the daemon's
    `ollama list`. Importing from a `.gguf` avoids this: the daemon reads the
    file and writes the blob into its own store, so no cross-user write is
    needed. That is why use_gguf=True is the default.
    """
    if use_gguf:
        model_ref = merged_to_gguf(merged_dir)
    else:
        model_ref = os.path.abspath(merged_dir)
    modelfile = os.path.join(merged_dir, "Modelfile")
    with open(modelfile, "w") as fh:
        fh.write(f"FROM {model_ref}\n")
    cmd = ["ollama", "create", tag]
    if not use_gguf:
        cmd.append("--experimental")
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
