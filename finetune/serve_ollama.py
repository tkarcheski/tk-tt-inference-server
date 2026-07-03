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


def create_tag(merged_dir, tag, quant="q4_K_M"):
    modelfile = os.path.join(merged_dir, "Modelfile")
    with open(modelfile, "w") as fh:
        fh.write(f"FROM {os.path.abspath(merged_dir)}\n")
    subprocess.run(["ollama", "create", tag, "--experimental", "-q", quant, "-f", modelfile], check=True)
    return tag
