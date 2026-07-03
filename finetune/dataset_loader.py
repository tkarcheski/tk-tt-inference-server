#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load a training dataset from either a local .jsonl file or a HF hub id.
Kept separate from train_lora.py so it can be imported and unit-tested without
pulling in torch/transformers/trl (which live only in the training venv)."""
from datasets import load_dataset


def load_local_or_hub(name):
    """A local path ending in .jsonl loads as a JSON dataset with a 'train'
    split; anything else is treated as a HuggingFace hub dataset id."""
    if name.endswith(".jsonl"):
        return load_dataset("json", data_files={"train": name})
    return load_dataset(name)
