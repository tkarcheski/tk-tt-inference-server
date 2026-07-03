# SPDX-License-Identifier: Apache-2.0
"""Put finetune/ on sys.path so `pytest tests/` resolves the loop's top-level
modules (rsi_common, split_suites, build_dataset, ...) regardless of which
rootdir pytest selects. Individual files already work via `PYTHONPATH=.`;
this makes whole-directory collection work too (CI + final review)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
