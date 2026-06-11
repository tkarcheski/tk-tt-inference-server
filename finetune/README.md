# finetune/ — LoRA fine-tuning on robotframework-chat skills

Pipeline: extract Q→A pairs from robotframework-chat scenario YAMLs, LoRA-tune
a Qwen instruct model on them, then (eventually) evaluate the adapter against
the rfc suites. See `docs/p100_demo_branch.md` for the bigger picture.

## Fast loop

```bash
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -r requirements-train.txt

# 1. Build the dataset (seconds; reads $RFC_ROOT, default ~/AI/robotframework-chat)
python3 build_dataset.py

# 2. Smoke test (minutes, CPU, tiny base model) — always run before a real run
./smoke_test.sh

# 3. Real run (GPU node; install the CUDA torch build first)
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct EPOCHS=3 MERGE=1 python3 train_lora.py
```

All training knobs are env vars — see the docstring in `train_lora.py`.

## Notes

- `out/` (datasets, adapters) is gitignored: the JSONL files contain rfc eval
  answer keys and must not be published.
- Base model is Qwen by design — never Llama.
- trl is pinned to 0.12.2 → full-sequence SFT; upgrade to trl ≥ 0.13 for
  `assistant_only_loss=True`.
