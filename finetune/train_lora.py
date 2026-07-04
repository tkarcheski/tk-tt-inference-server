#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LoRA SFT fine-tune of a Qwen instruct model on the robotframework-chat-skills
dataset (OpenAI chat format).

Designed to run unchanged on a GPU node for the real run, or on CPU with a small
base model for an end-to-end smoke test. All knobs are environment variables:

  BASE_MODEL   HF model id            (default Qwen/Qwen2.5-7B-Instruct)
  DATASET      HF dataset id          (default nutinspace/robotframework-chat-skills)
  OUTPUT_DIR   adapter output dir     (default ./out/lora-<base>)
  EPOCHS       float                  (default 3)
  MAX_STEPS    int, -1 = use epochs   (default -1)
  LORA_R / LORA_ALPHA / LR / MAX_SEQ_LEN / BATCH / GRAD_ACCUM
  SEED         int, seeds init + data shuffle  (default 42)
  MERGE        1 to also write a merged fp16 model to OUTPUT_DIR/merged

Never Llama — base must be a Qwen (or other non-Llama) instruct model.
"""
import os

import torch
from dataset_loader import load_local_or_hub
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def env(name, default):
    return os.environ.get(name, default)


def main():
    base = env("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    dataset_id = env("DATASET", "nutinspace/robotframework-chat-skills")
    out_dir = env("OUTPUT_DIR", "out/lora-" + base.split("/")[-1])
    epochs = float(env("EPOCHS", "3"))
    max_steps = int(env("MAX_STEPS", "-1"))
    max_seq_len = int(env("MAX_SEQ_LEN", "1024"))
    batch = int(env("BATCH", "1"))
    grad_accum = int(env("GRAD_ACCUM", "8"))
    lr = float(env("LR", "2e-4"))
    seed = int(env("SEED", "42"))
    set_seed(seed)  # reproducible per-round variation (weight init + data shuffle)

    cuda = torch.cuda.is_available()
    print(f"base={base}\ndataset={dataset_id}\nout={out_dir}\ncuda={cuda}")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None,
    )

    ds = load_local_or_hub(dataset_id)
    train_ds = ds["train"]
    # Split naming varies (build_dataset.py emits "val", HF convention is
    # "validation") — accept either rather than silently training without eval.
    eval_ds = next((ds[k] for k in ("validation", "val") if k in ds), None)
    if eval_ds is None:
        print("WARNING: no validation/val split found; training without eval")

    lora = LoraConfig(
        r=int(env("LORA_R", "16")),
        lora_alpha=int(env("LORA_ALPHA", "32")),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        seed=seed,
        data_seed=seed,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        bf16=cuda,
        max_seq_length=max_seq_len,
        packing=False,
        report_to=[],
        # NOTE: assistant-only loss masking isn't available in trl 0.12.2; this
        # does standard full-sequence SFT. On a newer trl (>=0.13) you can add
        # assistant_only_loss=True for completion-only loss.
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\nSaved LoRA adapter to {out_dir}")

    if env("MERGE", "0") == "1":
        from peft import PeftModel
        merged_dir = os.path.join(out_dir, "merged")
        base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
        merged = PeftModel.from_pretrained(base_model, out_dir).merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"Saved merged fp16 model to {merged_dir}")


if __name__ == "__main__":
    main()
