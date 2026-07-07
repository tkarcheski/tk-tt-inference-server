# RSI MODEL_TUNER — live shadow-loop status

> Auto-generated every round by the 24/7 [RSI MODEL_TUNER loop](https://github.com/tkarcheski/tk-tt-inference-server/blob/gh-pages/rsi-loop.md). **Shadow-only**: the loop LoRA-fine-tunes a small Qwen on robotframework-chat suites, evaluates the tuned model against the untuned base on a frozen holdout+canary split, and runs a McNemar promotion gate. Passing rounds are *proposed*, never auto-promoted — a human decides.

**Live dashboard:** https://tkarcheski.github.io/tk-tt-inference-server/

## Models

- **Base (control):** [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on Hugging Face, served as Ollama `qwen2.5:3b`.
- **Tuned (per round):** a LoRA fine-tune of the base, merged + served as Ollama `rsi-qwen:round`.
- **Published tuned models:** [`tkarcheski/rsi-ollama-models`](https://github.com/tkarcheski/rsi-ollama-models) — git-LFS registry (private); a round is pushed there only when it passes the gate.

## Summary

- Rounds run: **26** (24 graded)
- Rounds that passed the gate (proposed): **0**
- Best canary Δ so far: **-5.0 pp**
- Latest round: seed `1020` at 2026-07-07T14:32:17
- Generated: 2026-07-07 15:55 UTC

## Rounds (most recent first)

| Seed | Started | Holdout Δpp (p) | Canary Δpp (p) | Gate |
|---:|:--|:--|:--|:--|
| 1020 | 2026-07-07T14:32:17 | -16.1 (p=0.9922) | -10.0 (p=0.9375) | 🔴 held |
| 1019 | 2026-07-07T10:29:55 | -6.5 (p=1.0) | -15.0 (p=0.9688) | 🔴 held |
| 1018 | 2026-07-07T06:08:39 | -3.2 (p=1.0) | -5.0 (p=0.875) | 🔴 held |
| 1017 | 2026-07-07T01:52:22 | -6.5 (p=1.0) | -15.0 (p=0.9688) | 🔴 held |
| 1016 | 2026-07-06T23:25:26 | -7.1 (p=1.0) | +0 (p=1.0) | ⚪ degenerate |
| 1015 | 2026-07-06T19:19:16 | -6.5 (p=1.0) | -20.0 (p=0.9844) | 🔴 held |
| 1014 | 2026-07-06T14:57:50 | -3.2 (p=1.0) | -10.0 (p=0.9375) | 🔴 held |
| 1013 | 2026-07-06T10:41:03 | -6.5 (p=1.0) | -15.0 (p=0.9688) | 🔴 held |
| 1012 | 2026-07-06T06:34:58 | -3.2 (p=1.0) | -15.0 (p=0.9688) | 🔴 held |
| 1011 | 2026-07-06T04:07:32 | -38.7 (p=1.0) | -15.0 (p=0.9375) | 🔴 held |
| 1010 | 2026-07-06T01:07:50 | +12.9 (p=0.1094) | -15.0 (p=0.9688) | 🔴 held |
| 1009 | 2026-07-05T22:14:23 | -22.6 (p=0.998) | -15.0 (p=0.9688) | 🔴 held |
| 1008 | 2026-07-05T20:35:15 | -19.4 (p=0.9961) | -20.0 (p=0.9844) | 🔴 held |
| 1007 | 2026-07-05T19:15:57 | -22.6 (p=0.998) | -25.0 (p=0.9922) | 🔴 held |
| 1006 | 2026-07-05T17:24:55 | -19.4 (p=0.9961) | -15.0 (p=1.0) | 🔴 held |
| 1005 | 2026-07-05T14:26:33 | -22.6 (p=0.998) | -35.0 (p=0.998) | 🔴 held |
| 1004 | 2026-07-05T09:57:06 | +12.9 (p=0.1094) | -25.0 (p=0.9922) | 🔴 held |
| 1003 | 2026-07-05T05:24:12 | -3.2 (p=1.0) | -15.0 (p=0.9688) | 🔴 held |
| 1002 | 2026-07-05T02:07:02 | -29.0 (p=0.9995) | -10.0 (p=0.9375) | 🔴 held |
| 1001 | 2026-07-04T22:26:46 | +9.7 (p=0.2266) | -20.0 (p=0.9844) | 🔴 held |
| 1000 | 2026-07-04T18:32:13 | +12.9 (p=0.1094) | -10.0 (p=0.9375) | 🔴 held |
| 1001 | 2026-07-04T15:27:05 | +35.5 (p=0.0017) | -5.0 (p=0.7734) | 🔴 held |
| 1000 | 2026-07-04T06:07:28 | +22.6 (p=0.0327) | -10.0 (p=0.9375) | 🔴 held |
| 1000 | 2026-07-04T03:36:02 | +0 (p=1.0) | +0 (p=1.0) | ⚪ degenerate |
| None | 2026-07-04T01:57:07 | +19.4 (p=0.0898) | -25.0 (p=0.9805) | 🔴 held |
| None | 2026-07-04T00:10:58 | +19.4 (p=0.0898) | -25.0 (p=0.9805) | 🔴 held |

Gate rule: a round is **proposed** only when the tuned model beats base by **≥5pp AND p<0.05** (exact McNemar) on **both** holdout and canary.
