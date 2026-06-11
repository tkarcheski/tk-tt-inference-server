# Hosting the logwatch model on a Raspberry Pi / ARM SBC

Guide for serving the Qwen2.5-0.5B logwatch triage model on a small ARM
system — Raspberry Pi 4/5, Orange Pi 5, NVIDIA Jetson, or any aarch64 box —
so log review runs next to the logs instead of on the inference cluster.

## Why this works at all

The logwatch harness was designed for exactly this (see `logwatch/README.md`):
stateless calls, ~1.5k-token prompts, ~50-token JSON verdicts, no streaming
needed, latency tolerance of seconds. A quantized 0.5B fits those constraints
on a Pi with room to spare. What a Pi can NOT do is serve chat traffic or
train — this is a single-purpose triage appliance.

## Hardware reality check

| Device | RAM | Expected tok/s (0.5B Q4) | Verdict |
|---|---|---|---|
| Pi 4 (4 GB) | 4 GB | ~4–8 | works, slow; fine for nightly sweeps |
| Pi 5 (8 GB) | 8 GB | ~12–20 | comfortable; per-run sweeps |
| Orange Pi 5 / RK3588 | 8–16 GB | ~15–25 | comfortable |
| Jetson Orin Nano | 8 GB | 40+ (GPU) | overkill, use if you have one |

A Q4_K_M-quantized 0.5B is ~400 MB of weights; with KV cache for a 2k
context, total footprint stays under ~1 GB. Use a 64-bit OS (Raspberry Pi OS
64-bit or Ubuntu Server arm64) — 32-bit userlands cripple llama.cpp.

## Recommended stack: llama.cpp server

vLLM does not run on Pi-class hardware; llama.cpp is the standard choice and
its server exposes the same OpenAI-compatible endpoint the harness already
speaks (`LOGWATCH_MODEL_URL` just points at it).

### 1. Build llama.cpp (on the Pi, ~5 min)

```bash
sudo apt install -y build-essential cmake git
git clone https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-server -j4
```

ARM NEON/dotprod kernels are detected automatically on Pi 4/5. For RK3588,
the same build works; for Jetson add `-DGGML_CUDA=ON`.

### 2. Get the model in GGUF

Until the fine-tuned logwatch adapter exists (issue #2), use the base model:

```bash
# Prebuilt GGUF from HF (Qwen publishes official GGUFs)
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir ~/models
```

When the LoRA fine-tune lands, merge it first (`MERGE=1` in
`train_lora.py`), then convert the merged model on a workstation:

```bash
python3 llama.cpp/convert_hf_to_gguf.py finetune/out/lora-*/merged \
    --outfile logwatch-0.5b-f16.gguf
llama.cpp/build/bin/llama-quantize logwatch-0.5b-f16.gguf \
    logwatch-0.5b-q4_k_m.gguf Q4_K_M     # ~400 MB, scp to the Pi
```

### 3. Serve

```bash
llama.cpp/build/bin/llama-server \
    -m ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 --port 8000 \
    -c 2048 --parallel 2 --threads 4
```

`-c 2048` matches the harness chunk budget (~1.5k tokens + verdict); small
context = small KV cache = happy Pi. `--parallel 2` lets two chunks triage
concurrently; raise on 8 GB boards.

Note: llama.cpp's structured-output flavor differs from vLLM's `guided_json`
(it uses GBNF grammars / `response_format`). The harness tolerates this —
unparseable verdicts degrade to `benign` — but for strict enforcement pass a
grammar; see issue tracker for the planned `LOGWATCH_GUIDED=off` switch.

### 4. Run the harness against it

The harness is pure-Python + `requests` and runs on the Pi directly:

```bash
git clone <this repo> && cd tt-inference-server
pip install requests pyyaml
LOGWATCH_MODEL_URL=http://localhost:8000 \
LOGWATCH_MODEL=qwen2.5-0.5b-instruct \
LOGWATCH_MODEL_VERSION=logwatch-q4km-v0 \
python3 -m logwatch triage /var/log/myapp --engine llm
```

Ship logs to the Pi however you already do (rsync, syslog forwarding, NFS
mount); the harness only needs filesystem paths.

### 5. Make it an appliance (systemd)

`/etc/systemd/system/llama-server.service`:

```ini
[Unit]
Description=llama.cpp server (logwatch model)
After=network.target

[Service]
ExecStart=/home/pi/llama.cpp/build/bin/llama-server -m /home/pi/models/logwatch.gguf --host 0.0.0.0 --port 8000 -c 2048 --threads 4
Restart=on-failure
MemoryMax=2G

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/logwatch.timer` + `.service` running
`python3 -m logwatch file-issues <logdir> --engine llm --tracker github`
nightly, and `python3 -m logwatch feedback` after it. Tokens go in a
systemd `EnvironmentFile=`, not the unit file.

## Sizing the cadence

At a conservative 10 tok/s, one chunk costs ~10–20 s wall-clock (prefill
dominates: ~1.5k prompt tokens). A nightly sweep of ~100 fresh chunks (the
size of this repo's whole `workflow_logs/`) is ~20–30 min — fine for a
timer job. If sweep time ever matters, that's the signal to move serving to
a bigger box, not to grow the Pi.

## Alternatives, briefly

- **Ollama** (arm64 builds, wraps llama.cpp): easiest install
  (`ollama run qwen2.5:0.5b-instruct`), same OpenAI endpoint on :11434;
  trade fine-grained control for convenience. Good first step.
- **ONNX Runtime / ExecuTorch**: lower-level, only worth it if llama.cpp's
  numbers don't meet the cadence.
- **Cross-compiling**: unnecessary — on-device build is minutes for
  llama-server.
