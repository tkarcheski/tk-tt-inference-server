# P100 (Blackhole) Scripts for GPT-OSS-20b

**Status: EXPERIMENTAL** - These scripts provide experimental support for running GPT-OSS-20b on Tenstorrent P100 (Blackhole) cards.

## Prerequisites

1. **Hardware**: Tenstorrent P100 (Blackhole) card installed
2. **System**: Ubuntu 20.04 or 22.04 with tt-metal drivers installed
3. **Docker**: Docker with GPU support for Tenstorrent devices
4. **Hugging Face Token**: Required for downloading model weights
   ```bash
   export HF_TOKEN=your_token_here
   ```
5. **JWT Secret**: For API authentication (set during first run)

## Available Scripts

### 1. Start Inference Server

```bash
./scripts/p100/gpt-oss-20b-server.sh
```

Starts the GPT-OSS-20b inference server in a Docker container on your P100 card.

**Options:**
- `DEVICE_ID`: Specify which P100 device to use (default: 0)
  ```bash
  DEVICE_ID=0 ./scripts/p100/gpt-oss-20b-server.sh
  ```

### 2. Run Benchmarks

```bash
./scripts/p100/gpt-oss-20b-benchmark.sh
```

Runs performance benchmarks on the P100. This will automatically start the server, run benchmarks, and shut down.

### 3. Chat Test

```bash
# First, start the server in one terminal:
./scripts/p100/gpt-oss-20b-server.sh

# Then in another terminal, run:
./scripts/p100/gpt-oss-20b-chat.sh
```

Sends a simple chat completion request to test the server.

## Configuration

### Model Parameters (P100)

The P100 configuration for GPT-OSS-20b uses these conservative settings:

- **Max Concurrency**: 16 (requests processed simultaneously)
- **Max Context**: 32,768 tokens
- **Trace Region Size**: 30,000,000
- **VLLM Version**: V1 (recommended for P100)

These settings are optimized for stability on a single P100 card. You can override them in `workflows/model_spec.py` if needed.

## Troubleshooting

### Server won't start

1. Check device is recognized:
   ```bash
   tt-smi -ls
   ```

2. Verify Docker can access the device:
   ```bash
   docker run --rm --device /dev/tenstorrent/0 hello-world
   ```

3. Check logs:
   ```bash
   ls workflow_logs/docker_server/
   ```

### Out of memory errors

The 20B model is large for a single P100. If you encounter OOM errors:
- Reduce `max_concurrency` in model_spec.py
- Reduce `max_context` length
- Use smaller batch sizes

### Slow first startup

First run will:
1. Download model weights (~40GB)
2. Compile kernels for P100
3. Generate trace files

This can take 10-30 minutes. Subsequent runs will be faster.

## Manual Run Commands

If you prefer to run manually without the scripts:

```bash
# Start server
python3 run.py \
    --model gpt-oss-20b \
    --device p100 \
    --workflow server \
    --docker-server \
    --dev-mode

# Run benchmarks
python3 run.py \
    --model gpt-oss-20b \
    --device p100 \
    --workflow benchmarks \
    --docker-server \
    --dev-mode

# Run evals
python3 run.py \
    --model gpt-oss-20b \
    --device p100 \
    --workflow evals \
    --docker-server \
    --dev-mode
```

## Known Issues

- **Experimental Status**: This is not officially supported. Performance may vary.
- **Memory Constraints**: 20B params on P100 is memory-intensive. Monitor device temperature.
- **Limited Testing**: Has not been through full release validation.

## Contributing

If you find issues or improvements for P100 support:
1. Test changes locally
2. Update this README with findings
3. Consider submitting a PR to the main repo

## Resources

- [Tenstorrent Documentation](https://docs.tenstorrent.com/)
- [GPT-OSS Model Card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
- [tt-inference-server Issues](https://github.com/tenstorrent/tt-inference-server/issues)
