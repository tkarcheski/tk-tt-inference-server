#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Launch a vLLM server for Llama-3.1-8B on a single Tenstorrent P100 (Blackhole)
# by driving the release container DIRECTLY — bypassing run.py.
#
# WHY: at the current branch tip run.py is broken by an unfinished spec migration,
# and every runnable run.py on this branch emits the post-v0.11 docker entrypoint
# contract, which is incompatible with the only available P100 Llama images (<=0.10,
# old `docker-entrypoint.sh + gosu` contract). This script replicates the exact
# container interface those older images expect, reusing the already-generated
# 64K P100 tensor cache so the server comes up in seconds instead of regenerating.
#
# Produces a real, measured throughput number (see scripts/p100/bench_p100.py):
#   ~29 tok/s/user single-user decode, ~361 tok/s aggregate at 16 concurrent users.
#
# Prereqs: P100 visible (`tt-smi -ls`), HF_TOKEN with meta-llama access, the local
# image, base weights in the HF cache, and the persistent_volume with P100 cache.
set -euo pipefail

IMG=${IMG:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.7.0-55fd115-aa4ae1e}
REPO_ROOT=${REPO_ROOT:-/home/tyler/AI/github/tk-tt-inference-server}
VOL=${VOL:-$REPO_ROOT/persistent_volume/volume_id_tt_transformers-Llama-3.1-8B-v0.7.0}
# weights: base Llama-3.1-8B snapshot in the HF hub cache
WEIGHTS_REPO=${WEIGHTS_REPO:-/home/tyler/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B}
WSHA=${WSHA:-d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}
SPEC=${SPEC:-$REPO_ROOT/scripts/p100/tt_model_spec_llama31_8b_p100.json}
HOST_PORT=${HOST_PORT:-8011}
NAME=${NAME:-p100-llama-bench}

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN must be set (meta-llama gated repo)"; exit 1; }

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --cap-add ALL --device /dev/tenstorrent/0:/dev/tenstorrent/0 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src="$VOL",dst=/home/container_app_user/cache_root \
  --mount type=bind,src="$SPEC",dst=/home/container_app_user/model_spec/spec.json,readonly \
  --mount type=bind,src="$WEIGHTS_REPO",dst=/home/container_app_user/readonly_weights_mount/Llama-3.1-8B,readonly \
  --shm-size 32G --publish "$HOST_PORT":8000 \
  -e CACHE_ROOT=/home/container_app_user/cache_root \
  -e TT_CACHE_PATH=/home/container_app_user/cache_root/tt_metal_cache/cache_Llama-3.1-8B/P100 \
  -e MODEL_WEIGHTS_PATH=/home/container_app_user/readonly_weights_mount/Llama-3.1-8B/snapshots/$WSHA \
  -e TT_MODEL_SPEC_JSON_PATH=/home/container_app_user/model_spec/spec.json \
  -e ARCH_NAME=blackhole -e MESH_DEVICE=P100 -e HF_TOKEN="$HF_TOKEN" \
  "$IMG"

echo "Launched $NAME. Poll: curl http://localhost:$HOST_PORT/health"
echo "The spec has cli_args.disable_trace_capture=true — REQUIRED: the background"
echo "trace sweep crashes the 64K config at a 2048-token prefill (L1 buffer clash)."
echo "Keep request prompts well under ~2048 tokens."
