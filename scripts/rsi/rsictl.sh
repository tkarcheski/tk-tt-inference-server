#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# rsictl.sh — control the 24/7 shadow MODEL_TUNER loop as a systemd --user
# service. The loop (finetune/run_loop.py --forever) runs rounds back-to-back,
# SHADOW-ONLY: it never swaps a serving endpoint, never auto-merges, only ever
# proposes. See scripts/rsi/README.md.
#
# Usage: rsictl.sh {install|start|pause|resume|stop|restart|status|logs [N]|follow|uninstall}
set -euo pipefail

# Deploy dir = repo root that contains finetune/run_loop.py (this script lives in
# <deploy>/scripts/rsi/). All venv paths inside run_loop.py are absolute, so the
# deploy dir only needs to hold the source + scratch (out/, split.json, .rsi-stop).
RSI_DEPLOY="${RSI_DEPLOY:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GLUE_PY="${RSI_GLUE_PY:-/home/tyler/.rsi-loop-venv/bin/python}"
UNIT_NAME="rsi-loop.service"
UNIT="$HOME/.config/systemd/user/$UNIT_NAME"
STOP_FILE="${RSI_STOP_FILE:-$RSI_DEPLOY/finetune/.rsi-stop}"

# Service round config (override any of these in the environment at `install`).
: "${RFC_ROOT:=/home/tyler/AI/rfc/wt-rfc}"
: "${RSI_BASE_TAG:=qwen2.5:3b}"         # untuned base arm (paired control)
: "${RSI_SKIP_SUITES:=context_window,legal}"  # latency-bound needle/stress suites
: "${RSI_MODE:=shadow}"                  # NEVER promotes unless flipped to live
: "${RSI_SMOKE:=0}"                      # 1 = tiny/fast rounds (validation)
: "${RSI_LOOP_SLEEP:=300}"              # seconds between rounds

usage() { sed -n '13,14p' "${BASH_SOURCE[0]}"; exit 2; }
cmd="${1:-status}"

case "$cmd" in
  install)
    mkdir -p "$(dirname "$UNIT")"
    # Bake the live PATH so round subprocesses find ollama / uv / git.
    cat > "$UNIT" <<EOF
[Unit]
Description=RSI shadow MODEL_TUNER 24/7 loop (finetune/run_loop.py --forever)
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$RSI_DEPLOY/finetune
Environment=PATH=$PATH
Environment=HOME=$HOME
Environment=RFC_ROOT=$RFC_ROOT
Environment=RSI_BASE_TAG=$RSI_BASE_TAG
Environment=RSI_SKIP_SUITES=$RSI_SKIP_SUITES
Environment=RSI_MODE=$RSI_MODE
Environment=RSI_SMOKE=$RSI_SMOKE
Environment=RSI_LOOP_SLEEP=$RSI_LOOP_SLEEP
Environment=RSI_STOP_FILE=$STOP_FILE
ExecStart=$GLUE_PY $RSI_DEPLOY/finetune/run_loop.py --forever
# on-failure only: a crash restarts, but a graceful stop-file exit(0) stays down.
Restart=on-failure
RestartSec=60

[Install]
WantedBy=default.target
EOF
    loginctl enable-linger "$USER" >/dev/null 2>&1 || true   # survive logout/reboot
    systemctl --user daemon-reload
    echo "installed $UNIT"
    echo "  deploy=$RSI_DEPLOY  smoke=$RSI_SMOKE  mode=$RSI_MODE  base=$RSI_BASE_TAG  sleep=${RSI_LOOP_SLEEP}s"
    ;;
  start)   rm -f "$STOP_FILE"; systemctl --user enable --now "$UNIT_NAME"; echo "started (+enabled on boot)";;
  pause)   touch "$STOP_FILE"; echo "stop-file set → loop halts gracefully after the current round finishes";;
  resume)  rm -f "$STOP_FILE"; systemctl --user restart "$UNIT_NAME"; echo "resumed";;
  stop)    systemctl --user stop "$UNIT_NAME"; echo "stopped now (SIGTERM — may interrupt a round; use 'pause' to finish it first)";;
  restart) rm -f "$STOP_FILE"; systemctl --user restart "$UNIT_NAME"; echo "restarted";;
  status)  systemctl --user status "$UNIT_NAME" --no-pager || true;;
  logs)    journalctl --user -u "$UNIT_NAME" -n "${2:-80}" --no-pager || true;;
  follow)  journalctl --user -u "$UNIT_NAME" -f;;
  uninstall) touch "$STOP_FILE"; systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
             rm -f "$UNIT"; systemctl --user daemon-reload; echo "uninstalled";;
  *) usage;;
esac
