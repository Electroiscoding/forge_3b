#!/usr/bin/env bash
clear
echo "========================================================================"
echo "  🚀 FORGE-1B LIVE TRAINING MONITOR (NVIDIA H100 SXM)"
echo "========================================================================"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "------------------------------------------------------------------------"
if tmux has-session -t forge_train 2>/dev/null; then
    echo "Status: [RUNNING] Active in tmux session 'forge_train'"
    echo "Commands:"
    echo "  - Attach to live console : tmux attach -t forge_train"
    echo "  - Detach from console    : Press Ctrl+b then d"
else
    echo "Status: [STOPPED / IDLE] No active session found."
fi
echo "========================================================================"
echo "Live Stream Output (Press Ctrl+C to exit monitor):"
echo "------------------------------------------------------------------------"

if [ -f /workspace/forge_pipeline_master.log ]; then
    tail -n 25 /workspace/forge_pipeline_master.log
    echo "------------------------------------------------------------------------"
    tail -f /workspace/forge_pipeline_master.log
else
    LATEST_LOG=$(ls -t /workspace/checkpoints/forge_1b_pretrain/logs/*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        tail -n 25 "$LATEST_LOG"
        echo "------------------------------------------------------------------------"
        tail -f "$LATEST_LOG"
    else
        echo "No active log file found yet."
    fi
fi
