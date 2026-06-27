#!/usr/bin/env bash
# Debug the cuOT segfault: run ALICE under gdb (backtrace on SIGSEGV), BOB
# normally as its peer. Both need sudo (bind). Run from repo root.
#
# ARG FIX: binary takes <party> <port> [gpu] (3 positional, host hardcoded).
# CUDA_VISIBLE_DEVICES pins each process to one physical GPU.
#
#   sudo ./run_cuot_debug.sh [PORT]
set -u
PORT="${1:-32012}"
BIN="./build/bin/cuot_correlation_test"

pkill -9 -f cuot_correlation_test 2>/dev/null || true
sleep 1

# BOB (receiver) plain, pinned to physical GPU1 (logical dev 0).
CUDA_VISIBLE_DEVICES=1 "$BIN" 2 "$PORT" 0 > /tmp/cuot_bob_dbg.log 2>&1 &
BOB=$!
sleep 1

# ALICE (sender) under gdb, pinned to physical GPU0.
CUDA_VISIBLE_DEVICES=0 gdb -batch \
    -ex 'set pagination off' \
    -ex 'set environment LD_LIBRARY_PATH=/home/richorange/miniforge3/lib:/usr/local/cuda/lib64:./build/lib' \
    -ex 'run 1 '"$PORT"' 0' \
    -ex 'bt' \
    -ex 'thread apply all bt' \
    --args "$BIN" 1 "$PORT" 0 > /tmp/cuot_alice_dbg.log 2>&1
echo "gdb rc=$?"
wait "$BOB" 2>/dev/null

echo "================ ALICE gdb log ================"
cat /tmp/cuot_alice_dbg.log
echo "================ BOB log ================"
cat /tmp/cuot_bob_dbg.log
