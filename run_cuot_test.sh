#!/usr/bin/env bash
# Run the M2-T2 cuOT correlation test (two-party, two GPUs) on localhost.
#
# Tests BOTH:
#   - RCOT:        r[i] == send[i] ^ (lsb(r[i]) * Delta)        (cuOT native)
#   - chosen COT:  r[i] == x'[i] ^ (b[i] * Delta)               (paper §III-D,
#                  diff-vector d_i = lsb(r_i) ^ b_i sent over NetIO; sender
#                  x'_i = x_i ^ (d_i * Delta). No IPC, no cuOT patching.)
# for batch sizes {1, 2, 17, 1024, 4096}.
#
# Party 1 = sender   (ALICE, GPU0, server : binds 127.0.0.1:PORT)
# Party 2 = receiver (BOB,   GPU1, client : connects to 127.0.0.1:PORT)
#
# ARG FIX: the binary takes <party> <port> [gpu] (3 positional args, host is
# hardcoded 127.0.0.1 inside CuotProvider). Do NOT pass $HOST or port/gpu get
# misparsed (atoi("127.0.0.1")=127, atoi("32010")=32010 invalid device).
#
# CUDA_VISIBLE_DEVICES: each process is pinned to ONE physical GPU (exposed as
# logical device 0). This kills the per-thread CUDA primary-context mismatch
# (FerretCOT::setup spawns a ThreadPool worker that never calls cudaSetDevice;
# with only one visible device, the worker's default-0 matches the main thread).
#
# This box requires sudo for bind() (see memory openmm-build-env §4):
#   sudo ./run_cuot_test.sh [PORT]
#
# Usage:
#   ./run_cuot_test.sh [PORT] [GPU0] [GPU1]    # defaults: PORT=32010 GPU0=0 GPU1=1
set -u

PORT="${1:-32010}"
GPU0="${2:-0}"
GPU1="${3:-1}"
BIN="${BIN:-./build/bin/cuot_correlation_test}"
# CUDA_LAUNCH_BLOCKING=1 forces every kernel to block the host until complete,
# serializing all GPU work. Set to 1 to test whether cuOT's wrong-correlation
# bug is a missing-sync race (errors vanish => race confirmed). Default 0.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first (needs OPENMM_ENABLE_CUOT=ON):"
  echo "  cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON"
  echo "  cmake --build build -j --target cuot_correlation_test"
  exit 2
fi
if [[ ! -d data ]]; then
  echo "ERROR: no ./data dir here. Run from repo root (Ferret reads/writes"
  echo "       ./data/pre_ot_data_reg_* at runtime)."
  exit 2
fi

pkill -f "$(basename "$BIN")" 2>/dev/null || true
sleep 0.5

# CRITICAL: clear the Ferret pre-OT cache before EVERY run. cuOT's ~FerretCOT
# destructor writes ot_pre_data to ./data/pre_ot_data_reg_{send,recv} AFTER
# rcot has run — but by then extend_f2k has RECYCLED ot_pre_data (overwritten
# it with the last round's tail blocks via read_from_gpu, ferret_cot.hpp:99).
# So the on-disk cache holds a RECYCLED seed, not the original setup seed.
# A subsequent run that cache-HITs reads that recycled seed as the initial
# seed -> 100% wrong RCOT (not ~2%). Only a cache-MISS (fresh regenerate)
# is valid. Clearing every run forces the fresh path. (The ~2% residual on
# the fresh path is a separate, still-open kernel bug — see m2-cuot-standalone.md.)
sudo rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || \
  rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t cuot_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuot_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT correlation test  port=$PORT  GPU0=$GPU0(ALICE/sender) GPU1=$GPU1(BOB/receiver) ==="
echo "    (CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING arch=$(grep -o 'compute_[0-9]*' build/CMakeFiles/cuot_correlation_test.dir/flags.make 2>/dev/null | head -1))"

# Sender (ALICE, server): pinned to physical GPU0 via CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES="$GPU0" "$BIN" 1 "$PORT" 0 > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1

# Receiver (BOB, client): pinned to physical GPU1.
CUDA_VISIBLE_DEVICES="$GPU1" "$BIN" 2 "$PORT" 0 > "$R_LOG" 2>&1
R_RC=$?

wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1, GPU$GPU0) ---";  cat "$S_LOG"
echo "--- receiver (party 2, GPU$GPU1) ---"; cat "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"

if grep -q "cuOT correlation test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
