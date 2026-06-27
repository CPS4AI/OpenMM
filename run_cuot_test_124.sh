#!/usr/bin/env bash
# Run the M2-T2 cuOT correlation test BUILT WITH CUDA 12.4 + gcc-12
# (build_cuot124/) — isolates whether the ~2% RCOT corruption is a CUDA 13.2
# runtime/codegen semantic change vs a cuOT logic bug.
#
# Same protocol as run_cuot_test.sh (rcot + chosen-COT diff-vector, batch
# {1,2,17,1024,4096}, two-party same-machine + TCP, each pinned to one GPU),
# just a different binary + 12.4 runtime libs on LD_LIBRARY_PATH.
#
# Usage:  sudo ./run_cuot_test_124.sh [PORT]
set -u
PORT="${1:-32020}"
BIN="./build_cuot124/bin/cuot_correlation_test"
GPU0="${2:-0}"
GPU1="${3:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
# 12.4 runtime libs (driver 550+ runs 12.4 runtime; this box has 595).
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/home/richorange/miniforge3/lib:./build/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first:"
  echo "  cmake -B build_cuot124 -DABI=0 -DOPENMM_ENABLE_CUOT=ON \\"
  echo "    -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 \\"
  echo "    -DCMAKE_PREFIX_PATH=$PWD/build \\"
  echo "    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \\"
  echo "    -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \\"
  echo "    -DCMAKE_CUDA_ARCHITECTURES=80"
  echo "  cmake --build build_cuot124 -j --target cuot_correlation_test"
  exit 2
fi
if [[ ! -d data ]]; then
  echo "ERROR: no ./data dir. Run from repo root."; exit 2
fi

pkill -f cuot_correlation_test 2>/dev/null || true
sleep 0.5
# Clear Ferret cache every run (destructor writes recycled seed -> cache-HIT 100% wrong).
rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t cuot124_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuot124_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT correlation test (CUDA 12.4 / gcc-12 build)  port=$PORT  GPU0=$GPU0(ALICE) GPU1=$GPU1(BOB) ==="
echo "    nvcc: $(/usr/local/cuda-12.4/bin/nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"

CUDA_VISIBLE_DEVICES="$GPU0" "$BIN" 1 "$PORT" 0 > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1
CUDA_VISIBLE_DEVICES="$GPU1" "$BIN" 2 "$PORT" 0 > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1, GPU$GPU0) ---"; cat "$S_LOG"
echo "--- receiver (party 2, GPU$GPU1) ---"; cat "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"
if grep -q "cuOT correlation test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
