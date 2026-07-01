#!/usr/bin/env bash
# Run the M3-T6 Phase B cuOT block-ROT smoke test (two-party, two GPUs).
#
# Validates CuotProvider::send_rot_blocks / recv_rot_blocks in isolation before
# BitTripleGen composes on top. Inherits cuOT's ~2% RCOT floor; PASS = rate≤4%.
set -u
PORT="${1:-32120}"
GPU0="${2:-0}"
GPU1="${3:-1}"
BIN="./build/bin/cuot_rot_smoke_test"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first (needs OPENMM_ENABLE_CUOT=ON):"
  echo "  cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=\$PWD/build"
  echo "  cmake --build build -j --target cuot_rot_smoke_test"
  exit 2
fi
if [[ ! -d data ]]; then echo "ERROR: no ./data dir. Run from repo root."; exit 2; fi

for pid in $(pgrep -f "build/bin/cuot_rot_smoke_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
sudo rm -f data/pre_ot_data_reg_* 2>/dev/null || rm -f data/pre_ot_data_reg_* 2>/dev/null || true

S_LOG="$(mktemp -t cuotrot_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuotrot_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT ROT smoke test  port=$PORT  GPU0=$GPU0(ALICE/sender) GPU1=$GPU1(BOB/receiver) ==="

CUDA_VISIBLE_DEVICES="$GPU0" "$BIN" 1 "$PORT" 0 > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1
CUDA_VISIBLE_DEVICES="$GPU1" "$BIN" 2 "$PORT" 0 > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1, GPU$GPU0) ---";  cat "$S_LOG"
echo "--- receiver (party 2, GPU$GPU1) ---"; cat "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"

if grep -q "cuOT ROT smoke test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
