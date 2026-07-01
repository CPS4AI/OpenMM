#!/usr/bin/env bash
# Run the M3-T4 cuOT ring→field OR_AUX test (two-party, two GPUs).
#
# cuOT counterpart of run_ring_to_field_or_aux_test.sh (M3-T4 CPU baseline).
# Runs the SAME boundary block + 100000 random samples through RingToField
# backed by CuotProvider (GPU Ferret) instead of EmpOTProvider.
#
# Verifies:
#   (out_A + out_B) mod p == (msb_A | msb_B) * corr  mod p
# and reports the bad count / rate. The cuOT backend inherits cuOT's ~2% RCOT
# floor (M2-T2 shelved), so this is an ADAPTER-LOGIC / WIRING pass (rate ≤ 4%),
# NOT a clean 100k pass — see the test source header and
# docs/research/m3-cuot-integration.md.
#
# Party 1 = sender   (ALICE, GPU0, server)
# Party 2 = receiver (BOB,   GPU1, client)
#
# This box requires sudo for bind() (see memory openmm-build-env §4).
#   sudo ./run_cuot_ring_to_field_test.sh [PORT]
set -u
PORT="${1:-32070}"
GPU0="${2:-0}"
GPU1="${3:-1}"
BIN="./build/bin/cuot_ring_to_field_test"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first (needs OPENMM_ENABLE_CUOT=ON):"
  echo "  cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=\$PWD/build"
  echo "  cmake --build build -j --target cuot_ring_to_field_test"
  exit 2
fi
if [[ ! -d data ]]; then
  echo "ERROR: no ./data dir. Run from repo root (Ferret cache path)."; exit 2
fi

for pid in $(pgrep -f "build/bin/cuot_ring_to_field_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
sudo rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || \
  rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t cuotrtf_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuotrtf_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT ring_to_field OR_AUX test  port=$PORT  GPU0=$GPU0(ALICE/sender) GPU1=$GPU1(BOB/receiver) ==="

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

if grep -q "cuOT ring_to_field OR_AUX test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
