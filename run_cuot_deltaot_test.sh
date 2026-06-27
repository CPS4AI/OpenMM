#!/usr/bin/env bash
# Run the M2-T3 cuOT arithmetic Delta-OT adapter test (two-party, two GPUs).
#
# Verifies the MITCCRH-on-RCOT adapter produces the correct algebraic COT:
#   y_1 - y_0 = corr mod K,  K in {2^32, 2^64, p=4293918721}, n=16384.
#
# Party 1 = sender   (ALICE, GPU0, server)
# Party 2 = receiver (BOB,   GPU1, client)
#
# CAVEAT: inherits cuOT's ~2% RCOT error floor (M2-T2 shelved). The test
# PASSES the adapter-logic check if failure rate stays within ~4% (absorbs the
# inherited floor); above that the MITCCRH/packing port is buggy.
#
# This box requires sudo for bind() (see memory openmm-build-env §4).
#   sudo ./run_cuot_deltaot_test.sh [PORT]
set -u
PORT="${1:-32030}"
GPU0="${2:-0}"
GPU1="${3:-1}"
BIN="./build/bin/cuot_deltaot_arithmetic_test"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first (needs OPENMM_ENABLE_CUOT=ON):"
  echo "  cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON"
  echo "  cmake --build build -j --target cuot_deltaot_arithmetic_test"
  exit 2
fi
if [[ ! -d data ]]; then
  echo "ERROR: no ./data dir. Run from repo root."; exit 2
fi

pkill -f cuot_deltaot_arithmetic_test 2>/dev/null || true
sleep 0.5
# Clear Ferret cache every run (cuOT destructor writes a recycled seed;
# cache-HIT reads it as the initial seed -> 100% wrong). See
# docs/research/m2-cuot-standalone.md §7.4/§7.5.
sudo rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || \
  rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t cuotdt_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuotdt_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT Delta-OT arithmetic adapter test  port=$PORT  GPU0=$GPU0(ALICE) GPU1=$GPU1(BOB) ==="

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
if grep -q "cuOT Delta-OT arithmetic test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
