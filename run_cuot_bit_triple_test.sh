#!/usr/bin/env bash
# Run the M3-T6 Phase B cuOT bit-triple (_2ROT) diagnostic test.
#
# Two-party, two GPUs, FOUR CuotProviders (2 per party on two ports). Generates
# n=100000 random bit-AND triples; checks (a_A^a_B)&(b_A^b_B)==c_A^c_B.
# Inherits cuOT's ~2% RCOT floor compounded over 2 ROTs (~4%); PASS = rate≤8%.
#
# Provider wiring:
#   port1 = ROT#1 (BOB send <-> ALICE recv)
#   port2 = ROT#2 (ALICE send <-> BOB recv)
# Each party pinned to one GPU via CUDA_VISIBLE_DEVICES.
set -u
PORT1="${1:-32130}"
PORT2="${2:-32131}"
GPU0="${3:-0}"
GPU1="${4:-1}"
BIN="./build/bin/cuot_bit_triple_test"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first (needs OPENMM_ENABLE_CUOT=ON):"
  echo "  cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=\$PWD/build"
  echo "  cmake --build build -j --target cuot_bit_triple_test"
  exit 2
fi
if [[ ! -d data ]]; then echo "ERROR: no ./data dir. Run from repo root."; exit 2; fi

for pid in $(pgrep -f "build/bin/cuot_bit_triple_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
sudo rm -f data/pre_ot_data_reg_* 2>/dev/null || rm -f data/pre_ot_data_reg_* 2>/dev/null || true

S_LOG="$(mktemp -t cuotbt_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuotbt_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT bit-triple test  port1=$PORT1(ROT#1) port2=$PORT2(ROT#2)  GPU0=$GPU0(ALICE) GPU1=$GPU1(BOB) ==="

CUDA_VISIBLE_DEVICES="$GPU0" "$BIN" 1 "$PORT1" "$PORT2" 0 > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1
CUDA_VISIBLE_DEVICES="$GPU1" "$BIN" 2 "$PORT1" "$PORT2" 0 > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- alice (party 1, GPU$GPU0) ---"; cat "$S_LOG"
echo "--- bob   (party 2, GPU$GPU1) ---"; cat "$R_LOG"
echo "exit codes:  bob=$R_RC  alice=$S_RC"

if grep -q "cuOT bit-triple test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
