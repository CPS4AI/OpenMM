#!/usr/bin/env bash
# Run the M3-T6 Phase C cuOT full 32-bit Millionaire compare diagnostic.
#
# Two-party, two GPUs, FOUR CuotProviders (2/party, two ports). n=100000
# 32-bit compares through CuotCompare. Reports error rate + latency + comm.
# Diagnostic: ~2% RCOT floor compounded over 8-digit leaf + 11-triple AND-tree.
# PASS = rate not in chance band [45,55]% and <= 80%.
#
# Provider wiring (same as bit-triple test):
#   port1 = leaf-send/ROT#1 (BOB prov_send server <-> ALICE prov_recv client)
#   port2 = leaf-recv/ROT#2 (ALICE prov_send server <-> BOB prov_recv client)
set -u
PORT1="${1:-32910}"
PORT2="${2:-32911}"
GPU0="${3:-0}"
GPU1="${4:-1}"
N="${5:-100000}"
BIN="./build/bin/cuot_compare_test"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first (needs OPENMM_ENABLE_CUOT=ON):"
  echo "  cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=\$PWD/build"
  echo "  cmake --build build -j --target cuot_compare_test"
  exit 2
fi
if [[ ! -d data ]]; then echo "ERROR: no ./data dir. Run from repo root."; exit 2; fi

for pid in $(pgrep -f "build/bin/cuot_compare_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
sudo rm -f data/pre_ot_data_reg_* 2>/dev/null || rm -f data/pre_ot_data_reg_* 2>/dev/null || true

S_LOG="$(mktemp -t cuotcmp_s.XXXXXX.log)"
R_LOG="$(mktemp -t cuotcmp_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT compare test  port1=$PORT1 port2=$PORT2  GPU0=$GPU0(ALICE) GPU1=$GPU1(BOB) ==="

CUDA_VISIBLE_DEVICES="$GPU0" "$BIN" 1 "$PORT1" "$PORT2" 0 "$N" > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1
CUDA_VISIBLE_DEVICES="$GPU1" "$BIN" 2 "$PORT1" "$PORT2" 0 "$N" > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- alice (party 1, GPU$GPU0) ---"; cat "$S_LOG"
echo "--- bob   (party 2, GPU$GPU1) ---"; cat "$R_LOG"
echo "exit codes:  bob=$R_RC  alice=$S_RC"

if grep -q "cuOT compare test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
