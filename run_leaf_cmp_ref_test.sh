#!/usr/bin/env bash
# Run the M3-T6 Phase A CPU baseline single-digit compare leaf test.
#
# Verifies the SCI kkot leaf (1-OO-N-KOT, N∈{2,4,8,16}) computes the correct
# shared comparison bit for n=100000 random digit pairs per bitlength, and
# reports the real wire bytes per leaf batch (io->counter delta). This is the
# CPU correctness + comm baseline the cuOT leaf diagnostic is measured against.
#
# Party 1 = sender (ALICE, server), party 2 = receiver (BOB, client).
set -u
PORT="${1:-32100}"
BIN="${BIN:-./build/bin/leaf_cmp_ref_test}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first:"
  echo "  cmake -B build -DABI=0 -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=\$PWD/build"
  echo "  cmake --build build -j --target leaf_cmp_ref_test"
  exit 2
fi
if [[ ! -d data ]]; then echo "ERROR: no ./data dir. Run from repo root."; exit 2; fi

for pid in $(pgrep -f "build/bin/leaf_cmp_ref_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t lcr_s.XXXXXX.log)"
R_LOG="$(mktemp -t lcr_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== leaf_cmp_ref test  port=$PORT ==="
"$BIN" 1 "$PORT" > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1
"$BIN" 2 "$PORT" > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1) ---";   cat "$S_LOG"
echo "--- receiver (party 2) ---"; cat "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"

if grep -q "leaf_cmp_ref test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
