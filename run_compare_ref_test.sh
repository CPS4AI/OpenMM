#!/usr/bin/env bash
# Run the M3-T6 Phase C CPU baseline full 32-bit Millionaire compare test.
# Clean 100k PASS + per-compare latency + wire bytes (io->counter). The CPU
# numbers the cuOT compare is measured against.
set -u
PORT="${1:-32900}"
BIN="${BIN:-./build/bin/compare_ref_test}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first:"
  echo "  cmake -B build -DABI=0 -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=\$PWD/build"
  echo "  cmake --build build -j --target compare_ref_test"
  exit 2
fi
if [[ ! -d data ]]; then echo "ERROR: no ./data dir. Run from repo root."; exit 2; fi

for pid in $(pgrep -f "build/bin/compare_ref_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t cmpref_s.XXXXXX.log)"
R_LOG="$(mktemp -t cmpref_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== compare_ref test  port=$PORT ==="
"$BIN" 1 "$PORT" > "$S_LOG" 2>&1 &
S_PID=$!
sleep 1
"$BIN" 2 "$PORT" > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- party 1 (ALICE) ---"; cat "$S_LOG"
echo "--- party 2 (BOB)   ---"; cat "$R_LOG"
echo "exit codes:  bob=$R_RC  alice=$S_RC"

if grep -q "compare_ref test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
