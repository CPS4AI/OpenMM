#!/usr/bin/env bash
# Run the M3-T2 wrap-bit logical-OR primitive test (two-party) on localhost.
#
# Verifies WrapBitOr (on gpu_mm::OTProvider, CPU EmpOTProvider default) computes
# the correct shared logical-OR -> arithmetic wrap:
#   out_A + out_B == OR(local_bit_A, local_bit_B)  (mod 2^l),  l=32.
# Boundary cases {both-0, A-only-1, B-only-1, both-1} + 100000 random samples.
#
# Party 1 = sender   (ALICE, server : binds 127.0.0.1:PORT)
# Party 2 = receiver (BOB,   client : connects to 127.0.0.1:PORT)
#
# Usage:
#   ./run_wrap_bit_or_test.sh [PORT]      # default PORT=32090
#
# Run from the repo root so Ferret's ./data/pre_ot_data_reg_* cache is found.
set -u

PORT="${1:-32090}"
BIN="${BIN:-./build/bin/wrap_bit_or_test}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first:"
  echo "  cmake -B build -DABI=0 && cmake --build build -j --target wrap_bit_or_test"
  exit 2
fi
if [[ ! -d data ]]; then
  echo "ERROR: no ./data dir here. Run from repo root (Ferret cache path)."
  exit 2
fi

# Kill only stale test BINARY processes (not this script). pkill -f on the
# basename can match the launcher's own argv under some shells and signal the
# script; pgrep the binary path and kill those PIDs explicitly instead.
for pid in $(pgrep -f "build/bin/wrap_bit_or_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
# Clear Ferret cache every run (avoid stale-seed reuse; matches run_ot_test.sh hygiene).
rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t wbo_s.XXXXXX.log)"
R_LOG="$(mktemp -t wbo_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== wrap-bit OR test  port=$PORT ==="

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

if grep -q "wrap-bit OR test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
