#!/usr/bin/env bash
# Run the M1-T2 OT provider smoke test (two-party) on localhost.
#
# Verifies EmpOTProvider wraps SCI SilentOT behind gpu_mm::OTProvider and that
# the COT correlation holds for both modulus flavors M&M uses:
#   - COT over Z_{2^l}   (ring,   send_cot / recv_cot)
#   - COT over Z_p        (field,  send_cot_prime / recv_cot_prime)
#
# Party 1 = sender   (ALICE, server : binds 127.0.0.1:PORT, address arg ignored)
# Party 2 = receiver (BOB,   client : connects to 127.0.0.1:PORT)
#
# Usage:
#   ./run_ot_test.sh [PORT]        # default PORT=32000
#
# Run from the repo root so Ferret's ./data/pre_ot_data_reg_* cache is found.
set -u

PORT="${1:-32000}"
BIN="${BIN:-./build/bin/ot_provider_smoke_test}"
HOST="127.0.0.1"

# --- checks ---------------------------------------------------------------
if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first:"
  echo "  cmake -B build -DABI=0 && cmake --build build -j --target ot_provider_smoke_test"
  exit 2
fi
if [[ ! -d data ]]; then
  echo "ERROR: no ./data dir here. Run this script from the repo root"
  echo "       (Ferret reads/writes ./data/pre_ot_data_reg_* at runtime)."
  exit 2
fi

# --- clean up any stale instances -----------------------------------------
pkill -f "$(basename "$BIN")" 2>/dev/null || true
sleep 0.5

S_LOG="$(mktemp -t ot_s.XXXXXX.log)"
R_LOG="$(mktemp -t ot_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== OT provider smoke test  host=$HOST port=$PORT ==="

# Sender (ALICE) is the server: it binds and waits for the receiver.
"$BIN" 1 "$HOST" "$PORT" > "$S_LOG" 2>&1 &
S_PID=$!

# Give the server a moment to bind+listen before the client connects.
sleep 1

# Receiver (BOB) is the client: it connects and runs the correlation check.
"$BIN" 2 "$HOST" "$PORT" > "$R_LOG" 2>&1
R_RC=$?

wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1) ---";  cat "$S_LOG"
echo "--- receiver (party 2) ---"; cat "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"

if grep -q "OT provider smoke test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"
  exit 0
else
  echo "RESULT: FAIL"
  exit 1
fi
