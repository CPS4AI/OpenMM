#!/usr/bin/env bash
# Run the M3-T5 CPU baseline field→ring OR_AUX test (two-party) on localhost.
#
# Verifies FieldToRing::or_aux (on gpu_mm::OTProvider, CPU EmpOTProvider)
# computes the correct shared ring-domain OR_AUX term:
#   (out_A + out_B) & mask == ((msb_A | msb_B) * p) & mask,  p=4293918721, bw=32.
# Boundary cases {all-0, A-only, B-only, both-1} + 100000 random samples.
set -u
PORT="${1:-32088}"
BIN="${BIN:-./build/bin/field_to_ring_or_aux_test}"

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN"
  echo "build first:"
  echo "  cmake -B build -DABI=0 && cmake --build build -j --target field_to_ring_or_aux_test"
  exit 2
fi
if [[ ! -d data ]]; then echo "ERROR: no ./data dir. Run from repo root."; exit 2; fi

for pid in $(pgrep -f "build/bin/field_to_ring_or_aux_test" 2>/dev/null); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.5
rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true

S_LOG="$(mktemp -t ftr_s.XXXXXX.log)"
R_LOG="$(mktemp -t ftr_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== field_to_ring OR_AUX test  port=$PORT ==="
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

if grep -q "field_to_ring OR_AUX test PASSED" "$R_LOG"; then
  echo "RESULT: PASS"; exit 0
else
  echo "RESULT: FAIL"; exit 1
fi
