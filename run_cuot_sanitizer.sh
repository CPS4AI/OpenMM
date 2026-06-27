#!/usr/bin/env bash
# Run cuOT under compute-sanitizer to localize the remaining RCOT corruption.
#
# IMPORTANT: native test_ferret uses send_cot = rcot + online_sender(IPC). On a
# 4-GPU box the IPC path hits the known otherDev=dev<4?dev+4:dev-4 bug
# (cudaMemcpyPeer to nonexistent device 4) — that masks/is independent of the
# rcot corruption we're hunting. So we run the WRAPPER (cuot_correlation_test)
# under memcheck instead: it calls ferret_->rcot directly (no IPC), so any OOB
# hit is in the rcot/extend path itself.
#
# Usage:  sudo ./run_cuot_sanitizer.sh [tool] [PORT]
#   tool defaults to memcheck
set -u
TOOL="${1:-memcheck}"
PORT="${2:-12346}"
BIN="./build/bin/cuot_correlation_test"
SAN="/usr/local/cuda/bin/compute-sanitizer"

if [[ ! -x "$BIN" ]]; then echo "binary missing: $BIN (build with OPENMM_ENABLE_CUOT=ON)"; exit 2; fi
if [[ ! -x "$SAN" ]];  then echo "sanitizer missing: $SAN";  exit 2; fi
cd /home/richorange/dongye/OpenMM
mkdir -p data
rm -f data/pre_ot_data_reg_send data/pre_ot_data_reg_recv 2>/dev/null || true
pkill -9 -f cuot_correlation_test 2>/dev/null || true
sleep 1

S_LOG="$(mktemp -t san_s.XXXXXX.log)"
R_LOG="$(mktemp -t san_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT WRAPPER under compute-sanitizer ($TOOL)  port=$PORT  (rcot path, no IPC) ==="
# Both parties pinned to one GPU (logical dev 0) via CUDA_VISIBLE_DEVICES.
# NOTE: do NOT pass --force-blocking-launches for racecheck — it makes the
# tool 10-50x slower (every launch blocks) and the user Ctrl-C'd it last
# time. racecheck needs to observe natural warp interleaving anyway. Be
# PATIENT: racecheck instruments every shared-mem access; a ~10s test can
# take several minutes. Let it run to "ERROR SUMMARY".
( sleep 0.05; CUDA_VISIBLE_DEVICES=0 "$SAN" --tool "$TOOL" --error-exitcode 99 "$BIN" 1 "$PORT" 0 ) > "$S_LOG" 2>&1 &
S_PID=$!
CUDA_VISIBLE_DEVICES=1 "$SAN" --tool "$TOOL" --error-exitcode 99 "$BIN" 2 "$PORT" 0 > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1, GPU0) sanitizer (first 60 lines) ---"; head -60 "$S_LOG"
echo "--- receiver (party 2, GPU1) sanitizer (first 80 lines) ---"; head -80 "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"
echo
echo "=== For racecheck: look for 'Race reported' / 'Hazard' with kernel name"
echo "    + .cu:line. For memcheck: 'Invalid access'/'Out of bounds'. ==="
