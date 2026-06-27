#!/usr/bin/env bash
# Isolation test: run cuOT's OWN native test_ferret (no gpu_mm wrapper, no
# cuda_setdev, both parties on default device 0) two-party under sudo.
#
# Goal: determine whether the rcot/extend segfault is
#   (A) cuOT+CUDA13.2/sm86 incompatibility (native test_ferret ALSO segfaults), or
#   (B) my gpu_mm wrapper / device-forcing (native test_ferret PASSES).
#
# The native binary /tmp/test_ferret was built from cuOT's own
# ferret/emp-ot/test/ferret.cpp + gpu/*.cu + dev_layer.cu, linked against our
# emp-tool + miniforge3 OpenSSL + CUDA 13.2. It calls test_cot (chosen COT,
# 1<<10 = 1024 OTs) — note this exercises the IPC chosen-COT online path too,
# so on a 4-GPU box it may hit the otherDev=dev<4?dev+4:dev-4 topology bug.
# If it segfaults in SETUP/rcot (before online COT) that still answers (A).
#
# Usage:  sudo ./run_cuot_native_test.sh [PORT]
set -u
PORT="${1:-12345}"
BIN="/tmp/test_ferret"

if [[ ! -x "$BIN" ]]; then
  echo "native binary missing: $BIN"; exit 2
fi
cd /home/richorange/dongye/OpenMM
mkdir -p data

pkill -9 -f test_ferret 2>/dev/null || true
sleep 1

S_LOG="$(mktemp -t nat_s.XXXXXX.log)"
R_LOG="$(mktemp -t nat_r.XXXXXX.log)"
trap 'rm -f "$S_LOG" "$R_LOG"' EXIT

echo "=== cuOT NATIVE test_ferret  port=$PORT  (both parties device 0, no wrapper) ==="
# cuOT ferret.sh convention: (sleep 0.05; EXE 1 PORT LOGOT NGPU) & EXE 2 PORT LOGOT NGPU
# LOGOT=10 -> num = 1<<10 = 1024 COTs. NGPU=1.
( sleep 0.05; "$BIN" 1 "$PORT" 10 1 ) > "$S_LOG" 2>&1 &
S_PID=$!
"$BIN" 2 "$PORT" 10 1 > "$R_LOG" 2>&1
R_RC=$?
wait "$S_PID" 2>/dev/null
S_RC=$?

echo "--- sender (party 1) ---"; cat "$S_LOG"
echo "--- receiver (party 2) ---"; cat "$R_LOG"
echo "exit codes:  receiver=$R_RC  sender=$S_RC"
if [[ $R_RC -eq 0 && $S_RC -eq 0 ]] && grep -q "Tests passed" "$R_LOG"; then
  echo "RESULT: native test_ferret PASS -> cuOT works on this box; bug is in the gpu_mm wrapper/device-forcing"
else
  echo "RESULT: native test_ferret FAIL (rc=$R_RC/$S_RC) -> cuOT+CUDA13.2/sm86 incompatibility (not my wrapper)"
fi
