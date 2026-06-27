cuOT github: https://github.com/Andrew-Gan/cuOT

You download and place it in deps as cuOT.


Task ID: M2-T1

Goal:
Add optional build flag OPENMM_ENABLE_CUOT.

Scope:
You may modify:
- CMakeLists.txt
- scripts/build.sh
You may add:
- cmake/FindcuOT.cmake or equivalent minimal config

Forbidden:
- Do not change protocol code.
- Build must work when OPENMM_ENABLE_CUOT=OFF.

Acceptance criteria:
1. Default build without cuOT still passes.
2. With flag OFF, no CUDA/cuOT dependency is required.
3. Print clear message whether cuOT is enabled.

Task ID: M2-T2

Goal:
Add standalone cuOT correlation tests.

Scope:
You may add:
- include/gpu_mm/cuot_provider.h
- include/gpu_mm/cuot_provider.cc
- tests/cuot_correlation_test.cpp

Forbidden:
- Do not call M&M conversion code.
- Do not change existing OT code.

Acceptance criteria:
1. Test RCOT/COT correlation.
2. Test multiple batch sizes: 1, 2, 17, 1024, 4096.
3. Test both party roles.
4. Test deterministic seed mode if available.
5. Test fails loudly on role mismatch or correlation mismatch.


Task ID: M2-T3

Goal:
Build arithmetic DeltaOT adapter on top of cuOT output.

Scope:
You may modify:
- include/gpu_mm/cuot_provider.h
- include/gpu_mm/cuot_provider.cc
You may add:
- tests/cuot_deltaot_arithmetic_test.cpp

Acceptance criteria:
For modulus K in {2^32, 2^64, selected prime p}:
1. Receiver obtains y_b.
2. Sender messages satisfy y_1 - y_0 = Delta mod K.
3. Randomized test passes for at least 10000 samples.