Task ID: M1-T1

Goal:
Introduce HEBackend interface without changing behavior.

Scope:
You may add:
- include/gpu_mm/he_backend.h
- include/gpu_mm/seal_he_backend.h
- include/gpu_mm/seal_he_backend.cc
- tests/he_backend_smoke_test.cpp

Forbidden:
- Do not replace existing HomFCSS/HomBNSS calls yet.
- Do not add PhantomFHE.
- Do not change protocol logic.

Acceptance criteria:
1. SealHEBackend wraps basic SEAL operations:
   - encrypt_symmetric
   - decrypt
   - encode_coeff_plain
   - encode_simd_plain
   - multiply_plain
   - add_plain
   - add_inplace
2. Smoke test passes.
3. No existing code path is changed.

Task ID: M1-T2

Goal:
Introduce OTProvider interface with existing emp/SCI OT as the default backend.

Scope:
You may add:
- include/gpu_mm/ot_provider.h
- include/gpu_mm/emp_ot_provider.h
- include/gpu_mm/emp_ot_provider.cc
- tests/ot_provider_smoke_test.cpp

Forbidden:
- Do not add cuOT.
- Do not modify conversion protocol yet.

Acceptance criteria:
1. Interface supports COT / DeltaOT shape needed by M&M.
2. Existing emp OT can be wrapped.
3. Standalone correlation test passes.


Task ID: M1-T3

Goal:
Introduce CPU ShareTensorBackend as source-of-truth for future CUDA implementation.

Scope:
You may add:
- include/gpu_mm/share_tensor_backend.h
- include/gpu_mm/cpu_share_tensor_backend.h
- include/gpu_mm/cpu_share_tensor_backend.cc
- tests/share_tensor_backend_test.cpp

Acceptance criteria:
1. Support add/sub mod 2^k.
2. Support add/sub mod p.
3. Support scalar multiply mod p.
4. Support sign extraction reference implementation.
5. Randomized tests pass.