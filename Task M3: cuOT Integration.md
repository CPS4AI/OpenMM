Frist wrap-bit / logical OR，then conversion，finally comparison。

Task ID: M3-T1

Goal:
Produce a source-level map of all Ferret/OT usage in M&M.

Scope:
Read-only task. Do not modify source except adding docs:
- docs/gpu_mm/ot_callsite_map.md

Acceptance criteria:
Document:
1. file path
2. function name
3. OT type
4. modulus
5. batch size
6. caller protocol
7. test that covers it

Task ID: M3-T2

Goal:
Use OTProvider abstraction in wrap-bit logical OR path, defaulting to CPU emp backend.

Scope:
You may modify only the specific wrap-bit / auxiliary protocol files identified in M3-T1.

Forbidden:
- Do not enable cuOT by default.
- Do not modify comparison.
- Do not modify ring_to_field / field_to_ring equations.

Acceptance criteria:
1. CPU EmpOTProvider path produces identical results as original.
2. Existing conversion tests pass.
3. New wrap-bit boundary test passes.

Task ID: M3-T3

Goal:
Enable CuOTProvider only for wrap-bit OR path behind runtime flag.

Acceptance criteria:
1. CPU backend and cuOT backend both pass same wrap-bit tests.
2. Boundary cases pass:
   - 0
   - 1
   - -1
   - M/2 - 1
   - M/2
   - M - 1
3. Random 100000 samples pass.

Task ID: M3-T4

Goal: 
cuOT ring_to_field

- CPU Ferret backend pass
- cuOT backend pass
- reconstruction correctness
- random stress
- same communication count shape where applicable

Task ID: M3-T5

Goal: 
cuOT field_to_ring

- CPU Ferret backend pass
- cuOT backend pass
- reconstruction correctness
- random stress
- same communication count shape where applicable

Task ID: M3-T6

Goal: 
cuOT comparison

- DReLU / comparison primitive pass
- random signed values pass
- boundary values pass
- nonlinear block smoke test pass