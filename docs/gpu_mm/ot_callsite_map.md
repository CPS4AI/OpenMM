# M3-T1 — Source-Level Map of Ferret/OT Usage in M&M

> Read-only task (Task M3: cuOT Integration, M3-T1). No source modified; this
> file is the deliverable. Scope: enumerate every Ferret/OT callsite in M&M so
> M3-T2..T6 can target the right seams. Counterpart to
> [[m1-backend-abstraction]] (the OTProvider interface) and
> [[m2-cuot-standalone]] (the cuOT candidate backend).
>
> **Bottom line up front (BLUF).** M&M's *active* OT backend is
> `sci::OTPack<sci::NetIO>` built with `USE_CHEETAH=1` (Cheetah `SilentOT` =
> Ferret VOLE), constructed once per thread in
> `SCI/src/library_fixed_uniform.cpp:1447` and held in the `otpackArr[]` /
> `otpack` globals (`SCI/src/globals.{h,cpp}`). Every OT callsite dereferences
> one of five `OTPack` handles: `iknp_straight`, `iknp_reversed`,
> `silent_ot(_reversed)`, `kkot[i]`, or `mill->triple_gen`. The gpu_mm
> `OTProvider` interface (M1-T2) exposes **only** the `iknp_straight`-style
> correlated OT (`send_cot`/`recv_cot` over Z_{2^l}, `send_cot_prime`/
> `recv_cot_prime` over Z_p) — i.e. the wrap-bit / logical-OR / B2A /
> conversion shape, **not** Millionaire comparison (kkot) and **not** 1-bit
> `send_ot_cm_cc`. This map records which callsites fall inside the
> `OTProvider` surface (M3-T2..T5 candidates) and which do not (M3-T6
> comparison, out of OTProvider scope).

## 1. What "OT" means in M&M (active backend, verified from source)

| Layer | File | Handle | Underlying primitive |
|---|---|---|---|
| OT backend impl | `SCI/src/OT/ferret/silent_ot.h` | `cheetah::SilentOT<IO>` | Ferret VOLE + MITCCRH hashing → arithmetic COT; also 1-bit `cm_cc`/`rm_rc` |
| OT pack | `SCI/src/OT/cheetah-ot_pack.h` (`USE_CHEETAH=1`) | `otpack->iknp_straight/reversed`, `silent_ot(_reversed)`, `kkot[i]` | `iknp_*` = `silent_ot` aliases (line 58-59); `kkot[i]` = `SilentOTN` over `1<<(i+1)` choices |
| Comparison / triples | `SCI/src/Millionaire/millionaire.h`, `equality.h`, `bit-triple-generator.h` | `mill->compare`, `mill_and_eq->compare_with_eq`, `eq->check_equality`, `mill->triple_gen->generate` | Millionaire = kkot leaf + AND-tree; triples from `_2ROT`/`_16KKOT_to_4OT`/`_8KKOT` |
| Consumers | `SCI/src/BuildingBlocks/aux-protocols.cpp`, `truncation.cpp`, `LinearOT/linear-ot.cpp`, `Math/math-functions.cpp` | call into the above | wrap-bit, B2A, conversion, matmul cross-terms, ReLU/MSB/div/sqrt |

The dispatcher is `SCI/src/OT/ot_pack.h:25` (`#if USE_CHEETAH` → cheetah-ot_pack.h).
Every gemini test/network target compiles with `SCI_OT=1 USE_CHEETAH=1`
(`CMakeLists.txt:84,94,105`), so the CF2 `SplitIKNP`/`SplitKKOT` path
(`cf2-ot_pack.h`) is **not** built. `kkot_4/8/16/beta` appear only inside
`/* … */` comments in `millionaire.h` (dead); the live code uses
`kkot[beta-1]` / `kkot[r-1]` / `kkot[bitlength-1]`.

`gemini`'s FC/Conv/BatchNorm matmuls (`hom_fc_ss.cc`, `hom_conv2d_ss.cc`,
`hom_bn_ss.cc`, `CheetahLinear::fc/conv2d/BOLE` in `cheetah-api.cpp`) are
**HE (SEAL BFV), not OT** — 0 `otpack`/`send_cot` references. The network
`MatMul2D` (`main_resnet50.cpp:317` etc.) routes to `CheetahLinear` (HE). The
OT-based `LinearOT::matrix_multiplication` (`linear-ot.cpp`) is compiled into
`SCI-Cheetah` via `library_fixed.cpp` but is **not** on the gemini network's
hot path; it remains relevant as a batched-COT consumer for completeness.

## 2. OT callsite inventory

Columns: **file:line · function · OT type · modulus · batch size · caller protocol · test**.
"OT type" legend: `COT-l` = correlated OT over Z_{2^l}; `COT-p` = over Z_p;
`1-OT` = 1-out-of-2 (bit) OT; `KKOT-n` = 1-out-of-2^n k-out-of-n OT;
`RCOT` = random correlated OT (block); `Triple` = bit-triple generation.

### 2.1 Wrap-bit / logical-OR / B2A path — `OTProvider` surface (M3-T2 target)

| file:line | function | OT type | modulus | batch | caller protocol | test |
|---|---|---|---|---|---|---|
| `aux-protocols.cpp:80` | `AuxProtocols::multiplexer` (ALICE) | COT-l | Z_{2^bw_y} | `size` | `multiplexer` (y = sel·x) | (none direct; via ReLU/div) |
| `aux-protocols.cpp:81` | `multiplexer` (ALICE, reversed) | COT-l | Z_{2^bw_y} | `size` | multiplexer | — |
| `aux-protocols.cpp:83` | `multiplexer` (BOB) | COT-l | Z_{2^bw_y} | `size` | multiplexer | — |
| `aux-protocols.cpp:84` | `multiplexer` (BOB, reversed) | COT-l | Z_{2^bw_y} | `size` | multiplexer | — |
| `aux-protocols.cpp:110` | `AuxProtocols::B2A` (ALICE) | COT-l | Z_{2^bw_y} | `size` | Boolean→Arithmetic share | truncation `truncate_and_reduce` etc. |
| `aux-protocols.cpp:117` | `B2A` (BOB) | COT-l | Z_{2^bw_y} | `size` | B2A | — |
| `truncation.cpp:351` | `OR_then_B2A` λ (ALICE) | COT-l | Z_{2^bw_y}`=shift_bw` | `dim` | `ring_to_ring` (bit extension, `msb0‖msb1`) | `ring_extension_test.cpp` |
| `truncation.cpp:358` | `OR_then_B2A` λ (BOB) | COT-l | Z_{2^shift_bw} | `dim` | `ring_to_ring` | `ring_extension_test.cpp` |
| `truncation.cpp:412` | `OR_then_B2A` λ (ALICE) | COT-l | Z_{2^shift_bw} | `dim` | `ring_to_ring128` | `ring_extension_test.cpp`, `cpu_baseline_profile.cpp` |
| `truncation.cpp:419` | `OR_then_B2A` λ (BOB) | COT-l | Z_{2^shift_bw} | `dim` | `ring_to_ring128` | as above |

**Logical-OR semantics.** `OR_then_B2A` (truncation.cpp:344, 405) computes the
shared OR of the two parties' MSBs (`wrap = msb0 ‖ msb1`, per the inline
comment `// wrap = (msb0 || msb1)`) via a single COT-l whose correlation is
`input_x[i] ^ 1`, then local-adds to land in arithmetic. This is the
"wrap-bit / logical OR" path M3-T2 names. It calls **exactly**
`iknp_straight->send_cot/recv_cot` — the `OTProvider::send_cot/recv_cot`
contract — so it is the natural OTProvider seam.

### 2.2 Conversion (ring⇄field) OR path — `OTProvider` surface (M3-T4/T5 target)

| file:line | function | OT type | modulus | batch | caller protocol | test |
|---|---|---|---|---|---|---|
| `truncation.cpp:549` | `OR_AUX` λ (ALICE) | COT-p | Z_p (field_mod) | `dim` | `ring_to_field` | `ring_to_field_test.cpp`, `bole_test.cpp:132`, `cpu_baseline_profile.cpp:106` |
| `truncation.cpp:557` | `OR_AUX` λ (BOB) | COT-p | Z_p | `dim` | `ring_to_field` | as above |
| `truncation.cpp:607` | `OR_AUX` λ (ALICE) | COT-l | Z_{2^ring_bw} | `dim` | `field_to_ring` | `field_to_ring_test.cpp`, `bole_test.cpp:137` |
| `truncation.cpp:614` | `OR_AUX` λ (BOB) | COT-l | Z_{2^ring_bw} | `dim` | `field_to_ring` | as above |
| `truncation.cpp:665` | `OR_AUX` λ (ALICE) | COT-l | Z_{2^ring_bw} | `dim` | `field_to_ring_with_truncate` | `bole_test.cpp:30,496` |
| `truncation.cpp:672` | `OR_AUX` λ (BOB) | COT-l | Z_{2^ring_bw} | `dim` | `field_to_ring_with_truncate` | as above |

`ring_to_field`'s `OR_AUX` (truncation.cpp:543) is the **only COT-p** callsite
in M&M — it maps onto `OTProvider::send_cot_prime/recv_cot_prime`. M3-T4/T5
target these; M3-T2 explicitly forbids touching the conversion *equations*
(the surrounding big-number / secure_add-sub logic stays).

### 2.3 Comparison / Millionaire path — **outside** `OTProvider` (M3-T6)

| file:line | function | OT type | modulus | batch | caller protocol | test |
|---|---|---|---|---|---|---|
| `aux-protocols.cpp:57` | `wrap_computation` | KKOT (via `mill->compare`) | 1-bit out | `size`×digits | wrap-bit of `x` (≥/2^k) | (via truncate) |
| `aux-protocols.cpp:181` | `MSB` | KKOT (`mill->compare`, bw-1) | 1-bit | `size`×digits | MSB(x) | ReLU/div/sqrt |
| `millionaire.h:101,103` | `compare` leaf (≤beta) send | KKOT-(2^beta)/1-OT | 1-bit | `num_cmps` | Millionaire compare | (via MSB/ReLU) |
| `millionaire.h:115,117` | `compare` leaf recv | KKOT/1-OT | 1-bit | `num_cmps` | compare | — |
| `millionaire.h:200-241` | `compare` interior send | KKOT-beta / kkot[r-1] | 2-bit | `num_cmps·num_digits` | compare tree | — |
| `millionaire.h:255-300` | `compare` interior recv | KKOT-beta / kkot[r-1] | 2-bit | `num_cmps·num_digits` | compare tree | — |
| `equality.h:98-114` | `check_equality` leaf | KKOT/1-OT | 1-bit | `num_eqs` | zero-test | (via truncate div_correction) |
| `millionaire_with_equality.h:101-300` | `compare_with_eq` | KKOT-beta / kkot[r-1] + Triple | 2-bit | `num_cmps·num_digits` | compare+eq | (via MSB_to_Wrap, digit_decomp) |
| `bit-triple-generator.h:184-190` | triple gen (Cheetah) | RCOT (`send/recv_ot_rm_rc`) | 1-bit | `num_triples` | AND triples | `aux->AND` |
| `aux-protocols.cpp:308` | `AND` | Triple (`_16KKOT_to_4OT`) | 1-bit | `size` | bitwise AND | ReLU six-comp, div_correction |
| `millionaire.h:354-360` | `traverse_and_compute_ANDs` | Triple (`_2ROT` Cheetah / `_16KKOT_to_4OT`) | 1-bit | `num_cmps` | compare AND-tree | — |

These use `kkot[]`, `silent_ot->send_ot_rm_rc`, and `triple_gen->generate` —
**not** the `OTProvider` COT interface. M3-T6 (comparison) owns them; cuOT's
`CuotProvider` does not implement KKOT or rotated-message RC, so they stay on
emp/SCI regardless of the `--ot-backend` flag (documented limitation).

### 2.4 1-bit COT (Cheetah `cm_cc`) — `msb0/msb1_to_wrap`

| file:line | function | OT type | modulus | batch | caller protocol | test |
|---|---|---|---|---|---|---|
| `aux-protocols.cpp:238,251` | `msb0_to_wrap` | 1-OT (COT 1-bit, `send_ot_cm_cc`) | 1-bit | `size` | wrap when MSB=0 known | `truncate` (cheetah `apply_msb0_heuristic`) |
| `aux-protocols.cpp:240,253` | `msb0_to_wrap` (non-Cheetah) | 1-OT (`iknp_straight->send/recv`) | 1-bit | `size` | msb0 wrap | — |
| `aux-protocols.cpp:274,288` | `msb1_to_wrap` | 1-OT (`send_ot_cm_cc`) | 1-bit | `size` | wrap when MSB=1 known | `truncate` |
| `aux-protocols.cpp:277,290` | `msb1_to_wrap` (non-Cheetah) | 1-OT | 1-bit | `size` | msb1 wrap | — |

`send_ot_cm_cc<uint8_t>` is a 1-bit chosen-COT (SCI `SilentOT::send_ot_cm_cc`,
`silent_ot.h:301`) — a *different* shape from `OTProvider::send_cot` (which is
l-bit). It is **not** on the `OTProvider` surface; M3-T2 leaves it on emp.

### 2.5 LinearOT matmul cross-terms — batched COT (not on gemini hot path)

| file:line | function | OT type | modulus | batch | caller protocol | test |
|---|---|---|---|---|---|---|
| `linear-ot.cpp:254-284` | `matmul_cross_terms` | COT-l (batched) | Z_{2^bwC} | `msg_len`×msgs | `matrix_multiplication` (SCI OT matmul) | (SCI library; gemini uses HE) |
| `linear-ot.cpp:450-462` | `matmul_multiplexer` | GOT-l (batched) | Z_{2^bwC} | `dim`×msgs | matmul mux | — |

`send_batched_cot/recv_batched_cot` (`silent_ot.h:631,637`) pack many l-bit
COTs per OT message. Gemini's `MatMul2D` routes to HE `CheetahLinear::fc`
(`library_fixed_uniform_cheetah.cpp:71`), so this path is dormant for the ML
networks; recorded for completeness — **not** an M3 target.

## 3. OT primitive surface vs. `OTProvider` interface

| `OTProvider` method (M1-T2) | SCI callsite it replaces | M&M caller | M3 task |
|---|---|---|---|
| `send_cot / recv_cot` (Z_{2^l}) | `iknp_straight->send_cot/recv_cot` | `B2A`, `multiplexer`, `OR_then_B2A`, `OR_AUX` (field_to_ring) | T2 (wrap-bit OR), T5 (field_to_ring) |
| `send_cot_prime / recv_cot_prime` (Z_p) | `iknp_straight->send_cot_prime/recv_cot_prime` | `OR_AUX` (ring_to_field) | T4 (ring_to_field) |
| — (not exposed) | `iknp_straight->send/recv` (1-OT) | `msb0/msb1_to_wrap` fallback | T2 leaves on emp |
| — (not exposed) | `silent_ot->send_ot_cm_cc/recv_ot_cm_cc` (1-bit COT) | `msb0/msb1_to_wrap` (Cheetah) | T2 leaves on emp |
| — (not exposed) | `kkot[]->send/recv` | Millionaire `compare`/`check_equality` | T6 (comparison) |
| — (not exposed) | `triple_gen->generate`, `send_ot_rm_rc` | `AND`, compare tree | T6 |

**Implication for M3-T2.** The wrap-bit / logical-OR path that maps cleanly
onto `OTProvider` is the `OR_then_B2A`/`B2A`/`multiplexer` COT-l shape (§2.1).
`wrap_computation` and `MSB` route through `mill->compare` (§2.3) and are
explicitly *forbidden* to modify in M3-T2 ("Do not modify comparison"), so they
stay on emp regardless of `--ot-backend`. The 1-bit `msb0/msb1_to_wrap` (§2.4)
uses a shape `OTProvider` does not expose; they too stay on emp. M3-T2's
"wrap-bit logical OR path" is therefore scoped to the **COT-l OR primitive**
(`B2A` + a shared-bit OR), implemented in the gpu_mm layer behind
`OTProvider` (emp default, cuOT behind `--ot-backend`), without editing the SCI
conversion equations.

## 4. Construction & wiring (where OT is actually built)

- **Globals**: `otpack` / `otpackArr[MAX_THREADS]`, `truncation(…Arr)`,
  `mult(…Arr)` (= `LinearOT`), `math(…Arr)`, `aux(…Arr)` — declared
  `SCI/src/globals.h:58-98`, defined `SCI/src/globals.cpp:25-66`.
- **Allocation**: `SCI/src/library_fixed_uniform.cpp:1447` (`new
  sci::OTPack<sci::NetIO>(ioArr[i], …)`) and `:1461-1552` (`new LinearOT/
  Truncation/MathFunctions/AuxProtocols`). `USE_CHEETAH=1` ⇒ Cheetah
  `SilentOT`/`SilentOTN`.
- **CMake**: `SCI/src/CMakeLists.txt:41` builds `SCI-Cheetah` from
  `library_fixed_uniform_cheetah.cpp + library_fixed_uniform.cpp +
  library_fixed.cpp + globals.cpp`; `:43` sets `SCI_OT=1 USE_CHEETAH=1`.
  `CMakeLists.txt:83-105` links every gemini target to `gemini SCI-Cheetah`
  with the same defs.
- **gpu_mm seam (M1)**: `EmpOTProvider` (`include/gpu_mm/emp_ot_provider.cc`)
  owns its own `sci::NetIO` + `sci::OTPack` and wraps `iknp_straight->{
  send,recv}_cot{_prime}` behind `OTProvider`. `CuotProvider` (M2) implements
  the same four methods via the MITCCRH-on-RCOT adapter. Neither is wired into
  SCI yet — they are standalone, test-only.

## 5. Tests that cover each OT callsite

| Test binary | Source | Covers |
|---|---|---|
| `ring_to_field_test` | `tests/ring_to_field_test.cpp` | `Truncation::ring_to_field` → `OR_AUX` COT-p (§2.2). bw∈{16,32,40,48,56,60,64}, N=1e7 |
| `field_to_ring_test` | `tests/field_to_ring_test.cpp` | `Truncation::field_to_ring` → `OR_AUX` COT-l (§2.2) |
| `ring_extension_test` | `tests/ring_extension_test.cpp` | `Truncation::ring_to_ring128` → `OR_then_B2A` COT-l (§2.1) |
| `bole_test` | `tests/bole_test.cpp` | `ring_to_field`+`field_to_ring`+`field_to_ring_with_truncate`+BOLE(HE). kScale=12 |
| `cpu_baseline_profile` | `tests/cpu_baseline_profile.cpp` | ring_to_field/field_to_ring/ring_to_ring128/BOLE profiled under the gpu_mm profiler (M0) |
| `ot_provider_smoke_test` | `tests/ot_provider_smoke_test.cpp` | `EmpOTProvider` COT-l + COT-p correlation (M1 seam, not SCI) |
| `cuot_correlation_test` | `tests/cuot_correlation_test.cpp` | `CuotProvider` RCOT + block chosen-COT (M2 seam) |
| `cuot_deltaot_arithmetic_test` | `tests/cuot_deltaot_arithmetic_test.cpp` | `CuotProvider` COT-l + COT-p adapter (M2-T3 seam) |

No standalone test directly exercises `AuxProtocols::B2A`/`multiplexer`/`AND`/
`wrap_computation`/`MSB` in isolation; they are reached transitively through
`Truncation::truncate*` and `MathFunctions::ReLU/div/sqrt`. **Gap for M3-T2**:
the "new wrap-bit boundary test" acceptance criterion is not yet covered by an
existing test — M3-T2 must add one (mirroring `ot_provider_smoke_test`'s
send-x-back verification idiom).

## 6. Scope decisions for M3-T2..T6 (derived from this map)

- **M3-T2 (wrap-bit OR)**: target the COT-l `OR_then_B2A`/`B2A` shape via
  `OTProvider`; default emp; cuOT behind `--ot-backend`. Do **not** touch
  `wrap_computation`/`MSB` (comparison) or `msb0/msb1_to_wrap` (1-bit COT, not
  in `OTProvider` surface). Add a boundary test {0, 1, -1, M/2-1, M/2, M-1}.
- **M3-T4 (ring_to_field)**: route `OR_AUX` COT-p through `OTProvider`
  `send_cot_prime`/`recv_cot_prime`; keep the big-number/secure_add-sub
  equations byte-identical.
- **M3-T5 (field_to_ring)**: route `OR_AUX` COT-l through `send_cot`/`recv_cot`.
- **M3-T6 (comparison)**: `compare`/`MSB`/`AND` use KKOT + triples — outside
  `OTProvider`. cuOT cannot back these; they stay emp. This task is gated on a
  kkot/triple adapter that does not exist yet (honest limitation, do not fake).

## 7. Reproducibility

```bash
# Reproduce the callsite tallies (from repo root):
grep -rhoE "otpack->[a-z_0-9]+->[a-z_0-9]+(_prime)?\(|mill(->triple_gen)?->[a-z_0-9]+\(|mill_and_eq->[a-z_0-9]+\(" \
  SCI/src/BuildingBlocks/ SCI/src/Millionaire/ SCI/src/Math/ SCI/src/LinearOT/ | sort | uniq -c | sort -rn

# Confirm gemini matmul is HE (expect 0 OT hits):
grep -c "otpack\|send_cot\|recv_cot" include/gemini/cheetah/hom_fc_ss.cc include/gemini/cheetah/hom_conv2d_ss.cc include/gemini/cheetah/hom_bn_ss.cc

# Confirm active OT pack flavor:
grep -n "USE_CHEETAH\|cheetah-ot_pack\|cf2-ot_pack" SCI/src/OT/ot_pack.h
```

This map is the M3-T1 deliverable. No source files were modified; only this
document was added. It feeds M3-T2 (wrap-bit OR seam) and bounds M3-T4/T5/T6.
