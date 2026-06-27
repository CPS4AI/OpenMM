# Backend Abstraction — Change Record (M1 series)

> Status: complete. Branch `GPU-MM`. Date: 2026-06-25.
> This document records the M1 deliverable: three backend-abstraction seams
> (HE / OT / ShareTensor) introduced **without changing any existing protocol
> behavior**, each validated by a standalone test. It is the M1-series
> counterpart to [[cpu-baseline-profiling]] (the M0 measurement record) and is
> the reference later cuOT/Phantom/CUDA backends must match.

## 1. Purpose

CLAUDE.md mandates a staged migration toward a GPU-resident PPML runtime
(`cuOT Ferret backend + PhantomFHE HE backend + CUDA local share tensor
backend + GPU ML blocks`). The staged order is:

1. Profiling utility — ✅ M0-T1
2. CPU baseline profiling — ✅ M0-T2 (see [[cpu-baseline-profiling]])
3. **Backend interfaces without changing behavior — ✅ M1 (this doc)**
4. Standalone cuOT tests
5. Standalone PhantomFHE tests
6. Standalone CUDA share-tensor tests
7. Replace one primitive behind a flag
8. …

Step 3 introduces *seams*: abstract interfaces (`gpu_mm::HEBackend`,
`gpu_mm::OTProvider`, `gpu_mm::ShareTensorBackend`) plus one CPU/emp
reference implementation of each, compiled into the existing `gemini`
library / test exes. The existing protocol code (`HomFCSS`, `HomBNSS`,
`CheetahLinear`, all ring↔field / `ring_extension` / `BOLE` conversions)
is **untouched**. Later GPU backends implement the same interface and are
A/B-tested against this CPU reference, never replacing it unless a flag is set.

### Task-size compliance (CLAUDE.md)

The three M1 tasks were deliberately split so that each stays within the
3-existing-files / 2-new-test-files / 1-build-file budget, and **no single task
modifies both OT and HE** nor mixes correctness logic with performance work:

| Task | New interface/impl files | New test files | Modified existing files | OT&HE together? |
|---|---|---|---|---|
| M1-T1 HEBackend + SealHEBackend | 3 (`he_backend.h`, `seal_he_backend.{h,cc}`) | 1 (`he_backend_smoke_test.cpp`) | `include/CMakeLists.txt` (+`gpu_mm` subdir + its `CMakeLists.txt`) | no (HE only) |
| M1-T2 OTProvider + EmpOTProvider | 3 (`ot_provider.h`, `emp_ot_provider.{h,cc}`) | 1 (`ot_provider_smoke_test.cpp`) | `CMakeLists.txt` (`add_ot_test` macro + `add_test`) | no (OT only) |
| M1-T3 ShareTensorBackend + CPU | 3 (`share_tensor_backend.h`, `cpu_share_tensor_backend.{h,cc}`) | 1 (`share_tensor_backend_test.cpp`) | `include/gpu_mm/CMakeLists.txt` (add `.cc` to sources) | no (share only) |

The `gpu_mm` subdirectory and its `CMakeLists.txt` were created once (M1-T1)
and merely *added to* by M1-T3; no protocol source files were modified in any
task. Default backend remains CPU; every new backend is behind a plain
`gpu_mm` namespace / test-only path (no compile/runtime GPU flag is live yet).

## 2. Interfaces (what a future backend must implement)

All three live in `namespace gpu_mm`, header-only abstract bases with a
status enum and `void*`-typed opaque handles (so the abstract header pulls no
SEAL/SCI/cuOT dependency):

### 2.1 HEBackend (`include/gpu_mm/he_backend.h`)

```cpp
enum class HEEncoding { kCoeff, kSIMD };   // coeff=FC/Conv, simd=BOLE/OLE/BN
enum class HEStatus { kOk, kInvalidArg, kInternal, ... };

class HEBackend {
 public:
  virtual ~HEBackend() = default;
  virtual const char* backend_name() const = 0;
  virtual HEStatus setup(int N, uint64_t plain_mod,
                         const std::vector<int>& coeff_mod_bits) = 0;
  // encode / encrypt / decrypt / multiply_plain / add_plain / add_inplace ...
  // (convenience wrappers encode_coeff_plain / encode_simd_plain select encoding)
};
```

**Known architecture fact (enforced by the SEAL impl + tests):**
FC / matrix-vector / inner product uses Cheetah-style **coefficient encoding**
(no `BatchEncoder`; `multiply_plain` is a polynomial *convolution*, not
pointwise). Element-wise BOLE / OLE / BN uses BFV SIMD **BatchEncoder**
(pointwise). The two encodings must not be conflated — the `HEEncoding`
enum exists precisely to keep them apart.

### 2.2 OTProvider (`include/gpu_mm/ot_provider.h`)

```cpp
enum class OTParty { kSender = 1, kReceiver = 2 };
enum class OTStatus { kOk, kInvalidArg, kInternal, ... };

class OTProvider {
 public:
  virtual ~OTProvider() = default;
  virtual const char* backend_name() const = 0;
  virtual OTParty party() const = 0;
  // COT correlation (matches sci::SilentOT):
  //   sender gets random x; receiver with bit b gets x (b=0) or x+corr (b=1).
  virtual OTStatus send_cot(uint64_t* data0, const uint64_t* corr, int n, int l) = 0;
  virtual OTStatus recv_cot(uint64_t* data, const bool* b, int n, int l) = 0;
  // same correlation, but mod a prime p (field COT)
  virtual OTStatus send_cot_prime(uint64_t* data0, const uint64_t* corr, int n, uint64_t p) = 0;
  virtual OTStatus recv_cot_prime(uint64_t* data, const bool* b, int n, uint64_t p) = 0;
};
```

Two modulus flavors because M&M uses both: ring COT (mod 2^l) for
`Truncation::ring_to_field` style work, and field COT (mod p) for prime-field
conversion. `EmpOTProvider` delegates to `otpack_->iknp_straight->{send,recv}_cot{,_prime}`.

### 2.3 ShareTensorBackend (`include/gpu_mm/share_tensor_backend.h`)

```cpp
enum class STStatus { kOk, kInvalidArg, ... };

class ShareTensorBackend {
 public:
  virtual ~ShareTensorBackend() = default;
  virtual const char* backend_name() const = 0;
  // ring (mod 2^k)
  virtual STStatus add_mod2k(...) = 0;
  virtual STStatus sub_mod2k(...) = 0;
  // field (mod p)
  virtual STStatus add_modp(...) = 0;
  virtual STStatus sub_modp(...) = 0;
  virtual STStatus scalar_mul_modp(...) = 0;
  // two's-complement sign extraction in Z_{2^k} (mirrors sci::signed_val)
  virtual STStatus sign_extract(const uint64_t* a, int n, int k, uint64_t* out) = 0;
};
```

`sign_extract` is the subtle one: it must mirror `sci::signed_val` exactly,
including the `pow_x = 0` special case at `k == 64` (there 2^64 is
unrepresentable in `uint64_t`, so the bit pattern *is* the signed value and
the op is the identity). A future CUDA kernel must produce the same bits.

## 3. Reference implementations (CPU / emp, source-of-truth)

| Backend | Impl | Backend name | Mirrors |
|---|---|---|---|
| HE | `SealHEBackend` (`seal_he_backend.{h,cc}`) | `"seal"` | M&M's `HomFCSS`/`HomBNSS` SEAL setup: BFV, coeff-mod bits `{60,49}`, `modulo_poly_coeffs` for coeff packing (like `vec2PolyBFV`), `BatchEncoder` for SIMD. `PublicKey` materialized via stringstream round-trip. |
| OT | `EmpOTProvider` (`emp_ot_provider.{h,cc}`) | `"emp"` | M&M's `sci::NetIO` + `sci::OTPack<sci::NetIO>` (Ferret VOLE via `USE_CHEETAH`). ALICE=server (`addr=nullptr`), BOB=client. Calls `iknp_straight->send_cot/recv_cot(_prime)`, `io->flush()` after send. |
| ShareTensor | `CpuShareTensorBackend` (`cpu_share_tensor_backend.{h,cc}`) | `"cpu"` | M&M's mod semantics: wrapping mod 2^k for rings, `__uint128_t` mod p for fields, and `sci::signed_val` for `sign_extract`. |

None of these *replace* a protocol path; they are parallel, test-only seams.

## 4. Build wiring

The non-obvious part is the `-maes` flag propagation. M&M's OT headers
(`utils/aes-ni.h`) hard-`#error` without AES-NI. The `gemini` shared library
does **not** carry SCI's ISA flags, so `emp_ot_provider.cc` (which transitively
pulls `aes-ni.h`) cannot be a `gemini` source. Resolution:

- `include/gpu_mm/CMakeLists.txt` lists only `seal_he_backend.cc` and
  `cpu_share_tensor_backend.cc` as `gemini PUBLIC` sources.
- `emp_ot_provider.cc` is compiled **directly into the OT test executable** via
  the `add_ot_test` macro, which links `SCI-Cheetah` and thus inherits `-maes`
  transitively through the SCI-utils INTERFACE library chain.

```cmake
# top-level CMakeLists.txt
macro (add_ot_test _name)
  add_executable(${_name} "tests/${_name}.cpp" "include/gpu_mm/emp_ot_provider.cc")
  target_link_libraries(${_name} gemini SCI-Cheetah)
  target_compile_definitions(${_name} PUBLIC SCI_OT=1 USE_CHEETAH=1)
endmacro()
add_ot_test(ot_provider_smoke_test)
add_test(share_tensor_backend_test)   # and he_backend_smoke_test, etc.
```

No existing `add_test`/`add_network` macro was changed; only new targets added.

## 5. Validation evidence

All three tests built and passed on this machine (AMD EPYC 7443P, g++ 11.4,
ABI=0, same env as [[cpu-baseline-profiling]]).

### Configure + build

```text
$ cmake -B build -DABI=0
CFG_RC=0
$ cmake --build build -j --target he_backend_smoke_test \
        ot_provider_smoke_test share_tensor_backend_test
[ 77%] Built target SCI-Cheetah
Consolidate compiler generated dependencies of target share_tensor_backend_test
[100%] Built target share_tensor_backend_test
BUILD_RC=0
```

### M1-T1 — `he_backend_smoke_test` (single-process, no network)

Tests **both** encodings against a cleartext reference computed with
`__uint128_t` (independent of the backend's own helpers):

- **coeff path**: `encode_plain` → encrypt → `multiply_plain` → `add_plain` →
  `add_inplace` → decrypt. Reference uses `conv_coeff()` (a *polynomial
  convolution*, because coefficient packing makes `multiply_plain` a
  convolution, not pointwise — this caught an early test-logic bug where a
  pointwise ref gave `got=65527 want=65528`).
- **simd path**: reconstruct via `BatchEncoder`, `mul_mod` uses
  double-and-add (not `__uint128_t`), pointwise reference.

```text
$ ./build/bin/he_backend_smoke_test
[coeff] encrypt/decrypt, multiply_plain, add_plain, add_inplace OK
[simd]  encrypt/decrypt, multiply_plain, add_plain OK
HE backend smoke test PASSED
HE_RC=0
```

### M1-T3 — `share_tensor_backend_test` (single-process, randomized)

Randomized (xorshift `Rng`, seed `0x123456789abcdef`) against a cleartext
reference, `n = 4096`:

- ring `add/sub mod 2^k` for `k ∈ {1,8,16,32,37,64}`, incl. in-place alias
  (`out == a`);
- field `add/sub/scalar_mul mod p` for `p ∈ {65537, 4293918721, 1099511480321, 65521}`;
- `sign_extract` for `k ∈ {1,8,16,32,64}`; for `k < 64` a nonneg/neg branch
  cross-check, for `k == 64` only the exact `sci::signed_val` value
  (`pow_x=0` ⇒ identity) since the branch distinction is unrepresentable;
- invalid-arg paths (`nullptr`, `k=0`, `p<2`).

```text
$ ./build/bin/share_tensor_backend_test
Share-tensor backend test PASSED (cpu, n=4096)
ST_RC=0
```

### M1-T2 — `ot_provider_smoke_test` (two-party, **needs sudo**)

Two-party. Party 1 = sender (ALICE, server), party 2 = receiver (BOB,
client). Verifies the COT correlation for **both** modulus flavors:

```
(recv[i] - data0[i]) mod m == (b[i] ? corr[i] : 0)   for all i
```

The sender sends its secret `x = data0` back to the receiver over the
provider's own `net_io()` purely so the test can check the correlation
(`x` is secret in real use).

> **Machine-specific note (memory `openmm-build-env` §4):** on this box every
> test that opens a listening TCP socket fails at startup with
> `error: bind: Permission denied` (EACCES) for *every* port — including
> unprivileged ones — even though a trivial libc-only bind probe on the same
> port succeeds and `Seccomp=0`. This is a per-binary network restriction in
> the execution environment, not a code/port problem. **It runs cleanly under
> `sudo`.** The helper `./run_ot_test.sh [PORT]` (default 32000) launches
> party 1 then party 2 on `127.0.0.1` and prints `RESULT: PASS` iff
> `OT provider smoke test PASSED` appears. Run as `sudo ./run_ot_test.sh`.

Actual run (user, sudo, port 32000):

```text
$ sudo ./run_ot_test.sh 32000
=== OT provider smoke test  host=127.0.0.1 port=32000 ===
--- sender (party 1) ---
[sender] COT + COT' sent, n=1000, l=32, p=4293918721
--- receiver (party 2) ---
[receiver] COT + COT' correlation OK, n=1000, l=32, p=4293918721
OT provider smoke test PASSED
exit codes:  receiver=0  sender=0
RESULT: PASS
```

Both the ring COT (mod 2^32) and the field COT (mod p=4293918721)
correlations hold across all `n=1000` choices.

## 6. Change manifest

### New files (interface + impl + tests, all `gpu_mm` namespace, CPU default)

- `include/gpu_mm/he_backend.h`, `seal_he_backend.h`, `seal_he_backend.cc`
  (M1-T1)
- `include/gpu_mm/ot_provider.h`, `emp_ot_provider.h`, `emp_ot_provider.cc`
  (M1-T2)
- `include/gpu_mm/share_tensor_backend.h`, `cpu_share_tensor_backend.h`,
  `cpu_share_tensor_backend.cc` (M1-T3)
- `include/gpu_mm/CMakeLists.txt` (lists only `seal_he_backend.cc` +
  `cpu_share_tensor_backend.cc` as `gemini` sources; comment explains why
  `emp_ot_provider.cc` is absent — needs `-maes`)
- `tests/he_backend_smoke_test.cpp`, `tests/ot_provider_smoke_test.cpp`,
  `tests/share_tensor_backend_test.cpp`
- `run_ot_test.sh` (two-party OT launcher; runs under `sudo` on this box)

### Modified files (build wiring only, no logic)

- `CMakeLists.txt` — added the `add_ot_test` macro and four `add_test`
  entries (`profiler_smoke_test`, `cpu_baseline_profile`,
  `he_backend_smoke_test`, `ot_provider_smoke_test`,
  `share_tensor_backend_test`).
- `include/CMakeLists.txt` — added `add_subdirectory(gpu_mm)`.

### Untouched (the correctness baseline)

`HomFCSS` / `HomBNSS` / `CheetahLinear` and every conversion protocol
(`ring_to_field`, `field_to_ring`, `ring_extension`, `BOLE`) — **not modified
in any M1 task**. The `data/pre_ot_data_reg_*` Ferret cache files are runtime
artifacts rewritten by test runs; restore with `git checkout -- data/` before
committing.

## 7. Reproducibility

```bash
# from repo root (Ferret needs ./data; two-party tests need sudo for bind)
cmake -B build -DABI=0
cmake --build build -j --target he_backend_smoke_test \
        ot_provider_smoke_test share_tensor_backend_test

./build/bin/he_backend_smoke_test          # expect: HE backend smoke test PASSED
./build/bin/share_tensor_backend_test      # expect: Share-tensor backend test PASSED (cpu, n=4096)
sudo ./run_ot_test.sh 32000                # expect: OT provider smoke test PASSED
git checkout -- data/                      # restore Ferret cache before commit
```

## 8. What this unlocks (next per CLAUDE.md order)

The seams exist; the CPU/emp reference is bit-for-bit fixed. The next steps
(4–6) can each proceed **independently and behind their own flags**:

- **Step 4 — standalone cuOT tests**: implement a `cuOTProvider :
  OTProvider` behind `OPENMM_ENABLE_CUOT`, and a standalone correlation test
  that must match `EmpOTProvider`'s COT correlation (§2.2) before it can be
  considered as an OT path replacement.
- **Step 5 — standalone PhantomFHE tests**: implement a `PhantomHEBackend :
  HEBackend` behind `OPENMM_ENABLE_PHANTOM`, with SEAL-vs-Phantom equality
  tests on both encodings (§2.1) before replacing any HE path.
- **Step 6 — standalone CUDA share-tensor tests**: implement a
  `CudaShareTensorBackend : ShareTensorBackend` behind
  `OPENMM_ENABLE_CUDA_SHARE`, matching `CpuShareTensorBackend` including the
  `sign_extract` `k==64` identity (§2.3).

No full-system GPU execution until each primitive has cleared its standalone
match-against-CPU gate.
