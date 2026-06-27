# CPU Baseline Profiling — Research Record (M0-T2)

> Status: complete. Branch `GPU-MM`. Date: 2026-06-25.
> This document records the measured CPU baseline of every `/tests` benchmark so
> that later GPU/cuOT/Phantom/CUDA replacements can be compared against a fixed,
> reproducible reference. It is the deliverable of task M0-T2 and supersedes the
> inline summary.

## 1. Purpose

The long-term goal (CLAUDE.md) is a GPU-resident mixed-modulus PPML runtime. The
existing CPU implementation is the correctness **and** performance baseline.
Before any GPU/OT/HE backend is swapped in, we need a wall-clock breakdown of
each CPU primitive, decomposed by phase, measured with one consistent harness —
the `gpu_mm::Profiler` introduced in M0-T1.

This document captures that breakdown for all four `/tests` benchmarks:
`ring_to_field`, `field_to_ring`, `ring_extension`, and `bole`.

## 2. Harness

- **Profiler**: `include/gpu_mm/profiler.h` (M0-T1). RAII `TimerScope` +
  thread-safe `Profiler` static facade (`Reset`/`Enable`/`PrintSummary`/`Record`).
  Disabled by default; µs internally, reported in ms.
- **Benchmark driver**: `tests/cpu_baseline_profile.cpp`. One process per party.
  Each primitive: 1 warm-up round (profiler **off**) → `Reset()`+`Enable(true)`
  → 3 measured rounds → `PrintSummary`.
- **Per-iteration regions** (each primitive builds a fresh `OTPack`/
  `AuxProtocols`/`Truncation` — and `CheetahLinear` for `bole` — exactly like the
  original tests, so setup cost is visible):
  - `*.otpack_setup` — `OTPack` construction (base OT + Ferret VOLE bootstrap)
  - `*.aux_setup` — `AuxProtocols` + `Truncation` (+ `CheetahLinear` for bole)
  - `*.<primitive>` — the primitive under test
  - `*.total` — full iteration (sanity check ≈ sum of phases)
- Each `run_*` function mirrors the corresponding `tests/*.cpp` call-for-call;
  no protocol source files were modified. Default CPU path only; no CUDA/cuOT/
  Phantom.

## 3. Environment

| Item | Value |
|---|---|
| Repo | CPS4AI/OpenMM, branch `GPU-MM`, HEAD `6d54833` |
| CPU | AMD EPYC 7443P 24-Core (Zen 3, AVX2, **no AVX-512**) |
| Cores | 48 logical |
| OS | Ubuntu 22.04, kernel 5.19.0-1010-nvidia-lowlatency |
| Compiler | g++ 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3) |
| CMake | 3.22.1 |
| Build type | Release |
| ABI | `_GLIBCXX_USE_CXX11_ABI=0` (forced by top CMake; emp-tool needs old ABI) |
| `USE_APPROX_RESHARE` | ON |
| OpenSSL | 3.6.3 (from `~/miniforge3`; system has no `libssl-dev`) |
| emp-tool | `44b1dde` |
| emp-ot | `7f3d4f0` (Ferret VOLE) |
| Eigen | `1f05f51` (v3.3.3) |
| SEAL | `79234726` (v3.7.2), BFV, zstd-compressed, Intel-HEXL backend |
| Intel-HEXL | `343acab` (v1.2.2) |
| zstd | `5233c58e` |

> Note: HEXL's AVX-512 path is inactive on this Zen 3 CPU; it falls back to
> AVX2/non-AVX. SEAL BFV ops are therefore slower here than on an AVX-512 box.
> Any GPU/Phantom comparison must be run on this same machine to be fair, or the
> CPU numbers re-measured on the target box.

### Build/run recipe (machine-specific workarounds, see memory `openmm-build-env`)

```bash
# deps (one-time). Wrapper hides .git (no submodules) so build-deps clones;
# points OpenSSL at miniforge3; pins ABI=0.
bash /tmp/openmm_build_deps.sh   # -> installs into build/, exit 0

# top-level configure + build the profile target
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 \
  -DCMAKE_PREFIX_PATH=/home/richorange/dongye/OpenMM/build \
  -DUSE_APPROX_RESHARE=ON
make -j$(nproc) cpu_baseline_profile

# two-party run FROM REPO ROOT (Ferret writes ./data/pre_ot_data_reg_*)
./build/bin/cpu_baseline_profile 1 32 & ./build/bin/cpu_baseline_profile 2 32
```

## 4. Measured results

Parameters: `bw=32` (prime `4293918721`), `N=1,000,000` per primitive
(`ring_extension`: base `2^64` → `2^80`), `warm=1`, `repeats=3`. All times in ms;
values are **total over 3 repeats** unless marked `avg`.

### 4.1 ring_to_field — `Truncation::ring_to_field`

| region | count | total_ms | avg_ms |
|---|---|---|---|
| `rtf.otpack_setup` | 3 | 2206.0 | **735.3** |
| `rtf.aux_setup` | 3 | 0.015 | 0.005 |
| `rtf.ring_to_field` | 3 | 152.9 | **51.0** |
| `rtf.total` | 3 | 2597.9 | 866.0 |

### 4.2 field_to_ring — `Truncation::field_to_ring`

| region | count | total_ms | avg_ms |
|---|---|---|---|
| `ftr.otpack_setup` | 3 | 2234.8 | **744.9** |
| `ftr.aux_setup` | 3 | 0.015 | 0.005 |
| `ftr.field_to_ring` | 3 | 116.2 | **38.7** |
| `ftr.total` | 3 | 2584.2 | 861.4 |

### 4.3 ring_extension — `Truncation::ring_to_ring128` (64→80 bit)

| region | count | total_ms | avg_ms |
|---|---|---|---|
| `rext.otpack_setup` | 3 | 2272.0 | **757.3** |
| `rext.aux_setup` | 3 | 0.013 | 0.004 |
| `rext.ring_to_ring128` | 3 | 113.8 | **37.9** |
| `rext.total` | 3 | 2624.2 | 874.7 |

### 4.4 bole — `CheetahLinear::BOLE` (both shared-share, no truncate)

| region | count | total_ms | avg_ms |
|---|---|---|---|
| `bole.otpack_setup` | 3 | 2263.2 | **754.4** |
| `bole.aux_setup` | 3 | 29.8 | 9.9 |
| `bole.ring_to_field_x` | 3 | 266.7 | **88.9** |
| `bole.BOLE` | 3 | 2587.4 | **862.5** |
| `bole.field_to_ring` | 3 | 261.8 | **87.3** |
| `bole.total` | 3 | 5659.0 | **1886.3** |

BOB (party 2) numbers are within ~3% of ALICE on every region — timing is
symmetric and the measurements are self-consistent (`total ≈ Σ phases`).

## 5. Analysis

### 5.1 The three pure conversions are structurally identical in cost

`ring_to_field`, `field_to_ring`, and `ring_extension` all share the same shape:

```
~860 ms total  =  ~745 ms otpack_setup (86%)  +  ~40 ms conversion op (5%)  +  sync/IO
```

The conversion kernels themselves are **within 38–51 ms of each other** —
expected, because all three reduce to the same OT-based modulus-conversion
machinery in `Truncation`. The direction (ring→field vs field→ring) and the
width (64→80) barely move the needle.

### 5.2 `otpack_setup` is the single largest cost everywhere

Across all four primitives, **Ferret VOLE bootstrap (`OTPack` ctor) costs ~745 ms
per fresh OTPack and accounts for ~86% of each non-bole iteration.** Each
primitive rebuilds OTPack from scratch (matching the original tests), so a
4-primitive sweep spends ~3 s on setup alone.

**Implication for the GPU roadmap:** the thing a `cuOT` Ferret backend must beat
is this ~745 ms OT setup, **not** the ~40 ms conversion arithmetic. A GPU OT
that only speeds up the conversion math while leaving CPU Ferret setup in place
would save <5% — setup is where the latency is. This is the headline finding.

### 5.3 BOLE: HE multiplication is the new bottleneck

BOLE is a composite primitive (mirrors `bole_test.cpp` `BOLE_test`):

```
ring_to_field ×2  →  CheetahLinear::BOLE (BFV SIMD element-wise mul)  →  field_to_ring
```

Its breakdown:

```
bole.total 1886 ms ≈  otpack_setup 754  +  rtf×2 89  +  BOLE 862  +  ftr 87  +  aux 10
                       (40%)              (5%)        (46%)    (5%)
```

- `bole.BOLE` (the BFV homomorphic element-wise product + relinearization) is
  **862 ms — ~17× a single conversion kernel (51 ms)** and is the single largest
  phase. This is the HE path dominating, not the OT path.
- The two flanking conversions (`rtf_x` 89 ms for both, `ftr` 87 ms) cost about
  2× a standalone conversion, as expected.
- **Implication:** for BOLE-class primitives, a `PhantomFHE` GPU-HE backend is at
  least as important as `cuOT`. Cutting BOLE's 862 ms matters more here than
  cutting the OT setup.

### 5.4 What does NOT matter (yet)

- `aux_setup` is negligible (~0.005 ms for conversions; ~10 ms for bole only
  because `CheetahLinear` ctor builds SEAL keys/contexts). Not a target.
- Conversion *direction* and *width* are second-order. Don't tune these before
  attacking setup and HE.

### 5.5 Caveats / fairness notes for future comparisons

1. **AVX-512 is absent** on this Zen 3 box, so SEAL/HEXL BFV ops run on the
  AVX2 fallback. A GPU-HE or cuOT comparison should either run on this same
  machine or re-measure the CPU baseline on the comparison box first.
2. Each primitive rebuilds `OTPack` per iteration. In a real ML block, OTPack is
  typically constructed once and reused across many primitives, so the
  *amortized* setup cost per primitive would be far lower than the ~745 ms
  measured here. The per-iteration number is the worst case and matches the
  existing tests; for end-to-end ML-block profiling (later milestone), measure
  setup once and reuse.
3. `N=1M` was chosen to keep the 4-primitive sweep short while keeping OT work
  dominant over noise. The original tests use `N=10M` (conversions) and
  `N=2^20≈1.05M` (bole). Per-element costs scale linearly; absolute ms would
  differ at other N but the **phase ratios** (setup vs op) are stable.
4. Two-party loopback (`127.0.0.1`) — network is not a factor. A real WAN run
  would add latency to every `io->sync()` and inflate `total` without changing
  the per-phase compute breakdown.

## 6. Cost model (quick reference)

Per-iteration, N=1M, bw=32, loopback, this machine:

| Primitive | OTPack setup | Conversion op | HE op | Total |
|---|---|---|---|---|
| ring_to_field | ~745 ms | ~51 ms | — | ~866 ms |
| field_to_ring | ~745 ms | ~39 ms | — | ~861 ms |
| ring_extension | ~757 ms | ~38 ms | — | ~875 ms |
| bole | ~754 ms | ~177 ms (rtf×2+ftr) | ~862 ms | ~1886 ms |

**Targets ranked by potential win:**
1. OT/VOLE setup (~745 ms, every primitive) → `cuOT` Ferret.
2. HE multiplication (~862 ms, bole + future FC/Conv) → `PhantomFHE`.
3. Conversion kernels (~40 ms each) → low priority; only matters at high call
   counts inside an ML block.

## 7. Reproducibility checklist

- [x] `git diff --stat` shows only `CMakeLists.txt | 2 ++` (no protocol files
      touched).
- [x] `make -j cpu_baseline_profile` → `[100%] Built target`, exit 0.
- [x] Two-party run both exit 0; ALICE/BOB summaries within 3%.
- [x] `total ≈ Σ phases` for every primitive (self-consistent).
- [x] Warm-up run with profiler disabled does not appear in any summary.
- [x] `data/pre_ot_data_reg_*` (Ferret cache, runtime artifact) restored via
      `git checkout -- data/` before committing.

## 8. Raw logs

Full two-party output is archived alongside this doc in
`docs/logs/cpu_baseline_profile_bw32.log` (ALICE + BOB, all four primitives).

```
docs/
├── research/
│   └── cpu-baseline-profiling.md      ← this document
└── logs/
    └── cpu_baseline_profile_bw32.log  ← raw two-party output
```
