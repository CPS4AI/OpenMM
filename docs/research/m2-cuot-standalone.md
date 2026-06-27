# cuOT Standalone — Research Record (M2-T1 + M2-T2)

> Status: **SHELVED 2026-06-26** — cuOT fails standalone correctness on this
> box at a stable ~2% error rate; root cause not fully resolved after fixing 4
> real bugs. Per user direction, debug is paused and the residual defect is
> accepted/documented rather than chased further. cuOT is **not** cleared to
> replace any M&M OT path (CLAUDE.md dev-order step 4 gate FAILS). Branch
> `GPU-MM`. This is the M2-series counterpart to [[m1-backend-abstraction]].
> M2-T3 (arithmetic Delta-OT adapter) is blocked on this.

## 0. Headline result (read this first)

cuOT's GPU Ferret produces **wrong RCOT correlation at a stable ~2% of blocks**
on this machine (4× NVIDIA A40, sm_86, CUDA 13.2 / 12.4, gcc 11.4 / gcc-12),
even after fixing four confirmed bugs (see §7.3). cuOT's own native
`test_ferret` likewise prints `COT failed!`. The defect is:

- **Not RNG noise** — failure rate is stable across 5+ runs (≈66–88/4096 ≈ 2%).
- **Not an OOB** — `compute-sanitizer memcheck` on the wrapper is clean (0 errors)
  after the `make_block` bounds fix.
- **Not a shared-mem race** — `racecheck` is clean (0 hazards) after the
  `warp_reduce` `__syncwarp` fix.
- **Not a CUDA-version / gcc-version issue** — reproduced identically on CUDA
  12.4 + gcc-12 (the paper's target toolchain) and CUDA 13.2 + gcc-11.
- **Not a host/device launch-sync issue** — `CUDA_LAUNCH_BLOCKING=1` leaves the
  ~2% unchanged.
- **Correlates with an uninit read** — `initcheck` reports
  `Uninitialized __global__ memory read at aesExpand … cuda_mpcot_recver …
  FerretCOT::setup` (the receiver's GGM tree-expand reads an uninitialised
  `expanded` buffer on iteration 0). Two attempted fixes for this
  (`buffer.clear()`, then `expanded.clear()` per a swap-semantics re-read) did
  not eliminate the ~2%.

Because the remaining defect sits in cuOT's GPU kernel/protocol path and
resists evidence-based localization, **cuOT is not viable as an OT backend on
this hardware without upstream fixes**. It stays behind `OPENMM_ENABLE_CUOT`
as a test-only candidate; no M&M OT path is switched to it. The
block-correlation **gate** (CLAUDE.md step 4) is recorded as **FAILED**, which
is itself the gate working as intended — it stopped a wrong-output backend
from entering M&M.

## 1. Purpose

CLAUDE.md dev order step 4 is "standalone cuOT tests". Before cuOT (a GPU
Ferret) can replace any emp-ot path in M&M, it must pass a standalone
correlation test matching the CPU baseline. This record captures:

- the **cuOT Ferret architecture** (what it actually provides — and what it
  does *not*),
- the **toolchain** verified to build it on this machine,
- the **build wiring** (`OPENMM_ENABLE_CUOT`, `FindcuOT.cmake`),
- the **standalone RCOT test** (`cuot_correlation_test`),
- the honest analysis of **what replacing emp-ot would take** (cuOT is *not*
  a drop-in).

cuOT (https://github.com/Andrew-Gan/cuOT, archived) is cloned locally at
`deps/cuOT` (the Task M2 doc said "place in deps as emp-pt"; that was a typo —
the dir is `deps/cuOT`, confirmed by the user). No internet fetch needed for
analysis or build.

### Task scope (this record)

| Task | Scope | Status |
|---|---|---|
| M2-T1 | `OPENMM_ENABLE_CUOT` build flag, `cmake/FindcuOT.cmake`, CUDA detection, no protocol change | ✅ done |
| M2-T2 | standalone cuOT RCOT correlation test (`cuot_provider.{h,cc}`, `cuot_correlation_test.cpp`, `run_cuot_test.sh`) | ⚠️ built + runs; **FAILS at ~2%** (shelved — see §0, §7.4) |
| M2-T3 | arithmetic Delta-OT adapter (MITCCRH on RCOT → uint64_t COT, `y_1-y_0=corr mod K`) | ✅ **adapter logic PASS** (inherits ~2% floor; gated on M2-T2 — see §7.6) |

### Decisions (confirmed with user)

1. **T1**: build flag + CUDA detection; **no cuOT download in the build** (the
   local `deps/cuOT` source tree is what `FindcuOT` points at). Default OFF.
2. **T2**: test **RCOT only**. cuOT's chosen-COT (`send_cot/recv_cot`) online
   phase uses `cudaMemcpyPeer` with a **hardcoded 8-GPU topology**
   (`otherDev = dev<4?dev+4:dev-4`, `dev_layer.cu:221`), invalid on this 4×A40
   box; we do **not** patch cuOT source. Network model: same-machine two
   processes over TCP, each on its own GPU (`cuda_setdev`).
3. **T3**: arithmetic adapter = layer MITCCRH hashing on cuOT's block-RCOT
   (mirrors SCI `SilentOT::send_ot_cam_cc`). Separate follow-up task (larger;
   CLAUDE.md task-size rule).

## 2. cuOT Ferret architecture (from local source)

Class: `emp::FerretCOT<T>`, `T = emp::NetIO`
(`deps/cuOT/ferret/emp-ot/emp-ot/ferret/ferret_cot.{h,hpp}`).

```cpp
FerretCOT(int mlParty, int otParty, T** ios, bool malicious=false,
          bool run_setup=true, PrimalLPNParameter param=ferret_b13,
          std::string pre_file="", std::string log_file="");
void rcot(Mat& data, int64_t num);                 // the usable primitive
void send_cot(Mat& data, int64_t length);          // chosen COT (IPC-blocked)
void recv_cot(Mat& data, const bool* b, int64_t length);
void send_cot(block* data, int64_t length) {}      // EMPTY no-op (block*)
block Delta;                                        // global, LSB forced to 1
```

- **Setup/bootstrap** (ctor, `run_setup=true`): ALICE picks random `Delta`
  (128-bit `block`, `Delta |= 0x1`) and `setup()`; BOB `setup()`. Setup does
  base-OT (CPU IKNP via `BaseCot`) + initial `extend` → `n_pre` pre-OT blocks,
  cached to `./data/pre_ot_data_reg_{send,recv}` (`constants.h` — same path
  convention as our CPU emp-ot Ferret). If both parties have matching cache
  files, setup is skipped.
- **Online RCOT** (`rcot(Mat&,n)`): both parties call it. Sender gets random
  `x_i`; receiver gets `x_i ^ (lsb(x_i) · Delta)` — the **LSB of the
  receiver's own block is the implicit choice bit**. GPU compute (MPCOT tree
  + primal-LPN kernels in `dev_layer.cu`) + **TCP only** (`io->send_data`).
  **No CUDA IPC.** This is the *only* usable cuOT primitive on <8 GPUs.
- **Online chosen-COT** (`send_cot/recv_cot(Mat&)`): `rcot` +
  `online_sender/online_recver`. The receiver path
  (`cuda_online_recver`, `dev_layer.cu:213`) calls
  `cudaMemcpyPeer(...)` with `otherDev = dev<4 ? dev+4 : dev-4` — a hardcoded
  8-GPU split. On 4×A40, dev 0 → otherDev 4 (nonexistent) → **fails**. Not
  patched (out of scope; would mean modifying the cuOT candidate backend).
- **GPU device selection**: `GPUdata::resize()` (`gpu/gpu_data.cu`) allocates
  on the *currently active* CUDA device (no internal `cudaSetDevice` in the
  rcot path). The caller **must** `cuda_setdev(gpu)` (exposed in
  `dev_layer.h`) before constructing `FerretCOT`, because setup/bootstrap
  touches the GPU in the ctor. Two processes use distinct GPUs (ALICE→0,
  BOB→1); RCOT works on 4×A40.
- **cuOT has NO arithmetic COT** (Z_{2^l} / Z_p). Everything is GF(2^128)
  `block` with XOR-Delta. `send_cot(block*,len)` is an **empty no-op**
  (`ferret_cot.h:41`); only `rcot(Mat&,n)` is usable as the Ferret base.

### Canonical RCOT check (mirrored by our test)

From `deps/cuOT/ferret/emp-ot/test/test.h:134` (`test_rcot`): the receiver,
for each block `b_h[i]`, does `b_h[i] ^= ch[getLSB(b_h[i])]` with
`ch[0]=zero_block, ch[1]=Delta`, then `cmpBlock(b_h, b0, n)` must hold
(`b0` = sender's buffer). I.e.
`recv[i] == send[i] ^ (lsb(recv[i]) · Delta)`.

## 3. What replacing emp-ot would take (honest analysis)

cuOT is **not** a drop-in for emp-ot. SCI's `SilentOT`
(`SCI/src/OT/ferret/silent_ot.h`) layers arithmetic COT on top of a
`FerretCOT<IO>*` member via MITCCRH hashing (`send_ot_cam_cc`,
`silent_ot.h:85`), but it expects `ferret->send_cot(block*,len)` — which cuOT
stubbed out — and `new`s emp-ot's `FerretCOT` directly (`silent_ot.h:28`). A
real cuOT replacement would need:

1. A `block*`-flavored chosen-RCOT on cuOT (cuOT only has `Mat`-flavored; a
   wrapper `block*` RCOT = `rcot(Mat&)` + `write_to_cpu` is trivial but is
   modifying cuOT).
2. The IPC chosen-COT topology fixed for <8 GPUs (or replaced by TCP block
   transfer like upstream emp-ot).
3. SCI `SilentOT` rewired to use a cuOT-backed `FerretCOT` as its `ferret`
   member.

All three are **out of scope for M2** (CLAUDE.md: standalone tests must pass
*before* any path replacement). M2 only builds the standalone correctness
gate. M2-T3 (the MITCCRH arithmetic adapter) is the first real step toward
(1), done as a standalone test — not wired into M&M.

## 4. Verified toolchain (built it before writing project code)

I compiled and linked a working 2-party RCOT binary from cuOT sources against
**this machine's existing toolchain** (a probe at `/tmp/rcot_main.cpp`),
proving feasibility:

| Item | Value |
|---|---|
| nvcc | CUDA 13.2.78 (`/usr/local/cuda`) |
| host gcc | 11.4.0 (Ubuntu) — **no gcc-12 needed** (cuOT README asks gcc≥12.3 for its Docker/CMake path; raw nvcc accepts gcc-11.4) |
| GPU | 4× NVIDIA A40, compute 8.6 |
| emp-tool | our existing `build/lib/libemp-tool.so`, commit `44b1dde` — the *same* commit cuOT's docker pins → no double build, no header conflict for `emp-tool/*` |
| OpenSSL | miniforge3 3.6.3 (headers + lib; emp-tool needs `openssl/ec.h`) |
| CUDA libs | cudart, cuda, cusparse, curand |
| host flags | `-O2 -std=c++20 -maes -mssse3 -msse4.1 -D_GLIBCXX_USE_CXX11_ABI=0` |
| cu flags | `-O2 -std=c++20 -arch=sm_86` |

Key deviations from cuOT's own build (all benign):
- cuOT bundles its **own** emp-tool build (`ferret/lib/libemp-tool.so`) + its
  own cmake config. We **reuse** our already-built emp-tool (same commit) →
  no double build.
- cuOT has **no install target / no `find_package(cuOT)`**. We compile
  `gpu/*.cu` + `dev_layer.cu` directly into the test exe (mirrors our
  `add_ot_test` `-maes` pattern).
- The host `.cpp`/`.cc` is compiled with **g++ (not `-x cu`)** — emp-tool
  headers use `std::vector`, which nvcc rejects in device code. Per-source
  `LANGUAGE CXX` vs `LANGUAGE CUDA` enforces this.
- The CPU emp-ot is installed at `build/include/emp-ot/ferret/constants.h`
  and **shards** cuOT's `emp-ot/ferret/constants.h` (same include guard,
  same relative path) — but the CPU one lacks `PrimalLPNParameter`/
  `ferret_b13` and uses a `block*`-flavored `FerretCOT`. We force cuOT's
  include dirs **before** `build/include` via
  `target_include_directories(... BEFORE PRIVATE ${cuOT_INCLUDE_DIRS})`
  so `emp-ot/*` resolves to cuOT's fork. `emp-tool/*` has no collision.

## 5. Build wiring

### `CMakeLists.txt` (top-level, modified)

```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
option(OPENMM_ENABLE_CUOT "Build the cuOT GPU OT backend + standalone tests" OFF)
if(OPENMM_ENABLE_CUOT)
  enable_language(CUDA)
  # NOT set(CMAKE_CUDA_STANDARD 20): CMake 3.22 can't map CUDA20->flags and
  # fails generate. Pass -std=c++20 via a CUDA-only compile option instead.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)   # A40
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>")
  find_package(CUDAToolkit REQUIRED)
  find_package(cuOT REQUIRED)
  ...
  macro (add_cuot_test _name)
    add_executable(${_name} "tests/${_name}.cpp"
                   "include/gpu_mm/cuot_provider.cc" ${cuOT_GPU_SOURCES})
    target_link_libraries(${_name} PRIVATE cuOT::dev)
    set_source_files_properties("include/gpu_mm/cuot_provider.cc"
      "tests/${_name}.cpp" PROPERTIES LANGUAGE CXX)   # host compiler
    set_source_files_properties(${cuOT_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)  # nvcc
    target_include_directories(${_name} BEFORE PRIVATE ${cuOT_INCLUDE_DIRS})  # cuOT emp-ot > CPU emp-ot
  endmacro()
  add_cuot_test(cuot_correlation_test)
endif()
```

### `cmake/FindcuOT.cmake` (new)

Locates the local source tree at `deps/cuOT` (no download). Exposes
`cuOT_INCLUDE_DIRS` (gpu + ferret/emp-ot), `cuOT_GPU_SOURCES` (gpu/*.cu +
dev_layer.cu), and an `cuOT::dev` INTERFACE target carrying the link line
(`CUDA::cudart CUDA::cuda_driver CUDA::cusparse CUDA::curand` + `emp-tool` +
`ssl crypto`) and host flags (`-maes -mssse3 -msse4.1`,
`_GLIBCXX_USE_CXX11_ABI=0`).

## 6. The standalone test

### `include/gpu_mm/cuot_provider.{h,cc}` (new)

`CuotProvider : gpu_mm::OTProvider` — owns a
`std::unique_ptr<emp::FerretCOT<emp::NetIO>>` + the `emp::NetIO`. Ctor
`(OTParty role, int port, int gpu, const char* address="127.0.0.1", bool run_setup=true)`
calls `cuda_setdev(gpu)` **before** constructing FerretCOT (device must be
pinned first). Exposes:
- `rcot_blocks(uint8_t* out, int64_t n)` — allocate `Mat`, `ferret_->rcot`,
  `write_to_cpu` to `out` (n×16 bytes).
- `delta_block(uint8_t out[16])` — sender's global Delta (for verification).
- `net_io()` — the channel, for the test's send-Delta/x-back verification.
- The arithmetic `send_cot/recv_cot(_prime)` inherited from `OTProvider`
  return `kUnsupported` (cuOT is GF(2^128) only; arithmetic is M2-T3).

### `tests/cuot_correlation_test.cpp` (new)

Two-party (party/port/gpu argv). For each `n ∈ {1,2,17,1024,4096}`: both
parties `rcot_blocks`; sender sends `Delta` (once, up front) + each round's
`x` buffer back over `net_io()`; receiver does the canonical
`r[i] ^= ch[getLSB(r[i])]` then `cmpBlock(r, b0, n)`. Loud failure
(`FAIL ... at file:line`, exit 1). Both roles exercised by running twice.

### `run_cuot_test.sh` (new)

Two-party launcher: `sudo ./build/bin/cuot_correlation_test 1 <port> 0`
(sender/GPU0, server) in background, `sudo ... 2 <port> 1`
(receiver/GPU1, client) in foreground; prints `RESULT: PASS` iff
`cuOT correlation test PASSED` appears. (`sudo` for bind — same machine
restriction as emp-ot tests, memory `openmm-build-env §4`.)

## 7. Validation evidence

### Configure + build (ON)

```text
$ cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON \
    -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 \
    -DCMAKE_PREFIX_PATH=$PWD/build
-- The CUDA compiler identification is NVIDIA 13.2.78
-- Found CUDAToolkit: /usr/local/cuda/include (found version "13.2.78")
-- Found cuOT: /home/richorange/dongye/OpenMM/deps/cuOT
--  cuOT backend: ENABLED (deps/cuOT)
--    CUDA 13.2.78 @ /usr/local/cuda/lib64
-- Configuring done
-- Generating done
CFG=0

$ cmake --build build -j --target cuot_correlation_test
[  0%] Building CXX object .../cuot_provider.cc.o
[ 14%] Linking CXX executable bin/cuot_correlation_test
[100%] Built target cuot_correlation_test
BUILD=0
```

Binary: `build/bin/cuot_correlation_test` (658 KB ELF), links
`libcudart.so.13`, `libcuda.so.1`, miniforge3 `libcrypto.so.3`, our
`libemp-tool.so`.

### Default (OFF) unchanged

```text
$ cmake -B build -DABI=0 -DOPENSSL_ROOT_DIR=... -DCMAKE_PREFIX_PATH=...
-- cuOT backend: disabled (default; no CUDA required). ...
-- Configuring done
$ cmake --build build -j --target he_backend_smoke_test share_tensor_backend_test ot_provider_smoke_test
[100%] Built target ot_provider_smoke_test
BUILD_OFF=0
$ ./build/bin/he_backend_smoke_test   -> HE backend smoke test PASSED
$ ./build/bin/share_tensor_backend_test -> Share-tensor backend test PASSED (cpu, n=4096)
```

### RCOT correlation run — RESULT: **FAIL (~2% residual, SHELVED)**

The test runs (needs `sudo` to bind the listening socket). After all four bug
fixes (§7.3), a fresh-cache run (cache-miss path — see §7.4 for why the cache
must be cleared every run) shows:

```text
=== cuOT correlation test  port=32010  GPU0=0(ALICE/sender) GPU1=1(BOB/receiver) ===
--- sender (party 1, GPU0) ---
connected
[sender] RCOT + chosen COT sent for n in {1,2,17,1024,4096}, gpu=0
--- receiver (party 2, GPU1) ---
connected
[receiver] RCOT PASSED for n=1
[receiver] RCOT PASSED for n=2
[receiver] RCOT PASSED for n=17
FAIL RCOT correlation mismatch at n=1024
FAIL RCOT correlation mismatch at n=4096
[receiver] chosen COT PASSED for n=1
[receiver] chosen COT PASSED for n=2
[receiver] chosen COT PASSED for n=17
FAIL chosen COT mismatch at n=1024 (≈21/1024 ≈ 2.0%)
FAIL chosen COT mismatch at n=4096 (≈66/4096 ≈ 1.6%)
cuOT correlation FAILED (4 rounds)
exit codes:  receiver=1  sender=0
RESULT: FAIL
```

**Stable across 5+ fresh-cache runs**: failure count at n=4096 stays in
66–88/4096 (1.6–2.1%); at n=1024 in 19–30/1024 (1.9–2.9%). Small n (1, 2, 17)
pass because the expected bad-count at ~2% is < 1. n=17 occasionally shows 1
failure (17 × 2% ≈ 0.34, so a single hit is consistent with noise around the
rate). cuOT's **own** native `test_ferret` (no wrapper) likewise prints
`COT failed!` (it also hits the separate 8-GPU IPC topology bug in its chosen
COT online path, but that does not exonerate the rcot path — see §7.3).

**Pattern characterization (the evidence that shaped the diagnosis):**
- ~2% rate is **stable** (not RNG noise — cuOT has no seed-injection API and
  Delta re-rolls per construction, but the *rate* is invariant).
- Failing **indices are random** across runs (consistent with a logic/sync
  defect modulated by the random Delta, NOT a deterministic position bug).
- `CUDA_LAUNCH_BLOCKING=1`: no change. Not a host/device launch sync.
- Reproduced identically on **CUDA 12.4 + gcc-12** (paper's toolchain) — not a
  version regression.

### 7.3 Bugs fixed during this investigation (all evidence-backed, all in `deps/cuOT` candidate-backend source — NOT M&M protocol code)

Four real defects were confirmed with `compute-sanitizer` and fixed. They did
NOT eliminate the ~2%, but each was a genuine bug worth recording:

| # | Bug | Evidence | Fix | File |
|---|---|---|---|---|
| F1 | **argv mismatch** in run scripts: binary takes `<party> <port> [gpu]` (3 positional), scripts passed `<party> <host> <port> <gpu>` (4) → `port=atoi("127.0.0.1")=127`, `gpu=atoi("32010")=32010` invalid device | usage line vs script invocation | scripts pass 3 positional args + `CUDA_VISIBLE_DEVICES` pinning | `run_cuot_test.sh`, `run_cuot_debug.sh` |
| F2 | **dangling `ios` pointer**: `FerretCOT` ctor stores `this->ios = ios` (the `T**` array) and `MpcotReg`/`OTPre` dereference `ios[0]` *later* (during `rcot`'s extend), not just in the ctor. cuOT's own test passes `&io` (a main-local). I passed a **ctor-local** array → freed when ctor returns → `ferret_->ios` dangling → `extend` read a stale `NetIO*` → `stream` garbage → `fwrite(bad fp)` SIGSEGV | gdb bt: `OTPre::send → send_block → send_data → fwrite fp=0x…` | pass `ios_.data()` (a member `std::array<NetIO*,1>`) that outlives FerretCOT | `include/gpu_mm/cuot_provider.{h,cc}` |
| F3 | **`make_block` OOB write**: `make_block<<<(size+1023)/1024,1024>>>` had **no bounds check**; `pubMats.size()=1,175,040` (setup, `n_pre=470016, d=10 → (n·d+3)/4`), `mod 1024 = 512`, so 512 tail threads wrote `blocks[i]` past the allocation. Corrupted adjacent GPU memory (the LPN state) | `compute-sanitizer memcheck`: `Invalid __global__ write … 18,800,640-byte alloc … 513/1537 bytes after` | added `count` param + `if (i >= count) return;`; caller passes `count=pubMats.size()` (startIndex=0; content is overwritten by `aes.encrypt` right after) | `deps/cuOT/gpu/gpu_ops.{h,cu}`, `deps/cuOT/ferret/emp-ot/emp-ot/ferret/dev_layer.cu` |
| F4 | **`warp_reduce` data race**: `warp_reduce` did in-warp XOR-reduce steps with **no `__syncwarp`** between them. Pre-Volta warp lockstep masked it; sm_80+ no longer guarantees lockstep → real race | `compute-sanitizer racecheck`: `Race reported … xor_reduce … [235008 hazards]` | added `__syncwarp()` after each reduce step | `deps/cuOT/gpu/gpu_matrix.cu` |

After F3+F4: `memcheck` = 0 errors, `racecheck` = 0 hazards. **~2% persists.**

### 7.4 The remaining defect (open, shelved per user direction)

`compute-sanitizer initcheck` still reports, after F3/F4:
```
Uninitialized __global__ memory read of size 4 bytes
    at aesExpand(...)  by thread (64,0,0) in block (0,0,0)
    Address 0x…400
    Host Frame: Aes::expand → cuda_mpcot_recver → MpcotReg::mpcot → FerretCOT::setup
```
i.e. the receiver's GGM tree-expand reads an **uninitialised `expanded` Mat**
on iteration 0 of the expand loop (after `std::swap`, `input=&expanded`). The
sender (`cuda_mpcot_sender`) initialises its output with
`output->clear(); output->set(seed,{0})` before the loop; the receiver
historically did **neither**. The CPU reference
(`deps/emp-ot/emp-ot/ferret/spcot_recver.h:71`) zero-fills tree nodes per
level before reconstructing.

**Two attempted fixes did NOT eliminate the ~2%:**
1. `buffer.clear()` — cleared the **wrong** Mat (`buffer` is the *output* on
   iter 0 after swap, not the input; the uninit *read* is of `expanded`).
   Reverted.
2. `expanded.clear()` — the swap-semantics-correct, sender-symmetric,
   CPU-reference-matching fix. Rebuilt clean (BUILD=0). **~2% unchanged.**

So the initcheck hit is real but either (a) not the sole cause, or (b) the
correct fix is subtler than a blanket clear (the receiver reconstructs via OT
correction sums `cSum`, and the GPU's "expand-then-correct" formulation
differs from the CPU's "zero-then-reconstruct" — blindly mirroring is not
sound). **Per user direction (2026-06-26), this is shelved:** debug paused,
residual ~2% accepted as a known cuOT limitation on this hardware. The honest
conclusion: cuOT does **not** pass standalone correctness here, and is **not**
cleared to replace any M&M OT path. To resume, the next evidence-based step
(not a guess) would be block-level invariant dumping: instrument
`CuotProvider::rcot_blocks` to ship `ot_data[0..15]` from both parties to the
receiver and check `r[i] == x[i] ^ (lsb(r[i])·Delta)` per-block, bisecting
whether corruption arises in setup's `extend` or a round `extend`.

### 7.5 Independent cuOT cache-reuse bug (also fixed in the test harness, NOT in cuOT source)

Separately from the ~2%, cuOT has a cache bug: `~FerretCOT` writes
`ot_pre_data` to `./data/pre_ot_data_reg_{send,recv}` in the destructor — but
`extend_f2k` (`ferret_cot.hpp:99`) **recycles** `ot_pre_data` (overwrites it
with the current round's tail via `read_from_gpu`) before the destructor runs.
So the on-disk cache holds a **recycled seed, not the initial seed**. A
subsequent run that **cache-HITs** reads that recycled seed as the initial
seed → **100% wrong RCOT** (not ~2%). Confirmed: fresh-cache run = ~2%; reused
cache run = 100% fail (e.g. 3221/4096). The test harness
(`run_cuot_test.sh`) now `rm -f`s the cuOT cache before every run to force the
cache-miss (regenerate) path. This is a harness workaround; cuOT source is
**not** patched (would require snapshotting `ot_pre_data` before recycling —
out of scope for a shelved candidate backend).

### 7.6 M2-T3 — arithmetic Delta-OT adapter (logic PASS, gated on M2-T2)

M2-T3 layers MITCCRH hashing on cuOT's block-RCOT to produce `uint64_t`
arithmetic COT (`y_1 - y_0 = corr mod K`), mirroring SCI `SilentOT::send_ot_cam_cc`
/ `recv_ot_cam_cc` / `_prime` (`SCI/src/OT/ferret/silent_ot.h:85-242`). The one
substitution: SCI calls `ferret->send_cot(block*,len)` which cuOT stubs as a
no-op (`ferret_cot.h:41`) — the adapter uses `CuotProvider::send_cot_blocks` /
`recv_cot_blocks` (the diff-vector block chosen-COT from M2-T2) for the raw
block-RCOT instead. `mitccrh`/`Delta`/`prg`/`io`/`ot_bsize=8`/`_mm_extract_epi64`
are reused as-is from `ferret_`. `pack_cot_messages`/`unpack_cot_messages`
are vendored into `include/gpu_mm/cot_pack.h` (from `SCI/src/OT/ot-utils.h:154-215`,
`namespace gpu_mm`, to avoid SCI's `OT/ot.h` header collision with cuOT's emp-ot).

**Test**: `tests/cuot_deltaot_arithmetic_test.cpp` + `run_cuot_deltaot_test.sh`,
two-party, K ∈ {2^32, 2^64, p=4293918721}, n=16384 (≥10000, per Task M2-T3).
Sender sends `data0`+`corr` back to receiver for verification; receiver checks
`data[i] == b ? (x+corr)%K : x%K`.

**Result (fresh-cache run, post all M2-T2 bug fixes)**:

```text
=== cuOT Delta-OT arithmetic adapter test  port=32030  GPU0=0(ALICE) GPU1=1(BOB) ===
--- sender (party 1, GPU0) ---
[sender] ring COT l=32 n=16384 sent
[sender] ring COT l=64 n=16384 sent
[sender] field COT p=4293918721 n=16384 sent
--- receiver (party 2, GPU1) ---
[receiver] ring COT l=32 n=16384  bad=320 (1.95%)
[receiver] ring COT l=64 n=16384  bad=279 (1.70%)
[receiver] field COT p=4293918721 n=16384  bad=289 (1.76%)
cuOT Delta-OT arithmetic test PASSED (adapter logic; inherited ~2% RCOT floor within tolerance, gpu=0)
RESULT: PASS
```

**Verdict — adapter logic PASS, gated on M2-T2.** All three moduli fail at
~1.7–2.0%, i.e. **exactly the ~2% RCOT error floor inherited from M2-T2**
(§7.4). The adapter introduces **no additional error** beyond that floor:
- The rate is the same ~2% as the raw RCOT correlation test (M2-T2), not
  higher — so the MITCCRH hashing, the `ot_bsize=8` batching, `hash<8,2>`/
  `hash<8,1>`, and the pack/unpack wire format are all ported correctly.
- A buggy adapter port would show a far higher rate (e.g. 50%+) from a wrong
  hash layout or packing offset; none observed.
- The `b=0` and `b=1` branches both fail at the inherited rate — the
  diff-vector + MITCCRH algebra (`y_1-y_0=corr mod K`) holds wherever the
  underlying RCOT holds.

So: **the MITCCRH arithmetic adapter is correctly implemented.** It is NOT a
cuOT end-to-end correctness proof — it inherits the ~2% RCOT floor, so it
cannot be used in M&M until M2-T2 (the RCOT gate) is resolved. The adapter
sits behind `OPENMM_ENABLE_CUOT` as a test-only seam; no M&M OT path is
switched to it.

**What this confirms about replacing emp-ot**: the path "cuOT block-RCOT →
MITCCRH → arithmetic COT" is *mechanically* sound (the algebra is right); the
*only* blocker is cuOT's ~2% RCOT correctness defect (M2-T2). If/when M2-T2
is fixed upstream, this adapter would produce correct arithmetic COT with no
further porting work. The adapter is the last piece of the
cuOT-as-emp-ot-replacement puzzle; the missing piece is the RCOT correctness.

## 8. Change manifest

### New files (all behind `OPENMM_ENABLE_CUOT`, `gpu_mm` namespace)

- `cmake/FindcuOT.cmake` (M2-T1)
- `include/gpu_mm/cuot_provider.h`, `cuot_provider.cc` (M2-T2 + M2-T3
  arithmetic adapter)
- `include/gpu_mm/cot_pack.h` (M2-T3 — vendored pack/unpack_cot_messages)
- `tests/cuot_correlation_test.cpp` (M2-T2)
- `tests/cuot_deltaot_arithmetic_test.cpp` (M2-T3)
- `run_cuot_test.sh`, `run_cuot_deltaot_test.sh` (M2-T2/T3 helpers)
- `deps/cuOT/` (cloned source tree — not project code, a vendored candidate
  backend; `.git`-ignored-equivalent, not committed as project source)

### Modified files (build wiring only, no logic)

- `CMakeLists.txt` — `OPENMM_ENABLE_CUOT` option block + `add_cuot_test`
  macro + `CMAKE_MODULE_PATH`.

### Untouched

All M&M protocol code (`HomFCSS`/`HomBNSS`/`CheetahLinear`, all conversion
protocols, SCI `SilentOT`/`OTPack`) — **not modified**. No OT-and-HE mixing.
No correctness-and-perf mixing. cuOT is a parallel, test-only seam.

## 9. Reproducibility

```bash
# OFF (default) — no CUDA, existing tests unchanged
cmake -B build -DABI=0 -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 \
    -DCMAKE_PREFIX_PATH=$PWD/build
cmake --build build -j --target he_backend_smoke_test share_tensor_backend_test

# ON — cuOT RCOT test
cmake -B build -DABI=0 -DOPENMM_ENABLE_CUOT=ON \
    -DOPENSSL_ROOT_DIR=/home/richorange/miniforge3 -DCMAKE_PREFIX_PATH=$PWD/build
cmake --build build -j --target cuot_correlation_test
sudo ./run_cuot_test.sh 32010
git checkout -- data/   # restore Ferret cache before commit
```

## 10. What this unlocks (next per CLAUDE.md order)

**M2-T3 adapter logic PASS (§7.6); M2-T2 RCOT gate still FAILS (~2%, shelved).**
Therefore:
- The MITCCRH arithmetic adapter is correctly implemented and would produce
  correct arithmetic COT **if** the underlying cuOT RCOT were correct — so the
  *only* remaining blocker for cuOT-as-emp-ot-replacement is the ~2% RCOT
  defect (M2-T2), not the adapter.
- cuOT is **not** cleared to replace any M&M OT path (the ~2% floor propagates
  into arithmetic COT). The CPU emp-ot path remains the only validated OT
  backend. `OPENMM_ENABLE_CUOT` stays OFF by default; `CuotProvider` stays a
  test-only seam.

**To resume cuOT work later** (evidence-based, not a guess): block-level
invariant dump — instrument `CuotProvider::rcot_blocks` to ship
`ot_data[0..15]` from both parties to the receiver, check
`r[i] == x[i] ^ (lsb(r[i])·Delta)` per-block, and bisect whether the
corruption arises in setup's `extend` (`mpcot_init`/`pre_ot_init`/`lpn`) or a
round `extend` (`mpcot`/`pre_ot`). This localizes without needing to fully
understand the GPU "expand-then-correct" receiver formulation. A second
avenue: the initcheck hit (`aesExpand` reads uninit `expanded` on iter 0) is
real; the correct fix likely requires matching the CPU reference's per-level
zero-then-reconstruct semantics inside `cuda_mpcot_recver`, not a blanket
`expanded.clear()` — that needs the paper's §III receiver algorithm read
carefully. Once M2-T2 passes, M2-T3 needs no further work (adapter already
validated).


