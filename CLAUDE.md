# CLAUDE.md — GPU-M&M Development Discipline

## Project Goal

We are gradually developing a GPU-accelerated version of M&M / OpenMM.

The final long-term goal is:

GPU-resident mixed-modulus PPML runtime =
cuOT Ferret backend +
PhantomFHE HE backend +
CUDA local share tensor backend +
GPU ML blocks.

However, development must proceed in small, independently testable steps.

Do not attempt full-system GPU migration in one task.

---

## Source of Truth

The existing CPU implementation is the correctness baseline.

Unless a task explicitly says otherwise:

- Do not change protocol equations.
- Do not change security assumptions.
- Do not change public behavior.
- Do not remove CPU paths.
- Do not change expected outputs.
- Do not optimize before correctness is established.

Every GPU/cuOT/Phantom/CUDA implementation must first match the CPU baseline.

---

## Known Architecture Facts

1. FC / matrix-vector / inner product uses Cheetah-style coefficient encoding.
   Do not use BatchEncoder for FC / Conv coefficient-encoded paths.

2. Element-wise BOLE / OLE / BN uses BFV SIMD BatchEncoder.
   Do not treat BOLE as coefficient encoding.

3. cuOT is only an OT backend candidate.
   It must first pass standalone correlation tests before replacing any M&M OT path.

4. PhantomFHE is only an HE backend candidate.
   It must first pass standalone SEAL-vs-Phantom tests before replacing production HE paths.

5. CUDA local share kernels must first match CPU reference kernels before being used in protocols.

---

## Task Size Rule

Each task must be small.

A task may modify at most:

- 3 existing source files;
- 2 new test files;
- 1 CMake/build file if needed.

A task must not modify OT and HE in the same task.

A task must not modify correctness logic and performance optimization in the same task.

A task must not touch unrelated modules.

If completing the task requires a larger change, stop and propose a smaller diagnostic task.

---

## Required Development Order

Follow this order:

1. Profiling utility.
2. CPU baseline profiling.
3. Backend interfaces without changing behavior.
4. Standalone cuOT tests.
5. Standalone PhantomFHE tests.
6. Standalone CUDA share tensor tests.
7. Replace one primitive behind a flag.
8. Run primitive-level tests.
9. Compose ML blocks.
10. Run end-to-end system tests.

Never jump directly to full-system GPU execution.

---

## Default Backend Rule

The default backend must remain CPU unless the task explicitly says otherwise.

Any new backend must be behind:

- a compile flag, or
- a runtime flag, or
- a test-only path.

Examples:

- OPENMM_ENABLE_CUOT
- OPENMM_ENABLE_PHANTOM
- OPENMM_ENABLE_CUDA_SHARE
- --he-backend=seal|phantom
- --ot-backend=emp|cuot

---

## Testing Rule

Every task must include tests or evidence.

Do not claim completion without command output.

Minimum completion evidence:

```bash
git diff --stat
cmake ..
make -j
<relevant test command>