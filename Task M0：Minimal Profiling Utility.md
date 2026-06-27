You are working on CPS4AI/OpenMM branch GPU-MM.

Task ID:
M0-T1

Before starting this task, read CLAUDE.md at the repository root and follow it strictly.

This task is intentionally small. Do not expand its scope.

If you believe this task requires modifying protocol files, stop and explain why. Do not proceed

Goal:
Add a minimal lightweight profiling utility for OpenMM. This task must not change any protocol behavior. It only introduces a reusable TimerScope API and one smoke test.

Background:
The current top-level CMakeLists.txt defines an add_test macro for tests/<name>.cpp and links test binaries against gemini and SCI-Cheetah. Existing tests are added by calling add_test(ring_to_field_test), add_test(field_to_ring_test), add_test(ring_extension_test), and add_test(bole_test). Follow the same pattern.

Scope:
You may add:
- include/gpu_mm/profiler.h
- tests/profiler_smoke_test.cpp

You may modify:
- CMakeLists.txt

Forbidden:
- Do not modify FC, Conv, BOLE, ring_to_field, field_to_ring, comparison, OT, or HE logic.
- Do not add CUDA.
- Do not add cuOT.
- Do not add PhantomFHE.
- Do not change existing test behavior.
- Do not modify more than these three files.
- Do not claim completion without build/test output.

Implementation requirements:
1. Implement a header-only profiler in include/gpu_mm/profiler.h.
2. Provide an RAII TimerScope class:
   - constructor starts timing for a named region;
   - destructor records elapsed time;
   - timing unit should be microseconds or milliseconds.
3. Provide a Profiler singleton or static utility with:
   - Reset()
   - Enable(bool)
   - PrintSummary(std::ostream&)
   - Record(name, elapsed)
4. The profiler must be disabled/enabled at runtime.
5. It must be safe for simple multi-threaded use. Use std::mutex around shared state.
6. The summary should include:
   - region name
   - call count
   - total time
   - average time
7. Do not introduce external dependencies.

Expected API:
{
  gpu_mm::Profiler::Reset();
  gpu_mm::Profiler::Enable(true);

  {
    gpu_mm::TimerScope t("smoke.region");
    // some dummy work
  }

  gpu_mm::Profiler::PrintSummary(std::cout);
}

Expected output shape:
[PROFILE] smoke.region count=1 total_ms=... avg_ms=...

Smoke test:
Create tests/profiler_smoke_test.cpp that:
1. Enables the profiler.
2. Runs at least two TimerScope regions.
3. Runs one repeated region at least three times.
4. Calls PrintSummary.
5. Checks that the process exits successfully.
6. Does not require network, SEAL keys, two parties, CUDA, cuOT, or PhantomFHE.

CMake:
Add profiler_smoke_test using the existing add_test macro in CMakeLists.txt:
add_test(profiler_smoke_test)

Acceptance criteria:
1. Default build still works.
2. profiler_smoke_test builds.
3. profiler_smoke_test runs and prints a profiling summary.
4. No existing protocol files are modified.
5. No existing CPU backend behavior changes.

Required commands:
- mkdir -p build
- cd build
- cmake ..
- make -j
- ./bin/profiler_smoke_test

Deliverables:
1. List of modified files.
2. Exact build command output.
3. Exact profiler_smoke_test output.
4. Confirm that no protocol logic was modified.
5. State the next smallest task after this one.