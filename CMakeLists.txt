cmake_minimum_required(VERSION 3.11)
project(OpenMMs)
include(CMakeFindDependencyMacro)

# Local find-modules (e.g. FindcuOT.cmake).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# cuOT GPU OT backend (M2). Default OFF: no CUDA/cuOT dependency. When ON,
# enables the CUDA language, finds the local cuOT source tree at deps/cuOT
# (no download — cloned manually), and builds the standalone cuOT tests. Does
# NOT touch any M&M protocol path; cuOT is a parallel, test-only seam until
# its standalone tests pass (CLAUDE.md dev order step 4).
option(OPENMM_ENABLE_CUOT "Build the cuOT GPU OT backend + standalone tests" OFF)
if(OPENMM_ENABLE_CUOT)
  enable_language(CUDA)
  # nvcc 13.2 supports c++20. Do NOT use set(CMAKE_CUDA_STANDARD 20) here:
  # this CMake 3.22 reports it can't map the CUDA20 dialect to flags and fails
  # the generate step. Pass -std=c++20 directly via CMAKE_CUDA_FLAGS instead.
  #
  # CUDA arch: cuOT's kernels were validated by the paper on V100 (sm_80). On
  # this box's A40 (sm_86) with CUDA 13.2, building native sm_86 produces
  # WRONG RCOT/COT correlation (first few blocks correct, then corrupted from
  # ~index 3 onward — a kernel race/sync codegen bug). We build sm_80 PTX
  # (paper's target) AND let the A40 JIT it — A40 is forward-compatible with
  # sm_80 SASS. This is overridable: -DCMAKE_CUDA_ARCHITECTURES=86 to test the
  # native (known-broken) path.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>")
  find_package(CUDAToolkit REQUIRED)
  find_package(cuOT REQUIRED)
  message(STATUS "==============================================")
  message(STATUS " cuOT backend: ENABLED (deps/cuOT)")
  message(STATUS "   CUDA ${CUDAToolkit_VERSION} @ ${CUDAToolkit_LIBRARY_DIR}")
  message(STATUS "   cuOT source tree: ${cuOT_ROOT_DIR}")
  message(STATUS "   This adds GPU OT tests only; no protocol path is changed.")
  message(STATUS "==============================================")
else()
  message(STATUS "cuOT backend: disabled (default; no CUDA required). "
                 "Set -DOPENMM_ENABLE_CUOT=ON to enable the GPU OT tests.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

## Dependencies
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Eigen3 3.3.3 REQUIRED)
find_package(SEAL REQUIRED)
find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR}
                    ${EMP-OT_INCLUDE_DIRS} 
                    ${EMP-TOOL_INCLUDE_DIRS})

## cheetah-core
add_library(gemini SHARED)
include_directories(include)
add_subdirectory(include)
target_link_libraries(gemini SEAL::seal Eigen3::Eigen Threads::Threads)

if (USE_APPROX_RESHARE)
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=1)
else()
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=0)
endif()

## SCI
add_subdirectory(SCI)
include_directories(SCI/src/)

## demo
macro (add_network _name)
  add_executable(${_name} "networks/main_${_name}.cpp")
  target_link_libraries(${_name} gemini SCI-Cheetah)
  target_compile_definitions(${_name} PUBLIC SCI_OT=1 USE_CHEETAH=1)
endmacro()

add_network(resnet50)
add_network(sqnet)
add_network(densenet121)

macro (add_test _name)
  add_executable(${_name} "tests/${_name}.cpp")
  target_link_libraries(${_name} gemini SCI-Cheetah)
  target_compile_definitions(${_name} PUBLIC SCI_OT=1 USE_CHEETAH=1)
endmacro()

# gpu_mm OT provider source needs M&M's OT headers (utils/aes-ni.h), which
# require -maes. gemini does not carry SCI's ISA flags, so compile this source
# directly into the OT test executable (which links SCI-Cheetah and inherits
# -maes transitively via SCI-utils). No behavior change; test-only path.
macro (add_ot_test _name)
  add_executable(${_name} "tests/${_name}.cpp"
                 "include/gpu_mm/emp_ot_provider.cc")
  target_link_libraries(${_name} gemini SCI-Cheetah)
  target_compile_definitions(${_name} PUBLIC SCI_OT=1 USE_CHEETAH=1)
endmacro()

add_test(ring_to_field_test)
add_test(field_to_ring_test)
add_test(ring_extension_test)
add_test(bole_test)
add_test(profiler_smoke_test)
add_test(cpu_baseline_profile)
add_test(he_backend_smoke_test)
add_ot_test(ot_provider_smoke_test)
add_ot_test(wrap_bit_or_test)
# M3-T4: ring→field OR_AUX primitive (the ONLY COT-p callsite in M&M) on the
# gpu_mm OTProvider seam, CPU emp default. Mirrors truncation.cpp:543-559;
# SCI untouched. cuOT counterpart is cuot_ring_to_field_test (below).
add_ot_test(ring_to_field_or_aux_test)
# M3-T5: field→ring OR_AUX primitive (COT-l callsite, truncation.cpp:600-616)
# on the gpu_mm OTProvider seam, CPU emp default. SCI untouched. cuOT
# counterpart is cuot_field_to_ring_test (below).
add_ot_test(field_to_ring_or_aux_test)
# M3-T6 Phase A: CPU baseline single-digit compare leaf (SCI kkot 1-OO-N-KOT,
# N∈{2,4,8,16}) — the correctness + comm baseline for the cuOT leaf diagnostic.
# Mirrors millionaire.h:82-122 leaf; SCI untouched.
add_ot_test(leaf_cmp_ref_test)
# M3-T6 Phase C: CPU baseline full 32-bit Millionaire compare (SCI
# MillionaireProtocol::compare) — correctness + latency/comm baseline for the
# cuOT compare diagnostic. SCI untouched.
add_test(compare_ref_test)
add_test(share_tensor_backend_test)

# cuOT GPU OT backend + standalone tests (M2). Only built when
# OPENMM_ENABLE_CUOT=ON. The .cu sources are compiled by nvcc; the host
# .cc/.cpp by g++. No protocol path is touched.
if(OPENMM_ENABLE_CUOT)
  # cuot_provider.cc is host C++ (calls emp::FerretCOT templates + cuda_setdev
  # host wrappers). It MUST be compiled by the host compiler, NOT nvcc —
  # emp-tool headers pull std::vector into a device context otherwise. The
  # cuOT .cu sources (cuOT_GPU_SOURCES from FindcuOT) are compiled by nvcc.
  # cuOT::dev carries -maes/-mssse3/-msse4.1/-D_GLIBCXX_USE_CXX11_ABI=0 and the
  # CUDA + emp-tool + OpenSSL link line.
  macro (add_cuot_test _name)
    add_executable(${_name}
      "tests/${_name}.cpp"
      "include/gpu_mm/cuot_provider.cc"
      "include/gpu_mm/cuot_kernels.cu"
      ${cuOT_GPU_SOURCES})
    target_link_libraries(${_name} PRIVATE cuOT::dev)
    set_target_properties(${_name} PROPERTIES CXX_STANDARD 20)
    # Force .cc/.cpp to the host compiler (g++), NOT nvcc: emp-tool headers
    # use std::vector, which nvcc rejects in device context. The .cu sources
    # (cuot_kernels.cu + cuOT_GPU_SOURCES) stay CUDA (compiled by nvcc).
    set_source_files_properties("include/gpu_mm/cuot_provider.cc"
      "tests/${_name}.cpp" PROPERTIES LANGUAGE CXX)
    set_source_files_properties("include/gpu_mm/cuot_kernels.cu"
      ${cuOT_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
    # CRITICAL: prepend cuOT's include dirs BEFORE the project's build/include.
    # The CPU emp-ot is installed at build/include/emp-ot/ferret/constants.h
    # and SHARDS cuOT's emp-ot/ferret/constants.h (same guard, same relative
    # path) — but the CPU one lacks PrimalLPNParameter/ferret_b13 and uses a
    # block*-flavored FerretCOT. We must resolve emp-ot/* to cuOT's fork.
    # emp-tool/* has no collision (only in build/include) so it still resolves.
    target_include_directories(${_name} BEFORE PRIVATE ${cuOT_INCLUDE_DIRS})
  endmacro()
  add_cuot_test(cuot_correlation_test)
  # M3-T6 OPT-1 risk gate: GPU MITCCRH (cuot_kernels.cu) vs CPU NaryMitccrh
  # micro-bench. Pure AES timing, no network. Decides whether full GPU-resident
  # rewiring is worth it (GPU must beat CPU AES-NI on the leaf pad workload).
  add_cuot_test(cuot_mitccrh_bench)
  add_cuot_test(cuot_deltaot_arithmetic_test)
  # M3-T3: cuOT-backed wrap-bit logical-OR. Same WrapBitOr primitive as the
  # CPU wrap_bit_or_test, but backed by CuotProvider (GPU Ferret). Same boundary
  # + 100k cases. Inherits cuOT's ~2% RCOT floor — adapter-logic/wiring pass,
  # NOT a clean 100k pass (see tests/cuot_wrap_bit_or_test.cpp header).
  add_cuot_test(cuot_wrap_bit_or_test)
  # M3-T4: cuOT-backed ring→field OR_AUX. Same RingToField primitive as the
  # CPU ring_to_field_or_aux_test, backed by CuotProvider (GPU Ferret). Same
  # boundary + 100k cases. Inherits cuOT's ~2% RCOT floor — adapter-logic/
  # wiring pass, NOT a clean 100k pass (see tests/cuot_ring_to_field_test.cpp).
  add_cuot_test(cuot_ring_to_field_test)
  # M3-T5: cuOT-backed field→ring OR_AUX. Same FieldToRing primitive as the
  # CPU field_to_ring_or_aux_test, backed by CuotProvider. Same boundary +
  # 100k cases. Inherits cuOT's ~2% RCOT floor — adapter-logic/wiring pass,
  # NOT a clean 100k pass (see tests/cuot_field_to_ring_test.cpp).
  add_cuot_test(cuot_field_to_ring_test)
  # M3-T6 Phase A: cuOT single-digit compare leaf (1-OO-N-KOT reduced to
  # block chosen-COT + garbled table). Same bitlengths/cases as the CPU
  # leaf_cmp_ref_test, backed by CuotProvider. Diagnostic only — inherits
  # cuOT's ~2% RCOT floor compounded over logN (see tests/cuot_leaf_cmp_test.cpp).
  add_cuot_test(cuot_leaf_cmp_test)
  # M3-T6 Phase B: cuOT block-ROT primitive (send_rot_blocks/recv_rot_blocks,
  # re-derived on send_cot_blocks/recv_cot_blocks + MITCCRH decorrelation) +
  # cuOT bit-triple (_2ROT Beaver). Validates the ROT in isolation, then the
  # triple on top. Diagnostic — inherits ~2% RCOT floor (x2 ROTs ~4%).
  add_cuot_test(cuot_rot_smoke_test)
  add_cuot_test(cuot_bit_triple_test)
  # M3-T6 Phase C: cuOT full 32-bit Millionaire compare. Composes Phase A's
  # KkotLeafCmp (2-bit leaf) + Phase B's BitTripleGen (_2ROT AND-tree) into a
  # full compare. Diagnostic — inherits ~2% RCOT floor compounded over
  # 8-digit leaf + 11-triple tree; reports error rate + latency + comm.
  add_cuot_test(cuot_compare_test)
endif()
