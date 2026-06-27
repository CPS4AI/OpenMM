# FindcuOT.cmake — locate the local cuOT source tree (M2-T1).
#
# cuOT (https://github.com/Andrew-Gan/cuOT, archived) ships NO install target
# and NO `find_package(cuOT)` config. It is consumed as a *source tree*: the
# consumer compiles its `.cu` sources (nvcc) and links the CUDA + emp-tool +
# OpenSSL libraries itself. This module only locates the tree and exposes the
# source/include lists + link libs as a helper interface target.
#
# It does NOT download anything. The tree is expected at
# `${CMAKE_SOURCE_DIR}/deps/cuOT` (cloned manually per the M2 plan).
#
# Result variables:
#   cuOT_FOUND
#   cuOT_ROOT_DIR         - the deps/cuOT path
#   cuOT_INCLUDE_DIRS     - gpu/ + ferret/emp-ot (for emp-ot/ferret/*.h)
#   cuOT_GPU_SOURCES      - gpu/*.cu + ferret/.../dev_layer.cu (compile with nvcc)
#
# Result target:
#   cuOT::dev             - INTERFACE library carrying:
#                             include dirs + CUDA + emp-tool + OpenSSL link libs
#                             + the host compile flags cuOT host code needs
#                             (-maes -mssse3 -msse4.1 -D_GLIBCXX_USE_CXX11_ABI=0)
#
# Consumer recipe (see add_cuot_test in the top-level CMakeLists.txt):
#   add_executable(my_test tests/my_test.cpp include/gpu_mm/cuot_provider.cc
#                  ${cuOT_GPU_SOURCES})
#   target_link_libraries(my_test PRIVATE cuOT::dev)
#   set_target_properties(my_test PROPERTIES ... per-source CUDA vs host ...)
#
# Note on host-vs-device compilation: cuOT's .cu files must be compiled by
# nvcc (they contain __global__/__device__ kernels). cuOT host code
# (cuot_provider.cc) is plain C++ and MUST be compiled by the host compiler
# (g++), NOT nvcc — emp-tool headers use std::vector, which nvcc rejects in
# device context. The add_cuot_test macro sets per-source languages so .cu
# uses nvcc and .cc/.cpp use g++.

# Look for the source tree.
find_path(cuOT_ROOT_DIR
  NAMES gpu/gpu_define.h
  PATHS "${CMAKE_SOURCE_DIR}/deps/cuOT"
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuOT
  DEFAULT_MSG cuOT_ROOT_DIR)

if(cuOT_FOUND)
  set(cuOT_INCLUDE_DIRS
    "${cuOT_ROOT_DIR}/gpu"
    "${cuOT_ROOT_DIR}/ferret/emp-ot"
    CACHE PATH "cuOT include dirs")

  file(GLOB cuOT_GPU_SOURCES
    "${cuOT_ROOT_DIR}/gpu/*.cu")
  list(APPEND cuOT_GPU_SOURCES
    "${cuOT_ROOT_DIR}/ferret/emp-ot/emp-ot/ferret/dev_layer.cu")
  set(cuOT_GPU_SOURCES ${cuOT_GPU_SOURCES} CACHE FILEPATH "cuOT .cu sources")

  if(NOT TARGET cuOT::dev)
    add_library(cuOT::dev INTERFACE IMPORTED)
    # Include dirs: cuOT gpu + ferret/emp-ot, our emp-tool install, miniforge3
    # OpenSSL headers (emp-tool needs openssl/ec.h).
    set_target_properties(cuOT::dev PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
      "${cuOT_INCLUDE_DIRS};${CMAKE_SOURCE_DIR}/build/include;/home/richorange/miniforge3/include")
    # Link: CUDA runtime/driver/sparse/curand, our emp-tool, OpenSSL.
    # emp-tool was built ABI=0 (same as the rest of the project) at build/lib.
    set_target_properties(cuOT::dev PROPERTIES INTERFACE_LINK_LIBRARIES
      "CUDA::cudart;CUDA::cuda_driver;CUDA::cusparse;CUDA::curand;emp-tool;ssl;crypto")
    set_target_properties(cuOT::dev PROPERTIES INTERFACE_LINK_DIRECTORIES
      "${CMAKE_SOURCE_DIR}/build/lib;/home/richorange/miniforge3/lib")
    # Host compile flags for cuOT host sources: AES intrinsics + ABI=0 + c++20.
    # (The .cu sources compiled by nvcc get -std=c++20 from CMAKE_CUDA_STANDARD.)
    set_target_properties(cuOT::dev PROPERTIES INTERFACE_COMPILE_OPTIONS
      "-maes;-mssse3;-msse4.1")
    target_compile_definitions(cuOT::dev INTERFACE _GLIBCXX_USE_CXX11_ABI=0)
  endif()

  message(STATUS "cuOT source tree: ${cuOT_ROOT_DIR}")
  message(STATUS "cuOT GPU sources: ${cuOT_GPU_SOURCES}")
endif()
