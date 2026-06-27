# gpu_mm backend-abstraction sources (M1 series).
#
# These compile into the existing `gemini` shared library so that the test
# targets (which link gemini via the add_test macro) get them automatically.
# Nothing here changes existing protocol behavior; these are new, parallel
# backend seams.
#
# Note: emp_ot_provider.cc is NOT listed here — it includes M&M's OT headers
# (utils/aes-ni.h) which need -maes, and the gemini target does not carry SCI's
# ISA flags. That source is instead compiled directly into the OT test
# executable via the add_ot_test macro in the top-level CMakeLists.txt, which
# links SCI-Cheetah and inherits -maes transitively.
target_sources(gemini PUBLIC
${CMAKE_CURRENT_LIST_DIR}/seal_he_backend.cc
${CMAKE_CURRENT_LIST_DIR}/cpu_share_tensor_backend.cc
)
