// CpuShareTensorBackend — CPU reference share-tensor backend (Task M1-T3).
//
// The source-of-truth implementation that a future CUDA backend must match
// bit-for-bit. Plain host uint64_t arithmetic; no SIMD/HE. Mirrors the mod
// semantics used throughout M&M (wrapping mod 2^k for rings, mod p for fields)
// and sci::signed_val for sign extraction.
#ifndef GPU_MM_CPU_SHARE_TENSOR_BACKEND_H
#define GPU_MM_CPU_SHARE_TENSOR_BACKEND_H

#include "gpu_mm/share_tensor_backend.h"

namespace gpu_mm {

class CpuShareTensorBackend : public ShareTensorBackend {
   public:
    CpuShareTensorBackend() = default;
    ~CpuShareTensorBackend() override = default;

    const char* backend_name() const override { return "cpu"; }

    STStatus add_mod2k(const uint64_t* a, const uint64_t* b, int n, int k,
                       uint64_t* out) override;
    STStatus sub_mod2k(const uint64_t* a, const uint64_t* b, int n, int k,
                       uint64_t* out) override;
    STStatus add_modp(const uint64_t* a, const uint64_t* b, int n, uint64_t p,
                      uint64_t* out) override;
    STStatus sub_modp(const uint64_t* a, const uint64_t* b, int n, uint64_t p,
                      uint64_t* out) override;
    STStatus scalar_mul_modp(const uint64_t* a, uint64_t s, int n, uint64_t p,
                             uint64_t* out) override;
    STStatus sign_extract(const uint64_t* a, int n, int k,
                          uint64_t* out) override;
};

}  // namespace gpu_mm

#endif  // GPU_MM_CPU_SHARE_TENSOR_BACKEND_H
