// ShareTensorBackend — abstract share-tensor backend interface (Task M1-T3).
//
// Goal (CLAUDE.md dev order, step 3): a backend seam for the local share-tensor
// arithmetic (add/sub/scalar-mul/sign-extract) that the conversion protocols do
// on CPU today, so a future CUDA share-tensor backend can sit behind the same
// interface. The CPU impl (cpu_share_tensor_backend.{h,cc}) is the
// source-of-truth reference a CUDA impl must match bit-for-bit.
//
// Two modulus flavors, matching how M&M splits arithmetic:
//   - ring mod 2^k   : add/sub mod 2^k (wrapping), used by ring-domain shares.
//   - field mod p    : add/sub/scalar-mul mod prime p, used by field-domain shares
//                      (the BFV plaintext field after ring_to_field).
// Sign extraction maps a value in Z_{2^k} to its two's-complement signed
// interpretation (mirrors sci::signed_val in utils.hpp): the top valid bit is
// the sign bit, so values >= 2^{k-1} are interpreted as x - 2^k.
//
// This header is backend-agnostic; concrete backends pull in their own kernels.
// Shares are plain uint64_t buffers (no SIMD/HE types here) so the interface is
// trivially portable to CUDA (device pointers) later.
#ifndef GPU_MM_SHARE_TENSOR_BACKEND_H
#define GPU_MM_SHARE_TENSOR_BACKEND_H

#include <cstdint>
#include <cstddef>

namespace gpu_mm {

enum class STStatus {
    kOk = 0,
    kInvalidArg,
    kUnsupported,
    kInternal,
};

// Abstract share-tensor backend. All ops are elementwise over `n` uint64 values
// and operate in the modulus given by the argument (ring bitwidth k or prime p).
// Output may alias an input (in-place) unless noted; callers must ensure buffers
// are valid for `n` elements.
class ShareTensorBackend {
   public:
    virtual ~ShareTensorBackend() = default;
    virtual const char* backend_name() const = 0;

    // --- ring mod 2^k ------------------------------------------------------
    // out[i] = (a[i] + b[i]) mod 2^k
    virtual STStatus add_mod2k(const uint64_t* a, const uint64_t* b, int n,
                               int k, uint64_t* out) = 0;
    // out[i] = (a[i] - b[i]) mod 2^k
    virtual STStatus sub_mod2k(const uint64_t* a, const uint64_t* b, int n,
                               int k, uint64_t* out) = 0;

    // --- field mod p (prime) ----------------------------------------------
    // out[i] = (a[i] + b[i]) mod p
    virtual STStatus add_modp(const uint64_t* a, const uint64_t* b, int n,
                              uint64_t p, uint64_t* out) = 0;
    // out[i] = (a[i] - b[i]) mod p
    virtual STStatus sub_modp(const uint64_t* a, const uint64_t* b, int n,
                              uint64_t p, uint64_t* out) = 0;
    // out[i] = (s * a[i]) mod p   (scalar-multiply a vector by s)
    virtual STStatus scalar_mul_modp(const uint64_t* a, uint64_t s, int n,
                                     uint64_t p, uint64_t* out) = 0;

    // --- sign extraction (reference) --------------------------------------
    // out[i] = signed two's-complement interpretation of a[i] in Z_{2^k}:
    //   a[i] < 2^{k-1}  => a[i]
    //   a[i] >= 2^{k-1} => a[i] - 2^k   (as a non-mod uint64, i.e. wraps if k<64)
    // Mirrors sci::signed_val. Used by M&M's correctness checks and as a CPU
    // reference for future CUDA sign-bit extraction.
    virtual STStatus sign_extract(const uint64_t* a, int n, int k,
                                  uint64_t* out) = 0;
};

}  // namespace gpu_mm

#endif  // GPU_MM_SHARE_TENSOR_BACKEND_H
