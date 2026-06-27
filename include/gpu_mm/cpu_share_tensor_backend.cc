// CpuShareTensorBackend implementation (Task M1-T3).
#include "gpu_mm/cpu_share_tensor_backend.h"

#include <cstdint>

namespace gpu_mm {

namespace {

inline bool valid_args(const uint64_t* a, const uint64_t* b, const uint64_t* out,
                       int n) {
    return a != nullptr && out != nullptr && n > 0 &&
           (b != nullptr || b == nullptr);  // b optional per op; checked by caller
}

inline uint64_t ring_mask(int k) {
    // mod 2^k: keep the low k bits. k==64 => all bits.
    return (k >= 64) ? ~0ULL : ((1ULL << k) - 1);
}

// 128-bit mod reduction for field ops, matching M&M field arithmetic.
inline uint64_t mod_p(uint64_t x, uint64_t p) { return x % p; }
inline uint64_t add_mod_p(uint64_t a, uint64_t b, uint64_t p) {
    a %= p;
    b %= p;
    uint64_t r = a + b;
    if (r >= p || r < a) r -= p;  // wrapped or exceeded
    return r >= p ? r - p : r;
}
inline uint64_t sub_mod_p(uint64_t a, uint64_t b, uint64_t p) {
    a %= p;
    b %= p;
    return a >= b ? a - b : p - (b - a);
}
inline uint64_t mul_mod_p(uint64_t a, uint64_t b, uint64_t p) {
    return static_cast<__uint128_t>(a % p) * (b % p) % p;
}

}  // namespace

STStatus CpuShareTensorBackend::add_mod2k(const uint64_t* a, const uint64_t* b,
                                          int n, int k, uint64_t* out) {
    if (a == nullptr || b == nullptr || out == nullptr || n <= 0 || k <= 0 ||
        k > 64) {
        return STStatus::kInvalidArg;
    }
    const uint64_t m = ring_mask(k);
    for (int i = 0; i < n; ++i) out[i] = (a[i] + b[i]) & m;
    return STStatus::kOk;
}

STStatus CpuShareTensorBackend::sub_mod2k(const uint64_t* a, const uint64_t* b,
                                          int n, int k, uint64_t* out) {
    if (a == nullptr || b == nullptr || out == nullptr || n <= 0 || k <= 0 ||
        k > 64) {
        return STStatus::kInvalidArg;
    }
    const uint64_t m = ring_mask(k);
    for (int i = 0; i < n; ++i) out[i] = (a[i] - b[i]) & m;
    return STStatus::kOk;
}

STStatus CpuShareTensorBackend::add_modp(const uint64_t* a, const uint64_t* b,
                                         int n, uint64_t p, uint64_t* out) {
    if (a == nullptr || b == nullptr || out == nullptr || n <= 0 || p < 2) {
        return STStatus::kInvalidArg;
    }
    for (int i = 0; i < n; ++i) out[i] = add_mod_p(a[i], b[i], p);
    return STStatus::kOk;
}

STStatus CpuShareTensorBackend::sub_modp(const uint64_t* a, const uint64_t* b,
                                         int n, uint64_t p, uint64_t* out) {
    if (a == nullptr || b == nullptr || out == nullptr || n <= 0 || p < 2) {
        return STStatus::kInvalidArg;
    }
    for (int i = 0; i < n; ++i) out[i] = sub_mod_p(a[i], b[i], p);
    return STStatus::kOk;
}

STStatus CpuShareTensorBackend::scalar_mul_modp(const uint64_t* a, uint64_t s,
                                                int n, uint64_t p,
                                                uint64_t* out) {
    if (a == nullptr || out == nullptr || n <= 0 || p < 2) {
        return STStatus::kInvalidArg;
    }
    for (int i = 0; i < n; ++i) out[i] = mul_mod_p(a[i], s, p);
    return STStatus::kOk;
}

STStatus CpuShareTensorBackend::sign_extract(const uint64_t* a, int n, int k,
                                             uint64_t* out) {
    if (a == nullptr || out == nullptr || n <= 0 || k <= 0 || k > 64) {
        return STStatus::kInvalidArg;
    }
    // Mirror sci::signed_val: x in [0,2^k); values >= 2^{k-1} are negative.
    const uint64_t pow_x = (k >= 64) ? 0ULL : (1ULL << k);
    const uint64_t mask_x = (k >= 64) ? ~0ULL : (pow_x - 1);
    for (int i = 0; i < n; ++i) {
        uint64_t x = a[i] & mask_x;
        out[i] = x - ((x >= (pow_x / 2)) * pow_x);
    }
    return STStatus::kOk;
}

}  // namespace gpu_mm
