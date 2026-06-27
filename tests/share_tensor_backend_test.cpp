// Share-tensor backend test (Task M1-T3).
//
// Single-process, no network, no HE, no OT. Randomized tests against the CPU
// reference (CpuShareTensorBackend) for every op in the interface:
//   - add/sub mod 2^k  (ring)
//   - add/sub/scalar-mul mod p  (field)
//   - sign_extract
// Correctness is checked against a cleartext reference computed with __uint128_t,
// independent of the backend's own mod helpers where possible.
#include "gpu_mm/cpu_share_tensor_backend.h"
#include "gpu_mm/share_tensor_backend.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

uint64_t ring_mask(int k) { return (k >= 64) ? ~0ULL : ((1ULL << k) - 1); }

// Cleartext references (independent of the backend impl).
uint64_t ref_add2k(uint64_t a, uint64_t b, int k) {
    return (a + b) & ring_mask(k);
}
uint64_t ref_sub2k(uint64_t a, uint64_t b, int k) {
    return (a - b) & ring_mask(k);
}
uint64_t ref_addp(uint64_t a, uint64_t b, uint64_t p) {
    return (static_cast<__uint128_t>(a % p) + (b % p)) % p;
}
uint64_t ref_subp(uint64_t a, uint64_t b, uint64_t p) {
    __uint128_t r = (__uint128_t)(a % p) + p - (b % p);
    return r % p;
}
uint64_t ref_mulp(uint64_t a, uint64_t s, uint64_t p) {
    return (static_cast<__uint128_t>(a % p) * (s % p)) % p;
}
// sign_extract reference: signed two's-complement in Z_{2^k}.
// Mirrors sci::signed_val: pow_x=2^k for k<64, pow_x=0 for k==64 (=> identity).
uint64_t ref_sign(uint64_t a, int k) {
    const uint64_t pow_x = (k >= 64) ? 0ULL : (1ULL << k);
    const uint64_t mask_x = (k >= 64) ? ~0ULL : (pow_x - 1);
    uint64_t x = a & mask_x;
    return x - ((x >= (pow_x / 2)) * pow_x);
}

// simple xorshift PRNG (no Math.random in this env; deterministic seed)
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

int run() {
    using namespace gpu_mm;
    CpuShareTensorBackend st;
    EXPECT(std::string(st.backend_name()) == "cpu", "backend name");

    Rng rng(0x123456789abcdefULL);
    const int n = 4096;

    // ---- ring mod 2^k, several k ----
    for (int k : {1, 8, 16, 32, 37, 64}) {
        std::vector<uint64_t> a(n), b(n), out(n);
        for (int i = 0; i < n; ++i) {
            a[i] = rng.next();
            b[i] = rng.next();
        }
        EXPECT(st.add_mod2k(a.data(), b.data(), n, k, out.data()) == STStatus::kOk,
               "add_mod2k");
        for (int i = 0; i < n; ++i) {
            uint64_t want = ref_add2k(a[i], b[i], k);
            EXPECT(out[i] == want, "add_mod2k value");
        }
        EXPECT(st.sub_mod2k(a.data(), b.data(), n, k, out.data()) == STStatus::kOk,
               "sub_mod2k");
        for (int i = 0; i < n; ++i) {
            uint64_t want = ref_sub2k(a[i], b[i], k);
            EXPECT(out[i] == want, "sub_mod2k value");
        }
        // in-place alias check
        std::vector<uint64_t> c = a;
        EXPECT(st.add_mod2k(c.data(), b.data(), n, k, c.data()) == STStatus::kOk,
               "add_mod2k in-place");
        for (int i = 0; i < n; ++i)
            EXPECT(c[i] == ref_add2k(a[i], b[i], k), "add_mod2k in-place value");
    }

    // ---- field mod p (a few NTT-friendly + general primes) ----
    for (uint64_t p : {65537ULL, 4293918721ULL, 1099511480321ULL, 65521ULL}) {
        std::vector<uint64_t> a(n), b(n), out(n);
        uint64_t s = rng.next() % p;
        for (int i = 0; i < n; ++i) {
            a[i] = rng.next() % p;
            b[i] = rng.next() % p;
        }
        EXPECT(st.add_modp(a.data(), b.data(), n, p, out.data()) == STStatus::kOk,
               "add_modp");
        for (int i = 0; i < n; ++i)
            EXPECT(out[i] == ref_addp(a[i], b[i], p), "add_modp value");

        EXPECT(st.sub_modp(a.data(), b.data(), n, p, out.data()) == STStatus::kOk,
               "sub_modp");
        for (int i = 0; i < n; ++i)
            EXPECT(out[i] == ref_subp(a[i], b[i], p), "sub_modp value");

        EXPECT(st.scalar_mul_modp(a.data(), s, n, p, out.data()) == STStatus::kOk,
               "scalar_mul_modp");
        for (int i = 0; i < n; ++i)
            EXPECT(out[i] == ref_mulp(a[i], s, p), "scalar_mul_modp value");
    }

    // ---- sign_extract ----
    for (int k : {1, 8, 16, 32, 64}) {
        std::vector<uint64_t> a(n), out(n);
        for (int i = 0; i < n; ++i) a[i] = rng.next();
        EXPECT(st.sign_extract(a.data(), n, k, out.data()) == STStatus::kOk,
               "sign_extract");
        for (int i = 0; i < n; ++i) {
            uint64_t want = ref_sign(a[i], k);
            EXPECT(out[i] == want, "sign_extract value");
            // Branch cross-check (k<64 only): for k==64, 2^64 is unrepresentable
            // in uint64_t, so sign_extract is the identity (the bit pattern is
            // already the two's-complement value), matching sci::signed_val which
            // sets pow_x=0 for k==64. So the nonneg/neg output distinction only
            // holds for k<64.
            if (k < 64) {
                uint64_t m = ring_mask(k);
                uint64_t x = a[i] & m;
                uint64_t half = 1ULL << (k - 1);
                bool neg = (x >= half);
                EXPECT((out[i] == x) == (!neg), "sign_extract nonneg branch");
            }
        }
    }

    // ---- invalid-arg paths ----
    uint64_t dummy = 0;
    EXPECT(st.add_mod2k(nullptr, &dummy, 1, 8, &dummy) == STStatus::kInvalidArg,
           "add_mod2k null a");
    EXPECT(st.add_mod2k(&dummy, &dummy, 1, 0, &dummy) == STStatus::kInvalidArg,
           "add_mod2k k=0");
    EXPECT(st.add_modp(&dummy, &dummy, 1, 1, &dummy) == STStatus::kInvalidArg,
           "add_modp p<2");

    std::cout << "Share-tensor backend test PASSED (cpu, n=" << n << ")"
              << std::endl;
    return 0;
}

}  // namespace

int main() { return run(); }
