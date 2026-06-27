// HE backend smoke test (Task M1-T1).
//
// Single-process, single-party (BOB-style: holds the secret key). No network,
// no two parties, no CUDA/cuOT/Phantom. Verifies SealHEBackend wraps the basic
// SEAL BFV operations behind the gpu_mm::HEBackend interface for BOTH encodings:
//   - kCoeff (FC / matvec / inner-product path, no BatchEncoder)
//   - kSIMD  (BOLE / OLE / BN element-wise path, BatchEncoder)
//
// Checks:
//   encrypt_symmetric -> decrypt round-trips for both encodings
//   multiply_plain multiplies a ciphertext by an encoded plain correctly
//   add_plain adds an encoded plain to a ciphertext correctly
//   add_inplace accumulates ciphertexts
// All checks operate mod the plaintext modulus, matching M&M semantics.
#include "gpu_mm/he_backend.h"
#include "gpu_mm/seal_he_backend.h"

#include <seal/seal.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

uint64_t add_mod(uint64_t a, uint64_t b, uint64_t p) {
    a %= p;
    b %= p;
    uint64_t r = a + b;
    if (r >= p || r < a) r -= p;  // handle wrap
    return r >= p ? r - p : r;
}
uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t p) {
    a %= p;
    b %= p;
    uint64_t res = 0;
    while (b) {
        if (b & 1) res = add_mod(res, a, p);
        a = add_mod(a, a, p);
        b >>= 1;
    }
    return res;
}

const char* status_str(gpu_mm::HEStatus s) {
    switch (s) {
        case gpu_mm::HEStatus::kOk: return "ok";
        case gpu_mm::HEStatus::kInvalidArg: return "invalid_arg";
        case gpu_mm::HEStatus::kConfig: return "config";
        case gpu_mm::HEStatus::kUnsupported: return "unsupported";
        case gpu_mm::HEStatus::kInternal: return "internal";
    }
    return "?";
}

#define CHECK_HE(expr)                                                     \
    do {                                                                   \
        auto _s = (expr);                                                  \
        if (_s != gpu_mm::HEStatus::kOk) {                                 \
            std::cerr << "FAIL " << #expr << " -> " << status_str(_s)      \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1;                                                      \
        }                                                                  \
    } while (0)

// Read coefficient-encoded plaintext slots back from a Plaintext: the low `len`
// coefficients, each reduced mod p.
void read_coeff(const seal::Plaintext& pt, uint64_t p, size_t len,
                std::vector<uint64_t>& out) {
    out.assign(len, 0);
    for (size_t i = 0; i < len; ++i) {
        out[i] = (i < static_cast<size_t>(pt.coeff_count()))
                     ? (pt[i] % p)
                     : 0;
    }
}

// Convolution coefficient k of a(x)*b(x) mod p (a,b length `len`). Coefficient
// packing makes multiply_plain a polynomial product, NOT pointwise — this is the
// property FC/matvec exploits. So the coeff-mul check must compare convolutions.
uint64_t conv_coeff(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b,
                    size_t k, uint64_t p) {
    uint64_t acc = 0;
    for (size_t i = 0; i <= k && i < a.size(); ++i) {
        size_t j = k - i;
        if (j < b.size()) acc = add_mod(acc, mul_mod(a[i], b[j], p), p);
    }
    return acc;
}

// Read SIMD-encoded plaintext slots back via BatchEncoder.
void read_simd(seal::BatchEncoder& enc, const seal::Plaintext& pt,
               size_t len, std::vector<uint64_t>& out) {
    std::vector<uint64_t> all;
    enc.decode(pt, all);
    out.assign(len, 0);
    for (size_t i = 0; i < len; ++i) out[i] = all[i];
}

int run() {
    using namespace gpu_mm;
    SealHEBackend he;

    // NTT-friendly prime so BOTH encodings work (kSIMD needs batching).
    const size_t N = 4096;
    const uint64_t p = 65537;  // Fermat prime, NTT-friendly
    CHECK_HE(he.setup(N, p));

    if (he.poly_degree() != N || he.plain_modulus() != p) {
        std::cerr << "FAIL introspection mismatch\n";
        return 1;
    }

    // -------- kCoeff path (FC / matvec) --------
    {
        const size_t len = 8;
        std::vector<uint64_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<uint64_t> b(len);
        for (size_t i = 0; i < len; ++i) b[i] = (p - a[i]) % p;  // for mul test

        seal::Plaintext pa, pb, pdec;
        seal::Ciphertext ca, cmul, cadd, ctmp;

        CHECK_HE(he.encode_coeff_plain(a.data(), len, &pa));
        CHECK_HE(he.encode_coeff_plain(b.data(), len, &pb));
        CHECK_HE(he.encrypt_symmetric(&pa, &ca));

        // decrypt round-trip
        CHECK_HE(he.decrypt(&ca, &pdec));
        std::vector<uint64_t> got;
        read_coeff(pdec, p, len, got);
        for (size_t i = 0; i < len; ++i) {
            if (got[i] != a[i] % p) {
                std::cerr << "FAIL coeff decrypt @ " << i << " got=" << got[i]
                          << " want=" << a[i] % p << "\n";
                return 1;
            }
        }

        // multiply_plain: ct(a) * b(x)  =>  polynomial product a(x)*b(x) mod p.
        // Coefficient packing => this is a convolution, NOT pointwise. (This is
        // the property FC / matvec relies on.)
        CHECK_HE(he.multiply_plain(&ca, &pb, &cmul));
        CHECK_HE(he.decrypt(&cmul, &pdec));
        read_coeff(pdec, p, len, got);
        for (size_t k = 0; k < len; ++k) {
            uint64_t want = conv_coeff(a, b, k, p);
            if (got[k] != want) {
                std::cerr << "FAIL coeff mul_plain @ " << k << " got=" << got[k]
                          << " want=" << want << "\n";
                return 1;
            }
        }

        // add_plain: ct + pb => coeffs (a + b) mod p
        CHECK_HE(he.add_plain(&ca, &pb, &cadd));
        CHECK_HE(he.decrypt(&cadd, &pdec));
        read_coeff(pdec, p, len, got);
        for (size_t i = 0; i < len; ++i) {
            uint64_t want = add_mod(a[i], b[i], p);
            if (got[i] != want) {
                std::cerr << "FAIL coeff add_plain @ " << i << " got=" << got[i]
                          << " want=" << want << "\n";
                return 1;
            }
        }

        // add_inplace: cadd += cmul => (a+b) + a*b  (as polynomials, coeffwise add)
        CHECK_HE(he.add_inplace(&cadd, &cmul));
        CHECK_HE(he.decrypt(&cadd, &pdec));
        read_coeff(pdec, p, len, got);
        for (size_t k = 0; k < len; ++k) {
            uint64_t want = add_mod(add_mod(a[k], b[k], p), conv_coeff(a, b, k, p), p);
            if (got[k] != want) {
                std::cerr << "FAIL coeff add_inplace @ " << k << " got=" << got[k]
                          << " want=" << want << "\n";
                return 1;
            }
        }
        std::cout << "[coeff] encrypt/decrypt, multiply_plain, add_plain, "
                     "add_inplace OK"
                  << std::endl;
    }

    // -------- kSIMD path (BOLE / OLE / BN element-wise) --------
    {
        const size_t len = 16;
        std::vector<uint64_t> a(len), b(len);
        for (size_t i = 0; i < len; ++i) {
            a[i] = (i + 1) % p;
            b[i] = (3 * i + 7) % p;
        }

        seal::Plaintext pa, pb, pdec;
        seal::Ciphertext ca, cmul, cadd;
        CHECK_HE(he.encode_simd_plain(a.data(), len, &pa));
        CHECK_HE(he.encode_simd_plain(b.data(), len, &pb));
        CHECK_HE(he.encrypt_symmetric(&pa, &ca));

        // Need a BatchEncoder to decode back; reuse the backend's via a fresh
        // SEAL object is not exposed, so reconstruct from context-equivalent
        // params. Simpler: the backend's poly_degree()==N and plain mod==p,
        // build a BatchEncoder on an identical context.
        seal::EncryptionParameters parms(seal::scheme_type::bfv);
        parms.set_n_special_primes(0);
        parms.set_poly_modulus_degree(N);
        parms.set_plain_modulus(p);
        parms.set_coeff_modulus(seal::CoeffModulus::Create(N, {60, 49}));
        seal::SEALContext ctx(parms, true, seal::sec_level_type::tc128);
        seal::BatchEncoder benc(ctx);

        // round-trip
        CHECK_HE(he.decrypt(&ca, &pdec));
        std::vector<uint64_t> got;
        read_simd(benc, pdec, len, got);
        for (size_t i = 0; i < len; ++i) {
            if (got[i] != a[i]) {
                std::cerr << "FAIL simd decrypt @ " << i << " got=" << got[i]
                          << " want=" << a[i] << "\n";
                return 1;
            }
        }

        // multiply_plain: slot-wise a*b
        CHECK_HE(he.multiply_plain(&ca, &pb, &cmul));
        CHECK_HE(he.decrypt(&cmul, &pdec));
        read_simd(benc, pdec, len, got);
        for (size_t i = 0; i < len; ++i) {
            uint64_t want = mul_mod(a[i], b[i], p);
            if (got[i] != want) {
                std::cerr << "FAIL simd mul_plain @ " << i << " got=" << got[i]
                          << " want=" << want << "\n";
                return 1;
            }
        }

        // add_plain: slot-wise a+b
        CHECK_HE(he.add_plain(&ca, &pb, &cadd));
        CHECK_HE(he.decrypt(&cadd, &pdec));
        read_simd(benc, pdec, len, got);
        for (size_t i = 0; i < len; ++i) {
            uint64_t want = add_mod(a[i], b[i], p);
            if (got[i] != want) {
                std::cerr << "FAIL simd add_plain @ " << i << " got=" << got[i]
                          << " want=" << want << "\n";
                return 1;
            }
        }
        std::cout << "[simd]  encrypt/decrypt, multiply_plain, add_plain OK"
                  << std::endl;
    }

    std::cout << "HE backend smoke test PASSED" << std::endl;
    return 0;
}

}  // namespace

int main() { return run(); }
