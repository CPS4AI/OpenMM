// SealHEBackend — SEAL-based concrete HEBackend (Task M1-T1).
//
// Wraps the basic SEAL BFV operations the M&M codebase already uses, behind the
// gpu_mm::HEBackend interface. This does NOT replace HomFCSS/HomBNSS/CheetahLinear;
// it is a standalone, test-only seam so later PhantomFHE can implement the same
// interface and be compared.
//
// Two encodings, per CLAUDE.md Known Architecture Facts:
//   kCoeff — coefficient packing, mirrors HomFCSS::vec2PolyBFV (NO BatchEncoder).
//            Values occupy low polynomial coefficients reduced mod plain_modulus.
//   kSIMD  — BFV BatchEncoder slot packing, mirrors HomBNSS SIMD path.
//
// Opaque handle contract (documented for future Phantom parity):
//   plaintext handle  -> seal::Plaintext*
//   ciphertext handle -> seal::Ciphertext*
// Callers allocate these (default-constructed) and pass &obj; the backend fills
// them. The backend never owns/deletes them.
#ifndef GPU_MM_SEAL_HE_BACKEND_H
#define GPU_MM_SEAL_HE_BACKEND_H

#include "gpu_mm/he_backend.h"

#include <seal/seal.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gpu_mm {

class SealHEBackend : public HEBackend {
   public:
    SealHEBackend();
    ~SealHEBackend() override;

    // Disable copy: holds shared_ptr SEAL objects; copying is not needed and
    // would invite accidental double-management. Explicit setup is the way.
    SealHEBackend(const SealHEBackend&) = delete;
    SealHEBackend& operator=(const SealHEBackend&) = delete;

    // --- setup ---------------------------------------------------------------
    // Configure a fresh BFV context. `poly_degree` is N, `plain_mod` is the
    // plaintext modulus (field prime p for SIMD, or 2^k ring modulus for coeff),
    // `coeff_mod_bits` selects the ciphertext moduli bit sizes (default mirrors
    // CheetahLinear: {60,49} for <=109-bit export). Generates a keypair and
    // keeps the secret key for symmetric encryption + decryption.
    //
    // Returns kInvalidArg on bad params, kOk on success.
    HEStatus setup(size_t poly_degree, uint64_t plain_mod,
                   const std::vector<int>& coeff_mod_bits = {60, 49});

    // --- HEBackend interface -------------------------------------------------
    uint64_t plain_modulus() const override;
    size_t poly_degree() const override;
    const char* backend_name() const override { return "seal"; }

    HEStatus encode_plain(const uint64_t* values, size_t len, HEEncoding enc,
                          void* pt_out) override;
    HEStatus encrypt_symmetric(const void* pt_in, void* ct_out) override;
    HEStatus decrypt(const void* ct_in, void* pt_out) override;
    HEStatus multiply_plain(const void* ct_in, const void* pt,
                            void* ct_out) override;
    HEStatus add_plain(const void* ct_in, const void* pt,
                       void* ct_out) override;
    HEStatus add_inplace(void* ct_dst, const void* ct_src) override;

   private:
    // Validate an opaque handle is a non-null pointer we may cast to T*.
    template <typename T>
    static T* as(const void* h) {
        return static_cast<T*>(const_cast<void*>(h));
    }

    bool ready() const;

    std::shared_ptr<seal::SEALContext> context_;
    std::shared_ptr<seal::KeyGenerator> keygen_;
    std::shared_ptr<seal::SecretKey> sk_;
    std::shared_ptr<seal::PublicKey> pk_;
    std::shared_ptr<seal::Encryptor> encryptor_;   // symmetric (sk)
    std::shared_ptr<seal::Decryptor> decryptor_;
    std::shared_ptr<seal::Evaluator> evaluator_;
    std::shared_ptr<seal::BatchEncoder> batch_encoder_;
    uint64_t plain_mod_ = 0;
    size_t poly_degree_ = 0;
    bool has_batch_ = false;
};

}  // namespace gpu_mm

#endif  // GPU_MM_SEAL_HE_BACKEND_H
