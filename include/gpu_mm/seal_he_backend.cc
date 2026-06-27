// SealHEBackend implementation (Task M1-T1).
#include "gpu_mm/seal_he_backend.h"

#include <seal/seal.h>
#include <seal/util/polyarithsmallmod.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace gpu_mm {

SealHEBackend::SealHEBackend() = default;
SealHEBackend::~SealHEBackend() = default;

bool SealHEBackend::ready() const {
    return context_ && encryptor_ && decryptor_ && evaluator_;
}

HEStatus SealHEBackend::setup(size_t poly_degree, uint64_t plain_mod,
                              const std::vector<int>& coeff_mod_bits) {
    if (poly_degree == 0 || (poly_degree & (poly_degree - 1)) != 0) {
        return HEStatus::kInvalidArg;  // must be a power of two
    }
    if (plain_mod < 2) {
        return HEStatus::kInvalidArg;
    }
    if (coeff_mod_bits.size() < 2) {
        return HEStatus::kInvalidArg;
    }

    try {
        seal::EncryptionParameters parms(seal::scheme_type::bfv);
        parms.set_n_special_primes(0);
        parms.set_poly_modulus_degree(poly_degree);
        parms.set_plain_modulus(plain_mod);
        parms.set_coeff_modulus(
            seal::CoeffModulus::Create(poly_degree, coeff_mod_bits));

        context_ = std::make_shared<seal::SEALContext>(
            parms, true, seal::sec_level_type::tc128);
        if (!context_->parameters_set()) {
            context_.reset();
            return HEStatus::kInvalidArg;
        }

        keygen_ = std::make_shared<seal::KeyGenerator>(*context_);
        sk_ = std::make_shared<seal::SecretKey>(keygen_->secret_key());
        seal::Serializable<seal::PublicKey> spk = keygen_->create_public_key();
        // Materialize the serializable public key into a real PublicKey via a
        // stringstream round-trip (same pattern as CheetahLinear's ctor).
        std::stringstream os;
        spk.save(os);
        pk_ = std::make_shared<seal::PublicKey>();
        pk_->load(*context_, os);

        encryptor_ = std::make_shared<seal::Encryptor>(*context_, *sk_);
        decryptor_ = std::make_shared<seal::Decryptor>(*context_, *sk_);
        evaluator_ = std::make_shared<seal::Evaluator>(*context_);

        plain_mod_ = plain_mod;
        poly_degree_ = poly_degree;

        // BatchEncoder is available iff the plain modulus is NTT-friendly (the
        // SIMD batching qualifier). For non-NTT-friendly plain mods (e.g. 2^k
        // ring modulus used by the coeff-direct VOLE path), BatchEncoder cannot
        // be created and kSIMD encoding will return kUnsupported.
        const auto q = context_->first_context_data()->qualifiers();
        has_batch_ = static_cast<bool>(q.using_batching);
        if (has_batch_) {
            batch_encoder_ = std::make_shared<seal::BatchEncoder>(*context_);
        } else {
            batch_encoder_.reset();
        }
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
    return HEStatus::kOk;
}

uint64_t SealHEBackend::plain_modulus() const {
    return plain_mod_;
}

size_t SealHEBackend::poly_degree() const {
    return poly_degree_;
}

HEStatus SealHEBackend::encode_plain(const uint64_t* values, size_t len,
                                     HEEncoding enc, void* pt_out) {
    if (!ready() || values == nullptr || pt_out == nullptr) {
        return HEStatus::kInvalidArg;
    }
    if (len == 0 || len > poly_degree_) {
        return HEStatus::kInvalidArg;
    }
    auto* pt = as<seal::Plaintext>(pt_out);

    try {
        if (enc == HEEncoding::kCoeff) {
            // Coefficient packing — mirrors HomFCSS::vec2PolyBFV: reduce the low
            // `len` coefficients mod plain_mod_, zero the rest. No BatchEncoder.
            pt->parms_id() = seal::parms_id_zero;  // BFV non-NTT plaintext
            pt->resize(poly_degree_);
            if (pt->data() == nullptr) return HEStatus::kInternal;
            seal::util::modulo_poly_coeffs(values, len,
                                           seal::Modulus(plain_mod_), pt->data());
            std::fill_n(pt->data() + len, pt->coeff_count() - len, 0ULL);
            return HEStatus::kOk;
        } else if (enc == HEEncoding::kSIMD) {
            if (!has_batch_ || !batch_encoder_) {
                return HEStatus::kUnsupported;
            }
            std::vector<uint64_t> slot_vals(poly_degree_, 0);
            std::copy_n(values, len, slot_vals.data());
            batch_encoder_->encode(slot_vals, *pt);
            return HEStatus::kOk;
        }
        return HEStatus::kInvalidArg;
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
}

HEStatus SealHEBackend::encrypt_symmetric(const void* pt_in, void* ct_out) {
    if (!ready() || pt_in == nullptr || ct_out == nullptr) {
        return HEStatus::kInvalidArg;
    }
    auto* pt = as<seal::Plaintext>(pt_in);
    auto* ct = as<seal::Ciphertext>(ct_out);
    try {
        encryptor_->encrypt_symmetric(*pt, *ct);
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
    return HEStatus::kOk;
}

HEStatus SealHEBackend::decrypt(const void* ct_in, void* pt_out) {
    if (!ready() || ct_in == nullptr || pt_out == nullptr) {
        return HEStatus::kInvalidArg;
    }
    auto* ct = as<seal::Ciphertext>(ct_in);
    auto* pt = as<seal::Plaintext>(pt_out);
    try {
        decryptor_->decrypt(*ct, *pt);
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
    return HEStatus::kOk;
}

HEStatus SealHEBackend::multiply_plain(const void* ct_in, const void* pt,
                                       void* ct_out) {
    if (!ready() || ct_in == nullptr || pt == nullptr || ct_out == nullptr) {
        return HEStatus::kInvalidArg;
    }
    auto* ct_a = as<seal::Ciphertext>(ct_in);
    auto* plain = as<seal::Plaintext>(pt);
    auto* ct_r = as<seal::Ciphertext>(ct_out);
    try {
        evaluator_->multiply_plain(*ct_a, *plain, *ct_r);
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
    return HEStatus::kOk;
}

HEStatus SealHEBackend::add_plain(const void* ct_in, const void* pt,
                                  void* ct_out) {
    if (!ready() || ct_in == nullptr || pt == nullptr || ct_out == nullptr) {
        return HEStatus::kInvalidArg;
    }
    auto* ct_a = as<seal::Ciphertext>(ct_in);
    auto* plain = as<seal::Plaintext>(pt);
    auto* ct_r = as<seal::Ciphertext>(ct_out);
    try {
        evaluator_->add_plain(*ct_a, *plain, *ct_r);
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
    return HEStatus::kOk;
}

HEStatus SealHEBackend::add_inplace(void* ct_dst, const void* ct_src) {
    if (!ready() || ct_dst == nullptr || ct_src == nullptr) {
        return HEStatus::kInvalidArg;
    }
    auto* dst = as<seal::Ciphertext>(ct_dst);
    auto* src = as<seal::Ciphertext>(ct_src);
    try {
        evaluator_->add_inplace(*dst, *src);
    } catch (const std::exception&) {
        return HEStatus::kInternal;
    }
    return HEStatus::kOk;
}

}  // namespace gpu_mm
