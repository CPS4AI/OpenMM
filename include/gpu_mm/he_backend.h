// HEBackend — abstract homomorphic-encryption backend interface (Task M1-T1).
//
// Goal (CLAUDE.md dev order, step 3 "Backend interfaces without changing
// behavior"): introduce a backend seam for HE so that later GPU/PhantomFHE
// replacements can sit behind the same interface, WITHOUT touching any existing
// HomFCSS/HomBNSS/CheetahLinear call sites. Existing protocol paths keep using
// SEAL directly; this is a parallel, test-only abstraction.
//
// Scope: header-only interface + an enum for the concrete backend kind. The
// first concrete impl (SealHEBackend) lives in seal_he_backend.{h,cc}. No
// PhantomFHE here.
//
// Design constraints from Known Architecture Facts (CLAUDE.md):
//  - FC / matrix-vector / inner product use Cheetah-style COEFFICIENT encoding.
//    => encode_coeff_plain packs values into plaintext polynomial coefficients
//       (NOT BatchEncoder). This mirrors HomFCSS::vec2PolyBFV.
//  - Element-wise BOLE / OLE / BN uses BFV SIMD BatchEncoder.
//    => encode_simd_plain packs values via BatchEncoder SIMD slots. This mirrors
//       HomBNSS SIMD path.
// The interface exposes both encodings explicitly so callers cannot confuse
// them — a future Phantom backend must implement the same two encodings.
//
// This header intentionally does NOT depend on SEAL so the abstract interface
// stays backend-agnostic; concrete backends pull in their own HE headers. The
// plaintext/ciphertext handles are type-erased (void*) so the interface does not
// need a common FHE type. Concrete backends are responsible for casting to their
// own seal::* types and document the expected dynamic type.
#ifndef GPU_MM_HE_BACKEND_H
#define GPU_MM_HE_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_mm {

// Result / status code shared across gpu_mm backend interfaces.
enum class HEStatus {
    kOk = 0,
    kInvalidArg,
    kConfig,        // backend not set up (missing context/keys)
    kUnsupported,   // operation not implemented by this backend
    kInternal,
};

// Which encoding a plaintext uses. Mirrors the two M&M HE paths.
enum class HEEncoding {
    kCoeff,  // coefficient packing (FC / matvec / inner product) — no BatchEncoder
    kSIMD,   // BFV SIMD slot packing (element-wise BOLE / OLE / BN) — BatchEncoder
};

// Abstract HE backend. All plaintext/ciphertext objects are owned by the caller
// and passed as opaque handles (void*). Concrete backends document the expected
// dynamic type of each handle (e.g. seal::Plaintext* / seal::Ciphertext*).
//
// Lifetime contract: the caller owns all handles. The backend never deletes
// them. A handle is "a pointer to a value-initialized object of the backend's
// plaintext/ciphertext type"; encode/encrypt fill it, eval ops read+write it.
class HEBackend {
   public:
    virtual ~HEBackend() = default;

    // --- introspection -------------------------------------------------------
    // Plaintext modulus in use (== M&M field modulus p for SIMD, or 2^k ring
    // modulus for coeff-direct). Returns 0 if not set up.
    virtual uint64_t plain_modulus() const = 0;
    // BFV polynomial degree N (number of SIMD slots for kSIMD encoding).
    virtual size_t poly_degree() const = 0;
    // Human-readable backend name, e.g. "seal".
    virtual const char* backend_name() const = 0;

    // --- plaintext encoding --------------------------------------------------
    // Encode `len` values from `values` into the caller-owned plaintext handle
    // `pt_out` using the given encoding.
    //   kCoeff: values occupy the low `len` polynomial coefficients (mod p).
    //   kSIMD:  values occupy the first `len` BatchEncoder slots; rest zeroed.
    // `pt_out` must point to a value-initialized backend plaintext object.
    virtual HEStatus encode_plain(const uint64_t* values, size_t len,
                                  HEEncoding enc, void* pt_out) = 0;

    // Convenience: coefficient packing (FC path). Equivalent to
    // encode_plain(..., kCoeff).
    HEStatus encode_coeff_plain(const uint64_t* values, size_t len,
                                void* pt_out) {
        return encode_plain(values, len, HEEncoding::kCoeff, pt_out);
    }
    // Convenience: SIMD slot packing (BOLE/BN path).
    HEStatus encode_simd_plain(const uint64_t* values, size_t len,
                               void* pt_out) {
        return encode_plain(values, len, HEEncoding::kSIMD, pt_out);
    }

    // --- encryption / decryption -------------------------------------------
    // Symmetric-key encryption of a plaintext -> ciphertext. Mirrors SEAL's
    // Encryptor::encrypt_symmetric (the path HomFCSS uses for the encoder's
    // share). `pt_in` is a previously-encoded plaintext handle; `ct_out` is a
    // caller-owned ciphertext handle to fill.
    virtual HEStatus encrypt_symmetric(const void* pt_in, void* ct_out) = 0;

    // Decrypt `ct_in` into the caller-owned plaintext handle `pt_out`.
    // Requires the secret key to be configured.
    virtual HEStatus decrypt(const void* ct_in, void* pt_out) = 0;

    // --- homomorphic evaluation --------------------------------------------
    // ct_out = ct_in * pt (plain multiply). In-place safe to alias ct_in/out.
    virtual HEStatus multiply_plain(const void* ct_in, const void* pt,
                                    void* ct_out) = 0;
    // ct_out = ct_in + pt (plain add).
    virtual HEStatus add_plain(const void* ct_in, const void* pt,
                               void* ct_out) = 0;
    // ct_dst += ct_src (ciphertext-ciphertext add, in-place on dst).
    virtual HEStatus add_inplace(void* ct_dst, const void* ct_src) = 0;
};

}  // namespace gpu_mm

#endif  // GPU_MM_HE_BACKEND_H
