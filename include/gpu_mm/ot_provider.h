// OTProvider — abstract oblivious-transfer backend interface (Task M1-T2).
//
// Goal (CLAUDE.md dev order, step 3): a backend seam for OT so later cuOT can
// sit behind the same interface, WITHOUT modifying any conversion protocol.
// Existing M&M paths keep using sci::OTPack directly; this is a parallel,
// test-only abstraction.
//
// The M&M modulus-conversion protocols consume ONE OT shape: correlated OT
// (COT / "DeltaOT") in two modulus flavors — see Truncation::ring_to_field and
// friends calling otpack->iknp_straight->{send,recv}_cot / {send,recv}_cot_prime:
//   - send_cot/recv_cot   : COT over Z_{2^l}     (ring modulus, bitwidth l)
//   - send_cot_prime/...  : COT over Z_p          (field modulus, prime p)
// This interface exposes exactly that. A future cuOT backend implements the same.
//
// COT semantics (matching sci::SilentOT, see silent_ot.h):
//   Sender: chooses correlation `corr`. Receives random `data0` (= x).
//   Receiver: chooses choice bit `b`. Receives:
//       data = x            if b == 0
//       data = x + corr     if b == 1   (mod the modulus)
//   where x is the sender's data0. So sender holds x; receiver holds x or x+corr
//   depending on b. Both sides operate mod 2^l (ring) or mod p (field).
//
// This header is backend-agnostic (no emp/SCI includes) so it can wrap either
// emp-ot (M1-T2) or cuOT (later). Party role is passed at construction.
#ifndef GPU_MM_OT_PROVIDER_H
#define GPU_MM_OT_PROVIDER_H

#include <cstdint>
#include <cstddef>

namespace gpu_mm {

// Party roles (match sci::ALICE=1 / sci::BOB=2 conventions used in M&M).
enum class OTParty { kSender = 1, kReceiver = 2 };

enum class OTStatus {
    kOk = 0,
    kInvalidArg,
    kConfig,       // not set up
    kUnsupported,
    kInternal,
};

// Abstract OT provider. Stateful: a single instance serves as ONE party
// (sender OR receiver) over ONE channel. Two-party protocols construct two
// instances (one per party) backed by a connected NetIO pair.
class OTProvider {
   public:
    virtual ~OTProvider() = default;

    // Which party this instance acts as.
    virtual OTParty party() const = 0;
    // Human-readable backend name, e.g. "emp".
    virtual const char* backend_name() const = 0;

    // --- COT over Z_{2^l} (ring modulus) -----------------------------------
    // Sender side (party == kSender): `corr` is the chosen correlation (length
    // `n`). On return `data0[i]` holds the random message x_i (mod 2^l).
    virtual OTStatus send_cot(uint64_t* data0, const uint64_t* corr, int n,
                              int l) = 0;
    // Receiver side (party == kReceiver): `b` are the choice bits (length n).
    // On return `data[i]` = x_i + (b_i ? corr_i : 0) (mod 2^l).
    virtual OTStatus recv_cot(uint64_t* data, const bool* b, int n, int l) = 0;

    // --- COT over Z_p (field modulus, prime p) -----------------------------
    // Same COT semantics but mod p instead of mod 2^l. This is the "cot_prime"
    // path M&M uses in field-domain conversion (Truncation line ~549).
    virtual OTStatus send_cot_prime(uint64_t* data0, const uint64_t* corr,
                                    int n, uint64_t p) = 0;
    virtual OTStatus recv_cot_prime(uint64_t* data, const bool* b, int n,
                                    uint64_t p) = 0;
};

}  // namespace gpu_mm

#endif  // GPU_MM_OT_PROVIDER_H
