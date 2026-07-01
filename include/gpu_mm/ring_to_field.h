// RingToField — ring→field conversion primitive on gpu_mm::OTProvider
// (Task M3-T4).
//
// Goal (CLAUDE.md dev order, step 7: "replace one primitive behind a flag"):
// lift the ring→field modulus conversion off SCI's `otpack->iknp_straight`
// COT-p path and onto the gpu_mm::OTProvider seam (M1-T2), so a cuOT backend
// can later plug in behind `--ot-backend` (M3-T4). CPU emp default; no SCI path
// is touched — the conversion equations in truncation.cpp stay byte-identical
// (see docs/gpu_mm/ot_callsite_map.md §2.2).
//
// This mirrors Truncation::ring_to_field (SCI/src/BuildingBlocks/truncation.cpp
// :518-581) — the ONLY COT-p callsite in M&M — substituting OTProvider for
// otpack->iknp_straight. The COT-p shape:
//
//   OTProvider COT-p semantics (ot_provider.h:65-71): sender picks `corr`,
//   gets random x=data0; receiver with bit b gets x (b=0) or x+corr (b=1)
//   mod p.
//
//   ring_to_field equation (truncation.cpp:543-559), per-party ring share x_i
//   over Z_{2^ring_bw}, field modulus p = field_mod:
//     ALICE: corr = (ring_mod_mask % p + 1)        // the "wrap error"
//            corr_data[i] = msb[i] ? 0 : corr
//            send_cot_prime(y_A, corr_data, dim, p)
//            tmp   = (0 - y_A[i]) mod p            // secure_sub(0, y, p)
//            y_A[i] = msb[i] ? (corr + tmp) mod p : tmp
//     BOB:   recv_cot_prime(y_B, (bool*)msb, dim, p)
//            // y_B holds BOB's field share; no local adjust
//
//   Final field shares (truncation.cpp:569-577), after the big-number offset:
//     ALICE: outB[i] = (tmpA[i] - y_A[i] - big_number%p) mod p
//     BOB:   outB[i] = (tmpA[i] - y_B[i]) mod p
//   where tmpA = inA + big_number (ALICE) / inA (BOB), big_number=2^(ring_bw-2).
//   Reconstruction: outB_A + outB_B ≡ inA (mod p)   [the ring value in the field].
//
// This primitive owns ONLY the OR_AUX COT-p step (corr computation +
// send/recv_cot_prime + the ALICE local adjust). The big-number add, msb
// extraction, and final subtract are the caller's job — they are NOT OT and
// stay in the caller (mirroring how truncation.cpp wraps OR_AUX). Exposing
// just the OT-backed OR_AUX lets M3-T4 target the exact OTProvider seam the
// map identifies (§2.2), without re-implementing the surrounding arithmetic.
//
// secure_add / secure_sub are field-modulus helpers copied verbatim from
// truncation.cpp:533-541 so the ALICE local adjust matches byte-for-byte
// (overflow-safe mod-p add/sub without 128-bit intermediates).
#ifndef GPU_MM_RING_TO_FIELD_H
#define GPU_MM_RING_TO_FIELD_H

#include "gpu_mm/ot_provider.h"

#include <cstdint>

namespace gpu_mm {

namespace {
// Overflow-safe mod-p add/sub, copied verbatim from
// Truncation::ring_to_field (truncation.cpp:533-541). Avoids uint128; matches
// the CPU baseline's arithmetic exactly so ALICE's local adjust is identical.
inline uint64_t secure_add(uint64_t x, uint64_t y, uint64_t p) {
    if (x >= (p - y)) return x - (p - y);
    else return x + y;
}
inline uint64_t secure_sub(uint64_t x, uint64_t y, uint64_t p) {
    if (x >= y) return x - y;
    else return x + (p - y);
}
}  // namespace

// RingToField does NOT own the OTProvider; the caller owns it (EmpOTProvider
// or, behind the flag, CuotProvider). One instance = ONE party over ONE
// provider channel; two-party protocols construct two instances.
class RingToField {
   public:
    RingToField(OTProvider& prov, OTParty role) : prov_(prov), role_(role) {}

    // OT-backed OR_AUX step of ring_to_field (truncation.cpp:543-559).
    // Computes each party's field-domain share of the shared MSB-OR term
    //   wrap = (msb_A || msb_B),  reduced mod p.
    //
    //   msb    : length-dim vector, each element 0 or 1 (this party's MSB bit
    //            of tmpA = inA + big_number).
    //   out    : length-dim uint64, receives this party's field share of the
    //            OR_AUX result (mod p).
    //   corr   : the wrap-error correlation = ring_mod_mask % p + 1
    //            (caller computes it; matches truncation.cpp:566).
    //   p      : field modulus (prime).
    // Returns kOk on success; the provider's own status on failure.
    // After both parties return kOk:
    //   (out_A[i] + out_B[i]) mod p == (msb_A[i] | msb_B[i]) * corr  mod p
    // i.e. the shared OR (0 or corr) in the field. The caller then folds
    // out into the final field share via the truncation.cpp:569-577 subtract.
    OTStatus or_aux(const uint8_t* msb, uint64_t* out, int32_t dim,
                    uint64_t corr, uint64_t p) {
        if (msb == nullptr || out == nullptr || dim <= 0 || p < 2) {
            return OTStatus::kInvalidArg;
        }
        if (role_ == OTParty::kSender) {
            // corr_data[i] = msb[i] ? 0 : corr  (truncation.cpp:547)
            uint64_t* corr_data = new uint64_t[dim];
            for (int32_t i = 0; i < dim; ++i)
                corr_data[i] = msb[i] ? 0ULL : corr;
            OTStatus s = prov_.send_cot_prime(out, corr_data, dim, p);
            if (s != OTStatus::kOk) { delete[] corr_data; return s; }
            // tmp = (0 - out[i]) mod p; out[i] = msb[i] ? (corr+tmp)%p : tmp
            // (truncation.cpp:552-553). NOTE: EmpOTProvider / CuotProvider
            // flush internally on send_cot_prime, so no explicit flush here
            // (matches wrap_bit_or.h not calling otpack->io->flush()).
            for (int32_t i = 0; i < dim; ++i) {
                uint64_t tmp = secure_sub(0, out[i], p);
                out[i] = msb[i] ? secure_add(corr, tmp, p) : tmp;
            }
            delete[] corr_data;
        } else {  // OTParty::kReceiver
            // recv_cot_prime(out, (bool*)msb, dim, p)  (truncation.cpp:557).
            // out holds the receiver's field share; no local adjust.
            OTStatus s = prov_.recv_cot_prime(
                out, reinterpret_cast<const bool*>(msb), dim, p);
            if (s != OTStatus::kOk) return s;
        }
        return OTStatus::kOk;
    }

   private:
    OTProvider& prov_;
    OTParty role_;
};

}  // namespace gpu_mm

#endif  // GPU_MM_RING_TO_FIELD_H
