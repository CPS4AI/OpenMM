// FieldToRing — field→ring conversion primitive on gpu_mm::OTProvider
// (Task M3-T5).
//
// Goal (CLAUDE.md dev order, step 7: "replace one primitive behind a flag"):
// lift the field→ring modulus conversion's OT-backed OR_AUX step off SCI's
// `otpack->iknp_straight` COT-l path and onto the gpu_mm::OTProvider seam
// (M1-T2), so a cuOT backend can later plug in behind `--ot-backend` (M3-T5).
// CPU emp default; no SCI path is touched — the conversion equations in
// truncation.cpp stay byte-identical (see docs/gpu_mm/ot_callsite_map.md §2.2).
//
// This mirrors the OR_AUX lambda inside Truncation::field_to_ring
// (SCI/src/BuildingBlocks/truncation.cpp:600-616) — and the identical lambda
// in field_to_ring_with_truncate (truncation.cpp:658-674, only the corr value
// `p>>shift_bw` differs, passed in by the caller). It is the COT-l callsite
// the map identifies (§2.2). The substitution: OTProvider::send_cot/recv_cot
// for otpack->iknp_straight->send_cot/recv_cot.
//
//   OTProvider COT-l semantics (ot_provider.h:56-63): sender picks `corr`,
//   gets random x=data0; receiver with bit b gets x (b=0) or x+corr (b=1)
//   mod 2^bw_y.
//
//   field_to_ring OR_AUX equation (truncation.cpp:602-615), per-party field
//   share, field modulus p = field_mod, ring bitwidth bw_y = ring_bw:
//     ALICE: corr_data[i] = msb[i] ? 0 : p
//            send_cot(y_A, corr_data, dim, bw_y)
//            y_A[i] = msb[i] ? ((p - y_A[i]) & mask) : (-y_A[i] & mask)
//     BOB:   recv_cot(y_B, (bool*)msb, dim, bw_y)
//            // y_B holds BOB's ring share; no local adjust
//
//   Reconstruction (the wrap term, `// wrap = (msb0 || msb1)`):
//     (y_A[i] + y_B[i]) & mask == (msb_A[i] | msb_B[i]) * p  & mask
//   Proof: send_cot gives sender x; recv_cot gives receiver x + msb_B·corr_i
//   where corr_i = msb_A ? 0 : p. After ALICE's adjust y_A = msb_A ? (p-x) :
//   (-x), all mod 2^bw. Sum y_A+y_B mod 2^bw:
//     (0,0): -x + x         = 0
//     (0,1): -x + (x+p)     = p
//     (1,0): (p-x) + x      = p
//     (1,1): (p-x) + x      = p      (corr_i=0 when msb_A=1, so recv gets x)
//   = (msb_A | msb_B) · p. ∎
//   The caller folds y into the final ring share via truncation.cpp:626-633
//   (outB = tmpA - y - big_number) — not OT, stays in the caller.
//
// This primitive owns ONLY the OT-backed OR_AUX step (corr computation +
// send/recv_cot + the ALICE local adjust). The big-number add, msb (>=half)
// extraction, and final subtract are the caller's job — they are NOT OT and
// stay in the caller (mirroring how truncation.cpp wraps OR_AUX). Exposing
// just the OT-backed OR_AUX lets M3-T5 target the exact OTProvider seam the
// map identifies, without re-implementing the surrounding arithmetic.
#ifndef GPU_MM_FIELD_TO_RING_H
#define GPU_MM_FIELD_TO_RING_H

#include "gpu_mm/ot_provider.h"

#include <cstdint>

namespace gpu_mm {

// FieldToRing does NOT own the OTProvider; the caller owns it (EmpOTProvider
// or, behind the flag, CuotProvider). One instance = ONE party over ONE
// provider channel; two-party protocols construct two instances.
class FieldToRing {
   public:
    FieldToRing(OTProvider& prov, OTParty role) : prov_(prov), role_(role) {}

    // OT-backed OR_AUX step of field_to_ring (truncation.cpp:600-616).
    // Computes each party's ring-domain share of the wrap term
    //   wrap = (msb_A || msb_B) * p,  reduced mod 2^bw_y.
    //
    //   msb  : length-dim vector, each element 0 or 1 (this party's MSB bit
    //          of tmpA, i.e. tmpA >= half_field).
    //   out  : length-dim uint64, receives this party's ring share of the
    //          OR_AUX result (mod 2^bw_y).
    //   p    : field modulus (prime) — ALSO the COT correlation value here
    //          (truncation.cpp:605 corr_data[i] = msb[i] ? 0 : p).
    //   bw_y : ring bitwidth (ring_bw). M = 2^bw_y is the modulus.
    // Returns kOk on success; the provider's own status on failure.
    // After both parties return kOk:
    //   (out_A[i] + out_B[i]) & mask == ((msb_A[i] | msb_B[i]) * p) & mask
    // i.e. the shared OR (0 or p) in the ring. The caller then folds out
    // into the final ring share via the truncation.cpp:626-633 subtract.
    //
    // NOTE: field_to_ring_with_truncate uses the SAME lambda with corr =
    // p>>shift_bw instead of p; the caller can implement that variant by
    // passing a pre-shifted p — but the SCI lambda hardcodes `p` as both the
    // field modulus AND the corr, so for the truncate variant the caller
    // must pass p>>shift_bw here AND adjust its own final subtract. This
    // primitive matches the non-truncate field_to_ring lambda exactly.
    OTStatus or_aux(const uint8_t* msb, uint64_t* out, int32_t dim,
                    uint64_t p, int32_t bw_y) {
        if (msb == nullptr || out == nullptr || dim <= 0 || bw_y <= 0 ||
            bw_y > 64 || p < 2) {
            return OTStatus::kInvalidArg;
        }
        const uint64_t mask = (bw_y == 64) ? ~0ULL : ((1ULL << bw_y) - 1);
        if (role_ == OTParty::kSender) {
            // corr_data[i] = msb[i] ? 0 : p  (truncation.cpp:605)
            uint64_t* corr_data = new uint64_t[dim];
            for (int32_t i = 0; i < dim; ++i)
                corr_data[i] = msb[i] ? 0ULL : p;
            OTStatus s = prov_.send_cot(out, corr_data, dim, bw_y);
            if (s != OTStatus::kOk) { delete[] corr_data; return s; }
            // y_A[i] = msb[i] ? ((p - y_A[i]) & mask) : (-y_A[i] & mask)
            // (truncation.cpp:610). EmpOTProvider / CuotProvider flush
            // internally on send_cot, so no explicit flush here (matches
            // ring_to_field.h / wrap_bit_or.h).
            for (int32_t i = 0; i < dim; ++i) {
                out[i] = msb[i] ? ((p - out[i]) & mask)
                                : ((0ULL - out[i]) & mask);
            }
            delete[] corr_data;
        } else {  // OTParty::kReceiver
            // recv_cot(out, (bool*)msb, dim, bw_y)  (truncation.cpp:614).
            // out holds the receiver's ring share; no local adjust.
            OTStatus s = prov_.recv_cot(
                out, reinterpret_cast<const bool*>(msb), dim, bw_y);
            if (s != OTStatus::kOk) return s;
        }
        return OTStatus::kOk;
    }

   private:
    OTProvider& prov_;
    OTParty role_;
};

}  // namespace gpu_mm

#endif  // GPU_MM_FIELD_TO_RING_H
