// WrapBitOr — shared logical-OR of two parties' bits -> arithmetic wrap share,
// backed by gpu_mm::OTProvider (Task M3-T2).
//
// Goal (CLAUDE.md dev order, step 7: "replace one primitive behind a flag"):
// lift the wrap-bit / logical-OR primitive off SCI's `otpack->iknp_straight`
// COT and onto the gpu_mm::OTProvider seam (M1-T2), so a cuOT backend can later
// plug in behind `--ot-backend` (M3-T3, gated on M2-T2). CPU emp default; no
// SCI path is touched (the conversion equations in aux-protocols.cpp /
// truncation.cpp stay byte-identical — see docs/gpu_mm/ot_callsite_map.md §2.1).
//
// This mirrors Truncation::OR_then_B2A (SCI/src/BuildingBlocks/truncation.cpp
// :344-360) — the canonical "wrap = (msb0 || msb1)" path that ring_to_ring /
// ring_to_ring128 call. The mapping onto OTProvider is 1:1:
//
//   OTProvider COT semantics (ot_provider.h:11-21): sender picks `corr`, gets
//   random x=data0; receiver with bit b gets x (b=0) or x+corr (b=1) mod 2^l.
//
//   OR derivation (truncation.cpp:346-358), per-party bit x_i ∈ {0,1}:
//     ALICE (sender):  corr[i] = x_A[i] ^ 1
//                      send_cot(y_A, corr, n, l)
//                      y_A[i] = (x_A[i] - y_A[i]) & mask      // local adjust
//     BOB   (receiver): recv_cot(y_B, (bool*)x_B, n, l)
//                      y_B holds the receiver's share (no local adjust)
//
//   Reconstruction: y_A + y_B ≡ OR(x_A, x_B) (mod 2^l).
//   Proof sketch: COT gives y_A = x (=data0) and y_B ∈ {x, x+corr} depending on
//   b=x_B. After ALICE's adjust y_A' = x_A - x; BOB holds y_B = x + x_B·corr =
//   x + x_B·(x_A^1). Sum = x_A - x + x + x_B·(x_A^1) = x_A + x_B·(x_A^1) =
//   x_A + x_B·(1-x_A) = x_A + x_B - x_A·x_B = OR(x_A,x_B). ∎
//
// The recv side takes `const bool*`; `local_bit` here is `const uint8_t*` (one
// byte per bit, value exactly 0 or 1). The reinterpret_cast is safe ONLY under
// that invariant, which this primitive enforces (callers pass 0/1 bit vectors).
#ifndef GPU_MM_WRAP_BIT_OR_H
#define GPU_MM_WRAP_BIT_OR_H

#include "gpu_mm/ot_provider.h"

#include <cstdint>

namespace gpu_mm {

// WrapBitOr does NOT own the OTProvider; the caller owns it (e.g. an
// EmpOTProvider or, later, a CuotProvider). One WrapBitOr instance serves as
// ONE party over ONE provider's channel — two-party protocols construct two
// instances (one sender, one receiver) backed by a connected provider pair.
class WrapBitOr {
   public:
    WrapBitOr(OTProvider& prov, OTParty role) : prov_(prov), role_(role) {}

    // Compute arithmetic shares of OR(local_bit_A, local_bit_B) mod 2^l.
    //   local_bit : length-n vector, each element 0 or 1 (this party's bit).
    //   out       : length-n uint64, receives this party's arithmetic share.
    //   l         : ring bitwidth, 1..64. M = 2^l is the modulus.
    // Returns kOk on success; the provider's own status on failure.
    // After both parties return kOk:
    //   (out_A[i] + out_B[i]) & mask == (local_bit_A[i] | local_bit_B[i]).
    OTStatus or_share(const uint8_t* local_bit, uint64_t* out, int32_t n,
                      int32_t l) {
        if (local_bit == nullptr || out == nullptr || n <= 0 || l <= 0 ||
            l > 64) {
            return OTStatus::kInvalidArg;
        }
        const uint64_t mask = (l == 64) ? ~0ULL : ((1ULL << l) - 1);

        if (role_ == OTParty::kSender) {
            // corr[i] = x_A[i] ^ 1  (truncation.cpp:349). Heap-alloc because n
            // may be large (100k in the test); a fixed stack buffer would not
            // scale. Freed before return.
            uint64_t* corr = new uint64_t[n];
            for (int32_t i = 0; i < n; ++i) corr[i] = local_bit[i] ? 0ULL : 1ULL;
            OTStatus s = prov_.send_cot(out, corr, n, l);
            if (s != OTStatus::kOk) {
                delete[] corr;
                return s;
            }
            // y_A[i] = (x_A[i] - y_A[i]) & mask  (truncation.cpp:354)
            for (int32_t i = 0; i < n; ++i)
                out[i] = ((uint64_t)local_bit[i] - out[i]) & mask;
            delete[] corr;
        } else {  // OTParty::kReceiver
            // recv_cot(y_B, (bool*)x_B, n, l)  (truncation.cpp:358). y_B already
            // holds the receiver's OR share; no local adjust.
            OTStatus s = prov_.recv_cot(
                out, reinterpret_cast<const bool*>(local_bit), n, l);
            if (s != OTStatus::kOk) return s;
        }
        return OTStatus::kOk;
    }

   private:
    OTProvider& prov_;
    OTParty role_;
};

}  // namespace gpu_mm

#endif  // GPU_MM_WRAP_BIT_OR_H
