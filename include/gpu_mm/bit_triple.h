// BitTriple — cuOT-backed Beaver bit-triple generation via _2ROT
// (Task M3-T6 Phase B).
//
// Generates a RANDOM bit-AND triple (a, b, c) with c = a & b, shared across
// parties, mirroring SCI's TripleGenerator::generate(_2ROT)
// (SCI/src/Millionaire/bit-triple-generator.h:166-217). That is the triple the
// Millionaire AND-tree (millionaire.h traverse_and_compute_ANDs) consumes to
// fold per-digit leaf compare bits into a full multi-digit compare result.
//
// IMPORTANT — a, b, c are all OUTPUTS (a RANDOM triple), not party-chosen
// inputs. SCI's _2ROT derives a and b from the ROTs themselves (the receiver's
// random choice bit becomes a; the sender's m1 becomes b). Neither party
// "picks" the AND inputs; the triple is random masking material.
//
// THE _2ROT ALGORITHM (bit-triple-generator.h:184-199), two 1-2 ROTs in
// opposite directions:
//   ROT#1 (BOB sender, ALICE receiver): ALICE recv with random choice a_rot1
//     -> ALICE gets u = m_{a_rot1}, and a = a_rot1 (the random choice bit).
//        BOB send -> BOB gets v = m0_rot1, b = m1_rot1.
//   ROT#2 (ALICE sender, BOB receiver): BOB recv with random choice a_rot2
//     -> BOB gets u = m_{a_rot2}, and a = a_rot2. ALICE send -> ALICE gets
//        v = m0_rot2, b = m1_rot2.
//   Finalize (BOTH parties, identical): b ^= v; c = (a & b) ^ u ^ v.
//   Reconstruction: a_A^a_B = a, b_A^b_B = b, c_A^c_B = a&b. (Verified: the
//   ROT property m_{r} = m0 ^ r*(m1^m0) makes c_A^c_B collapse to a*b.)
//
// cuOT REALIZATION:
//   - ROT = CuotProvider::send_rot_blocks / recv_rot_blocks (added for Phase
//     B; re-derives emp COT<T>::send_rot/recv_rot on send_cot_blocks/
//     recv_cot_blocks + MITCCRH decorrelation, since cuOT's inherited send_rot
//     calls the no-op send_cot(block*)).
//   - send_rot_blocks needs the Ferret SENDER (owns Delta, for m1=H(x^Delta));
//     recv_rot_blocks needs the Ferret RECEIVER. A single Ferret fixes one
//     role, so each party needs TWO CuotProviders:
//       prov_send (OTParty::kSender, Ferret-ALICE/Delta, NetIO server)
//       prov_recv (OTParty::kReceiver, Ferret-BOB, NetIO client)
//     On port P1: BOB.prov_send (server) <-> ALICE.prov_recv (client) = ROT#1.
//     On port P2: ALICE.prov_send (server) <-> BOB.prov_recv (client) = ROT#2.
//   - Each Ferret needs a DISTINCT pre_file (4 total) to avoid cache
//     cross-contamination + the recycle bug (ferret_cot.hpp:99). The run
//     script `rm -f`s all pre_ot_data_reg_* before each run.
//
// CAVEAT (docs/research/m2-cuot-standalone.md §0,§7.4): cuOT's block-RCOT has
// a STABLE ~2% error floor (M2-T2 SHELVED). A triple uses TWO ROTs, so the
// triple error rate compounds to ~4%. DIAGNOSTIC only; not correctness-clear.
// Per user direction (2026-06-28, "继续堆 Phase B/C"), this is the intended
// measurement. The ROT primitive itself is validated first by
// cuot_rot_smoke_test (send_rot_blocks/recv_rot_blocks in isolation) before
// this triple composes on top of it.
#ifndef GPU_MM_BIT_TRIPLE_H
#define GPU_MM_BIT_TRIPLE_H

#include "gpu_mm/cuot_provider.h"   // CuotProvider + send/recv_rot_blocks
#include "gpu_mm/ot_provider.h"

#include <emp-tool/emp-tool.h>      // emp::block, getLSB

#include <cstdint>
#include <cstring>
#include <vector>

namespace gpu_mm {

namespace {
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
};
}  // namespace

// BitTripleGen owns TWO CuotProviders (prov_send = Ferret-SENDER, prov_recv =
// Ferret-RECEIVER) for the two opposite-direction ROTs. The caller constructs
// one BitTripleGen per party; the two parties' provider pairs must be connected
// on two distinct ports. role = this party's Beaver role (kSender=ALICE,
// kReceiver=BOB).
class BitTripleGen {
   public:
    BitTripleGen(CuotProvider& prov_send, CuotProvider& prov_recv,
                 OTParty role)
        : prov_send_(prov_send), prov_recv_(prov_recv), role_(role) {}

    // Generate `n` RANDOM bit-AND triples. a, b, c are OUTPUTS (length n,
    // 1 byte each). After both parties return kOk:
    //   (a_A[i]^a_B[i]) & (b_A[i]^b_B[i]) == c_A[i]^c_B[i].
    OTStatus generate(uint8_t* a, uint8_t* b, uint8_t* c, int32_t n) {
        if (a == nullptr || b == nullptr || c == nullptr || n <= 0) {
            return OTStatus::kInvalidArg;
        }
        // u, v: the ROT receiver output (u) and the ROT sender's m0 (v), 1 bit.
        std::vector<uint8_t> u((size_t)n), v((size_t)n);

        // Block buffers. send_rot_blocks writes n*2 blocks (m0,m1 interleave);
        // recv_rot_blocks writes n blocks (m_{r}).
        std::vector<emp::block> send_buf((size_t)n * 2);
        std::vector<emp::block> recv_buf((size_t)n);

        // Random choice bits for THIS party's recv ROT (rm_rc = random choice).
        // ALICE's recv is ROT#1 (choice a_rot1); BOB's recv is ROT#2 (a_rot2).
        std::vector<uint8_t> rbuf((size_t)n);
        Rng rng(role_ == OTParty::kSender ? 0x9e3779b97f4a7c15ULL
                                          : 0xd1b54a32d192ed03ULL);
        for (int i = 0; i < n; ++i) rbuf[i] = (uint8_t)(rng.next() & 1);
        const bool* r = reinterpret_cast<const bool*>(rbuf.data());

        // Order: ROT#1 first (BOB send + ALICE recv), then ROT#2 (ALICE send +
        // BOB recv). ALICE does recv#1 then send#2; BOB does send#1 then
        // recv#2. The send_cot_blocks diff-vector handshake (recv sends diff,
        // sender recvs) is matched on each port; no inversion deadlock.
        if (role_ == OTParty::kSender) {  // ALICE
            // ROT#1: ALICE recv (prov_recv_), choice = a_rot1.
            OTStatus s = prov_recv_.recv_rot_blocks(
                reinterpret_cast<uint8_t*>(recv_buf.data()), r, n);
            if (s != OTStatus::kOk) return s;
            for (int i = 0; i < n; ++i) {
                a[i] = rbuf[i];  // a-share = the random choice bit
                u[i] = (uint8_t)emp::getLSB(recv_buf[i]);  // m_{a_rot1}
            }
            // ROT#2: ALICE send (prov_send_).
            s = prov_send_.send_rot_blocks(
                reinterpret_cast<uint8_t*>(send_buf.data()), n);
            if (s != OTStatus::kOk) return s;
            for (int i = 0; i < n; ++i) {
                b[i] = (uint8_t)emp::getLSB(send_buf[2 * i + 1]);  // m1_rot2
                v[i] = (uint8_t)emp::getLSB(send_buf[2 * i]);      // m0_rot2
            }
        } else {  // BOB
            // ROT#1: BOB send (prov_send_).
            OTStatus s = prov_send_.send_rot_blocks(
                reinterpret_cast<uint8_t*>(send_buf.data()), n);
            if (s != OTStatus::kOk) return s;
            for (int i = 0; i < n; ++i) {
                b[i] = (uint8_t)emp::getLSB(send_buf[2 * i + 1]);  // m1_rot1
                v[i] = (uint8_t)emp::getLSB(send_buf[2 * i]);      // m0_rot1
            }
            // ROT#2: BOB recv (prov_recv_), choice = a_rot2.
            s = prov_recv_.recv_rot_blocks(
                reinterpret_cast<uint8_t*>(recv_buf.data()), r, n);
            if (s != OTStatus::kOk) return s;
            for (int i = 0; i < n; ++i) {
                a[i] = rbuf[i];  // a-share = the random choice bit
                u[i] = (uint8_t)emp::getLSB(recv_buf[i]);  // m_{a_rot2}
            }
        }

        // Finalize (bit-triple-generator.h:196-199), SAME on both parties.
        for (int i = 0; i < n; ++i) b[i] ^= v[i];
        for (int i = 0; i < n; ++i) c[i] = (uint8_t)((a[i] & b[i]) ^ u[i] ^ v[i]);
        return OTStatus::kOk;
    }

   private:
    CuotProvider& prov_send_;
    CuotProvider& prov_recv_;
    OTParty role_;
};

}  // namespace gpu_mm

#endif  // GPU_MM_BIT_TRIPLE_H
