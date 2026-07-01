// Field→ring conversion primitive test (Task M3-T5).
//
// Two-party. Verifies FieldToRing::or_aux (on gpu_mm::OTProvider, CPU
// EmpOTProvider default) computes the correct shared ring-domain OR_AUX term:
//   (out_A[i] + out_B[i]) & mask == ((msb_A[i] | msb_B[i]) * p) & mask
// for ring_bw = bw_y = 32, p = field_mod = 4293918721 (the 32-bit prime M&M
// uses).
//
// This isolates the OT-backed OR_AUX step (truncation.cpp:600-616) — the COT-l
// callsite the M3-T1 map identifies (§2.2). It does NOT exercise the
// surrounding big-number add / half-field MSB / final subtract (not OT; stay
// in the caller). "Reconstruction correctness" = the OR_AUX reconstruction
// above. "Same communication count shape" is structural: OR_AUX issues exactly
// one send_cot / recv_cot of length `dim` (one COT-l batch), same as SCI's
// iknp_straight->send_cot.
//
// Coverage (per Task M3-T5 acceptance — reused as the CPU baseline so the cuOT
// variant runs the same cases):
//   - boundary MSB pairs covering all four OR outcomes:
//       (0,0)->0, (1,0)->p, (0,1)->p, (1,1)->p   (all mod 2^bw)
//   - random 100000 samples (M3 "random stress"; 100k per the M3 bar).
//
// Verification channel: the sender sends its `out_A` share + its `msb_A` bits
// back to the receiver over the provider's NetIO (both secret in real use;
// exposed here only for the test, exactly like ring_to_field_or_aux_test.cpp).
//
// Usage (two-party, run from repo root so ./data exists for Ferret; bind
// needs sudo on this box — see memory openmm-build-env §4):
//   sudo ./build/bin/field_to_ring_or_aux_test 1 <port> & \
//   sudo ./build/bin/field_to_ring_or_aux_test 2 <port>
#include "gpu_mm/emp_ot_provider.h"
#include "gpu_mm/field_to_ring.h"
#include "gpu_mm/ot_provider.h"
#include "utils/net_io_channel.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
};

constexpr int32_t bw_y = 32;
const uint64_t mask = (bw_y == 64) ? ~0ULL : ((1ULL << bw_y) - 1);
const uint64_t p = 4293918721ULL;  // field modulus AND COT correlation (truncation.cpp:605)

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

// ---- sender (ALICE) ----
int run_sender(int port) {
    using namespace gpu_mm;
    EmpOTProvider prov(OTParty::kSender, port);
    sci::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "sender net_io null");
    FieldToRing conv(prov, OTParty::kSender);

    const uint8_t msbA[] = {0, 1, 0, 1};
    const int32_t nb = (int32_t)(sizeof(msbA) / sizeof(msbA[0]));
    std::vector<uint64_t> outA(nb);
    EXPECT(conv.or_aux(msbA, outA.data(), nb, p, bw_y) == OTStatus::kOk,
           "or_aux boundary");
    io->send_data(outA.data(), nb * sizeof(uint64_t));
    io->send_data(msbA, nb);
    io->flush();

    const int32_t n = 100000;
    std::vector<uint8_t> msbsA(n);
    Rng rng(0x9e3779b97f4a7c15ULL);
    for (int32_t i = 0; i < n; ++i) msbsA[i] = (uint8_t)(rng.next() & 1);
    std::vector<uint64_t> outA_r(n);
    EXPECT(conv.or_aux(msbsA.data(), outA_r.data(), n, p, bw_y) == OTStatus::kOk,
           "or_aux random");
    io->send_data(outA_r.data(), n * sizeof(uint64_t));
    io->send_data(msbsA.data(), n);
    io->flush();

    std::cout << "[sender] field_to_ring OR_AUX sent: boundary=" << nb
              << " random=" << n << " p=" << p << " bw=" << bw_y << std::endl;
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port) {
    using namespace gpu_mm;
    EmpOTProvider prov(OTParty::kReceiver, port);
    sci::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");
    FieldToRing conv(prov, OTParty::kReceiver);

    const uint8_t msbB[] = {0, 0, 1, 1};
    const int32_t nb = (int32_t)(sizeof(msbB) / sizeof(msbB[0]));
    std::vector<uint64_t> outB(nb);
    EXPECT(conv.or_aux(msbB, outB.data(), nb, p, bw_y) == OTStatus::kOk,
           "or_aux boundary recv");

    std::vector<uint64_t> outA(nb);
    std::vector<uint8_t> msbA(nb);
    io->recv_data(outA.data(), nb * sizeof(uint64_t));
    io->recv_data(msbA.data(), nb);

    int fails = 0;
    for (int32_t i = 0; i < nb; ++i) {
        uint64_t got = (outA[i] + outB[i]) & mask;
        uint64_t want = ((uint64_t)(msbA[i] | msbB[i]) * p) & mask;
        if (got != want) {
            std::cerr << "FAIL boundary @ " << i << " A=" << (int)msbA[i]
                      << " B=" << (int)msbB[i] << " got=" << got
                      << " want=" << want << "\n";
            ++fails;
        }
    }

    const int32_t n = 100000;
    std::vector<uint8_t> msbsB(n);
    Rng rng(0xd1b54a32d192ed03ULL);
    for (int32_t i = 0; i < n; ++i) msbsB[i] = (uint8_t)(rng.next() & 1);
    std::vector<uint64_t> outB_r(n);
    EXPECT(conv.or_aux(msbsB.data(), outB_r.data(), n, p, bw_y) == OTStatus::kOk,
           "or_aux random recv");

    std::vector<uint64_t> outA_r(n);
    std::vector<uint8_t> msbsA(n);
    io->recv_data(outA_r.data(), n * sizeof(uint64_t));
    io->recv_data(msbsA.data(), n);

    for (int32_t i = 0; i < n; ++i) {
        uint64_t got = (outA_r[i] + outB_r[i]) & mask;
        uint64_t want = ((uint64_t)(msbsA[i] | msbsB[i]) * p) & mask;
        if (got != want) {
            if (fails < 5)
                std::cerr << "FAIL random @ " << i << " A=" << (int)msbsA[i]
                          << " B=" << (int)msbsB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++fails;
        }
    }

    if (fails != 0) {
        std::cerr << "field_to_ring OR_AUX test FAILED (" << fails
                  << " mismatches)\n";
        return 1;
    }
    std::cout << "[receiver] field_to_ring OR_AUX OK: boundary=" << nb
              << " random=" << n << " p=" << p << " bw=" << bw_y << std::endl;
    std::cout << "field_to_ring OR_AUX test PASSED" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: field_to_ring_or_aux_test <party 1|2> <port>\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./build/bin/field_to_ring_or_aux_test 1 <port> &\n"
                  << "  sudo ./build/bin/field_to_ring_or_aux_test 2 <port>\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    if (party == 1) return run_sender(port);
    if (party == 2) return run_receiver(port);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
