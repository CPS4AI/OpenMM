// Ring→field conversion primitive test (Task M3-T4).
//
// Two-party. Verifies RingToField::or_aux (on gpu_mm::OTProvider, CPU
// EmpOTProvider default) computes the correct shared field-domain OR_AUX term:
//   (out_A[i] + out_B[i]) mod p == (msb_A[i] | msb_B[i]) * corr  mod p
// for ring_bw=32, field_mod = p = 4293918721 (the 32-bit prime M&M uses).
//
// This isolates the OT-backed OR_AUX step (truncation.cpp:543-559) — the ONLY
// COT-p callsite in M&M — exactly as the M3-T1 map (§2.2) scopes it. It does
// NOT exercise the surrounding big-number add / final subtract (those are not
// OT and stay in the caller); the acceptance criterion "reconstruction
// correctness" is therefore the OR_AUX reconstruction above. "Same
// communication count shape" is satisfied structurally: OR_AUX issues exactly
// one send_cot_prime / recv_cot_prime of length `dim` (one COT-p batch), same
// as SCI's iknp_straight->send_cot_prime — no extra rounds.
//
// Coverage (per Task M3-T4 acceptance — reused as the CPU baseline so the cuOT
// variant in cuot_ring_to_field_test.cpp runs the same cases):
//   - boundary MSB pairs covering all four OR outcomes:
//       (0,0)->0, (1,0)->corr, (0,1)->corr, (1,1)->corr
//   - random 100000 samples (M3 "random stress"; 100k per the M3 acceptance
//     bar). MSB bits drawn independently per party from a fixed xorshift seed.
//
// Verification channel: the sender sends its `out_A` share + its `msb_A` bits
// back to the receiver over the provider's NetIO (both secret in real use;
// exposed here only for the test, exactly like wrap_bit_or_test.cpp:80).
//
// Usage (two-party, run from repo root so ./data exists for Ferret; bind
// needs sudo on this box — see memory openmm-build-env §4):
//   sudo ./build/bin/ring_to_field_test 1 <port> & \
//   sudo ./build/bin/ring_to_field_test 2 <port>
//   party 1 = sender (ALICE), party 2 = receiver (BOB).
#include "gpu_mm/emp_ot_provider.h"
#include "gpu_mm/ot_provider.h"
#include "gpu_mm/ring_to_field.h"
#include "utils/net_io_channel.h"  // sci::NetIO send/recv for verification

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

// xorshift PRNG (deterministic, no Math.random). Mirrors
// share_tensor_backend_test.cpp's Rng so both parties generate reproducible
// MSB bits and the receiver can predict the sender's bits for the check.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
};

constexpr int32_t ring_bw = 32;
const uint64_t ring_mod_mask = (ring_bw == 64) ? ~0ULL : ((1ULL << ring_bw) - 1);
const uint64_t p = 4293918721ULL;  // 32-bit prime (matches ot_provider_smoke_test)
// wrap-error correlation = ring_mod_mask % p + 1  (truncation.cpp:566)
const uint64_t corr = (ring_mod_mask % p) + 1;

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
    RingToField conv(prov, OTParty::kSender);

    // --- boundary block: msb_A = {0,1,0,1} (covers all OR outcomes w/ B) ---
    const uint8_t msbA[] = {0, 1, 0, 1};
    const int32_t nb = (int32_t)(sizeof(msbA) / sizeof(msbA[0]));
    std::vector<uint64_t> outA(nb);
    EXPECT(conv.or_aux(msbA, outA.data(), nb, corr, p) == OTStatus::kOk,
           "or_aux boundary");
    io->send_data(outA.data(), nb * sizeof(uint64_t));
    io->send_data(msbA, nb);  // A's MSB bits (so receiver knows the pair)
    io->flush();

    // --- random 100k block ---
    const int32_t n = 100000;
    std::vector<uint8_t> msbsA(n);
    Rng rng(0x9e3779b97f4a7c15ULL);
    for (int32_t i = 0; i < n; ++i) msbsA[i] = (uint8_t)(rng.next() & 1);
    std::vector<uint64_t> outA_r(n);
    EXPECT(conv.or_aux(msbsA.data(), outA_r.data(), n, corr, p) == OTStatus::kOk,
           "or_aux random");
    io->send_data(outA_r.data(), n * sizeof(uint64_t));
    io->send_data(msbsA.data(), n);
    io->flush();

    std::cout << "[sender] ring_to_field OR_AUX sent: boundary=" << nb
              << " random=" << n << " p=" << p << std::endl;
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port) {
    using namespace gpu_mm;
    EmpOTProvider prov(OTParty::kReceiver, port);
    sci::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");
    RingToField conv(prov, OTParty::kReceiver);

    // --- boundary block: msb_B = {0,0,1,1} (pairs with A's {0,1,0,1}) ---
    const uint8_t msbB[] = {0, 0, 1, 1};
    const int32_t nb = (int32_t)(sizeof(msbB) / sizeof(msbB[0]));
    std::vector<uint64_t> outB(nb);
    EXPECT(conv.or_aux(msbB, outB.data(), nb, corr, p) == OTStatus::kOk,
           "or_aux boundary recv");

    std::vector<uint64_t> outA(nb);
    std::vector<uint8_t> msbA(nb);
    io->recv_data(outA.data(), nb * sizeof(uint64_t));
    io->recv_data(msbA.data(), nb);

    int fails = 0;
    for (int32_t i = 0; i < nb; ++i) {
        // reconstruction: (out_A + out_B) mod p == (msb_A | msb_B) * corr mod p
        uint64_t got = (outA[i] + outB[i]) % p;
        uint64_t want = ((uint64_t)(msbA[i] | msbB[i]) * corr) % p;
        if (got != want) {
            std::cerr << "FAIL boundary @ " << i << " A=" << (int)msbA[i]
                      << " B=" << (int)msbB[i] << " got=" << got
                      << " want=" << want << "\n";
            ++fails;
        }
    }

    // --- random 100k block: B draws its own MSB bits (independent of A) ---
    const int32_t n = 100000;
    std::vector<uint8_t> msbsB(n);
    Rng rng(0xd1b54a32d192ed03ULL);
    for (int32_t i = 0; i < n; ++i) msbsB[i] = (uint8_t)(rng.next() & 1);
    std::vector<uint64_t> outB_r(n);
    EXPECT(conv.or_aux(msbsB.data(), outB_r.data(), n, corr, p) == OTStatus::kOk,
           "or_aux random recv");

    std::vector<uint64_t> outA_r(n);
    std::vector<uint8_t> msbsA(n);
    io->recv_data(outA_r.data(), n * sizeof(uint64_t));
    io->recv_data(msbsA.data(), n);

    for (int32_t i = 0; i < n; ++i) {
        uint64_t got = (outA_r[i] + outB_r[i]) % p;
        uint64_t want = ((uint64_t)(msbsA[i] | msbsB[i]) * corr) % p;
        if (got != want) {
            if (fails < 5)
                std::cerr << "FAIL random @ " << i << " A=" << (int)msbsA[i]
                          << " B=" << (int)msbsB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++fails;
        }
    }

    if (fails != 0) {
        std::cerr << "ring_to_field OR_AUX test FAILED (" << fails
                  << " mismatches)\n";
        return 1;
    }
    std::cout << "[receiver] ring_to_field OR_AUX OK: boundary=" << nb
              << " random=" << n << " p=" << p << std::endl;
    std::cout << "ring_to_field OR_AUX test PASSED" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ring_to_field_test <party 1|2> <port>\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./build/bin/ring_to_field_test 1 <port> &\n"
                  << "  sudo ./build/bin/ring_to_field_test 2 <port>\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    if (party == 1) return run_sender(port);
    if (party == 2) return run_receiver(port);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
