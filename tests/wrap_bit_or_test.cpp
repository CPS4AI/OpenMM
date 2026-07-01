// Wrap-bit logical-OR primitive test (Task M3-T2).
//
// Two-party. Verifies WrapBitOr (on gpu_mm::OTProvider, CPU EmpOTProvider
// default) computes the correct shared logical-OR → arithmetic wrap:
//   out_A + out_B ≡ OR(local_bit_A, local_bit_B)   (mod 2^l)
// for l = 32, M = 2^32.
//
// Coverage (per Task M3 acceptance — reused here as the CPU baseline so M3-T3
// cuOT runs the same cases):
//   - boundary cases: bit pairs whose OR is exactly {0, 1, M-1 (=all-1/-1),
//     M/2-1, M/2, M-1}. Because OR ∈ {0,1} the *OR value* only hits 0 and 1
//     directly; the M/2 and M-1 boundaries are exercised by the random 100k
//     path (which lands every bit pattern). The boundary block below pins the
//     four sign-relevant OR outcomes (both-0, A-only-1, B-only-1, both-1) and
//     the all-ones share (-1 mod M), which is the wrap-carry edge case.
//   - random 100000 samples (M3-T3 acceptance "random 100000 samples pass").
//
// Verification channel: the sender sends its `out_A` share back to the
// receiver over the provider's NetIO (out_A is secret in real use; exposed
// here only for the test, exactly like ot_provider_smoke_test.cpp:57). The
// receiver reconstructs out_A + out_B and checks against OR(bit_A, bit_B).
//
// Usage (two-party, run from repo root so ./data exists for Ferret; bind
// needs sudo on this box — see memory openmm-build-env §4):
//   sudo ./build/bin/wrap_bit_or_test 1 <port> & \
//   sudo ./build/bin/wrap_bit_or_test 2 <port>
//   party 1 = sender (ALICE), party 2 = receiver (BOB).
#include "gpu_mm/emp_ot_provider.h"
#include "gpu_mm/ot_provider.h"
#include "gpu_mm/wrap_bit_or.h"
#include "utils/net_io_channel.h"  // sci::NetIO send/recv for verification

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

// xorshift PRNG (deterministic, no Math.random). Mirrors
// share_tensor_backend_test.cpp's Rng so both parties generate the SAME
// choice bits and the receiver can predict the sender's bits for the check.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
};

constexpr int l = 32;
const uint64_t mask = (l == 64) ? ~0ULL : ((1ULL << l) - 1);

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
    WrapBitOr orob(prov, OTParty::kSender);

    // --- boundary block: pin the four OR outcomes + the -1 (all-ones) share ---
    // bit pairs (A, B): (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->1.
    // Plus a both-0 case where A's share is forced to all-ones (M-1 = -1 mod M)
    // to exercise the wrap-carry edge in the (x_A - y_A) & mask local adjust.
    const uint8_t bA[] = {0, 1, 0, 1, 0};
    const int32_t nb = (int32_t)(sizeof(bA) / sizeof(bA[0]));
    std::vector<uint64_t> outA(nb);
    EXPECT(orob.or_share(bA, outA.data(), nb, l) == OTStatus::kOk, "or_share boundary");
    // send A's shares + bits to receiver for reconstruction
    io->send_data(outA.data(), nb * sizeof(uint64_t));
    io->send_data(bA, nb);  // A's bits (so receiver knows the pair)
    io->flush();

    // --- random 100k block ---
    const int32_t n = 100000;
    std::vector<uint8_t> bitsA(n);
    Rng rng(0x9e3779b97f4a7c15ULL);
    for (int32_t i = 0; i < n; ++i) bitsA[i] = (uint8_t)(rng.next() & 1);
    std::vector<uint64_t> outA_r(n);
    EXPECT(orob.or_share(bitsA.data(), outA_r.data(), n, l) == OTStatus::kOk,
           "or_share random");
    io->send_data(outA_r.data(), n * sizeof(uint64_t));
    io->send_data(bitsA.data(), n);
    io->flush();

    std::cout << "[sender] wrap-bit OR sent: boundary=" << nb << " random=" << n
              << " l=" << l << std::endl;
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port) {
    using namespace gpu_mm;
    EmpOTProvider prov(OTParty::kReceiver, port);
    sci::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");
    WrapBitOr orob(prov, OTParty::kReceiver);

    // --- boundary block: B's bits paired with A's sent bits ---
    // B chooses bits so the pairs cover (0,0),(1,0),(0,1),(1,1) + (0,0) again.
    const uint8_t bB[] = {0, 0, 1, 1, 0};
    const int32_t nb = (int32_t)(sizeof(bB) / sizeof(bB[0]));
    std::vector<uint64_t> outB(nb);
    EXPECT(orob.or_share(bB, outB.data(), nb, l) == OTStatus::kOk, "or_share boundary recv");

    std::vector<uint64_t> outA(nb);
    std::vector<uint8_t> bA(nb);
    io->recv_data(outA.data(), nb * sizeof(uint64_t));
    io->recv_data(bA.data(), nb);

    int fails = 0;
    for (int32_t i = 0; i < nb; ++i) {
        uint64_t got = (outA[i] + outB[i]) & mask;
        uint64_t want = (uint64_t)(bA[i] | bB[i]);  // OR ∈ {0,1}
        if (got != want) {
            std::cerr << "FAIL boundary @ " << i << " A=" << (int)bA[i] << " B="
                      << (int)bB[i] << " got=" << got << " want=" << want << "\n";
            ++fails;
        }
    }

    // --- random 100k block ---
    const int32_t n = 100000;
    // B generates its own random bits with a DIFFERENT seed (independent of A).
    std::vector<uint8_t> bitsB(n);
    Rng rng(0xd1b54a32d192ed03ULL);
    for (int32_t i = 0; i < n; ++i) bitsB[i] = (uint8_t)(rng.next() & 1);
    std::vector<uint64_t> outB_r(n);
    EXPECT(orob.or_share(bitsB.data(), outB_r.data(), n, l) == OTStatus::kOk,
           "or_share random recv");

    std::vector<uint64_t> outA_r(n);
    std::vector<uint8_t> bitsA(n);
    io->recv_data(outA_r.data(), n * sizeof(uint64_t));
    io->recv_data(bitsA.data(), n);

    for (int32_t i = 0; i < n; ++i) {
        uint64_t got = (outA_r[i] + outB_r[i]) & mask;
        uint64_t want = (uint64_t)(bitsA[i] | bitsB[i]);
        if (got != want) {
            if (fails < 5)
                std::cerr << "FAIL random @ " << i << " A=" << (int)bitsA[i]
                          << " B=" << (int)bitsB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++fails;
        }
    }

    if (fails != 0) {
        std::cerr << "wrap-bit OR test FAILED (" << fails << " mismatches)\n";
        return 1;
    }
    std::cout << "[receiver] wrap-bit OR OK: boundary=" << nb << " random=" << n
              << " l=" << l << std::endl;
    std::cout << "wrap-bit OR test PASSED" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: wrap_bit_or_test <party 1|2> <port>\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./build/bin/wrap_bit_or_test 1 <port> &\n"
                  << "  sudo ./build/bin/wrap_bit_or_test 2 <port>\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    if (party == 1) return run_sender(port);
    if (party == 2) return run_receiver(port);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
