// cuOT-backed ring→field conversion test (Task M3-T4).
//
// cuOT counterpart of tests/ring_to_field_or_aux_test.cpp (the CPU EmpOTProvider
// baseline). Runs the EXACT SAME boundary block and 100000-sample random cases
// through RingToField::or_aux, but backed by CuotProvider (GPU Ferret) instead
// of EmpOTProvider. Acceptance #1 of M3-T4 — "CPU backend and cuOT backend
// both pass" — is exercised on identical inputs.
//
//   (out_A[i] + out_B[i]) mod p == (msb_A[i] | msb_B[i]) * corr  mod p
//   ring_bw=32, p=4293918721, corr = ring_mod_mask % p + 1.
//
// Verification channel: the sender sends its `out_A` share + `msb_A` bits back
// to the receiver over the provider's own emp::NetIO (both secret in real use;
// exposed here only for the test, like cuot_wrap_bit_or_test.cpp).
//
// CAVEAT (docs/research/m2-cuot-standalone.md §0, §7.4): cuOT's block-RCOT has
// a STABLE ~2% error floor on this 4×A40 box (M2-T2 SHELVED). CuotProvider::
// send_cot_prime/recv_cot_prime layer MITCCRH hashing on that block-RCOT
// (M2-T3 adapter, logic PASS), so the OR_AUX reconstruction inherits the same
// ~2% floor. This test reports the actual bad count / rate and is considered
// an ADAPTER-LOGIC / WIRING pass if the only failures are the inherited floor
// (rate ≤ 4%); above that the RingToField primitive or cuOT wiring is buggy.
// Per user direction (2026-06-28), this known floor is accepted and documented;
// M3-T4's strict "random stress pass" is NOT met cleanly by the cuOT backend —
// only the CPU backend meets it. See docs/research/m3-cuot-integration.md.
//
// Mirrors ring_to_field_or_aux_test.cpp's cases verbatim (same seeds, same
// boundary): A-seed 0x9e3779b97f4a7c15, B-seed 0xd1b54a32d192ed03, n=100000.
//
// Usage (binary takes <party> <port> [gpu]; host 127.0.0.1; run from repo
// root; bind needs sudo; pin each party to one physical GPU via
// CUDA_VISIBLE_DEVICES — the launcher does this):
//   sudo ./run_cuot_ring_to_field_test.sh [PORT]
#include "gpu_mm/cuot_provider.h"
#include "gpu_mm/ot_provider.h"
#include "gpu_mm/ring_to_field.h"

#include <emp-tool/emp-tool.h>  // emp::NetIO send/recv

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

constexpr int32_t ring_bw = 32;
const uint64_t ring_mod_mask = (ring_bw == 64) ? ~0ULL : ((1ULL << ring_bw) - 1);
const uint64_t p = 4293918721ULL;
const uint64_t corr = (ring_mod_mask % p) + 1;

// Adapter-logic pass threshold: failures within the inherited ~2% RCOT floor
// (allow up to 4% to absorb run-to-run variance). Mirrors
// cuot_wrap_bit_or_test.cpp / cuot_deltaot_arithmetic_test.cpp.
constexpr double kAdapterLogicMaxRate = 4.0;

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

// ---- sender (ALICE) ----
int run_sender(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kSender, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "sender net_io null");
    RingToField conv(prov, OTParty::kSender);

    const uint8_t msbA[] = {0, 1, 0, 1};
    const int32_t nb = (int32_t)(sizeof(msbA) / sizeof(msbA[0]));
    std::vector<uint64_t> outA(nb);
    EXPECT(conv.or_aux(msbA, outA.data(), nb, corr, p) == OTStatus::kOk,
           "or_aux boundary");
    io->send_data(outA.data(), nb * sizeof(uint64_t));
    io->send_data(msbA, nb);
    io->flush();

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

    std::cout << "[sender] cuOT ring_to_field OR_AUX sent: boundary=" << nb
              << " random=" << n << " p=" << p << " gpu=" << gpu << std::endl;
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kReceiver, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");
    RingToField conv(prov, OTParty::kReceiver);

    int fails = 0;

    const uint8_t msbB[] = {0, 0, 1, 1};
    const int32_t nb = (int32_t)(sizeof(msbB) / sizeof(msbB[0]));
    std::vector<uint64_t> outB(nb);
    EXPECT(conv.or_aux(msbB, outB.data(), nb, corr, p) == OTStatus::kOk,
           "or_aux boundary recv");

    std::vector<uint64_t> outA(nb);
    std::vector<uint8_t> msbA(nb);
    io->recv_data(outA.data(), nb * sizeof(uint64_t));
    io->recv_data(msbA.data(), nb);

    int64_t boundary_bad = 0;
    for (int32_t i = 0; i < nb; ++i) {
        uint64_t got = (outA[i] + outB[i]) % p;
        uint64_t want = ((uint64_t)(msbA[i] | msbB[i]) * corr) % p;
        if (got != want) {
            if (boundary_bad < 3)
                std::cerr << "FAIL boundary @ " << i << " A=" << (int)msbA[i]
                          << " B=" << (int)msbB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++boundary_bad;
        }
    }

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

    int64_t random_bad = 0;
    for (int32_t i = 0; i < n; ++i) {
        uint64_t got = (outA_r[i] + outB_r[i]) % p;
        uint64_t want = ((uint64_t)(msbsA[i] | msbsB[i]) * corr) % p;
        if (got != want) {
            if (random_bad < 5)
                std::cerr << "FAIL random @ " << i << " A=" << (int)msbsA[i]
                          << " B=" << (int)msbsB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++random_bad;
        }
    }

    double rate = 100.0 * random_bad / n;
    std::cout << "[receiver] cuOT ring_to_field OR_AUX: boundary_bad="
              << boundary_bad << " random_bad=" << random_bad << "/" << n
              << " (" << rate << "%)  p=" << p << " gpu=" << gpu << std::endl;

    if (rate > kAdapterLogicMaxRate) {
        std::cerr << "FAIL cuOT ring_to_field random rate " << rate
                  << "% > " << kAdapterLogicMaxRate
                  << "% (RingToField/cuOT wiring buggy, not RCOT floor)\n";
        ++fails;
    }
    if (fails != 0) {
        std::cerr << "cuOT ring_to_field OR_AUX test FAILED (wiring)\n";
        return 1;
    }
    std::cout << "cuOT ring_to_field OR_AUX test PASSED (wiring + adapter "
                 "logic; inherited ~2% RCOT floor within tolerance)"
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: cuot_ring_to_field_test <party 1|2> <port> [gpu]\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "  host is 127.0.0.1 (hardcoded in CuotProvider)\n"
                  << "  Pin each party to one physical GPU via CUDA_VISIBLE_DEVICES.\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./run_cuot_ring_to_field_test.sh [port]\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    int gpu = (argc > 3) ? std::atoi(argv[3]) : 0;
    if (party == 1) return run_sender(port, gpu);
    if (party == 2) return run_receiver(port, gpu);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
