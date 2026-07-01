// cuOT-backed wrap-bit logical-OR test (Task M3-T3).
//
// This is the cuOT counterpart of tests/wrap_bit_or_test.cpp (the CPU
// EmpOTProvider baseline, M3-T2). It runs the EXACT SAME boundary block and
// 100000-sample random cases through WrapBitOr, but backed by CuotProvider
// (GPU Ferret) instead of EmpOTProvider. Acceptance #1 of M3-T3 — "CPU backend
// and cuOT backend both pass same wrap-bit tests" — is therefore exercised on
// identical inputs.
//
//   out_A + out_B ≡ OR(local_bit_A, local_bit_B)   (mod 2^l),  l = 32.
//
// Verification channel: the sender sends its `out_A` share back to the receiver
// over the provider's own emp::NetIO (out_A is secret in real use; exposed here
// only for the test, exactly like cuot_deltaot_arithmetic_test.cpp / cuot_correlation_test.cpp).
//
// CAVEAT (recorded in docs/research/m2-cuot-standalone.md §0, §7.4): cuOT's
// underlying block-RCOT has a STABLE ~2% error floor on this 4×A40 box (M2-T2
// SHELVED). CuotProvider::send_cot/recv_cot layer MITCCRH hashing on that
// block-RCOT (M2-T3 adapter, logic PASS), so the wrap-bit OR reconstruction
// inherits the same ~2% floor: roughly 2 of every 100 OR outputs reconstruct
// wrong. This test reports the actual bad count and rate. It is considered an
// ADAPTER-LOGIC / WIRING pass if the only failures are the inherited ~2% floor
// (rate ≤ 4% to absorb variance); a rate far above ~2% means the WrapBitOr
// primitive or the cuOT wiring is buggy, not the RCOT. Per user direction
// (2026-06-28), this known floor is accepted and documented; M3-T3's strict
// "100000 samples pass" criterion is NOT met cleanly by the cuOT backend —
// only the CPU backend meets it. See docs/research/m3-cuot-integration.md.
//
// Mirrors wrap_bit_or_test.cpp's cases verbatim (same seeds, same boundary):
//   boundary  bA = {0,1,0,1,0},  bB = {0,0,1,1,0}   (covers all four OR outcomes)
//   random    A-seed 0x9e3779b97f4a7c15,  B-seed 0xd1b54a32d192ed03,  n=100000
//
// Usage (binary takes <party> <port> [gpu]; host 127.0.0.1; run from repo
// root; bind needs sudo; pin each party to one physical GPU via
// CUDA_VISIBLE_DEVICES — the launcher does this):
//   sudo ./run_cuot_wrap_bit_or_test.sh [PORT]
#include "gpu_mm/cuot_provider.h"
#include "gpu_mm/ot_provider.h"
#include "gpu_mm/wrap_bit_or.h"

#include <emp-tool/emp-tool.h>  // emp::NetIO send/recv

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

// xorshift PRNG (deterministic, no Math.random). Mirrors
// wrap_bit_or_test.cpp's Rng so the SAME choice bits are generated on each
// party as in the CPU baseline.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    }
};

constexpr int l = 32;
const uint64_t mask = (l == 64) ? ~0ULL : ((1ULL << l) - 1);

// Adapter-logic pass threshold: failures within the inherited ~2% RCOT floor
// (allow up to 4% to absorb run-to-run variance; above that the WrapBitOr
// primitive / cuOT wiring is buggy, not the RCOT). Mirrors
// cuot_deltaot_arithmetic_test.cpp's 4% gate.
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
    WrapBitOr orob(prov, OTParty::kSender);

    // --- boundary block: bA = {0,1,0,1,0} (matches wrap_bit_or_test.cpp) ---
    const uint8_t bA[] = {0, 1, 0, 1, 0};
    const int32_t nb = (int32_t)(sizeof(bA) / sizeof(bA[0]));
    std::vector<uint64_t> outA(nb);
    EXPECT(orob.or_share(bA, outA.data(), nb, l) == OTStatus::kOk,
           "or_share boundary");
    io->send_data(outA.data(), nb * sizeof(uint64_t));
    io->send_data(bA, nb);  // A's bits (so receiver knows the pair)
    io->flush();

    // --- random 100k block: A-seed matches wrap_bit_or_test.cpp ---
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

    std::cout << "[sender] cuOT wrap-bit OR sent: boundary=" << nb
              << " random=" << n << " l=" << l << " gpu=" << gpu << std::endl;
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kReceiver, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");
    WrapBitOr orob(prov, OTParty::kReceiver);

    int fails = 0;

    // --- boundary block: bB = {0,0,1,1,0} (matches wrap_bit_or_test.cpp) ---
    const uint8_t bB[] = {0, 0, 1, 1, 0};
    const int32_t nb = (int32_t)(sizeof(bB) / sizeof(bB[0]));
    std::vector<uint64_t> outB(nb);
    EXPECT(orob.or_share(bB, outB.data(), nb, l) == OTStatus::kOk,
           "or_share boundary recv");

    std::vector<uint64_t> outA(nb);
    std::vector<uint8_t> bA(nb);
    io->recv_data(outA.data(), nb * sizeof(uint64_t));
    io->recv_data(bA.data(), nb);

    int64_t boundary_bad = 0;
    for (int32_t i = 0; i < nb; ++i) {
        uint64_t got = (outA[i] + outB[i]) & mask;
        uint64_t want = (uint64_t)(bA[i] | bB[i]);  // OR ∈ {0,1}
        if (got != want) {
            if (boundary_bad < 3)
                std::cerr << "FAIL boundary @ " << i << " A=" << (int)bA[i]
                          << " B=" << (int)bB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++boundary_bad;
        }
    }

    // --- random 100k block: B-seed matches wrap_bit_or_test.cpp ---
    const int32_t n = 100000;
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

    int64_t random_bad = 0;
    for (int32_t i = 0; i < n; ++i) {
        uint64_t got = (outA_r[i] + outB_r[i]) & mask;
        uint64_t want = (uint64_t)(bitsA[i] | bitsB[i]);
        if (got != want) {
            if (random_bad < 5)
                std::cerr << "FAIL random @ " << i << " A=" << (int)bitsA[i]
                          << " B=" << (int)bitsB[i] << " got=" << got
                          << " want=" << want << "\n";
            ++random_bad;
        }
    }

    double rate = 100.0 * random_bad / n;
    std::cout << "[receiver] cuOT wrap-bit OR: boundary_bad=" << boundary_bad
              << " random_bad=" << random_bad << "/" << n << " (" << rate
              << "%)  l=" << l << " gpu=" << gpu << std::endl;

    // Adapter-logic / wiring pass: the ONLY acceptable failures are the
    // inherited ~2% RCOT floor. Above 4% the WrapBitOr port or cuOT wiring is
    // buggy. (Boundary is 5 elements; ~2% expected ≈ 0, but a stray hit is
    // consistent with the floor — reported, not separately gated.)
    if (rate > kAdapterLogicMaxRate) {
        std::cerr << "FAIL cuOT wrap-bit OR random rate " << rate
                  << "% > " << kAdapterLogicMaxRate
                  << "% (WrapBitOr/cuOT wiring buggy, not RCOT floor)\n";
        ++fails;
    }
    if (fails != 0) {
        std::cerr << "cuOT wrap-bit OR test FAILED (wiring)\n";
        return 1;
    }
    std::cout << "cuOT wrap-bit OR test PASSED (wiring + adapter logic; "
                 "inherited ~2% RCOT floor within tolerance)"
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: cuot_wrap_bit_or_test <party 1|2> <port> [gpu]\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "  host is 127.0.0.1 (hardcoded in CuotProvider)\n"
                  << "  Pin each party to one physical GPU via CUDA_VISIBLE_DEVICES.\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./run_cuot_wrap_bit_or_test.sh [port]\n";
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
