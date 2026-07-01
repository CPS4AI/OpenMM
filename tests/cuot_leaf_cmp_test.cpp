// cuOT-backed single-digit compare leaf diagnostic (Task M3-T6 Phase A).
//
// Two-party, two GPUs. The cuOT counterpart of leaf_cmp_ref_test (the CPU
// baseline). Runs the SAME bitlengths ∈ {1,2,3,4} (N ∈ {2,4,8,16}) and
// n=100000 random digit pairs through KkotLeafCmp (on CuotProvider) instead of
// SCI kkot. Same seeds so inputs match the CPU baseline.
//
// Verifies: res_A[i] ^ res_B[i] == (dataA[i] > dataB[i])  (greater_than=true)
// and REPORTS the actual bad count / rate per bitlength. This is the
// DIAGNOSTIC: the cuOT leaf inherits cuOT's ~2% RCOT floor, compounded over
// logN = bitlength block-COTs per output, so the expected leaf error rate is
// ≈ 1 - 0.98^bitlength: bl=1 ≈ 2.0%, bl=2 ≈ 4.0%, bl=3 ≈ 5.9%, bl=4 ≈ 7.8%.
// The test PASSES iff the rate stays within [0, 15%] (absorbs the floor ×
// bitlength + run-to-run variance); a rate far above the floor (e.g. ~50%)
// means the garbled-table / MITCCRH reduction port is buggy, NOT the RCOT.
//
// ALSO reports, per bitlength, the STRUCTURAL wire byte count the cuOT leaf
// consumes (emp::NetIO has no counter, so this is computed analytically from
// the reduction: n bytes diff-vector + 1 block seed + ceil(n*N/8) garbled
// table bytes, per the send/recv_cot_blocks + pack_leaf_messages layout). This
// is the number compared against the CPU kkot leaf's real io->counter delta
// (printed by leaf_cmp_ref_test) in docs/research/m3-cuot-integration.md §8.
//
// CAVEAT: NOT a correctness-clear comparison. Per user direction (2026-06-28,
// "先建 cuOT 比较诊断"), this measures the inherited error + comm shape, it
// does not claim a working cuOT compare.
//
// Usage (binary takes <party> <port> [gpu]; host 127.0.0.1; run from repo
// root; bind needs sudo; pin each party to one physical GPU via
// CUDA_VISIBLE_DEVICES — the launcher does this):
//   sudo ./run_cuot_leaf_cmp_test.sh [PORT]
#include "gpu_mm/cuot_provider.h"
#include "gpu_mm/kkot_leaf_cmp.h"
#include "gpu_mm/ot_provider.h"

#include <emp-tool/emp-tool.h>  // emp::NetIO

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

constexpr int32_t n = 100000;
const int bitlengths[] = {1, 2, 3, 4};

// Diagnostic pass threshold: failures within the inherited ~2% RCOT floor
// compounded over logN block-COTs. Allow up to 15% (absorbs floor × bitlength
// + variance); above that the reduction port is buggy.
constexpr double kDiagMaxRate = 15.0;

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

// Structural wire bytes for ONE cuOT leaf batch (n outputs, N choices, l=1):
//   - n bytes  : receiver->sender diff vector (send_cot_blocks: d_i = lsb^b)
//   - 16 bytes : sender->receiver MITCCRH seed (1 block)
//   - ceil(n*N/8) bytes : garbled table (pack_leaf_messages, l=1)
// (rcot_blocks itself sends nothing on the wire — pure GPU compute + the diff
// vector above; Ferret base-OT/extend amortized to setup, not per-leaf.)
inline uint64_t cuot_leaf_comm_bytes(int n_, int N) {
    uint64_t diff = (uint64_t)n_;                 // n bytes (1 bit/element, sent as bytes)
    uint64_t seed = 16;                           // 1 block
    uint64_t table = ((uint64_t)n_ * (uint64_t)N + 7) / 8;  // l=1 packed bits
    return diff + seed + table;
}

// ---- sender (ALICE) ----
int run_sender(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kSender, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "sender net_io null");
    KkotLeafCmp leaf(prov, OTParty::kSender);

    for (int bl : bitlengths) {
        const int N = 1 << bl;
        const uint8_t mask = (uint8_t)(N - 1);
        std::vector<uint8_t> dataA(n);
        Rng rng(0x9e3779b97f4a7c15ULL ^ (uint64_t)bl);
        for (int i = 0; i < n; ++i) dataA[i] = (uint8_t)(rng.next() & mask);

        std::vector<uint8_t> res(n);
        EXPECT(leaf.send_leaf(dataA.data(), res.data(), n, N, /*l=*/1,
                              /*greater=*/true) == OTStatus::kOk,
               "send_leaf");

        // send res + dataA to receiver for reconstruction check
        io->send_data(res.data(), n);
        io->send_data(dataA.data(), n);
        io->flush();

        std::cout << "[sender] bl=" << bl << " N=" << N << " n=" << n
                  << " cuot_leaf_comm≈" << cuot_leaf_comm_bytes(n, N)
                  << " bytes gpu=" << gpu << std::endl;
    }
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kReceiver, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");
    KkotLeafCmp leaf(prov, OTParty::kReceiver);

    int total_fails = 0;
    for (int bl : bitlengths) {
        const int N = 1 << bl;
        const uint8_t mask = (uint8_t)(N - 1);
        std::vector<uint8_t> dataB(n);
        Rng rng(0xd1b54a32d192ed03ULL ^ (uint64_t)bl);
        for (int i = 0; i < n; ++i) dataB[i] = (uint8_t)(rng.next() & mask);

        std::vector<uint8_t> resB(n);
        EXPECT(leaf.recv_leaf(dataB.data(), resB.data(), n, N, /*l=*/1) ==
                   OTStatus::kOk,
               "recv_leaf");

        std::vector<uint8_t> resA(n), dataA(n);
        io->recv_data(resA.data(), n);
        io->recv_data(dataA.data(), n);

        int64_t bad = 0;
        for (int i = 0; i < n; ++i) {
            uint8_t got = (uint8_t)(resA[i] ^ resB[i]);
            uint8_t want = (uint8_t)(dataA[i] > dataB[i] ? 1 : 0);
            if (got != want) {
                if (bad < 3)
                    std::cerr << "FAIL bl=" << bl << " @ " << i << " A="
                              << (int)dataA[i] << " B=" << (int)dataB[i]
                              << " got=" << (int)got << " want=" << (int)want
                              << "\n";
                ++bad;
            }
        }
        double rate = 100.0 * bad / n;
        std::cout << "[receiver] bl=" << bl << " N=" << N << " n=" << n
                  << " bad=" << bad << " (" << rate << "%)"
                  << " cuot_leaf_comm≈" << cuot_leaf_comm_bytes(n, N)
                  << " bytes gpu=" << gpu << std::endl;
        if (rate > kDiagMaxRate) {
            std::cerr << "FAIL bl=" << bl << " rate " << rate << "% > "
                      << kDiagMaxRate << "% (reduction port buggy, not RCOT)\n";
            ++total_fails;
        }
    }
    if (total_fails != 0) {
        std::cerr << "cuOT leaf_cmp test FAILED (reduction port)\n";
        return 1;
    }
    std::cout << "cuOT leaf_cmp test PASSED (diagnostic; inherited ~2% RCOT "
                 "floor compounded over logN, within tolerance)"
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: cuot_leaf_cmp_test <party 1|2> <port> [gpu]\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "  host is 127.0.0.1 (hardcoded in CuotProvider)\n"
                  << "  Pin each party to one physical GPU via CUDA_VISIBLE_DEVICES.\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./run_cuot_leaf_cmp_test.sh [port]\n";
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
