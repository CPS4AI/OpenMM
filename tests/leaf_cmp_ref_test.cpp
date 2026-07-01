// CPU baseline: single-digit compare leaf via SCI kkot (Task M3-T6 Phase A).
//
// Two-party. The CORRECTNESS baseline for the cuOT KkotLeafCmp diagnostic.
// Mirrors the `bitlength <= beta` early-exit leaf of MillionaireProtocol
// (millionaire.h:82-122) EXACTLY: builds the leaf table msg[k] =
// ((dataA > k) ^ res) and serves it with otpack->kkot[bitlength-1]->send/recv
// (a Cheetah SilentOTN = 1-OO-N-KOT, N = 2^bitlength, l = 1).
//
// Verifies, for bitlength ∈ {1,2,3,4} (N ∈ {2,4,8,16}), n = 100000 random
// digit pairs:
//   res_A[i] ^ res_B[i] == (dataA[i] > dataB[i])   (greater_than=true)
// This is the SHARED comparison bit. CLEAN 100k PASS = the CPU correctness
// baseline the cuOT variant (cuot_leaf_cmp_test) is measured against.
//
// ALSO reports, per bitlength, the wire bytes consumed by ONE leaf batch
// (io->counter delta) — the real CPU kkot cost (includes Ferret RCOT extend
// + the garbled table). This is the number the cuOT leaf's structural bytes
// are compared against in docs/research/m3-cuot-integration.md §8.
//
// Usage (two-party, run from repo root so ./data exists for Ferret; bind
// needs sudo on this box):
//   sudo ./build/bin/leaf_cmp_ref_test 1 <port> & \
//   sudo ./build/bin/leaf_cmp_ref_test 2 <port>
#include "OT/ot_pack.h"           // sci::OTPack<sci::NetIO> (kkot)
#include "utils/net_io_channel.h" // sci::NetIO (+ counter)

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
const int bitlengths[] = {1, 2, 3, 4};  // => N ∈ {2,4,8,16}

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
    int party = sci::ALICE;
    sci::NetIO* io = new sci::NetIO(/*server=*/nullptr, port);
    sci::OTPack<sci::NetIO>* otpack = new sci::OTPack<sci::NetIO>(io, party);

    for (int bl : bitlengths) {
        const int N = 1 << bl;
        const uint8_t mask = (uint8_t)(N - 1);
        // dataA ∈ [0,N)
        std::vector<uint8_t> dataA(n);
        Rng rng(0x9e3779b97f4a7c15ULL ^ (uint64_t)bl);
        for (int i = 0; i < n; ++i) dataA[i] = (uint8_t)(rng.next() & mask);

        // random mask res (mirrors millionaire.h:87 prg.random_data &1)
        std::vector<uint8_t> res(n);
        sci::PRG128 prg;
        prg.random_data(res.data(), n * sizeof(uint8_t));
        for (int i = 0; i < n; ++i) res[i] &= 1;

        // leaf table msg[i*N + k] = ((dataA[i] > k) ^ res[i])  (millionaire.h:93-94)
        std::vector<uint8_t*> msg(n);
        std::vector<uint8_t> msgbuf((size_t)n * N);
        for (int i = 0; i < n; ++i) {
            msg[i] = msgbuf.data() + (size_t)i * N;
            for (int k = 0; k < N; ++k)
                msg[i][k] = (uint8_t)((dataA[i] > k) ^ res[i]);
        }

        uint64_t cnt0 = io->counter;
        otpack->kkot[bl - 1]->send(msg.data(), n, /*l=*/1);
        uint64_t cnt1 = io->counter;

        // send res + dataA to receiver for reconstruction check
        io->send_data(res.data(), n);
        io->send_data(dataA.data(), n);
        io->flush();

        std::cout << "[sender] bl=" << bl << " N=" << N << " n=" << n
                  << " leaf_comm=" << (cnt1 - cnt0) << " bytes" << std::endl;
    }
    delete otpack;
    delete io;
    return 0;
}

// ---- receiver (BOB) ----
int run_receiver(int port) {
    int party = sci::BOB;
    sci::NetIO* io = new sci::NetIO("127.0.0.1", port);
    sci::OTPack<sci::NetIO>* otpack = new sci::OTPack<sci::NetIO>(io, party);

    int total_fails = 0;
    for (int bl : bitlengths) {
        const int N = 1 << bl;
        const uint8_t mask = (uint8_t)(N - 1);
        std::vector<uint8_t> dataB(n);
        Rng rng(0xd1b54a32d192ed03ULL ^ (uint64_t)bl);
        for (int i = 0; i < n; ++i) dataB[i] = (uint8_t)(rng.next() & mask);

        std::vector<uint8_t> choice(n);
        for (int i = 0; i < n; ++i) choice[i] = dataB[i] & mask;

        std::vector<uint8_t> resB(n);
        otpack->kkot[bl - 1]->recv(resB.data(), choice.data(), n, /*l=*/1);

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
                  << " bad=" << bad << " (" << rate << "%)" << std::endl;
        if (bad != 0) ++total_fails;
    }
    delete otpack;
    delete io;
    if (total_fails != 0) {
        std::cerr << "leaf_cmp_ref test FAILED (" << total_fails
                  << " bitlengths with mismatches)\n";
        return 1;
    }
    std::cout << "leaf_cmp_ref test PASSED (CPU kkot leaf, clean 100k x 4 N)"
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: leaf_cmp_ref_test <party 1|2> <port>\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./build/bin/leaf_cmp_ref_test 1 <port> &\n"
                  << "  sudo ./build/bin/leaf_cmp_ref_test 2 <port>\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    if (party == 1) return run_sender(port);
    if (party == 2) return run_receiver(port);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
