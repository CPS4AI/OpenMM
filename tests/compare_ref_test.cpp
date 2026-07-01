// CPU baseline: full 32-bit Millionaire compare via SCI (Task M3-T6 Phase C).
//
// Two-party. The CORRECTNESS + perf baseline for the cuOT CuotCompare
// diagnostic. Directly calls sci::MillionaireProtocol::compare
// (SCI/src/Millionaire/millionaire.h:76) on 32-bit shares, n=100000 random
// pairs, verifies res_A ^ res_B == (dataA > dataB). CLEAN 100k PASS.
//
// ALSO reports per-compare wall-clock latency (warm-up 1 + repeat 5, average,
// mirrors ring_to_field_test.cpp:36-50) and wire bytes (io->counter delta) —
// the CPU numbers the cuOT compare is measured against.
//
// Usage (two-party, run from repo root; bind needs sudo on this box):
//   sudo ./build/bin/compare_ref_test 1 <port> & \
//   sudo ./build/bin/compare_ref_test 2 <port>
#include "Millionaire/millionaire.h"
#include "OT/ot_pack.h"
#include "utils/net_io_channel.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

constexpr int32_t kN_default = 100000;
int32_t kN = kN_default;
constexpr int32_t kBits = 32;
const uint64_t kMask = (1ULL << kBits) - 1;
const uint32_t kWarm = 1;
const uint32_t kRepeat = 5;

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

int run_party(int party, int port) {
    sci::NetIO* io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);
    sci::OTPack<sci::NetIO>* otpack = new sci::OTPack<sci::NetIO>(io, party);
    MillionaireProtocol<sci::NetIO> mill(party, io, otpack);

    // Shared data: both parties draw the SAME dataA (seed) and SAME dataB.
    // dataA is party-1's share, dataB party-2's share; the compare is on the
    // RECONSTRUCTED value. For a clean test, give party 1 share=a, party 2
    // share=b, compare a>b. (Both parties know a,b here — test only.)
    std::vector<uint64_t> dataA(kN), dataB(kN);
    Rng rngA(0x9e3779b97f4a7c15ULL), rngB(0xd1b54a32d192ed03ULL);
    for (int i = 0; i < kN; ++i) {
        dataA[i] = rngA.next() & kMask;
        dataB[i] = rngB.next() & kMask;
    }
    // This party's input: party 1 holds dataA, party 2 holds dataB.
    const uint64_t* mine = (party == 1) ? dataA.data() : dataB.data();

    std::vector<uint64_t> tmp(kN);
    for (int i = 0; i < kN; ++i) tmp[i] = mine[i] & kMask;
    std::vector<uint8_t> res(kN);

    std::vector<uint64_t> runtimes;
    std::vector<uint64_t> comms;
    for (uint32_t round = 0; round < kWarm + kRepeat; ++round) {
        io->sync();
        uint64_t c0 = io->counter;
        auto t0 = std::chrono::high_resolution_clock::now();
        mill.compare(res.data(), tmp.data(), kN, kBits, /*greater=*/true);
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t c1 = io->counter;
        runtimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                               t1 - t0).count());
        comms.push_back(c1 - c0);
    }
    uint64_t avg_rt = 0, avg_cm = 0;
    for (uint32_t i = kWarm; i < kWarm + kRepeat; ++i) {
        avg_rt += runtimes[i]; avg_cm += comms[i];
    }
    avg_rt /= kRepeat; avg_cm /= kRepeat;

    // Verification: party 1 sends res_A to party 2, which checks
    // res_A ^ res_B == (dataA > dataB).
    int64_t bad = 0;
    if (party == 1) {
        io->send_data(res.data(), kN); io->flush();
        io->send_data(dataA.data(), kN * sizeof(uint64_t)); io->flush();
    } else {
        std::vector<uint8_t> resA(kN);
        std::vector<uint64_t> dataA2(kN);
        io->recv_data(resA.data(), kN);
        io->recv_data(dataA2.data(), kN * sizeof(uint64_t));
        for (int i = 0; i < kN; ++i) {
            uint8_t got = (uint8_t)(resA[i] ^ res[i]);
            uint8_t want = (uint8_t)(dataA2[i] > dataB[i] ? 1 : 0);
            if (got != want) {
                if (bad < 3)
                    std::cerr << "FAIL @ " << i << " A=" << dataA2[i]
                              << " B=" << dataB[i] << " got=" << (int)got
                              << " want=" << (int)want << "\n";
                ++bad;
            }
        }
    }

    std::cout << "[party " << party << "] compare_ref n=" << kN
              << " bits=" << kBits
              << " avg_time=" << avg_rt << "us (" << (avg_rt / 1000.0) << "ms)"
              << " avg_comm=" << avg_cm << " bytes"
              << " bad=" << bad << std::endl;

    delete otpack;
    delete io;
    if (party == 2 && bad != 0) {
        std::cerr << "compare_ref test FAILED (" << bad << " mismatches)\n";
        return 1;
    }
    if (party == 2)
        std::cout << "compare_ref test PASSED (CPU Millionaire, clean 100k)"
                  << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: compare_ref_test <party 1|2> <port>\n"
                  << "  party 1 = ALICE (server), 2 = BOB (client)\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./build/bin/compare_ref_test 1 <port> &\n"
                  << "  sudo ./build/bin/compare_ref_test 2 <port>\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    if (argc > 3) kN = std::atoi(argv[3]);
    if (party == 1 || party == 2) return run_party(party, port);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
