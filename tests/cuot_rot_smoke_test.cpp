// cuOT block-ROT smoke test (Task M3-T6 Phase B, ROT primitive validation).
//
// Validates CuotProvider::send_rot_blocks / recv_rot_blocks IN ISOLATION before
// BitTripleGen composes on top. A single 1-2 ROT:
//   sender (ALICE) gets (m0, m1) two independent random blocks;
//   receiver (BOB) gets m_{r} for its random choice r.
// Verification: m_{r} == r ? m1 : m0 (per element), checked by the sender
// sending (m0, m1) back to the receiver over the same NetIO.
//
// This MUST pass (within the ~2% floor) before bit_triple_test is trusted — a
// broken ROT primitive would corrupt the triple algebra at ~50%, not ~4%.
//
// CAVEAT: inherits cuOT's ~2% RCOT floor (M2-T2). PASS = rate ≤ 4%.
//
// Usage: sudo ./run_cuot_rot_smoke_test.sh [PORT]
#include "gpu_mm/cuot_provider.h"
#include "gpu_mm/ot_provider.h"

#include <emp-tool/emp-tool.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

constexpr int64_t kN = 100000;
constexpr double kMaxRate = 4.0;

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

int run_sender(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kSender, port, gpu, "127.0.0.1", true,
                      "data/pre_ot_data_reg_send_rot");
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "sender net_io null");

    std::vector<emp::block> out((size_t)kN * 2);
    EXPECT(prov.send_rot_blocks(reinterpret_cast<uint8_t*>(out.data()), kN) ==
               OTStatus::kOk,
           "send_rot_blocks");
    // send (m0,m1) back for verification (secret in real use; test only).
    io->send_data(out.data(), kN * 2 * sizeof(emp::block));
    io->flush();
    std::cout << "[sender] cuOT ROT sent n=" << kN << " gpu=" << gpu << std::endl;
    return 0;
}

int run_receiver(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kReceiver, port, gpu, "127.0.0.1", true,
                      "data/pre_ot_data_reg_recv_rot");
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");

    std::vector<uint8_t> rbuf((size_t)kN);
    Rng rng(0xd1b54a32d192ed03ULL);
    for (int i = 0; i < kN; ++i) rbuf[i] = (uint8_t)(rng.next() & 1);
    const bool* r = reinterpret_cast<const bool*>(rbuf.data());

    std::vector<emp::block> mr((size_t)kN);
    EXPECT(prov.recv_rot_blocks(reinterpret_cast<uint8_t*>(mr.data()), r, kN) ==
               OTStatus::kOk,
           "recv_rot_blocks");

    std::vector<emp::block> m01((size_t)kN * 2);
    io->recv_data(m01.data(), kN * 2 * sizeof(emp::block));

    int64_t bad = 0;
    for (int i = 0; i < kN; ++i) {
        emp::block want = rbuf[i] ? m01[2 * i + 1] : m01[2 * i];
        // compare 16 bytes (block equality). Use XOR==0.
        emp::block diff = _mm_xor_si128(mr[i], want);
        if (!_mm_testz_si128(diff, diff)) ++bad;
    }
    double rate = 100.0 * bad / kN;
    std::cout << "[receiver] cuOT ROT bad=" << bad << "/" << kN << " (" << rate
              << "%) gpu=" << gpu << std::endl;
    if (rate > kMaxRate) {
        std::cerr << "FAIL cuOT ROT rate " << rate << "% > " << kMaxRate
                  << "% (send_rot_blocks/recv_rot_blocks port buggy)\n";
        return 1;
    }
    std::cout << "cuOT ROT smoke test PASSED (within ~2% RCOT floor)" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: cuot_rot_smoke_test <party 1|2> <port> [gpu]\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "  sudo ./run_cuot_rot_smoke_test.sh [port]\n";
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
