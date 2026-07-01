// cuOT bit-triple (_2ROT) diagnostic test (Task M3-T6 Phase B).
//
// Two-party, two GPUs, FOUR CuotProviders total (2 per party: prov_send =
// Ferret-SENDER, prov_recv = Ferret-RECEIVER, on two ports). Generates n
// random bit-AND triples via BitTripleGen and checks:
//   (a_A[i]^a_B[i]) & (b_A[i]^b_B[i]) == c_A[i]^c_B[i]
// for n = 100000 triples.
//
// CAVEAT: each triple uses TWO ROTs, so the error rate compounds the ~2% RCOT
// floor to ~4%. DIAGNOSTIC PASS = rate ≤ 8% (absorbs 2×floor + variance). A
// rate near 50% means the ROT/triple algebra is wrong, not the RCOT.
//
// Provider wiring (4 Ferrets, 4 distinct pre_file, 2 ports):
//   port P1: BOB.prov_send (server, pre_file "..._bob_send_rot1")
//            <-> ALICE.prov_recv (client, "..._alice_recv_rot1")  = ROT#1
//   port P2: ALICE.prov_send (server, "..._alice_send_rot2")
//            <-> BOB.prov_recv (client, "..._bob_recv_rot2")      = ROT#2
// The launcher pins each party to one GPU via CUDA_VISIBLE_DEVICES and clears
// all pre_ot_data_reg_* caches before the run.
//
// Usage: sudo ./run_cuot_bit_triple_test.sh [P1] [P2]
#include "gpu_mm/bit_triple.h"
#include "gpu_mm/cuot_provider.h"
#include "gpu_mm/ot_provider.h"

#include <emp-tool/emp-tool.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

constexpr int32_t kN = 100000;
constexpr double kMaxRate = 8.0;

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

// ALICE: prov_send on P2 (server), prov_recv on P1 (client).
int run_alice(int port1, int port2, int gpu) {
    using namespace gpu_mm;
    // prov_send = Ferret-SENDER (Delta owner), server on port2.
    auto prov_send = std::make_unique<CuotProvider>(
        OTParty::kSender, port2, gpu, "127.0.0.1", true,
        "data/pre_ot_data_reg_send_alice_rot2");
    // prov_recv = Ferret-RECEIVER, client on port1 (BOB is server there).
    auto prov_recv = std::make_unique<CuotProvider>(
        OTParty::kReceiver, port1, gpu, "127.0.0.1", true,
        "data/pre_ot_data_reg_recv_alice_rot1");
    BitTripleGen gen(*prov_send, *prov_recv, OTParty::kSender);

    std::vector<uint8_t> a(kN), b(kN), c(kN);
    EXPECT(gen.generate(a.data(), b.data(), c.data(), kN) == OTStatus::kOk,
           "generate");
    // send a,b,c to BOB for reconstruction check. Use prov_SEND's channel
    // (port2: ALICE-prov_send server <-> BOB-prov_recv client) — Bob reads
    // on HIS prov_recv (same port2). (Earlier used prov_recv->net_io() =
    // port1, but Bob's prov_recv is port2 -> mismatch -> deadlock. Both
    // parties must use the SAME port for the verification exchange; port2 is
    // the ALICE-send/BOB-recv direction, natural for ALICE->BOB verify.)
    emp::NetIO* io = prov_send->net_io();
    EXPECT(io != nullptr, "alice net_io null");
    io->send_data(a.data(), kN);
    io->send_data(b.data(), kN);
    io->send_data(c.data(), kN);
    io->flush();
    // Wait for Bob's ack before exiting, so Bob consumes the verification
    // data before this socket closes (avoids a close/recv race that hung Bob).
    uint8_t ack = 0;
    io->recv_data(&ack, 1);
    std::cout << "[alice] cuOT bit-triple sent n=" << kN << " gpu=" << gpu
              << std::endl;
    return 0;
}

// BOB: prov_send on P1 (server), prov_recv on P2 (client).
// CONSTRUCTION ORDER (deadlock fix): ALICE builds server(port2) then
// client(port1); BOB MUST build in the OPPOSITE order — client(port2) then
// server(port1). emp::NetIO server ctor blocks in accept() until a client
// connects, and the client ctor retries until the server is up. If both
// parties built server-first, each would block in accept() forever waiting
// for a client neither ever constructs (confirmed: alice printed nothing,
// hung indefinitely). With BOB client-first: ALICE server(port2) accepts
// BOB client(port2) [both unblock] -> BOB server(port1) accepts ALICE
// client(port1) [both unblock]. The (prov_send, prov_recv) ARGUMENT order to
// BitTripleGen is unchanged — only construction order swaps.
int run_bob(int port1, int port2, int gpu) {
    using namespace gpu_mm;
    auto prov_recv = std::make_unique<CuotProvider>(
        OTParty::kReceiver, port2, gpu, "127.0.0.1", true,
        "data/pre_ot_data_reg_recv_bob_rot2");
    auto prov_send = std::make_unique<CuotProvider>(
        OTParty::kSender, port1, gpu, "127.0.0.1", true,
        "data/pre_ot_data_reg_send_bob_rot1");
    BitTripleGen gen(*prov_send, *prov_recv, OTParty::kReceiver);

    std::vector<uint8_t> a(kN), b(kN), c(kN);
    EXPECT(gen.generate(a.data(), b.data(), c.data(), kN) == OTStatus::kOk,
           "generate");
    emp::NetIO* io = prov_recv->net_io();
    EXPECT(io != nullptr, "bob net_io null");
    std::vector<uint8_t> aA(kN), bA(kN), cA(kN);
    io->recv_data(aA.data(), kN);
    io->recv_data(bA.data(), kN);
    io->recv_data(cA.data(), kN);
    // Ack so Alice knows the verification data was consumed before she exits
    // and closes the socket (without this, Alice exiting first made her
    // ~CuotProvider/NetIO close port2 while Bob's prior recv_data could race
    // the socket teardown -> Bob spinning). The ack is a single byte.
    uint8_t ack = 1;
    io->send_data(&ack, 1);
    io->flush();

    int64_t bad = 0;
    for (int i = 0; i < kN; ++i) {
        uint8_t a_full = aA[i] ^ a[i];
        uint8_t b_full = bA[i] ^ b[i];
        uint8_t c_full = cA[i] ^ c[i];
        if (((a_full & b_full) & 1) != (c_full & 1)) {
            if (bad < 3)
                std::cerr << "FAIL triple @ " << i << " a=" << (int)a_full
                          << " b=" << (int)b_full << " c=" << (int)c_full
                          << " want=" << (int)(a_full & b_full) << "\n";
            ++bad;
        }
    }
    double rate = 100.0 * bad / kN;
    std::cout << "[bob] cuOT bit-triple bad=" << bad << "/" << kN << " ("
              << rate << "%) gpu=" << gpu << std::endl;
    if (rate > kMaxRate) {
        std::cerr << "FAIL cuOT bit-triple rate " << rate << "% > " << kMaxRate
                  << "% (ROT/triple algebra buggy)\n";
        return 1;
    }
    std::cout << "cuOT bit-triple test PASSED (within ~4% compounded RCOT floor)"
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: cuot_bit_triple_test <party 1|2> <port1> <port2> [gpu]\n"
                  << "  party 1 = ALICE, 2 = BOB\n"
                  << "  port1 = ROT#1 (BOB send <-> ALICE recv)\n"
                  << "  port2 = ROT#2 (ALICE send <-> BOB recv)\n"
                  << "  sudo ./run_cuot_bit_triple_test.sh [port1] [port2]\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port1 = std::atoi(argv[2]);
    int port2 = std::atoi(argv[3]);
    int gpu = (argc > 4) ? std::atoi(argv[4]) : 0;
    if (party == 1) return run_alice(port1, port2, gpu);
    if (party == 2) return run_bob(port1, port2, gpu);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
