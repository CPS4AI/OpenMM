// cuOT standalone correlation test (Task M2-T2).
//
// Two-party, same machine, two GPUs (sender->GPU0, receiver->GPU1 over
// CUDA_VISIBLE_DEVICES, each exposed as logical dev 0). Over TCP (emp::NetIO).
//
// Tests TWO correlations, both over GF(2^128):
//
// 1. RCOT (cuOT native): sender holds random x_i + global Delta (LSB=1);
//    receiver holds r_i = x_i ^ (lsb(r_i) * Delta) — the LSB of the receiver's
//    OWN block is the implicit (random) choice bit.
//    Check (mirrors cuOT test_rcot): r[i] ^= ch[getLSB(r[i])]; r == x.
//
// 2. Chosen COT (paper §III-D, "Generating COTs from RCOTs"): receiver chooses
//    b; the diff vector d_i = lsb(r_i) ^ b_i is sent to the sender over NetIO;
//    sender adjusts x'_i = x_i ^ (d_i ? Delta : 0). Correlation:
//    r_i == x'_i ^ (b_i * Delta). d is a one-time pad (lsb(r) is the receiver's
//    private uniform bit, unknown to the sender) -> sender learns nothing about
//    b; receiver learns nothing about Delta online. This is the
//    receiver-chosen-choice-bit path M&M actually consumes (Truncation::
//    ring_to_field -> iknp_straight->send_cot), done here at block level.
//
// Both correlations are checked with the sender sending Delta + its blocks
// back to the receiver for verification (Delta/x are secret in real use;
// exposed here only for the test, exactly like ot_provider_smoke_test).
//
// Batch sizes: {1, 2, 17, 1024, 4096} (per M2-T2 acceptance). Both roles are
// exercised by running the binary twice (party 1 and party 2). Loud failure:
// any mismatch prints FAIL + the round and exits 1.
//
// Determinism: Ferret caches ./data/pre_ot_data_reg_* after the first run, so
// re-runs reuse state; Delta is random per construction (cuOT has no
// seed-injection API). The CORRELATION checks are deterministic regardless.
//
// Usage (binary takes <party> <port> [gpu]; host is 127.0.0.1 inside the
// provider; run from repo root; bind needs sudo on this box):
//   sudo CUDA_VISIBLE_DEVICES=0 ./build/bin/cuot_correlation_test 1 <port> 0 & \
//   sudo CUDA_VISIBLE_DEVICES=1 ./build/bin/cuot_correlation_test 2 <port> 0
// or just:  sudo ./run_cuot_test.sh [port]
#include "gpu_mm/cuot_provider.h"

#include <emp-tool/emp-tool.h>  // emp::NetIO, block, getLSB, cmpBlock, zero_block

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr int64_t kBlockBytes = 16;
const int64_t kSizes[] = {1LL, 2LL, 17LL, 1024LL, 4096LL};

// xorshift PRNG for the receiver's chosen bits (deterministic, like
// share_tensor_backend_test's Rng). No Math.random.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    int operator()() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (int)(s & 1);
    }
};

#define EXPECT(cond, msg)                                          \
    do {                                                           \
        if (!(cond)) {                                             \
            std::cerr << "FAIL " << (msg) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;            \
            return 1;                                              \
        }                                                          \
    } while (0)

// ---- RCOT round (both sides) ----
// Sender: rcot -> x; send Delta(once, up front) + x buffer.
// Receiver: rcot -> r; recv Delta; r[i] ^= ch[getLSB(r[i])]; check r == x.
int rcot_round_sender(gpu_mm::CuotProvider& prov, emp::NetIO* io, int64_t n) {
    std::vector<uint8_t> buf(n * kBlockBytes);
    EXPECT(prov.rcot_blocks(buf.data(), n) == gpu_mm::OTStatus::kOk,
           "rcot_blocks sender");
    io->send_data(&n, sizeof(int64_t));
    io->send_data(buf.data(), n * kBlockBytes);
    io->flush();
    return 0;
}
int rcot_round_receiver(gpu_mm::CuotProvider& prov, emp::NetIO* io,
                        emp::block ch[2], int64_t n, int& fails) {
    std::vector<uint8_t> r(n * kBlockBytes);
    EXPECT(prov.rcot_blocks(r.data(), n) == gpu_mm::OTStatus::kOk,
           "rcot_blocks receiver");
    int64_t got_n = 0;
    std::vector<uint8_t> b0(n * kBlockBytes);
    io->recv_data(&got_n, sizeof(int64_t));
    EXPECT(got_n == n, "rcot recv n mismatch");
    io->recv_data(b0.data(), n * kBlockBytes);
    emp::block* r_blks = reinterpret_cast<emp::block*>(r.data());
    emp::block* b0_blks = reinterpret_cast<emp::block*>(b0.data());
    for (int64_t i = 0; i < n; ++i) r_blks[i] = r_blks[i] ^ ch[emp::getLSB(r_blks[i])];
    if (!emp::cmpBlock(r_blks, b0_blks, (int)n)) {
        std::cerr << "FAIL RCOT correlation mismatch at n=" << n << std::endl;
        ++fails;
    } else {
        std::cout << "[receiver] RCOT PASSED for n=" << n << std::endl;
    }
    return 0;
}

// ---- Chosen COT round (paper §III-D diff-vector method) ----
// Sender: send_cot_blocks -> x' (already adjusted by recv'd diff); send x' back.
// Receiver: recv_cot_blocks(b) -> r; check r == x' ^ (b * Delta).
int chosen_cot_round_sender(gpu_mm::CuotProvider& prov, emp::NetIO* io, int64_t n) {
    std::vector<uint8_t> buf(n * kBlockBytes);
    EXPECT(prov.send_cot_blocks(buf.data(), n) == gpu_mm::OTStatus::kOk,
           "send_cot_blocks sender");
    io->send_data(&n, sizeof(int64_t));
    io->send_data(buf.data(), n * kBlockBytes);
    io->flush();
    return 0;
}
int chosen_cot_round_receiver(gpu_mm::CuotProvider& prov, emp::NetIO* io,
                              emp::block delta, Rng& rng, int64_t n, int& fails) {
    // receiver-chosen choice bits
    std::vector<uint8_t> bbuf((size_t)n);
    for (int64_t i = 0; i < n; ++i) bbuf[i] = (uint8_t)rng();
    const bool* b = reinterpret_cast<const bool*>(bbuf.data());

    std::vector<uint8_t> r(n * kBlockBytes);
    EXPECT(prov.recv_cot_blocks(r.data(), b, n) == gpu_mm::OTStatus::kOk,
           "recv_cot_blocks receiver");

    int64_t got_n = 0;
    std::vector<uint8_t> xp(n * kBlockBytes);
    io->recv_data(&got_n, sizeof(int64_t));
    EXPECT(got_n == n, "chosen cot recv n mismatch");
    io->recv_data(xp.data(), n * kBlockBytes);

    // check r[i] == x'[i] ^ (b[i] * Delta)
    emp::block* r_blks = reinterpret_cast<emp::block*>(r.data());
    emp::block* xp_blks = reinterpret_cast<emp::block*>(xp.data());
    int local_fail = 0;
    for (int64_t i = 0; i < n; ++i) {
        emp::block want = bbuf[i] ? (xp_blks[i] ^ delta) : xp_blks[i];
        if (!emp::cmpBlock(&r_blks[i], &want, 1)) {
            if (local_fail < 3)
                std::cerr << "FAIL chosen COT @ " << i << " b=" << (int)bbuf[i]
                          << std::endl;
            ++local_fail;
        }
    }
    if (local_fail) { std::cerr << "FAIL chosen COT mismatch at n=" << n
                               << " (" << local_fail << "/" << n << ")\n"; ++fails; }
    else { std::cout << "[receiver] chosen COT PASSED for n=" << n << std::endl; }
    return 0;
}

int run_sender(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kSender, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "sender net_io null");

    // Send Delta ONCE up front (receiver needs it for both RCOT and chosen checks).
    uint8_t delta_bytes[16];
    EXPECT(prov.delta_block(delta_bytes) == OTStatus::kOk, "delta_block");
    io->send_data(delta_bytes, 16);
    io->flush();

    Rng rng(0xabcdef0123456789ULL);  // sender doesn't choose bits; rng unused but symmetrical
    (void)rng;
    for (int64_t n : kSizes) {
        EXPECT(rcot_round_sender(prov, io, n) == 0, "rcot sender round");
    }
    for (int64_t n : kSizes) {
        EXPECT(chosen_cot_round_sender(prov, io, n) == 0, "chosen cot sender round");
    }
    std::cout << "[sender] RCOT + chosen COT sent for n in {1,2,17,1024,4096}, gpu="
              << gpu << std::endl;
    return 0;
}

int run_receiver(int port, int gpu) {
    using namespace gpu_mm;
    CuotProvider prov(OTParty::kReceiver, port, gpu);
    emp::NetIO* io = prov.net_io();
    EXPECT(io != nullptr, "receiver net_io null");

    // Receive Delta once.
    uint8_t delta_bytes[16];
    io->recv_data(delta_bytes, 16);
    emp::block delta;
    std::memcpy(&delta, delta_bytes, 16);
    emp::block ch[2] = {emp::zero_block, delta};

    int fails = 0;
    Rng rng(0xabcdef0123456789ULL);
    for (int64_t n : kSizes) {
        EXPECT(rcot_round_receiver(prov, io, ch, n, fails) == 0, "rcot receiver round");
    }
    for (int64_t n : kSizes) {
        EXPECT(chosen_cot_round_receiver(prov, io, delta, rng, n, fails) == 0,
               "chosen cot receiver round");
    }

    if (fails != 0) {
        std::cerr << "cuOT correlation FAILED (" << fails << " rounds)\n";
        return 1;
    }
    std::cout << "cuOT correlation test PASSED (rcot+chosen, gpu=" << gpu << ")"
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: cuot_correlation_test <party 1|2> <port> [gpu]\n"
                  << "  party 1 = sender (ALICE)\n"
                  << "  party 2 = receiver (BOB)\n"
                  << "  host is 127.0.0.1 (hardcoded in CuotProvider)\n"
                  << "  Pin each party to one physical GPU via CUDA_VISIBLE_DEVICES.\n"
                  << "Run both concurrently from repo root (needs sudo for bind):\n"
                  << "  sudo ./run_cuot_test.sh [port]\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    int gpu = (argc > 3) ? std::atoi(argv[3]) : 0;  // logical device (use 0 w/ CVD)
    if (party == 1) return run_sender(port, gpu);
    if (party == 2) return run_receiver(port, gpu);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
