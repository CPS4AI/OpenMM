// OT provider smoke / standalone correlation test (Task M1-T2).
//
// Two-party. Verifies EmpOTProvider wraps the existing emp/SCI SilentOT behind
// the gpu_mm::OTProvider interface and that the COT correlation holds for BOTH
// modulus flavors M&M uses:
//   - COT over Z_{2^l}     (ring, send_cot/recv_cot)
//   - COT over Z_p         (field, send_cot_prime/recv_cot_prime)
//
// Correlation check (matching sci::SilentOT semantics):
//   Sender gets random x (data0); receiver with choice bit b gets:
//       x            if b == 0
//       x + corr     if b == 1      (mod modulus)
//   => (recv[i] - data0[i]) mod m == (b[i] ? corr[i] : 0)  for all i.
// To check this the sender sends its random `x` back to the receiver over the
// provider's NetIO after the OT (x is secret in real use; here it is exposed
// only for verification).
//
// Usage (two-party, run from repo root so ./data exists for Ferret):
//   ./build/bin/ot_provider_smoke_test 1 <port> & \
//   ./build/bin/ot_provider_smoke_test 2 <port>
//   party 1 = sender (ALICE), party 2 = receiver (BOB).
#include "gpu_mm/emp_ot_provider.h"
#include "gpu_mm/ot_provider.h"
#include "utils/net_io_channel.h"  // sci::NetIO send/recv for verification

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

uint64_t sub_mod_ring(uint64_t a, uint64_t b, int l) {
    uint64_t m = (l >= 64) ? ~0ULL : ((1ULL << l) - 1);
    return (a - b) & m;
}
uint64_t sub_mod_field(uint64_t a, uint64_t b, uint64_t p) {
    a %= p;
    b %= p;
    return a >= b ? a - b : p - (b - a);
}

int run_sender(int port) {
    using namespace gpu_mm;
    EmpOTProvider prov(OTParty::kSender, port);
    sci::NetIO* io = prov.net_io();

    const int n = 1000;
    const int l = 32;
    std::vector<uint64_t> corr(n), data0(n);
    for (int i = 0; i < n; ++i) corr[i] = (uint64_t)(i * 7 + 3);

    if (prov.send_cot(data0.data(), corr.data(), n, l) != OTStatus::kOk) {
        std::cerr << "FAIL send_cot\n";
        return 1;
    }
    // send x = data0 back to receiver for verification
    io->send_data(data0.data(), n * sizeof(uint64_t));
    io->flush();

    const uint64_t p = 4293918721ULL;
    std::vector<uint64_t> corr_p(n), data0_p(n);
    for (int i = 0; i < n; ++i) corr_p[i] = (uint64_t)(i + 1) % p;
    if (prov.send_cot_prime(data0_p.data(), corr_p.data(), n, p) !=
        OTStatus::kOk) {
        std::cerr << "FAIL send_cot_prime\n";
        return 1;
    }
    io->send_data(data0_p.data(), n * sizeof(uint64_t));
    io->flush();

    std::cout << "[sender] COT + COT' sent, n=" << n << ", l=" << l
              << ", p=" << p << std::endl;
    return 0;
}

int run_receiver(int port) {
    using namespace gpu_mm;
    EmpOTProvider prov(OTParty::kReceiver, port);
    sci::NetIO* io = prov.net_io();

    const int n = 1000;
    const int l = 32;
    // std::vector<bool> is space-optimized and has no contiguous bool*; use a
    // raw bool array so we can pass &b[0] as const bool* to recv_cot.
    std::unique_ptr<bool[]> b(new bool[n]);
    for (int i = 0; i < n; ++i) b[i] = (i % 2 == 0);
    std::vector<uint64_t> data(n), x(n);
    if (prov.recv_cot(data.data(), b.get(), n, l) != OTStatus::kOk) {
        std::cerr << "FAIL recv_cot\n";
        return 1;
    }
    io->recv_data(x.data(), n * sizeof(uint64_t));

    // ring COT correlation check: (data[i] - x[i]) mod 2^l == b[i] ? corr[i] : 0
    int fails = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t diff = sub_mod_ring(data[i], x[i], l);
        // corr[i] on the sender side = i*7+3; replicate here (public in this test).
        uint64_t corr_i = (uint64_t)(i * 7 + 3);
        uint64_t want = b[i] ? (corr_i & ((1ULL << l) - 1)) : 0;
        if (diff != want) {
            if (fails < 5)
                std::cerr << "FAIL ring cot @ " << i << " diff=" << diff
                          << " want=" << want << " b=" << b[i] << "\n";
            ++fails;
        }
    }

    const uint64_t p = 4293918721ULL;
    std::vector<uint64_t> data_p(n), x_p(n);
    if (prov.recv_cot_prime(data_p.data(), b.get(), n, p) != OTStatus::kOk) {
        std::cerr << "FAIL recv_cot_prime\n";
        return 1;
    }
    io->recv_data(x_p.data(), n * sizeof(uint64_t));
    for (int i = 0; i < n; ++i) {
        uint64_t diff = sub_mod_field(data_p[i], x_p[i], p);
        uint64_t corr_i = (uint64_t)(i + 1) % p;
        uint64_t want = b[i] ? corr_i : 0;
        if (diff != want) {
            if (fails < 5)
                std::cerr << "FAIL field cot @ " << i << " diff=" << diff
                          << " want=" << want << " b=" << b[i] << "\n";
            ++fails;
        }
    }

    if (fails != 0) {
        std::cerr << "OT correlation check FAILED (" << fails << " mismatches)\n";
        return 1;
    }
    std::cout << "[receiver] COT + COT' correlation OK, n=" << n << ", l=" << l
              << ", p=" << p << std::endl;
    std::cout << "OT provider smoke test PASSED" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ot_provider_smoke_test <party 1|2> <port>\n"
                  << "  party 1 = sender (ALICE), 2 = receiver (BOB)\n"
                  << "Run both concurrently from repo root.\n";
        return 0;
    }
    int party = std::atoi(argv[1]);
    int port = std::atoi(argv[2]);
    if (party == 1) return run_sender(port);
    if (party == 2) return run_receiver(port);
    std::cerr << "party must be 1 or 2\n";
    return 1;
}
