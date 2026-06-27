// CPU baseline profiling harness (Task M0-T2).
//
// Goal: measure the wall-clock breakdown of the CPU baseline modulus-conversion
// + element-wise primitives so that later GPU/cuOT/Phantom/CUDA replacements
// can be compared against the SAME measuring stick (the gpu_mm profiler from
// M0-T1).
//
// Scope rules followed (CLAUDE.md / Task Size Rule):
//  - This is a NEW test file; no protocol source files are modified.
//  - Only public API of the existing CPU baseline is used (NetIO, OTPack,
//    AuxProtocols, Truncation, CheetahLinear) — exactly the same calls the
//    existing tests make, in the same order.
//  - No protocol equations, OT, or HE logic is touched. Default CPU path only.
//  - No CUDA/cuOT/Phantom.
//
// Primitives profiled (each mirrors an existing tests/*.cpp):
//   ring_to_field   — Truncation::ring_to_field            (tests/ring_to_field_test.cpp)
//   field_to_ring   — Truncation::field_to_ring            (tests/field_to_ring_test.cpp)
//   ring_extension  — Truncation::ring_to_ring128          (tests/ring_extension_test.cpp)
//   bole            — CheetahLinear::BOLE                  (tests/bole_test.cpp)
//
// Profiling protocol per primitive:
//   phase 0  warm-up round  — profiler DISABLED (lets Ferret/OTPack caches warm,
//                             avoids one-time setup polluting the breakdown)
//   phase 1  measured rounds — Profiler::Reset()+Enable(true), REPEAT measured
//                             iterations each wrapping the per-region TimerScopes.
//
// Regions measured (per iteration, per party, per primitive). A fresh
// OTPack/AuxProtocols/Truncation (and CheetahLinear for bole) is built per
// iteration, exactly like the original tests, so setup cost is visible:
//   *.otpack_setup   — OTPack construction (base OT + Ferret bootstrap)
//   *.aux_setup      — AuxProtocols + Truncation (+ CheetahLinear for bole)
//   *.<primitive>    — the primitive under test
//   *.total          — full iteration (sanity check vs sum of above)
//
// Usage (two-party, run from repo root so ./data exists for Ferret):
//   ./build/bin/cpu_baseline_profile 1 [bw] & ./build/bin/cpu_baseline_profile 2 [bw]
//   bw ∈ {16,32,40,48,56,60,64}   (default 32)
#include "BuildingBlocks/truncation.h"
#include "cheetah/cheetah-api.h"
#include "gpu_mm/profiler.h"

#include <iostream>
#include <vector>

using namespace sci;
using namespace std;

int party, port = 32000;
sci::NetIO *io = nullptr;
sci::OTPack<sci::NetIO> *otpack = nullptr;
AuxProtocols *aux_protocol = nullptr;
Truncation *truncation_protocol = nullptr;
gemini::CheetahLinear *cheetahLinear_protocol = nullptr;

// Same prime table the existing tests use (ring_to_field_test.cpp).
const std::map<int32_t, uint64_t> prime_mod{
    {16, 65521ULL},
    {32, 4293918721ULL},
    {40, 1099511480321ULL},
    {48, 281474976694273ULL},
    {56, 72057594037641217ULL},
    {60, 1152921504606830593ULL},
    {64, 18446744073709551557ULL},
};

PRG128 prg;

// ---------------------------------------------------------------------------
// ring_to_field  (mirrors tests/ring_to_field_test.cpp)
// ---------------------------------------------------------------------------
static void run_ring_to_field(const uint32_t N, const int32_t bw_in,
                              const uint64_t prime) {
    const uint64_t maskA = (bw_in == 64 ? -1ULL : ((1ULL << bw_in) - 1));
    vector<uint64_t> inA(N), outB(N), inA_bob;
    vector<int64_t> data_value;
    const uint64_t max_value = min((uint64_t)1 << (bw_in - 2), prime / 4);

    if (party == sci::ALICE) {
        data_value.resize(N);
        inA_bob.resize(N);
        prg.random_data(data_value.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++)
            data_value[i] = (data_value[i] % (2 * max_value)) - max_value;
        io->recv_data(inA_bob.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) inA[i] = (data_value[i] - inA_bob[i]) & maskA;
    } else {
        prg.random_data(inA.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) inA[i] &= maskA;
        io->send_data(inA.data(), N * sizeof(uint64_t));
    }
    io->sync();

    gpu_mm::TimerScope t_total("rtf.total");
    {
        gpu_mm::TimerScope t("rtf.otpack_setup");
        otpack = new OTPack<sci::NetIO>(io, party);
    }
    {
        gpu_mm::TimerScope t("rtf.aux_setup");
        aux_protocol = new AuxProtocols(party, io, otpack);
        truncation_protocol = new Truncation(party, io, otpack, aux_protocol);
    }
    {
        gpu_mm::TimerScope t("rtf.ring_to_field");
        truncation_protocol->ring_to_field(N, inA.data(), outB.data(), bw_in, prime);
    }
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    truncation_protocol = nullptr;
    aux_protocol = nullptr;
    otpack = nullptr;
}

// ---------------------------------------------------------------------------
// field_to_ring  (mirrors tests/field_to_ring_test.cpp)
// ---------------------------------------------------------------------------
static void run_field_to_ring(const uint32_t N, const uint64_t prime,
                              const int32_t bw_out) {
    vector<uint64_t> inA(N), outB(N), inA_bob;
    vector<int64_t> data_value;
    const int64_t value_mod = prime / 2;
    const int64_t big_number = prime / 4;

    if (party == sci::ALICE) {
        data_value.resize(N);
        inA_bob.resize(N);
        prg.random_data(data_value.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++)
            data_value[i] = abs(data_value[i]) % value_mod - big_number;
        io->recv_data(inA_bob.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) inA[i] = (data_value[i] - inA_bob[i]) % prime;
    } else {
        prg.random_data(inA.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) inA[i] %= prime;
        io->send_data(inA.data(), N * sizeof(uint64_t));
    }
    io->sync();

    gpu_mm::TimerScope t_total("ftr.total");
    {
        gpu_mm::TimerScope t("ftr.otpack_setup");
        otpack = new OTPack<sci::NetIO>(io, party);
    }
    {
        gpu_mm::TimerScope t("ftr.aux_setup");
        aux_protocol = new AuxProtocols(party, io, otpack);
        truncation_protocol = new Truncation(party, io, otpack, aux_protocol);
    }
    {
        gpu_mm::TimerScope t("ftr.field_to_ring");
        truncation_protocol->field_to_ring(N, inA.data(), outB.data(), prime, bw_out);
    }
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    truncation_protocol = nullptr;
    aux_protocol = nullptr;
    otpack = nullptr;
}

// ---------------------------------------------------------------------------
// ring_extension  (mirrors tests/ring_extension_test.cpp)
//   base ring 2^64 -> target ring 2^(64+shift), uses ring_to_ring128.
// ---------------------------------------------------------------------------
static void run_ring_extension(const uint32_t N, const int32_t base_bw,
                               const int32_t bw_out) {
    const uint64_t maskA = (base_bw == 64 ? -1ULL : ((1ULL << base_bw) - 1));
    const uint64_t value_mask = (1ULL << (base_bw - 1)) - 1;
    const uint64_t big_number = (1ULL << (base_bw - 2));

    vector<uint64_t> inA(N), inA_bob;
    vector<__uint128_t> outB(N);
    vector<int64_t> data_value;

    if (party == sci::ALICE) {
        data_value.resize(N);
        inA_bob.resize(N);
        prg.random_data(data_value.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++)
            data_value[i] = (data_value[i] & value_mask) - big_number;
        io->recv_data(inA_bob.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) inA[i] = (data_value[i] - inA_bob[i]) & maskA;
    } else {
        prg.random_data(inA.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) inA[i] &= maskA;
        io->send_data(inA.data(), N * sizeof(uint64_t));
    }
    io->sync();

    gpu_mm::TimerScope t_total("rext.total");
    {
        gpu_mm::TimerScope t("rext.otpack_setup");
        otpack = new OTPack<sci::NetIO>(io, party);
    }
    {
        gpu_mm::TimerScope t("rext.aux_setup");
        aux_protocol = new AuxProtocols(party, io, otpack);
        truncation_protocol = new Truncation(party, io, otpack, aux_protocol);
    }
    {
        gpu_mm::TimerScope t("rext.ring_to_ring128");
        truncation_protocol->ring_to_ring128(N, inA.data(), outB.data(), base_bw, bw_out);
    }
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    truncation_protocol = nullptr;
    aux_protocol = nullptr;
    otpack = nullptr;
}

// ---------------------------------------------------------------------------
// bole  (mirrors tests/bole_test.cpp BOLE_test: both_ss=true, no truncate)
//   two shared-share vectors -> ring_to_field x2 -> BOLE -> field_to_ring
// ---------------------------------------------------------------------------
static void run_bole(const uint32_t N, const int32_t bw_in,
                     const uint64_t prime) {
    const uint64_t maskA = (bw_in == 64 ? -1ULL : ((1ULL << bw_in) - 1));
    const uint64_t max_value = 1ULL << 12;

    vector<uint64_t> input_x(N), client_input_x(N), input_y(N),
        client_input_y(N), output_z(N);
    vector<int64_t> x_value(N), y_value(N);

    gemini::CheetahLinear::BNMeta meta;
    meta.target_base_mod = prime;
    meta.is_shared_input = true;
    meta.vec_shape = gemini::TensorShape({N});
    gemini::Tensor<uint64_t> field_x, field_y, field_z;
    field_x.Reshape(meta.vec_shape);
    field_y.Reshape(meta.vec_shape);
    field_z.Reshape(meta.vec_shape);

    if (party == sci::ALICE) {
        prg.random_data(x_value.data(), N * sizeof(uint64_t));
        for (auto &p : x_value) p = ((uint64_t)p % (2 * max_value)) - max_value;
        io->recv_data(client_input_x.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++)
            input_x[i] = (x_value[i] - client_input_x[i]) & maskA;

        prg.random_data(y_value.data(), N * sizeof(uint64_t));
        for (auto &p : y_value) p = ((uint64_t)p % (2 * max_value)) - max_value;
        io->recv_data(client_input_y.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++)
            input_y[i] = (y_value[i] - client_input_y[i]) & maskA;
    } else {
        prg.random_data(input_x.data(), N * sizeof(uint64_t));
        for (auto &p : input_x) p &= maskA;
        io->send_data(input_x.data(), N * sizeof(uint64_t));
        prg.random_data(input_y.data(), N * sizeof(uint64_t));
        for (auto &p : input_y) p &= maskA;
        io->send_data(input_y.data(), N * sizeof(uint64_t));
    }
    io->sync();

    gpu_mm::TimerScope t_total("bole.total");
    {
        gpu_mm::TimerScope t("bole.otpack_setup");
        otpack = new OTPack<sci::NetIO>(io, party);
    }
    {
        gpu_mm::TimerScope t("bole.aux_setup");
        aux_protocol = new AuxProtocols(party, io, otpack);
        truncation_protocol = new Truncation(party, io, otpack, aux_protocol);
        cheetahLinear_protocol =
            new gemini::CheetahLinear(party, io, 1ULL << bw_in, 1);
    }
    {
        gpu_mm::TimerScope t("bole.ring_to_field_x");
        truncation_protocol->ring_to_field(N, input_x.data(), field_x.data(), bw_in, prime);
        truncation_protocol->ring_to_field(N, input_y.data(), field_y.data(), bw_in, prime);
    }
    {
        gpu_mm::TimerScope t("bole.BOLE");
        cheetahLinear_protocol->BOLE(field_x, field_y, meta, field_z);
    }
    {
        gpu_mm::TimerScope t("bole.field_to_ring");
        truncation_protocol->field_to_ring(N, field_z.data(), output_z.data(), prime, bw_in);
    }
    io->sync();
    delete cheetahLinear_protocol;
    cheetahLinear_protocol = nullptr;
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    truncation_protocol = nullptr;
    aux_protocol = nullptr;
    otpack = nullptr;
}

// Generic driver: warm-up (profiler off) then `repeats` measured iterations
// (profiler on), printing the summary for the side that ran them.
template <typename Fn>
static void profile_primitive(const char *label, uint32_t warm_up,
                              uint32_t repeats, Fn fn) {
    if (party == sci::ALICE) cout << "--- " << label << " ---" << endl;

    gpu_mm::Profiler::Enable(false);
    for (uint32_t i = 0; i < warm_up; ++i) fn();

    gpu_mm::Profiler::Reset();
    gpu_mm::Profiler::Enable(true);
    for (uint32_t i = 0; i < repeats; ++i) fn();
    gpu_mm::Profiler::Enable(false);

    cout << "=== [" << label << "] profiler summary (party "
         << (party == sci::ALICE ? "ALICE" : "BOB") << ") ===" << endl;
    gpu_mm::Profiler::PrintSummary(cout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Usage: cpu_baseline_profile <party 1|2> [bw 16|32|40|48|56|60|64]"
             << endl;
        return 0;
    }
    party = atoi(argv[1]);
    const int32_t bitwidth = (argc >= 3) ? std::atoi(argv[2]) : 32;
    if (!prime_mod.count(bitwidth)) {
        cout << "current support bitwidth :";
        for (const auto &p : prime_mod) cout << " " << p.first;
        cout << endl;
        return 0;
    }
    const uint64_t prime = prime_mod.at(bitwidth);

    // Smaller N than the original tests to keep the multi-primitive sweep short;
    // large enough that OT work (not noise) dominates. ring_extension uses a
    // separate, larger N like its original test.
    const uint32_t N = 1'000'000;
    const uint32_t N_rext = 1'000'000;
    const uint32_t warm_up = 1;
    const uint32_t repeats = 3;
    const int32_t base_bw = 64;
    const int32_t shift_bw = 16;  // 64 -> 80

    io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);

    if (party == sci::ALICE) {
        cout << "CPU baseline profile: bw=" << bitwidth << ", N=" << N
             << " (rext N=" << N_rext << "), warm=" << warm_up
             << ", repeats=" << repeats << endl;
    }

    profile_primitive("ring_to_field", warm_up, repeats, [&]() {
        run_ring_to_field(N, bitwidth, prime);
    });
    profile_primitive("field_to_ring", warm_up, repeats, [&]() {
        run_field_to_ring(N, prime, bitwidth);
    });
    profile_primitive("ring_extension", warm_up, repeats, [&]() {
        run_ring_extension(N_rext, base_bw, base_bw + shift_bw);
    });
    profile_primitive("bole", warm_up, repeats, [&]() {
        run_bole(N, bitwidth, prime);
    });

    io->sync();
    delete io;
    return 0;
}
