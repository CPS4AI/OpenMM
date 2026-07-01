// Micro-benchmark: GPU MITCCRH (cuot_kernels.cu) vs CPU NaryMitccrh
// (kkot_leaf_cmp.h) — the OPT-1 risk gate (plan risk #1).
//
// Runs the leaf garbled-table pad construction (logN=4, N=16) over n=800000
// outputs with BOTH the GPU kernel (leaf_pad_sender) and the CPU loop
// (NaryMitccrh hash_exp), on identical inputs (random rm0, rm1=rm0^Delta,
// fixed hash_in0/hash_in1). Checks the two produce IDENTICAL pad blocks
// (correctness) and times both (GPU incl. the kernel launch only; CPU incl.
// the serial per-output AES-NI loop).
//
// If GPU wins (us < CPU us) => OPT-1 full rewiring is worth it. If GPU loses
// (CPU AES-NI per-op beats GPU software T-table despite parallelism) =>
// OPT-1 reverts; only OPT-2 (diff-vector) proceeds.
//
// This needs NO cuOT Ferret / NO network — it's pure AES timing with random
// blocks. Single process, single GPU.
//
// Usage: ./build/bin/cuot_mitccrh_bench [n]
#include "gpu_mm/kkot_leaf_cmp.h"   // NaryMitccrh (CPU), but it's in gpu_mm + needs emp headers
// kkot_leaf_cmp.h includes cuot_provider.h -> emp-tool. For a no-network
// bench we still need the emp headers for emp::block/AES. Fine (no NetIO used).
#include "gpu_mm/cuot_provider.h"
#include "gpu_define.h"   // cuOT `blk` (16-byte union), for the launcher sig

#include <emp-tool/emp-tool.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// The GPU kernels live in cuot_kernels.cu (compiled into this target as
// LANGUAGE CUDA). cuOT `blk` is a 16-byte union; emp::block is __m128i —
// same size, trivially-copyable, so reinterpret_cast between them is safe.
namespace gpu_mm::gpu_kern {
template <int logN>
void launch_leaf_pad_sender(const blk* rm0, const blk* rm1,
                            const blk* hash_in0, const blk* hash_in1,
                            blk* pad, int n);
void launch_leaf_pad_recv(const blk* rm, blk* pad_choice,
                          const uint8_t* choice, int n, int logN);
void launch_dbg_single_aes(const blk* key, const blk* inb, blk* out);
}  // namespace gpu_mm::gpu_kern

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

int run(int n) {
    constexpr int logN = 4;
    constexpr int N = 1 << logN;       // 16
    constexpr int nhash = (1 << logN) - 1;  // 15

    // Random rm0 (n*logN base blocks) + Delta + rm1 = rm0 ^ Delta.
    std::vector<emp::block> rm0_h((size_t)n * logN);
    Rng rng(0x9e3779b97f4a7c15ULL);
    for (auto& b : rm0_h) {
        uint64_t* p = reinterpret_cast<uint64_t*>(&b);
        p[0] = rng.next(); p[1] = rng.next();
    }
    emp::block delta;
    { uint64_t* p = reinterpret_cast<uint64_t*>(&delta); p[0]=rng.next(); p[1]=rng.next(); }
    std::vector<emp::block> rm1_h((size_t)n * logN);
    for (size_t i = 0; i < rm0_h.size(); ++i)
        rm1_h[i] = rm0_h[i] ^ delta;

    // hash_in0/hash_in1 (the (2^logN-1) fixed plaintext blocks). Mirror
    // kkot_leaf_cmp.h:330-337 layout: idx runs over (x,y) tree, in0=makeBlock(y,0),
    // in1=makeBlock(2^x+y,0).
    std::vector<emp::block> hin0_h((size_t)nhash), hin1_h((size_t)nhash);
    {
        int idx = 0;
        for (int x = 0; x < logN; ++x) {
            for (int y = 0; y < (1 << x); ++y) {
                hin0_h[idx] = emp::makeBlock((uint64_t)y, 0);
                hin1_h[idx] = emp::makeBlock((uint64_t)((1 << x) + y), 0);
                ++idx;
            }
        }
    }

    // ---------- CPU path (NaryMitccrh, mirrors kkot_leaf_cmp.h:342-362) ----------
    // NaryMitccrh lives in kkot_leaf_cmp.h's anonymous namespace, so it is
    // TU-local: by including that header HERE, this TU gets its OWN identical
    // copy. We reach it via `using namespace gpu_mm;` then the unqualified
    // name resolves to this TU's anon-namespace copy.
    std::vector<emp::block> pad_cpu((size_t)n * N);
    auto t0 = std::chrono::high_resolution_clock::now();
    {
        using namespace gpu_mm;
        NaryMitccrh mitccrh;
        std::vector<emp::block> hash_out((size_t)2 * nhash);
        for (int i = 0; i < n; ++i) {
            mitccrh.renew_ks(rm0_h.data() + (size_t)i * logN, logN);
            mitccrh.hash_exp(hash_out.data(), hin0_h.data(), logN);
            mitccrh.renew_ks(rm1_h.data() + (size_t)i * logN, logN);
            mitccrh.hash_exp(hash_out.data() + nhash, hin1_h.data(), logN);
            for (int k = 0; k < N; ++k) {
                emp::block p = emp::zero_block;
                int idx = 0;
                for (int s = 0; s < logN; ++s) {
                    int pref = k & ((1 << s) - 1);
                    if ((k & (1 << s)) == 0) p = p ^ hash_out[idx + pref];
                    else p = p ^ hash_out[nhash + idx + pref];
                    idx += 1 << s;
                }
                pad_cpu[(size_t)i * N + k] = p;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    // ---------- GPU path (leaf_pad_sender kernel) ----------
    // Upload rm0, rm1, hash_in0/1 to device; alloc pad_gpu; launch; download.
    // Use cuOT `blk` (16-byte union) for the device buffers since the kernel
    // signature takes blk*. emp::block/__m128i is also 16 bytes — copy as
    // raw bytes, cast the pointer for the launch.
    blk *rm0_d, *rm1_d, *hin0_d, *hin1_d, *pad_d;
    cudaMalloc(&rm0_d, n * logN * sizeof(blk));
    cudaMalloc(&rm1_d, n * logN * sizeof(blk));
    cudaMalloc(&hin0_d, nhash * sizeof(blk));
    cudaMalloc(&hin1_d, nhash * sizeof(blk));
    cudaMalloc(&pad_d, n * N * sizeof(blk));
    cudaMemcpy(rm0_d, rm0_h.data(), n * logN * sizeof(blk), cudaMemcpyHostToDevice);
    cudaMemcpy(rm1_d, rm1_h.data(), n * logN * sizeof(blk), cudaMemcpyHostToDevice);
    cudaMemcpy(hin0_d, hin0_h.data(), nhash * sizeof(blk), cudaMemcpyHostToDevice);
    cudaMemcpy(hin1_d, hin1_h.data(), nhash * sizeof(blk), cudaMemcpyHostToDevice);

    // warm-up launch (first launch pays JIT/alloc)
    gpu_mm::gpu_kern::launch_leaf_pad_sender<logN>(rm0_d, rm1_d, hin0_d, hin1_d, pad_d, n);
    cudaDeviceSynchronize();

    auto g0 = std::chrono::high_resolution_clock::now();
    gpu_mm::gpu_kern::launch_leaf_pad_sender<logN>(rm0_d, rm1_d, hin0_d, hin1_d, pad_d, n);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::high_resolution_clock::now();
    double gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(g1 - g0).count();

    std::vector<emp::block> pad_gpu_h((size_t)n * N);
    cudaMemcpy(pad_gpu_h.data(), pad_d, n * N * sizeof(blk), cudaMemcpyDeviceToHost);

    // ---------- correctness: pad_cpu == pad_gpu ----------
    int64_t bad = 0;
    for (size_t i = 0; i < pad_cpu.size(); ++i) {
        emp::block d = pad_cpu[i] ^ pad_gpu_h[i];
        if (!_mm_testz_si128(d, d)) ++bad;
    }

    std::cout << "=== MITCCRH micro-bench (leaf pad, logN=" << logN
              << " N=" << N << " n=" << n << ") ===\n";
    std::cout << "CPU (AES-NI, serial per-output): " << cpu_us << " us ("
              << cpu_us / 1000.0 << " ms)\n";
    std::cout << "GPU (T-table, parallel):         " << gpu_us << " us ("
              << gpu_us / 1000.0 << " ms)  [launch+compute, no H2D/D2H]\n";
    std::cout << "correctness (pad_cpu == pad_gpu): bad=" << bad << "/"
              << pad_cpu.size() << (bad == 0 ? "  OK" : "  MISMATCH") << "\n";
    std::cout << "speedup GPU/CPU: " << (cpu_us / gpu_us) << "x\n";

    // Debug: dump output 0, pad block k=0..3, CPU vs GPU hex (find assembly bug).
    if (bad != 0) {
        auto pr = [](const char* lbl, const emp::block& b) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&b);
            std::cout << lbl << " ";
            for (int i = 0; i < 16; ++i) printf("%02x", p[i]);
            std::cout << "\n";
        };
        std::cout << "--- debug out[0] pad[k=0..3] ---\n";
        for (int k = 0; k < 4; ++k) {
            char lb[32];
            snprintf(lb, sizeof(lb), "cpu pad[%d] ", k); pr(lb, pad_cpu[k]);
            snprintf(lb, sizeof(lb), "gpu pad[%d] ", k); pr(lb, pad_gpu_h[k]);
        }
        pr("rm0[0] (key0)", rm0_h[0]);
        pr("hin0[0] (pt)  ", hin0_h[0]);
        // Also dump what emp AES-NI alone produces (in^key, 9x aesenc, aesenclast)
        // to compare against the GPU single AES.
        emp::block aesref;
        {
            emp::AES_KEY keys[1];
            emp::AES_opt_key_schedule<1>((emp::block*)&rm0_h[0], keys);
            emp::block s = hin0_h[0] ^ keys[0].rd_key[0];
            for (int r = 1; r < 10; ++r) s = _mm_aesenc_si128(s, keys[0].rd_key[r]);
            s = _mm_aesenclast_si128(s, keys[0].rd_key[10]);
            aesref = s ^ hin0_h[0];  // Davies-Meyer
        }
        pr("empAES(ref) ", aesref);
        // GPU single AES(key0, hin0[0]) Davies-Meyer — should equal empAES(ref).
        // dbg_single_aes takes DEVICE blk*; upload key/pt to device first.
        blk *key_d, *in_d, *out_d;
        cudaMalloc(&key_d, sizeof(blk)); cudaMalloc(&in_d, sizeof(blk));
        cudaMalloc(&out_d, sizeof(blk));
        cudaMemcpy(key_d, &rm0_h[0], sizeof(blk), cudaMemcpyHostToDevice);
        cudaMemcpy(in_d, &hin0_h[0], sizeof(blk), cudaMemcpyHostToDevice);
        gpu_mm::gpu_kern::launch_dbg_single_aes(key_d, in_d, out_d);
        cudaDeviceSynchronize();
        blk gpu_sng;
        cudaMemcpy(&gpu_sng, out_d, sizeof(blk), cudaMemcpyDeviceToHost);
        emp::block gpu_aes;
        std::memcpy(&gpu_aes, &gpu_sng, sizeof(blk));
        pr("gpuAES(sng) ", gpu_aes);
        cudaFree(key_d); cudaFree(in_d); cudaFree(out_d);
        // FIPS known vector: AES-128(key=0, pt=0) = 66e94bd4ef8a2c3b884cfa59ca342b2e
        {
            blk zero{};  // all-zero key + pt
            blk *zk_d, *zi_d, *zo_d;
            cudaMalloc(&zk_d, sizeof(blk)); cudaMalloc(&zi_d, sizeof(blk));
            cudaMalloc(&zo_d, sizeof(blk));
            cudaMemcpy(zk_d, &zero, sizeof(blk), cudaMemcpyHostToDevice);
            cudaMemcpy(zi_d, &zero, sizeof(blk), cudaMemcpyHostToDevice);
            gpu_mm::gpu_kern::launch_dbg_single_aes(zk_d, zi_d, zo_d);
            cudaDeviceSynchronize();
            blk zg;
            cudaMemcpy(&zg, zo_d, sizeof(blk), cudaMemcpyDeviceToHost);
            emp::block zgh;
            std::memcpy(&zgh, &zg, sizeof(blk));
            pr("gpuAES(0,0)", zgh);
            std::cout << "  expected:     66e94bd4ef8a2c3b884cfa59ca342b2e\n";
            cudaFree(zk_d); cudaFree(zi_d); cudaFree(zo_d);
        }
    }

    cudaFree(rm0_d); cudaFree(rm1_d); cudaFree(hin0_d); cudaFree(hin1_d);
    cudaFree(pad_d);
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 800000;
    return run(n);
}
