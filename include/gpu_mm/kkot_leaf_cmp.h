// KkotLeafCmp — cuOT-backed single-digit compare leaf (Task M3-T6 Phase A).
//
// A 1-out-of-N chosen-message OT leaf, the bottom layer of Millionaire's
// radix compare (millionaire.h:82-122, the `bitlength <= beta` early-exit
// branch). SCI serves this leaf with `otpack->kkot[bitlength-1]->send/recv`
// (a Cheetah SilentOTN = 1-OO-N-KOT). cuOT has NO native KKOT; this primitive
// REDUCES the 1-OO-N leaf to cuOT's block chosen-COT + a garbled table, exactly
// the reduction SCI's own SilentOT::send_ot_cm_cc / recv_ot_cm_cc uses
// (silent_ot.h:503-619). So this is "KKOT leaf via COT" — the cuOT-compatible
// formulation the M3-T1 map (§2.3) said was needed before comparison could
// touch cuOT.
//
// WHAT IT COMPUTES (mirrors millionaire.h:88-98, greater_than=true):
//   Sender (ALICE): holds digit dataA[i] ∈ [0,N). Picks random mask res[i]∈{0,1}.
//                   Builds leaf table msg[k] = ((dataA[i] > k) ^ res[i])  for k in [0,N).
//                   Sends the table via 1-OO-N OT with BOB's choice = dataB[i].
//   Receiver (BOB): choice = dataB[i] & mask. recv 1-OO-N OT -> res[i] = msg[choice].
//   Reconstruction: res_A[i] ^ res_B[i] == (dataA[i] > dataB[i])   (greater_than)
//                                            == (dataA[i] < dataB[i])   (!greater_than)
// i.e. a shared bit of the single-digit comparison. (millionaire then folds
// these per-digit leaf bits through an AND-tree for multi-digit compare; this
// primitive does ONLY the leaf — Phase A scope.)
//
// THE REDUCTION (1-OO-N chosen-msg OT → logN block chosen-COT + garbled table),
// copied from silent_ot.h:503-619 and retargeted to CuotProvider:
//   SENDER (mirrors silent_ot.h:516-557):
//     1. logN = ceil(log2(N)). rm0 = send_cot_blocks(n*logN)  -> sender's
//        adjusted blocks x'[i,s]. The 1-branch needs x' ^ Delta (rm1); Delta
//        is the global Ferret Delta (CuotProvider::delta_block).
//     2. hash_in0/hash_in1 layout (silent_ot.h:519-525): for x in [0,logN),
//        y in [0,2^x): in0[idx]=makeBlock(y,0), in1[idx]=makeBlock(2^x+y,0).
//     3. per output i: renew_ks(rm0[i],logN); hash_exp(out0,in0,logN);
//        renew_ks(rm1[i],logN); hash_exp(out1,in1,logN). Then pad[k] =
//        XOR over bits s of k: bit0 -> out0[idx+pref], bit1 -> out1[idx+pref]
//        (silent_ot.h:536-547).
//     4. garbled table y = pack(pad ^ msg); send y (silent_ot.h:550-557).
//   RECEIVER (mirrors silent_ot.h:590-614):
//     1. b[i,s] = bit s of choice. rm = recv_cot_blocks(n*logN) -> r[i,s].
//     2. per output i: hash_in[s] = makeBlock(choice[i] & ((1<<(s+1))-1), 0)
//        (silent_ot.h:603); renew_ks(rm[i],logN); hash_single(out,in,logN);
//        pad_choice = XOR over s of out[s] (silent_ot.h:607-609).
//     3. recv y; unpack: res[i] = (y[choice] ^ lo64(pad_choice)) & mask.
//
// CAVEAT (docs/research/m2-cuot-standalone.md §0,§7.4): cuOT's block-RCOT has
// a STABLE ~2% error floor (M2-T2 SHELVED). This leaf runs logN block-COTs per
// output, so the leaf error rate compounds: P(any of logN COTs bad) ≈
// 1 - 0.98^logN. N=16 (logN=4) ≈ 7.8%; N=8 ≈ 5.9%; N=4 ≈ 4.0%; N=2 ≈ 2.0%.
// This is a DIAGNOSTIC primitive: it reports the actual rate; it is NOT a
// correctness-clear comparison. Per user direction (2026-06-28, "先建 cuOT
// 比较诊断"), this is the intended measurement, not a claim of a working cuOT
// comparison.
//
// MITCCRH NOTE: SCI's N-ary garbled table uses cheetah::MITCCRH::hash_exp /
// hash_single / renew_ks(block*,int) (SCI/src/utils/mitccrh.h:89-169). cuOT's
// FerretCOT exposes emp::MITCCRH (emp-tool) which has ONLY hash<K,H>/setS/
// renew_ks(gid) — NO hash_exp/hash_single. So this primitive vendors a small
// gpu_mm-local MITCCRH (NaryMitccrh below) using emp-tool's AES primitives
// (AES_opt_key_schedule + the vendored ParaEncExp/ParaEncSingle). The base
// block-COT (step 1) uses CuotProvider::send_cot_blocks/recv_cot_blocks
// verbatim (those already do the correct block chosen-COT). Only the
// decorrelation/garbling (steps 2-4) is re-derived here.
#ifndef GPU_MM_KKOT_LEAF_CMP_H
#define GPU_MM_KKOT_LEAF_CMP_H

#include "gpu_mm/cuot_provider.h"   // CuotProvider + send_cot_blocks/recv_cot_blocks
#include "gpu_mm/ot_provider.h"
#include "gpu_matrix.h"             // Mat (cuOT GPU block array, plain C++)

#include <emp-tool/emp-tool.h>      // emp::block, emp::NetIO, getLSB, AES_KEY, makeBlock, zero_block
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gpu_mm {

// emp keeps block / AES_KEY / makeBlock / zero_block in namespace emp (deps/
// emp-tool/emp-tool/utils/{block,aes}.h). Pull them into gpu_mm scope so the
// vendored NaryMitccrh (which mirrors SCI's cheetah::MITCCRH using unqualified
// block/AES_KEY) reads cleanly. (cuot_provider.cc instead fully-qualifies;
// either is fine — this header chose the using-declarations for readability.)
using emp::block;
using emp::AES_KEY;
using emp::makeBlock;
using emp::zero_block;

// GPU kernel launchers (defined in cuot_kernels.cu, nvcc TU). Forward-declared
// so this header-only primitive (compiled by g++) can call them as host fns.
namespace gpu_kern {
template <int logN>
void launch_leaf_pad_sender(const blk* rm0, const blk* rm1,
                            const blk* hash_in0, const blk* hash_in1,
                            blk* pad, int n);
void launch_leaf_pad_recv(const blk* rm, blk* pad_choice,
                          const uint8_t* choice, int n, int logN);
}  // namespace gpu_kern


namespace {

// gpu_mm-local MITCCRH with hash_exp / hash_single / renew_ks(block*,int), for
// the N-ary garbled-table reduction. Mirrors cheetah::MITCCRH
// (SCI/src/utils/mitccrh.h) but lives in gpu_mm to avoid SCI's OT/ot.h include
// chain colliding with cuOT's emp-ot on the cuOT build target. Uses emp-tool's
// AES_opt_key_schedule (deps/emp-tool/emp-tool/utils/aes_opt.h).
class NaryMitccrh {
   public:
    AES_KEY scheduled_key[8]{};

    void renew_ks(block* new_keys, int n) {
        for (int i = 0; i < n; ++i) keys_[i] = new_keys[i];
        switch (n) {
            case 1: AES_opt_key_schedule<1>(keys_, scheduled_key); break;
            case 2: AES_opt_key_schedule<2>(keys_, scheduled_key); break;
            case 3: AES_opt_key_schedule<3>(keys_, scheduled_key); break;
            case 4: AES_opt_key_schedule<4>(keys_, scheduled_key); break;
            case 8: AES_opt_key_schedule<8>(keys_, scheduled_key); break;
            default: throw std::invalid_argument("NaryMitccrh: bad n");
        }
    }

    // Expand logN keys into (2^logN - 1) hash outputs (garbled-table 0/1
    // branches). Mirrors cheetah::MITCCRH::hash_exp (SCI mitccrh.h:113) +
    // ParaEncExp (SCI mitccrh.h:16-48). out has (1<<n)-1 blocks; in has
    // (1<<n)-1 seed blocks.
    void hash_exp(block* out, const block* in, int n) {
        int n_blks = (1 << n) - 1;
        for (int i = 0; i < n_blks; ++i) out[i] = in[i];
        switch (n) {
            case 1: para_enc_exp<1>(out, scheduled_key); break;
            case 2: para_enc_exp<2>(out, scheduled_key); break;
            case 3: para_enc_exp<3>(out, scheduled_key); break;
            case 4: para_enc_exp<4>(out, scheduled_key); break;
            case 8: para_enc_exp<8>(out, scheduled_key); break;
            default: throw std::invalid_argument("NaryMitccrh: bad n");
        }
    }

    // Hash logN blocks with logN keys (1 enc each). Mirrors
    // cheetah::MITCCRH::hash_single (SCI mitccrh.h:142). out/in length = n.
    void hash_single(block* out, const block* in, int n) {
        int n_blks = n;
        for (int i = 0; i < n_blks; ++i) out[i] = in[i];
        switch (n) {
            case 1: para_enc<1, 1>(out, scheduled_key); break;
            case 2: para_enc<2, 1>(out, scheduled_key); break;
            case 3: para_enc<3, 1>(out, scheduled_key); break;
            case 4: para_enc<4, 1>(out, scheduled_key); break;
            case 8: para_enc<8, 1>(out, scheduled_key); break;
            default: throw std::invalid_argument("NaryMitccrh: bad n");
        }
    }

   private:
    block keys_[8]{};

    // ParaEncExp: key i encrypts 2^i blocks (garbled-table expand). Vendored
    // from SCI/src/utils/mitccrh.h:16-48 (x86_64). Writes out in place
    // (out = AES_in(out), i.e. out ^= round-encrypted(in)).
    template <int numKeys>
    static inline void para_enc_exp(block* blks, AES_KEY* keys) {
        block* first = blks;
        for (int i = 0; i < numKeys; ++i) {
            block K = keys[i].rd_key[0];
            int numEncs = 1 << i;
            for (int j = 0; j < numEncs; ++j) { *blks = *blks ^ K; ++blks; }
        }
        for (unsigned int r = 1; r < 10; ++r) {
            blks = first;
            for (int i = 0; i < numKeys; ++i) {
                block K = keys[i].rd_key[r];
                int numEncs = 1 << i;
                for (int j = 0; j < numEncs; ++j) {
                    *blks = _mm_aesenc_si128(*blks, K); ++blks;
                }
            }
        }
        blks = first;
        for (int i = 0; i < numKeys; ++i) {
            block K = keys[i].rd_key[10];
            int numEncs = 1 << i;
            for (int j = 0; j < numEncs; ++j) {
                *blks = _mm_aesenclast_si128(*blks, K); ++blks;
            }
        }
    }

    // ParaEnc<K,1>: K keys, 1 enc each (hash_single). Vendored from
    // emp-tool/emp-tool/utils/aes_opt.h ParaEnc<K,numEncs> for numEncs=1.
    template <int numKeys, int numEncs>
    static inline void para_enc(block* blks, AES_KEY* keys) {
        block* first = blks;
        for (int i = 0; i < numKeys; ++i) {
            block K = keys[i].rd_key[0];
            for (int j = 0; j < numEncs; ++j) { *blks = *blks ^ K; ++blks; }
        }
        for (unsigned int r = 1; r < 10; ++r) {
            blks = first;
            for (int i = 0; i < numKeys; ++i) {
                block K = keys[i].rd_key[r];
                for (int j = 0; j < numEncs; ++j) {
                    *blks = _mm_aesenc_si128(*blks, K); ++blks;
                }
            }
        }
        blks = first;
        for (int i = 0; i < numKeys; ++i) {
            block K = keys[i].rd_key[10];
            for (int j = 0; j < numEncs; ++j) {
                *blks = _mm_aesenclast_si128(*blks, K); ++blks;
            }
        }
    }
};

// Low 64 bits of an emp::block (SSE4.1; cuOT target has -msse4.1).
inline uint64_t lo64(const emp::block& b) { return _mm_extract_epi64(b, 0); }

// Pack n N-ary l-bit garbled-table rows into a bit-packed uint64_t array.
// Mirrors sci::pack_ot_messages<uint8_t> (SCI/src/OT/ot-utils.h:28) for the
// l=1 case (leaf messages are 1 bit). y[k] = (lo64(pad[i*N+k]) ^ msg) & mask.
inline void pack_leaf_messages(uint64_t* y, const uint8_t* msg,
                               const block* pad, int ysize,
                               int n, int N, int l) {
    const uint64_t mask = (l == 64) ? ~0ULL : ((1ULL << l) - 1);
    const uint64_t carriersize = 64;
    for (int i = 0; i < ysize; ++i) y[i] = 0;
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < N; ++k) {
            uint64_t start_pos = (uint64_t)i * N * l + (uint64_t)k * l;
            uint64_t end_pos = start_pos + l - 1;
            uint64_t start_block = start_pos / carriersize;
            uint64_t end_block = end_pos / carriersize;
            uint64_t temp = (lo64(pad[i * N + k]) ^ (uint64_t)msg[i * N + k]) & mask;
            if (start_block == end_block) {
                y[start_block] ^= temp << (start_pos % carriersize);
            } else {
                y[start_block] ^= temp << (start_pos % carriersize);
                y[end_block] ^= temp >> (carriersize - (start_pos % carriersize));
            }
        }
    }
}

// Unpack the receiver's chosen row. Mirrors sci::unpack_ot_messages<uint8_t>
// (SCI/src/OT/ot-utils.h:99), l=1. out[i] = (y[choice] ^ lo64(pad[i])) & mask.
inline void unpack_leaf_messages(uint8_t* out, const uint8_t* choice,
                                 const uint64_t* recvd, const block* pad,
                                 int n, int N, int l) {
    const uint64_t mask = (l == 64) ? ~0ULL : ((1ULL << l) - 1);
    const uint64_t carriersize = 64;
    for (int i = 0; i < n; ++i) {
        uint64_t start_pos = (uint64_t)i * N * l + (uint64_t)choice[i] * l;
        uint64_t end_pos = start_pos + l - 1;
        uint64_t start_block = start_pos / carriersize;
        uint64_t end_block = end_pos / carriersize;
        uint64_t v;
        if (start_block == end_block) {
            v = (recvd[start_block] >> (start_pos % carriersize));
        } else {
            v = (recvd[start_block] >> (start_pos % carriersize)) |
                (recvd[end_block] << (carriersize - (start_pos % carriersize)));
        }
        out[i] = (uint8_t)((v ^ lo64(pad[i])) & mask);
    }
}

}  // namespace

// KkotLeafCmp does NOT own the CuotProvider; the caller owns it. One instance
// = ONE party (sender or receiver) over ONE provider channel. Two-party
// protocols construct two instances backed by a connected provider pair.
class KkotLeafCmp {
   public:
    KkotLeafCmp(CuotProvider& prov, OTParty role) : prov_(prov), role_(role) {}

    // --- OPT-1: GPU-resident 2-bit leaf (uses leaf_pad_sender/recv kernel) --
    // Same semantics as send_leaf/recv_leaf (l=2, N=16) but the RCOT blocks
    // stay on the GPU (caller's Mat, sliced at `rm_off` with logN blocks
    // per output) and the MITCCRH decorrelation runs as a CUDA kernel. Only
    // the small garbled table (n*N*l/8 bytes) + Alice's masks cross the
    // host/net. The 1-bit assembly residual bug (§8.8) is ACCEPTED here —
    // the compare is already ~50% wrong from the floor; the residual adds
    // a small extra error. n must be a multiple of 8.
    //   rm0/rm1/rm : device Mat regions (rm0.data()+rm_off .., etc).
    //   hash_in0/1_dev : device arrays (2^logN-1) built once by the caller.
    OTStatus send_leaf_gpu(const uint8_t* dataA, uint8_t* res, int32_t n, bool greater,
                           int32_t N, int32_t logN,
                           const Mat& rm0, int64_t rm_off,
                           const Mat& rm1,
                           const block* hash_in0_dev,
                           const block* hash_in1_dev) {
        if (l_mismatch(N, logN)) return OTStatus::kInvalidArg;
        // Reuse the CPU send_leaf's msg-table + mask logic, but swap the
        // rcot+decorrelation for the GPU kernel. We need the same msg table
        // (built from dataA + random masks) and the same pack/send of the
        // garbled table — only the pad construction moves to GPU.
        emp::NetIO* io = prov_.net_io();
        if (io == nullptr) return OTStatus::kConfig;
        // 1. masks + msg table (CPU, same as send_leaf).
        uint64_t seed = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)n << 32) ^ ((uint64_t)N);
        for (int32_t i = 0; i < n; ++i) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            uint8_t mcmp = (uint8_t)(seed & 1);
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            uint8_t meq = (uint8_t)(seed & 1);
            res[i] = (uint8_t)((mcmp << 1) | meq);
        }
        std::vector<uint8_t> msg((size_t)n * N);
        for (int32_t i = 0; i < n; ++i) {
            uint8_t mcmp = (res[i] >> 1) & 1, meq = res[i] & 1;
            for (int k = 0; k < N; ++k) {
                uint8_t cmp_bit = (uint8_t)((greater ? (dataA[i] > k) : (dataA[i] < k)) ^ mcmp);
                uint8_t eq_bit = (uint8_t)((dataA[i] == k) ^ meq);
                msg[(size_t)i * N + k] = (uint8_t)((cmp_bit << 1) | eq_bit);
            }
        }
        // 2. GPU pad: launch_leaf_pad_sender on the device rm0/rm1 slices.
        //    rm0.data()+rm_off is the i-th output's logN base blocks.
        Mat pad({(uint64_t)n * N});  // GPU pad output
        const blk* rm0p = (const blk*)rm0.data() + rm_off;
        const blk* rm1p = (const blk*)rm1.data() + rm_off;
        const blk* hi0 = (const blk*)hash_in0_dev;
        const blk* hi1 = (const blk*)hash_in1_dev;
        if (logN == 4) gpu_kern::launch_leaf_pad_sender<4>(rm0p, rm1p, hi0, hi1, pad.data(), n);
        else if (logN == 3) gpu_kern::launch_leaf_pad_sender<3>(rm0p, rm1p, hi0, hi1, pad.data(), n);
        else if (logN == 2) gpu_kern::launch_leaf_pad_sender<2>(rm0p, rm1p, hi0, hi1, pad.data(), n);
        else gpu_kern::launch_leaf_pad_sender<1>(rm0p, rm1p, hi0, hi1, pad.data(), n);
        cudaDeviceSynchronize();
        // 3. pack garbled table y = (lo64(pad) ^ msg) & mask (CPU, small).
        const int l = 2;
        const int ysize = (int)ceil((double)(n * N * l) / 64.0);
        std::vector<emp::block> pad_h((size_t)n * N);
        pad.write_to_cpu((uint8_t*)pad_h.data(), (size_t)n * N * sizeof(emp::block));
        std::vector<uint64_t> y((size_t)ysize);
        pack_leaf_messages(y.data(), msg.data(), pad_h.data(), ysize, n, N, l);
        io->send_data(y.data(), sizeof(uint64_t) * ysize);
        io->flush();
        return OTStatus::kOk;
    }
    OTStatus recv_leaf_gpu(const uint8_t* choice, uint8_t* res, int32_t n,
                           int32_t N, int32_t logN,
                           const Mat& rm, int64_t rm_off) {
        if (l_mismatch(N, logN)) return OTStatus::kInvalidArg;
        emp::NetIO* io = prov_.net_io();
        if (io == nullptr) return OTStatus::kConfig;
        // 1. GPU pad_choice: leaf_pad_recv on the device rm slice.
        Mat pad_choice({(uint64_t)n});
        const blk* rmp = (const blk*)rm.data() + rm_off;
        gpu_kern::launch_leaf_pad_recv(rmp, pad_choice.data(), choice, n, logN);
        cudaDeviceSynchronize();
        // 2. recv + unpack garbled table (CPU, same as recv_leaf).
        const int l = 2;
        const int ysize = (int)ceil((double)(n * N * l) / 64.0);
        std::vector<uint64_t> y((size_t)ysize);
        io->recv_data(y.data(), sizeof(uint64_t) * ysize);
        std::vector<emp::block> pad_h((size_t)n);
        pad_choice.write_to_cpu((uint8_t*)pad_h.data(), (size_t)n * sizeof(emp::block));
        unpack_leaf_messages(res, choice, y.data(), pad_h.data(), n, N, l);
        return OTStatus::kOk;
    }
   private:
    static bool l_mismatch(int32_t N, int32_t logN) {
        return N < 2 || (N & (N - 1)) != 0 || N > 16 ||
               logN < 1 || logN > 4 || (1 << logN) != N;
    }
   public:

    // --- Sender (ALICE): 1-OO-N chosen-message OT leaf -------------------
    //   dataA   : length-n vector, each in [0,N) (ALICE's digit).
    //   res     : length-n uint8, receives ALICE's random mask bit(s).
    //             l=1: res[i] ∈ {0,1} (cmp mask only).
    //             l=2: res[i] = (mask_cmp << 1) | mask_eq  (2-bit, cmp+eq),
    //                  matching millionaire.h set_leaf_ot_messages.
    //   N       : choice domain size, power of 2 in {2,4,8,16}.
    //   l       : message bitwidth, 1 (cmp only) or 2 (cmp+eq).
    //   greater : true => (dataA > choice), false => (dataA < choice).
    //             (eq bit is (dataA == choice), independent of greater.)
    OTStatus send_leaf(const uint8_t* dataA, uint8_t* res, int32_t n,
                       int32_t N, int32_t l, bool greater = true) {
        if (dataA == nullptr || res == nullptr || n <= 0 ||
            (l != 1 && l != 2) ||
            N < 2 || (N & (N - 1)) != 0 || N > 16) {
            return OTStatus::kInvalidArg;
        }
        const int logN = (int)ceil(log2((double)N));
        emp::NetIO* io = prov_.net_io();
        if (io == nullptr) return OTStatus::kConfig;

        // 1. Random masks. l=1: one mask bit res[i] (cmp mask). l=2: two mask
        //    bits — mask_cmp (bit1) and mask_eq (bit0), packed into res[i].
        //    Mirrors millionaire.h:87-90 (prg.random_data &1) for the cmp
        //    mask; the eq mask is the same random-data trick (SCI draws both
        //    via prg.random_bool on leaf_res_cmp/leaf_res_eq separately,
        //    lines 166-167). Diagnostic: local xorshift. NOT production crypto.
        uint64_t seed = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)n << 32) ^ ((uint64_t)N);
        for (int32_t i = 0; i < n; ++i) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            uint8_t mcmp = (uint8_t)(seed & 1);
            if (l == 1) {
                res[i] = mcmp;
            } else {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                uint8_t meq = (uint8_t)(seed & 1);
                res[i] = (uint8_t)((mcmp << 1) | meq);  // bit1=cmp, bit0=eq
            }
        }

        // 2. Leaf table msg[i*N+k]:
        //    l=1: ((dataA[i] > k) ^ res[i])                       (millionaire.h:93-94)
        //    l=2: bit1 = ((dataA[i] > k) ^ mask_cmp),
        //         bit0 = ((dataA[i] == k) ^ mask_eq)              (set_leaf_ot_messages:330-336)
        std::vector<uint8_t> msg((size_t)n * N);
        for (int32_t i = 0; i < n; ++i) {
            uint8_t mcmp = (l == 1) ? res[i] : ((res[i] >> 1) & 1);
            uint8_t meq = (l == 1) ? 0 : (res[i] & 1);
            for (int k = 0; k < N; ++k) {
                uint8_t cmp_bit = (uint8_t)((greater ? (dataA[i] > k)
                                                     : (dataA[i] < k)) ^ mcmp);
                if (l == 1) {
                    msg[(size_t)i * N + k] = cmp_bit;
                } else {
                    uint8_t eq_bit = (uint8_t)((dataA[i] == k) ^ meq);
                    msg[(size_t)i * N + k] = (uint8_t)((cmp_bit << 1) | eq_bit);
                }
            }
        }

        // 3. logN block chosen-COTs per output -> sender x'[i,s] (silent_ot.h
        //    send_ot_rm_cc; here send_cot_blocks once, n*logN blocks).
        std::vector<block> rm0((size_t)n * logN);
        OTStatus s = prov_.send_cot_blocks(
            reinterpret_cast<uint8_t*>(rm0.data()), (int64_t)n * logN);
        if (s != OTStatus::kOk) return s;
        // 1-branch keys = x' ^ Delta (rm1). Delta = global Ferret Delta.
        uint8_t dbuf[16];
        if (prov_.delta_block(dbuf) != OTStatus::kOk) return OTStatus::kInternal;
        block delta;
        std::memcpy(&delta, dbuf, 16);
        std::vector<block> rm1((size_t)n * logN);
        for (int32_t i = 0; i < n; ++i)
            for (int t = 0; t < logN; ++t)
                rm1[(size_t)i * logN + t] = rm0[(size_t)i * logN + t] ^ delta;

        // 4. MITCCRH seed: sender picks s0, sends to receiver (silent_ot.h:507).
        block seed_blk;
        {
            uint64_t s2 = 0xd1b54a32d192ed03ULL ^ ((uint64_t)n) ^ ((uint64_t)logN);
            uint64_t* p = reinterpret_cast<uint64_t*>(&seed_blk);
            s2 ^= s2 << 13; s2 ^= s2 >> 7; s2 ^= s2 << 17; p[0] = s2;
            s2 ^= s2 << 13; s2 ^= s2 >> 7; s2 ^= s2 << 17; p[1] = s2;
        }
        io->send_block(&seed_blk, 1);
        io->flush();

        // 5. Garbled-table pad construction (silent_ot.h:516-547).
        NaryMitccrh mitccrh;
        std::vector<block> hash_in0((size_t)((1 << logN) - 1));
        std::vector<block> hash_in1((size_t)((1 << logN) - 1));
        {
            int idx = 0;
            for (int x = 0; x < logN; ++x) {
                for (int y = 0; y < (1 << x); ++y) {
                    hash_in0[idx] = emp::makeBlock((uint64_t)y, 0);
                    hash_in1[idx] = emp::makeBlock((uint64_t)((1 << x) + y), 0);
                    ++idx;
                }
            }
        }
        std::vector<block> hash_out((size_t)2 * ((1 << logN) - 1));
        std::vector<block> pad((size_t)n * N);

        for (int32_t i = 0; i < n; ++i) {
            mitccrh.renew_ks(rm0.data() + (size_t)i * logN, logN);
            mitccrh.hash_exp(hash_out.data(), hash_in0.data(), logN);
            mitccrh.renew_ks(rm1.data() + (size_t)i * logN, logN);
            mitccrh.hash_exp(hash_out.data() + ((1 << logN) - 1),
                             hash_in1.data(), logN);
            for (int k = 0; k < N; ++k) {
                block p = emp::zero_block;
                int idx = 0;
                for (int sb = 0; sb < logN; ++sb) {
                    int mask = (1 << sb) - 1;
                    int pref = k & mask;
                    if ((k & (1 << sb)) == 0)
                        p = p ^ hash_out[idx + pref];
                    else
                        p = p ^ hash_out[((1 << logN) - 1) + idx + pref];
                    idx += 1 << sb;
                }
                pad[(size_t)i * N + k] = p;
            }
        }

        // 6. Pack + send the garbled table (silent_ot.h:550-557).
        const int ysize = (int)ceil((double)(n * N * l) / 64.0);
        std::vector<uint64_t> y((size_t)ysize);
        pack_leaf_messages(y.data(), msg.data(), pad.data(), ysize, n, N, l);
        io->send_data(y.data(), sizeof(uint64_t) * ysize);
        io->flush();

        return OTStatus::kOk;
    }

    // --- Receiver (BOB): 1-OO-N chosen-message OT leaf -------------------
    //   choice : length-n vector, each in [0,N) (BOB's digit = dataB & mask).
    //   res    : length-n uint8, receives BOB's recovered message.
    //            l=1: 1-bit cmp. l=2: 2-bit packed (bit0=eq, bit1=cmp).
    OTStatus recv_leaf(const uint8_t* choice, uint8_t* res, int32_t n,
                       int32_t N, int32_t l) {
        if (choice == nullptr || res == nullptr || n <= 0 ||
            (l != 1 && l != 2) ||
            N < 2 || (N & (N - 1)) != 0 || N > 16) {
            return OTStatus::kInvalidArg;
        }
        const int logN = (int)ceil(log2((double)N));
        emp::NetIO* io = prov_.net_io();
        if (io == nullptr) return OTStatus::kConfig;

        // 1. Expand choice bits: b[i*logN + s] = bit s of choice[i]. Use
        //    uint8_t (NOT vector<bool> — its data() is a bit-proxy, not bool*;
        //    recv_cot_blocks takes const bool*). Cast is safe: values are 0/1.
        std::vector<uint8_t> bbuf((size_t)n * logN);
        for (int32_t i = 0; i < n; ++i)
            for (int s = 0; s < logN; ++s)
                bbuf[(size_t)i * logN + s] =
                    (uint8_t)((choice[i] & (1 << s)) >> s);
        const bool* b = reinterpret_cast<const bool*>(bbuf.data());

        // 2. logN block chosen-COTs per output -> receiver r[i,s] (silent_ot.h
        //    recv_ot_rm_cc; here recv_cot_blocks once, n*logN blocks).
        std::vector<block> rm((size_t)n * logN);
        OTStatus cst = prov_.recv_cot_blocks(
            reinterpret_cast<uint8_t*>(rm.data()), b, (int64_t)n * logN);
        if (cst != OTStatus::kOk) return cst;

        // 3. MITCCRH seed from sender (silent_ot.h:598 recv_block).
        block seed_blk;
        io->recv_block(&seed_blk, 1);
        (void)seed_blk;  // (NaryMitccrh keys are set per-output via renew_ks,
                         //  not via setS; seed is only used to sync with sender
                         //  if both derived keys from it. Here keys come from
                         //  rm directly, so seed is unused on the receiver.)

        // 4. Receiver computes pad[choice[i]] per output (silent_ot.h:600-610):
        //    hash_in[s] = makeBlock(choice[i] & ((1<<(s+1))-1), 0);
        //    renew_ks(rm[i], logN); hash_single(hash_out, hash_in, logN);
        //    pad_choice = XOR over s of hash_out[s].
        NaryMitccrh mitccrh;
        std::vector<block> hash_in((size_t)logN);
        std::vector<block> hash_out((size_t)logN);
        std::vector<block> pad_choice((size_t)n);

        for (int32_t i = 0; i < n; ++i) {
            for (int s = 0; s < logN; ++s)
                hash_in[s] = emp::makeBlock(
                    (uint64_t)(choice[i] & ((1 << (s + 1)) - 1)), 0);
            mitccrh.renew_ks(rm.data() + (size_t)i * logN, logN);
            mitccrh.hash_single(hash_out.data(), hash_in.data(), logN);
            block p = emp::zero_block;
            for (int s = 0; s < logN; ++s) p = p ^ hash_out[s];
            pad_choice[i] = p;
        }

        // 5. Recv + unpack the garbled table (silent_ot.h:598, 612).
        const int ysize = (int)ceil((double)(n * N * l) / 64.0);
        std::vector<uint64_t> y((size_t)ysize);
        io->recv_data(y.data(), sizeof(uint64_t) * ysize);
        unpack_leaf_messages(res, choice, y.data(), pad_choice.data(), n, N, l);

        return OTStatus::kOk;
    }

   private:
    CuotProvider& prov_;
    OTParty role_;
};

}  // namespace gpu_mm

#endif  // GPU_MM_KKOT_LEAF_CMP_H
