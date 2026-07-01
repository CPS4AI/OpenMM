// CuotCompare — cuOT-backed full Millionaire compare (Task M3-T6 Phase C).
//
// Composes Phase A's `KkotLeafCmp` (1-OO-N leaf via block-COT reduction) and
// Phase B's `BitTripleGen` (_2ROT Beaver triple) into a full 32-bit
// Millionaire-style compare, mirroring SCI's MillionaireProtocol::compare
// (SCI/src/Millionaire/millionaire.h:76-323). This is the first end-to-end
// cuOT compare primitive — it produces the actual comparison bit
// (share_A > share_B) and is where the ~2% RCOT floor compounds across the
// 8-digit leaf + 11-triple AND-tree.
//
// ALGORITHM (millionaire.h, beta=4 = MILL_PARAM, USE_CHEETAH=1):
//   configure: l=bitlength, beta=4, num_digits=ceil(l/beta), r=l%beta,
//     log_alpha = bitlen_quirk(num_digits)-1, log_num_digits=log_alpha+1,
//     num_triples = (2*num_digits - 2 - 2*log_num_digits) + log_num_digits.
//     For l=32: num_digits=8, log_num_digits=3 (bitlen(8)=3 quirk), num_triples=11.
//   1. pad num_cmps -> ceil(n/8)*8.
//   2. digit extract (LSB-first): digits[d][c] = (share[c] >> d*4) & 0xF.
//   3. 2-bit leaf OT (KkotLeafCmp, N=16, l=2): Alice builds
//      msg[k]=pack2(cmp=(digit>k)^mcmp, eq=(digit==k)^meq); Bob recv with
//      choice=digit -> leaf_res (2-bit). Bob splits: eq=leaf_res&1, cmp=leaf_res>>1
//      (millionaire.h:305-309, for d>=1; d=0 has no eq).
//   4. generate num_triples*num_cmps _2ROT triples (BitTripleGen).
//   5. AND-tree (3 levels for num_digits=8), each AND = Beaver gate:
//      step1: e=x^a, f=y^b (8-bit packed); exchange ei,fi; reconstruct e,f.
//      step2: z = (ALICE: e&f) ^ f&a ^ e&b ^ c  ; (BOB: f&a ^ e&b ^ c).
//      update: cmp[j] ^= cmp[j+i]; if j!=0: eq[j] = z (the eq-AND result).
//      Schedule (millionaire.h:373-540):
//        i=1: j=0 (std, 1 AND: cmp[0]&eq[1]); j=2,4,6 (corr, 2 ANDs each)
//        i=2: j=0 (std); j=4 (corr, 2 ANDs)
//        i=4: j=0 (std)
//   6. res = cmp[0] (millionaire.h:314-315).
//
// provider wiring: CuotCompare holds ONE pair of CuotProviders (prov_send =
// Ferret-SENDER, prov_recv = Ferret-RECEIVER) on two ports — the same pair
// BitTripleGen uses. The leaf (KkotLeafCmp) is single-direction chosen-COT:
// Alice (sender) uses prov_send's send_cot_blocks; Bob (receiver) uses
// prov_recv's recv_cot_blocks. Leaf and triple run SEQUENTIALLY (leaf first,
// then triple gen + AND-tree), reusing the same NetIO channels — no
// concurrency, no deadlock (verified pattern from Phase A/B).
//
// CAVEAT (docs/research/m2-cuot-standalone.md §0,§7.4): cuOT's ~2% RCOT floor
// compounds across 8 digits × leaf(logN=4 block-COTs each) + 11 triples
// (2 ROTs each). Worst-case naive ~65%, but Phase A/B measured "1-bit
// absorption" (wrong ROT often still yields the right 1-bit output), so the
// real rate is an empirical question — this primitive MEASURES it. Diagnostic
// only; not correctness-clear. Per user direction (2026-06-28, "继续推进
// M3-T6"), this is the intended measurement.
#ifndef GPU_MM_CUOT_COMPARE_H
#define GPU_MM_CUOT_COMPARE_H

#include "gpu_mm/bit_triple.h"      // BitTripleGen
#include "gpu_mm/cuot_provider.h"   // CuotProvider
#include "gpu_mm/kkot_leaf_cmp.h"   // KkotLeafCmp (2-bit leaf, +GPU leaf)
#include "gpu_mm/ot_provider.h"
#include "gpu_matrix.h"             // Mat (cuOT GPU block array)

#include <emp-tool/emp-tool.h>      // emp::NetIO
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gpu_mm {

namespace {

// 8-bit bool pack/unpack, vendored from SCI/src/utils/utils.hpp:167,200
// (add_cuot_test does not link SCI-Cheetah, so these can't be included).
// Pack 8 (or fewer) bool-bytes (LSB-first) into one uint8.
inline uint8_t bool_to_uint8(const uint8_t* data, int len) {
    if (len != 0) len = (len > 8 ? 8 : len); else len = 8;
    uint8_t res = 0;
    for (int i = 0; i < len; ++i) if (data[i]) res |= (1 << i);
    return res;
}
// Unpack a uint8 into `length` bool-bytes (LSB-first).
inline void uint8_to_bool(uint8_t* data, uint8_t input, int length) {
    for (int i = 0; i < length; ++i) { data[i] = (uint8_t)((input & 1) == 1); input >>= 1; }
}

// SCI's bitlen quirk (utils/utils.hpp:207-215): returns smallest i s.t.
// 2^i >= x, i.e. ceil(log2(x)) for x>1. NOT the bit-length of x. Used by
// Millionaire to compute log_alpha.
inline int bitlen_quirk(int x) {
    if (x < 1) return 0;
    for (int i = 0; i < 32; ++i) { if ((1 << i) >= x) return i; }
    return 0;
}

}  // namespace

// CuotCompare owns no providers; the caller owns the CuotProvider pair. One
// instance = ONE party over ONE connected provider pair. role = this party's
// role (kSender=ALICE, kReceiver=BOB).
class CuotCompare {
   public:
    CuotCompare(CuotProvider& prov_send, CuotProvider& prov_recv,
                OTParty role)
        : prov_send_(prov_send), prov_recv_(prov_recv), role_(role),
          leaf_send_(prov_send, role), leaf_recv_(prov_recv, role),
          triple_(prov_send, prov_recv, role) {}

    // Full Millionaire compare. Mirrors millionaire.h:76-323.
    //   data : length-num_cmps uint64, this party's l-bit share (only low l
    //          bits used; for BOB the caller passes the COMPLEMENTED share,
    //          matching millionaire's wrap/MSB convention — but this method
    //          itself does NOT complement; it compares data_A vs data_B as
    //          greater_than directly).
    //   res  : length-num_cmps uint8, receives this party's share of
    //          (greater ? data_A > data_B : data_A < data_B).
    //   bitlength, greater_than : as SCI.
    // Returns kOk on success.
    // After both parties return kOk: res_A[i] ^ res_B[i] == (greater ?
    //   data_A[i] > data_B[i] : data_A[i] < data_B[i]).
    OTStatus compare(uint8_t* res, const uint64_t* data, int32_t num_cmps,
                     int32_t bitlength, bool greater_than = true) {
        if (res == nullptr || data == nullptr || num_cmps <= 0 ||
            bitlength <= 0 || bitlength > 64) {
            return OTStatus::kInvalidArg;
        }
        // configure (millionaire.h:53-72). beta = MILL_PARAM = 4.
        const int beta = 4;
        const int l = bitlength;
        const int num_digits = (int)ceil((double)l / beta);
        const int r = l % beta;
        const int log_alpha = bitlen_quirk(num_digits) - 1;
        const int log_num_digits = log_alpha + 1;
        const int num_triples_corr = 2 * num_digits - 2 - 2 * log_num_digits;
        const int num_triples_std = log_num_digits;
        const int num_triples = num_triples_std + num_triples_corr;
        const uint8_t mask_beta = (uint8_t)((1 << beta) - 1);
        const uint8_t mask_r = (uint8_t)((1 << r) - 1);
        const int N = 1 << beta;  // 16

        // pad num_cmps to multiple of 8 (millionaire.h:127).
        int old_num_cmps = num_cmps;
        num_cmps = (int)ceil(num_cmps / 8.0) * 8;
        std::vector<uint64_t> data_ext_buf;
        const uint64_t* data_ext = data;
        if (old_num_cmps != num_cmps) {
            data_ext_buf.assign((size_t)num_cmps, 0);
            std::memcpy(data_ext_buf.data(), data, old_num_cmps * sizeof(uint64_t));
            data_ext = data_ext_buf.data();
        }

        // digit extract (millionaire.h:147-155), LSB-first.
        // digits[d*num_cmps + c] = (data_ext[c] >> d*beta) & mask.
        std::vector<uint8_t> digits((size_t)num_digits * num_cmps);
        for (int d = 0; d < num_digits; ++d) {
            for (int c = 0; c < num_cmps; ++c) {
                if (d == num_digits - 1 && r != 0)
                    digits[(size_t)d * num_cmps + c] =
                        (uint8_t)((data_ext[c] >> (d * beta)) & mask_r);
                else
                    digits[(size_t)d * num_cmps + c] =
                        (uint8_t)((data_ext[c] >> (d * beta)) & mask_beta);
            }
        }

        // ---- OPT-1: GPU-resident 2-bit leaf ----
        // One big RCOT for the leaf (n_leaf_outs * logN blocks) into a GPU Mat;
        // the leaf pad construction runs as a CUDA kernel (leaf_pad_sender/recv),
        // NO per-output CPU AES and NO per-leaf D2H. Only the small garbled
        // table crosses host/net. (The triple ROT decorrelation stays CPU for
        // now — it's a separate Mat; wiring it GPU is future OPT-1 work.)
        // The 1-bit assembly residual (§8.8) is accepted per user direction.
        const int logN = 4;  // N=16
        const int n_leaf_outs = num_digits * num_cmps;
        const int64_t leaf_blocks = (int64_t)n_leaf_outs * logN;
        // hash_in0/in1 device arrays (2^logN-1 each), built once per call.
        // (Could be cached across calls; built here for simplicity.)
        const int nhash = (1 << logN) - 1;
        std::vector<emp::block> hin0_h((size_t)nhash), hin1_h((size_t)nhash);
        {
            int idx = 0;
            for (int x = 0; x < logN; ++x)
                for (int y = 0; y < (1 << x); ++y) {
                    hin0_h[idx] = emp::makeBlock((uint64_t)y, 0);
                    hin1_h[idx] = emp::makeBlock((uint64_t)((1 << x) + y), 0);
                    ++idx;
                }
        }
        emp::block *hin0_d, *hin1_d;
        cudaMalloc(&hin0_d, nhash * sizeof(emp::block));
        cudaMalloc(&hin1_d, nhash * sizeof(emp::block));
        cudaMemcpy(hin0_d, hin0_h.data(), nhash * sizeof(emp::block), cudaMemcpyHostToDevice);
        cudaMemcpy(hin1_d, hin1_h.data(), nhash * sizeof(emp::block), cudaMemcpyHostToDevice);

        // RCOT into a GPU Mat (leaf blocks). ALICE: rm0 = send-side adjusted
        // blocks (via send_cot_blocks_gpu). BOB: rm = recv-side (recv_cot_blocks_gpu).
        // Both need the GPU-resident chosen-COT so blocks never leave the GPU.
        // cmp_sh/eq_sh declared here (used by split_leaf_ below, and by the
        // AND-tree afterward) — must precede the GPU-leaf branch that fills them.
        std::vector<uint8_t> cmp_sh((size_t)num_digits * num_cmps);
        std::vector<uint8_t> eq_sh((size_t)num_digits * num_cmps);
        Mat rm_mat({(uint64_t)leaf_blocks});
        OTStatus s;
        if (role_ == OTParty::kSender) {
            // ALICE: send_cot_blocks_gpu gives x' (adjusted sender blocks) in rm_mat.
            s = prov_send_.send_cot_blocks_gpu(rm_mat, leaf_blocks);
            if (s != OTStatus::kOk) { cudaFree(hin0_d); cudaFree(hin1_d); return s; }
            // rm1 = rm0 ^ Delta (on GPU via apply... no, leaf_pad_sender takes rm0+rm1).
            // Build rm1 Mat = rm0 ^ Delta on GPU with a tiny kernel? Simplest: the
            // leaf_pad_sender kernel reads rm0 and rm1; rm1 = rm0 ^ Delta. Compute on
            // GPU with apply_diff_vector(Delta as the "diff"=all-ones)? Cleaner: a
            // dedicated xor_scalar kernel. For now, do it on CPU: download rm0, XOR,
            // re-upload rm1. (One D2H + H2D — still far cheaper than the per-output
            // CPU AES it replaces.) TODO: a 1-line xor_scalar kernel.
            std::vector<emp::block> rm0_h((size_t)leaf_blocks);
            rm_mat.write_to_cpu((uint8_t*)rm0_h.data(), (size_t)leaf_blocks * sizeof(emp::block));
            uint8_t dbuf[16];
            if (prov_send_.delta_block(dbuf) != OTStatus::kOk) {
                cudaFree(hin0_d); cudaFree(hin1_d); return OTStatus::kInternal;
            }
            emp::block delta_h;
            std::memcpy(&delta_h, dbuf, sizeof(emp::block));
            std::vector<emp::block> rm1_h((size_t)leaf_blocks);
            for (int64_t i = 0; i < leaf_blocks; ++i) rm1_h[i] = rm0_h[i] ^ delta_h;
            Mat rm1_mat({(uint64_t)leaf_blocks});
            rm1_mat.read_from_cpu((uint8_t*)rm1_h.data(), (size_t)leaf_blocks * sizeof(emp::block));
            // GPU leaf send.
            std::vector<uint8_t> leaf_res((size_t)n_leaf_outs);
            s = leaf_send_.send_leaf_gpu(digits.data(), leaf_res.data(),
                                         n_leaf_outs, greater_than,
                                         N, logN,
                                         rm_mat, 0, rm1_mat, hin0_d, hin1_d);
            if (s != OTStatus::kOk) { cudaFree(hin0_d); cudaFree(hin1_d); return s; }
            // Split (CPU, same as before).
            split_leaf_(leaf_res, cmp_sh, eq_sh, num_digits, num_cmps);
        } else {
            // BOB: choice bits on device. recv_cot_blocks_gpu needs ONE choice
            // bit per RCOT block (leaf_blocks = n_leaf_outs*logN), NOT one per
            // leaf output. Expand each digit's logN bits: b[i*logN+s] = bit s
            // of digits[i] (mirrors recv_leaf's bbuf expansion,
            // kkot_leaf_cmp.h:526-530). Sizing choice_d to n_leaf_outs (the old
            // bug) made diff_vector_pack8 read b[n_leaf_outs..leaf_blocks] OOB
            // → sticky "illegal memory access" → "[cuot] rcot failed".
            std::vector<uint8_t> bbuf((size_t)n_leaf_outs * logN);
            for (int32_t i = 0; i < n_leaf_outs; ++i)
                for (int s = 0; s < logN; ++s)
                    bbuf[(size_t)i * logN + s] =
                        (uint8_t)((digits[i] & (1 << s)) >> s);
            uint8_t* choice_d = nullptr;
            cudaMalloc(&choice_d, n_leaf_outs * logN);
            cudaMemcpy(choice_d, bbuf.data(), n_leaf_outs * logN,
                       cudaMemcpyHostToDevice);
            s = prov_recv_.recv_cot_blocks_gpu(rm_mat, choice_d, leaf_blocks);
            cudaFree(choice_d);
            if (s != OTStatus::kOk) { cudaFree(hin0_d); cudaFree(hin1_d); return s; }
            std::vector<uint8_t> leaf_res((size_t)n_leaf_outs);
            s = leaf_recv_.recv_leaf_gpu(digits.data(), leaf_res.data(),
                                         n_leaf_outs, N, logN, rm_mat, 0);
            if (s != OTStatus::kOk) { cudaFree(hin0_d); cudaFree(hin1_d); return s; }
            split_leaf_(leaf_res, cmp_sh, eq_sh, num_digits, num_cmps);
        }
        cudaFree(hin0_d); cudaFree(hin1_d);

        // (The 2-bit leaf is now done GPU-resident above — OPT-1. The old CPU
        // send_leaf/recv_leaf path is replaced by send_leaf_gpu/recv_leaf_gpu.)
        // greater_than is baked into the leaf msg table (send_leaf_gpu builds it
        // with `dataA > k`); the AND-tree is greater_than-independent.
        // cmp_sh/eq_sh were declared + filled above (split_leaf_ in the GPU-leaf
        // branch). d=0 eq is unused (no equality for the LSB digit per
        // millionaire); leave eq_sh[0..num_cmps) as-is.
        (void)greater_than;

        // Generate num_triples*num_cmps _2ROT triples (millionaire.h:347,354).
        // For num_digits==1 (bitlength<=beta, e.g. l=4), num_triples==0 and
        // the AND-tree is skipped entirely — res is just the single-digit
        // leaf cmp (millionaire.h:82-122 early-exit equivalent). Guard the
        // triple gen + AND-tree on n_trips>0.
        const int n_trips = num_triples * num_cmps;
        // BitTripleGen produces UNPACKED triples (1 byte/triple, value 0/1).
        // The Beaver AND gate (and_step1/2) operates on PACKED byte arrays
        // (8 bits/byte), matching SCI's Triple(packed=true). So pack ta/tb/tc
        // into ai/bi/ci (n_trips/8 bytes) via bool_to_uint8.
        std::vector<uint8_t> tu((size_t)std::max(0, n_trips)),
                               tv((size_t)std::max(0, n_trips)),
                               tw((size_t)std::max(0, n_trips));
        if (n_trips > 0) {
            s = triple_.generate(tu.data(), tv.data(), tw.data(), n_trips);
            if (s != OTStatus::kOk) return s;
        }
        const int nbytes = n_trips / 8;
        std::vector<uint8_t> ai((size_t)std::max(0, nbytes)),
                              bi((size_t)std::max(0, nbytes)),
                              ci((size_t)std::max(0, nbytes));
        for (int i = 0; i < n_trips; i += 8)
            ai[i / 8] = bool_to_uint8(tu.data() + i, 8);
        for (int i = 0; i < n_trips; i += 8)
            bi[i / 8] = bool_to_uint8(tv.data() + i, 8);
        for (int i = 0; i < n_trips; i += 8)
            ci[i / 8] = bool_to_uint8(tw.data() + i, 8);

        // AND-tree (millionaire.h:373-540). Only runs if num_triples>0
        // (num_digits>1). For num_digits==1 the leaf cmp IS the result.
        if (num_triples > 0) {
        // Packed 8-per-byte Beaver gates. ei/fi/e/f byte arrays.
        std::vector<uint8_t> ei((size_t)nbytes), fi((size_t)nbytes),
                              e((size_t)nbytes), f((size_t)nbytes);
        // The ei/fi exchange uses ONE agreed socket: ALICE prov_send (server
        // port2) <-> BOB prov_recv (client port2) — same port2.
        emp::NetIO* io = (role_ == OTParty::kSender) ? prov_send_.net_io()
                                                     : prov_recv_.net_io();
        if (io == nullptr) return OTStatus::kConfig;

        int counter_std = 0, old_counter_std = 0;
        int counter_corr = 0, old_counter_corr = 0;
        int counter_combined = 0, old_counter_combined = 0;
        // helper: byte offset for triple index t (packed 8/cmp).
        auto toff = [&](int t) { return (t * num_cmps) / 8; };

        for (int i = 1; i < num_digits; i *= 2) {
            // AND_step_1 for all (j, j+i) pairs at this level.
            // TRIPLE LAYOUT (millionaire.h, USE_CHEETAH): there is ONE flat
            // triples_std array of num_triples entries. std-ANDs consume
            // entries [counter_std] (incremented per std-AND); corr-ANDs
            // consume entries [num_triples_std + 2*counter_corr] and
            // [...+1] (2 per corr-pair). ei/fi/e/f mirror this layout:
            // std region [0, num_triples_std), corr region
            // [num_triples_std, num_triples). counter_combined tracks the
            // TOTAL triple consumed for the a/b/c arrays (which are flat);
            // counter_std / counter_corr track the ei/fi regions separately.
            for (int j = 0; j < num_digits && j + i < num_digits; j += 2 * i) {
                if (j == 0) {
                    // std: AND(cmp[0], eq[i]).
                    and_step1(ei.data() + toff(counter_std),
                              fi.data() + toff(counter_std),
                              cmp_sh.data(),            // x = cmp[0]
                              eq_sh.data() + (size_t)i * num_cmps,  // y = eq[i]
                              ai.data() + toff(counter_combined),
                              bi.data() + toff(counter_combined),
                              num_cmps);
                    counter_combined++; counter_std++;
                } else {
                    // corr: 2 ANDs. (1) cmp[j] & eq[j+i] -> triple [base]
                    // (2) eq[j] & eq[j+i] -> triple [base+1], base=num_triples_std+2*counter_corr
                    int base = num_triples_std + 2 * counter_corr;
                    and_step1(ei.data() + toff(base),
                              fi.data() + toff(base),
                              cmp_sh.data() + (size_t)j * num_cmps,
                              eq_sh.data() + (size_t)(j + i) * num_cmps,
                              ai.data() + toff(counter_combined),
                              bi.data() + toff(counter_combined),
                              num_cmps);
                    counter_combined++;
                    and_step1(ei.data() + toff(base + 1),
                              fi.data() + toff(base + 1),
                              eq_sh.data() + (size_t)j * num_cmps,
                              eq_sh.data() + (size_t)(j + i) * num_cmps,
                              ai.data() + toff(counter_combined),
                              bi.data() + toff(counter_combined),
                              num_cmps);
                    counter_combined++; counter_corr++;
                }
            }
            // exchange ei,fi for this level's range (millionaire.h:435-454).
            int off_std = (old_counter_std * num_cmps) / 8;
            int sz_std = ((counter_std - old_counter_std) * num_cmps) / 8;
            int off_corr = (num_triples_std + 2 * old_counter_corr) * num_cmps / 8;
            int sz_corr = (2 * (counter_corr - old_counter_corr) * num_cmps) / 8;
            if (role_ == OTParty::kSender) {  // ALICE
                io->send_data(ei.data() + off_std, sz_std);
                io->send_data(ei.data() + off_corr, sz_corr);
                io->send_data(fi.data() + off_std, sz_std);
                io->send_data(fi.data() + off_corr, sz_corr);
                io->recv_data(e.data() + off_std, sz_std);
                io->recv_data(e.data() + off_corr, sz_corr);
                io->recv_data(f.data() + off_std, sz_std);
                io->recv_data(f.data() + off_corr, sz_corr);
            } else {  // BOB
                io->recv_data(e.data() + off_std, sz_std);
                io->recv_data(e.data() + off_corr, sz_corr);
                io->recv_data(f.data() + off_std, sz_std);
                io->recv_data(f.data() + off_corr, sz_corr);
                io->send_data(ei.data() + off_std, sz_std);
                io->send_data(ei.data() + off_corr, sz_corr);
                io->send_data(fi.data() + off_std, sz_std);
                io->send_data(fi.data() + off_corr, sz_corr);
            }
            // reconstruct e = ei_A ^ ei_B, f = fi_A ^ fi_B (XOR shares).
            for (int k = 0; k < sz_std; ++k) {
                e[off_std + k] ^= ei[off_std + k];
                f[off_std + k] ^= fi[off_std + k];
            }
            for (int k = 0; k < sz_corr; ++k) {
                e[off_corr + k] ^= ei[off_corr + k];
                f[off_corr + k] ^= fi[off_corr + k];
            }

            // reset counters for AND_step_2 pass (millionaire.h:464-467).
            counter_std = old_counter_std;
            counter_corr = old_counter_corr;
            counter_combined = old_counter_combined;

            // AND_step_2 + update for all pairs at this level.
            // e/f use the SAME std/corr region split as ei/fi (step1).
            // a/b/c use counter_combined (flat). SCI millionaire.h:472-540.
            for (int j = 0; j < num_digits && j + i < num_digits; j += 2 * i) {
                if (j == 0) {
                    // cmp[0] = AND(cmp[0], eq[i]) ^ cmp[i]
                    uint8_t z[(num_cmps + 7) / 8];
                    and_step2(z, e.data() + toff(counter_std),
                              f.data() + toff(counter_std),
                              ei.data() + toff(counter_std),
                              fi.data() + toff(counter_std),
                              ai.data() + toff(counter_combined),
                              bi.data() + toff(counter_combined),
                              ci.data() + toff(counter_combined), num_cmps);
                    counter_combined++; counter_std++;
                    for (int c = 0; c < num_cmps; c += 8) {
                        uint8_t zc = z[c / 8];
                        uint8_t zi[8]; uint8_to_bool(zi, zc, 8);
                        for (int b = 0; b < 8 && c + b < num_cmps; ++b)
                            cmp_sh[c + b] = (uint8_t)(zi[b] ^
                                cmp_sh[(size_t)i * num_cmps + c + b]);
                    }
                } else {
                    // (1) cmp[j] = AND(cmp[j],eq[j+i]) ^ cmp[j+i]
                    // (2) eq[j]  = AND(eq[j],eq[j+i])
                    int base = num_triples_std + 2 * counter_corr;
                    uint8_t z1[(num_cmps + 7) / 8], z2[(num_cmps + 7) / 8];
                    and_step2(z1, e.data() + toff(base), f.data() + toff(base),
                              ei.data() + toff(base), fi.data() + toff(base),
                              ai.data() + toff(counter_combined),
                              bi.data() + toff(counter_combined),
                              ci.data() + toff(counter_combined), num_cmps);
                    counter_combined++;
                    and_step2(z2, e.data() + toff(base + 1),
                              f.data() + toff(base + 1),
                              ei.data() + toff(base + 1),
                              fi.data() + toff(base + 1),
                              ai.data() + toff(counter_combined),
                              bi.data() + toff(counter_combined),
                              ci.data() + toff(counter_combined), num_cmps);
                    counter_combined++; counter_corr++;
                    // cmp[j] = z1 ^ cmp[j+i]; eq[j] = z2
                    for (int c = 0; c < num_cmps; c += 8) {
                        uint8_t z1c = z1[c / 8]; uint8_t z1i[8]; uint8_to_bool(z1i, z1c, 8);
                        for (int b = 0; b < 8 && c + b < num_cmps; ++b)
                            cmp_sh[(size_t)j * num_cmps + c + b] =
                                z1i[b] ^ cmp_sh[(size_t)(j + i) * num_cmps + c + b];
                        uint8_t z2c = z2[c / 8]; uint8_t z2i[8]; uint8_to_bool(z2i, z2c, 8);
                        for (int b = 0; b < 8 && c + b < num_cmps; ++b)
                            eq_sh[(size_t)j * num_cmps + c + b] = z2i[b];
                    }
                }
            }
            old_counter_std = counter_std;
            old_counter_corr = counter_corr;
            old_counter_combined = counter_combined;
        }
        }  // end if (num_triples > 0)

        // res = cmp[0] (millionaire.h:314-315).
        for (int i = 0; i < old_num_cmps; ++i)
            res[i] = cmp_sh[i] & 1;
        return OTStatus::kOk;
    }

   private:
    // Split the 2-bit leaf outputs into separate cmp/eq share planes.
    // leaf_res[i] = (mcmp<<1)|meq for output i (= digit d, compare c, i=d*num_cmps+c).
    // cmp_sh[d*num_cmps+c] = leaf_res[i]>>1 (cmp mask share), eq_sh[...] = leaf_res[i]&1.
    // Mirrors millionaire.h:305-309. (d=0 eq is unused downstream but filled here
    // for uniformity.)
    static void split_leaf_(const std::vector<uint8_t>& leaf_res,
                            std::vector<uint8_t>& cmp_sh,
                            std::vector<uint8_t>& eq_sh,
                            int num_digits, int num_cmps) {
        for (int d = 0; d < num_digits; ++d) {
            for (int c = 0; c < num_cmps; ++c) {
                int i = d * num_cmps + c;
                cmp_sh[(size_t)d * num_cmps + c] = (uint8_t)((leaf_res[i] >> 1) & 1);
                eq_sh[(size_t)d * num_cmps + c]  = (uint8_t)(leaf_res[i] & 1);
            }
        }
    }
    // Beaver AND step 1 (millionaire.h:565-575): ei = a ^ pack(x), fi = b ^ pack(y).
    // x,y,a,b are bool-byte arrays (num_ANDs = num_cmps, multiple of 8).
    static void and_step1(uint8_t* ei, uint8_t* fi, const uint8_t* x,
                          const uint8_t* y, const uint8_t* a, const uint8_t* b,
                          int num_ANDs) {
        for (int i = 0; i < num_ANDs; i += 8) {
            ei[i / 8] = a[i / 8] ^ bool_to_uint8(x + i, 8);
            fi[i / 8] = b[i / 8] ^ bool_to_uint8(y + i, 8);
        }
    }
    // Beaver AND step 2 (millionaire.h:576-591): z = (ALICE: e&f) ^ f&a ^ e&b ^ c.
    // Output z as packed bytes (num_ANDs/8). num_ANDs multiple of 8.
    void and_step2(uint8_t* z, const uint8_t* e, const uint8_t* f,
                   const uint8_t* /*ei*/, const uint8_t* /*fi*/,
                   const uint8_t* a, const uint8_t* b, const uint8_t* c,
                   int num_ANDs) {
        for (int i = 0; i < num_ANDs; i += 8) {
            uint8_t tz;
            if (role_ == OTParty::kSender)  // ALICE
                tz = (uint8_t)(e[i / 8] & f[i / 8]);
            else
                tz = 0;
            tz = (uint8_t)(tz ^ (f[i / 8] & a[i / 8]));
            tz = (uint8_t)(tz ^ (e[i / 8] & b[i / 8]));
            tz = (uint8_t)(tz ^ c[i / 8]);
            z[i / 8] = tz;
        }
    }

    CuotProvider& prov_send_;
    CuotProvider& prov_recv_;
    OTParty role_;
    KkotLeafCmp leaf_send_;
    KkotLeafCmp leaf_recv_;
    BitTripleGen triple_;
};

}  // namespace gpu_mm

#endif  // GPU_MM_CUOT_COMPARE_H
