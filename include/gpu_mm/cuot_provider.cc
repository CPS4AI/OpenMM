// CuotProvider implementation (Task M2-T2).
//
// Host C++. Compiled by g++ (NOT nvcc): set_source_files_properties in the
// add_cuot_test macro forces LANGUAGE CXX for this file. It references symbols
// defined in the cuOT .cu sources (cuda_setdev) and emp::FerretCOT templates
// (header-only); those .cu files are compiled into the same exe by nvcc.
#include "gpu_mm/cuot_provider.h"
#include "gpu_mm/cot_pack.h"  // gpu_mm::pack/unpack_cot_messages (M2-T3)

#include "emp-ot/ferret/ferret_cot.h"   // emp::FerretCOT<emp::NetIO>
#include "emp-ot/ferret/constants.h"     // emp::ferret_b13, PRE_OT_DATA_REG_*_FILE
#include "emp-ot/ferret/dev_layer.h"     // cuda_setdev, Mat (via gpu_matrix.h)
#include <emp-tool/emp-tool.h>           // emp::NetIO, emp::block, zero_block, getLSB, MITCCRH
#include <cuda_runtime.h>               // cudaMalloc/Memcpy/Free (host-callable, g++ ok)

// GPU kernel launchers (defined in cuot_kernels.cu, nvcc TU). Forward-declared
// here so this g++ TU can call them as ordinary host functions (the <<<>>>
// lives inside the .cu definitions, not here).
namespace gpu_mm::gpu_kern {
void launch_diff_vector_pack8(const blk* r, const uint8_t* b,
                              uint8_t* d_packed, int nbytes);
void launch_apply_diff_vector(blk* r, const uint8_t* d_packed,
                              const blk* delta, int n);
}

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gpu_mm {

namespace {
inline int sci_party(OTParty r) { return r == OTParty::kSender ? 1 : 2; }  // ALICE=1, BOB=2
}  // namespace

CuotProvider::CuotProvider(OTParty role, int port, int gpu, const char* address,
                           bool run_setup, const std::string& pre_file)
    : role_(role), gpu_(gpu) {
    // cuOT convention (mirrors ferret/emp-ot/test/ferret.cpp): ALICE is the
    // server (NetIO(nullptr, port)), BOB is the client (NetIO("127.0.0.1", port)).
    const char* addr = (role == OTParty::kSender) ? nullptr : address;

    // Pin the CUDA device BEFORE anything allocates GPU memory. Ferret's
    // ctor runs setup() (base-OT + initial extend) which calls GPUdata::resize
    // -> cudaMalloc on the *currently active* device. Two parties must use
    // distinct GPUs (e.g. 0 and 1); rcot uses only the local GPU + TCP, so no
    // IPC / peer access is needed.
    cuda_setdev(gpu_);

    try {
        io_ = std::make_unique<emp::NetIO>(addr, port);
        // CRITICAL: cuOT's FerretCOT stores `this->ios = ios` (the T** array
        // POINTER, not a copy) and MpcotReg/OTPre dereference ios[0] LATER
        // (during rcot's extend), not just in the ctor. cuOT's own test passes
        // `&io` where `io` is a main()-local whose lifetime covers rcot. If we
        // pass a ctor-local array, it dies when the ctor returns, leaving
        // ferret_->ios dangling — extend then reads a stale NetIO* -> stream
        // member is garbage -> fwrite(bad fp) SIGSEGV (gdb bt: OTPre::send ->
        // send_block -> send_data -> fwrite). So pass the ADDRESS of the
        // MEMBER array ios_, which outlives FerretCOT. (Earlier I kept a
        // ctor-local `emp::NetIO* ios[1]` and passed THAT — the member ios_
        // was set but unused; the dangling local was the actual bug.)
        ios_[0] = io_.get();
        // FerretCOT(mlParty, otParty, ios, malicious, run_setup, param, pre_file, log_file).
        // mlParty is unused for OT logic; pass the same as otParty. param=ferret_b13
        // (the default/only tuned preset). pre_file (default "") -> the standard
        // ./data/pre_ot_data_reg_{send,recv} cache path from constants.h; a
        // non-empty override lets one party own TWO FerretCOTs (bit-triple gen)
        // without cache cross-contamination / the recycle bug (ferret_cot.hpp:99).
        ferret_ = std::make_unique<emp::FerretCOT<emp::NetIO>>(
            sci_party(role), sci_party(role), ios_.data(), /*malicious=*/false,
            run_setup, emp::ferret_b13, pre_file, /*log_file=*/"");
        ready_ = true;
    } catch (const std::exception& e) {
        std::cerr << "[cuot] init failed: " << e.what() << std::endl;
        ready_ = false;
    }
}

CuotProvider::~CuotProvider() {
    // ferret_ must die before io_ (it holds a raw NetIO*). unique_ptr members
    // are destroyed in reverse declaration order: ferret_ then io_. Good.
    ferret_.reset();
    io_.reset();
}

emp::NetIO* CuotProvider::net_io() { return io_.get(); }

OTStatus CuotProvider::rcot_blocks(uint8_t* out, int64_t n) {
    if (!ready_ || out == nullptr || n <= 0) return OTStatus::kInvalidArg;
    try {
        // Re-pin the device in case the host thread changed it since ctor.
        cuda_setdev(gpu_);
        // Mat is cuOT's GPU matrix type (global scope, from gpu_matrix.h,
        // pulled in transitively via ferret_cot.h). n blocks x 16 bytes.
        Mat data({(uint64_t)n});
        ferret_->rcot(data, n);
        data.write_to_cpu(out, data.size_bytes());
    } catch (const std::exception& e) {
        std::cerr << "[cuot] rcot failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// OPT-1: GPU-resident RCOT — fill the caller's GPU Mat with n RCOT blocks,
// NO D2H (rcot_blocks does write_to_cpu; this does not). The blocks stay on
// the GPU for the MITCCRH/Beaver kernels to consume by device pointer.
OTStatus CuotProvider::rcot_blocks_gpu(Mat& out, int64_t n) {
    if (!ready_ || n <= 0) return OTStatus::kInvalidArg;
    try {
        cuda_setdev(gpu_);
        // out must be pre-sized to n blocks by the caller. ferret_->rcot
        // writes n blocks (16n bytes) starting at out.data(). (Mirrors
        // rcot_blocks but skips the write_to_cpu.)
        ferret_->rcot(out, n);
    } catch (const std::exception& e) {
        std::cerr << "[cuot] rcot_blocks_gpu failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// OPT-1/2: GPU-resident block chosen-COT. RCOT stays in `out` (Mat, GPU); the
// diff-vector is computed + bit-packed on the GPU (OPT-2) and exchanged as
// n/8 bytes. Only n/8 bytes touch the host<->net; blocks never leave the GPU.
// Mirrors send_cot_blocks (cuot_provider.cc:156) but GPU-resident.
OTStatus CuotProvider::send_cot_blocks_gpu(Mat& out, int64_t n) {
    if (!ready_ || role_ != OTParty::kSender || n <= 0) return OTStatus::kInvalidArg;
    try {
        cuda_setdev(gpu_);
        // x = rcot(n) into the GPU Mat (no D2H).
        OTStatus s = rcot_blocks_gpu(out, n);
        if (s != OTStatus::kOk) return s;
        // recv packed diff (n/8 bytes) from receiver.
        int nbytes = (int)((n + 7) / 8);
        std::vector<uint8_t> d_packed((size_t)nbytes);
        io_->recv_data(d_packed.data(), nbytes);
        // upload d_packed to a small device buffer, apply_diff_vector on GPU.
        uint8_t* d_dev = nullptr;
        cudaMalloc(&d_dev, nbytes);
        cudaMemcpy(d_dev, d_packed.data(), nbytes, cudaMemcpyHostToDevice);
        // Delta as a device blk.
        emp::block delta_h = ferret_->Delta;
        blk delta_d;
        std::memcpy(&delta_d, &delta_h, sizeof(blk));
        blk* delta_dev = nullptr;
        cudaMalloc(&delta_dev, sizeof(blk));
        cudaMemcpy(delta_dev, &delta_d, sizeof(blk), cudaMemcpyHostToDevice);
        gpu_mm::gpu_kern::launch_apply_diff_vector(out.data(), d_dev, delta_dev, (int)n);
        cudaDeviceSynchronize();
        cudaFree(d_dev); cudaFree(delta_dev);
    } catch (const std::exception& e) {
        std::cerr << "[cuot] send_cot_blocks_gpu failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus CuotProvider::recv_cot_blocks_gpu(Mat& out, const uint8_t* b_dev, int64_t n) {
    if (!ready_ || role_ != OTParty::kReceiver || b_dev == nullptr || n <= 0)
        return OTStatus::kInvalidArg;
    try {
        cuda_setdev(gpu_);
        // r = rcot(n) into the GPU Mat (no D2H).
        OTStatus s = rcot_blocks_gpu(out, n);
        if (s != OTStatus::kOk) return s;
        // OPT-2: diff-vector on GPU, bit-packed (n/8 bytes). b_dev is already
        // on the device (caller's choice bits). d_i = lsb(r_i) ^ b_i.
        int nbytes = (int)((n + 7) / 8);
        uint8_t* d_packed_dev = nullptr;
        cudaMalloc(&d_packed_dev, nbytes);
        gpu_mm::gpu_kern::launch_diff_vector_pack8(out.data(), b_dev,
                                                   d_packed_dev, nbytes);
        cudaDeviceSynchronize();
        // download just the packed diff (n/8 bytes) and send to sender.
        std::vector<uint8_t> d_packed((size_t)nbytes);
        cudaMemcpy(d_packed.data(), d_packed_dev, nbytes, cudaMemcpyDeviceToHost);
        cudaFree(d_packed_dev);
        io_->send_data(d_packed.data(), nbytes);
        io_->flush();
        // out (the GPU Mat) already holds r; that's the receiver's chosen-COT.
    } catch (const std::exception& e) {
        std::cerr << "[cuot] recv_cot_blocks_gpu failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus CuotProvider::delta_block(uint8_t out[16]) const {
    if (!ready_ || role_ != OTParty::kSender) return OTStatus::kInvalidArg;
    // Delta is emp::block = __m128i. Store 16 bytes via memcpy (safe, no UB).
    emp::block d = ferret_->Delta;
    std::memcpy(out, &d, 16);
    return OTStatus::kOk;
}

// Block-level chosen COT (paper §III-D). Reuses rcot_blocks + a diff-vector
// exchange over this provider's own NetIO — NOT cuOT's IPC online path. The
// protocol is symmetric up to rcot, then:
//   receiver: d_i = getLSB(r_i) XOR b_i  -> send d (n bits) to sender
//   sender:   recv d; x'_i = x_i ^ (d_i ? Delta : 0)
// Correlation: r_i == x'_i XOR (b_i * Delta). d is a one-time pad (LSB(r) is
// the receiver's private uniform bit, unknown to the sender), so the sender
// learns nothing about b; the receiver learns nothing about Delta online.
OTStatus CuotProvider::recv_cot_blocks(uint8_t* out, const bool* b, int64_t n) {
    if (!ready_ || role_ != OTParty::kReceiver || out == nullptr ||
        b == nullptr || n <= 0) {
        return OTStatus::kInvalidArg;
    }
    try {
        cuda_setdev(gpu_);
        // r = rcot(n). r[i] == x[i] ^ (lsb(r[i]) * Delta) at this point.
        OTStatus s = rcot_blocks(out, n);
        if (s != OTStatus::kOk) return s;
        // d_i = getLSB(r_i) XOR b_i; pack bits, send to sender.
        std::vector<uint8_t> d((size_t)n);
        const emp::block* r = reinterpret_cast<const emp::block*>(out);
        for (int64_t i = 0; i < n; ++i)
            d[i] = (uint8_t)(emp::getLSB(r[i]) ^ (b[i] ? 1 : 0));
        io_->send_data(d.data(), (int)n);
        io_->flush();
        // out already holds r; that's the receiver's chosen-COT output.
    } catch (const std::exception& e) {
        std::cerr << "[cuot] recv_cot_blocks failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus CuotProvider::send_cot_blocks(uint8_t* out, int64_t n) {
    if (!ready_ || role_ != OTParty::kSender || out == nullptr || n <= 0) {
        return OTStatus::kInvalidArg;
    }
    try {
        cuda_setdev(gpu_);
        // x = rcot(n). x[i] is the raw sender block; receiver holds
        // r[i] == x[i] ^ (lsb(r[i]) * Delta).
        OTStatus s = rcot_blocks(out, n);
        if (s != OTStatus::kOk) return s;
        // recv d (n bits) from receiver.
        std::vector<uint8_t> d((size_t)n);
        io_->recv_data(d.data(), (int)n);
        // x'_i = x_i ^ (d_i ? Delta : 0). After this, r_i == x'_i ^ (b_i*Delta).
        emp::block delta = ferret_->Delta;
        emp::block* x = reinterpret_cast<emp::block*>(out);
        for (int64_t i = 0; i < n; ++i)
            if (d[i]) x[i] = x[i] ^ delta;
    } catch (const std::exception& e) {
        std::cerr << "[cuot] send_cot_blocks failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// Block-level random OT (ROT) — M3-T6 Phase B (_2ROT bit-triple). Re-derives
// emp COT<T>::send_rot/recv_rot (deps/emp-tool/.../cot.h) on top of our
// send_cot_blocks/recv_cot_blocks, because cuOT's inherited send_rot calls the
// no-op send_cot(block*). Mirrors cot.h send_rot/recv_rot + silent_ot.h
// send_ot_rm_rc/recv_ot_rm_rc (the 2-ROT decorrelation): a chosen-COT with the
// receiver's RANDOM choice bit r, then MITCCRH hash<ot_bsize,2> (sender, gives
// (m0,m1)) / hash<ot_bsize,1> (receiver, gives m_{r}) to decorrelate the RCOT
// correlation into independent random messages. Seed sync: sender picks s,
// sends to receiver (cot.h / silent_ot.h:507).
//
// Correlation: sender holds (m0_i, m1_i); receiver holds m_{r_i} == m0_i if
// r_i==0 else m1_i, with m0,m1 independent random blocks. This is the standard
// 1-2 ROT that Beaver's AND-triple needs (bit-triple-generator.h:184-199).
//
// CAVEAT: inherits cuOT's ~2% RCOT floor (M2-T2). A triple uses TWO ROTs (one
// per direction), so the triple error rate compounds to ~4%.
OTStatus CuotProvider::send_rot_blocks(uint8_t* out, int64_t n) {
    if (!ready_ || role_ != OTParty::kSender || out == nullptr || n <= 0) {
        return OTStatus::kInvalidArg;
    }
    try {
        cuda_setdev(gpu_);
        // data0 = send_cot_blocks(n): the adjusted sender blocks x' for the
        // receiver's random choice r. (recv_cot_blocks on the peer drew r and
        // sent the diff vector.) x'[i] is the "0-branch" base.
        emp::block* data0 = reinterpret_cast<emp::block*>(out);
        OTStatus s = send_cot_blocks(out, n);
        if (s != OTStatus::kOk) return s;

        // MITCCRH seed: sender picks s, sends to receiver (cot.h send_rot /
        // silent_ot.h:507). Use a FRESH local MITCCRH (NOT ferret_->mitccrh):
        // emp::MITCCRH::setS only resets start_point, NOT gid/key_used, so a
        // shared ferret_->mitccrh carries divergent gid state from setup +
        // prior OT calls -> sender/receiver renew with different keys ->
        // m_{r} != m0/m1 -> ~100% wrong (confirmed: 99.99% bad before this
        // fix). A local MITCCRH starts at gid=0,key_used=BatchSize on BOTH
        // sides, so setS+hash stay byte-synchronized.
        emp::block seed;
        ferret_->prg.random_block(&seed, 1);
        io_->send_block(&seed, 1);
        emp::MITCCRH<emp::ot_bsize> rot_mitccrh;
        rot_mitccrh.setS(seed);
        io_->flush();

        // Decorrelate: m0 = H(x'), m1 = H(x' ^ Delta), independent random
        // blocks (cot.h:67-80). Write interleaved [m0,m1] per element.
        // CRITICAL: write to a SEPARATE output (out, 2n blocks) reading from a
        // COPY of x' — writing interleaved [m0,m1] in-place on data0 would
        // clobber later batches' x' source (batch 0 writes [0,16), batch 1
        // reads [8,16) -> reads decorrelated m0/m1, not x' -> ~100% wrong,
        // only batch 0 survives = exactly 8/100000 correct, the symptom that
        // localized this bug).
        std::vector<emp::block> base(data0, data0 + n);  // copy of x'
        emp::block pad[2 * emp::ot_bsize];
        for (int64_t i = 0; i < n; i += emp::ot_bsize) {
            int64_t bsz = std::min((int64_t)emp::ot_bsize, n - i);
            for (int64_t j = i; j < i + bsz; ++j) {
                pad[2 * (j - i)] = base[j];
                pad[2 * (j - i) + 1] = base[j] ^ ferret_->Delta;
            }
            rot_mitccrh.template hash<emp::ot_bsize, 2>(pad);
            for (int64_t j = i; j < i + bsz; ++j) {
                data0[2 * j] = pad[2 * (j - i)];        // m0
                data0[2 * j + 1] = pad[2 * (j - i) + 1]; // m1
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[cuot] send_rot_blocks failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus CuotProvider::recv_rot_blocks(uint8_t* out, const bool* r, int64_t n) {
    if (!ready_ || role_ != OTParty::kReceiver || out == nullptr ||
        r == nullptr || n <= 0) {
        return OTStatus::kInvalidArg;
    }
    try {
        cuda_setdev(gpu_);
        // data = recv_cot_blocks(r, n): receiver's chosen-COT output for its
        // random choice r. data[i] = x'[i] ^ (r_i * Delta).
        emp::block* data = reinterpret_cast<emp::block*>(out);
        OTStatus s = recv_cot_blocks(out, r, n);
        if (s != OTStatus::kOk) return s;

        // MITCCRH seed from sender (cot.h recv_rot / silent_ot.h:452). FRESH
        // local MITCCRH (see send_rot_blocks: shared ferret_->mitccrh has
        // divergent gid -> ~100% wrong). gid starts at 0 on both sides here.
        emp::block seed;
        io_->recv_block(&seed, 1);
        emp::MITCCRH<emp::ot_bsize> rot_mitccrh;
        rot_mitccrh.setS(seed);

        // Decorrelate: m_{r} = H(data) (cot.h:90-99). data already encodes the
        // chosen branch via r, so a single hash<ot_bsize,1> suffices.
        emp::block pad[emp::ot_bsize];
        for (int64_t i = 0; i < n; i += emp::ot_bsize) {
            int64_t bsz = std::min((int64_t)emp::ot_bsize, n - i);
            std::memcpy(pad, data + i, bsz * sizeof(emp::block));
            rot_mitccrh.template hash<emp::ot_bsize, 1>(pad);
            std::memcpy(data + i, pad, bsz * sizeof(emp::block));
        }
    } catch (const std::exception& e) {
        std::cerr << "[cuot] recv_rot_blocks failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// recv_ot_cam_cc / _prime (SCI/src/OT/ferret/silent_ot.h:85-242). The only
// substitution: SCI's send_ot_rcm_cc/recv_ot_rcm_cc call ferret->send_cot/
// recv_cot (block*) which cuOT stubs as a NO-OP (ferret_cot.h:41) — we use our
// own send_cot_blocks/recv_cot_blocks (the diff-vector block chosen-COT) for
// the raw block-RCOT instead. Everything else (mitccrh, Delta, prg, io,
// ot_bsize, _mm_extract_epi64) is reused as-is from ferret_.
//
// Correlation (matching SCI / M&M): sender picks corr, gets random x=data0;
// receiver with bit b gets x (b=0) or x+corr (b=1) mod K. i.e. y_1-y_0=corr.
//
// CAVEAT: inherits cuOT's ~2% RCOT error floor (M2-T2 shelved,
// docs/research/m2-cuot-standalone.md §7.4). These methods validate the
// adapter's MITCCRH/packing logic; they are NOT a cuOT end-to-end correctness
// proof.

namespace {
// Pull the low 64 bits of an emp::block (__m128i). SSE4.1 intrinsic; the cuot
// target compiles with -msse4.1 (FindcuOT.cmake). Mirrors SCI's
// _mm_extract_epi64(pad[..], 0).
inline uint64_t lo64(const emp::block& b) { return _mm_extract_epi64(b, 0); }
}  // namespace

// --- COT over Z_{2^l} (ring) : send_ot_cam_cc (silent_ot.h:85-127) ---------
OTStatus CuotProvider::send_cot(uint64_t* data0, const uint64_t* corr, int n,
                                int l) {
    if (!ready_ || role_ != OTParty::kSender || data0 == nullptr ||
        corr == nullptr || n <= 0 || l <= 0 || l > 64) {
        return OTStatus::kInvalidArg;
    }
    const uint64_t modulo_mask = (l == 64) ? (uint64_t)-1 : ((1ULL << l) - 1);
    try {
        cuda_setdev(gpu_);
        // Raw block chosen-COT (substitutes the no-op ferret_->send_cot).
        // send_cot_blocks fills `rcm` with the sender's adjusted blocks x'.
        emp::block* rcm = new emp::block[n];
        OTStatus s = send_cot_blocks(reinterpret_cast<uint8_t*>(rcm), n);
        if (s != OTStatus::kOk) { delete[] rcm; return s; }

        // MITCCRH seed: sender picks s, sends to receiver (matches SCI).
        emp::block seed;
        ferret_->prg.random_block(&seed, 1);
        io_->send_block(&seed, 1);
        ferret_->mitccrh.setS(seed);
        io_->flush();

        emp::block pad[2 * emp::ot_bsize];
        const uint32_t y_size =
            (uint32_t)ceil((emp::ot_bsize * l) / (float(64)));
        uint64_t y[y_size];
        uint64_t corr_data[emp::ot_bsize];

        for (int i = 0; i < n; i += emp::ot_bsize) {
            int bsz = std::min((int64_t)emp::ot_bsize, (int64_t)(n - i));
            for (int j = 0; j < bsz; ++j) {
                pad[2 * j] = rcm[i + j];
                pad[2 * j + 1] = rcm[i + j] ^ ferret_->Delta;
            }
            ferret_->mitccrh.template hash<emp::ot_bsize, 2>(pad);
            for (int j = 0; j < bsz; ++j) {
                data0[i + j] = lo64(pad[2 * j]) & modulo_mask;
                corr_data[j] = (corr[i + j] + data0[i + j] +
                                lo64(pad[2 * j + 1])) & modulo_mask;
            }
            uint32_t corrected_y_size =
                (uint32_t)ceil((bsz * l) / ((float)sizeof(uint64_t) * 8));
            gpu_mm::pack_cot_messages(y, corr_data, corrected_y_size, bsz, l);
            io_->send_data(y, sizeof(uint64_t) * corrected_y_size);
        }
        io_->flush();
        delete[] rcm;
    } catch (const std::exception& e) {
        std::cerr << "[cuot] send_cot failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// --- COT over Z_{2^l} (ring) : recv_ot_cam_cc (silent_ot.h:168-207) -------
OTStatus CuotProvider::recv_cot(uint64_t* data, const bool* b, int n, int l) {
    if (!ready_ || role_ != OTParty::kReceiver || data == nullptr ||
        b == nullptr || n <= 0 || l <= 0 || l > 64) {
        return OTStatus::kInvalidArg;
    }
    const uint64_t modulo_mask = (l == 64) ? (uint64_t)-1 : ((1ULL << l) - 1);
    try {
        cuda_setdev(gpu_);
        // Raw block chosen-COT (substitutes the no-op ferret_->recv_cot).
        emp::block* rcm = new emp::block[n];
        OTStatus s = recv_cot_blocks(reinterpret_cast<uint8_t*>(rcm), b, n);
        if (s != OTStatus::kOk) { delete[] rcm; return s; }

        emp::block seed;
        io_->recv_block(&seed, 1);
        ferret_->mitccrh.setS(seed);

        emp::block pad[emp::ot_bsize];
        const uint32_t recvd_size =
            (uint32_t)ceil((emp::ot_bsize * l) / (float(64)));
        uint64_t corr_data[emp::ot_bsize];
        uint64_t recvd[recvd_size];

        for (int i = 0; i < n; i += emp::ot_bsize) {
            int bsz = std::min((int64_t)emp::ot_bsize, (int64_t)(n - i));
            uint32_t corrected_recvd_size =
                (uint32_t)ceil((bsz * l) / ((float)sizeof(uint64_t) * 8));
            io_->recv_data(recvd, sizeof(uint64_t) * corrected_recvd_size);
            memcpy(pad, rcm + i, bsz * sizeof(emp::block));
            ferret_->mitccrh.template hash<emp::ot_bsize, 1>(pad);
            gpu_mm::unpack_cot_messages(corr_data, recvd, bsz, l);
            for (int j = 0; j < bsz; ++j) {
                if (b[i + j])
                    data[i + j] = (corr_data[j] - lo64(pad[j])) & modulo_mask;
                else
                    data[i + j] = lo64(pad[j]) & modulo_mask;
            }
        }
        delete[] rcm;
    } catch (const std::exception& e) {
        std::cerr << "[cuot] recv_cot failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// --- COT over Z_p (field) : send_ot_cam_cc_prime (silent_ot.h:129-163) ----
OTStatus CuotProvider::send_cot_prime(uint64_t* data0, const uint64_t* corr,
                                      int n, uint64_t p) {
    if (!ready_ || role_ != OTParty::kSender || data0 == nullptr ||
        corr == nullptr || n <= 0 || p < 2) {
        return OTStatus::kInvalidArg;
    }
    try {
        cuda_setdev(gpu_);
        emp::block* rcm = new emp::block[n];
        OTStatus s = send_cot_blocks(reinterpret_cast<uint8_t*>(rcm), n);
        if (s != OTStatus::kOk) { delete[] rcm; return s; }

        emp::block seed;
        ferret_->prg.random_block(&seed, 1);
        io_->send_block(&seed, 1);
        ferret_->mitccrh.setS(seed);
        io_->flush();

        const int l = (int)ceil(log2((double)p) / 8.0);  // byte width per corr
        emp::block pad[2 * emp::ot_bsize];
        uint64_t corr_data[emp::ot_bsize];
        uint8_t* send_buff = new uint8_t[emp::ot_bsize * l];

        for (int i = 0; i < n; i += emp::ot_bsize) {
            int bsz = std::min((int64_t)emp::ot_bsize, (int64_t)(n - i));
            for (int j = 0; j < bsz; ++j) {
                pad[2 * j] = rcm[i + j];
                pad[2 * j + 1] = rcm[i + j] ^ ferret_->Delta;
            }
            ferret_->mitccrh.template hash<emp::ot_bsize, 2>(pad);
            for (int j = 0; j < bsz; ++j) {
                __uint128_t tmp0 = (*(__uint128_t*)&pad[2 * j]) % p;
                __uint128_t tmp1 = (*(__uint128_t*)&pad[2 * j + 1]) % p;
                data0[i + j] = static_cast<uint64_t>(tmp0);
                corr_data[j] =
                    static_cast<uint64_t>((tmp0 + tmp1 + corr[i + j]) % p);
                std::memcpy(&send_buff[j * l], &corr_data[j], l);
            }
            io_->send_data(send_buff, bsz * l);
        }
        io_->flush();
        delete[] rcm;
        delete[] send_buff;
    } catch (const std::exception& e) {
        std::cerr << "[cuot] send_cot_prime failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

// --- COT over Z_p (field) : recv_ot_cam_cc_prime (silent_ot.h:209-242) ----
OTStatus CuotProvider::recv_cot_prime(uint64_t* data, const bool* b, int n,
                                      uint64_t p) {
    if (!ready_ || role_ != OTParty::kReceiver || data == nullptr ||
        b == nullptr || n <= 0 || p < 2) {
        return OTStatus::kInvalidArg;
    }
    try {
        cuda_setdev(gpu_);
        emp::block* rcm = new emp::block[n];
        OTStatus s = recv_cot_blocks(reinterpret_cast<uint8_t*>(rcm), b, n);
        if (s != OTStatus::kOk) { delete[] rcm; return s; }

        emp::block seed;
        io_->recv_block(&seed, 1);
        ferret_->mitccrh.setS(seed);

        const int l = (int)ceil(log2((double)p) / 8.0);
        emp::block pad[emp::ot_bsize];
        uint64_t corr_data[emp::ot_bsize];
        uint8_t* recv_buff = new uint8_t[emp::ot_bsize * l];

        for (int i = 0; i < n; i += emp::ot_bsize) {
            int bsz = std::min((int64_t)emp::ot_bsize, (int64_t)(n - i));
            memcpy(pad, rcm + i, bsz * sizeof(emp::block));
            ferret_->mitccrh.template hash<emp::ot_bsize, 1>(pad);
            io_->recv_data(recv_buff, bsz * l);
            for (int j = 0; j < bsz; ++j) {
                uint64_t tmp = 0;
                std::memcpy(&tmp, &recv_buff[j * l], l);
                corr_data[j] = tmp;
            }
            for (int j = 0; j < bsz; ++j) {
                __uint128_t tmp = (*(__uint128_t*)&pad[j]) % p;
                if (b[i + j])
                    data[i + j] =
                        static_cast<uint64_t>((corr_data[j] + (p - tmp)) % p);
                else
                    data[i + j] = static_cast<uint64_t>(tmp);
            }
        }
        delete[] rcm;
        delete[] recv_buff;
    } catch (const std::exception& e) {
        std::cerr << "[cuot] recv_cot_prime failed: " << e.what() << std::endl;
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

}  // namespace gpu_mm
