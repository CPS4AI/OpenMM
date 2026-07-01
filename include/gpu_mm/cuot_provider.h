// CuotProvider — cuOT (GPU Ferret) backed OTProvider (Task M2-T2).
//
// Wraps emp::FerretCOT<emp::NetIO> from deps/cuOT behind the gpu_mm::OTProvider
// interface. This is a standalone, test-only seam: it does NOT replace any
// M&M OT path (CLAUDE.md dev order step 4 — cuOT must pass standalone tests
// first). Built only when OPENMM_ENABLE_CUOT=ON.
//
// IMPORTANT — what cuOT actually provides (verified from deps/cuOT source):
//   - rcot(Mat& data, int64_t num): random correlated OT over GF(2^128).
//     Both parties call it. Sender gets random x_i; receiver gets
//     x_i ^ (lsb(x_i) * Delta) — the LSB of the receiver's own block is the
//     implicit choice bit. Pure GPU compute + TCP (no CUDA IPC). This is the
//     ONLY usable cuOT primitive on a <8-GPU box.
//   - send_cot/recv_cot (chosen COT): exist but the receiver online path uses
//     cudaMemcpyPeer with a HARDCODED 8-GPU topology (otherDev = dev<4?dev+4:dev-4),
//     which is invalid on this 4x A40 box. So chosen COT is NOT exposed here.
//   - cuOT has NO arithmetic COT (Z_{2^l} / Z_p); everything is GF(2^128)
//     block with XOR-Delta. The arithmetic send_cot(uint64_t*,...) inherited
//     from OTProvider therefore returns kUnsupported. The arithmetic adapter
//     (layering MITCCRH hashing on rcot) is M2-T3, not this task.
//
// Network model: same-machine two processes over TCP (emp::NetIO), each on its
// OWN GPU. The caller must pass distinct `gpu` indices (e.g. ALICE->0, BOB->1).
// cuda_setdev(gpu) is called BEFORE constructing FerretCOT, because
// GPUdata::resize() allocates on the currently-active device and the Ferret
// setup/bootstrap (base-OT + initial extend) touches the GPU in the ctor.
//
// Run from repo root so Ferret finds ./data/pre_ot_data_reg_* (cache); the
// bind() of the listening socket needs sudo on this box (see memory
// openmm-build-env §4 and run_cuot_test.sh).
#ifndef GPU_MM_CUOT_PROVIDER_H
#define GPU_MM_CUOT_PROVIDER_H

#include "gpu_mm/ot_provider.h"

#include <cstdint>
#include <memory>
#include <array>
#include <string>

namespace emp {
class NetIO;
template <typename T>
class FerretCOT;
}  // namespace emp

// cuOT's GPU block-array type (deps/cuOT/gpu/gpu_matrix.h), global scope.
// Forward-declared so cuot_provider.h can expose rcot_blocks_gpu(Mat&, ...)
// without pulling the full cuOT GPU headers (which would collide with the
// emp-tool headers on the cuot build target).
class Mat;

namespace gpu_mm {

class CuotProvider : public OTProvider {
   public:
    // Construct as `role` on `port`, pinned to GPU index `gpu` (the caller
    // must give the two parties DISTINCT gpus, e.g. sender->0, receiver->1).
    // Sender (ALICE) listens (address==nullptr), receiver (BOB) connects to
    // `address` (default 127.0.0.1). The matching peer must be constructed
    // concurrently (NetIO connects). run_setup=true runs Ferret bootstrap
    // (base-OT + initial extend) in the ctor, writing/reading the
    // ./data/pre_ot_data_reg_* cache.
    //
    // pre_file: Ferret pre-OT cache path override (empty = the default
    //   ./data/pre_ot_data_reg_{send,recv} from constants.h). A party that
    //   builds TWO FerretCOTs (e.g. bit-triple gen needs one sender + one
    //   receiver Ferret on the SAME party — opposite Delta ownership for the
    //   two ROT directions) MUST give them distinct pre_file so the cache-HIT
    //   path doesn't cross-contaminate (and the recycle bug, ferret_cot.hpp
    //   :99, writes a recycled seed into whichever file). Pass e.g.
    //   "data/pre_ot_data_reg_send_fwd"/"_recv_fwd" for the "reversed"
    //   (BOB-sends) Ferret.
    CuotProvider(OTParty role, int port, int gpu,
                 const char* address = "127.0.0.1", bool run_setup = true,
                 const std::string& pre_file = "");
    ~CuotProvider() override;

    CuotProvider(const CuotProvider&) = delete;
    CuotProvider& operator=(const CuotProvider&) = delete;

    OTParty party() const override { return role_; }
    const char* backend_name() const override { return "cuot"; }

    // The underlying network channel. Exposed so the test harness can do an
    // auxiliary verification exchange (sender sends Delta + its x buffer back
    // to the receiver to check the RCOT correlation). NOT used by protocol
    // code; the backend owns the channel.
    emp::NetIO* net_io();

    // --- cuOT RCOT (the usable primitive) ----------------------------------
    // Both parties: generate `n` random correlated 128-bit blocks into `out`
    // (which must point to n*16 bytes; cast to uint64_t* for 2x uint64 per
    // block). Returns kOk on success. This is a NON-arbitrary-modulus,
    // block-level RCOT — NOT the arithmetic COT of OTProvider::send_cot.
    OTStatus rcot_blocks(uint8_t* out, int64_t n);

    // --- OPT-1: GPU-resident RCOT (no D2H) ---------------------------------
    // Fill the caller-owned GPU `Mat` with `n` RCOT blocks. The blocks stay
    // on the GPU (no write_to_cpu) so the GPU MITCCRH/Beaver kernels can
    // consume them by device pointer. The caller slices the Mat via
    // `data() + offset` for leaf vs triple (one big rcot, not 3 small ones).
    // Mat must be pre-sized to n blocks (Mat({(uint64_t)n})). cuda_setdev is
    // called internally (re-pin in case the host thread changed it).
    OTStatus rcot_blocks_gpu(Mat& out, int64_t n);

    // The underlying emp::FerretCOT — exposed so the GPU kernels (cuot_kernels.cu)
    // and the gpu_mm compare primitive can reach Delta / NetIO directly without
    // round-tripping bytes through the host. NOT for protocol code; the backend
    // owns the channel.
    emp::FerretCOT<emp::NetIO>* ferret() { return ferret_.get(); }

    // --- OPT-1/2: GPU-resident block chosen-COT (no D2H, diff-vector on GPU) -
    // Like send/recv_cot_blocks but the RCOT blocks stay on the GPU (caller's
    // Mat) and the diff-vector is computed + bit-packed on the GPU (OPT-2),
    // exchanged as n/8 bytes (vs the CPU path's n bytes). Only n/8 bytes cross
    // host<->net; the blocks never leave the GPU. Mirrors send/recv_cot_blocks
    // (cuot_provider.cc) but GPU-resident.
    //   send_cot_blocks_gpu(out_mat, n): sender. rcot into Mat; recv packed d
    //     (n/8 bytes); apply_diff_vector on GPU (x' = x ^ (d?Delta:0)).
    //   recv_cot_blocks_gpu(out_mat, b_dev, n): receiver. rcot into Mat;
    //     diff_vector_pack8 on GPU (d = lsb(r)^b, packed); send n/8 bytes.
    // b_dev is a DEVICE pointer (uint8_t*, n bytes, 0/1) — the choice bits,
    // already on the GPU (e.g. extracted from digits by the caller).
    OTStatus send_cot_blocks_gpu(Mat& out, int64_t n);
    OTStatus recv_cot_blocks_gpu(Mat& out, const uint8_t* b_dev, int64_t n);

    // --- Block-level chosen COT (paper §III-D "Generating COTs from RCOTs") --
    // Implemented WITHOUT cuOT's IPC online path (which is hardcoded to an
    // 8-GPU topology). Instead we reuse rcot_blocks + an exchange of the diff
    // vector over this provider's own NetIO — exactly the paper's method:
    //   sender: x = rcot(); recv d (n bits) over NetIO; x'[i] = x[i] ^ (d_i?Delta:0)
    //   receiver: r = rcot(); d_i = getLSB(r[i]) XOR b_i; send d over NetIO
    // Correlation: r[i] == x'[i] ^ (b[i] * Delta). Semi-honest: d is a
    // one-time-pad (LSB(r) is the receiver's private uniform bit unknown to
    // the sender), so the sender learns nothing about b; the receiver learns
    // nothing about Delta in the diff step. Both operate in GF(2^128).
    //
    // `out` must hold n*16 bytes. On the sender side `out` receives x' (the
    // adjusted sender blocks); on the receiver side `out` receives r (and the
    // caller's `b` is consumed). `b` is the receiver's choice bits (length n).
    OTStatus send_cot_blocks(uint8_t* out, int64_t n);
    OTStatus recv_cot_blocks(uint8_t* out, const bool* b, int64_t n);

    // --- Block-level random OT (ROT) — for _2ROT bit-triple generation (M3-T6 Phase B) --
    // SCI's _2ROT triple (bit-triple-generator.h:184-199) needs a 1-2 random OT:
    //   sender gets (m0, m1) two INDEPENDENT random blocks; receiver gets m_{r}
    //   for its random choice bit r. This is emp COT<T>::send_rot/recv_rot
    //   (deps/emp-tool/.../cot.h): rcot + MITCCRH decorrelation. cuOT's
    //   FerretCOT inherits send_rot/recv_rot, but they call the no-op
    //   send_cot(block*) (ferret_cot.h:41) -> broken on cuOT. So we re-derive
    //   ROT on top of our send_cot_blocks/recv_cot_blocks (the diff-vector
    //   block chosen-COT): run a chosen-COT with RECEIVER's random choice r,
    //   then MITCCRH-hash (hash<ot_bsize,2> sender / hash<ot_bsize,1> recv) to
    //   decorrelate into independent (m0,m1)/m_{r}. Mirrors cot.h send_rot/
    //   recv_rot + send_ot_rm_rc (silent_ot.h:425-465). The seed sync (sender
    //   picks s, sends to receiver) matches SCI.
    //   out (sender): n*2 blocks, [m0_0,m1_0, m0_1,m1_1, ...] interleaved.
    //   out (receiver): n blocks, m_{r_i} per element. r (receiver): n choice bits.
    OTStatus send_rot_blocks(uint8_t* out, int64_t n);
    OTStatus recv_rot_blocks(uint8_t* out, const bool* r, int64_t n);

    // Sender (ALICE) only: the global Ferret Delta (128-bit), LSB forced to 1.
    // Exposed for the standalone correlation test to verify
    //   recv[i] == send[i] ^ (lsb(recv[i]) * Delta)   (RCOT), and
    //   recv[i] == send[i] ^ (b[i]     * Delta)       (chosen COT).
    // Returns kInvalidArg if called on the receiver or before setup.
    OTStatus delta_block(uint8_t out[16]) const;

    // --- OTProvider arithmetic COT (Z_{2^l} / Z_p) — M2-T3 adapter ---------
    // cuOT is GF(2^128) only; these layer MITCCRH hashing on the block chosen-COT
    // (send_cot_blocks/recv_cot_blocks) to produce uint64_t arithmetic COT,
    // mirroring SCI SilentOT::send_ot_cam_cc / recv_ot_cam_cc / _prime
    // (SCI/src/OT/ferret/silent_ot.h:85-242). The substitution: SCI calls
    // ferret->send_cot(block*,len) which cuOT stubs as a no-op (ferret_cot.h:41),
    // so we use our own send_cot_blocks/recv_cot_blocks for the raw block-RCOT.
    // Correlation: receiver with bit b gets x (b=0) or x+corr (b=1) mod K, where
    // x is the sender's data0; i.e. y_1 - y_0 = corr mod K (per Task M2-T3).
    // CAVEAT: inherits cuOT's ~2% RCOT error floor (M2-T2 shelved); these
    // validate the ADAPTER's MITCCRH/packing logic, not cuOT end-to-end.
    OTStatus send_cot(uint64_t* data0, const uint64_t* corr, int n,
                      int l) override;
    OTStatus recv_cot(uint64_t* data, const bool* b, int n, int l) override;
    OTStatus send_cot_prime(uint64_t* data0, const uint64_t* corr, int n,
                            uint64_t p) override;
    OTStatus recv_cot_prime(uint64_t* data, const bool* b, int n,
                            uint64_t p) override;

   private:
    OTParty role_;
    int gpu_;
    std::unique_ptr<emp::NetIO> io_;
    // CRITICAL: FerretCOT stores `this->ios = ios` (a T**) and MpcotReg/OTPre
    // later read ios[0]. The array MUST outlive FerretCOT — cuOT's own test
    // passes &io where `io` is a main()-local. Declaring the array inside the
    // CuotProvider ctor (as I first did) leaves ferret_->ios dangling after
    // the ctor returns, so extend_initialization reads freed stack -> bad
    // NetIO* -> stream=NULL -> fwrite(NULL) SIGSEGV. Keep it as a member.
    std::array<emp::NetIO*, 1> ios_;
    std::unique_ptr<emp::FerretCOT<emp::NetIO>> ferret_;
    bool ready_ = false;
};

}  // namespace gpu_mm

#endif  // GPU_MM_CUOT_PROVIDER_H
