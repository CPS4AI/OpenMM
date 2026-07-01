// cuot_kernels.cu — GPU-resident MITCCRH + diff-vector kernels for the cuOT
// compare (Task M3-T6 Phase C-Opt, OPT-1/OPT-2).
//
// GOAL: move the ~9.2M per-output AES ops (garbled-table pad construction +
// triple ROT decorrelation) off the CPU (emp::MITCCRH / AES-NI, serial per
// output) onto the GPU, AND eliminate the per-rcot D2H (write_to_cpu). The
// RCOT blocks stay GPU-resident in a `Mat`; these kernels consume them by
// device pointer and write only small results (garbled table bits, 1-bit
// pads) back to host.
//
// WHY NOT reuse cuOT's `aesEncrypt` (deps/cuOT/gpu/aes_op.cu:90)? It assumes
// ONE expanded key for the whole launch (`rk` is a single 44-uint32 array
// shared across all threads). MITCCRH needs a DIFFERENT key per output (each
// output's logN RCOT base blocks are its keys). So we need a per-output-key
// AES. We DO reuse the T0c T-table (re-declared here in __constant__, since
// a separate .cu can't reference aes_op.cu's __constant__) and the exact
// round macros (ROTR/GET_BYTE/FLIP_ENDIANESS) copied from aes_op.cu:78-88.
//
// The AES round body is copied VERBATIM from aes_op.cu:204-256 (the proven
// IMPROVED path) but refactored so each thread uses its OWN expanded key
// (key-scheduled in-thread registers). Per-op this is SLOWER than CPU AES-NI
// (1 instr/round) — the win is parallelism: 800k+ outputs in one sweep vs a
// CPU serial loop. (If the GPU doesn't win on the bench, OPT-1 reverts; see
// plan risk #1.)
//
// MITCCRH semantics mirrored (emp mitccrh.h + SCI cheetah MITCCRH):
//   hash(key, in) = AES(key, in) ^ in   (Davies-Meyer)
//   hash<K,H>: for each of K keys, AES-encrypt the H blocks, XOR into them.
//   hash_exp (sender, N-ary): K=logN keys, expands 2^logN-1 pad blocks.
//   hash_single (receiver): K=logN keys, 1 block each, XOR-reduce.
#include "gpu_define.h"   // blk union (16 bytes)
#include "gpu_matrix.h"   // Mat / GPUdata — blk* data()

#include <cstdint>

namespace gpu_mm {
namespace gpu_kern {

// Re-declare the AES T0 table in THIS TU's __constant__ memory. (aes_op.cu
// owns its own; a separate .cu can't reference another TU's __constant__.)
// Standard AES Te0, copied from aes_op.cu:12-76.
__constant__
uint32_t kT0[256] = {
    0xc66363a5,0xf87c7c84,0xee777799,0xf67b7b8d,0xfff2f20d,0xd66b6bbd,0xde6f6fb1,0x91c5c554,
    0x60303050,0x02010103,0xce6767a9,0x562b2b7d,0xe7fefe19,0xb5d7d762,0x4dababe6,0xec76769a,
    0x8fcaca45,0x1f82829d,0x89c9c940,0xfa7d7d87,0xeffafa15,0xb25959eb,0x8e4747c9,0xfbf0f00b,
    0x41adadec,0xb3d4d467,0x5fa2a2fd,0x45afafea,0x239c9cbf,0x53a4a4f7,0xe4727296,0x9bc0c05b,
    0x75b7b7c2,0xe1fdfd1c,0x3d9393ae,0x4c26266a,0x6c36365a,0x7e3f3f41,0xf5f7f702,0x83cccc4f,
    0x6834345c,0x51a5a5f4,0xd1e5e534,0xf9f1f108,0xe2717193,0xabd8d873,0x62313153,0x2a15153f,
    0x0804040c,0x95c7c752,0x46232365,0x9dc3c35e,0x30181828,0x379696a1,0x0a05050f,0x2f9a9ab5,
    0x0e070709,0x24121236,0x1b80809b,0xdfe2e23d,0xcdebeb26,0x4e272769,0x7fb2b2cd,0xea75759f,
    0x1209091b,0x1d83839e,0x582c2c74,0x341a1a2e,0x361b1b2d,0xdc6e6eb2,0xb45a5aee,0x5ba0a0fb,
    0xa45252f6,0x763b3b4d,0xb7d6d661,0x7db3b3ce,0x5229297b,0xdde3e33e,0x5e2f2f71,0x13848497,
    0xa65353f5,0xb9d1d168,0x00000000,0xc1eded2c,0x40202060,0xe3fcfc1f,0x79b1b1c8,0xb65b5bed,
    0xd46a6abe,0x8dcbcb46,0x67bebed9,0x7239394b,0x944a4ade,0x984c4cd4,0xb05858e8,0x85cfcf4a,
    0xbbd0d06b,0xc5efef2a,0x4faaaae5,0xedfbfb16,0x864343c5,0x9a4d4dd7,0x66333355,0x11858594,
    0x8a4545cf,0xe9f9f910,0x04020206,0xfe7f7f81,0xa05050f0,0x783c3c44,0x259f9fba,0x4ba8a8e3,
    0xa25151f3,0x5da3a3fe,0x804040c0,0x058f8f8a,0x3f9292ad,0x219d9dbc,0x70383848,0xf1f5f504,
    0x63bcbcdf,0x77b6b6c1,0xafdada75,0x42212163,0x20101030,0xe5ffff1a,0xfdf3f30e,0xbfd2d26d,
    0x81cdcd4c,0x180c0c14,0x26131335,0xc3ecec2f,0xbe5f5fe1,0x359797a2,0x884444cc,0x2e171739,
    0x93c4c457,0x55a7a7f2,0xfc7e7e82,0x7a3d3d47,0xc86464ac,0xba5d5de7,0x3219192b,0xe6737395,
    0xc06060a0,0x19818198,0x9e4f4fd1,0xa3dcdc7f,0x44222266,0x542a2a7e,0x3b9090ab,0x0b888883,
    0x8c4646ca,0xc7eeee29,0x6bb8b8d3,0x2814143c,0xa7dede79,0xbc5e5ee2,0x160b0b1d,0xaddbdb76,
    0xdbe0e03b,0x64323256,0x743a3a4e,0x140a0a1e,0x924949db,0x0c06060a,0x4824246c,0xb85c5ce4,
    0x9fc2c25d,0xbdd3d36e,0x43acacef,0xc46262a6,0x399191a8,0x319595a4,0xd3e4e437,0xf279798b,
    0xd5e7e732,0x8bc8c843,0x6e373759,0xda6d6db7,0x018d8d8c,0xb1d5d564,0x9c4e4ed2,0x49a9a9e0,
    0xd86c6cb4,0xac5656fa,0xf3f4f407,0xcfeaea25,0xca6565af,0xf47a7a8e,0x47aeaee9,0x10080818,
    0x6fbabad5,0xf0787888,0x4a25256f,0x5c2e2e72,0x381c1c24,0x57a6a6f1,0x73b4b4c7,0x97c6c651,
    0xcbe8e823,0xa1dddd7c,0xe874749c,0x3e1f1f21,0x964b4bdd,0x61bdbddc,0x0d8b8b86,0x0f8a8a85,
    0xe0707090,0x7c3e3e42,0x71b5b5c4,0xcc6666aa,0x904848d8,0x06030305,0xf7f6f601,0x1c0e0e12,
    0xc26161a3,0x6a35355f,0xae5757f9,0x69b9b9d0,0x17868691,0x99c1c158,0x3a1d1d27,0x279e9eb9,
    0xd9e1e138,0xebf8f813,0x2b9898b3,0x22111133,0xd26969bb,0xa9d9d970,0x078e8e89,0x339494a7,
    0x2d9b9bb6,0x3c1e1e22,0x15878792,0xc9e9e920,0x87cece49,0xaa5555ff,0x50282878,0xa5dfdf7a,
    0x038c8c8f,0x59a1a1f8,0x09898980,0x1a0d0d17,0x65bfbfda,0xd7e6e631,0x844242c6,0xd06868b8,
    0x824141c3,0x299999b0,0x5a2d2d77,0x1e0f0f11,0x7bb0b0cb,0xa85454fc,0x6dbbbbd6,0x2c16163a
};

// AES S-box (for the key schedule SubWord step). Standard.
__constant__
uint8_t kSbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__constant__ uint8_t kRcon[11] = {0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};

// Byte-perm macros copied from aes_op.cu:78-88 (proven correct there).
#define CK_ROT(x, sel) __byte_perm(x, x, sel)
#define CK_GETB(x, sel) __byte_perm(x, 0, sel)
#define CK_FLIP(x) __byte_perm(x, x, 0x0123)
#define CK_B3 0x00004321
#define CK_B2 0x00005432
#define CK_B1 0x00006543
#define CK_G3 0x00004560
#define CK_G2 0x00004561
#define CK_G1 0x00004562
#define CK_G0 0x00004563

// ---- per-thread AES-128 primitives (BYTE-ORIENTED, tiny-AES style) ----
//
// Rewritten for CORRECTNESS after the T-table version produced mostly-zero
// output (round/byte-order indexing bug). This byte-oriented form mirrors
// the standard FIPS-197 AES-128 (tiny-AES) byte layout, which matches emp's
// _mm_aesenc_si128 byte semantics exactly (block = 16 bytes in natural
// order; SubBytes per-byte; ShiftRows shifts row r by r; MixColumns = GF
// multiply). Slower than a T-table but correct — and the bench showed 33x
// GPU headroom even before fixing, so a correct-but-slower kernel still wins.
//
// emp note: emp's _mm_aesenc_si128(state, rk) = MixColumns(ShiftRows(
// SubBytes(state))) ^ rk, with state bytes in __m128i natural order. The
// key schedule (emp AES_opt_key_schedule) is the "fast garbling" vectorized
// form but produces the SAME 11 round keys as FIPS Rijndael expansion. So a
// standard FIPS AES-128 here is byte-compatible with emp.

// AES round-key expansion: 16-byte key -> 176 bytes (11 round keys x 16).
// Operates on byte arrays. `key` = 16 bytes (the block, natural order).
__device__ __forceinline__
void aes_key_schedule(const uint8_t key[16], uint8_t rk[176]) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) rk[i] = key[i];
    uint8_t rc = 1;
    for (int i = 16; i < 176; i += 16) {
        uint8_t t[4];
        // RotWord + SubWord on the last word.
        t[0] = kSbox[rk[i - 3]] ^ rc;
        t[1] = kSbox[rk[i - 2]];
        t[2] = kSbox[rk[i - 1]];
        t[3] = kSbox[rk[i - 4]];
        rk[i + 0] = rk[i - 16] ^ t[0];
        rk[i + 1] = rk[i - 15] ^ t[1];
        rk[i + 2] = rk[i - 14] ^ t[2];
        rk[i + 3] = rk[i - 13] ^ t[3];
        #pragma unroll
        for (int j = 4; j < 16; ++j)
            rk[i + j] = rk[i - 16 + j] ^ rk[i + j - 4];
        // Rcon update (xtime in GF(2)).
        rc = (rc << 1) ^ (((rc >> 7) & 1) * 0x1b);
    }
}

// GF(2^8) multiply-by-2 (xtime) and the MixColumns helper.
__device__ __forceinline__
uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}
__device__ __forceinline__
uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        p ^= (uint8_t)((b & 1) * a);
        uint8_t hi = a & 0x80;
        a = (uint8_t)(a << 1) ^ ((hi ? 0x1b : 0));
        b >>= 1;
    }
    return p;
}

// SubBytes + ShiftRows + MixColumns + AddRoundKey, in place on s[16].
// FIPS state layout: s[r + 4*c] (row r, col c). ShiftRows: row r shifted
// left by r. MixColumns: standard column mix.
__device__ __forceinline__
void aes_round(uint8_t s[16], const uint8_t rk[16]) {
    uint8_t t[16];
    // SubBytes + ShiftRows.
    t[0]  = kSbox[s[0]];  t[1] = kSbox[s[5]];  t[2] = kSbox[s[10]]; t[3] = kSbox[s[15]];
    t[4]  = kSbox[s[4]];  t[5] = kSbox[s[9]];  t[6] = kSbox[s[14]]; t[7] = kSbox[s[3]];
    t[8]  = kSbox[s[8]];  t[9] = kSbox[s[13]]; t[10]= kSbox[s[2]];  t[11]= kSbox[s[7]];
    t[12] = kSbox[s[12]]; t[13]= kSbox[s[1]];  t[14]= kSbox[s[6]];  t[15]= kSbox[s[11]];
    // MixColumns.
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = t[4*c+0], a1 = t[4*c+1], a2 = t[4*c+2], a3 = t[4*c+3];
        s[4*c+0] = (uint8_t)(gmul(a0,2) ^ gmul(a1,3) ^ a2 ^ a3);
        s[4*c+1] = (uint8_t)(a0 ^ gmul(a1,2) ^ gmul(a2,3) ^ a3);
        s[4*c+2] = (uint8_t)(a0 ^ a1 ^ gmul(a2,2) ^ gmul(a3,3));
        s[4*c+3] = (uint8_t)(gmul(a0,3) ^ a1 ^ a2 ^ gmul(a3,2));
    }
    // AddRoundKey.
    #pragma unroll
    for (int i = 0; i < 16; ++i) s[i] ^= rk[i];
}

// Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns).
__device__ __forceinline__
void aes_final_round(uint8_t s[16], const uint8_t rk[16]) {
    uint8_t t[16];
    t[0]  = kSbox[s[0]];  t[1] = kSbox[s[5]];  t[2] = kSbox[s[10]]; t[3] = kSbox[s[15]];
    t[4]  = kSbox[s[4]];  t[5] = kSbox[s[9]];  t[6] = kSbox[s[14]]; t[7] = kSbox[s[3]];
    t[8]  = kSbox[s[8]];  t[9] = kSbox[s[13]]; t[10]= kSbox[s[2]];  t[11]= kSbox[s[7]];
    t[12] = kSbox[s[12]]; t[13]= kSbox[s[1]];  t[14]= kSbox[s[6]];  t[15]= kSbox[s[11]];
    #pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = t[i] ^ rk[i];
}

// Encrypt one 16-byte block (natural byte order) under expanded key rk[176].
__device__ __forceinline__
void aes_encrypt(const uint8_t in[16], const uint8_t rk[176], uint8_t out[16]) {
    uint8_t s[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = in[i] ^ rk[i];  // AddRoundKey round 0
    #pragma unroll
    for (int r = 1; r < 10; ++r) aes_round(s, rk + 16 * r);
    aes_final_round(s, rk + 160);
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = s[i];
}

// MITCCRH Davies-Meyer hash: out = AES(key, in) ^ in. Byte arrays (16 bytes).
__device__ __forceinline__
void mitccrh_hash(const uint8_t key[16], const uint8_t in[16], uint8_t out[16]) {
    uint8_t rk[176];
    aes_key_schedule(key, rk);
    uint8_t e[16];
    aes_encrypt(in, rk, e);
    #pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = e[i] ^ in[i];
}

// Byte-array ↔ uint32[4] bridges (the kernels above read blk.data_8 / .data_32).
// blk.data_8[0..15] is natural byte order; data_32[0..3] is LE words. The
// byte-oriented AES uses data_8 directly. (The old T-table uint32 AES below
// was removed — it had a round/byte-order bug; the byte-oriented form above
// is the correct, emp-compatible AES-128.)

// ============================================================================
// OPT-1b: leaf garbled-table pad construction, SENDER side (mirrors
// KkotLeafCmp::send_leaf's NaryMitccrh hash_exp loop, kkot_leaf_cmp.h:371-392,
// which mirrors SCI silent_ot.h:516-547).
//
// Each thread = 1 leaf output. It has logN base RCOT blocks (rm0[i*logN..])
// for the 0-branch keys and rm1[i*logN..] = rm0^Delta for the 1-branch.
// hash_in0/hash_in1 are the (2^logN-1) fixed plaintext blocks (same for all
// outputs, passed as device arrays). The kernel computes, for each output,
// the N=2^logN pad blocks pad[i*N + k] by XOR-ing the appropriate 0/1-branch
// hash outputs (silent_ot.h:536-547). Writes pad (n*N blocks) to device.
//
// Only logN in {1,2,3,4} (N in {2,4,8,16}) supported (Millionaire beta=4).
// ============================================================================
template <int logN>
__global__
void leaf_pad_sender(const blk* __restrict__ rm0, const blk* __restrict__ rm1,
                     const blk* __restrict__ hash_in0,
                     const blk* __restrict__ hash_in1,
                     blk* __restrict__ pad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int N = 1 << logN;
    const int nhash = (1 << logN) - 1;  // = N-1
    // ho0/ho1: the (2^logN-1) per-branch hash outputs, byte arrays (16 bytes
    // each). nhash <= 15 for logN=4. We XOR into them across the logN keys
    // (each key contributes to its 2^s subtree blocks).
    uint8_t ho0[15][16], ho1[15][16];
    #pragma unroll
    for (int j = 0; j < 15; ++j)
        #pragma unroll
        for (int t = 0; t < 16; ++t) { ho0[j][t] = 0; ho1[j][t] = 0; }
    // For each key s, it encrypts the 2^s blocks at tree level s
    // (start=2^s-1, cnt=2^s). XOR the Davies-Meyer hash into ho0/ho1.
    for (int s = 0; s < logN; ++s) {
        const uint8_t* k0 = (rm0 + i * logN + s)->data_8;
        const uint8_t* k1 = (rm1 + i * logN + s)->data_8;
        int start = (1 << s) - 1;
        int cnt = 1 << s;
        for (int b = 0; b < cnt; ++b) {
            uint8_t h[16];
            mitccrh_hash(k0, (hash_in0 + start + b)->data_8, h);
            #pragma unroll
            for (int t = 0; t < 16; ++t) ho0[start + b][t] ^= h[t];
            mitccrh_hash(k1, (hash_in1 + start + b)->data_8, h);
            #pragma unroll
            for (int t = 0; t < 16; ++t) ho1[start + b][t] ^= h[t];
        }
    }
    // Assemble pad[k] for k in [0,N): XOR, per bit s of k, the hash_out at
    // idx (kkot_leaf_cmp.h:348-360). bit0 -> ho0[idx+pref], bit1 -> ho1[idx+pref].
    for (int k = 0; k < N; ++k) {
        uint8_t p[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) p[t] = 0;
        int idx = 0;
        for (int s = 0; s < logN; ++s) {
            int pref = k & ((1 << s) - 1);
            const uint8_t* src = ((k & (1 << s)) == 0) ? ho0[idx + pref]
                                                       : ho1[idx + pref];
            #pragma unroll
            for (int t = 0; t < 16; ++t) p[t] ^= src[t];
            idx += 1 << s;
        }
        blk* outb = pad + i * N + k;
        #pragma unroll
        for (int t = 0; t < 16; ++t) outb->data_8[t] = p[t];
    }
}

// ============================================================================
// OPT-1b: leaf pad, RECEIVER side (mirrors KkotLeafCmp::recv_leaf's
// hash_single loop, kkot_leaf_cmp.h:452-460, = SCI silent_ot.h:600-610).
// Each thread = 1 output. key s = rm[i*logN+s] (the chosen block). hash_in[s]
// = makeBlock(choice & ((1<<(s+1))-1), 0). pad_choice = XOR_s hash_out[s].
// ============================================================================
__global__
void leaf_pad_recv(const blk* __restrict__ rm, blk* __restrict__ pad_choice,
                   const uint8_t* __restrict__ choice, int n, int logN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t p[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) p[t] = 0;
    for (int s = 0; s < logN; ++s) {
        const uint8_t* k = (rm + i * logN + s)->data_8;
        // hash_in = makeBlock(pref, 0): emp block has high 64 = 0, low 64 = pref.
        // In byte order (LE), byte 0 = pref's low byte. pref < 2^logN <= 16, so
        // pref fits in byte 0; bytes 1..15 = 0.
        uint8_t inb[16];
        #pragma unroll
        for (int t = 0; t < 16; ++t) inb[t] = 0;
        inb[0] = (uint8_t)(choice[i] & ((1 << (s + 1)) - 1));
        uint8_t h[16];
        mitccrh_hash(k, inb, h);
        #pragma unroll
        for (int t = 0; t < 16; ++t) p[t] ^= h[t];
    }
    blk* outb = pad_choice + i;
    #pragma unroll
    for (int t = 0; t < 16; ++t) outb->data_8[t] = p[t];
}

// ============================================================================
// OPT-2: diff-vector on GPU. Receiver computes d_i = lsb(r_i) ^ b_i for n
// blocks, bit-packs into d_packed (n/8 bytes). lsb = byte 0 bit 0 of the blk.
// ============================================================================
// OPT-2 placeholder: see diff_vector_pack8 / apply_diff_vector below.
// (The 1-thread-per-output pack above is a leftover stub; use pack8.)
__global__
void diff_vector_pack(const blk* __restrict__ /*r*/, const uint8_t* __restrict__ /*b*/,
                      uint8_t* __restrict__ /*d_packed*/, int /*n*/) {
    // superseded by diff_vector_pack8 (one thread per 8 outputs, bit-packs).
}

// One thread per 8 outputs, writes one packed byte.
__global__
void diff_vector_pack8(const blk* __restrict__ r, const uint8_t* __restrict__ b,
                       uint8_t* __restrict__ d_packed, int nbytes) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nbytes) return;
    uint8_t out = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int i = j * 8 + k;
        uint8_t lsb = r[i].data_8[0] & 1;
        uint8_t di = (uint8_t)(lsb ^ (b[i] ? 1 : 0));
        out |= (uint8_t)(di << k);
    }
    d_packed[j] = out;
}

// Sender: unpack d_packed (n/8 bytes) and XOR Delta into r[i] where d_i=1.
// r is in-place GPU. (Mirrors cuot_provider.cc:154-155.)
__global__
void apply_diff_vector(blk* __restrict__ r, const uint8_t* __restrict__ d_packed,
                       const blk* __restrict__ delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t di = (d_packed[i / 8] >> (i % 8)) & 1;
    if (di) {
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            r[i].data_32[t] ^= delta->data_32[t];
    }
}
}  // namespace gpu_kern (kernels)

// Host launchers for OPT-2 diff-vector kernels.
namespace gpu_kern {
void launch_diff_vector_pack8(const blk* r, const uint8_t* b,
                                     uint8_t* d_packed, int nbytes) {
    int blkdim = 256;
    int grid = (nbytes + blkdim - 1) / blkdim;
    diff_vector_pack8<<<grid, blkdim>>>(r, b, d_packed, nbytes);
}
void launch_apply_diff_vector(blk* r, const uint8_t* d_packed,
                                     const blk* delta, int n) {
    int blkdim = 256;
    int grid = (n + blkdim - 1) / blkdim;
    apply_diff_vector<<<grid, blkdim>>>(r, d_packed, delta, n);
}
}  // namespace gpu_kern


// ============================================================================
// Host launchers (called from .cpp/.cc). Take raw `blk*` (cuOT's 16-byte GPU
// block type; emp::block/__m128i is also 16 bytes — callers reinterpret_cast).
// ============================================================================
namespace gpu_kern {

template <int logN>
void launch_leaf_pad_sender(const blk* rm0, const blk* rm1,
                            const blk* hash_in0, const blk* hash_in1,
                            blk* pad, int n) {
    int blkdim = 256;
    int grid = (n + blkdim - 1) / blkdim;
    leaf_pad_sender<logN><<<grid, blkdim>>>(rm0, rm1, hash_in0, hash_in1, pad, n);
}

// NOTE: must NOT be `inline` — this is a free function defined in a .cu TU that
// does NOT itself call it, so an `inline` def would not be emitted here (no
// COMDAT use-site in this TU), leaving the symbol undefined for callers in
// other TUs. The non-inline def makes this TU the single emission site.
void launch_leaf_pad_recv(const blk* rm, blk* pad_choice,
                          const uint8_t* choice, int n, int logN) {
    int blkdim = 256;
    int grid = (n + blkdim - 1) / blkdim;
    leaf_pad_recv<<<grid, blkdim>>>(rm, pad_choice, choice, n, logN);
}

// Debug: raw single AES-128 Davies-Meyer hash of ONE block. out=AES(key,in)^in.
__global__
inline void dbg_single_aes(const blk* key, const blk* inb, blk* out) {
    mitccrh_hash(key->data_8, inb->data_8, out->data_8);
}
void launch_dbg_single_aes(const blk* key, const blk* inb, blk* out) {
    dbg_single_aes<<<1, 1>>>(key, inb, out);
}

// Explicit instantiations (template defs above aren't emitted otherwise).
// Inside gpu_kern so nvcc accepts them (qualified form triggered errors).
template void launch_leaf_pad_sender<1>(const blk*, const blk*, const blk*, const blk*, blk*, int);
template void launch_leaf_pad_sender<2>(const blk*, const blk*, const blk*, const blk*, blk*, int);
template void launch_leaf_pad_sender<3>(const blk*, const blk*, const blk*, const blk*, blk*, int);
template void launch_leaf_pad_sender<4>(const blk*, const blk*, const blk*, const blk*, blk*, int);

}  // namespace gpu_kern
}  // namespace gpu_mm
