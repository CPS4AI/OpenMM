// cot_pack.h — vendored bit-packing helpers for arithmetic COT (Task M2-T3).
//
// Copied verbatim from SCI/src/OT/ot-utils.h:154-216 (pack_cot_messages /
// unpack_cot_messages) and placed in namespace gpu_mm to avoid pulling SCI's
// OT/ot.h include chain (which collides with cuOT's emp-ot headers on the
// cuot build target). Pure bit-packing, no crypto deps.
//
// These pack `bsize` l-bit correction values into a packed uint64_t array
// (and unpack the reverse), matching SCI's SilentOT send_ot_cam_cc /
// recv_ot_cam_cc wire format — so a gpu_mm::CuotProvider arithmetic COT is
// byte-compatible with SCI's SilentOT on the wire (were they ever paired).
//
// Original authors: Mayank Rathee, Deevashwer Rathee (MSR Research).
// License: MIT (see below — preserved from the source).
//
// Copyright (c) 2020 Microsoft Research
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the
// Software, and to permit persons to whom the Software is furnished to do so,
// subject to the condition that the above copyright notice and this permission
// notice are included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
#ifndef GPU_MM_COT_PACK_H
#define GPU_MM_COT_PACK_H

#include <cstdint>
#include <cassert>

namespace gpu_mm {

// Pack `bsize` l-bit correction values from `corr_data` into the bit-packed
// uint64_t array `y` (length `ysize`). Wire format matches SCI.
inline void pack_cot_messages(uint64_t *y, uint64_t *corr_data, int ysize,
                              int bsize, int bitsize) {
  assert(y != nullptr && corr_data != nullptr);
  uint64_t start_pos = 0;
  uint64_t end_pos = 0;
  uint64_t start_block = 0;
  uint64_t end_block = 0;
  uint64_t temp_bl = 0;
  uint64_t mask = (1ULL << bitsize) - 1;
  if (bitsize == 64)
    mask = -1;

  uint64_t carriersize = 64;
  for (int i = 0; i < ysize; i++) {
    y[i] = 0;
  }
  for (int i = 0; i < bsize; i++) {
    start_pos = i * bitsize; // inclusive
    end_pos = start_pos + bitsize;
    end_pos -= 1; // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (carriersize == 64) {
      if (start_block == end_block) {
        y[start_block] ^= (corr_data[i] & mask) << (start_pos % carriersize);
      } else {
        temp_bl = (corr_data[i] & mask);
        y[start_block] ^= (temp_bl) << (start_pos % carriersize);
        y[end_block] ^= (temp_bl) >> (carriersize - (start_pos % carriersize));
      }
    }
  }
}

// Unpack `bsize` l-bit correction values from the bit-packed `recvd` array
// into `corr_data`. Inverse of pack_cot_messages.
inline void unpack_cot_messages(uint64_t *corr_data, uint64_t *recvd, int bsize,
                                int bitsize) {
  assert(corr_data != nullptr && recvd != nullptr);
  uint64_t start_pos = 0;
  uint64_t end_pos = 0;
  uint64_t start_block = 0;
  uint64_t end_block = 0;
  uint64_t mask = (1ULL << bitsize) - 1;
  if (bitsize == 64)
    mask = -1;
  uint64_t carriersize = 64;

  for (int i = 0; i < bsize; i++) {
    start_pos = i * bitsize;
    end_pos = start_pos + bitsize - 1; // inclusive
    start_block = start_pos / carriersize;
    end_block = end_pos / carriersize;
    if (carriersize == 64) {
      if (start_block == end_block) {
        corr_data[i] = (recvd[start_block] >> (start_pos % carriersize)) & mask;
      } else {
        corr_data[i] = 0;
        corr_data[i] ^= (recvd[start_block] >> (start_pos % carriersize));
        corr_data[i] ^=
            (recvd[end_block] << (carriersize - (start_pos % carriersize)));
      }
    }
  }
}

}  // namespace gpu_mm

#endif  // GPU_MM_COT_PACK_H
