/*
Authors: Mayank Rathee, Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "BuildingBlocks/truncation.h"

#include "BuildingBlocks/value-extension.h"
#include "utils/performance.h"

// using namespace std;
using namespace sci;

Truncation::Truncation(int party, sci::NetIO *io, OTPack<sci::NetIO> *otpack,
                       AuxProtocols *auxp,
                       MillionaireWithEquality<sci::NetIO> *mill_eq_in) {
  this->party = party;
  this->io = io;
  this->otpack = otpack;
  if (auxp == nullptr) {
    del_aux = true;
    this->aux = new AuxProtocols(party, io, otpack);
  } else {
    this->aux = auxp;
  }
  if (mill_eq_in == nullptr) {
    del_milleq = true;
    this->mill_eq = new MillionaireWithEquality<sci::NetIO>(party, io, otpack,
                                                            this->aux->mill);
  } else {
    this->mill_eq = mill_eq_in;
  }
  this->mill = this->aux->mill;
  this->eq = new Equality<sci::NetIO>(party, io, otpack, this->mill);
  this->triple_gen = this->mill->triple_gen;
}

Truncation::~Truncation() {
  if (del_aux) {
    delete this->aux;
  }
  if (del_milleq) {
    delete this->mill_eq;
  }
  delete this->eq;
}

void Truncation::div_pow2(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t shift, int32_t bw, bool signed_arithmetic,
                          uint8_t *msb_x) {
  if (signed_arithmetic == false) {
    truncate(dim, inA, outB, shift, bw, false, msb_x);
    return;
  }
  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Division shouldn't truncate the full bitwidth");
  assert(signed_arithmetic && (bw - shift - 1 >= 0));
  assert((msb_x == nullptr) && "Not yet implemented");
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  // mask_upper extracts the upper bw-shift-1 bits after the MSB
  uint64_t mask_upper =
      ((bw - shift - 1) == 64 ? -1 : ((1ULL << (bw - shift - 1)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (party == sci::ALICE) {
    for (int i = 0; i < dim; i++) {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint64_t *inA_lower = new uint64_t[dim];
  uint64_t *inA_upper = new uint64_t[dim];
  uint8_t *wrap_lower = new uint8_t[dim];
  uint8_t *wrap_upper = new uint8_t[dim];
  uint8_t *msb_upper = new uint8_t[dim];
  uint8_t *zero_test_lower = new uint8_t[dim];
  uint8_t *eq_upper = new uint8_t[dim];
  uint8_t *and_upper = new uint8_t[dim];
  uint8_t *div_correction = new uint8_t[dim];
  for (int i = 0; i < dim; i++) {
    inA_lower[i] = inA[i] & mask_shift;
    inA_upper[i] = (inA[i] >> shift) & mask_upper;
    if (party == sci::BOB) {
      inA_upper[i] = (mask_upper - inA_upper[i]) & mask_upper;
    }
  }

  this->aux->wrap_computation(inA_lower, wrap_lower, dim, shift);
  for (int i = 0; i < dim; i++) {
    if (party == sci::BOB) {
      inA_lower[i] = (-1 * inA_lower[i]) & mask_shift;
    }
  }
  this->eq->check_equality(zero_test_lower, inA_lower, dim, shift);
  this->mill_eq->compare_with_eq(msb_upper, eq_upper, inA_upper, dim,
                                 (bw - shift - 1));
  this->aux->AND(wrap_lower, eq_upper, and_upper, dim);
  for (int i = 0; i < dim; i++) {
    msb_upper[i] = (msb_upper[i] ^ and_upper[i] ^ (inA[i] >> (bw - 1))) & 1;
  }
  this->aux->MSB_to_Wrap(inA, msb_upper, wrap_upper, dim, bw);
  // negate zero_test_lower and msb_upper
  // if signed_arithmetic == true, MSB of inA is flipped in the beginning
  // if MSB was 1, and the lower shift bits were not all 0, add 1 as
  // div_correction
  for (int i = 0; i < dim; i++) {
    if (party == sci::ALICE) {
      zero_test_lower[i] ^= 1;
      msb_upper[i] ^= 1;
    }
  }
  this->aux->AND(zero_test_lower, msb_upper, div_correction, dim);

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  uint64_t *arith_wrap_lower = new uint64_t[dim];
  uint64_t *arith_div_correction = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  this->aux->B2A(wrap_lower, arith_wrap_lower, dim, bw);
  this->aux->B2A(div_correction, arith_div_correction, dim, bw);

  for (int i = 0; i < dim; i++) {
    outB[i] =
        ((inA[i] >> shift) + arith_div_correction[i] + arith_wrap_lower[i] -
         (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
        mask_bw;
  }

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] inA_lower;
  delete[] inA_upper;
  delete[] wrap_lower;
  delete[] wrap_upper;
  delete[] msb_upper;
  delete[] zero_test_lower;
  delete[] eq_upper;
  delete[] and_upper;
  delete[] div_correction;
  delete[] arith_wrap_lower;
  delete[] arith_wrap_upper;
  delete[] arith_div_correction;

  return;
}

#if !USE_CHEETAH
void Truncation::truncate(int32_t dim, uint64_t *inA, uint64_t *outB,
                          int32_t shift, int32_t bw, bool signed_arithmetic,
                          uint8_t *msb_x, bool _dummy) {
  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");
  assert((signed_arithmetic && (bw - shift - 1 >= 0)) || !signed_arithmetic);
  assert(inA != outB);

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_upper =
      ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_orig = new uint64_t[dim];

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      inA_orig[i] = inA[i];
      inA[i] = ((inA[i] + (1ULL << (bw - 1))) & mask_bw);
    }
  }

  uint64_t *inA_lower = new uint64_t[dim];
  uint64_t *inA_upper = new uint64_t[dim];
  uint8_t *wrap_lower = new uint8_t[dim];
  uint8_t *wrap_upper = new uint8_t[dim];
  uint8_t *eq_upper = new uint8_t[dim];
  uint8_t *and_upper = new uint8_t[dim];
  for (int i = 0; i < dim; i++) {
    inA_lower[i] = inA[i] & mask_shift;
    inA_upper[i] = (inA[i] >> shift) & mask_upper;
    if (party == sci::BOB) {
      inA_upper[i] = (mask_upper - inA_upper[i]) & mask_upper;
    }
  }

  this->aux->wrap_computation(inA_lower, wrap_lower, dim, shift);
  if (msb_x == nullptr) {
    this->mill_eq->compare_with_eq(wrap_upper, eq_upper, inA_upper, dim,
                                   (bw - shift));
    this->aux->AND(wrap_lower, eq_upper, and_upper, dim);
    for (int i = 0; i < dim; i++) {
      wrap_upper[i] ^= and_upper[i];
    }
  } else {
    if (signed_arithmetic) {
      uint8_t *inv_msb_x = new uint8_t[dim];
      for (int i = 0; i < dim; i++) {
        inv_msb_x[i] = msb_x[i] ^ (party == sci::ALICE ? 1 : 0);
      }
      this->aux->MSB_to_Wrap(inA, inv_msb_x, wrap_upper, dim, bw);
      delete[] inv_msb_x;
    } else {
      this->aux->MSB_to_Wrap(inA, msb_x, wrap_upper, dim, bw);
    }
  }

  uint64_t *arith_wrap_upper = new uint64_t[dim];
  uint64_t *arith_wrap_lower = new uint64_t[dim];
  this->aux->B2A(wrap_upper, arith_wrap_upper, dim, shift);
  this->aux->B2A(wrap_lower, arith_wrap_lower, dim, bw);

  for (int i = 0; i < dim; i++) {
    outB[i] = (((inA[i] >> shift) & mask_upper) + arith_wrap_lower[i] -
               (1ULL << (bw - shift)) * arith_wrap_upper[i]) &
              mask_bw;
  }

  if (signed_arithmetic && (party == sci::ALICE)) {
    for (int i = 0; i < dim; i++) {
      outB[i] = ((outB[i] - (1ULL << (bw - shift - 1))) & mask_bw);
      inA[i] = inA_orig[i];
    }
  }
  delete[] inA_orig;
  delete[] inA_lower;
  delete[] inA_upper;
  delete[] wrap_lower;
  delete[] wrap_upper;
  delete[] eq_upper;
  delete[] and_upper;
  delete[] arith_wrap_lower;
  delete[] arith_wrap_upper;

  return;
}
#else
#include "truncation-cheetah.cpp"
#endif

void Truncation::truncate_red_then_ext(int32_t dim, uint64_t *inA,
                                       uint64_t *outB, int32_t shift,
                                       int32_t bw, bool signed_arithmetic,
                                       uint8_t *msb_x) {
  if (shift == 0) {
    memcpy(outB, inA, dim * sizeof(uint64_t));
    return;
  }
  uint64_t *tmpB = new uint64_t[dim];
  truncate_and_reduce(dim, inA, tmpB, shift, bw);
  XTProtocol xt(this->party, this->io, this->otpack, this->aux);
  if (signed_arithmetic)
    xt.s_extend(dim, tmpB, outB, bw - shift, bw);
  else
    xt.z_extend(dim, tmpB, outB, bw - shift, bw);

  delete[] tmpB;
  return;
}

void Truncation::truncate_and_reduce(int32_t dim, uint64_t *inA, uint64_t *outB,
                                     int32_t shift, int32_t bw) {
  if (shift == 0) {
    memcpy(outB, inA, sizeof(uint64_t) * dim);
    return;
  }
  assert((bw - shift) > 0 && "Truncation shouldn't truncate the full bitwidth");

  uint64_t mask_bw = (bw == 64 ? -1 : ((1ULL << bw) - 1));
  uint64_t mask_shift = (shift == 64 ? -1 : ((1ULL << shift) - 1));
  uint64_t mask_out = ((bw - shift) == 64 ? -1 : ((1ULL << (bw - shift)) - 1));

  uint64_t *inA_lower = new uint64_t[dim];
  uint8_t *wrap = new uint8_t[dim];
  for (int i = 0; i < dim; i++) {
    inA_lower[i] = inA[i] & mask_shift;
  }

  this->aux->wrap_computation(inA_lower, wrap, dim, shift);

  uint64_t *arith_wrap = new uint64_t[dim];
  this->aux->B2A(wrap, arith_wrap, dim, (bw - shift));

  for (int i = 0; i < dim; i++) {
    outB[i] = ((inA[i] >> shift) + arith_wrap[i]) & mask_out;
  }

  return;
}

void Truncation::ring_to_ring(int32_t dim, uint64_t *inA, uint64_t *outB, int32_t bw_A, int32_t bw_B) {
    const uint64_t A_mod_mask = (bw_A == 64 ? -1ULL : (1ULL << bw_A) - 1);
    const uint64_t A_msb_mask = 1ULL << (bw_A - 1);
    const uint64_t big_number = 1ULL << (bw_A - 2); // big number = 2^l / 4

    const uint64_t B_mod_mask = (bw_B == 64 ? -1ULL : (1ULL << bw_B) - 1);

    if (bw_A >= bw_B) {
        std::transform(inA, inA + dim, outB, [B_mod_mask] (uint64_t p) { return p & B_mod_mask; });
        return;
    }

    const int32_t shift_bw = bw_B - bw_A;

    uint64_t *tmpA = new uint64_t[dim]; // add a big number

    // add a big number
    if (party == sci::ALICE) {
        for (uint32_t i = 0; i < dim; i++) {
            tmpA[i] = (inA[i] + big_number) & A_mod_mask;
        }
    } else {
        memcpy(tmpA, inA, dim * sizeof(uint64_t));
    }

    auto OR_then_B2A = [&] (int32_t dim, uint8_t * input_x, uint64_t * output_y, int32_t bw_y) {
        const uint64_t cur_mod_mask = bw_y == 64 ? -1ULL : (1ULL << bw_y) - 1;
        if (party == sci::ALICE) {
            uint64_t *corr_data = new uint64_t[dim];
            for (size_t i = 0; i < dim; i++) {
                corr_data[i] = input_x[i] ^ 1;
            }
            this->otpack->iknp_straight->send_cot(output_y, corr_data, dim, bw_y);
            this->otpack->io->flush();
            for (size_t i = 0; i < dim; i++) {
                output_y[i] = ((uint64_t) input_x[i] - output_y[i]) & cur_mod_mask;
            }
            delete[] corr_data;
        } else {
            this->otpack->iknp_straight->recv_cot(output_y, (bool *)input_x, dim, bw_y);
        }
    };

    // extend bit-width
    uint8_t * msb = new uint8_t[dim];
    for (size_t i = 0; i < dim; i++) {
        msb[i] = (bool) (tmpA[i] & A_msb_mask);
    }
    uint64_t * msb_or_res = new uint64_t[dim]; // wrap = (msb0 || msb1)
    OR_then_B2A(dim, msb, msb_or_res, shift_bw);
    if (party == sci::ALICE) {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (tmpA[i] - (msb_or_res[i] << bw_A) - big_number) & B_mod_mask;
        }
    } else {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (tmpA[i] - (msb_or_res[i] << bw_A)) & B_mod_mask;
        }
    }

    delete[] msb;
    delete[] msb_or_res;
}

void Truncation::ring_to_ring128(int32_t dim, uint64_t *inA, __uint128_t *outB, int32_t bw_A, int32_t bw_B) {
    const uint64_t A_mod_mask = (bw_A == 64 ? -1ULL : (1ULL << bw_A) - 1);
    const uint64_t A_msb_mask = 1ULL << (bw_A - 1);
    const uint64_t big_number = 1ULL << (bw_A - 2); // big number = 2^l / 4
    const __uint128_t big_number_128 = static_cast<__uint128_t>(big_number);

    const __uint128_t B_mod_mask = (bw_B == 128 ? static_cast<__uint128_t>(-1) : (static_cast<__uint128_t>(1) << bw_B) - 1);

    const int32_t shift_bw = bw_B - bw_A;
    assert(shift_bw <= 64);

    uint64_t *tmpA = new uint64_t[dim]; // add a big number

    // add a big number
    if (party == sci::ALICE) {
        for (uint32_t i = 0; i < dim; i++) {
            tmpA[i] = (inA[i] + big_number) & A_mod_mask;
        }
    } else {
        memcpy(tmpA, inA, dim * sizeof(uint64_t));
    }

    auto OR_then_B2A = [&] (int32_t dim, uint8_t * input_x, uint64_t * output_y, int32_t bw_y) {
        const uint64_t cur_mod_mask = bw_y == 64 ? -1ULL : (1ULL << bw_y) - 1;
        if (party == sci::ALICE) {
            uint64_t *corr_data = new uint64_t[dim];
            for (size_t i = 0; i < dim; i++) {
                corr_data[i] = input_x[i] ^ 1;
            }
            this->otpack->iknp_straight->send_cot(output_y, corr_data, dim, bw_y);
            this->otpack->io->flush();
            for (size_t i = 0; i < dim; i++) {
                output_y[i] = ((uint64_t) input_x[i] - output_y[i]) & cur_mod_mask;
            }
            delete[] corr_data;
        } else {
            this->otpack->iknp_straight->recv_cot(output_y, (bool *)input_x, dim, bw_y);
        }
    };

    // extend bit-width
    uint8_t * msb = new uint8_t[dim];
    for (size_t i = 0; i < dim; i++) {
        msb[i] = (bool) (tmpA[i] & A_msb_mask);
    }
    uint64_t * msb_or_res = new uint64_t[dim]; // wrap = (msb0 || msb1)
    OR_then_B2A(dim, msb, msb_or_res, shift_bw);
    if (party == sci::ALICE) {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (static_cast<__uint128_t>(tmpA[i]) - (static_cast<__uint128_t>(msb_or_res[i]) << bw_A) - big_number_128) & B_mod_mask;
        }
    } else {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (static_cast<__uint128_t>(tmpA[i]) - (static_cast<__uint128_t>(msb_or_res[i]) << bw_A)) & B_mod_mask;
        }
    }

    delete[] msb;
    delete[] msb_or_res;
}


// bit extension then local convert
//void Truncation::ring_to_field(int32_t dim, uint64_t *inA, uint64_t *outB, int32_t ring_bw, uint64_t field_mod) {
//    const uint64_t ring_mod_mask = (ring_bw == 64 ? -1ULL : (1ULL << ring_bw) - 1);
//    const uint64_t ring_msb_mask = 1ULL << (ring_bw - 1);
//    const uint64_t big_number = 1ULL << (ring_bw - 2);
//    const __uint128_t big_number_128 = static_cast<__uint128_t>(big_number);
//    const int32_t shift_bw = 40;
//    const int32_t tmp_bw = ring_bw + shift_bw;
//    const __uint128_t tmp_mod_mask = (static_cast<__uint128_t>(1) << tmp_bw) - 1;
//    uint64_t *tmpA = new uint64_t[dim]; // add a big number
//    __uint128_t *tmpB = new __uint128_t[dim]; // bit-width extend
//
//    // add a big number
//    if (party == sci::ALICE && shift_bw != 0) {
//        for (uint32_t i = 0; i < dim; i++) {
//            tmpA[i] = (inA[i] + big_number) & ring_mod_mask;
//        }
//    } else {
//        memcpy(tmpA, inA, dim * sizeof(uint64_t));
//    }
//
//    auto OR_then_B2A = [&] (int32_t dim, uint8_t * input_x, uint64_t * output_y, int32_t bw_y) {
//        const uint64_t cur_mod_mask = bw_y == 64 ? -1ULL : (1ULL << bw_y) - 1;
//        if (party == sci::ALICE) {
//            uint64_t *corr_data = new uint64_t[dim];
//            for (size_t i = 0; i < dim; i++) {
//                corr_data[i] = input_x[i] ^ 1;
//            }
//            this->otpack->iknp_straight->send_cot(output_y, corr_data, dim, bw_y);
//            this->otpack->io->flush();
//            for (size_t i = 0; i < dim; i++) {
//                output_y[i] = ((uint64_t) input_x[i] - output_y[i]) & cur_mod_mask;
//            }
//            delete[] corr_data;
//        } else {
//            this->otpack->iknp_straight->recv_cot(output_y, (bool *)input_x, dim, bw_y);
//        }
//    };
//
//    // extend bit-width
//    uint8_t * msb = new uint8_t[dim];
//    for (size_t i = 0; i < dim; i++) {
//        msb[i] = (bool) (tmpA[i] & ring_msb_mask);
//    }
//    uint64_t * msb_or_res = new uint64_t[dim]; // wrap = (msb0 || msb1)
//    OR_then_B2A(dim, msb, msb_or_res, shift_bw);
//    if (party == sci::ALICE) {
//        for (size_t i = 0; i < dim; i++) {
//            tmpB[i] = (static_cast<__uint128_t>(tmpA[i]) - (static_cast<__uint128_t>(msb_or_res[i]) << ring_bw)) & tmp_mod_mask;
//        }
//    } else {
//        for (size_t i = 0; i < dim; i++) {
//            tmpB[i] = (static_cast<__uint128_t>(tmpA[i]) - (static_cast<__uint128_t>(msb_or_res[i]) << ring_bw) - big_number_128) & tmp_mod_mask;
//        }
//    }
//    delete[] msb;
//    delete[] msb_or_res;
//
//    // local convert
//    if (party == sci::ALICE) {
//        for (size_t i = 0; i < dim; i++) {
//            outB[i] = tmpB[i] % field_mod;
//        }
//    } else {
//        const uint64_t wrap_error = tmp_mod_mask % field_mod + 1;
//        for (size_t i = 0; i < dim; i++) {
//            outB[i] = (tmpB[i] + field_mod - wrap_error) % field_mod;
//        }
//    }
//    delete[] tmpA;
//    delete[] tmpB;
//}

void Truncation::ring_to_field(int32_t dim, uint64_t *inA, uint64_t *outB, int32_t ring_bw, uint64_t field_mod) {
    const uint64_t ring_mod_mask = (ring_bw == 64 ? -1ULL : (1ULL << ring_bw) - 1);
    const uint64_t ring_msb_mask = 1ULL << (ring_bw - 1);
    const uint64_t big_number = 1ULL << (ring_bw - 2);
    uint64_t *tmpA = new uint64_t[dim]; // add a big number

    // add a big number
    if (party == sci::ALICE) {
        for (uint32_t i = 0; i < dim; i++) {
            tmpA[i] = (inA[i] + big_number) & ring_mod_mask;
        }
    } else {
        memcpy(tmpA, inA, dim * sizeof(uint64_t));
    }

    auto secure_add = [] (const uint64_t x, const uint64_t y, const uint64_t p) -> uint64_t {
        if (x >= (p - y)) return x - (p - y);
        else return x + y;
    };

    auto secure_sub = [] (const uint64_t x, const uint64_t y, const uint64_t p) -> uint64_t {
        if (x >= y) return x - y;
        else return x + (p - y);
    };

    auto OR_AUX = [&] (int32_t dim, uint64_t corr, const uint8_t * input_x, uint64_t * output_y, uint64_t p) {
        if (party == sci::ALICE) {
            uint64_t *corr_data = new uint64_t[dim];
            for (size_t i = 0; i < dim; i++) {
                corr_data[i] = input_x[i] ? 0 : corr;
            }
            this->otpack->iknp_straight->send_cot_prime(output_y, corr_data, dim, p);
            this->otpack->io->flush();
            for (size_t i = 0; i < dim; i++) {
                uint64_t tmp = secure_sub(0, output_y[i], p);
                output_y[i] = input_x[i] ? secure_add(corr, tmp, p) : tmp;
            }
            delete[] corr_data;
        } else {
            this->otpack->iknp_straight->recv_cot_prime(output_y, (bool *)input_x, dim, p);
        }
    };

    uint8_t * msb = new uint8_t[dim];
    for (size_t i = 0; i < dim; i++) {
        msb[i] = (bool) (tmpA[i] & ring_msb_mask);
    }
    uint64_t * msb_or_aux_res = new uint64_t[dim];
    OR_AUX(dim, ring_mod_mask % field_mod + 1, msb, msb_or_aux_res, field_mod);
    delete[] msb;

    if (party == sci::ALICE) {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = secure_sub(tmpA[i], msb_or_aux_res[i], field_mod);
            outB[i] = secure_sub(outB[i], big_number % field_mod, field_mod);
        }
    } else {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = secure_sub(tmpA[i], msb_or_aux_res[i], field_mod);
        }
    }
    delete[] tmpA;
    delete[] msb_or_aux_res;
}

void Truncation::field_to_ring(int32_t dim, uint64_t *inA, uint64_t *outB, uint64_t field_mod, int32_t ring_bw) { // 1 COT F2R
    const uint64_t ring_mod_mask = (ring_bw == 64 ? -1ULL : (1ULL << ring_bw) - 1);
    const uint64_t big_number = field_mod / 4; // big number = p / 4
    const uint64_t half_field = field_mod / 2 + 1;

    uint64_t *tmpA = new uint64_t[dim]; // add a big number

    // add a big number
    if (this->party == sci::ALICE) {
        for (uint32_t i = 0; i < dim; i++) {
            if (inA[i] >= field_mod - big_number) tmpA[i] = (inA[i] - field_mod + big_number);
            else tmpA[i] = inA[i] + big_number;
        }
    } else {
        memcpy(tmpA, inA, dim * sizeof(uint64_t));
    }

    auto OR_AUX = [&] (int32_t dim, const uint64_t p, uint8_t * input_x, uint64_t * output_y, int32_t bw_y) {
        const uint64_t cur_mod_mask = bw_y == 64 ? -1ULL : (1ULL << bw_y) - 1;
        if (this->party == sci::ALICE) {
            uint64_t *corr_data = new uint64_t[dim];
            for (size_t i = 0; i < dim; i++) {
                corr_data[i] = input_x[i] ? 0 : p;
            }
            this->otpack->iknp_straight->send_cot(output_y, corr_data, dim, bw_y);
            this->otpack->io->flush();
            for (size_t i = 0; i < dim; i++) {
                output_y[i] = input_x[i] ? ((p - output_y[i]) & cur_mod_mask) : (-output_y[i] & cur_mod_mask);
            }
            delete[] corr_data;
        } else {
            this->otpack->iknp_straight->recv_cot(output_y, (bool *)input_x, dim, bw_y);
        }
    };

    uint8_t * msb = new uint8_t[dim];
    for (size_t i = 0; i < dim; i++) {
        msb[i] = (tmpA[i] >= half_field);
    }
    uint64_t * or_aux_res = new uint64_t[dim]; // wrap = (msb0 || msb1)
    OR_AUX(dim, field_mod, msb, or_aux_res, ring_bw);
    delete[] msb;

    if (this->party == sci::ALICE) {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (tmpA[i] - or_aux_res[i] - big_number) & ring_mod_mask;
        }
    } else {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (tmpA[i] - or_aux_res[i]) & ring_mod_mask;
        }
    }
    delete[] or_aux_res;
    delete[] tmpA;
}

void Truncation::field_to_ring_with_truncate(int32_t dim,
                                             uint64_t *inA, uint64_t *outB,
                                             uint64_t field_mod, int32_t ring_bw, int32_t shift_bw) {
    const uint64_t ring_mod_mask = (ring_bw == 64 ? -1ULL : (1ULL << ring_bw) - 1);
    const uint64_t big_number = field_mod / 4; // big number = p / 4
    const uint64_t half_field = field_mod / 2 + 1;

    uint64_t *tmpA = new uint64_t[dim]; // add a big number

    // add a big number
    if (this->party == sci::ALICE) {
        for (uint32_t i = 0; i < dim; i++) {
            if (inA[i] >= field_mod - big_number) tmpA[i] = (inA[i] - field_mod + big_number);
            else tmpA[i] = inA[i] + big_number;
        }
    } else {
        memcpy(tmpA, inA, dim * sizeof(uint64_t));
    }

    auto OR_AUX = [&] (int32_t dim, const uint64_t p, uint8_t * input_x, uint64_t * output_y, int32_t bw_y) {
        const uint64_t cur_mod_mask = bw_y == 64 ? -1ULL : (1ULL << bw_y) - 1;
        if (this->party == sci::ALICE) {
            uint64_t *corr_data = new uint64_t[dim];
            for (size_t i = 0; i < dim; i++) {
                corr_data[i] = input_x[i] ? 0 : (p >> shift_bw);
            }
            this->otpack->iknp_straight->send_cot(output_y, corr_data, dim, bw_y);
            this->otpack->io->flush();
            for (size_t i = 0; i < dim; i++) {
                output_y[i] = input_x[i] ? (((p >> shift_bw) - output_y[i]) & cur_mod_mask) : (-output_y[i] & cur_mod_mask);
            }
            delete[] corr_data;
        } else {
            this->otpack->iknp_straight->recv_cot(output_y, (bool *)input_x, dim, bw_y);
        }
    };

    uint8_t * msb = new uint8_t[dim];
    for (size_t i = 0; i < dim; i++) {
        msb[i] = (tmpA[i] >= half_field);
    }
    uint64_t * or_aux_res = new uint64_t[dim]; // wrap = (msb0 || msb1)
    OR_AUX(dim, field_mod, msb, or_aux_res, ring_bw);
    delete[] msb;

    if (this->party == sci::ALICE) {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = (((tmpA[i] - big_number) >> shift_bw) - or_aux_res[i]) & ring_mod_mask;
        }
    } else {
        for (size_t i = 0; i < dim; i++) {
            outB[i] = ((tmpA[i] >> shift_bw) - or_aux_res[i] + 1) & ring_mod_mask;
        }
    }
    delete[] or_aux_res;
    delete[] tmpA;
}
