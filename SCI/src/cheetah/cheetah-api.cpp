// Author: Wen-jie Lu on 2021/9/14.
#include "cheetah/cheetah-api.h"

#include <seal/seal.h>

#include "gemini/cheetah/shape_inference.h"
#include "gemini/cheetah/tensor_encoder.h"
#include "utils/constants.h"  // ALICE & BOB
#include "utils/net_io_channel.h"

template <class CtType>
void send_ciphertext(sci::NetIO *io, const CtType &ct) {
  std::stringstream os;
  uint64_t ct_size;
  ct.save(os);
  ct_size = os.tellp();
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
}

template <class EncVecCtType>
static void send_encrypted_vector(sci::NetIO *io, const EncVecCtType &ct_vec) {
  uint32_t ncts = ct_vec.size();
  io->send_data(&ncts, sizeof(uint32_t));
  for (size_t i = 0; i < ncts; ++i) {
    send_ciphertext(io, ct_vec.at(i));
  }
}

static void recv_encrypted_vector(sci::NetIO *io,
                                  const seal::SEALContext &context,
                                  std::vector<seal::Ciphertext> &ct_vec,
                                  bool is_truncated = false);
static void recv_ciphertext(sci::NetIO *io, const seal::SEALContext &context,
                            seal::Ciphertext &ct, bool is_truncated = false);

namespace gemini {

TensorShape GetConv2DOutShape(const HomConv2DSS::Meta &meta) {
  auto o = shape_inference::Conv2D(meta.ishape, meta.fshape, meta.padding,
                                   meta.stride);
  if (!o) {
    printf("GetConv2DOutShape failed\n");
    return TensorShape({0, 0, 0});
  }
  o->Update(0, meta.n_filters);
  return *o;
}

uint64_t CheetahLinear::io_counter() const { return io_ ? io_->counter : 0; }

int64_t CheetahLinear::get_signed(uint64_t x) const {
  if (x >= base_mod_) {
    LOG(FATAL) << "CheetahLinear::get_signed input out-of-bound";
  }

  // [-2^{k-1}, 2^{k-1})
  if (x > positive_upper_)
    return static_cast<int64_t>(x - base_mod_);
  else
    return static_cast<int64_t>(x);
}

uint64_t CheetahLinear::reduce(uint64_t x) const {
  if (barrett_reducer_) {
    return seal::util::barrett_reduce_64(x, *barrett_reducer_);
  } else {
    return x & mod_mask_;
  }
}

CheetahLinear::CheetahLinear(int party, sci::NetIO *io, uint64_t base_mod,
                             size_t nthreads)
    : party_(party), io_(io), nthreads_(nthreads), base_mod_(base_mod) {
  if (base_mod < 2ULL || (int)std::log2(base_mod) >= 45) {
    throw std::logic_error("CheetahLinear: base_mod out-of-bound [2, 2^45)");
  }

  const bool is_mod_2k = IsTwoPower(base_mod_);

  if (is_mod_2k) {
    mod_mask_ = base_mod_ - 1;
    positive_upper_ = base_mod_ / 2;
  } else {
    barrett_reducer_ = seal::Modulus(base_mod_);
    // [0, 7) -> (-4, 3]
    // [0, 8) -> [-4, 4]
    // [0, odd) -> [-floor(odd/2), floor(odd/2)]
    positive_upper_ = (base_mod_ + 1) >> 1;
  }

  const uint64_t plain_mod = base_mod;  // [0, 2^k)

  using namespace seal;
  EncryptionParameters seal_parms(scheme_type::bfv);
  seal_parms.set_n_special_primes(0);
  // We are not exporting the pk/ct with more than 109-bit.
  std::vector<int> moduli_bits{60, 49};

  seal_parms.set_poly_modulus_degree(4096);
  seal_parms.set_coeff_modulus(CoeffModulus::Create(4096, moduli_bits));
  seal_parms.set_plain_modulus(plain_mod);
  context_ =
      std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);

  if (party == sci::BOB) {
    // Bob generate keys
    KeyGenerator keygen(*context_);
    // Keep secret key
    sk_ = std::make_shared<SecretKey>(keygen.secret_key());
    // Send public key
    Serializable<PublicKey> s_pk = keygen.create_public_key();

    std::stringstream os;
    s_pk.save(os);
    uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
    const std::string &keys_str = os.str();

    io_->send_data(&pk_sze, sizeof(uint64_t));
    io_->send_data(keys_str.c_str(), pk_sze);

    conv2d_impl_.setUp(*context_, *sk_);
    fc_impl_.setUp(*context_, *sk_);
    bn_impl_.setUp_Base(base_mod, *context_, *sk_);
  } else {
    pk_ = std::make_shared<PublicKey>();

    uint64_t pk_sze{0};
    io_->recv_data(&pk_sze, sizeof(uint64_t));
    char *key_buf = new char[pk_sze];
    io_->recv_data(key_buf, pk_sze);
    std::stringstream is;
    is.write(key_buf, pk_sze);
    pk_->load(*context_, is);
    delete[] key_buf;

    conv2d_impl_.setUp(*context_, std::nullopt, pk_);
    fc_impl_.setUp(*context_, std::nullopt, pk_);
    bn_impl_.setUp_Base(base_mod, *context_, std::nullopt, pk_);
  }

  if (is_mod_2k) {
      setUpForBN();
  } else {
      throw std::runtime_error("Not implemented yet.");
  }
}

void CheetahLinear::setUpForBN() {
    using namespace seal;
    const size_t N = 4096;
    const uint64_t field_mod = sci::default_prime_mod.at((int)std::log2(base_mod_));
    EncryptionParameters seal_parms(scheme_type::bfv);
    seal_parms.set_n_special_primes(0);
    seal_parms.set_plain_modulus(field_mod);
    // We are not exporting the pk/ct with more than 109-bit.
    std::vector<int> cipher_moduli_bits{60, 49};
    seal_parms.set_poly_modulus_degree(N);
    seal_parms.set_coeff_modulus(CoeffModulus::Create(N, cipher_moduli_bits));

    simd_bn_context_ = std::make_shared<SEALContext>(seal_parms, true, sec_level_type::tc128);

    if (party_ == sci::BOB) {
        KeyGenerator keygen(*simd_bn_context_);
        simd_bn_sk_ = std::make_shared<SecretKey>(keygen.secret_key());
        Serializable<PublicKey> s_pk = keygen.create_public_key();

        std::stringstream os;
        s_pk.save(os);
        uint64_t pk_sze = static_cast<uint64_t>(os.tellp());
        const std::string &keys_str = os.str();

        io_->send_data(&pk_sze, sizeof(uint64_t));
        io_->send_data(keys_str.c_str(), pk_sze);

        auto code = bn_impl_.setUp_Field(field_mod, *simd_bn_context_, *simd_bn_sk_, {});
        if (code != Code::OK) {
            throw std::runtime_error("BN setUp failed [" + CodeMessage(code) + "]");
        }
    } else {
        simd_bn_pk_ = std::make_shared<PublicKey>();
        uint64_t pk_sze{0};
        io_->recv_data(&pk_sze, sizeof(uint64_t));
        char *key_buf = new char[pk_sze];
        io_->recv_data(key_buf, pk_sze);
        std::stringstream is;
        is.write(key_buf, pk_sze);
        simd_bn_pk_->load(*simd_bn_context_, is);
        delete[] key_buf;

        auto code = bn_impl_.setUp_Field(field_mod, *simd_bn_context_, std::nullopt, simd_bn_pk_);
        if (code != Code::OK) {
            throw std::runtime_error("BN setUp failed [" + CodeMessage(code) + "]");
        }
    }
}

void SummaryTensor(Tensor<double> const &t, std::string tag) {
  double mn = *std::min_element(t.data(), t.data() + t.NumElements());
  double mx = *std::max_element(t.data(), t.data() + t.NumElements());
  std::cout << tag << " shape " << t.shape() << " values in [" << mn << ","
            << mx << "]\n";
}

bool CheetahLinear::verify(const Tensor<uint64_t> &in_tensor_share,
                           const std::vector<Tensor<uint64_t>> &filters,
                           const ConvMeta &meta,
                           const Tensor<uint64_t> &out_tensor_share,
                           int nbit_precision) const {
  size_t n_fitlers = filters.size();
  if (n_fitlers < 1) {
    LOG(WARNING) << "CheetahLinear::verify number of filters = 0";
    return false;
  }

  TensorShape ishape = in_tensor_share.shape();
  TensorShape fshape = filters[0].shape();

  auto oshape = GetConv2DOutShape(meta);

  if (!oshape.IsSameSize(out_tensor_share.shape())) {
    LOG(WARNING) << "CheetahLinear::verify oshape mismatch";
    return false;
  }
  if (!ishape.IsSameSize(meta.ishape)) {
    LOG(WARNING) << "CheetahLinear::verify ishape mismatch";
    return false;
  }
  if (!fshape.IsSameSize(meta.fshape)) {
    LOG(WARNING) << "CheetahLinear::verify fshape mismatch";
    return false;
  }

  if (party_ == sci::BOB) {
    io_->send_data(in_tensor_share.data(),
                   sizeof(uint64_t) * in_tensor_share.NumElements());
    io_->send_data(out_tensor_share.data(),
                   sizeof(uint64_t) * out_tensor_share.NumElements());
  } else {
    std::vector<uint64_t> in_tensor_raw(ishape.num_elements());
    std::vector<uint64_t> out_tensor_raw(oshape.num_elements());
    io_->recv_data(in_tensor_raw.data(),
                   sizeof(uint64_t) * in_tensor_raw.size());
    io_->recv_data(out_tensor_raw.data(),
                   sizeof(uint64_t) * out_tensor_raw.size());

    auto in_tensor = Tensor<uint64_t>::Wrap(in_tensor_raw.data(), ishape);
    auto out_tensor = Tensor<uint64_t>::Wrap(out_tensor_raw.data(), oshape);

    // Reconstruct
    in_tensor.tensor() += in_tensor_share.tensor();
    out_tensor.tensor() += out_tensor_share.tensor();

    in_tensor.tensor() =
        in_tensor.tensor().unaryExpr([this](uint64_t v) { return reduce(v); });
    out_tensor.tensor() =
        out_tensor.tensor().unaryExpr([this](uint64_t v) { return reduce(v); });

    auto cast_to_double = [this](uint64_t v, int nbits) -> double {
      // reduce to [0, p) from [0, 2p)
      int64_t sv = get_signed(reduce(v));
      return sv / std::pow(2., nbits);
    };

    Tensor<double> f64_in(in_tensor.shape());
    f64_in.tensor() = in_tensor.tensor().unaryExpr(
        [&](uint64_t v) { return cast_to_double(v, nbit_precision); });

    SummaryTensor(f64_in, "in_tensor");

    Tensor<uint64_t> ground;
    conv2d_impl_.idealFunctionality(in_tensor, filters, meta, ground);

    int cnt_err{0};
    for (auto c = 0; c < out_tensor.channels(); ++c) {
      for (auto h = 0; h < out_tensor.height(); ++h) {
        for (auto w = 0; w < out_tensor.width(); ++w) {
          int64_t g = get_signed(ground(c, h, w));
          int64_t g_ = get_signed(out_tensor(c, h, w));
          if (g != g_) {
            ++cnt_err;
          }
        }
      }
    }

    if (cnt_err == 0) {
      std::cout << "HomConv: matches" << std::endl;
    } else {
      std::cout << "HomConv: failed" << std::endl;
    }
  }

  return true;
}

void CheetahLinear::fc(const Tensor<uint64_t> &input_vector,
                       const Tensor<uint64_t> &weight_matrix,
                       const FCMeta &meta,
                       Tensor<uint64_t> &out_vec_share) const {
  // out_matrix = input_matrix * weight_matrix
  if (!input_vector.shape().IsSameSize(meta.input_shape)) {
    throw std::invalid_argument("CheetahLinear::fc input shape mismatch");
  }

  if (party_ == sci::ALICE &&
      !weight_matrix.shape().IsSameSize(meta.weight_shape)) {
    throw std::invalid_argument("CheetahLinear::fc weight shape mismatch");
  }

  TensorShape out_shape({meta.weight_shape.dim_size(0)});
  if (!out_vec_share.shape().IsSameSize(out_shape)) {
    // NOTE(Wen-jie) If the out_matrix may already wrap some memory
    // Then this Reshape will raise error.
    out_vec_share.Reshape(out_shape);
  }

  const auto &impl = fc_impl_;

  Code code;
  int nthreads = nthreads_;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = impl.encryptInputVector(input_vector, meta, ct_buff, nthreads);
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::fc encryptInputVector [" +
                                 CodeMessage(code) + "]");
      }
      send_encrypted_vector(io_, ct_buff);
    }

    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff);
    code = impl.decryptToVector(ct_buff, meta, out_vec_share, nthreads);

    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc decryptToVector [" +
                               CodeMessage(code) + "]");
    }
  } else {
    std::vector<std::vector<seal::Plaintext>> encoded_matrix;
    code =
        impl.encodeWeightMatrix(weight_matrix, meta, encoded_matrix, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc encodeWeightMatrix error [" +
                               CodeMessage(code) + "]");
    }
    std::vector<seal::Plaintext> vec_share1;
    if (meta.is_shared_input) {
      code = impl.encodeInputVector(input_vector, meta, vec_share1, nthreads);
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::fc encodeInputVector error [" +
                                 CodeMessage(code) + "]");
      }
    }

    uint32_t ncts{0};
    io_->recv_data(&ncts, sizeof(uint32_t));
    std::vector<seal::Ciphertext> vec_share0(ncts);
    for (size_t i = 0; i < ncts; ++i) {
      recv_ciphertext(io_, *context_, vec_share0[i]);
    }

    std::vector<seal::Ciphertext> out_vec_share0;
    auto code = impl.matVecMul(encoded_matrix, vec_share0, vec_share1, meta,
                               out_vec_share0, out_vec_share, nthreads);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::fc matmul2D error [" +
                               CodeMessage(code) + "]");
    }
    send_encrypted_vector(io_, out_vec_share0);
  }
}

void CheetahLinear::conv2d(const Tensor<uint64_t> &in_tensor,
                           const std::vector<Tensor<uint64_t>> &filters,
                           const ConvMeta &meta,
                           Tensor<uint64_t> &out_tensor) const {
  if (!meta.ishape.IsSameSize(in_tensor.shape())) {
    throw std::invalid_argument("CheetahLinear::conv2d meta.ishape mismatch");
  }
  if (meta.n_filters != filters.size()) {
    throw std::invalid_argument(
        "CheetahLinear::conv2d meta.n_filters mismatch");
  }
  for (const auto &f : filters) {
    if (!meta.fshape.IsSameSize(f.shape())) {
      throw std::invalid_argument("CheetahLinear::conv2d meta.fshape mismatch");
    }
  }

  const auto &impl = conv2d_impl_;

  Code code;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = impl.encryptImage(in_tensor, meta, ct_buff, nthreads_);
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::conv2d encryptImage " +
                                 CodeMessage(code));
      }
      send_encrypted_vector(io_, ct_buff);
    }

    // Wait for result
    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff, true);

    code = impl.decryptToTensor(ct_buff, meta, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv2d decryptToTensor " +
                               CodeMessage(code));
    }
  } else {
    std::vector<std::vector<seal::Plaintext>> encoded_filters;
    code = impl.encodeFilters(filters, meta, encoded_filters, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv2d ecnodeFilters " +
                               CodeMessage(code));
    }

    std::vector<seal::Plaintext> encoded_share;
    if (meta.is_shared_input) {
      code = impl.encodeImage(in_tensor, meta, encoded_share, nthreads_);
      if (code != Code::OK) {
        throw std::runtime_error("CheetahLinear::conv2d encodeImage " +
                                 CodeMessage(code));
      }
    }

    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff, false);

    std::vector<seal::Ciphertext> out_ct;
    auto code = impl.conv2DSS(ct_buff, encoded_share, encoded_filters, meta,
                              out_ct, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("CheetahLinear::conv2d conv2DSS: " +
                               CodeMessage(code));
    }
    send_encrypted_vector(io_, out_ct);
  }
}

//void CheetahLinear::bn(const Tensor<uint64_t> &input_vector,
//                       const Tensor<uint64_t> &scale_vector, const BNMeta &meta,
//                       Tensor<uint64_t> &out_vector) const { // original CRT SIMD 3 sub-modulus BOLE
//  if (meta.is_shared_input &&
//      !input_vector.shape().IsSameSize(meta.vec_shape)) {
//    throw std::runtime_error("bn input_vector shape mismatch");
//  }
//  Code code;
//  if (party_ == sci::BOB) {
//    {
//      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
//      code = bn_impl_.encryptVector(input_vector, meta, ct_buff, nthreads_);
//      if (code != Code::OK) {
//        throw std::runtime_error("bn encryptVector [" + CodeMessage(code) +
//                                 "]");
//      }
//      code = bn_impl_.sendEncryptVector(io_, ct_buff, meta);
//      if (code != Code::OK) {
//        throw std::runtime_error("bn sendEncryptVector [" + CodeMessage(code) +
//                                 "]");
//      }
//    }
//
//    std::vector<seal::Ciphertext> ct_buff;
//    code = bn_impl_.recvEncryptVector(io_, ct_buff, meta);
//    if (code != Code::OK) {
//      throw std::runtime_error("bn recvEncryptVector [" + CodeMessage(code) +
//                               "]");
//    }
//
//    code = bn_impl_.decryptToVector(ct_buff, meta, out_vector, nthreads_);
//    if (code != Code::OK) {
//      throw std::runtime_error("bn decryptToVector [" + CodeMessage(code) +
//                               "]");
//    }
//  } else {
//    if (!scale_vector.shape().IsSameSize(meta.vec_shape)) {
//      throw std::runtime_error("bn scale_vector shape mismatch");
//    }
//
//    std::vector<seal::Plaintext> encoded_vector;
//    std::vector<seal::Plaintext> encoded_scales;
//    if (meta.is_shared_input) {
//      code =
//          bn_impl_.encodeVector(input_vector, meta, encoded_vector, nthreads_);
//      if (code != Code::OK) {
//        throw std::runtime_error("bn encodeVector [" + CodeMessage(code) + "]");
//      }
//    }
//
//    code = bn_impl_.encodeScales(scale_vector, meta, encoded_scales, nthreads_);
//    if (code != Code::OK) {
//      throw std::runtime_error("bn encodeScales [" + CodeMessage(code) + "]");
//    }
//
//    std::vector<seal::Ciphertext> encrypted_vector;
//    code = bn_impl_.recvEncryptVector(io_, encrypted_vector, meta);
//    if (code != Code::OK) {
//      throw std::runtime_error("bn recvEncryptVector [" + CodeMessage(code) +
//                               "]");
//    }
//    if (encrypted_vector.size() != encoded_scales.size()) {
//      LOG(FATAL) << "vector / scales size mismatch";
//    }
//    std::vector<seal::Ciphertext> out_ct;
//    code = bn_impl_.bn(encrypted_vector, encoded_vector, encoded_scales, meta,
//                       out_ct, out_vector, nthreads_);
//    if (code != Code::OK) {
//      throw std::runtime_error("bn failed [" + CodeMessage(code) + "]");
//    }
//
//    code = bn_impl_.sendEncryptVector(io_, out_ct, meta);
//    if (code != Code::OK) {
//      throw std::runtime_error("bn sendEncryptVector [" + CodeMessage(code) +
//                               "]");
//    }
//  }
//}

void CheetahLinear::bn(const Tensor<uint64_t> &input_vector,
                       const Tensor<uint64_t> &scale_vector, const BNMeta &meta,
                       Tensor<uint64_t> &out_vector) const { // modified single modulus SIMD BOLE
    if (meta.is_shared_input &&
        !input_vector.shape().IsSameSize(meta.vec_shape)) {
        throw std::runtime_error("bn input_vector shape mismatch");
    }
    Code code;
    if (party_ == sci::BOB) { // BOB
        // encrypt then send
        // std::cerr << "BOLE BOB encrypt then send" << std::endl;
        std::vector<seal::Serializable<seal::Ciphertext>> send_ct_buff;
        code = bn_impl_.encryptVector(input_vector, meta, send_ct_buff, nthreads_);
        if (code != Code::OK) {
            throw std::runtime_error("bn encryptVector [" + CodeMessage(code) + "]");
        }
        code = bn_impl_.sendEncryptVector(io_, send_ct_buff, meta);
        if (code != Code::OK) {
            throw std::runtime_error("bn sendEncryptVector [" + CodeMessage(code) + "]");
        }
        // send ct over

        // std::cerr << "BOLE BOB recv then decrypt" << std::endl;
        std::vector<seal::Ciphertext> recv_ct_buff;
        code = bn_impl_.recvEncryptVector(io_, recv_ct_buff, meta);
        if (code != Code::OK) {
            throw std::runtime_error("bn recvEncryptVector [" + CodeMessage(code) + "]");
        }

        code = bn_impl_.decryptToVector(recv_ct_buff, meta, out_vector, nthreads_);
        if (code != Code::OK) {
            throw std::runtime_error("bn decryptToVector [" + CodeMessage(code) + "]");
        }
    } else { // ALICE
        if (!scale_vector.shape().IsSameSize(meta.vec_shape)) {
            throw std::runtime_error("bn scale_vector shape mismatch");
        }
        // encode input vector and scale vector
        // std::cerr << "BOLE ALICE encode" << std::endl;
        std::vector<seal::Plaintext> encoded_vector;
        std::vector<seal::Plaintext> encoded_scales;
        if (meta.is_shared_input) {
            code = bn_impl_.encodeVector(input_vector, meta, encoded_vector, nthreads_);
            if (code != Code::OK) {
                throw std::runtime_error("bn encodeVector [" + CodeMessage(code) + "]");
            }
        }
        code = bn_impl_.encodeScales(scale_vector, meta, encoded_scales, nthreads_);
        if (code != Code::OK) {
            throw std::runtime_error("bn encodeScales [" + CodeMessage(code) + "]");
        }
        // recv ciphertexts
        // std::cerr << "BOLE ALICE recv" << std::endl;
        std::vector<seal::Ciphertext> encrypted_vector;
        code = bn_impl_.recvEncryptVector(io_, encrypted_vector, meta);

        if (code != Code::OK) {
            throw std::runtime_error("bn recvEncryptVector [" + CodeMessage(code) + "]");
        }
        if (encrypted_vector.size() != encoded_scales.size()) {
            LOG(FATAL) << "vector / scales size mismatch";
        }
        // bn (simd-based element-wise product)
        std::vector<seal::Ciphertext> out_ct;
        // std::cerr << "BOLE ALICE cp mul" << std::endl;
        code = bn_impl_.bn(encrypted_vector, encoded_vector, encoded_scales, meta, out_ct, out_vector, nthreads_);
        if (code != Code::OK) {
            throw std::runtime_error("bn failed [" + CodeMessage(code) + "]");
        }
        // response ciphertexts
        // std::cerr << "BOLE ALICE response" << std::endl;
        code = bn_impl_.sendEncryptVector(io_, out_ct, meta);
        if (code != Code::OK) {
            throw std::runtime_error("bn sendEncryptVector [" + CodeMessage(code) + "]");
        }
    } // end of ALICE
}

void CheetahLinear::BOLE(const Tensor<uint64_t> & input_x_vector,
                             const Tensor<uint64_t> & input_y_vector,
                             const BNMeta & meta,
                             Tensor<uint64_t> &out_vector) const {
    if (!input_x_vector.shape().IsSameSize(meta.vec_shape) ||
        !input_y_vector.shape().IsSameSize(meta.vec_shape)) {
        throw std::runtime_error("BOLE input_vector shape mismatch");
    }
    Code code;
    if (party_ == sci::BOB) { // BOB
        // encrypt then send
        std::vector<seal::Serializable<seal::Ciphertext>> send_x_ct_buff, send_y_ct_buff;
        bn_impl_.encryptVector(input_x_vector, meta, send_x_ct_buff, nthreads_);
        bn_impl_.encryptVector(input_y_vector, meta, send_y_ct_buff, nthreads_);

        bn_impl_.sendEncryptVector(io_, send_x_ct_buff, meta);
        bn_impl_.sendEncryptVector(io_, send_y_ct_buff, meta);
        // send ct over

        // recv then decrypt
        std::vector<seal::Ciphertext> recv_ct_buff;
        bn_impl_.recvEncryptVector(io_, recv_ct_buff, meta);
        bn_impl_.decryptToVector(recv_ct_buff, meta, out_vector, nthreads_);
    } else { // ALICE
        // encode input vector
        std::vector<seal::Plaintext> encoded_x_vector, encoded_y_vector;
        bn_impl_.encodeVector(input_x_vector, meta, encoded_x_vector, nthreads_);
        bn_impl_.encodeVector(input_y_vector, meta, encoded_y_vector, nthreads_);

        // recv ciphertexts
        std::vector<seal::Ciphertext> encrypted_x_vector, encrypted_y_vector;
        bn_impl_.recvEncryptVector(io_, encrypted_x_vector, meta);
        bn_impl_.recvEncryptVector(io_, encrypted_y_vector, meta);

        // bole (simd-based element-wise product)
        std::vector<seal::Ciphertext> out_ct;
        bn_impl_.bole(encrypted_x_vector, encrypted_y_vector,
                      encoded_x_vector, encoded_y_vector,
                      meta, out_ct, out_vector, nthreads_);

        // response ciphertexts
        bn_impl_.sendEncryptVector(io_, out_ct, meta);
    } // end of ALICE
    for (size_t i = 0; i < meta.vec_shape.length(); i++)
        out_vector.data()[i] = (out_vector.data()[i] + sci::moduloMult(input_x_vector.data()[i], input_y_vector.data()[i], meta.target_base_mod)) % meta.target_base_mod;
}

void CheetahLinear::bn_direct(const Tensor<uint64_t> &input_tensor,
                              const Tensor<uint64_t> &scale_vector,
                              const BNMeta &meta,
                              Tensor<uint64_t> &out_tensor) const {
  if (meta.is_shared_input && !input_tensor.shape().IsSameSize(meta.ishape)) {
    throw std::runtime_error("bn_direct input_vector shape mismatch");
  }
  Code code;
  if (party_ == sci::BOB) {
    {
      std::vector<seal::Serializable<seal::Ciphertext>> ct_buff;
      code = bn_impl_.encryptTensor(input_tensor, meta, ct_buff, nthreads_);
      if (code != Code::OK) {
        throw std::runtime_error("bn_direct encryptVector [" +
                                 CodeMessage(code) + "]");
      }
      send_encrypted_vector(io_, ct_buff);
    }

    std::vector<seal::Ciphertext> ct_buff;
    recv_encrypted_vector(io_, *context_, ct_buff);

    code = bn_impl_.decryptToTensor(ct_buff, meta, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("bn_direct decryptToTensor [" +
                               CodeMessage(code) + "]");
    }
  } else {
    if (scale_vector.dims() != 1 ||
        scale_vector.length() != input_tensor.channels()) {
      throw std::runtime_error("bn_direct scale_vector shape mismatch");
    }

    std::vector<seal::Plaintext> encoded_tensor;
    if (meta.is_shared_input) {
      code =
          bn_impl_.encodeTensor(input_tensor, meta, encoded_tensor, nthreads_);
      if (code != Code::OK) {
        throw std::runtime_error("bn_direct encodeVector [" +
                                 CodeMessage(code) + "]");
      }
    }

    std::vector<seal::Ciphertext> encrypted_tensor;
    recv_encrypted_vector(io_, *context_, encrypted_tensor);

    std::vector<seal::Ciphertext> out_ct;
    code = bn_impl_.bn_direct(encrypted_tensor, encoded_tensor, scale_vector,
                              meta, out_ct, out_tensor, nthreads_);
    if (code != Code::OK) {
      throw std::runtime_error("bn_direct failed [" + CodeMessage(code) + "]");
    }
    send_encrypted_vector(io_, out_ct);
  }
}

}  // namespace gemini

void recv_encrypted_vector(sci::NetIO *io, const seal::SEALContext &context,
                           std::vector<seal::Ciphertext> &ct_vec,
                           bool is_truncated) {
  uint32_t ncts{0};
  io->recv_data(&ncts, sizeof(uint32_t));
  if (ncts > 0) {
    ct_vec.resize(ncts);
    for (size_t i = 0; i < ncts; ++i) {
      recv_ciphertext(io, context, ct_vec[i], is_truncated);
    }
  }
}

void recv_ciphertext(sci::NetIO *io, const seal::SEALContext &context,
                     seal::Ciphertext &ct, bool is_truncated) {
  std::stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  if (is_truncated) {
    ct.unsafe_load(context, is);
  } else {
    ct.load(context, is);
  }
  delete[] c_enc_result;
}
