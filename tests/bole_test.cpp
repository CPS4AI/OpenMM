#include "BuildingBlocks/truncation.h"
#include "cheetah/cheetah-api.h"
#include <iostream>

using namespace sci;
using namespace std;
#define CHECK
// #undef CHECK

int party, port = 33000;
sci::NetIO *io;
sci::OTPack<sci::NetIO> * otpack;
sci::IKNP<sci::NetIO> * iknpOT;
AuxProtocols * aux_protocol;
Truncation * truncation_protocol;
gemini::CheetahLinear * cheetahLinear_protocol;

// 32 : 4294828033ULL
// Polynomial Degree = 2^12, |p| = 32, |q| = 109
const int32_t bitwidth = 32;
const int32_t kScale = 12;
const uint64_t field_mod = sci::default_prime_mod.at(bitwidth);

PRG128 prg;

const uint32_t warm_up_time = 0;
const uint32_t repeat_time = 1;
vector<uint64_t> all_runtime;
vector<uint64_t> all_communication_0, all_communication_1;

uint64_t get_average_runtime() {
    assert(all_runtime.size() == warm_up_time + repeat_time);
    return accumulate(all_runtime.begin() + warm_up_time, all_runtime.end(), 0ULL) / repeat_time;
}

uint64_t get_average_communication(int p) {
    auto comm_vct = (p ? all_communication_1 : all_communication_0);
    assert(comm_vct.size() == warm_up_time + repeat_time);
    return accumulate(comm_vct.begin() + warm_up_time, comm_vct.end(), 0ULL) / repeat_time;
}

static inline uint64_t getFieldElt(int64_t x) {
    if (x >= 0) return x % field_mod;
    else return field_mod - 1 - ((-x - 1) % field_mod);
}

static inline int64_t getSignedVal_Field(uint64_t x, uint64_t p) {
    assert(x < field_mod);
    int64_t sx = x;
    if (x >= (p + 1) / 2)
        sx = x - p;
    return sx;
}

uint64_t mulmod(uint64_t x, uint64_t y, uint64_t mod) {
    uint64_t cur = x % mod;
    uint64_t res = 0;
    while (y) {
        if (y & 1) {
            res += cur;
            if (mod <= res)
                res -= mod;
        }
        y >>= 1;
        cur += cur;
        if (mod <= cur)
            cur -= mod;
    }
    return res;
}


void BOLE_test(const uint32_t N, const int32_t bw_in, const uint64_t prime_mod) {
    io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);
    cheetahLinear_protocol = new gemini::CheetahLinear(party, io, 1ULL << bw_in, 1);

    const uint64_t maskA = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));
    const uint64_t max_value = 1ULL << 12;

    // generate random input data
    vector<uint64_t> input_x(N), client_input_x(N), input_y(N), client_input_y(N), output_z(N), client_output_z(N);
    vector<int64_t> x_value(N), y_value(N);

    gemini::CheetahLinear::BNMeta meta;
    meta.target_base_mod = prime_mod;
    meta.is_shared_input = true;
    meta.vec_shape = gemini::TensorShape({N});

    gemini::Tensor<uint64_t> field_x, field_y, field_z;
    field_x.Reshape(meta.vec_shape);
    field_y.Reshape(meta.vec_shape);
    field_z.Reshape(meta.vec_shape);

    // generate input data x, y \in [-2^15, 2^15)
    if (party == sci::ALICE) {
        prg.random_data(x_value.data(), N * sizeof(uint64_t));
        for (auto &p: x_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;
        io->recv_data(client_input_x.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++) {
            input_x[i] = (x_value[i] - client_input_x[i]) & maskA;
        }

        prg.random_data(y_value.data(), N * sizeof(uint64_t));
        for (auto &p: y_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;

        io->recv_data(client_input_y.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++) {
            input_y[i] = (y_value[i] - client_input_y[i]) & maskA;
        }
    } else {
        prg.random_data(input_x.data(), N * sizeof(uint64_t));
        for (auto &p: input_x)
            p &= maskA;
        io->send_data(input_x.data(), N * sizeof(uint64_t));

        prg.random_data(input_y.data(), N * sizeof(uint64_t));
        for (auto &p: input_y)
            p &= maskA;
        io->send_data(input_y.data(), N * sizeof(uint64_t));
    }

    io->sync();
    auto time_stamp0 = std::chrono::high_resolution_clock::now();
    auto io_cnt_0 = io->counter;

    otpack = new OTPack<sci::NetIO>(io, party);
    aux_protocol = new AuxProtocols(party, io, otpack);
    truncation_protocol = new Truncation(party, io, otpack, aux_protocol);

    truncation_protocol->ring_to_field(N, input_x.data(), field_x.data(), bw_in, prime_mod);
    truncation_protocol->ring_to_field(N, input_y.data(), field_y.data(), bw_in, prime_mod);

    cheetahLinear_protocol->BOLE(field_x, field_y, meta, field_z); // element-wise product BOLE

    truncation_protocol->field_to_ring(N, field_z.data(), output_z.data(), prime_mod, bw_in);

    auto io_cnt_1 = io->counter;
    io->sync();
    auto time_stamp1 = std::chrono::high_resolution_clock::now();

    uint64_t client_comm, server_comm;
    if (party == sci::ALICE) {
        server_comm = io_cnt_1 - io_cnt_0;
        io->recv_data(&client_comm, sizeof(uint64_t));
        all_runtime.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count());
        all_communication_0.emplace_back(client_comm);
        all_communication_1.emplace_back(server_comm);
    } else {
        client_comm = io_cnt_1 - io_cnt_0;
        io->send_data(&client_comm, sizeof(uint64_t));
    }

#ifdef CHECK
    if (party == sci::ALICE) {
        vector<uint64_t> client_f_x(N), client_f_y(N), client_f_z(N);
        io->recv_data(client_f_x.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_y.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_z.data(), N * sizeof(uint64_t));
        io->recv_data(client_output_z.data(), N * sizeof(uint64_t));
        cout << "Testing for correctness..." << endl;
        bool flag = true;
        for (int i = 0; i < N && flag; i++) {
            int64_t correct_res = x_value[i] * y_value[i];
            int64_t bole_res = signed_val(output_z[i] + client_output_z[i], bw_in);
            if (correct_res != bole_res) {
                flag = false;
                cout << "error at " << i << endl;
                cout << getSignedVal_Field((field_x.data()[i] + client_f_x[i]) % prime_mod, prime_mod) << " "
                     << getSignedVal_Field((field_y.data()[i] + client_f_y[i]) % prime_mod, prime_mod) << endl;
                cout << "field z = "
                     << getSignedVal_Field((field_z.data()[i] + client_f_z[i]) % prime_mod, prime_mod) << endl;
                cout << x_value[i] << " " << y_value[i] << endl;
                cout << correct_res << " " << bole_res << endl;
            }
        }
        if (flag) cout << "Correct!" << endl;
        cout << endl << flush;
    } else { // BOB
        io->send_data(field_x.data(), N * sizeof(uint64_t));
        io->send_data(field_y.data(), N * sizeof(uint64_t));
        io->send_data(field_z.data(), N * sizeof(uint64_t));
        io->send_data(output_z.data(), N * sizeof(uint64_t));
    }
#endif // CHECK
    // cout << endl;
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    delete io;
}

void BOLE_truncate_test(const uint32_t N, const int32_t bw_in, const uint64_t prime_mod) {
    io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);
    cheetahLinear_protocol = new gemini::CheetahLinear(party, io, 1ULL << bw_in, 1);

    const uint64_t maskA = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));
    const uint64_t max_value = 1ULL << 12;

    // generate random input data
    vector<uint64_t> input_x(N), client_input_x(N), input_y(N), client_input_y(N), output_z(N), client_output_z(N);
    vector<int64_t> x_value(N), y_value(N);

    gemini::CheetahLinear::BNMeta meta;
    meta.target_base_mod = prime_mod;
    meta.is_shared_input = true;
    meta.vec_shape = gemini::TensorShape({N});

    gemini::Tensor<uint64_t> field_x, field_y, field_z;
    field_x.Reshape(meta.vec_shape);
    field_y.Reshape(meta.vec_shape);
    field_z.Reshape(meta.vec_shape);

    // generate input data x, y \in [-2^15, 2^15)
    if (party == sci::ALICE) {
        prg.random_data(x_value.data(), N * sizeof(uint64_t));
        for (auto &p: x_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;
        io->recv_data(client_input_x.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++) {
            input_x[i] = (x_value[i] - client_input_x[i]) & maskA;
        }

        prg.random_data(y_value.data(), N * sizeof(uint64_t));
        for (auto &p: y_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;

        io->recv_data(client_input_y.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++) {
            input_y[i] = (y_value[i] - client_input_y[i]) & maskA;
        }
    } else {
        prg.random_data(input_x.data(), N * sizeof(uint64_t));
        for (auto &p: input_x)
            p &= maskA;
        io->send_data(input_x.data(), N * sizeof(uint64_t));

        prg.random_data(input_y.data(), N * sizeof(uint64_t));
        for (auto &p: input_y)
            p &= maskA;
        io->send_data(input_y.data(), N * sizeof(uint64_t));
    }

    io->sync();
    auto time_stamp0 = std::chrono::high_resolution_clock::now();
    auto io_cnt_0 = io->counter;

    otpack = new OTPack<sci::NetIO>(io, party);
    aux_protocol = new AuxProtocols(party, io, otpack);
    truncation_protocol = new Truncation(party, io, otpack, aux_protocol);

    truncation_protocol->ring_to_field(N, input_x.data(), field_x.data(), bw_in, prime_mod);
    truncation_protocol->ring_to_field(N, input_y.data(), field_y.data(), bw_in, prime_mod);

    cheetahLinear_protocol->BOLE(field_x, field_y, meta, field_z); // element-wise product BOLE

    truncation_protocol->field_to_ring_with_truncate(N, field_z.data(), output_z.data(), prime_mod, bw_in, kScale);

    auto io_cnt_1 = io->counter;
    io->sync();
    auto time_stamp1 = std::chrono::high_resolution_clock::now();

    uint64_t client_comm, server_comm;
    if (party == sci::ALICE) {
        server_comm = io_cnt_1 - io_cnt_0;
        io->recv_data(&client_comm, sizeof(uint64_t));
        all_runtime.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count());
        all_communication_0.emplace_back(client_comm);
        all_communication_1.emplace_back(server_comm);
    } else {
        client_comm = io_cnt_1 - io_cnt_0;
        io->send_data(&client_comm, sizeof(uint64_t));
    }

#ifdef CHECK
    if (party == sci::ALICE) {
        vector<uint64_t> client_f_x(N), client_f_y(N), client_f_z(N);
        io->recv_data(client_f_x.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_y.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_z.data(), N * sizeof(uint64_t));
        io->recv_data(client_output_z.data(), N * sizeof(uint64_t));
        cout << "Testing for correctness..." << endl;
        bool flag = true;
        for (int i = 0; i < N && flag; i++) {
            int64_t correct_res = (x_value[i] * y_value[i]) / ((int64_t)1 << kScale);
            int64_t bole_res = signed_val(output_z[i] + client_output_z[i], bw_in);
            if (abs(correct_res - bole_res) > 1) {
                flag = false;
                cout << "error at " << i << endl;
                cout << getSignedVal_Field((field_x.data()[i] + client_f_x[i]) % prime_mod, prime_mod) << " "
                     << getSignedVal_Field((field_y.data()[i] + client_f_y[i]) % prime_mod, prime_mod) << endl;
                cout << "field z = "
                     << getSignedVal_Field((field_z.data()[i] + client_f_z[i]) % prime_mod, prime_mod) << endl;
                cout << x_value[i] << " " << y_value[i] << endl;
                cout << correct_res << " " << bole_res << endl;
            }
        }
        if (flag) cout << "Correct!" << endl;
        cout << endl << flush;
    } else { // BOB
        io->send_data(field_x.data(), N * sizeof(uint64_t));
        io->send_data(field_y.data(), N * sizeof(uint64_t));
        io->send_data(field_z.data(), N * sizeof(uint64_t));
        io->send_data(output_z.data(), N * sizeof(uint64_t));
    }
#endif // CHECK
    // cout << endl;
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    delete io;
}

void BatchNorm_test(const uint32_t N, const int32_t bw_in, const uint64_t prime_mod) {
    io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);
    cheetahLinear_protocol = new gemini::CheetahLinear(party, io, 1ULL << bw_in, 1);

    const uint64_t maskA = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));
    const uint64_t max_value = 1ULL << 12;
    // generate random input data

    vector<uint64_t> input_x(N), client_input_x(N), input_y(N), output_z(N), client_output_z(N);
    vector<int64_t> x_value(N), y_value(N);

    gemini::CheetahLinear::BNMeta meta;
    meta.target_base_mod = prime_mod;
    meta.is_shared_input = true;
    meta.vec_shape = gemini::TensorShape({N});

    gemini::Tensor<uint64_t> field_x, field_y, field_z;
    field_x.Reshape(meta.vec_shape);
    field_y.Reshape(meta.vec_shape);
    field_z.Reshape(meta.vec_shape);

    // generate input data x, y \in [-2^15, 2^15)
    if (party == sci::ALICE) {
        prg.random_data(x_value.data(), N * sizeof(uint64_t));
        for (auto &p: x_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;
        io->recv_data(client_input_x.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++) {
            input_x[i] = (x_value[i] - client_input_x[i]) & maskA;
        }

        prg.random_data(y_value.data(), N * sizeof(uint64_t));
        for (auto &p: y_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;

        for (uint32_t i = 0; i < N; i++) {
            input_y[i] = y_value[i]& maskA;
            field_y.data()[i] = getFieldElt(y_value[i]);
        }
    } else {
        prg.random_data(input_x.data(), N * sizeof(uint64_t));
        for (auto &p: input_x)
            p &= maskA;
        io->send_data(input_x.data(), N * sizeof(uint64_t));

        memset(input_y.data(), 0, N * sizeof(uint64_t));
        memset(field_y.data(), 0, N * sizeof(uint64_t));
    }

    io->sync();
    auto io_cnt_0 = io->counter;
    auto time_stamp0 = std::chrono::high_resolution_clock::now();

    otpack = new OTPack<sci::NetIO>(io, party);
    aux_protocol = new AuxProtocols(party, io, otpack);
    truncation_protocol = new Truncation(party, io, otpack, aux_protocol);

    truncation_protocol->ring_to_field(N, input_x.data(), field_x.data(), bw_in, prime_mod);

    cheetahLinear_protocol->bn(field_x, field_y, meta, field_z); // element-wise product BOLE

    truncation_protocol->field_to_ring(N, field_z.data(), output_z.data(), prime_mod, bw_in);

    auto io_cnt_1 = io->counter;
    io->sync();
    auto time_stamp1 = std::chrono::high_resolution_clock::now();

    uint64_t client_comm, server_comm;
    if (party == sci::ALICE) {
        server_comm = io_cnt_1 - io_cnt_0;
        io->recv_data(&client_comm, sizeof(uint64_t));
        all_runtime.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count());
        all_communication_0.emplace_back(client_comm);
        all_communication_1.emplace_back(server_comm);
    } else {
        client_comm = io_cnt_1 - io_cnt_0;
        io->send_data(&client_comm, sizeof(uint64_t));
    }
#ifdef CHECK
    if (party == sci::ALICE) {
        vector<uint64_t> client_f_x(N), client_f_y(N), client_f_z(N);
        io->recv_data(client_f_x.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_y.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_z.data(), N * sizeof(uint64_t));
        io->recv_data(client_output_z.data(), N * sizeof(uint64_t));
        cout << "Testing for correctness..." << endl;
        bool flag = true;
        for (int i = 0; i < N && flag; i++) {
            int64_t correct_res = x_value[i] * y_value[i];
            int64_t bole_res = signed_val(output_z[i] + client_output_z[i], bw_in);
            if (correct_res != bole_res) {
                flag = false;
                cout << "error at " << i << endl;
                cout << getSignedVal_Field((field_x.data()[i] + client_f_x[i]) % prime_mod, prime_mod) << " "
                     << getSignedVal_Field((field_y.data()[i] + client_f_y[i]) % prime_mod, prime_mod) << endl;
                cout << "field z = "
                     << getSignedVal_Field((field_z.data()[i] + client_f_z[i]) % prime_mod, prime_mod) << endl;
                cout << x_value[i] << " " << y_value[i] << endl;
                cout << correct_res << " " << bole_res << endl;
            }
        }
        if (flag) cout << "Correct!" << endl;
        cout << endl << flush;
    } else { // BOB
        io->send_data(field_x.data(), N * sizeof(uint64_t));
        io->send_data(field_y.data(), N * sizeof(uint64_t));
        io->send_data(field_z.data(), N * sizeof(uint64_t));
        io->send_data(output_z.data(), N * sizeof(uint64_t));
    }
#endif // CHECK
    // cout << endl;
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    delete io;
}

void BatchNorm_truncate_test(const uint32_t N, const int32_t bw_in, const uint64_t prime_mod) {
    io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);
    cheetahLinear_protocol = new gemini::CheetahLinear(party, io, 1ULL << bw_in, 1);

    const uint64_t maskA = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));
    const uint64_t max_value = 1ULL << 12;
    // generate random input data

    vector<uint64_t> input_x(N), client_input_x(N), input_y(N), output_z(N), client_output_z(N);
    vector<int64_t> x_value(N), y_value(N);

    gemini::CheetahLinear::BNMeta meta;
    meta.target_base_mod = prime_mod;
    meta.is_shared_input = true;
    meta.vec_shape = gemini::TensorShape({N});

    gemini::Tensor<uint64_t> field_x, field_y, field_z;
    field_x.Reshape(meta.vec_shape);
    field_y.Reshape(meta.vec_shape);
    field_z.Reshape(meta.vec_shape);

    // generate input data x, y \in [-2^15, 2^15)
    if (party == sci::ALICE) {
        prg.random_data(x_value.data(), N * sizeof(uint64_t));
        for (auto &p: x_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;
        io->recv_data(client_input_x.data(), N * sizeof(uint64_t));
        for (uint32_t i = 0; i < N; i++) {
            input_x[i] = (x_value[i] - client_input_x[i]) & maskA;
        }

        prg.random_data(y_value.data(), N * sizeof(uint64_t));
        for (auto &p: y_value)
            p = ((uint64_t) p % (2 * max_value)) - max_value;

        for (uint32_t i = 0; i < N; i++) {
            input_y[i] = y_value[i]& maskA;
            field_y.data()[i] = getFieldElt(y_value[i]);
        }
    } else {
        prg.random_data(input_x.data(), N * sizeof(uint64_t));
        for (auto &p: input_x)
            p &= maskA;
        io->send_data(input_x.data(), N * sizeof(uint64_t));

        memset(input_y.data(), 0, N * sizeof(uint64_t));
        memset(field_y.data(), 0, N * sizeof(uint64_t));
    }

    io->sync();
    auto io_cnt_0 = io->counter;
    auto time_stamp0 = std::chrono::high_resolution_clock::now();

    otpack = new OTPack<sci::NetIO>(io, party);
    aux_protocol = new AuxProtocols(party, io, otpack);
    truncation_protocol = new Truncation(party, io, otpack, aux_protocol);

    truncation_protocol->ring_to_field(N, input_x.data(), field_x.data(), bw_in, prime_mod);

    cheetahLinear_protocol->bn(field_x, field_y, meta, field_z); // element-wise product BOLE

    truncation_protocol->field_to_ring_with_truncate(N, field_z.data(), output_z.data(), prime_mod, bw_in, kScale);

    auto io_cnt_1 = io->counter;
    io->sync();
    auto time_stamp1 = std::chrono::high_resolution_clock::now();

    uint64_t client_comm, server_comm;
    if (party == sci::ALICE) {
        server_comm = io_cnt_1 - io_cnt_0;
        io->recv_data(&client_comm, sizeof(uint64_t));
        all_runtime.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count());
        all_communication_0.emplace_back(client_comm);
        all_communication_1.emplace_back(server_comm);
    } else {
        client_comm = io_cnt_1 - io_cnt_0;
        io->send_data(&client_comm, sizeof(uint64_t));
    }
#ifdef CHECK
    if (party == sci::ALICE) {
        vector<uint64_t> client_f_x(N), client_f_y(N), client_f_z(N);
        io->recv_data(client_f_x.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_y.data(), N * sizeof(uint64_t));
        io->recv_data(client_f_z.data(), N * sizeof(uint64_t));
        io->recv_data(client_output_z.data(), N * sizeof(uint64_t));
        cout << "Testing for correctness..." << endl;
        bool flag = true;
        for (int i = 0; i < N && flag; i++) {
            int64_t correct_res = x_value[i] * y_value[i] / ((int64_t)1 << kScale);
            int64_t bole_res = signed_val(output_z[i] + client_output_z[i], bw_in);
            if (abs(correct_res - bole_res) > 1) {
                flag = false;
                cout << "error at " << i << endl;
                cout << getSignedVal_Field((field_x.data()[i] + client_f_x[i]) % prime_mod, prime_mod) << " "
                     << getSignedVal_Field((field_y.data()[i] + client_f_y[i]) % prime_mod, prime_mod) << endl;
                cout << "field z = "
                     << getSignedVal_Field((field_z.data()[i] + client_f_z[i]) % prime_mod, prime_mod) << endl;
                cout << x_value[i] << " " << y_value[i] << endl;
                cout << correct_res << " " << bole_res << endl;
            }
        }
        if (flag) cout << "Correct!" << endl;
        cout << endl << flush;
    } else { // BOB
        io->send_data(field_x.data(), N * sizeof(uint64_t));
        io->send_data(field_y.data(), N * sizeof(uint64_t));
        io->send_data(field_z.data(), N * sizeof(uint64_t));
        io->send_data(output_z.data(), N * sizeof(uint64_t));
    }
#endif // CHECK
    // cout << endl;
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    delete io;
}

int main(int argc, char **argv) {
    party = atoi(argv[1]);
    int32_t both_ss_flag = atoi(argv[2]);
    int32_t do_truncate = atoi(argv[3]);

    string test_name = "BOLE test, bit-width = 32, ";
    if (both_ss_flag) test_name += "secret share vector X secret share vector";
    else test_name += "secret share vector X private vector";

    if (do_truncate) test_name += ", with free truncate";

    if (party == sci::ALICE) {
        cout << test_name << endl;
    }

    const uint32_t N = 1ULL << 20;
    for (size_t i = 0; i < warm_up_time + repeat_time; i++) {
        if (!both_ss_flag && !do_truncate) BatchNorm_test(N, bitwidth, field_mod);
        if (!both_ss_flag && do_truncate) BatchNorm_truncate_test(N, bitwidth, field_mod);
        if (both_ss_flag && !do_truncate) BOLE_test(N, bitwidth, field_mod);
        if (both_ss_flag && do_truncate) BOLE_truncate_test(N, bitwidth, field_mod);
    }

    if (party == sci::ALICE) {
        cout << "warm round = " << warm_up_time << ", repeat round = " << repeat_time << endl;
        cout << "batch size = " << N << endl;
        cout << "input bit width = " << bitwidth << " bits" << endl;
        cout << "intermedia field mod = " << field_mod << ", ≈ " << round(log2(field_mod)) << "bits" << endl;
        cout << "runtime = " << get_average_runtime() << "ms" << endl;
        cout << "P0 -> P1 communication = " << get_average_communication(0) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "P1 -> P0 communication = " << get_average_communication(1) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "All communication = " << (get_average_communication(0) + get_average_communication(1)) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "P0 -> P1 communication per bole = " << get_average_communication(0) * 8.0 / N << " bits" << endl;
        cout << "P1 -> P0 communication per bole = " << get_average_communication(1) * 8.0 / N << " bits" << endl;
        cout << "All communication per bole = " << (get_average_communication(0) + get_average_communication(1)) * 8.0 / N << " bits" << endl;
    }
    return 0;
}