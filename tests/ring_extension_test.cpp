#include "BuildingBlocks/truncation.h"
#include <iostream>

using namespace sci;
using namespace std;
#define CHECK
// #undef CHECK

int party, port = 32000;
sci::NetIO *io;
sci::OTPack<sci::NetIO> * otpack;
sci::IKNP<sci::NetIO> * iknpOT;
AuxProtocols * aux_protocol;
Truncation * truncation_protocol;

PRG128 prg;

const uint32_t warm_up_time = 1;
const uint32_t repeat_time = 5;
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

void ring_extension128_test(const uint32_t N, const int32_t bw_in, const int32_t bw_out) {
     io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);

    const uint64_t maskA = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));
    const __uint128_t maskB = (bw_out == 128 ? static_cast<__uint128_t>(-1) : (static_cast<__uint128_t>(1) << bw_out) - 1);

    vector<uint64_t> inA(N), inA_bob;
    vector<__uint128_t> outB(N);
    vector<int64_t> data_value;

    const uint64_t value_mask = (1ULL << (bw_in - 1)) - 1;
    const uint64_t big_number = (1ULL << (bw_in - 2));

    // generate input data
    if (party == sci::ALICE) {
        data_value.resize(N);
        inA_bob.resize(N);
        prg.random_data(data_value.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) {
            data_value[i] = (data_value[i] & value_mask) - big_number;
        }

        io->recv_data(inA_bob.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) {
            inA[i] = (data_value[i] - inA_bob[i]) & maskA;
        }
    } else {
        prg.random_data(inA.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) {
            inA[i] &= maskA;
        }
        io->send_data(inA.data(), N * sizeof(uint64_t));
    }

    io->sync();
    auto time_stamp0 = std::chrono::high_resolution_clock::now();
    auto io_cnt_0 = io->counter;

    otpack = new OTPack<sci::NetIO>(io, party);
    aux_protocol = new AuxProtocols(party, io, otpack);
    truncation_protocol = new Truncation(party, io, otpack, aux_protocol);

    truncation_protocol->ring_to_ring128(N, inA.data(), outB.data(), bw_in, bw_out);

    auto io_cnt_1 = io->counter;
    io->sync();
    auto time_stamp1 = std::chrono::high_resolution_clock::now();

    uint64_t client_comm, server_comm;
    if (party == sci::ALICE) {
        server_comm = io_cnt_1 - io_cnt_0;
        io->recv_data(&client_comm, sizeof(uint64_t));
//        cout << "N = 2^" << log2(N) << endl;
//        cout << "ring extension " << bw_in << " bits -> " << bw_out << " bits" << ", K - M = " << (bw_out - bw_in) << " bits" << endl;
//        cout << "ring extension comm = " << (server_comm + client_comm) * 8.0 / 1024.0 / 1024.0 << " M bits" << endl;
        // cout << "average ring extension comm = " << (server_comm + client_comm) * 8.0 / N << " bits" << endl;
//        cout << bw_in << " -> " << bw_out << ", runtime = "
//             << std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count() << "ms" << endl;
        all_runtime.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count());
        all_communication_0.emplace_back(client_comm);
        all_communication_1.emplace_back(server_comm);
    } else {
        client_comm = io_cnt_1 - io_cnt_0;
        io->send_data(&client_comm, sizeof(uint64_t));
    }
    #ifdef CHECK
    auto signed_val_128 = [] (__uint128_t x, int32_t bw_x) -> int64_t {
        __uint128_t half_pow_x = static_cast<__uint128_t>(1) << (bw_x - 1);
        __uint128_t mask_x = ((half_pow_x - 1) << 1) | 1;
        if (bw_x != 128) x = x & mask_x;
        if (x >= half_pow_x) return -int64_t(half_pow_x - (x - half_pow_x));
        else return int64_t(x);
    };
    if (party == sci::ALICE) {
        vector<__uint128_t> outB_bob(N);
        io->recv_data(outB_bob.data(), N * sizeof(__uint128_t));
        cout << "Testing for correctness..." << endl;
        bool flag = true;
        for (int i = 0; i < N && flag; i++) {
            flag &= (data_value[i] == signed_val_128(outB[i] + outB_bob[i], bw_out));
            if (!flag) {
                cout << "error at " << i << endl;
                cout << data_value[i] << " -> " << signed_val_128(outB[i] + outB_bob[i], bw_out) << endl;
            }
        }
        if (flag) cout << "Correct!" << endl;
        cout << endl;
    } else { // BOB
        io->send_data(outB.data(), N * sizeof(__uint128_t));
    }
#endif // CHECK
    io->sync();
    delete truncation_protocol;
    delete aux_protocol;
    delete otpack;
    delete io;
}

int main(int argc, char **argv) {
    party = atoi(argv[1]);

    if (party == sci::ALICE) {
        cout << "Ring Extension test" << endl;
    }

    const int32_t shift_bw = std::atoi(argv[2]);
    if (shift_bw < 1 || 64 < shift_bw) {
        cout << "current support shift_bw : [1, 64]"  << endl;
        return 0;
    }

    const uint32_t N = 10'000'000;
    const int32_t base_bw = 64;
    for (size_t i = 0; i < warm_up_time + repeat_time; i++) {
        ring_extension128_test(N, base_bw, base_bw + shift_bw);
    }

    if (party == sci::ALICE) {
        cout << "warm round = " << warm_up_time << ", repeat round = " << repeat_time << endl;
        cout << "batch size = " << N << endl;
        cout << "input bit width = " << base_bw << " bits" << endl;
        cout << "target bit width = " << base_bw << " + " << shift_bw << " bits" << endl;
        cout << "runtime = " << get_average_runtime() << "ms" << endl;
        cout << "P0 -> P1 communication = " << get_average_communication(0) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "P1 -> P0 communication = " << get_average_communication(1) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "All communication = " << (get_average_communication(0) + get_average_communication(1)) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "P0 -> P1 communication per ring_extension = " << get_average_communication(0) * 8.0 / N << " bits" << endl;
        cout << "P1 -> P0 communication per ring_extension = " << get_average_communication(1) * 8.0 / N << " bits" << endl;
        cout << "All communication per ring_extension = " << (get_average_communication(0) + get_average_communication(1)) * 8.0 / N << " bits" << endl;
    }

    return 0;
}