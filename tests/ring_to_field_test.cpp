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

// 16 : 65521ULL
// 32 : 4294828033ULL
// 40 : 1099511480321ULL
// 48 : 281474976694273ULL
// 56 : 72057594037641217ULL
// 60 : 1152921504606830593ULL
// 64 : 18446744073709551557ULL

const std::map<int32_t, uint64_t> prime_mod{
        {16, 65521ULL},
        {32, 4293918721ULL},
        {40, 1099511480321ULL},
        {48, 281474976694273ULL},
        {56, 72057594037641217ULL},
        {60, 1152921504606830593ULL},
        {64, 18446744073709551557ULL},
};

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

uint64_t add_mod(uint64_t x, uint64_t y, uint64_t p) {
    x %= p;
    y %= p;
    if (x >= p - y) return x - (p - y);
    else return x + y;
}

static inline int64_t getSignedVal_Field(uint64_t x, uint64_t p) {
    x %= p;
    int64_t sx = x;
    if (x >= (p + 1) / 2)
        sx = x - p;
    return sx;
}

void ring_to_field_test(const uint32_t N, const int32_t bw_in, const uint64_t prime_mod) {
    io = new sci::NetIO(party == 1 ? nullptr : "127.0.0.1", port);

    const uint64_t maskA = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));
    vector<uint64_t> inA(N), outB(N), inA_bob;
    vector<int64_t> data_value;

    const uint64_t max_value = min((uint64_t)1 << (bw_in - 2), prime_mod / 4);

    if (party == sci::ALICE) {
        data_value.resize(N);
        inA_bob.resize(N);
        prg.random_data(data_value.data(), N * sizeof(uint64_t));
        for (int i = 0; i < N; i++) {
            data_value[i] = (data_value[i] % (2 * max_value)) - max_value;
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

    truncation_protocol->ring_to_field(N, inA.data(), outB.data(), bw_in, prime_mod);

    auto io_cnt_1 = io->counter;
    io->sync();
    auto time_stamp1 = std::chrono::high_resolution_clock::now();

    uint64_t client_comm, server_comm;
    if (party == sci::ALICE) {
        server_comm = io_cnt_1 - io_cnt_0;
        io->recv_data(&client_comm, sizeof(uint64_t));
        // cout << "N = " << N << endl;
        // cout << "input bit width = " << bw_in << " bits" << endl;
        // cout << "target field mod = " << prime_mod << ", ≈ " << round(log2(prime_mod)) << "bits" << endl;
        // cout << "all communication = " << (server_comm + client_comm) / 1024.0 / 1024.0 << " MB" << endl;
        // cout << "average communication per ring-to-field = " << (server_comm + client_comm) * 8.0 / N << " bits" << endl;
        // cout << "runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count() << "ms" << endl;
        all_runtime.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(time_stamp1 - time_stamp0).count());
        all_communication_0.emplace_back(client_comm);
        all_communication_1.emplace_back(server_comm);
    } else {
        client_comm = io_cnt_1 - io_cnt_0;
        io->send_data(&client_comm, sizeof(uint64_t));
    }

#ifdef CHECK
    if (party == sci::ALICE) {
        vector<uint64_t> outB_bob(N);
        io->recv_data(outB_bob.data(), N * sizeof(uint64_t));
        cout << "Testing for correctness..." << endl;
        bool flag = true;
        for (int i = 0; i < N && flag; i++) {
            flag &= (data_value[i] == getSignedVal_Field(add_mod(outB[i], outB_bob[i], prime_mod), prime_mod));
            if (!flag) {
                cout << "error at " << i << endl;
                cout << data_value[i] << " " << getSignedVal_Field(add_mod(outB[i], outB_bob[i], prime_mod), prime_mod) << endl;
                cout << "diff = " <<  abs(data_value[i] - getSignedVal_Field(add_mod(outB[i], outB_bob[i], prime_mod), prime_mod)) << endl;
            }
        }
        if (flag) cout << "Correct!" << endl;
        cout << endl << flush;
    } else { // BOB
        io->send_data(outB.data(), N * sizeof(uint64_t));
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
        cout << "Ring to Field test" << endl;
    }

    const int32_t bitwidth = std::atoi(argv[2]);
    if (!prime_mod.count(bitwidth)) {
        cout << "current support bitwidth :";
        for (const auto & p : prime_mod)
            cout << " " << p.first;
        cout << endl;
        return 0;
    }

    const uint32_t N = 10'000'000;
    for (size_t i = 0; i < warm_up_time + repeat_time; i++) {
        ring_to_field_test(N, bitwidth, prime_mod.at(bitwidth));
    }

    if (party == sci::ALICE) {
        cout << "warm round = " << warm_up_time << ", repeat round = " << repeat_time << endl;
        cout << "batch size = " << N << endl;
        cout << "input bit width = " << bitwidth << " bits" << endl;
        cout << "target field mod = " << prime_mod.at(bitwidth) << ", ≈ " << round(log2(prime_mod.at(bitwidth))) << " bits" << endl;
        cout << "runtime = " << get_average_runtime() << "ms" << endl;
        cout << "P0 -> P1 communication = " << get_average_communication(0) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "P1 -> P0 communication = " << get_average_communication(1) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "All communication = " << (get_average_communication(0) + get_average_communication(1)) / 1024.0 / 1024.0 << " MB" << endl;
        cout << "P0 -> P1 communication per ring_to_field = " << get_average_communication(0) * 8.0 / N << " bits" << endl;
        cout << "P1 -> P0 communication per ring_to_field = " << get_average_communication(1) * 8.0 / N << " bits" << endl;
        cout << "All communication per ring_to_field = " << (get_average_communication(0) + get_average_communication(1)) * 8.0 / N << " bits" << endl;
    }

    return 0;
}