// EmpOTProvider implementation (Task M1-T2).
#include "gpu_mm/emp_ot_provider.h"

// Pull M&M's OT headers through the project's chosen entry points. The active
// OTPack flavor (Cheetah SilentOT vs CF2 SplitIKNP) is selected by the existing
// `USE_CHEETAH` macro the project defines for every test target — so we include
// the dispatcher `OT/emp-ot.h` rather than hard-coding cheetah-ot_pack.h. This
// keeps the wrapper consistent with whichever OT backend the project actually
// builds, and avoids a class-redefinition error from including both flavors.
#include "OT/emp-ot.h"            // sci::OTPack<sci::NetIO> (+ Ferret when USE_CHEETAH)
#include "utils/constants.h"      // sci::ALICE/BOB
#include "utils/net_io_channel.h"  // sci::NetIO

#include <stdexcept>

namespace gpu_mm {

EmpOTProvider::EmpOTProvider(OTParty role, int port, const char* address,
                             bool run_setup)
    : role_(role), sci_party_(role == OTParty::kSender ? sci::ALICE : sci::BOB) {
    // Mirror the existing tests' NetIO convention: ALICE (sender) is the server
    // (nullptr address), BOB (receiver) is the client (connects to `address`).
    const char* addr = (role == OTParty::kSender) ? nullptr : address;
    try {
        io_ = std::make_unique<sci::NetIO>(addr, port, /*quiet=*/true);
        otpack_ = std::make_unique<sci::OTPack<sci::NetIO>>(io_.get(),
                                                            sci_party_, run_setup);
        ready_ = true;
    } catch (const std::exception&) {
        ready_ = false;
    }
}

EmpOTProvider::~EmpOTProvider() {
    // otpack_ must die before io_ (it holds a raw NetIO*). unique_ptr members
    // are destroyed in reverse declaration order: otpack_ then io_. Good.
    otpack_.reset();
    io_.reset();
}

// COT uses the "straight" OT instance. In M&M, the sender (ALICE) calls
// iknp_straight->send_cot and the receiver (BOB) calls iknp_straight->recv_cot
// (see Truncation::ring_to_field etc.). iknp_straight is the SilentOT bound to
// this party's own role, so both sides use the same field.
// (Per-call error mapping is handled inline below; this helper kept minimal.)

OTStatus EmpOTProvider::send_cot(uint64_t* data0, const uint64_t* corr, int n,
                                 int l) {
    if (!ready_ || role_ != OTParty::kSender || data0 == nullptr ||
        corr == nullptr || n <= 0 || l <= 0 || l > 64) {
        return OTStatus::kInvalidArg;
    }
    try {
        // SCI's SilentOT takes a non-const corr pointer (it does not mutate, but
        // the signature is `const uint64_t*`? actually non-const in silent_ot.h).
        // Cast away const to satisfy the existing SCI signature without copying.
        otpack_->iknp_straight->send_cot(data0, const_cast<uint64_t*>(corr), n, l);
        otpack_->io->flush();
    } catch (const std::exception&) {
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus EmpOTProvider::recv_cot(uint64_t* data, const bool* b, int n, int l) {
    if (!ready_ || role_ != OTParty::kReceiver || data == nullptr ||
        b == nullptr || n <= 0 || l <= 0 || l > 64) {
        return OTStatus::kInvalidArg;
    }
    try {
        otpack_->iknp_straight->recv_cot(data, const_cast<bool*>(b), n, l);
    } catch (const std::exception&) {
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus EmpOTProvider::send_cot_prime(uint64_t* data0, const uint64_t* corr,
                                       int n, uint64_t p) {
    if (!ready_ || role_ != OTParty::kSender || data0 == nullptr ||
        corr == nullptr || n <= 0 || p < 2) {
        return OTStatus::kInvalidArg;
    }
    try {
        otpack_->iknp_straight->send_cot_prime(data0, const_cast<uint64_t*>(corr),
                                               n, p);
        otpack_->io->flush();
    } catch (const std::exception&) {
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

OTStatus EmpOTProvider::recv_cot_prime(uint64_t* data, const bool* b, int n,
                                       uint64_t p) {
    if (!ready_ || role_ != OTParty::kReceiver || data == nullptr ||
        b == nullptr || n <= 0 || p < 2) {
        return OTStatus::kInvalidArg;
    }
    try {
        otpack_->iknp_straight->recv_cot_prime(data, const_cast<bool*>(b), n, p);
    } catch (const std::exception&) {
        return OTStatus::kInternal;
    }
    return OTStatus::kOk;
}

}  // namespace gpu_mm
