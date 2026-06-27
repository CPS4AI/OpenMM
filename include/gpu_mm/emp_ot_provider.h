// EmpOTProvider — emp/SCI SilentOT-based concrete OTProvider (Task M1-T2).
//
// Wraps the existing sci::OTPack (Cheetah SilentOT = Ferret VOLE) behind the
// gpu_mm::OTProvider interface. Does NOT replace any M&M OT path; it is a
// standalone, test-only seam so later cuOT can implement the same interface and
// be compared. No cuOT here.
//
// Construction owns a sci::NetIO + sci::OTPack pair. The two parties must
// construct concurrently (NetIO connects); like the existing tests, the sender
// is ALICE (party 1, server: address==nullptr) and the receiver is BOB
// (party 2, client: address=="127.0.0.1"), on a shared port.
//
// run_setup defaults true (matches OTPack default) so base OTs / Ferret
// bootstrap happen at construction. The standalone correlation test relies on
// this.
#ifndef GPU_MM_EMP_OT_PROVIDER_H
#define GPU_MM_EMP_OT_PROVIDER_H

#include "gpu_mm/ot_provider.h"

#include <memory>
#include <string>

namespace sci {
class NetIO;
template <typename T>
class OTPack;
}  // namespace sci

namespace gpu_mm {

class EmpOTProvider : public OTProvider {
   public:
    // Construct as `role` on `port`. Sender (ALICE) listens, receiver (BOB)
    // connects to `address` (default loopback). The matching peer must be
    // constructed at the same time.
    EmpOTProvider(OTParty role, int port,
                  const char* address = "127.0.0.1",
                  bool run_setup = true);
    ~EmpOTProvider() override;

    EmpOTProvider(const EmpOTProvider&) = delete;
    EmpOTProvider& operator=(const EmpOTProvider&) = delete;

    OTParty party() const override { return role_; }
    const char* backend_name() const override { return "emp"; }

    // Access the underlying network channel. Exposed so test harnesses can do
    // auxiliary verification exchanges (e.g. the sender sending its random `x`
    // back to the receiver to check the COT correlation). NOT used by protocol
    // code; the backend owns the channel.
    sci::NetIO* net_io() { return io_.get(); }

    OTStatus send_cot(uint64_t* data0, const uint64_t* corr, int n,
                      int l) override;
    OTStatus recv_cot(uint64_t* data, const bool* b, int n, int l) override;
    OTStatus send_cot_prime(uint64_t* data0, const uint64_t* corr, int n,
                            uint64_t p) override;
    OTStatus recv_cot_prime(uint64_t* data, const bool* b, int n,
                            uint64_t p) override;

   private:
    OTParty role_;
    int sci_party_;  // 1 == ALICE (sender), 2 == BOB (receiver)
    std::unique_ptr<sci::NetIO> io_;
    std::unique_ptr<sci::OTPack<sci::NetIO>> otpack_;
    bool ready_ = false;
};

}  // namespace gpu_mm

#endif  // GPU_MM_EMP_OT_PROVIDER_H
