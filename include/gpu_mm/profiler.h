// Minimal lightweight profiling utility for OpenMM (GPU-M&M project).
//
// Task M0-T1: a reusable TimerScope + Profiler facade. This header introduces
// NO external dependencies, touches no protocol logic, and is disabled by
// default. It exists only to give later tasks a uniform timing harness around
// CPU-baseline primitives so that GPU/cuOT/Phantom/CUDA replacements can be
// compared against the same measuring stick.
//
// Design notes:
//  - Header-only so tests can use it without linking a new library.
//  - Profiler is a static facade over a Meyers-singleton holding the shared
//    state. std::mutex guards all shared access for simple multi-threaded use.
//  - Disabled by default: TimerScope is a cheap no-op when Enable(false).
//    Checking a single atomic bool avoids taking the mutex on the hot path.
//  - Time unit is microseconds internally; summaries are reported in
//    milliseconds (matching the [PROFILE] ... total_ms=... shape in the task).
#ifndef GPU_MM_PROFILER_H
#define GPU_MM_PROFILER_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <map>
#include <mutex>
#include <ostream>
#include <string>

namespace gpu_mm {

// A single named timing region aggregated across all TimerScope exits.
struct ProfileStat {
    uint64_t count = 0;          // number of TimerScope exits recorded
    uint64_t total_us = 0;       // total elapsed microseconds
};

// Thread-safe accumulator of named timing regions.
//
// Usage:
//   gpu_mm::Profiler::Reset();
//   gpu_mm::Profiler::Enable(true);
//   { gpu_mm::TimerScope t("smoke.region"); /* work */ }
//   gpu_mm::Profiler::PrintSummary(std::cout);
class Profiler {
   public:
    // Discard all recorded regions.
    static void Reset() {
        State& s = instance();
        std::lock_guard<std::mutex> lk(s.mutex);
        s.stats.clear();
    }

    // Enable/disable recording at runtime. Disabled by default.
    static void Enable(bool on) { instance().enabled.store(on); }

    // Whether recording is currently enabled.
    static bool IsEnabled() { return instance().enabled.load(); }

    // Manually record an elapsed duration (microseconds) for a named region.
    // Useful when a caller times something without a TimerScope. No-op when
    // disabled, so manual recording respects the global flag too.
    static void Record(const std::string& name, uint64_t elapsed_us) {
        State& s = instance();
        if (!s.enabled.load()) return;
        std::lock_guard<std::mutex> lk(s.mutex);
        ProfileStat& st = s.stats[name];
        st.count += 1;
        st.total_us += elapsed_us;
    }

    // Print one [PROFILE] line per region, in insertion order (std::map).
    // Format: [PROFILE] <name> count=<n> total_ms=<x> avg_ms=<y>
    static void PrintSummary(std::ostream& os) {
        State& s = instance();
        std::lock_guard<std::mutex> lk(s.mutex);
        for (const auto& kv : s.stats) {
            const ProfileStat& st = kv.second;
            double total_ms = static_cast<double>(st.total_us) / 1000.0;
            double avg_ms = (st.count > 0)
                                ? total_ms / static_cast<double>(st.count)
                                : 0.0;
            os << "[PROFILE] " << kv.first << " count=" << st.count
               << " total_ms=" << std::fixed << std::setprecision(3) << total_ms
               << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms
               << "\n";
        }
        os.flush();
    }

   private:
    struct State {
        std::mutex mutex;
        std::atomic<bool> enabled{false};
        std::map<std::string, ProfileStat> stats;
    };

    static State& instance() {
        static State s;
        return s;
    }
};

// RAII timing scope. Constructor captures the start clock; destructor records
// the elapsed microseconds under `name` via Profiler::Record (which honors the
// global enable flag). When disabled, no state is touched and no clock is read
// on destruction beyond a single atomic load.
class TimerScope {
   public:
    explicit TimerScope(const std::string& name) : name_(name) {
        if (Profiler::IsEnabled()) {
            start_ = Clock::now();
            active_ = true;
        }
    }

    ~TimerScope() {
        if (active_) {
            auto end = Clock::now();
            uint64_t us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end - start_)
                    .count());
            Profiler::Record(name_, us);
        }
    }

    // Non-copyable, non-movable: a scope owns its single timing interval.
    TimerScope(const TimerScope&) = delete;
    TimerScope& operator=(const TimerScope&) = delete;
    TimerScope(TimerScope&&) = delete;
    TimerScope& operator=(TimerScope&&) = delete;

   private:
    using Clock = std::chrono::high_resolution_clock;
    std::string name_;
    Clock::time_point start_;
    bool active_ = false;
};

}  // namespace gpu_mm

#endif  // GPU_MM_PROFILER_H
