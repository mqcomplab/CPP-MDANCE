#pragma once

// Cancellation and progress for long calls.
//
// A caller that runs an algorithm on a worker thread (the C API's
// mdance_control_t, the Tcl extension's jobs) installs a Control for the
// duration of the call with ControlScope. The algorithms' loops call
// MD::checkCancel(), which throws MD::Cancelled once the caller asked for a
// stop, and MD::progress() to say how far they are. Without a Control (the
// CLI, the tests) both do nothing.
//
// The pointer is per thread: concurrent calls on different threads have
// their own.

#include <atomic>
#include <stdexcept>

namespace MD {

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

struct Control {
    std::atomic<int> cancel{0};
    std::atomic<double> fraction{0.0};
    std::atomic<const char*> stage{""};     // a string literal
};

inline Control*& currentControl() {
    thread_local Control* c = nullptr;
    return c;
}

inline void checkCancel() {
    Control* c = currentControl();
    if (c && c->cancel.load(std::memory_order_relaxed)) throw Cancelled();
}

// fraction in [0, 1] of the current stage; stage a string literal
inline void progress(double fraction, const char* stage = nullptr) {
    Control* c = currentControl();
    if (!c) return;
    c->fraction.store(fraction, std::memory_order_relaxed);
    if (stage) c->stage.store(stage, std::memory_order_relaxed);
    if (c->cancel.load(std::memory_order_relaxed)) throw Cancelled();
}

class ControlScope {
    Control* prev;
public:
    explicit ControlScope(Control* c) : prev(currentControl()) { currentControl() = c; }
    ~ControlScope() { currentControl() = prev; }
    ControlScope(const ControlScope&) = delete;
    ControlScope& operator=(const ControlScope&) = delete;
};

}  // namespace MD
