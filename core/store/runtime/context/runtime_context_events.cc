// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/context/runtime_context_events.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "folly/MPMCQueue.h"

namespace tensorcast::store::runtime {

namespace {
constexpr size_t kDefaultQueueDepth = 4096;
constexpr absl::Duration kQueueBackoff = absl::Microseconds(50);
} // namespace

struct RuntimeContextEvents::DispatcherState : public std::enable_shared_from_this<DispatcherState> {
  struct Subscriber {
    int64_t id;
    std::shared_ptr<Callback> callback;
  };

  DispatcherState() : queue(kDefaultQueueDepth) {
    worker = std::thread([this]() { dispatch_loop(); });
  }

  ~DispatcherState() {
    shutdown();
  }

  void publish(RuntimeEvent event) {
    if (!running.load(std::memory_order_acquire)) {
      return;
    }
    pending_events.fetch_add(1, std::memory_order_acq_rel);
    while (running.load(std::memory_order_acquire)) {
      if (queue.write(std::move(event))) {
        event_available.SignalAll();
        return;
      }
      absl::SleepFor(kQueueBackoff);
    }
    pending_events.fetch_sub(1, std::memory_order_acq_rel);
  }

  std::unique_ptr<Subscription> subscribe(Callback callback) {
    if (!callback) {
      return nullptr;
    }
    auto holder = std::make_shared<Callback>(std::move(callback));
    int64_t id = 0;
    {
      absl::MutexLock lock(&subscriber_mu);
      id = next_id++;
      subscribers.push_back(Subscriber{.id = id, .callback = std::move(holder)});
    }
    return std::make_unique<Subscription>(shared_from_this(), id);
  }

  void unsubscribe(int64_t id) {
    absl::MutexLock lock(&subscriber_mu);
    auto it =
        std::remove_if(subscribers.begin(), subscribers.end(), [id](const Subscriber& sub) { return sub.id == id; });
    subscribers.erase(it, subscribers.end());
  }

  void drain() {
    absl::MutexLock lock(&drain_mu);
    while (pending_events.load(std::memory_order_acquire) != 0) {
      drain_cv.Wait(&drain_mu);
    }
  }

  void shutdown() {
    bool expected = true;
    if (!running.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
      return;
    }
    event_available.SignalAll();
    if (worker.joinable()) {
      worker.join();
    }
  }

  void dispatch_loop() {
    RuntimeEvent event;
    while (running.load(std::memory_order_acquire) || pending_events.load(std::memory_order_acquire) > 0) {
      if (queue.read(event)) {
        notify(event);
        pending_events.fetch_sub(1, std::memory_order_acq_rel);
        notify_drain_waiters();
        continue;
      }
      absl::MutexLock lock(&event_mu);
      event_available.WaitWithTimeout(&event_mu, kQueueBackoff);
    }

    while (queue.read(event)) {
      notify(event);
      pending_events.fetch_sub(1, std::memory_order_acq_rel);
      notify_drain_waiters();
    }
    notify_drain_waiters();
  }

  void notify(const RuntimeEvent& event) {
    std::vector<std::shared_ptr<Callback>> callbacks;
    {
      absl::MutexLock lock(&subscriber_mu);
      callbacks.reserve(subscribers.size());
      for (const auto& sub : subscribers) {
        callbacks.push_back(sub.callback);
      }
    }

    for (const auto& cb : callbacks) {
      if (cb && *cb) {
        (*cb)(event);
      }
    }
  }

  void notify_drain_waiters() {
    if (pending_events.load(std::memory_order_acquire) == 0) {
      absl::MutexLock lock(&drain_mu);
      if (pending_events.load(std::memory_order_relaxed) == 0) {
        drain_cv.SignalAll();
      }
    }
  }

  folly::MPMCQueue<RuntimeEvent> queue;
  absl::Mutex subscriber_mu;
  std::vector<Subscriber> subscribers ABSL_GUARDED_BY(subscriber_mu);
  std::atomic<int64_t> next_id{1};
  std::atomic<bool> running{true};
  std::atomic<int64_t> pending_events{0};
  absl::Mutex event_mu;
  absl::CondVar event_available;
  absl::Mutex drain_mu;
  absl::CondVar drain_cv;
  std::thread worker;
};

RuntimeContextEvents::RuntimeContextEvents() : state_(std::make_shared<DispatcherState>()) {}

RuntimeContextEvents::~RuntimeContextEvents() {
  if (state_) {
    state_->shutdown();
  }
}

RuntimeContextEvents::Publisher RuntimeContextEvents::publisher() {
  return Publisher(state_);
}

std::unique_ptr<RuntimeContextEvents::Subscription> RuntimeContextEvents::subscribe(Callback callback) {
  if (!state_) {
    return nullptr;
  }
  return state_->subscribe(std::move(callback));
}

void RuntimeContextEvents::drain() {
  if (state_) {
    state_->drain();
  }
}

RuntimeContextEvents::Subscription::Subscription() = default;

RuntimeContextEvents::Subscription::Subscription(std::shared_ptr<DispatcherState> state, int64_t id)
    : state_(std::move(state)), id_(id) {}

RuntimeContextEvents::Subscription::~Subscription() {
  release();
}

RuntimeContextEvents::Subscription::Subscription(Subscription&& other) noexcept {
  state_ = std::move(other.state_);
  id_ = other.id_;
  other.id_ = 0;
}

RuntimeContextEvents::Subscription& RuntimeContextEvents::Subscription::operator=(Subscription&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  state_ = std::move(other.state_);
  id_ = other.id_;
  other.id_ = 0;
  return *this;
}

void RuntimeContextEvents::Subscription::release() {
  if (state_ && id_ != 0) {
    state_->unsubscribe(id_);
    id_ = 0;
  }
}

RuntimeContextEvents::Publisher::Publisher() = default;

RuntimeContextEvents::Publisher::Publisher(std::shared_ptr<DispatcherState> state) : state_(std::move(state)) {}

void RuntimeContextEvents::Publisher::publish(RuntimeEvent event) const {
  if (state_) {
    state_->publish(std::move(event));
  }
}

} // namespace tensorcast::store::runtime
