// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/runtime_event_hub.h"

#include <algorithm>
#include <utility>

namespace tensorcast::store::runtime {

RuntimeEventHub::Subscription::Subscription(RuntimeEventHub* hub, int64_t id) : hub_(hub), id_(id) {}

RuntimeEventHub::Subscription::~Subscription() {
  release();
}

RuntimeEventHub::Subscription::Subscription(Subscription&& other) noexcept {
  hub_ = other.hub_;
  id_ = other.id_;
  other.hub_ = nullptr;
  other.id_ = 0;
}

RuntimeEventHub::Subscription& RuntimeEventHub::Subscription::operator=(Subscription&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  hub_ = other.hub_;
  id_ = other.id_;
  other.hub_ = nullptr;
  other.id_ = 0;
  return *this;
}

void RuntimeEventHub::Subscription::release() {
  if (hub_ && id_ != 0) {
    hub_->unsubscribe(id_);
    hub_ = nullptr;
    id_ = 0;
  }
}

std::unique_ptr<RuntimeEventHub::Subscription> RuntimeEventHub::subscribe(Callback callback) {
  auto cb_holder = std::make_shared<Callback>(std::move(callback));
  int64_t sub_id = 0;
  {
    absl::MutexLock lock(&mu_);
    sub_id = next_id_++;
    subscribers_.push_back(Subscriber{.id = sub_id, .callback = std::move(cb_holder)});
  }
  return std::make_unique<Subscription>(this, sub_id);
}

void RuntimeEventHub::publish(const RuntimeEvent& event) {
  std::vector<std::shared_ptr<Callback>> callbacks;
  {
    absl::MutexLock lock(&mu_);
    if (subscribers_.empty()) {
      return;
    }
    callbacks.reserve(subscribers_.size());
    for (const auto& subscriber : subscribers_) {
      callbacks.push_back(subscriber.callback);
    }
    ++active_publishers_;
  }

  for (const auto& cb : callbacks) {
    if (cb) {
      (*cb)(event);
    }
  }

  {
    absl::MutexLock lock(&mu_);
    --active_publishers_;
    if (active_publishers_ == 0) {
      drain_cv_.SignalAll();
    }
  }
}

void RuntimeEventHub::drain() {
  absl::MutexLock lock(&mu_);
  while (active_publishers_ > 0) {
    drain_cv_.Wait(&mu_);
  }
}

void RuntimeEventHub::unsubscribe(int64_t id) {
  absl::MutexLock lock(&mu_);
  auto it =
      std::remove_if(subscribers_.begin(), subscribers_.end(), [id](const Subscriber& sub) { return sub.id == id; });
  subscribers_.erase(it, subscribers_.end());
}

} // namespace tensorcast::store::runtime
