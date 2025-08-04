// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_MISC_QUEUE_H_
#define CORE_COMMUNICATOR_MISC_QUEUE_H_

#include <mutex>
#include <queue>
#include <utility>

#include "core/communicator/misc/common.h"

namespace stepcast::communicator {

template <class T>
class Queue {
 public:
  explicit Queue(int capacity_num = -1) : capacity_num_(capacity_num), stop_(false) {}

  ~Queue() {
    stop_.store(true);
    cv_.notify_all();
    std::unique_lock<std::mutex> lock(mu_);
    while (!queue_.empty()) {
      queue_.pop();
    }
  }

  bool full() {
    if (capacity_num_ <= 0) {
      return false;
    }
    std::unique_lock<std::mutex> lock(mu_);
    return queue_.size() >= static_cast<uint64_t>(capacity_num_);
  }

  bool empty() {
    std::unique_lock<std::mutex> lock(mu_);
    return queue_.empty();
  }

  result_t push(const T& elem, bool block = true, uint32_t timeout = 1000000) {
    auto result = do_push(elem, block, timeout);
    notify();
    return result;
  }

  result_t do_push(const T& elem, bool block = true, uint32_t timeout = 1000000) {
    std::unique_lock<std::mutex> lock(mu_);
    if (block) {
      cv_.wait_for(lock, std::chrono::milliseconds(timeout), [this] {
        if (stop_.load()) {
          return true;
        }
        if (capacity_num_ <= 0) {
          return true;
        }
        return queue_.size() < static_cast<uint64_t>(capacity_num_);
      });
    }
    if (stop_.load()) {
      return FAILED;
    }
    if (capacity_num_ >= 0) {
      if (queue_.size() >= static_cast<uint64_t>(capacity_num_)) {
        LOG(WARNING) << "failed to push a element into a full queue!";
        return FAILED;
      }
    }

    queue_.push(elem);
    return SUCCESS;
  }

  T pop(bool block = false, uint32_t timeout = 1000000000) {
    auto val = do_pop(block, timeout);
    notify();
    return val;
  }

  T do_pop(bool block = false, uint32_t timeout = 1000000000) {
    std::unique_lock<std::mutex> lock(mu_);
    if (block) {
      cv_.wait_for(lock, std::chrono::milliseconds(timeout), [this] {
        if (stop_.load()) {
          return true;
        }
        return !queue_.empty();
      });
    }

    if (stop_.load() || queue_.empty()) {
      return T();
    }

    auto val = queue_.front();
    queue_.pop();
    return val;
  }

  result_t notify() {
    cv_.notify_all();
    return SUCCESS;
  }

  result_t stop() {
    stop_.store(true);
    cv_.notify_all();
    return SUCCESS;
  }

  result_t clear() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!queue_.empty()) {
      queue_.pop();
    }
    return SUCCESS;
  }

  int size() {
    std::unique_lock<std::mutex> lock(mu_);
    return queue_.size();
  }

 private:
  mutable std::mutex mu_;
  std::queue<T> queue_;
  std::condition_variable cv_;
  const int capacity_num_;
  std::atomic_bool stop_;
};

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_MISC_QUEUE_H_
