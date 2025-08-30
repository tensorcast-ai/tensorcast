
// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/misc/metric.h"
#include "core/communicator/misc/utils.h"

namespace tensorcast::communicator {

Timer::Timer(bool init_record) : start_(0), end_(0) {
  if (init_record) {
    record();
  }
}

uint64_t Timer::record() {
  // Always record timing metrics (env gating removed)
  end_ = get_us();
  if (start_ == 0) {
    start_ = end_;
    return 0;
  }
  uint64_t cost = end_ - start_;
  start_ = end_;
  return cost;
}

} // namespace tensorcast::communicator
