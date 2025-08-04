
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/communicator/misc/metric.h"
#include "core/communicator/misc/envs.h"
#include "core/communicator/misc/utils.h"

namespace stepcast::communicator {

ENV_PARAM(TIMER, 1);

Timer::Timer(bool init_record) : start_(0), end_(0) {
  if (init_record) {
    record();
  }
}

uint64_t Timer::record() {
  if (!TIMER) {
    return 0;
  }
  end_ = get_us();
  if (start_ == 0) {
    start_ = end_;
    return 0;
  }
  uint64_t cost = end_ - start_;
  start_ = end_;
  return cost;
}

} // namespace stepcast::communicator
