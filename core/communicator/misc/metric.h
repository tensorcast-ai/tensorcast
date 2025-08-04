
// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef COMMUNICATOR_MISC_METRIC_H_
#define COMMUNICATOR_MISC_METRIC_H_

#include <cstdint>

namespace stepcast::communicator {

class Timer {
 public:
  Timer(bool init_record = false);
  uint64_t record();

 private:
  uint64_t start_;
  uint64_t end_;
};

} // namespace stepcast::communicator

#endif // COMMUNICATOR_MISC_METRIC_H_
