
// Copyright (c) 2025, TensorCast Team.

#ifndef COMMUNICATOR_MISC_METRIC_H_
#define COMMUNICATOR_MISC_METRIC_H_

#include <cstdint>

namespace tensorcast::communicator {

class Timer {
 public:
  Timer(bool init_record = false);
  uint64_t record();

 private:
  uint64_t start_;
  uint64_t end_;
};

} // namespace tensorcast::communicator

#endif // COMMUNICATOR_MISC_METRIC_H_
