// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tensorcast::daemon::representation_layout {

struct TensorLayoutSpec {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t storage_offset{0};
  uint64_t logical_length{0};
  uint64_t element_size{0};
};

struct ViewNarrowSpec {
  int32_t dim{0};
  int64_t start{0};
  int64_t end{0};
};

inline bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  if (shape.empty()) {
    return stride.empty();
  }
  if (shape.size() != stride.size()) {
    return false;
  }
  int64_t expected_stride = 1;
  for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0; --index) {
    if (stride[static_cast<size_t>(index)] != expected_stride) {
      return false;
    }
    expected_stride *= shape[static_cast<size_t>(index)];
  }
  return true;
}

} // namespace tensorcast::daemon::representation_layout
