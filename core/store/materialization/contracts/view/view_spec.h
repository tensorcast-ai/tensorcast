// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tensorcast::store::materialization::view {

struct NarrowOp {
  int32_t dim{0};
  int64_t start{0};
  uint64_t length{0};
};

struct TransposeOp {
  int32_t dim0{0};
  int32_t dim1{1};
};

struct ViewOp {
  enum class Kind : uint8_t { kNarrow = 0, kTranspose = 1 };

  Kind kind{Kind::kNarrow};
  NarrowOp narrow{};
  TransposeOp transpose{};

  static ViewOp Narrow(const NarrowOp& op) {
    ViewOp out;
    out.kind = Kind::kNarrow;
    out.narrow = op;
    return out;
  }

  static ViewOp Transpose(const TransposeOp& op) {
    ViewOp out;
    out.kind = Kind::kTranspose;
    out.transpose = op;
    return out;
  }
};

struct TensorViewOps {
  std::vector<ViewOp> ops;
};

struct ViewSpec {
  std::map<std::string, TensorViewOps> tensors;
};

} // namespace tensorcast::store::materialization::view
