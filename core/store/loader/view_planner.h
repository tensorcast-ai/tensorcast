// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"

namespace tensorcast::store::loader {

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

struct SelectionPlan {
  struct Range {
    enum class Kind : uint8_t { kData = 0, kPad = 1 };
    Kind kind{Kind::kData};
    uint64_t src_offset{0};
    uint64_t dst_offset{0};
    uint64_t length{0};
  };

  std::vector<Range> ranges;
  bool is_contiguous{false};
  uint32_t num_ranges{0};
  uint64_t total_bytes{0};
  bool is_segment_aligned{false};
  bool requires_materialization{false};
};

struct TensorTransformPlan {
  std::string tensor_name;
  uint64_t dst_offset{0};
  uint64_t canonical_offset{0};
  uint64_t storage_offset_elements{0};
  std::vector<int64_t> canonical_shape;
  std::vector<int64_t> canonical_stride;
  std::vector<int64_t> view_shape;
  std::vector<int64_t> view_stride;
  std::vector<int64_t> permutation;
  std::string dtype;
  uint64_t element_size_bytes{0};
};

struct TransformPlan {
  std::vector<TensorTransformPlan> tensors;
  bool requires_materialization{false};

  [[nodiscard]] bool empty() const {
    return tensors.empty();
  }
};

struct ViewPlan {
  bool is_identity{true};
  uint64_t view_size_bytes{0};
  std::string view_index_json;
  SelectionPlan selection;
  TransformPlan transform;
};

struct ViewWritePlan {
  struct Chunk {
    uint64_t canonical_offset{0};
    uint64_t view_offset{0};
    uint64_t length{0};
    bool segment_aligned{false};
  };

  std::vector<Chunk> chunks;

  [[nodiscard]] bool empty() const {
    return chunks.empty();
  }
};

struct BidirectionalViewPlan {
  ViewPlan forward;
  ViewWritePlan write;
  TransformPlan inverse_transform;
};

class ViewPlanner {
 public:
  [[nodiscard]] static absl::StatusOr<ViewPlan> compute_view_plan(
      std::string_view canonical_index_json,
      const ViewSpec& spec);

  [[nodiscard]] static absl::StatusOr<BidirectionalViewPlan> compute_bidirectional_view_plan(
      std::string_view canonical_index_json,
      const ViewSpec& spec);
};

} // namespace tensorcast::store::loader
