// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tensorcast::store::materialization::view {

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

} // namespace tensorcast::store::materialization::view
