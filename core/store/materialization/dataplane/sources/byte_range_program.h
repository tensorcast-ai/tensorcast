// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::loader {

struct ByteRangeRun {
  enum class Kind : uint8_t { kPad = 0, kContiguous = 1, kStrided = 2 };

  Kind kind{Kind::kPad};
  uint64_t dst_begin{0};
  uint64_t dst_end{0};
  uint32_t source_index{0};
  uint64_t src_begin{0};
  uint64_t src_base{0};
  uint64_t row_len{0};
  uint64_t stride{0};
  uint64_t rows{0};
  uint64_t rows_per_block{0};
};

struct ByteRangeProgram {
  std::vector<ByteRangeRun> runs;
  std::vector<uint64_t> run_starts;
  uint64_t total_bytes{0};
  ByteRangeFingerprint map_fingerprint;
  ByteRangeFingerprint config_fingerprint;
  uint64_t strided_candidate_runs{0};
  uint64_t strided_compile_fallback_runs{0};
  uint64_t strided_block_max_bytes{0};
  bool has_strided_runs{false};
};

ByteRangeFingerprint fingerprint_byte_range_config(const StoreEngineOptions::ByteMappingConfig& config);

class ByteRangeCompiler {
 public:
  explicit ByteRangeCompiler(StoreEngineOptions::ByteMappingConfig config, std::string path);

  absl::StatusOr<std::shared_ptr<const ByteRangeProgram>> Compile(ByteRangeMap map);

  [[nodiscard]] const StoreEngineOptions::ByteMappingConfig& config() const {
    return config_;
  }

  [[nodiscard]] const ByteRangeFingerprint& config_fingerprint() const {
    return config_fingerprint_;
  }

 private:
  StoreEngineOptions::ByteMappingConfig config_;
  ByteRangeFingerprint config_fingerprint_;
  std::string path_;
};

} // namespace tensorcast::store::loader
