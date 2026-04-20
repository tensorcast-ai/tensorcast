// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/sources/byte_range_program.h"

namespace tensorcast::store::loader {

class ByteRangeMappedSource final : public SeekableSource {
 public:
  struct Options {
    std::string path;
    bool enable_direct_write_at{true};
    uint64_t direct_gather_min_row_len_bytes{4ULL * 1024};
    size_t direct_gather_min_total_bytes{4ULL * 1024 * 1024};
    uint64_t direct_gather_max_rows_touched{12ULL * 1024};
  };

  static absl::StatusOr<std::unique_ptr<ByteRangeMappedSource>> Create(
      ByteRangeMap map,
      std::shared_ptr<const ByteRangeProgram> program,
      std::vector<std::shared_ptr<SeekableSource>> sources,
      Options options);

  ~ByteRangeMappedSource() override;

  [[nodiscard]] uint64_t total_bytes() const override;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  [[nodiscard]] bool supports_direct_write_at() const override;
  [[nodiscard]] bool supports_batched_direct_write_at() const override;
  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override;
  absl::StatusOr<size_t> readv_into_at(absl::Span<const DirectWriteOp> ops, const DirectWriteGrant& grant) override;

 private:
  struct StridedBlockCache;

  ByteRangeMappedSource(
      ByteRangeMap map,
      std::shared_ptr<const ByteRangeProgram> program,
      std::vector<std::shared_ptr<SeekableSource>> sources,
      Options options,
      bool direct_write_supported);

  absl::Status validate_sources(const ByteRangeMap& map) const;
  [[nodiscard]] size_t find_run_index(uint64_t offset) const;

  absl::StatusOr<size_t> read_base(uint32_t source_index, uint64_t offset, uint8_t* dst, size_t bytes);
  absl::StatusOr<size_t> copy_from_strided_rows(
      const ByteRangeRun& run,
      uint64_t run_offset,
      uint8_t* dst,
      size_t bytes);
  absl::StatusOr<size_t> fill_strided_run(
      size_t run_index,
      const ByteRangeRun& run,
      uint64_t run_offset,
      uint8_t* dst,
      size_t bytes,
      uint64_t* pack_us_total,
      size_t* pack_bytes_total,
      uint64_t* cache_lookup_us_total,
      uint64_t* block_prepare_us_total,
      uint64_t* block_load_us_total,
      uint64_t* row_copy_us_total,
      size_t* row_copy_bytes_total);
  absl::Status zero_fill_to_grant(uint64_t dest_va_offset, size_t bytes, const DirectWriteGrant& grant);
  absl::StatusOr<size_t> execute_grouped_direct_write(
      uint32_t source_index,
      absl::Span<const DirectWriteOp> ops,
      const DirectWriteGrant& grant);
  absl::StatusOr<size_t> execute_grouped_direct_write_fallback(
      uint32_t source_index,
      absl::Span<const DirectWriteOp> ops,
      const DirectWriteGrant& grant);

  struct Stats {
    std::atomic<uint64_t> base_read_calls{0};
    std::atomic<uint64_t> base_read_bytes{0};
    std::atomic<uint64_t> output_bytes{0};
    std::atomic<uint64_t> pad_bytes{0};
    std::atomic<uint64_t> pack_bytes{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> strided_runtime_fallbacks{0};
    std::atomic<uint64_t> direct_write_calls{0};
    std::atomic<uint64_t> direct_write_bytes{0};
    std::atomic<uint64_t> direct_write_fallback_calls{0};
  };

  ByteRangeMap map_;
  std::shared_ptr<const ByteRangeProgram> program_;
  std::vector<std::shared_ptr<SeekableSource>> sources_;
  Options options_;
  bool direct_write_supported_{false};
  std::unique_ptr<std::atomic<uint8_t>[]> strided_disabled_;
  std::unique_ptr<StridedBlockCache> strided_cache_;
  Stats stats_;
  uint64_t cursor_{0};
};

} // namespace tensorcast::store::loader
