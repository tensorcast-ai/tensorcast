// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>

#include "absl/status/status.h"
#include "core/store/materialization/dataplane/metadata/disk_artifact_context.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"
#include "core/store/materialization/dataplane/sources/multi_safetensors_source.h"

namespace tensorcast::store::loader {
namespace {

constexpr uint64_t kHashLeafChunkBytes = 4ULL * 1024ULL * 1024ULL;

uint64_t saturated_mul_u64(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) {
    return 0;
  }
  if (a > (std::numeric_limits<uint64_t>::max() / b)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return a * b;
}

std::function<void(uint64_t hashed_leaf_count, uint64_t total_hash_leaves)> to_leaf_progress_callback(
    uint64_t total_size,
    std::function<void(uint64_t processed_bytes, uint64_t total_bytes)> progress_cb) {
  if (!progress_cb || total_size == 0) {
    return {};
  }
  return [total_size, cb = std::move(progress_cb)](uint64_t hashed_leaf_count, uint64_t /*total_hash_leaves*/) {
    const uint64_t processed_bytes =
        std::min<uint64_t>(total_size, saturated_mul_u64(hashed_leaf_count, kHashLeafChunkBytes));
    cb(processed_bytes, total_size);
  };
}

} // namespace

absl::StatusOr<std::string> compute_data_multihash_from_disk_dir(
    const std::string& artifact_dir,
    std::function<void(uint64_t processed_bytes, uint64_t total_bytes)> progress_cb) {
  namespace fs = std::filesystem;
  fs::path dir(artifact_dir);
  auto context_or = get_disk_artifact_context(dir);
  if (!context_or.ok()) {
    return context_or.status();
  }
  const auto& context = *context_or.value();

  std::optional<uint64_t> logical_total_size;
  std::optional<uint64_t> source_total_size;
  if (context.is_safetensors() || context.tensor_index_json_present() || context.tensor_index_cbor_present()) {
    auto index_info_or = context.get_index_info(/*target_device_id=*/0);
    if (!index_info_or.ok()) {
      return index_info_or.status();
    }
    if (index_info_or->total_size_bytes > 0) {
      logical_total_size = index_info_or->total_size_bytes;
    }
    if (index_info_or->source_total_size_bytes > 0) {
      source_total_size = index_info_or->source_total_size_bytes;
    }
  }

  if (!context.is_safetensors()) {
    FilePartitionSource::Options opts;
    opts.partition_paths = context.partition_paths();
    opts.partition_sizes = context.partition_sizes();
    opts.total_size = logical_total_size.value_or(context.total_size());
    opts.io_batch_bytes = 128 * 1024 * 1024;
    FilePartitionSource src(std::move(opts));
    if (progress_cb) {
      progress_cb(0, opts.total_size);
    }
    return compute_data_multihash_from_seekable_source(
        src,
        opts.total_size,
        static_cast<size_t>(kHashLeafChunkBytes),
        to_leaf_progress_callback(opts.total_size, std::move(progress_cb)));
  }

  MultiSafetensorsSource source(context.safetensors_segments());
  const uint64_t physical_payload_bytes = source_total_size.value_or(context.total_size());
  if (logical_total_size.has_value() && physical_payload_bytes > 0 && *logical_total_size > physical_payload_bytes) {
    return absl::FailedPreconditionError("tensor_index.json total size exceeds safetensors payload bytes");
  }
  const uint64_t effective_total_size = logical_total_size.value_or(physical_payload_bytes);
  if (progress_cb) {
    progress_cb(0, effective_total_size);
  }
  return compute_data_multihash_from_seekable_source(
      source,
      effective_total_size,
      static_cast<size_t>(kHashLeafChunkBytes),
      to_leaf_progress_callback(effective_total_size, std::move(progress_cb)));
}

} // namespace tensorcast::store::loader
