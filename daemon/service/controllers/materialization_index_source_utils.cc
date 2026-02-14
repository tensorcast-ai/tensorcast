// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_index_source_utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_hash.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon::materialization_index_source {

namespace {

class TargetLayoutGpuSource final : public store::loader::SeekableSource {
 public:
  TargetLayoutGpuSource(std::vector<TargetLayoutSpan> spans, uint64_t total_size, int device_id)
      : spans_(std::move(spans)), total_size_(total_size), device_id_(device_id) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_size_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto st = read_at(current_offset_, dst, max_bytes);
    if (!st.ok()) {
      return st;
    }
    current_offset_ += *st;
    return st;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_size_) {
      return static_cast<size_t>(0);
    }
    if (auto st = cuda::set_device(device_id_); !st.ok()) {
      return st;
    }

    size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
    auto* out = static_cast<uint8_t*>(dst);

    size_t idx = 0;
    for (; idx < spans_.size(); ++idx) {
      const auto& span = spans_[idx];
      if (offset < span.offset + span.length) {
        break;
      }
    }
    if (idx == spans_.size()) {
      return static_cast<size_t>(0);
    }

    while (remaining > 0 && idx < spans_.size()) {
      const auto& span = spans_[idx];
      const uint64_t local = offset - span.offset;
      const size_t avail = static_cast<size_t>(span.length - local);
      const size_t take = std::min(remaining, avail);
      const auto* src = static_cast<uint8_t*>(span.base_ptr.get()) + local;
      auto st = cuda::memcpy(out, src, take, cudaMemcpyDeviceToHost);
      if (!st.ok()) {
        return st;
      }
      if (auto sync = cuda::device_synchronize(); !sync.ok()) {
        return sync;
      }
      out += take;
      offset += take;
      remaining -= take;
      if (take == avail) {
        ++idx;
      }
    }
    return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
  }

 private:
  std::vector<TargetLayoutSpan> spans_;
  uint64_t total_size_{0};
  int device_id_{0};
  uint64_t current_offset_{0};
};

} // namespace

absl::Status ensure_tensor_index_present(const std::filesystem::path& artifact_dir) {
  std::error_code ec;
  const auto json_path = artifact_dir / "tensor_index.json";
  const auto cbor_path = artifact_dir / "tensor_index.cbor";
  const bool has_json = std::filesystem::exists(json_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to stat tensor_index.json at ", json_path.string()));
  }
  const bool has_cbor = std::filesystem::exists(cbor_path, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to stat tensor_index.cbor at ", cbor_path.string()));
  }
  if (!has_json && !has_cbor) {
    bool has_safetensors = false;
    std::filesystem::directory_iterator iter(artifact_dir, ec);
    if (ec) {
      return absl::ErrnoToStatus(
          ec.value(), absl::StrCat("Failed to enumerate artifact directory '", artifact_dir.string(), "'"));
    }
    for (const auto& entry : iter) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string name = entry.path().filename().string();
      if (name.ends_with(".safetensors")) {
        has_safetensors = true;
        break;
      }
    }
    if (!has_safetensors) {
      return absl::NotFoundError(
          absl::StrCat(
              "tensor index not found under ",
              artifact_dir.string(),
              " (expected tensor_index.json, tensor_index.cbor, or .safetensors files)"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> load_canonical_index_with_disk_fallback(
    store::StoreEngine& engine,
    std::string_view resolved_artifact_id,
    const std::optional<std::filesystem::path>& normalized_disk_path,
    int device_ordinal,
    bool gs_connected) {
  auto read_canonical_from_disk = [&]() -> absl::StatusOr<std::string> {
    if (!normalized_disk_path.has_value()) {
      return absl::FailedPreconditionError("disk source path required when Global Store is unavailable");
    }
    auto idx_status = ensure_tensor_index_present(*normalized_disk_path);
    if (!idx_status.ok()) {
      return idx_status;
    }
    auto local_or = store::loader::read_from_artifact_dir(*normalized_disk_path, device_ordinal);
    if (!local_or.ok()) {
      return local_or.status();
    }
    return local_or->canonical_index_json;
  };

  const bool prefer_disk_index = normalized_disk_path.has_value() && !gs_connected;
  absl::StatusOr<std::string> canonical_json_or =
      prefer_disk_index ? read_canonical_from_disk() : engine.get_canonical_index_by_id(resolved_artifact_id);
  if (!canonical_json_or.ok() && normalized_disk_path.has_value() && !prefer_disk_index) {
    auto disk_or = read_canonical_from_disk();
    if (disk_or.ok()) {
      canonical_json_or = std::move(disk_or);
    }
  }
  return canonical_json_or;
}

std::optional<std::string> parse_mi2_data_multihash(std::string_view artifact_id) {
  constexpr std::string_view kPrefix = "mi2:";
  if (!artifact_id.starts_with(kPrefix)) {
    return std::nullopt;
  }
  const std::string_view tail = artifact_id.substr(kPrefix.size());
  const size_t sep = tail.find(':');
  if (sep == std::string_view::npos) {
    return std::nullopt;
  }
  const std::string_view data_mh = tail.substr(sep + 1);
  if (data_mh.empty()) {
    return std::nullopt;
  }
  return std::string(data_mh);
}

absl::StatusOr<std::string> compute_target_layout_multihash(
    std::vector<TargetLayoutSpan> spans,
    uint64_t total_size,
    int device_id) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("target layout total_size is zero");
  }
  if (spans.empty()) {
    return absl::InvalidArgumentError("target layout spans are empty");
  }
  TargetLayoutGpuSource src(std::move(spans), total_size, device_id);
  return store::loader::compute_data_multihash_from_seekable_source(src, total_size);
}

absl::StatusOr<DescriptorMetadata> load_descriptor_metadata(const std::filesystem::path& artifact_dir) {
  DescriptorMetadata metadata;
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  std::error_code ec;
  if (!std::filesystem::exists(descriptor_path, ec)) {
    if (ec) {
      return absl::ErrnoToStatus(
          ec.value(), absl::StrCat("Failed to stat artifact_descriptor.json at ", descriptor_path.string()));
    }
    return metadata;
  }
  metadata.found = true;
  std::ifstream in(descriptor_path);
  if (!in.is_open()) {
    return absl::PermissionDeniedError(
        absl::StrCat("artifact_descriptor.json not readable at ", descriptor_path.string()));
  }
  try {
    nlohmann::json j;
    in >> j;
    const auto get_string = [&](const char* key) -> std::optional<std::string> {
      auto it = j.find(key);
      if (it == j.end() || it->is_null()) {
        return std::optional<std::string>{};
      }
      if (!it->is_string()) {
        throw std::invalid_argument(absl::StrCat(key, " must be a string"));
      }
      const std::string trimmed = std::string(absl::StripAsciiWhitespace(it->get<std::string>()));
      if (trimmed.empty()) {
        return std::optional<std::string>{};
      }
      return trimmed;
    };
    metadata.artifact_id = get_string("artifact_id");
    metadata.schema_version = get_string("schema_version");
    metadata.index_multihash = get_string("index_multihash");
    metadata.data_multihash = get_string("data_multihash");
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse artifact_descriptor.json at ", descriptor_path.string(), ": ", ex.what()));
  }
  return metadata;
}

absl::Status validate_descriptor_against_index(
    const DescriptorMetadata& descriptor,
    const store::loader::IndexInfo& index_info,
    bool verify_checksums) {
  if (!verify_checksums) {
    return absl::OkStatus();
  }
  if (!descriptor.found) {
    return absl::FailedPreconditionError("artifact_descriptor.json required when verify_checksums=true");
  }
  if (!descriptor.index_multihash.has_value() || descriptor.index_multihash->empty()) {
    return absl::FailedPreconditionError("index_multihash missing from artifact_descriptor.json");
  }
  auto computed_or = common::compute_index_multihash(
      std::optional<std::string>(index_info.canonical_index_json), /*index_key_hex=*/"");
  if (!computed_or.ok()) {
    return computed_or.status();
  }
  if (*descriptor.index_multihash != *computed_or) {
    return absl::FailedPreconditionError("index_multihash mismatch for disk artifact");
  }
  if (descriptor.artifact_id.has_value() && descriptor.data_multihash.has_value() &&
      !descriptor.data_multihash->empty()) {
    const std::string expected_artifact_id =
        absl::StrCat("mi2:", *descriptor.index_multihash, ":", *descriptor.data_multihash);
    if (*descriptor.artifact_id != expected_artifact_id) {
      return absl::FailedPreconditionError("artifact_id does not match descriptor multihashes");
    }
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::materialization_index_source
