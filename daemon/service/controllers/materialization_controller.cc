// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/util/time_util.h"
#include "gsl/pointers"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/common/selection_identity.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/view_utils.h"
#include "daemon/util/deadline_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/path_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using store::loader::ViewOp;
using store::loader::ViewSpec;

constexpr absl::Duration kTargetWriteTokenTtl = absl::Minutes(5);

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id);
using store::loading::MaterializationSource;
using store::loading::SourcePreference;

std::string mint_write_id(absl::BitGen& bitgen) {
  std::string raw;
  raw.resize(16);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen, 0u, 256u));
  }
  return absl::BytesToHexString(raw);
}

std::string compute_target_layout_hash(const v2::TargetLayout& layout) {
  std::string buffer;
  buffer.reserve(512);
  absl::StrAppend(
      &buffer,
      "lk:",
      static_cast<int>(layout.layout_kind()),
      "|ik:",
      static_cast<int>(layout.index_kind()),
      "|tk:",
      static_cast<int>(layout.tensor_spec_kind()),
      "|vid:",
      layout.view_id(),
      "|");
  buffer.append(layout.logical_layout_hash().data(), layout.logical_layout_hash().size());
  for (const auto& storage : layout.storages()) {
    absl::StrAppend(
        &buffer,
        "|s:",
        storage.storage_id(),
        ":",
        storage.device_id(),
        ":",
        storage.storage_length(),
        ":",
        storage.mapping_base_offset(),
        ":");
    if (storage.storage_source_case() == v2::StorageEntry::kVramRegionId) {
      absl::StrAppend(&buffer, "r:", storage.vram_region_id());
    } else if (storage.storage_source_case() == v2::StorageEntry::kCudaIpcHandle) {
      buffer.append("h:");
      buffer.append(storage.cuda_ipc_handle().data(), storage.cuda_ipc_handle().size());
    }
  }
  for (const auto& entry : layout.offsets()) {
    absl::StrAppend(
        &buffer,
        "|o:",
        entry.name(),
        ":",
        entry.storage_id(),
        ":",
        entry.storage_offset(),
        ":",
        entry.logical_length());
  }
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

void record_disk_path_denied() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_disk_path_denied_total");
    if (counter) {
      counter->Add(1);
    }
  } catch (...) {
  }
}

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
    return absl::NotFoundError(
        absl::StrCat("tensor index not found under ", artifact_dir.string(), " (expected tensor_index.json or .cbor)"));
  }
  return absl::OkStatus();
}

void record_disk_resolution_outcome(std::string_view outcome) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_disk_path_resolve_total");
    if (counter) {
      counter->Add(1, {{"outcome", std::string(outcome)}});
    }
  } catch (...) {
  }
}

void record_lease_create_failed() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
    ctr->Add(1.0);
  } catch (...) {
  }
}

void record_materialize_into_target_verification_enabled() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_materialize_into_target_verification_enabled_total");
    if (counter) {
      counter->Add(1);
    }
  } catch (...) {
  }
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

absl::StatusOr<std::optional<std::string>> resolve_artifact_binding(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    std::string_view artifact_id) {
  if (!client || !client->is_connected()) {
    return std::nullopt;
  }
  if (common::infer_artifact_id_kind(artifact_id) != common::ArtifactIdKind::kCgid) {
    return std::nullopt;
  }
  auto binding_or = client->get_artifact_binding(artifact_id);
  if (binding_or.ok()) {
    return binding_or->to_artifact_id;
  }
  if (absl::IsNotFound(binding_or.status())) {
    return std::nullopt;
  }
  return binding_or.status();
}

struct TargetLayoutSpan {
  gsl::not_null<void*> base_ptr;
  uint64_t offset{0};
  uint64_t length{0};
};

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

absl::Status register_session_and_refs(
    SessionsService& sessions,
    RefTracker& refs,
    const store::loading::ReplicaKey& replica_key,
    std::shared_ptr<tensorcast::common::ReadySignal<absl::Status>> ready_signal,
    const std::string& replica_uuid,
    int32_t pid,
    bool allow_pid_ref) {
  if (!replica_uuid.empty()) {
    auto st = sessions.put_with_verification(replica_uuid, replica_key, std::move(ready_signal));
    if (!st.ok()) {
      return st;
    }
  }
  if (!allow_pid_ref || pid <= 0) {
    return absl::OkStatus();
  }
  refs.add_ref(replica_key, pid);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<uint32_t>> build_export_chunks_for_replica(
    store::StoreEngine& engine,
    const store::loading::ReplicaKey& key,
    std::optional<uint64_t> size_bytes_override = std::nullopt) {
  uint64_t size_bytes = 0;
  if (size_bytes_override.has_value()) {
    size_bytes = *size_bytes_override;
  } else {
    auto size_or = engine.get_replica_size(key);
    if (!size_or.ok()) {
      return size_or.status();
    }
    size_bytes = *size_or;
  }
  const uint64_t chunk_bytes = static_cast<uint64_t>(engine.get_artifact_chunk_bytes());
  if (chunk_bytes == 0) {
    return absl::FailedPreconditionError("artifact_chunk_bytes is zero");
  }
  const uint64_t num_chunks = (size_bytes + chunk_bytes - 1) / chunk_bytes;
  if (num_chunks == 0) {
    return absl::InvalidArgumentError("replica size is zero");
  }
  if (num_chunks > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return absl::InvalidArgumentError("replica has too many chunks");
  }
  std::vector<uint32_t> chunks;
  chunks.reserve(static_cast<size_t>(num_chunks));
  for (uint32_t i = 0; i < static_cast<uint32_t>(num_chunks); ++i) {
    chunks.push_back(i);
  }
  return chunks;
}

struct DescriptorMetadata {
  bool found{false};
  std::optional<std::string> schema_version;
  std::optional<std::string> artifact_id;
  std::optional<std::string> index_multihash;
  std::optional<std::string> data_multihash;
};

struct CanonicalIndexEntry {
  uint64_t logical_offset{0};
  uint64_t logical_length{0};
  uint64_t storage_offset{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
};

struct CanonicalIndexTable {
  absl::flat_hash_map<std::string, CanonicalIndexEntry> entries;
  uint64_t logical_total_size{0};
};

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

absl::StatusOr<CanonicalIndexTable> parse_canonical_index(std::string_view index_json) {
  if (index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  CanonicalIndexTable table;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() != 6) {
      return absl::InvalidArgumentError("Invalid canonical index entry");
    }
    CanonicalIndexEntry entry;
    entry.logical_offset = arr[0].get<uint64_t>();
    entry.logical_length = arr[1].get<uint64_t>();
    entry.shape.reserve(arr[2].size());
    for (const auto& dim : arr[2]) {
      entry.shape.push_back(dim.get<int64_t>());
    }
    entry.stride.reserve(arr[3].size());
    for (const auto& dim : arr[3]) {
      entry.stride.push_back(dim.get<int64_t>());
    }
    entry.dtype = arr[4].get<std::string>();
    entry.storage_offset = arr[5].get<uint64_t>();
    table.logical_total_size =
        std::max<uint64_t>(table.logical_total_size, entry.logical_offset + entry.logical_length);
    table.entries.emplace(it.key(), std::move(entry));
  }
  return table;
}

struct TargetOffsetEntry {
  std::string name;
  std::string storage_id;
  uint64_t storage_offset{0};
  uint64_t logical_length{0};
};

absl::StatusOr<std::vector<TargetOffsetEntry>> resolve_target_offsets(const v2::TargetLayout& layout) {
  std::vector<TargetOffsetEntry> offsets;
  offsets.reserve(layout.offsets_size() + layout.aliases_size());
  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS) {
    for (const auto& entry : layout.offsets()) {
      TargetOffsetEntry resolved;
      resolved.name = entry.name();
      resolved.storage_id = entry.storage_id();
      resolved.storage_offset = entry.storage_offset();
      resolved.logical_length = entry.logical_length();
      offsets.push_back(std::move(resolved));
    }
    return offsets;
  }
  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    for (const auto& entry : layout.aliases()) {
      TargetOffsetEntry resolved;
      resolved.name = entry.name();
      resolved.storage_id = entry.storage_id();
      resolved.storage_offset = entry.storage_offset();
      resolved.logical_length = entry.logical_length();
      offsets.push_back(std::move(resolved));
    }
    return offsets;
  }
  return absl::InvalidArgumentError("Unsupported tensor_spec_kind for target layout");
}

void record_materialize_into_target(
    std::string_view result,
    std::string_view reason,
    v2::MaterializationSource source) {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_materialize_into_target_total");
    if (counter) {
      counter->Add(
          1,
          {{"result", std::string(result)}, {"reason", std::string(reason)}, {"source", static_cast<int64_t>(source)}});
    }
  } catch (...) {
  }
}

void record_materialize_into_target_verification_skipped() {
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateUInt64Counter("tc_store_materialize_into_target_verification_skipped_total");
    if (counter) {
      counter->Add(1);
    }
  } catch (...) {
  }
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

SourcePreference to_hint_preference(v2::SourcePreference preference) {
  switch (preference) {
    case v2::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P:
      return SourcePreference::kPreferP2P;
    case v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK:
      return SourcePreference::kPreferDisk;
    case v2::SourcePreference::SOURCE_PREFERENCE_AUTO:
    case v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED:
    default:
      return SourcePreference::kAuto;
  }
}

struct ResolvedSourcePolicy {
  v2::SourcePreference preference{v2::SourcePreference::SOURCE_PREFERENCE_AUTO};
  bool allow_p2p{true};
  bool allow_disk{true};
};

ResolvedSourcePolicy resolve_source_policy(const v2::SourcePolicy* policy, v2::SourcePreference legacy_preference) {
  ResolvedSourcePolicy resolved;
  resolved.preference = legacy_preference;
  if (policy != nullptr) {
    if (policy->preference() != v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED) {
      resolved.preference = policy->preference();
    }
    if (policy->has_allow_p2p()) {
      resolved.allow_p2p = policy->allow_p2p();
    }
    if (policy->has_allow_disk()) {
      resolved.allow_disk = policy->allow_disk();
    }
  }
  if (resolved.preference == v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED) {
    resolved.preference = v2::SourcePreference::SOURCE_PREFERENCE_AUTO;
  }
  return resolved;
}

absl::Status validate_source_policy(const ResolvedSourcePolicy& policy) {
  if (!policy.allow_p2p && policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P) {
    return absl::InvalidArgumentError("source_policy disallows P2P but preference=PREFER_P2P was requested");
  }
  if (!policy.allow_disk && policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK) {
    return absl::InvalidArgumentError("source_policy disallows disk but preference=PREFER_DISK was requested");
  }
  return absl::OkStatus();
}

v2::MaterializationSource to_proto_source(MaterializationSource source) {
  switch (source) {
    case MaterializationSource::kDisk:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_DISK;
    case MaterializationSource::kP2P:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_P2P;
    case MaterializationSource::kLocalReplica:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA;
    case MaterializationSource::kUnspecified:
    default:
      return v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED;
  }
}

uint64_t compute_generation_from_index(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return 0;
  }
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(
          reinterpret_cast<const uint8_t*>(canonical_index_json.data()), canonical_index_json.size()));
  uint64_t value = 0;
  const size_t limit = std::min<size_t>(8, digest.size());
  for (size_t i = 0; i < limit; ++i) {
    value = (value << 8) | static_cast<uint64_t>(digest[i]);
  }
  return value;
}

absl::StatusOr<ViewSpec> convert_view_spec(const tensorcast::common::v1::ViewSpec& proto) {
  ViewSpec spec;
  for (const auto& [tensor_name, ops_proto] : proto.tensors()) {
    store::loader::TensorViewOps ops;
    for (const auto& op_proto : ops_proto.ops()) {
      switch (op_proto.kind_case()) {
        case tensorcast::common::v1::Op::kNarrow: {
          const auto& narrow = op_proto.narrow();
          store::loader::NarrowOp op{
              .dim = static_cast<int32_t>(narrow.dim()),
              .start = narrow.start(),
              .length = narrow.length(),
          };
          ops.ops.push_back(ViewOp::Narrow(op));
          break;
        }
        case tensorcast::common::v1::Op::kTranspose: {
          const auto& transpose = op_proto.transpose();
          store::loader::TransposeOp op{
              .dim0 = static_cast<int32_t>(transpose.dim0()),
              .dim1 = static_cast<int32_t>(transpose.dim1()),
          };
          ops.ops.push_back(ViewOp::Transpose(op));
          break;
        }
        case tensorcast::common::v1::Op::KIND_NOT_SET:
          return absl::InvalidArgumentError("view op kind not set");
      }
    }
    spec.tensors.emplace(tensor_name, std::move(ops));
  }
  return spec;
}

absl::StatusOr<std::string> serialize_deterministic_proto(const google::protobuf::Message& message) {
  const size_t size = message.ByteSizeLong();
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::OutOfRangeError("proto message too large for deterministic serialization");
  }
  std::string buffer;
  buffer.resize(size);
  google::protobuf::io::ArrayOutputStream stream(buffer.data(), static_cast<int>(size));
  google::protobuf::io::CodedOutputStream coded(&stream);
  coded.SetSerializationDeterministic(true);
  if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
    return absl::InternalError("deterministic proto serialization failed");
  }
  return buffer;
}

absl::StatusOr<std::string> compute_view_id_from_spec(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json) {
  auto index_mh_or = common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  auto proto_bytes_or = serialize_deterministic_proto(view_spec);
  if (!proto_bytes_or.ok()) {
    return proto_bytes_or.status();
  }
  std::vector<uint8_t> buffer;
  buffer.reserve(proto_bytes_or->size() + index_mh_or->size());
  buffer.insert(buffer.end(), proto_bytes_or->begin(), proto_bytes_or->end());
  buffer.insert(buffer.end(), index_mh_or->begin(), index_mh_or->end());
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer));
  return common::multibase_multihash_sha256(digest);
}

tensorcast::common::v1::ViewSpec build_view_spec_proto(const ViewSpec& spec) {
  tensorcast::common::v1::ViewSpec proto;
  auto* tensors = proto.mutable_tensors();
  for (const auto& [tensor_name, ops] : spec.tensors) {
    auto& container = (*tensors)[tensor_name];
    for (const auto& op : ops.ops) {
      auto* op_proto = container.add_ops();
      switch (op.kind) {
        case ViewOp::Kind::kNarrow:
          op_proto->mutable_narrow()->set_dim(static_cast<uint32_t>(op.narrow.dim));
          op_proto->mutable_narrow()->set_start(op.narrow.start);
          op_proto->mutable_narrow()->set_length(op.narrow.length);
          break;
        case ViewOp::Kind::kTranspose:
          op_proto->mutable_transpose()->set_dim0(static_cast<uint32_t>(op.transpose.dim0));
          op_proto->mutable_transpose()->set_dim1(static_cast<uint32_t>(op.transpose.dim1));
          break;
      }
    }
  }
  return proto;
}

bool spec_includes_transpose(const ViewSpec& spec) {
  for (const auto& [_, ops] : spec.tensors) {
    for (const auto& op : ops.ops) {
      if (op.kind == ViewOp::Kind::kTranspose) {
        return true;
      }
    }
  }
  return false;
}

store::loading::TransformPlacement resolve_placement(
    const v2::MaterializeReplicaRequest& req,
    const std::optional<ViewSpec>& spec) {
  switch (req.placement()) {
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_SERVER:
      return store::loading::TransformPlacement::kServer;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_CLIENT:
      return store::loading::TransformPlacement::kClient;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_UNSPECIFIED:
    default:
      break;
  }
  if (spec.has_value() && spec_includes_transpose(*spec)) {
    return store::loading::TransformPlacement::kClient;
  }
  return store::loading::TransformPlacement::kServer;
}

store::loading::TransformPlacement resolve_placement(
    const v2::MaterializeIntoTargetRequest& req,
    const std::optional<ViewSpec>& spec) {
  switch (req.placement()) {
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_SERVER:
      return store::loading::TransformPlacement::kServer;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_CLIENT:
      return store::loading::TransformPlacement::kClient;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_UNSPECIFIED:
    default:
      break;
  }
  if (spec.has_value() && spec_includes_transpose(*spec)) {
    return store::loading::TransformPlacement::kClient;
  }
  return store::loading::TransformPlacement::kServer;
}

struct DescriptorBuildResult {
  std::vector<v2::TensorPayloadDescriptor> descriptors;
  std::vector<std::string> included_names;
};

absl::StatusOr<DescriptorBuildResult> build_descriptors_from_view_plan(
    const store::loader::ViewPlan& plan,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid) {
  absl::flat_hash_set<std::string> filter;
  std::vector<std::string> requested_order;
  for (const auto& name : tensor_names) {
    filter.insert(name);
    requested_order.push_back(name);
  }
  const bool filter_enabled = !filter.empty();

  struct OrderedDescriptor {
    uint64_t offset{0};
    v2::TensorPayloadDescriptor desc;
  };

  std::vector<OrderedDescriptor> built;
  built.reserve(plan.transform.tensors.size());

  absl::flat_hash_set<std::string> seen;
  for (const auto& tensor : plan.transform.tensors) {
    if (filter_enabled && !filter.contains(tensor.tensor_name)) {
      continue;
    }
    OrderedDescriptor out;
    out.offset = tensor.dst_offset;
    out.desc.set_name(tensor.tensor_name);
    out.desc.set_buffer_offset(tensor.dst_offset);
    out.desc.set_storage_offset(tensor.storage_offset_elements);
    out.desc.set_dtype(tensor.dtype);
    if (!device_uuid.empty()) {
      out.desc.set_device_uuid(std::string(device_uuid));
    }
    uint64_t elements = 1;
    for (const auto dim : tensor.view_shape) {
      out.desc.add_shape(dim);
      elements *= static_cast<uint64_t>(dim);
    }
    for (const auto stride : tensor.view_stride) {
      out.desc.add_stride(stride);
    }
    if (tensor.element_size_bytes == 0) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid element_size_bytes for tensor ", tensor.tensor_name));
    }
    out.desc.set_byte_length(elements * tensor.element_size_bytes);
    built.push_back(std::move(out));
    seen.insert(tensor.tensor_name);
  }

  if (filter_enabled) {
    for (const auto& name : requested_order) {
      if (!seen.contains(name)) {
        return absl::NotFoundError(absl::StrCat("requested tensor '", name, "' missing from view transform"));
      }
    }
  }

  std::sort(built.begin(), built.end(), [](const OrderedDescriptor& a, const OrderedDescriptor& b) {
    return a.offset < b.offset;
  });

  DescriptorBuildResult result;
  result.descriptors.reserve(built.size());
  result.included_names.reserve(built.size());
  for (auto& entry : built) {
    result.included_names.push_back(entry.desc.name());
    result.descriptors.push_back(std::move(entry.desc));
  }
  return result;
}

absl::StatusOr<DescriptorBuildResult> build_descriptors_from_index(
    std::string_view canonical_index_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical index JSON is required for descriptor building");
  }

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON for descriptors: ", e.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  absl::flat_hash_set<std::string> filter;
  std::vector<std::string> requested_order;
  for (const auto& name : tensor_names) {
    filter.insert(name);
    requested_order.push_back(name);
  }
  const bool filter_enabled = !filter.empty();

  struct OrderedDescriptor {
    uint64_t offset{0};
    v2::TensorPayloadDescriptor desc;
  };

  std::vector<OrderedDescriptor> built;
  built.reserve(j.size());

  auto parse_entry = [&](const std::string& name, const nlohmann::json& meta) -> absl::StatusOr<OrderedDescriptor> {
    if (!meta.is_array() || meta.size() < 6) {
      return absl::InvalidArgumentError(
          absl::StrCat("canonical index entry for ", name, " must be array [offset,size,shape,stride,dtype,storage]"));
    }
    OrderedDescriptor out;
    out.offset = meta[0].get<uint64_t>();
    const uint64_t size_bytes = meta[1].get<uint64_t>();
    out.desc.set_name(name);
    out.desc.set_buffer_offset(out.offset);
    out.desc.set_byte_length(size_bytes);
    out.desc.set_storage_offset(meta[5].get<uint64_t>());
    out.desc.set_dtype(meta[4].get<std::string>());
    if (!device_uuid.empty()) {
      out.desc.set_device_uuid(std::string(device_uuid));
    }
    for (const auto& dim : meta[2]) {
      out.desc.add_shape(dim.get<int64_t>());
    }
    for (const auto& stride : meta[3]) {
      out.desc.add_stride(stride.get<int64_t>());
    }
    return out;
  };

  if (filter_enabled) {
    absl::flat_hash_set<std::string> seen;
    for (const auto& name : requested_order) {
      const auto it = j.find(name);
      if (it == j.end()) {
        return absl::NotFoundError(absl::StrCat("requested tensor '", name, "' missing from canonical index"));
      }
      auto parsed_or = parse_entry(name, *it);
      if (!parsed_or.ok()) {
        return parsed_or.status();
      }
      built.push_back(std::move(*parsed_or));
      seen.insert(name);
    }
  } else {
    for (auto it = j.begin(); it != j.end(); ++it) {
      auto parsed_or = parse_entry(it.key(), it.value());
      if (!parsed_or.ok()) {
        return parsed_or.status();
      }
      built.push_back(std::move(*parsed_or));
    }
  }

  std::sort(built.begin(), built.end(), [](const OrderedDescriptor& a, const OrderedDescriptor& b) {
    return a.offset < b.offset;
  });

  DescriptorBuildResult result;
  result.descriptors.reserve(built.size());
  result.included_names.reserve(built.size());
  for (auto& entry : built) {
    result.included_names.push_back(entry.desc.name());
    result.descriptors.push_back(std::move(entry.desc));
  }
  return result;
}

absl::StatusOr<std::string> resolve_layout_json(
    const v2::MaterializeReplicaResponse& v1_resp,
    const v2::MaterializeReplicaRequest& v2_req,
    store::StoreEngine& engine) {
  if (!v1_resp.view_index_json().empty()) {
    return v1_resp.view_index_json();
  }
  if (!v2_req.artifact_id().empty()) {
    return engine.get_canonical_index_by_id(v2_req.artifact_id());
  }
  if (!v1_resp.artifact_id().empty()) {
    return engine.get_canonical_index_by_id(v1_resp.artifact_id());
  }
  if (!v1_resp.disk_path().empty()) {
    auto local_or = store::loader::read_from_artifact_dir(v1_resp.disk_path(), /*target_device_id=*/0);
    if (!local_or.ok()) {
      return local_or.status();
    }
    return local_or->canonical_index_json;
  }
  if (v2_req.has_disk_fallback() && !v2_req.disk_fallback().disk_path().empty()) {
    auto local_or = store::loader::read_from_artifact_dir(v2_req.disk_fallback().disk_path(), /*target_device_id=*/0);
    if (!local_or.ok()) {
      return local_or.status();
    }
    return local_or->canonical_index_json;
  }
  return absl::NotFoundError("canonical index JSON unavailable for materialization response");
}

absl::StatusOr<std::string> resolve_layout_json_by_key(
    const v2::MaterializeByKeyResponse& v1_resp,
    store::StoreEngine& engine) {
  if (!v1_resp.artifact_id().empty()) {
    return engine.get_canonical_index_by_id(v1_resp.artifact_id());
  }
  return absl::NotFoundError("canonical index JSON unavailable for key materialization");
}

template <typename ResponseT>
absl::Status populate_materialize_payloads(
    ResponseT& resp,
    const std::string& layout_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    const std::string& device_uuid,
    const std::string& view_subset_hash,
    bool wait_for_completion,
    const std::string& replica_uuid,
    const std::string* ticket_device_uuid,
    const std::optional<store::loader::ViewPlan>& view_plan,
    bool prefer_view_plan,
    bool fill_view_index_bytes) {
  const uint64_t generation = compute_generation_from_index(layout_json);
  resp.set_canonical_index_bytes(layout_json);
  resp.set_generation(generation);
  if (fill_view_index_bytes && resp.view_index_bytes().empty()) {
    resp.set_view_index_bytes(layout_json);
  }

  std::optional<DescriptorBuildResult> desc_result;
  if (prefer_view_plan && view_plan.has_value() && !view_plan->is_identity) {
    auto desc_or = build_descriptors_from_view_plan(*view_plan, tensor_names, device_uuid);
    if (!desc_or.ok()) {
      return desc_or.status();
    }
    desc_result = std::move(*desc_or);
  }
  if (!desc_result.has_value()) {
    auto desc_or = build_descriptors_from_index(layout_json, tensor_names, device_uuid);
    if (!desc_or.ok()) {
      return desc_or.status();
    }
    desc_result = std::move(*desc_or);
  }

  for (auto& desc : desc_result->descriptors) {
    *resp.add_payloads() = std::move(desc);
  }
  if (!view_subset_hash.empty() || !desc_result->included_names.empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!view_subset_hash.empty()) {
      subset->set_subset_hash(view_subset_hash);
    }
    for (const auto& name : desc_result->included_names) {
      subset->add_tensor_names(name);
    }
  }
  if (!wait_for_completion && !replica_uuid.empty()) {
    auto* ticket = resp.mutable_ticket();
    ticket->set_replica_uuid(replica_uuid);
    ticket->set_status(resp.status());
    if (ticket_device_uuid != nullptr && !ticket_device_uuid->empty()) {
      ticket->set_device_uuid(*ticket_device_uuid);
    }
    *ticket->mutable_created_at() = google::protobuf::util::TimeUtil::GetCurrentTime();
  }
  return absl::OkStatus();
}

} // namespace

MaterializationController::MaterializationController(Dep d)
    : d_(std::move(d)),
      capability_tokens_(d_.capability_tokens),
      target_write_registry_(TargetWriteRegistry::Options{.ttl = kTargetWriteTokenTtl}) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

TargetWriteRegistry::Record MaterializationController::insert_target_write_for_testing(
    TargetWriteRegistry::Record record) {
  return target_write_registry_.insert(std::move(record));
}

grpc::Status MaterializationController::materialize_replica(
    RpcContext& rctx,
    const v2::MaterializeReplicaRequest& req,
    v2::MaterializeReplicaResponse& resp) {
  auto& span = rctx.span();
  const auto policy = resolve_source_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  ResolvedSourcePolicy effective_policy = policy;
  if (storage_path_.empty()) {
    effective_policy.allow_disk = false;
  }
  const bool prefer_disk = effective_policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK;
  const bool prefer_p2p = effective_policy.preference == v2::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P;
  bool verify_checksums = true;
  std::string disk_path;
  const bool has_disk_fallback = req.has_disk_fallback() && !req.disk_fallback().disk_path().empty();
  const bool has_legacy_disk = req.has_disk_path() && !req.disk_path().empty();
  if (has_disk_fallback) {
    disk_path = req.disk_fallback().disk_path();
    verify_checksums = req.disk_fallback().verify_checksums();
  } else if (has_legacy_disk) {
    disk_path = req.disk_path();
    verify_checksums = req.has_verify_checksums() ? req.verify_checksums() : true;
  }

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);

  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.device.uuid", req.device_uuid());
  }
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.size_bytes()));
  span->SetAttribute("tc.store.preference", static_cast<int64_t>(effective_policy.preference));
  span->SetAttribute("tc.store.allow_p2p", effective_policy.allow_p2p);
  span->SetAttribute("tc.store.allow_disk", effective_policy.allow_disk);

  using v2::MaterializeReplicaStatus;
  if (d_.shutdown_signal.is_shutting_down()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  absl::Status policy_status = validate_source_policy(effective_policy);
  if (!policy_status.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(policy_status);
  }

  const bool request_has_artifact = req.has_artifact_id() && !req.artifact_id().empty();
  const bool request_has_disk = has_disk_fallback || has_legacy_disk;
  if (!request_has_artifact && !request_has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id or disk_fallback.disk_path is required"};
  }
  if (request_has_disk && !effective_policy.allow_disk) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::INVALID_ARGUMENT, "disk fallback requested but source_policy disallows disk"};
  }

  const auto dev = d_.devices.From(req.target_device_type(), req.device_uuid(), std::nullopt);
  const bool cpu_target = dev.type == DeviceType::CPU;
  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  const bool no_lease = req.lease_mode() == v2::LeaseMode::LEASE_MODE_NO_LEASE;
  const int32_t effective_pid = (loopback_peer && !no_lease) ? req.pid() : 0;
  if (req.wait_for_completion() && !loopback_peer) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::PERMISSION_DENIED, "wait_for_completion materialization is local-only (loopback/UDS)"};
  }
  if (no_lease && req.wait_for_completion()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::INVALID_ARGUMENT, "lease_mode=NO_LEASE requires wait_for_completion=false"};
  }
  if (cpu_target) {
    if (!loopback_peer) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::PERMISSION_DENIED, "CPU shared-memory materialization is local-only"};
    }
    if (effective_pid <= 0) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::INVALID_ARGUMENT, "pid is required for CPU handle leases"};
    }
    if (!d_.cpu_shared_memory_enabled) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "cpu_shared_memory is disabled"};
    }
    if (d_.handle_leases == nullptr) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "local handle plane is disabled (no handle leases)"};
    }
  }

  std::optional<std::filesystem::path> normalized_disk_path;
  if (request_has_disk) {
    auto normalized_or = normalize_disk_path(disk_path, storage_path_);
    if (!normalized_or.ok()) {
      record_disk_path_denied();
      LOG(WARNING) << "disk_path rejected: " << disk_path << ": " << normalized_or.status();
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(normalized_or.status());
    }
    const auto& normalized = *normalized_or;
    normalized_disk_path = normalized;
    resp.set_disk_path(normalized.string());
    if (rctx.allow_high_card_attrs()) {
      span->SetAttribute("tc.disk.path", normalized.string());
    }
  }

  DescriptorMetadata descriptor_meta;
  std::optional<store::loader::IndexInfo> disk_index;
  std::optional<store::loading::DiskMetadata> disk_metadata;
  if (normalized_disk_path.has_value()) {
    auto descriptor_or = load_descriptor_metadata(*normalized_disk_path);
    if (!descriptor_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(descriptor_or.status());
    }
    descriptor_meta = *descriptor_or;
    if (verify_checksums && !descriptor_meta.found) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "artifact_descriptor.json required when verify_checksums=true"};
    }
    if (verify_checksums) {
      auto index_or = store::loader::read_from_artifact_dir(*normalized_disk_path, dev.ordinal);
      if (!index_or.ok()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(index_or.status());
      }
      auto validation_status = validate_descriptor_against_index(descriptor_meta, *index_or, /*verify_checksums=*/true);
      if (!validation_status.ok()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(validation_status);
      }
      disk_index = std::move(*index_or);
    } else if (prefer_disk) {
      auto idx_status = ensure_tensor_index_present(*normalized_disk_path);
      if (!idx_status.ok()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(idx_status);
      }
    }

    store::loading::DiskMetadata metadata;
    metadata.descriptor_present = descriptor_meta.found;
    metadata.schema_version = descriptor_meta.schema_version;
    metadata.index_multihash = descriptor_meta.index_multihash;
    metadata.data_multihash = descriptor_meta.data_multihash;
    if (disk_index.has_value()) {
      metadata.canonical_index_json = disk_index->canonical_index_json;
      if (!disk_index->index_multihash.empty()) {
        metadata.index_multihash = disk_index->index_multihash;
      }
      if (disk_index->total_size_bytes > 0) {
        metadata.logical_total_size = disk_index->total_size_bytes;
      }
      metadata.is_safetensors = disk_index->is_safetensors;
    }
    disk_metadata = std::move(metadata);
  }

  std::optional<std::string> artifact_id =
      request_has_artifact ? std::optional<std::string>(req.artifact_id()) : std::nullopt;
  if (!artifact_id.has_value() && descriptor_meta.artifact_id.has_value()) {
    artifact_id = descriptor_meta.artifact_id;
  }
  if (artifact_id.has_value() && descriptor_meta.artifact_id.has_value() &&
      *artifact_id != *descriptor_meta.artifact_id) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::FAILED_PRECONDITION, "artifact_id mismatch between request and artifact_descriptor.json"};
  }
  if (!artifact_id.has_value() && normalized_disk_path.has_value()) {
    artifact_id = normalized_disk_path->string();
  }
  const bool has_artifact = artifact_id.has_value() && !artifact_id->empty();
  const bool has_disk = normalized_disk_path.has_value();
  std::string resolved_artifact_id = has_artifact ? *artifact_id : std::string();
  std::optional<std::string> bound_artifact_id;
  std::optional<std::string> fallback_artifact_id;
  if (has_artifact) {
    auto binding_or = resolve_artifact_binding(d_.global_store_client, resolved_artifact_id);
    if (!binding_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(binding_or.status());
    }
    if (binding_or->has_value()) {
      const auto& bound = binding_or->value();
      fallback_artifact_id = resolved_artifact_id;
      bound_artifact_id = bound;
      resolved_artifact_id = bound;
    }
  }
  if (has_artifact) {
    span->SetAttribute("tc.artifact.id", resolved_artifact_id);
    resp.set_artifact_id(resolved_artifact_id);
    if (bound_artifact_id.has_value()) {
      span->SetAttribute("tc.artifact.bound", *bound_artifact_id);
    }
  }
  if (prefer_p2p && !has_artifact) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required when preference=PREFER_P2P"};
  }

  // View identity handling
  std::optional<ViewSpec> view_spec;
  std::optional<store::loader::ViewPlan> view_plan;
  std::optional<std::string> canonical_index_json;
  std::optional<std::string> request_view_id;
  const std::string& index_source_artifact_id =
      fallback_artifact_id.has_value() ? *fallback_artifact_id : resolved_artifact_id;

  switch (req.view_identity_case()) {
    case v2::MaterializeReplicaRequest::kView: {
      if (!has_artifact) {
        return {StatusCode::INVALID_ARGUMENT, "view spec requires artifact_id for canonical planning"};
      }
      auto spec_or = convert_view_spec(req.view());
      if (!spec_or.ok()) {
        return to_grpc_status(spec_or.status());
      }
      view_spec = std::move(*spec_or);
      auto index_or = d_.engine.get_canonical_index_by_id(index_source_artifact_id);
      if (!index_or.ok()) {
        return to_grpc_status(index_or.status());
      }
      canonical_index_json = std::move(*index_or);
      auto plan_or = store::StoreEngine::compute_view_plan(*canonical_index_json, *view_spec);
      if (!plan_or.ok()) {
        return to_grpc_status(plan_or.status());
      }
      if (!plan_or->is_identity) {
        view_plan = *plan_or;
        auto view_id_or = compute_view_id_from_spec(req.view(), *canonical_index_json);
        if (!view_id_or.ok()) {
          return to_grpc_status(view_id_or.status());
        }
        request_view_id = std::move(*view_id_or);
      } else {
        // Identity view collapses to canonical path
        view_spec.reset();
        view_plan.reset();
        canonical_index_json.reset();
      }
      break;
    }
    case v2::MaterializeReplicaRequest::kViewId: {
      if (!req.view_id().empty()) {
        if (!has_artifact) {
          return {StatusCode::INVALID_ARGUMENT, "view_id requires artifact_id for routing"};
        }
        request_view_id = req.view_id();
      }
      break;
    }
    case v2::MaterializeReplicaRequest::VIEW_IDENTITY_NOT_SET:
      break;
  }
  if (request_view_id.has_value()) {
    span->SetAttribute("tc.view.id", *request_view_id);
  }

  auto finalize_response = [&]() -> grpc::Status {
    if (no_lease) {
      resp.clear_mem_handle();
    }
    if (!resp.view_index_json().empty() && resp.view_index_bytes().empty()) {
      resp.set_view_index_bytes(resp.view_index_json());
    }

    auto layout_or = resolve_layout_json(resp, req, d_.engine);
    if (!layout_or.ok()) {
      return to_grpc_status(layout_or.status());
    }
    const bool prefer_view_plan =
        req.view_identity_case() == v2::MaterializeReplicaRequest::kView && resp.view_index_json().empty();
    const std::string* ticket_device_uuid = req.device_uuid().empty() ? nullptr : &req.device_uuid();
    absl::Status payload_status = populate_materialize_payloads(
        resp,
        *layout_or,
        req.tensor_names(),
        req.device_uuid(),
        req.view_subset_hash(),
        req.wait_for_completion(),
        req.replica_uuid(),
        ticket_device_uuid,
        view_plan,
        prefer_view_plan,
        /*fill_view_index_bytes=*/false);
    if (!payload_status.ok()) {
      return to_grpc_status(payload_status);
    }
    rctx.mark_success();
    return Status::OK;
  };

  // Artifact LIP fast path: try cross-device consumption
  const bool view_requested = view_spec.has_value() || request_view_id.has_value();
  std::optional<store::loading::ReplicaKey> lip_replica_key;
  if (has_artifact && !view_requested && dev.type == DeviceType::GPU) {
    absl::Status session_status = absl::OkStatus();
    auto satisfied = d_.lip.try_satisfy_from_lip(
        resolved_artifact_id,
        dev.ordinal,
        [&](const store::loading::ReplicaKey& rkey) {
          lip_replica_key = rkey;
          session_status = register_session_and_refs(
              d_.sessions, d_.refs, rkey, nullptr, req.replica_uuid(), effective_pid, loopback_peer);
        },
        resp.mutable_mem_handle());
    if (!session_status.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(session_status);
    }
    if (!satisfied.ok()) {
      // Same-device denial should fall back to the engine path just like MaterializeByKey.
      if (!absl::IsFailedPrecondition(satisfied.status())) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(satisfied.status());
      }
    } else if (*satisfied) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
      resp.set_source(v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA);
      span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
      bool lease_created = false;
      if (d_.handle_leases != nullptr && effective_pid > 0 && lip_replica_key.has_value()) {
        auto token_or = d_.handle_leases->mint_cuda_ipc_lease(*lip_replica_key, effective_pid);
        if (token_or.ok()) {
          resp.mutable_mem_handle()->set_lease_token(*token_or);
          lease_created = true;
        } else {
          LOG(WARNING) << "mint_cuda_ipc_lease failed (LIP path): key=" << *lip_replica_key << " pid=" << effective_pid
                       << ": " << token_or.status();
          record_lease_create_failed();
        }
      }
      if (!lease_created && d_.lifecycle != nullptr && effective_pid > 0 && lip_replica_key.has_value()) {
        auto lid_or = d_.lifecycle->create_use_lease(*lip_replica_key, effective_pid);
        if (!lid_or.ok()) {
          LOG(WARNING) << "create_use_lease failed (LIP path): key=" << *lip_replica_key << " pid=" << effective_pid
                       << ": " << lid_or.status();
          record_lease_create_failed();
        }
      }
      return finalize_response();
    }
  }

  // Engine-backed materialization
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.verify = verify_checksums ? store::loading::MaterializeHints::Verify::CHECKSUM
                                  : store::loading::MaterializeHints::Verify::NONE;
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  if (has_artifact)
    hints.artifact_id = resolved_artifact_id;
  if (has_disk)
    hints.disk_path = normalized_disk_path->string();
  if (disk_metadata.has_value()) {
    hints.disk_metadata = std::move(*disk_metadata);
  }
  if (view_spec.has_value() || request_view_id.has_value()) {
    store::loading::VariantIdentity variant;
    if (has_artifact) {
      variant.canonical_artifact_id = resolved_artifact_id;
    }
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    if (canonical_index_json.has_value()) {
      variant.canonical_index_json = canonical_index_json;
    }
    if (view_plan.has_value()) {
      variant.cached_plan = view_plan;
    }
    if (request_view_id.has_value()) {
      variant.view_id = request_view_id;
    }
    variant.placement = resolve_placement(req, view_spec);
    hints.variant = std::move(variant);
  }
  const auto mode = (has_disk && !has_artifact && !prefer_disk) ? store::StoreEngine::MaterializeMode::LOAD_ONLY
                                                                : store::StoreEngine::MaterializeMode::AUTO;

  auto result = d_.engine.materialize_replica(dev, mode, hints);
  if (!result.ok() && view_requested && fallback_artifact_id.has_value() && absl::IsNotFound(result.status())) {
    bool allow_reuse = false;
    if (d_.post_seal_policy.reuse_views_if_safe) {
      if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
      }
      auto safe_or =
          check_post_seal_view_reuse_safe(*d_.global_store_client, *fallback_artifact_id, resolved_artifact_id);
      if (!safe_or.ok()) {
        LOG(WARNING) << "post-seal view reuse check failed for assembly=" << *fallback_artifact_id
                     << " mi2=" << resolved_artifact_id << ": " << safe_or.status();
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(safe_or.status());
      }
      allow_reuse = *safe_or;
      if (!allow_reuse) {
        LOG(WARNING) << "post-seal view reuse disabled: proof commitments mismatch for assembly="
                     << *fallback_artifact_id << " mi2=" << resolved_artifact_id;
      }
    }

    if (allow_reuse) {
      hints.artifact_id = *fallback_artifact_id;
      if (hints.variant.has_value()) {
        hints.variant->canonical_artifact_id = *fallback_artifact_id;
      }
      auto fallback_or = d_.engine.materialize_replica(dev, mode, hints);
      if (fallback_or.ok()) {
        result = std::move(fallback_or);
      } else {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(fallback_or.status());
      }
    }
  }
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  resp.set_source(to_proto_source(handle.source));
  span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
  {
    const absl::Status session_status = register_session_and_refs(
        d_.sessions,
        d_.refs,
        handle.replica_key,
        handle.ready_signal,
        req.replica_uuid(),
        effective_pid,
        loopback_peer);
    if (!session_status.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(session_status);
    }
  }
  if (has_disk)
    resp.set_disk_path(normalized_disk_path->string());
  if (cpu_target) {
    if (!handle.cpu_memfd_region.has_value()) {
      LOG(WARNING) << "MaterializationController: cpu_target but engine handle missing cpu_memfd_region for key="
                   << handle.replica_key << " cpu_state=" << static_cast<int>(handle.cpu_state)
                   << " gpu_state=" << static_cast<int>(handle.gpu_state);
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "CPU memfd handle unavailable for replica"};
    }
    const auto& region = *handle.cpu_memfd_region;
    auto chunks_or = build_export_chunks_for_replica(d_.engine, handle.replica_key);
    if (!chunks_or.ok()) {
      chunks_or = build_export_chunks_for_replica(d_.engine, handle.replica_key, region.size_bytes);
    }
    if (!chunks_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(chunks_or.status());
    }
    HandleLeaseRegistry::CpuMemfdDescriptor memfd_desc{
        .fd = region.fd,
        .size_bytes = region.size_bytes,
        .offset_bytes = region.offset_bytes,
    };
    auto token_or = d_.handle_leases->mint_cpu_memfd_lease(handle.replica_key, effective_pid, memfd_desc, *chunks_or);
    if (!token_or.ok()) {
      // Best-effort rollback: drop ref and unload.
      if (!req.replica_uuid().empty()) {
        (void)d_.sessions.erase(req.replica_uuid());
      }
      if (effective_pid > 0) {
        d_.refs.drop_ref(handle.replica_key, effective_pid);
      }
      (void)d_.engine.unload_replica(handle.replica_key);
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(token_or.status());
    }
    auto* cpu = resp.mutable_mem_handle()->mutable_cpu_memfd();
    cpu->set_size_bytes(region.size_bytes);
    cpu->set_offset_bytes(region.offset_bytes);
    resp.mutable_mem_handle()->set_lease_token(*token_or);
  } else {
    if (handle.cuda_ipc_handle.is_valid()) {
      auto handle_view = handle.cuda_ipc_handle.as_string_view();
      resp.mutable_mem_handle()->set_cuda_ipc_handle(handle_view.data(), handle_view.size());
    }
    bool lease_created = false;
    if (d_.handle_leases != nullptr && effective_pid > 0) {
      auto token_or = d_.handle_leases->mint_cuda_ipc_lease(handle.replica_key, effective_pid);
      if (token_or.ok()) {
        resp.mutable_mem_handle()->set_lease_token(*token_or);
        lease_created = true;
      } else {
        LOG(WARNING) << "mint_cuda_ipc_lease failed (engine path): key=" << handle.replica_key
                     << " pid=" << effective_pid << ": " << token_or.status();
        record_lease_create_failed();
      }
    }
    if (!lease_created && d_.lifecycle != nullptr && effective_pid > 0) {
      auto lid_or = d_.lifecycle->create_use_lease(handle.replica_key, effective_pid);
      if (!lid_or.ok()) {
        LOG(WARNING) << "create_use_lease failed (engine path): key=" << handle.replica_key << " pid=" << effective_pid
                     << ": " << lid_or.status();
        record_lease_create_failed();
      }
    }
  }
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  if (handle.view_index_json.has_value()) {
    resp.set_view_index_json(*handle.view_index_json);
  }
  if (resp.view_index_json().empty() && normalized_disk_path.has_value()) {
    // Prefer disk-local canonical index to avoid Global Store dependency when the client provides a disk_path,
    // even if the engine serves the request from an already-loaded local replica.
    if (disk_index.has_value()) {
      resp.set_view_index_json(disk_index->canonical_index_json);
    } else {
      auto local_index_or = store::loader::read_from_artifact_dir(*normalized_disk_path, dev.ordinal);
      if (local_index_or.ok()) {
        resp.set_view_index_json(local_index_or->canonical_index_json);
        VLOG(1) << "MaterializationController: filled view_index_json from disk for artifact_id="
                << handle.replica_key.artifact_id;
      } else {
        LOG(WARNING) << "Failed to read canonical index from disk for artifact_id=" << handle.replica_key.artifact_id
                     << ": " << local_index_or.status();
      }
    }
  }
  if (resp.view_index_json().empty() && handle.source == store::loading::MaterializationSource::kDisk) {
    auto index_or = d_.engine.get_canonical_index_by_id(handle.replica_key.artifact_id);
    if (index_or.ok()) {
      resp.set_view_index_json(*index_or);
      VLOG(1) << "MaterializationController: filled view_index_json from engine for disk artifact_id="
              << handle.replica_key.artifact_id;
    } else {
      LOG(WARNING) << "Failed to fetch canonical index for disk materialization response: " << index_or.status();
    }
  }
  if (handle.view_data_hash.has_value()) {
    resp.set_view_data_hash(*handle.view_data_hash);
  }
  return finalize_response();
}

grpc::Status MaterializationController::materialize_by_key(
    RpcContext& rctx,
    const v2::MaterializeByKeyRequest& req,
    v2::MaterializeByKeyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());
  const v2::DeviceType requested_type = (req.target_device_type() == v2::DeviceType::DEVICE_TYPE_CPU)
      ? v2::DeviceType::DEVICE_TYPE_CPU
      : v2::DeviceType::DEVICE_TYPE_GPU;
  const bool cpu_target = requested_type == v2::DeviceType::DEVICE_TYPE_CPU;
  const bool loopback_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  const bool no_lease = req.lease_mode() == v2::LeaseMode::LEASE_MODE_NO_LEASE;
  const int32_t effective_pid = (loopback_peer && !no_lease) ? req.pid() : 0;
  if (req.wait_for_completion() && !loopback_peer) {
    resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::PERMISSION_DENIED, "wait_for_completion materialization is local-only (loopback/UDS)"};
  }
  if (no_lease && req.wait_for_completion()) {
    resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::INVALID_ARGUMENT, "lease_mode=NO_LEASE requires wait_for_completion=false"};
  }
  span->SetAttribute("tc.device.type", static_cast<int64_t>(requested_type));
  const auto policy = resolve_source_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  ResolvedSourcePolicy effective_policy = policy;
  if (storage_path_.empty()) {
    effective_policy.allow_disk = false;
  }
  span->SetAttribute("tc.store.preference", static_cast<int64_t>(effective_policy.preference));
  span->SetAttribute("tc.store.allow_p2p", effective_policy.allow_p2p);
  span->SetAttribute("tc.store.allow_disk", effective_policy.allow_disk);

  using v2::MaterializeReplicaStatus;
  if (d_.shutdown_signal.is_shutting_down()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  absl::Status policy_status = validate_source_policy(effective_policy);
  if (!policy_status.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(policy_status);
  }
  if (req.key().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (cpu_target) {
    if (!loopback_peer) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::PERMISSION_DENIED, "CPU shared-memory materialization is local-only"};
    }
    if (effective_pid <= 0) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::INVALID_ARGUMENT, "pid is required for CPU handle leases"};
    }
    if (!d_.cpu_shared_memory_enabled) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "cpu_shared_memory is disabled"};
    }
    if (d_.handle_leases == nullptr) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "local handle plane is disabled (no handle leases)"};
    }
  }

  auto finalize_response = [&]() -> grpc::Status {
    if (no_lease) {
      resp.clear_mem_handle();
    }
    auto layout_or = resolve_layout_json_by_key(resp, d_.engine);
    if (!layout_or.ok()) {
      return to_grpc_status(layout_or.status());
    }
    absl::Status payload_status = populate_materialize_payloads(
        resp,
        *layout_or,
        req.tensor_names(),
        /*device_uuid=*/"",
        req.view_subset_hash(),
        req.wait_for_completion(),
        req.replica_uuid(),
        /*ticket_device_uuid=*/nullptr,
        /*view_plan=*/std::nullopt,
        /*prefer_view_plan=*/false,
        /*fill_view_index_bytes=*/true);
    if (!payload_status.ok()) {
      return to_grpc_status(payload_status);
    }
    rctx.mark_success();
    return Status::OK;
  };

  auto mapping_or = d_.engine.resolve_key_mapping(req.key());
  if (!mapping_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(mapping_or.status());
  }
  const auto& mapping = *mapping_or;
  std::string resolved_artifact_id = mapping.artifact_id;
  auto binding_or = resolve_artifact_binding(d_.global_store_client, resolved_artifact_id);
  if (!binding_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(binding_or.status());
  }
  if (binding_or->has_value()) {
    const auto& bound = binding_or->value();
    span->SetAttribute("tc.artifact.bound", bound);
    resolved_artifact_id = bound;
  }
  span->SetAttribute("tc.artifact.id", resolved_artifact_id);
  std::string used_disk_path;
  if (!mapping.disk_path.empty() && effective_policy.allow_disk) {
    auto normalized_or = normalize_disk_path(mapping.disk_path, storage_path_);
    if (!normalized_or.ok()) {
      record_disk_path_denied();
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(normalized_or.status());
    }
    used_disk_path = normalized_or->string();
  }

  // Try LIP fast path first (GPU only)
  std::optional<store::loading::ReplicaKey> lip_replica_key;
  if (!cpu_target) {
    absl::Status session_status = absl::OkStatus();
    auto satisfied = d_.lip.try_satisfy_from_lip(
        resolved_artifact_id,
        req.device_id(),
        [&](const store::loading::ReplicaKey& rkey) {
          lip_replica_key = rkey;
          session_status = register_session_and_refs(
              d_.sessions, d_.refs, rkey, nullptr, req.replica_uuid(), effective_pid, loopback_peer);
        },
        resp.mutable_mem_handle());
    if (!session_status.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(session_status);
    }
    if (!satisfied.ok()) {
      // If LIP path fails for reasons like same-device denial, fall back to engine path.
      // Only propagate errors that indicate a broader failure.
      // For simple parity, we treat FailedPrecondition as a miss and continue.
      if (!absl::IsFailedPrecondition(satisfied.status())) {
        return to_grpc_status(satisfied.status());
      }
    }
    if (*satisfied) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
      resp.set_artifact_id(mapping.artifact_id);
      resp.set_used_disk_path(used_disk_path);
      resp.set_source(v2::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA);
      span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
      bool lease_created = false;
      if (d_.handle_leases != nullptr && effective_pid > 0 && lip_replica_key.has_value()) {
        auto token_or = d_.handle_leases->mint_cuda_ipc_lease(*lip_replica_key, effective_pid);
        if (token_or.ok()) {
          resp.mutable_mem_handle()->set_lease_token(*token_or);
          lease_created = true;
        } else {
          LOG(WARNING) << "mint_cuda_ipc_lease failed (LIP by-key): key=" << *lip_replica_key
                       << " pid=" << effective_pid << ": " << token_or.status();
          record_lease_create_failed();
        }
      }
      if (!lease_created && d_.lifecycle != nullptr && effective_pid > 0 && lip_replica_key.has_value()) {
        auto lid_or = d_.lifecycle->create_use_lease(*lip_replica_key, effective_pid);
        if (!lid_or.ok()) {
          LOG(WARNING) << "create_use_lease failed (LIP by-key): key=" << *lip_replica_key << " pid=" << effective_pid
                       << ": " << lid_or.status();
          record_lease_create_failed();
        }
      }
      return finalize_response();
    }
  }

  // Engine path
  store::DeviceKey dev;
  if (cpu_target) {
    dev = d_.devices.From(v2::DeviceType::DEVICE_TYPE_CPU, /*uuid=*/"", /*ordinal_hint=*/std::nullopt);
  } else {
    // Validate device_id
    if (req.device_id() < 0 || req.device_id() >= d_.engine.get_num_gpus()) {
      return {StatusCode::INVALID_ARGUMENT, "invalid device_id"};
    }
    dev = store::DeviceRegistry::instance().gpu_key(req.device_id());
  }
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.artifact_id = resolved_artifact_id;
  if (!used_disk_path.empty()) {
    hints.disk_path = used_disk_path;
  }
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;

  auto result = d_.engine.materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, hints);
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  {
    const absl::Status session_status = register_session_and_refs(
        d_.sessions,
        d_.refs,
        handle.replica_key,
        handle.ready_signal,
        req.replica_uuid(),
        effective_pid,
        loopback_peer);
    if (!session_status.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(session_status);
    }
  }
  if (cpu_target) {
    if (!handle.cpu_memfd_region.has_value()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return {StatusCode::FAILED_PRECONDITION, "CPU memfd handle unavailable for replica"};
    }
    const auto& region = *handle.cpu_memfd_region;
    auto chunks_or = build_export_chunks_for_replica(d_.engine, handle.replica_key);
    if (!chunks_or.ok()) {
      chunks_or = build_export_chunks_for_replica(d_.engine, handle.replica_key, region.size_bytes);
    }
    if (!chunks_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(chunks_or.status());
    }
    HandleLeaseRegistry::CpuMemfdDescriptor memfd_desc{
        .fd = region.fd,
        .size_bytes = region.size_bytes,
        .offset_bytes = region.offset_bytes,
    };
    auto token_or = d_.handle_leases->mint_cpu_memfd_lease(handle.replica_key, effective_pid, memfd_desc, *chunks_or);
    if (!token_or.ok()) {
      if (!req.replica_uuid().empty()) {
        (void)d_.sessions.erase(req.replica_uuid());
      }
      if (effective_pid > 0) {
        d_.refs.drop_ref(handle.replica_key, effective_pid);
      }
      (void)d_.engine.unload_replica(handle.replica_key);
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(token_or.status());
    }
    auto* cpu = resp.mutable_mem_handle()->mutable_cpu_memfd();
    cpu->set_size_bytes(region.size_bytes);
    cpu->set_offset_bytes(region.offset_bytes);
    resp.mutable_mem_handle()->set_lease_token(*token_or);
  } else {
    if (handle.cuda_ipc_handle.is_valid()) {
      auto handle_view = handle.cuda_ipc_handle.as_string_view();
      resp.mutable_mem_handle()->set_cuda_ipc_handle(handle_view.data(), handle_view.size());
    }
    bool lease_created = false;
    if (d_.handle_leases != nullptr && effective_pid > 0) {
      auto token_or = d_.handle_leases->mint_cuda_ipc_lease(handle.replica_key, effective_pid);
      if (token_or.ok()) {
        resp.mutable_mem_handle()->set_lease_token(*token_or);
        lease_created = true;
      } else {
        LOG(WARNING) << "mint_cuda_ipc_lease failed (engine by-key): key=" << handle.replica_key
                     << " pid=" << effective_pid << ": " << token_or.status();
        record_lease_create_failed();
      }
    }
    if (!lease_created && d_.lifecycle != nullptr && effective_pid > 0) {
      auto lid_or = d_.lifecycle->create_use_lease(handle.replica_key, effective_pid);
      if (!lid_or.ok()) {
        LOG(WARNING) << "create_use_lease failed (engine by-key): key=" << handle.replica_key
                     << " pid=" << effective_pid << ": " << lid_or.status();
        record_lease_create_failed();
      }
    }
  }
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_artifact_id(resolved_artifact_id);
  resp.set_used_disk_path(used_disk_path);
  resp.set_source(to_proto_source(handle.source));
  span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
  return finalize_response();
}

grpc::Status MaterializationController::materialize_into_target(
    RpcContext& rctx,
    const v2::MaterializeIntoTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    record_materialize_into_target(
        "error", "unavailable", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const auto policy = resolve_source_policy(req.has_source_policy() ? &req.source_policy() : nullptr, req.preference());
  ResolvedSourcePolicy effective_policy = policy;
  if (storage_path_.empty()) {
    effective_policy.allow_disk = false;
  }
  absl::Status policy_status = validate_source_policy(effective_policy);
  if (!policy_status.ok()) {
    record_materialize_into_target(
        "error", "policy_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(policy_status);
  }
  if (!is_loopback_grpc_peer(rctx.server_context().peer())) {
    record_materialize_into_target(
        "error", "non_loopback_peer", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::PERMISSION_DENIED, "MaterializeIntoTarget is local-only (loopback/UDS)"};
  }

  const bool has_artifact_id = req.has_artifact_id() && !req.artifact_id().empty();
  const bool has_key = req.has_key() && !req.key().empty();
  if (has_key) {
    record_materialize_into_target(
        "error", "key_not_supported", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "key-based requests not supported for MaterializeIntoTarget"};
  }
  if (!has_artifact_id) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required for MaterializeIntoTarget"};
  }

  std::string resolved_artifact_id = req.artifact_id();
  auto binding_or = resolve_artifact_binding(d_.global_store_client, resolved_artifact_id);
  if (!binding_or.ok()) {
    record_materialize_into_target(
        "error", "binding_error", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(binding_or.status());
  }
  if (binding_or->has_value()) {
    resolved_artifact_id = binding_or->value();
  }
  if (req.has_disk_fallback() && req.disk_fallback().disk_path().empty()) {
    record_materialize_into_target(
        "error", "disk_fallback_empty", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "disk_fallback.disk_path must not be empty"};
  }
  if (!req.has_target_layout()) {
    record_materialize_into_target(
        "error", "layout_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout is required"};
  }
  if (req.device_uuid().empty()) {
    record_materialize_into_target(
        "error", "device_uuid_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "device_uuid is required"};
  }
  if (req.pid() <= 0) {
    record_materialize_into_target(
        "error", "owner_pid_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "pid is required for MaterializeIntoTarget"};
  }

  const auto& layout = req.target_layout();
  if (layout.layout_kind() != v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "layout_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Only LAYOUT_KIND_COALESCED_UNSPECIFIED is supported"};
  }
  if (layout.index_kind() != v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED &&
      layout.index_kind() != v2::TargetLayout::INDEX_KIND_VIEW) {
    record_materialize_into_target(
        "error", "index_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Unsupported index_kind for MaterializeIntoTarget"};
  }
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS &&
      layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "tensor_spec_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Unsupported tensor_spec_kind for MaterializeIntoTarget"};
  }
  if (layout.storages_size() == 0) {
    record_materialize_into_target(
        "error", "storage_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include at least one storage entry"};
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  for (const auto& storage : layout.storages()) {
    if (storage.storage_source_case() != v2::StorageEntry::kVramRegionId) {
      record_materialize_into_target(
          "error", "storage_not_region", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "Target storage must reference a vram_region_id"};
    }
    if (storage.device_id() != device.ordinal) {
      record_materialize_into_target(
          "error", "device_uuid_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage.device_id does not match device_uuid"};
    }
  }

  auto offsets_or = resolve_target_offsets(layout);
  if (!offsets_or.ok()) {
    record_materialize_into_target(
        "error", "offsets_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(offsets_or.status());
  }
  const auto& offsets = *offsets_or;
  if (offsets.empty()) {
    record_materialize_into_target(
        "error", "offsets_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout offsets are required"};
  }

  absl::flat_hash_set<std::string> layout_name_set;
  layout_name_set.reserve(offsets.size());
  std::vector<std::string> layout_names;
  layout_names.reserve(offsets.size());
  for (const auto& entry : offsets) {
    if (entry.name.empty()) {
      record_materialize_into_target(
          "error", "tensor_name_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes empty tensor name"};
    }
    if (!layout_name_set.insert(entry.name).second) {
      record_materialize_into_target(
          "error", "tensor_name_duplicate", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes duplicate tensor name"};
    }
    layout_names.push_back(entry.name);
  }

  std::vector<std::string> request_names;
  absl::flat_hash_set<std::string> request_name_set;
  if (req.tensor_names_size() > 0) {
    request_names.reserve(req.tensor_names_size());
    request_name_set.reserve(req.tensor_names_size());
    for (const auto& name : req.tensor_names()) {
      request_names.push_back(name);
      request_name_set.insert(name);
    }
    if (request_name_set.size() != static_cast<size_t>(req.tensor_names_size())) {
      record_materialize_into_target(
          "error", "tensor_name_duplicate", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "tensor_names must not contain duplicates"};
    }
    if (request_name_set.size() != layout_name_set.size()) {
      record_materialize_into_target(
          "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "tensor_names do not match target_layout entries"};
    }
    for (const auto& name : layout_name_set) {
      if (!request_name_set.contains(name)) {
        record_materialize_into_target(
            "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return {StatusCode::INVALID_ARGUMENT, "tensor_names do not match target_layout entries"};
      }
    }
  }

  auto canonical_json_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!canonical_json_or.ok()) {
    record_materialize_into_target(
        "error", "index_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_json_or.status());
  }
  auto canonical_table_or = parse_canonical_index(*canonical_json_or);
  if (!canonical_table_or.ok()) {
    record_materialize_into_target(
        "error", "index_parse_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_table_or.status());
  }
  const CanonicalIndexTable& canonical_table = *canonical_table_or;
  for (const auto& name : layout_name_set) {
    if (!canonical_table.entries.contains(name)) {
      record_materialize_into_target(
          "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes unknown tensor name"};
    }
  }

  bool has_subset = canonical_table.entries.size() != layout_name_set.size();
  if (!has_subset) {
    for (const auto& [name, _] : canonical_table.entries) {
      if (!layout_name_set.contains(name)) {
        has_subset = true;
        break;
      }
    }
  }
  const bool has_ordered_selection = !request_names.empty();

  std::optional<ViewSpec> view_spec;
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  std::optional<std::string> request_view_id;
  std::optional<std::string> view_data_hash;
  bool view_id_requested = false;

  switch (req.view_identity_case()) {
    case v2::MaterializeIntoTargetRequest::kView: {
      auto spec_or = convert_view_spec(req.view());
      if (!spec_or.ok()) {
        return to_grpc_status(spec_or.status());
      }
      view_spec = std::move(*spec_or);
      view_spec_proto = req.view();
      break;
    }
    case v2::MaterializeIntoTargetRequest::kViewId: {
      if (!req.view_id().empty()) {
        view_id_requested = true;
        request_view_id = req.view_id();
      }
      break;
    }
    case v2::MaterializeIntoTargetRequest::VIEW_IDENTITY_NOT_SET:
      break;
  }

  if (view_id_requested && request_view_id.has_value()) {
    auto view_meta_or = d_.engine.get_view_metadata(req.artifact_id(), *request_view_id);
    if (!view_meta_or.ok()) {
      record_materialize_into_target(
          "error", "view_meta_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(view_meta_or.status());
    }
    auto spec_or = store::view::parse_view_spec_json(view_meta_or->view_spec_json);
    if (!spec_or.ok()) {
      record_materialize_into_target(
          "error", "view_parse_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(spec_or.status());
    }
    view_spec = std::move(*spec_or);
    view_spec_proto = build_view_spec_proto(*view_spec);
    view_data_hash = view_meta_or->view_data_hash;
  }

  std::optional<store::loader::ViewPlan> view_plan;
  if (view_spec.has_value() || has_subset || layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW ||
      has_ordered_selection) {
    ViewSpec plan_spec = view_spec.value_or(ViewSpec{});
    std::vector<std::string> subset_names;
    if (has_ordered_selection) {
      subset_names = request_names;
    } else if (has_subset) {
      subset_names = layout_names;
    }
    auto plan_or = store::StoreEngine::compute_view_plan(*canonical_json_or, plan_spec, subset_names);
    if (!plan_or.ok()) {
      record_materialize_into_target(
          "error", "view_plan_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(plan_or.status());
    }
    view_plan = std::move(*plan_or);
  }

  const bool has_view_transform =
      view_id_requested || (view_spec.has_value() && view_plan.has_value() && !view_plan->is_identity);
  if (view_id_requested && view_plan.has_value() && view_plan->is_identity && !has_subset) {
    record_materialize_into_target(
        "error", "view_identity_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "view_id requires a non-identity view spec"};
  }

  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED) {
    if (has_view_transform || has_subset || has_ordered_selection) {
      record_materialize_into_target(
          "error", "index_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "index_kind CANONICAL cannot be used with view/subset selection"};
    }
  } else {
    if (!has_view_transform && !has_subset && !has_ordered_selection) {
      record_materialize_into_target(
          "error", "index_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "index_kind VIEW requires view or selection order"};
    }
  }

  std::optional<std::string> resolved_view_id;
  if (has_view_transform) {
    if (!view_spec_proto.has_value()) {
      record_materialize_into_target(
          "error", "view_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "view spec required for view transforms"};
    }
    auto view_id_or = compute_view_id_from_spec(*view_spec_proto, *canonical_json_or);
    if (!view_id_or.ok()) {
      return to_grpc_status(view_id_or.status());
    }
    if (request_view_id.has_value() && *request_view_id != *view_id_or) {
      record_materialize_into_target(
          "error", "view_id_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "view_id does not match view spec"};
    }
    resolved_view_id = *view_id_or;
  }

  if (d_.external_target_verification_enabled && resolved_view_id.has_value() && !view_data_hash.has_value()) {
    auto view_meta_or = d_.engine.get_view_metadata(req.artifact_id(), *resolved_view_id);
    if (view_meta_or.ok()) {
      view_data_hash = view_meta_or->view_data_hash;
    } else {
      VLOG(1) << "MaterializeIntoTarget: view metadata unavailable for verification: " << view_meta_or.status();
    }
  }

  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    if (has_view_transform) {
      if (layout.view_id().empty() || (resolved_view_id.has_value() && layout.view_id() != *resolved_view_id)) {
        record_materialize_into_target(
            "error", "view_id_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return {StatusCode::INVALID_ARGUMENT, "target_layout.view_id must match resolved view_id"};
      }
    } else if (!layout.view_id().empty()) {
      record_materialize_into_target(
          "error", "view_id_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout.view_id must be empty for subset-only layouts"};
    }
  } else if (!layout.view_id().empty()) {
    record_materialize_into_target(
        "error", "view_id_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout.view_id not allowed for canonical layout"};
  }

  std::string selected_index_json;
  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    if (!view_plan.has_value()) {
      record_materialize_into_target(
          "error", "view_plan_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::FAILED_PRECONDITION, "view plan missing for VIEW layout"};
    }
    selected_index_json = view_plan->view_index_json;
  } else {
    selected_index_json = *canonical_json_or;
  }

  auto index_table_or = parse_canonical_index(selected_index_json);
  if (!index_table_or.ok()) {
    record_materialize_into_target(
        "error", "index_parse_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(index_table_or.status());
  }
  const CanonicalIndexTable& index_table = *index_table_or;
  const uint64_t logical_total_size = index_table.logical_total_size;
  if (index_table.entries.size() != layout_name_set.size()) {
    record_materialize_into_target(
        "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include every selected tensor"};
  }
  for (const auto& name : layout_name_set) {
    if (!index_table.entries.contains(name)) {
      record_materialize_into_target(
          "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes unknown tensor name"};
    }
  }
  if (view_plan.has_value() && view_plan->view_size_bytes > 0 && view_plan->view_size_bytes != logical_total_size) {
    record_materialize_into_target(
        "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "view plan size does not match selected index"};
  }

  if (layout.tensor_spec_kind() == v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    for (const auto& alias : layout.aliases()) {
      auto it = index_table.entries.find(alias.name());
      if (it == index_table.entries.end()) {
        record_materialize_into_target(
            "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return {StatusCode::INVALID_ARGUMENT, "target_layout alias includes unknown tensor name"};
      }
      const auto& entry = it->second;
      if (alias.logical_length() != entry.logical_length) {
        record_materialize_into_target(
            "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return {StatusCode::INVALID_ARGUMENT, "target_layout alias logical_length mismatch"};
      }
      if (alias.dtype() != entry.dtype) {
        record_materialize_into_target(
            "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return {StatusCode::INVALID_ARGUMENT, "target_layout alias dtype mismatch"};
      }
      if (alias.shape_size() != static_cast<int>(entry.shape.size()) ||
          alias.stride_size() != static_cast<int>(entry.stride.size())) {
        record_materialize_into_target(
            "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return {StatusCode::INVALID_ARGUMENT, "target_layout alias shape/stride mismatch"};
      }
      for (int i = 0; i < alias.shape_size(); ++i) {
        if (alias.shape(i) != entry.shape[static_cast<size_t>(i)]) {
          record_materialize_into_target(
              "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
          return {StatusCode::INVALID_ARGUMENT, "target_layout alias shape mismatch"};
        }
      }
      for (int i = 0; i < alias.stride_size(); ++i) {
        if (alias.stride(i) != entry.stride[static_cast<size_t>(i)]) {
          record_materialize_into_target(
              "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
          return {StatusCode::INVALID_ARGUMENT, "target_layout alias stride mismatch"};
        }
      }
    }
  }

  struct StorageRange {
    std::string storage_id;
    uint64_t base_offset{0};
    uint64_t length{0};
  };

  std::vector<RegisterStorageMeta> publish_storages;
  std::vector<LeaseSegMeta> publish_segments;
  publish_storages.reserve(layout.storages_size());
  publish_segments.reserve(layout.storages_size());

  absl::flat_hash_map<std::string, StorageRange> storage_ranges;
  storage_ranges.reserve(layout.storages_size());
  uint64_t range_cursor = 0;
  for (const auto& storage : layout.storages()) {
    if (storage.storage_id().empty()) {
      record_materialize_into_target(
          "error", "storage_id_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage_id is required for each storage entry"};
    }
    if (storage.storage_length() == 0) {
      record_materialize_into_target(
          "error", "storage_length_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage_length must be non-zero"};
    }
    if (storage_ranges.contains(storage.storage_id())) {
      record_materialize_into_target(
          "error", "storage_id_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage_id must be unique in target_layout"};
    }
    StorageRange range;
    range.storage_id = storage.storage_id();
    range.base_offset = range_cursor;
    range.length = storage.storage_length();
    storage_ranges.emplace(range.storage_id, range);
    RegisterStorageMeta meta;
    meta.storage_id = storage.storage_id();
    meta.device_id = storage.device_id();
    meta.storage_length = storage.storage_length();
    if (!storage.vram_region_id().empty()) {
      meta.region_id = storage.vram_region_id();
    }
    if (!storage.cuda_ipc_handle().empty()) {
      meta.handle_bytes = storage.cuda_ipc_handle();
    }
    meta.mapping_base_offset = storage.mapping_base_offset();
    publish_storages.push_back(std::move(meta));

    LeaseSegMeta seg;
    seg.storage_id = storage.storage_id();
    seg.storage_offset = 0;
    seg.artifact_offset = range_cursor;
    seg.length = storage.storage_length();
    publish_segments.push_back(std::move(seg));
    if (storage.storage_length() > std::numeric_limits<uint64_t>::max() - range_cursor) {
      record_materialize_into_target(
          "error", "storage_length_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "storage_length sum overflow"};
    }
    range_cursor += storage.storage_length();
  }
  if (range_cursor != logical_total_size) {
    record_materialize_into_target(
        "error", "storage_length_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "storage_length must span logical byte space"};
  }

  absl::flat_hash_set<std::string> seen_offsets;
  seen_offsets.reserve(offsets.size());
  for (const auto& entry : offsets) {
    if (!seen_offsets.insert(entry.name).second) {
      record_materialize_into_target(
          "error", "tensor_name_duplicate", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes duplicate tensor name"};
    }
    auto it = index_table.entries.find(entry.name);
    if (it == index_table.entries.end()) {
      record_materialize_into_target(
          "error", "tensor_name_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes unknown tensor name"};
    }
    const auto range_it = storage_ranges.find(entry.storage_id);
    if (range_it == storage_ranges.end()) {
      record_materialize_into_target(
          "error", "storage_id_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout references unknown storage_id"};
    }
    const auto& range = range_it->second;
    const auto& index_entry = it->second;
    if (entry.logical_length != index_entry.logical_length) {
      record_materialize_into_target(
          "error", "layout_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout logical_length mismatch"};
    }
    if (index_entry.logical_offset < range.base_offset) {
      record_materialize_into_target(
          "error", "offset_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout offset out of storage bounds"};
    }
    const uint64_t expected_storage_offset = index_entry.logical_offset - range.base_offset;
    if (entry.storage_offset != expected_storage_offset) {
      record_materialize_into_target(
          "error", "offset_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout storage_offset mismatch"};
    }
    if (expected_storage_offset + entry.logical_length > range.length) {
      record_materialize_into_target(
          "error", "offset_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout exceeds storage bounds"};
    }
  }

  std::string view_subset_hash;
  if (has_subset) {
    const auto& subset_names = has_ordered_selection ? request_names : layout_names;
    view_subset_hash = common::compute_view_subset_hash_bytes(absl::MakeSpan(subset_names));
    if (!req.view_subset_hash().empty() && req.view_subset_hash() != view_subset_hash) {
      record_materialize_into_target(
          "error", "subset_hash_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "view_subset_hash does not match tensor_names"};
    }
  } else if (!req.view_subset_hash().empty()) {
    record_materialize_into_target(
        "error", "subset_hash_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "view_subset_hash must be empty for full selection"};
  }

  std::optional<std::string> expected_data_hash;
  bool verify_external_target = false;
  if (d_.external_target_verification_enabled) {
    if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED && !has_view_transform &&
        !has_subset) {
      expected_data_hash = parse_mi2_data_multihash(resolved_artifact_id);
    } else if (resolved_view_id.has_value()) {
      expected_data_hash = view_data_hash;
    }
    if (expected_data_hash.has_value()) {
      record_materialize_into_target_verification_enabled();
      verify_external_target = true;
    } else {
      record_materialize_into_target_verification_skipped();
    }
  } else {
    record_materialize_into_target_verification_skipped();
  }

  if (rctx.server_context().IsCancelled()) {
    record_materialize_into_target("error", "cancelled", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::CANCELLED, "request cancelled before transfer"};
  }

  struct RegionReleaseGuard {
    IpcRegionRegistry& registry;
    std::vector<std::string>& region_ids;
    bool active{true};

    ~RegionReleaseGuard() {
      if (active) {
        for (const auto& region_id : region_ids) {
          (void)registry.release(region_id);
        }
      }
    }
  };

  struct RegionMapping {
    IpcRegionRegistry::RegionDescriptor desc;
    std::unique_ptr<cuda::IpcMapping> mapping;
  };

  absl::flat_hash_map<std::string, RegionMapping> region_map;
  region_map.reserve(layout.storages_size());
  std::vector<std::string> acquired_regions;
  acquired_regions.reserve(layout.storages_size());
  RegionReleaseGuard guard{d_.regions, acquired_regions, true};

  std::vector<store::loading::IntoTargetStorage> target_storages;
  target_storages.reserve(layout.storages_size());
  for (const auto& storage : layout.storages()) {
    auto it = region_map.find(storage.vram_region_id());
    if (it == region_map.end()) {
      auto region_desc_or = d_.regions.acquire(storage.vram_region_id(), req.pid());
      if (!region_desc_or.ok()) {
        const absl::Status& st = region_desc_or.status();
        const bool poisoned = absl::IsFailedPrecondition(st) && st.message() == "region is poisoned";
        record_materialize_into_target(
            "error",
            poisoned ? "region_poisoned" : "region_missing",
            v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return to_grpc_status(st);
      }
      auto handle_or = d_.regions.get_handle_bytes(storage.vram_region_id());
      if (!handle_or.ok()) {
        record_materialize_into_target(
            "error", "region_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return to_grpc_status(handle_or.status());
      }
      auto map_or = cuda::IpcMapping::open(*handle_or, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
      if (!map_or.ok()) {
        record_materialize_into_target(
            "error", "map_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
        return to_grpc_status(map_or.status());
      }
      RegionMapping mapping{.desc = *region_desc_or, .mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or))};
      auto [inserted_it, _] = region_map.emplace(storage.vram_region_id(), std::move(mapping));
      it = inserted_it;
      acquired_regions.push_back(storage.vram_region_id());
    }
    const auto& region_desc = it->second.desc;
    if (region_desc.device_id != storage.device_id()) {
      record_materialize_into_target(
          "error", "device_uuid_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::FAILED_PRECONDITION, "region device does not match storage device"};
    }
    const uint64_t region_end = storage.mapping_base_offset() + storage.storage_length();
    if (region_end > region_desc.size_bytes) {
      record_materialize_into_target("error", "bounds", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::FAILED_PRECONDITION, "region-backed storage exceeds region bounds"};
    }
    void* region_base_ptr =
        static_cast<uint8_t*>(it->second.mapping->get()) + static_cast<uint64_t>(storage.mapping_base_offset());
    target_storages.push_back(
        store::loading::IntoTargetStorage{
            .base_ptr = gsl::not_null<void*>{region_base_ptr},
            .length = storage.storage_length(),
        });
  }

  std::vector<TargetLayoutSpan> verification_spans;
  if (verify_external_target) {
    verification_spans.reserve(target_storages.size());
    uint64_t cursor = 0;
    for (const auto& storage : target_storages) {
      verification_spans.push_back(
          TargetLayoutSpan{
              .base_ptr = storage.base_ptr,
              .offset = cursor,
              .length = storage.length,
          });
      cursor += storage.length;
    }
  }

  store::loading::MaterializeHints hints;
  hints.artifact_id = resolved_artifact_id;
  if (req.has_disk_fallback() && !req.disk_fallback().disk_path().empty()) {
    if (!effective_policy.allow_disk) {
      record_materialize_into_target(
          "error", "disk_fallback_disallowed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "disk fallback requested but source_policy disallows disk"};
    }
    auto normalized_or = normalize_disk_path(req.disk_fallback().disk_path(), storage_path_);
    if (!normalized_or.ok()) {
      record_disk_path_denied();
      record_materialize_into_target(
          "error", "disk_fallback_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(normalized_or.status());
    }
    hints.disk_path = normalized_or->string();
  }
  hints.source_preference = to_hint_preference(effective_policy.preference);
  hints.allow_p2p = effective_policy.allow_p2p;
  hints.allow_disk = effective_policy.allow_disk;
  hints.verify = store::loading::MaterializeHints::Verify::NONE;

  if (view_plan.has_value() && layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolved_artifact_id;
    if (resolved_view_id.has_value()) {
      variant.view_id = *resolved_view_id;
    }
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    variant.cached_plan = view_plan;
    variant.canonical_index_json = *canonical_json_or;
    variant.placement = resolve_placement(req, view_spec);
    hints.variant = std::move(variant);
  }

  const uint64_t generation = compute_generation_from_index(*canonical_json_or);
  store::loading::IntoTargetLayout target_layout;
  target_layout.storages = std::move(target_storages);
  target_layout.total_size = logical_total_size;
  auto result_or = d_.engine.materialize_into_target(device, target_layout, *canonical_json_or, generation, hints);
  if (!result_or.ok()) {
    if (absl::IsDataLoss(result_or.status())) {
      for (const auto& region_id : acquired_regions) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "transfer_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    } else {
      record_materialize_into_target(
          "error", "transfer_error", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    }
    return to_grpc_status(result_or.status());
  }

  if (verify_external_target) {
    auto actual_hash_or =
        compute_target_layout_multihash(std::move(verification_spans), logical_total_size, device.ordinal);
    if (!actual_hash_or.ok()) {
      for (const auto& region_id : acquired_regions) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "verification_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(
          absl::DataLossError(
              absl::StrCat("external target verification failed: ", actual_hash_or.status().message())));
    }
    if (expected_data_hash.has_value() && *expected_data_hash != *actual_hash_or) {
      for (const auto& region_id : acquired_regions) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "verification_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::DATA_LOSS, "external target verification failed: data hash mismatch"};
    }
  }

  resp.set_artifact_id(resolved_artifact_id);
  resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_source(to_proto_source(result_or->source));
  resp.set_canonical_index_bytes(*canonical_json_or);
  if (view_plan.has_value() && layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    resp.set_view_index_bytes(view_plan->view_index_json);
  }
  if (!layout_names.empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!view_subset_hash.empty()) {
      subset->set_subset_hash(view_subset_hash);
    }
    for (const auto& name : layout_names) {
      subset->add_tensor_names(name);
    }
  }
  resp.set_generation(generation);
  if (capability_tokens_ != nullptr && capability_tokens_->configured()) {
    const std::string view_id_value = resolved_view_id.value_or("");
    const bool needs_view_index = layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW;
    const std::string logical_layout_hash = common::compute_logical_layout_hash_bytes(
        absl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(selected_index_json.data()), selected_index_json.size()),
        needs_view_index);
    std::optional<std::string_view> subset_hash_opt;
    if (!view_subset_hash.empty()) {
      subset_hash_opt = view_subset_hash;
    }
    const std::string selection_hash = common::compute_selection_hash_bytes(view_id_value, subset_hash_opt);

    tensorcast::common::v1::ArtifactSelection selection;
    selection.set_artifact_id(resolved_artifact_id);
    selection.set_view_id(view_id_value);
    selection.set_logical_layout_hash(logical_layout_hash);
    selection.set_selection_hash(selection_hash);
    if (!view_subset_hash.empty()) {
      selection.set_view_subset_hash(view_subset_hash);
    }
    for (const auto& name : req.tensor_names()) {
      selection.add_tensor_names(name);
    }
    if (view_spec_proto.has_value()) {
      selection.mutable_view_spec()->CopyFrom(*view_spec_proto);
    }

    tensorcast::common::v1::ByteSpaceRef byte_space;
    if (!view_id_value.empty()) {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
      byte_space.set_id(view_id_value);
    } else {
      byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
      byte_space.set_id("");
    }

    const std::string layout_hash = compute_target_layout_hash(layout);
    const std::string write_id = mint_write_id(bitgen_);
    const absl::Time expires_at = absl::Now() + kTargetWriteTokenTtl;

    auto stable_index_or = store::loader::rebuild_stable_canonical_index(*canonical_json_or, device.ordinal);
    if (!stable_index_or.ok()) {
      VLOG(1) << "MaterializeIntoTarget: failed to rebuild canonical index for target write token: "
              << stable_index_or.status();
    } else {
      std::string stable_index_json = std::move(*stable_index_or);
      const auto digest = common::sha256_digest_bytes(
          absl::Span<const uint8_t>(
              reinterpret_cast<const uint8_t*>(stable_index_json.data()), stable_index_json.size()));
      std::string index_key_hex =
          absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));

      tensorcast::common::v1::TargetWriteScope scope;
      scope.set_write_id(write_id);
      scope.mutable_selection()->CopyFrom(selection);
      scope.mutable_byte_space()->CopyFrom(byte_space);
      scope.set_device_uuid(req.device_uuid());
      scope.set_owner_pid(req.pid());
      scope.set_target_layout_hash(layout_hash);

      auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
      if (scope_or.ok()) {
        const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(expires_at));
        auto token_or = capability_tokens_->mint(
            d_.identity.daemon_id(),
            tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_WRITE,
            *scope_or,
            expires_at_ms);
        if (token_or.ok()) {
          TargetWriteRegistry::Record record;
          record.write_id = write_id;
          record.layout_key = layout_hash;
          record.target_layout_hash = layout_hash;
          record.selection = selection;
          record.byte_space = byte_space;
          record.canonical_index_json = std::move(stable_index_json);
          record.index_key_hex = std::move(index_key_hex);
          record.device_uuid = req.device_uuid();
          record.owner_pid = req.pid();
          if (req.has_operation_id()) {
            record.operation_id = req.operation_id();
          }
          record.expires_at = expires_at;
          record.segments = std::move(publish_segments);
          record.storages = std::move(publish_storages);
          auto inserted = target_write_registry_.insert(std::move(record));
          (void)inserted;
          resp.set_target_write_token(*token_or);
        } else {
          VLOG(1) << "MaterializeIntoTarget: failed to mint target_write_token: " << token_or.status();
        }
      } else {
        VLOG(1) << "MaterializeIntoTarget: failed to serialize target_write scope: " << scope_or.status();
      }
    }
  }
  record_materialize_into_target("ok", "ok", resp.source());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs() && req.has_operation_id()) {
    span->SetAttribute("tc.operation.id", req.operation_id());
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.target_write_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_write_token is required"};
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return {StatusCode::FAILED_PRECONDITION, "capability tokens not configured"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "Global Store client unavailable"};
  }

  auto normalize_space =
      [](const tensorcast::common::v1::ByteSpaceRef& space) -> absl::StatusOr<tensorcast::common::v1::ByteSpaceRef> {
    tensorcast::common::v1::ByteSpaceRef out;
    switch (space.kind()) {
      case tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED:
      case tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL:
        out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
        out.set_id("");
        return out;
      case tensorcast::common::v1::BYTE_SPACE_KIND_VIEW:
        if (space.id().empty()) {
          return absl::InvalidArgumentError("byte_space VIEW requires id");
        }
        out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
        out.set_id(space.id());
        return out;
      default:
        return absl::InvalidArgumentError("unsupported byte_space kind");
    }
  };

  auto normalized_req_or = normalize_space(req.byte_space());
  if (!normalized_req_or.ok()) {
    return to_grpc_status(normalized_req_or.status());
  }
  tensorcast::common::v1::ByteSpaceRef normalized_req = std::move(*normalized_req_or);

  auto env_or = capability_tokens_->verify(
      req.target_write_token(),
      tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_WRITE,
      d_.identity.daemon_id(),
      absl::Now(),
      /*require_not_expired=*/true);
  if (!env_or.ok()) {
    return to_grpc_status(env_or.status());
  }

  tensorcast::common::v1::TargetWriteScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return {StatusCode::INVALID_ARGUMENT, "target_write_token scope parse failed"};
  }
  if (req.has_owner_pid() && scope.owner_pid() != req.owner_pid()) {
    return {StatusCode::PERMISSION_DENIED, "owner_pid mismatch for target_write_token"};
  }

  auto normalized_scope_or = normalize_space(scope.byte_space());
  if (!normalized_scope_or.ok()) {
    return to_grpc_status(normalized_scope_or.status());
  }
  tensorcast::common::v1::ByteSpaceRef normalized_scope = std::move(*normalized_scope_or);
  if (normalized_scope.kind() != normalized_req.kind() || normalized_scope.id() != normalized_req.id()) {
    return {StatusCode::INVALID_ARGUMENT, "byte_space does not match target_write_token"};
  }

  if (scope.write_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "target_write_token missing write_id"};
  }

  auto record_opt = target_write_registry_.lookup(scope.write_id(), absl::Now(), /*require_not_expired=*/true);
  if (!record_opt.has_value()) {
    return {StatusCode::NOT_FOUND, "target_write_token is no longer valid"};
  }
  auto record = std::move(*record_opt);
  if (!target_write_registry_.is_current_for_layout(record.layout_key, scope.write_id())) {
    return {StatusCode::FAILED_PRECONDITION, "target_write_token is stale for layout"};
  }
  if (record.device_uuid != scope.device_uuid()) {
    return {StatusCode::FAILED_PRECONDITION, "device_uuid mismatch for target_write_token"};
  }
  if (record.owner_pid != scope.owner_pid()) {
    return {StatusCode::FAILED_PRECONDITION, "owner_pid mismatch for target_write_token"};
  }
  if (record.target_layout_hash != scope.target_layout_hash()) {
    return {StatusCode::FAILED_PRECONDITION, "target_layout_hash mismatch for target_write_token"};
  }
  if (record.byte_space.kind() != normalized_scope.kind() || record.byte_space.id() != normalized_scope.id()) {
    return {StatusCode::FAILED_PRECONDITION, "byte_space mismatch for target_write_token"};
  }
  if (record.selection.artifact_id() != scope.selection().artifact_id() ||
      record.selection.view_id() != scope.selection().view_id() ||
      record.selection.logical_layout_hash() != scope.selection().logical_layout_hash() ||
      record.selection.selection_hash() != scope.selection().selection_hash() ||
      record.selection.view_subset_hash() != scope.selection().view_subset_hash() ||
      record.selection.tensor_names_size() != scope.selection().tensor_names_size()) {
    return {StatusCode::FAILED_PRECONDITION, "selection mismatch for target_write_token"};
  }
  for (int i = 0; i < record.selection.tensor_names_size(); ++i) {
    if (record.selection.tensor_names(i) != scope.selection().tensor_names(i)) {
      return {StatusCode::FAILED_PRECONDITION, "selection tensor_names mismatch for target_write_token"};
    }
  }

  if (!scope.selection().tensor_names().empty() || !scope.selection().view_subset_hash().empty()) {
    return {StatusCode::FAILED_PRECONDITION, "selection is not publishable (packed or subset)"};
  }
  if (scope.selection().artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id missing from target_write_token"};
  }

  const auto device = d_.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, scope.device_uuid(), std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return {StatusCode::INVALID_ARGUMENT, "invalid device_uuid for target_write_token"};
  }

  const std::string view_id =
      normalized_scope.kind() == tensorcast::common::v1::BYTE_SPACE_KIND_VIEW ? normalized_scope.id() : "";
  ArtifactDeviceKey key{
      .artifact_id = scope.selection().artifact_id(), .view_id = view_id, .device_id = device.ordinal};

  if (auto active = d_.lip_manager.find_active_by_key(key); active.has_value()) {
    if (active->registration_id == scope.write_id()) {
      auto replica_id = d_.lip_manager.find_replica_id(key);
      if (!replica_id.has_value()) {
        return {StatusCode::FAILED_PRECONDITION, "target already published without replica_id"};
      }
      resp.set_lease_id(scope.write_id());
      resp.set_replica_id(*replica_id);
      rctx.mark_success();
      return Status::OK;
    }
    return {StatusCode::ALREADY_EXISTS, "another lease already exists for target"};
  }

  uint64_t total_size = 0;
  for (const auto& seg : record.segments) {
    if (seg.length == 0) {
      continue;
    }
    const uint64_t end = seg.artifact_offset + seg.length;
    if (end > total_size) {
      total_size = end;
    }
  }
  if (total_size == 0) {
    return {StatusCode::FAILED_PRECONDITION, "target_write_token has empty segments"};
  }

  struct LipRollback {
    LipManager* lip{nullptr};
    std::string registration_id;
    bool active{true};

    ~LipRollback() {
      if (!active || lip == nullptr) {
        return;
      }
      absl::Status st = lip->revoke_by_registration_id(registration_id);
      if (!st.ok()) {
        LOG(WARNING) << "PublishTargetReplica rollback: revoke failed for id=" << registration_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } lip_rollback{.lip = &d_.lip_manager, .registration_id = scope.write_id()};

  const uint32_t ttl_ms = req.has_ttl_ms() ? req.ttl_ms() : 0U;
  const uint64_t epoch = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  auto lease_or = d_.lip_manager.commit_routable_view_lease_in_place(
      scope.write_id(),
      scope.selection().artifact_id(),
      view_id,
      device.ordinal,
      scope.owner_pid(),
      ttl_ms,
      epoch,
      total_size,
      std::move(record.segments),
      std::move(record.storages));
  if (!lease_or.ok()) {
    lip_rollback.release();
    return to_grpc_status(lease_or.status());
  }

  std::string worker_id = d_.identity.worker_id();
  if (worker_id.empty()) {
    worker_id = "local";
  }

  auto replica_id_or = d_.global_store_client->register_memory_replica(
      scope.selection().artifact_id(),
      worker_id,
      device,
      total_size,
      record.index_key_hex,
      lease_or->remote_memory_keys,
      lease_or->buffer_sizes,
      record.canonical_index_json,
      /*encoding=*/"json",
      /*schema_version=*/"v3",
      /*max_concurrency=*/1,
      /*verification_json=*/std::nullopt,
      view_id.empty() ? std::nullopt : std::optional<std::string_view>(view_id));
  if (!replica_id_or.ok()) {
    return to_grpc_status(replica_id_or.status());
  }
  const std::string replica_id = *replica_id_or;
  d_.lip_manager.attach_replica_id(scope.write_id(), replica_id);

  lip_rollback.release();
  resp.set_lease_id(scope.write_id());
  resp.set_replica_id(replica_id);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::resolve_artifact_from_disk(
    RpcContext& rctx,
    const v2::ResolveArtifactFromDiskRequest& req,
    v2::ResolveArtifactFromDiskResponse& resp) {
  auto& span = rctx.span();
  const bool verify_checksums = req.verify_checksums();
  if (req.disk_path().empty()) {
    record_disk_resolution_outcome("invalid_argument");
    return {StatusCode::INVALID_ARGUMENT, "disk_path is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    record_disk_resolution_outcome("unavailable");
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto normalized_or = normalize_disk_path(req.disk_path(), storage_path_);
  if (!normalized_or.ok()) {
    record_disk_path_denied();
    record_disk_resolution_outcome("invalid_argument");
    return to_grpc_status(normalized_or.status());
  }
  const auto& normalized = *normalized_or;
  resp.set_disk_path(normalized.string());
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", normalized.string());
  }
  span->SetAttribute("tc.store.verify_checksums", verify_checksums);

  auto descriptor_or = load_descriptor_metadata(normalized);
  if (!descriptor_or.ok()) {
    record_disk_resolution_outcome("invalid_descriptor");
    return to_grpc_status(descriptor_or.status());
  }
  const DescriptorMetadata& descriptor = *descriptor_or;
  if (descriptor.artifact_id.has_value()) {
    resp.set_artifact_id(*descriptor.artifact_id);
    span->SetAttribute("tc.artifact.id", *descriptor.artifact_id);
  }

  auto index_presence_status = ensure_tensor_index_present(normalized);
  if (!index_presence_status.ok()) {
    record_disk_resolution_outcome("not_found");
    return to_grpc_status(index_presence_status);
  }

  auto index_or = store::loader::read_from_artifact_dir(normalized, /*target_device_id=*/0);
  if (!index_or.ok()) {
    record_disk_resolution_outcome("not_found");
    return to_grpc_status(index_or.status());
  }

  auto validation_status = validate_descriptor_against_index(descriptor, *index_or, verify_checksums);
  if (!validation_status.ok()) {
    record_disk_resolution_outcome("checksum_failed");
    return to_grpc_status(validation_status);
  }

  if (!resp.artifact_id().empty() && descriptor.artifact_id.has_value() &&
      *descriptor.artifact_id != resp.artifact_id()) {
    record_disk_resolution_outcome("checksum_failed");
    return to_grpc_status(
        absl::FailedPreconditionError("artifact_id mismatch between resolved descriptor and request"));
  }
  if (resp.artifact_id().empty() && descriptor.index_multihash.has_value() && descriptor.data_multihash.has_value() &&
      !descriptor.index_multihash->empty() && !descriptor.data_multihash->empty()) {
    const std::string artifact_id = absl::StrCat("mi2:", *descriptor.index_multihash, ":", *descriptor.data_multihash);
    resp.set_artifact_id(artifact_id);
    if (rctx.allow_high_card_attrs()) {
      span->SetAttribute("tc.artifact.id", artifact_id);
    }
  }

  resp.set_canonical_index_bytes(index_or->canonical_index_json);
  const uint64_t generation = compute_generation_from_index(index_or->canonical_index_json);
  resp.set_generation(generation);
  span->SetAttribute("tc.artifact.generation", static_cast<int64_t>(generation));
  record_disk_resolution_outcome("ok");
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::get_artifact_index_by_id(
    RpcContext& rctx,
    const v2::GetArtifactIndexByIdRequest& req,
    v2::GetArtifactIndexByIdResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());

  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  auto bytes_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!bytes_or.ok())
    return to_grpc_status(bytes_or.status());
  resp.set_tensor_index_data(*bytes_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto result_or = d_.engine.seal_assembly(req.assembly_id(), req.publish_canonical());
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  const auto& result = *result_or;
  resp.set_sealed_artifact_id(result.sealed_artifact_id);
  resp.set_already_sealed(result.already_sealed);
  auto* desc = resp.mutable_descriptor_();
  desc->set_artifact_id(result.sealed_artifact_id);
  if (!result.index_multihash.empty()) {
    desc->set_index_multihash(result.index_multihash);
  }
  if (!result.data_multihash.empty()) {
    desc->set_data_multihash(result.data_multihash);
  }
  if (!result.schema_version.empty()) {
    desc->set_schema_version(result.schema_version);
  }
  if (!result.encoding.empty()) {
    desc->set_encoding(result.encoding);
  }
  if (result.total_size > 0) {
    desc->set_total_size(result.total_size);
  }
  desc->set_id_kind(tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_MI2);
  rctx.mark_success();
  return Status::OK;
}

namespace {

std::string compute_seal_operation_id(std::string_view assembly_id) {
  const std::string payload = absl::StrCat("seal_assembly:", assembly_id);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::string owner_id_for_operation(const WorkerIdentityStore& identity) {
  auto daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  auto worker_id = identity.worker_id();
  if (!worker_id.empty()) {
    return worker_id;
  }
  return "unknown";
}

bool retryable_status(const absl::Status& st) {
  return absl::IsUnavailable(st) || absl::IsDeadlineExceeded(st) || absl::IsAborted(st) || absl::IsInternal(st) ||
      absl::IsUnknown(st);
}

constexpr uint64_t kProofChunkBytesV1 = 4ULL * 1024 * 1024;

struct TensorInterval {
  std::string tensor_name;
  uint64_t offset{0};
  uint64_t size_bytes{0};
};

absl::StatusOr<std::vector<TensorInterval>> parse_tensor_intervals(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical_index_json must not be empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", ex.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  std::vector<TensorInterval> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string tensor_name = it.key();
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    TensorInterval interval;
    interval.tensor_name = tensor_name;
    interval.offset = arr[0].get<uint64_t>();
    interval.size_bytes = arr[1].get<uint64_t>();
    out.push_back(std::move(interval));
  }

  std::sort(
      out.begin(), out.end(), [](const TensorInterval& a, const TensorInterval& b) { return a.offset < b.offset; });
  return out;
}

std::vector<uint8_t> compute_view_meta_digest(const store::components::ViewInfo& view) {
  std::vector<store::components::CanonicalRange> ranges = view.canonical_ranges;
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    if (a.offset != b.offset) {
      return a.offset < b.offset;
    }
    return a.length < b.length;
  });

  std::string payload;
  payload.reserve(256 + ranges.size() * 32);
  absl::StrAppend(&payload, "view_id=", view.view_id, ";");
  absl::StrAppend(&payload, "view_data_hash=", view.view_data_hash.value_or(""), ";");
  absl::StrAppend(&payload, "view_size_bytes=", view.view_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_size_bytes=", view.canonical_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_bytes_covered=", view.canonical_bytes_covered, ";");
  for (const auto& range : ranges) {
    absl::StrAppend(&payload, range.offset, ":", range.length, ";");
  }

  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  return tensorcast::common::sha256_digest_bytes(bytes);
}

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id) {
  if (assembly_id.empty() || mi2_id.empty()) {
    return absl::InvalidArgumentError("check_post_seal_view_reuse_safe requires assembly_id and mi2_id");
  }

  auto layouts_or = client.list_artifact_layouts(mi2_id);
  if (!layouts_or.ok()) {
    return layouts_or.status();
  }
  if (layouts_or->empty()) {
    return true;
  }

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> tensors_by_schema;
  for (const auto& layout_id : *layouts_or) {
    if (layout_id.empty()) {
      continue;
    }
    auto spec_or = client.get_layout_spec(layout_id);
    if (!spec_or.ok()) {
      return spec_or.status();
    }
    const auto& layout_spec = spec_or->layout();
    const std::string schema_version = layout_spec.proof_schema_version();
    for (const auto& entry : layout_spec.tensors()) {
      if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
        if (schema_version.empty()) {
          return absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
        }
        tensors_by_schema[schema_version].insert(entry.first);
      }
    }
  }

  if (tensors_by_schema.empty()) {
    return true;
  }

  for (const auto& [schema_version, tensors] : tensors_by_schema) {
    if (tensors.empty()) {
      continue;
    }
    tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest req;
    req.set_assembly_id(std::string(assembly_id));
    req.set_mi2_id(std::string(mi2_id));
    req.set_proof_schema_version(schema_version);
    for (const auto& name : tensors) {
      if (!name.empty()) {
        req.add_tensor_names(name);
      }
    }
    auto resp_or = client.check_proof_commitments_match(req);
    if (!resp_or.ok()) {
      return resp_or.status();
    }
    if (!resp_or->match()) {
      return false;
    }
  }
  return true;
}

} // namespace

grpc::Status MaterializationController::start_seal_assembly(
    RpcContext& rctx,
    const v2::StartSealAssemblyRequest& req,
    v2::StartSealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const std::string operation_id = compute_seal_operation_id(req.assembly_id());
  auto* out_ref = resp.mutable_operation();
  out_ref->set_operation_id(operation_id);
  out_ref->set_kind("seal_assembly");
  out_ref->set_target_artifact_id(req.assembly_id());

  tensorcast::operation::v1::AcquireOperationLeaseRequest lease_req;
  lease_req.set_operation_id(operation_id);
  lease_req.set_kind("seal_assembly");
  lease_req.set_target_artifact_id(req.assembly_id());
  lease_req.set_owner_id(owner_id_for_operation(d_.identity));
  // Let Global Store apply defaults and clamp to limits.
  lease_req.set_ttl_ms(0);

  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_req);
  if (!lease_or.ok()) {
    if (absl::IsAlreadyExists(lease_or.status())) {
      rctx.mark_success();
      return Status::OK;
    }
    return to_grpc_status(lease_or.status());
  }
  const auto& lease_resp = *lease_or;
  if (!lease_resp.acquired()) {
    rctx.mark_success();
    return Status::OK;
  }

  const auto lease = lease_resp.lease();
  const uint64_t lease_generation = lease.lease_generation();
  const std::string lease_token = lease.lease_token();
  const std::string assembly_id = req.assembly_id();
  const std::string layout_id = req.layout_id();

  bool should_start = false;
  {
    absl::MutexLock lock(&seal_mu_);
    should_start = active_seal_operations_.insert(operation_id).second;
  }

  if (should_start) {
    auto* client = d_.global_store_client.get();
    auto executor = d_.async_runtime.blocking_executor();
    executor->add(
        [this, client, operation_id, assembly_id, layout_id, lease_generation, lease_token]() mutable -> void {
          absl::Status final_status = absl::OkStatus();
          auto cleanup = absl::MakeCleanup([this, operation_id]() {
            absl::MutexLock lock(&seal_mu_);
            active_seal_operations_.erase(operation_id);
          });

          auto keepalive_stop = std::make_shared<std::atomic<bool>>(false);
          auto keepalive_exec = d_.async_runtime.blocking_executor();
          auto keepalive = std::make_shared<std::function<void()>>();
          std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
          *keepalive = [client,
                        keepalive_stop,
                        keepalive_exec,
                        keepalive_weak,
                        &timekeeper = d_.async_runtime.timekeeper(),
                        lease_token]() mutable {
            if (keepalive_stop->load(std::memory_order_relaxed)) {
              return;
            }
            timekeeper.after(std::chrono::milliseconds(5000))
                .via(keepalive_exec)
                .thenValue([client, keepalive_stop, lease_token, keepalive_weak](folly::Unit) mutable {
                  if (keepalive_stop->load(std::memory_order_relaxed)) {
                    return;
                  }
                  tensorcast::operation::v1::KeepaliveOperationLeaseRequest req;
                  req.set_lease_token(lease_token);
                  req.set_ttl_ms(0);
                  auto resp_or = client->keepalive_operation_lease(req);
                  if (!resp_or.ok()) {
                    LOG(WARNING) << "keepalive_operation_lease failed for op=" << lease_token << ": "
                                 << resp_or.status();
                  }
                  auto next = keepalive_weak.lock();
                  if (next != nullptr) {
                    (*next)();
                  }
                });
          };
          (*keepalive)();

          tensorcast::daemon::v2::SealAssemblySnapshot snapshot_msg;
          google::protobuf::Any snapshot_any;
          bool snapshot_loaded = false;
          {
            tensorcast::operation::v1::GetOperationRequest get_req;
            get_req.set_operation_id(operation_id);
            auto existing_or = client->get_operation(get_req);
            if (existing_or.ok()) {
              const auto& existing_snapshot = existing_or->snapshot();
              const bool has_snapshot = !existing_snapshot.type_url().empty() || !existing_snapshot.value().empty();
              if (has_snapshot) {
                snapshot_any = existing_snapshot;
                snapshot_loaded = snapshot_any.UnpackTo(&snapshot_msg);
                if (!snapshot_loaded) {
                  final_status = absl::FailedPreconditionError("unsupported seal snapshot payload");
                }
              }
            } else {
              LOG(WARNING) << "get_operation failed while loading seal snapshot (op=" << operation_id
                           << "): " << existing_or.status();
            }
          }

          if (!snapshot_loaded && final_status.ok()) {
            snapshot_msg.set_assembly_id(assembly_id);
            if (!layout_id.empty()) {
              snapshot_msg.set_layout_id(layout_id);
              snapshot_msg.set_assembly_layout_binding_version(0);
            } else {
              auto binding_or = client->get_assembly_layout_binding(assembly_id);
              if (binding_or.ok()) {
                snapshot_msg.set_layout_id(binding_or->layout_id());
                snapshot_msg.set_assembly_layout_binding_version(binding_or->binding_version());
              }
            }

            auto views_or = client->list_views(assembly_id);
            if (!views_or.ok()) {
              final_status = views_or.status();
            } else {
              std::vector<store::components::ViewInfo> views = std::move(*views_or);
              std::sort(views.begin(), views.end(), [](const auto& a, const auto& b) { return a.view_id < b.view_id; });
              for (const auto& view : views) {
                if (view.view_id.empty()) {
                  continue;
                }
                auto* out = snapshot_msg.add_views();
                out->set_view_id(view.view_id);
                auto digest = compute_view_meta_digest(view);
                out->set_meta_digest(digest.data(), static_cast<int>(digest.size()));
              }
            }

            if (final_status.ok()) {
              snapshot_any.PackFrom(snapshot_msg);
              snapshot_loaded = true;
            }
          }

          tensorcast::operation::v1::UpdateOperationRequest running;
          running.set_operation_id(operation_id);
          running.set_lease_generation(lease_generation);
          auto* status = running.mutable_status();
          status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
          status->set_message("sealing");
          status->set_progress(0.0);
          *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
          if (snapshot_loaded) {
            running.mutable_snapshot()->CopyFrom(snapshot_any);
          }
          final_status = client->update_operation(running);
          if (!final_status.ok()) {
            LOG(WARNING) << "update_operation(RUNNING) failed for op=" << operation_id << ": " << final_status;
            keepalive_stop->store(true, std::memory_order_relaxed);
            return;
          }

          std::vector<std::string> allowed_view_ids;
          allowed_view_ids.reserve(static_cast<size_t>(snapshot_msg.views_size()));
          for (const auto& view : snapshot_msg.views()) {
            if (!view.view_id().empty()) {
              allowed_view_ids.push_back(view.view_id());
            }
          }

          if (snapshot_msg.views_size() > 0) {
            auto current_views_or = client->list_views(assembly_id);
            if (!current_views_or.ok()) {
              final_status = current_views_or.status();
            } else {
              absl::flat_hash_map<std::string, std::string> expected;
              expected.reserve(static_cast<size_t>(snapshot_msg.views_size()));
              for (const auto& view : snapshot_msg.views()) {
                expected.emplace(view.view_id(), view.meta_digest());
              }
              for (const auto& view : *current_views_or) {
                auto it = expected.find(view.view_id);
                if (it == expected.end()) {
                  continue;
                }
                auto digest = compute_view_meta_digest(view);
                const std::string computed(reinterpret_cast<const char*>(digest.data()), digest.size());
                if (computed != it->second) {
                  final_status = absl::FailedPreconditionError(
                      absl::StrCat("seal snapshot view metadata mismatch for view_id=", view.view_id));
                  break;
                }
                expected.erase(it);
              }
              if (final_status.ok() && !expected.empty()) {
                final_status = absl::FailedPreconditionError("seal snapshot view missing from current view set");
              }
            }
          }

          if (!final_status.ok()) {
            keepalive_stop->store(true, std::memory_order_relaxed);
            // Fall through to FAILED status update below.
          }

          auto last_progress_ms = std::make_shared<std::atomic<int64_t>>(0);
          auto max_hashed = std::make_shared<std::atomic<uint64_t>>(0);
          auto enable_updates = std::make_shared<std::atomic<bool>>(true);
          store::runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb =
              [client, operation_id, lease_generation, last_progress_ms, max_hashed, enable_updates](
                  uint64_t hashed_leaf_count, uint64_t total_hash_leaves) mutable {
                if (!enable_updates->load(std::memory_order_relaxed) || total_hash_leaves == 0) {
                  return;
                }
                const uint64_t prev_max = max_hashed->load(std::memory_order_relaxed);
                if (hashed_leaf_count <= prev_max && hashed_leaf_count != total_hash_leaves) {
                  return;
                }
                max_hashed->store(std::max(prev_max, hashed_leaf_count), std::memory_order_relaxed);

                const int64_t now_ms = absl::ToUnixMillis(absl::Now());
                const int64_t last_ms = last_progress_ms->load(std::memory_order_relaxed);
                if (hashed_leaf_count != total_hash_leaves && last_ms != 0 && now_ms - last_ms < 1000) {
                  return;
                }
                last_progress_ms->store(now_ms, std::memory_order_relaxed);

                tensorcast::operation::v1::UpdateOperationRequest update;
                update.set_operation_id(operation_id);
                update.set_lease_generation(lease_generation);
                auto* status = update.mutable_status();
                status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
                status->set_message(absl::StrCat("hashing ", hashed_leaf_count, "/", total_hash_leaves));
                status->set_progress(static_cast<double>(hashed_leaf_count) / static_cast<double>(total_hash_leaves));
                *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
                absl::Status st = client->update_operation(update);
                if (!st.ok()) {
                  enable_updates->store(false, std::memory_order_relaxed);
                  LOG(WARNING) << "update_operation(progress) failed for op=" << operation_id << ": " << st;
                }
              };

          const std::vector<std::string>* allowed_ptr = snapshot_loaded ? &allowed_view_ids : nullptr;
          auto seal_or = final_status.ok()
              ? d_.engine.seal_assembly(assembly_id, /*publish_canonical=*/true, std::move(progress_cb), allowed_ptr)
              : absl::StatusOr<store::SealAssemblyResult>(final_status);
          if (!seal_or.ok()) {
            final_status = seal_or.status();
          } else {
            const std::string sealed_artifact_id = seal_or->sealed_artifact_id;
            if (final_status.ok() && !snapshot_msg.layout_id().empty()) {
              final_status = client->attach_layout_to_artifact(sealed_artifact_id, snapshot_msg.layout_id());
            }

            std::optional<tensorcast::layout::v1::LayoutSpec> layout_spec_for_post_seal;
            if (final_status.ok() && !snapshot_msg.layout_id().empty()) {
              auto layout_or = client->get_layout_spec(snapshot_msg.layout_id());
              if (!layout_or.ok()) {
                final_status = layout_or.status();
              } else {
                layout_spec_for_post_seal = layout_or->layout();
                const auto& layout_spec = *layout_spec_for_post_seal;
                const std::string proof_schema_version = layout_spec.proof_schema_version();
                absl::flat_hash_set<std::string> replicated_tensors;
                replicated_tensors.reserve(layout_spec.tensors_size());
                for (const auto& entry : layout_spec.tensors()) {
                  if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
                    replicated_tensors.insert(entry.first);
                  }
                }

                if (!replicated_tensors.empty()) {
                  if (proof_schema_version.empty()) {
                    final_status =
                        absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
                  } else if (proof_schema_version != "v1") {
                    final_status = absl::UnimplementedError("unsupported proof_schema_version");
                  } else {
                    auto index_or = client->get_artifact_index_by_id(sealed_artifact_id);
                    if (!index_or.ok()) {
                      final_status = index_or.status();
                    } else {
                      auto intervals_or = parse_tensor_intervals(*index_or);
                      if (!intervals_or.ok()) {
                        final_status = intervals_or.status();
                      } else {
                        auto devices = d_.engine.get_resident_devices(sealed_artifact_id);
                        auto gpu_it = std::find_if(devices.begin(), devices.end(), [](const store::DeviceKey& d) {
                          return d.type == DeviceType::GPU;
                        });
                        if (gpu_it == devices.end()) {
                          final_status =
                              absl::FailedPreconditionError("sealed artifact GPU replica unavailable for proofs");
                        } else {
                          store::loading::ReplicaKey replica_key;
                          replica_key.artifact_id = sealed_artifact_id;
                          replica_key.view_id = std::nullopt;
                          replica_key.device = *gpu_it;
                          replica_key.replica = 0;

                          auto size_or = d_.engine.get_replica_size(replica_key);
                          auto ptr_or = d_.engine.get_replica_gpu_ptr(replica_key);
                          if (!size_or.ok()) {
                            final_status = size_or.status();
                          } else if (!ptr_or.ok()) {
                            final_status = ptr_or.status();
                          } else {
                            store::loader::GpuMemorySource src(
                                gsl::not_null<void*>{reinterpret_cast<void*>(*ptr_or)},
                                /*device_id=*/gpu_it->ordinal,
                                *size_or);

                            std::vector<tensorcast::global_store::v1::TensorProofCommitmentWrite> writes;
                            for (const auto& interval : *intervals_or) {
                              if (interval.size_bytes == 0) {
                                continue;
                              }
                              if (!replicated_tensors.contains(interval.tensor_name)) {
                                continue;
                              }
                              const uint64_t expected_chunks =
                                  (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
                              for (uint64_t chunk_idx = 0; chunk_idx < expected_chunks; ++chunk_idx) {
                                const uint64_t local_start = chunk_idx * kProofChunkBytesV1;
                                const uint64_t local_end =
                                    std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
                                if (local_end <= local_start) {
                                  continue;
                                }
                                const uint64_t abs_start = interval.offset + local_start;
                                const uint64_t read_len = local_end - local_start;
                                if (read_len > std::numeric_limits<size_t>::max()) {
                                  final_status = absl::OutOfRangeError("proof chunk exceeds host memory limits");
                                  break;
                                }
                                std::vector<uint8_t> buffer(static_cast<size_t>(read_len));
                                auto read_or = src.read_at(abs_start, buffer.data(), static_cast<size_t>(read_len));
                                if (!read_or.ok()) {
                                  final_status = read_or.status();
                                  break;
                                }
                                if (*read_or != buffer.size()) {
                                  final_status = absl::DataLossError("short read while computing MI2 proof digest");
                                  break;
                                }
                                std::vector<uint8_t> digest =
                                    tensorcast::common::sha256_digest_bytes(absl::MakeSpan(buffer));
                                if (digest.size() != 32) {
                                  final_status = absl::InternalError("sha256 digest size mismatch");
                                  break;
                                }
                                tensorcast::global_store::v1::TensorProofCommitmentWrite write;
                                write.set_tensor_name(interval.tensor_name);
                                write.set_proof_chunk_idx(chunk_idx);
                                write.set_digest(digest.data(), static_cast<int>(digest.size()));
                                writes.push_back(std::move(write));
                              }
                              if (!final_status.ok()) {
                                break;
                              }
                            }

                            if (final_status.ok() && !writes.empty()) {
                              constexpr size_t kBatchEntries = 1024;
                              for (size_t i = 0; i < writes.size(); i += kBatchEntries) {
                                tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest write_req;
                                write_req.set_mi2_id(sealed_artifact_id);
                                write_req.set_proof_schema_version(proof_schema_version);
                                const size_t end = std::min(writes.size(), i + kBatchEntries);
                                for (size_t j = i; j < end; ++j) {
                                  *write_req.add_commitments() = writes[j];
                                }
                                auto write_resp_or = client->write_tensor_proof_commitments(write_req);
                                if (!write_resp_or.ok()) {
                                  final_status = write_resp_or.status();
                                  break;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            if (final_status.ok()) {
              const auto& policy = d_.post_seal_policy;
              const bool allow_migration = policy.migrate_views;
              const bool allow_retire = policy.retire_pieces;
              if (allow_retire && !policy.migrate_views && !policy.reuse_views_if_safe) {
                LOG(WARNING) << "post-seal retire_pieces enabled without migrate_views or reuse_views_if_safe; "
                             << "view reads may fail after seal";
              }

              if (allow_migration) {
                auto views_or = client->list_views(assembly_id);
                if (!views_or.ok()) {
                  LOG(WARNING) << "post-seal migrate_views list_views failed for assembly=" << assembly_id << ": "
                               << views_or.status();
                } else {
                  absl::flat_hash_set<std::string> allowed_set;
                  if (!allowed_view_ids.empty()) {
                    allowed_set.reserve(allowed_view_ids.size());
                    for (const auto& id : allowed_view_ids) {
                      if (!id.empty()) {
                        allowed_set.insert(id);
                      }
                    }
                  }
                  absl::flat_hash_set<std::string> expected_set;
                  if (layout_spec_for_post_seal.has_value()) {
                    const auto& expected = layout_spec_for_post_seal->expected_view_ids();
                    expected_set.reserve(static_cast<size_t>(expected.size()));
                    for (const auto& id : expected) {
                      if (!id.empty()) {
                        expected_set.insert(id);
                      }
                    }
                  }

                  const std::vector<std::string>* allowed_ptr = allowed_view_ids.empty() ? nullptr : &allowed_view_ids;
                  if (d_.engine.get_num_gpus() == 0) {
                    LOG(WARNING) << "post-seal migrate_views skipped: no GPU devices available";
                  } else {
                    for (const auto& view : *views_or) {
                      if (view.view_id.empty()) {
                        continue;
                      }
                      if (!allowed_set.empty() && !allowed_set.contains(view.view_id)) {
                        continue;
                      }
                      if (!expected_set.empty() && !expected_set.contains(view.view_id)) {
                        continue;
                      }
                      if (view.view_spec_json.empty()) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (missing view_spec_json)";
                        continue;
                      }
                      if (view.view_size_bytes == 0) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (view_size_bytes=0)";
                        continue;
                      }
                      if (policy.migrate_transpose_only) {
                        auto spec_or = store::view::parse_view_spec_json(view.view_spec_json);
                        if (!spec_or.ok()) {
                          LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                       << " (invalid view_spec_json): " << spec_or.status();
                          continue;
                        }
                        if (!spec_includes_transpose(*spec_or)) {
                          continue;
                        }
                      }

                      const store::DeviceKey target_device = d_.devices.DefaultGpu();
                      auto handle_or = d_.engine.materialize_view_from_assembly(
                          assembly_id,
                          sealed_artifact_id,
                          view.view_id,
                          view.view_spec_json,
                          target_device,
                          store::loading::TransformPlacement::kServer,
                          allowed_ptr);
                      if (!handle_or.ok()) {
                        LOG(WARNING) << "post-seal migrate_views failed for view_id=" << view.view_id << ": "
                                     << handle_or.status();
                        continue;
                      }

                      auto publish_status = d_.engine.register_replica_with_global_store(handle_or->replica_key, {});
                      if (!publish_status.ok() && !absl::IsAlreadyExists(publish_status)) {
                        LOG(WARNING) << "post-seal migrate_views register_replica failed for view_id=" << view.view_id
                                     << ": " << publish_status;
                      }

                      store::components::ViewStateUpdate update;
                      update.artifact_id = sealed_artifact_id;
                      update.view_id = view.view_id;
                      update.view_spec_json = view.view_spec_json;
                      update.view_size_bytes = view.view_size_bytes;
                      if (view.view_data_hash.has_value()) {
                        update.view_data_hash = view.view_data_hash;
                      }
                      update.mark_verified = view.verified_at.has_value();
                      update.canonical_size_bytes = view.canonical_size_bytes;
                      update.canonical_bytes_covered = view.canonical_bytes_covered;
                      update.canonical_ranges = view.canonical_ranges;
                      auto view_status = client->update_artifact_view_state(update);
                      if (!view_status.ok()) {
                        LOG(WARNING) << "post-seal migrate_views update_view_state failed for view_id=" << view.view_id
                                     << ": " << view_status;
                      }
                    }
                  }
                }
              }

              if (allow_retire) {
                const std::string worker_id = d_.identity.worker_id();
                if (!worker_id.empty()) {
                  auto unreg_status = client->unregister_replica_by_worker(assembly_id, worker_id);
                  if (!unreg_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unregister_replica_by_worker failed for assembly="
                                 << assembly_id << ": " << unreg_status;
                  }
                } else {
                  LOG(WARNING) << "post-seal retire_pieces skipped unregister_replica_by_worker: worker_id unavailable";
                }

                std::vector<store::loading::ReplicaKey> to_unload;
                for (const auto& info : d_.engine.get_all_replicas_info()) {
                  if (info.key.artifact_id == assembly_id) {
                    to_unload.push_back(info.key);
                  }
                }
                for (const auto& key : to_unload) {
                  auto unload_status = d_.engine.unload_replica_status(key);
                  if (!unload_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unload_replica failed for key=" << key << ": "
                                 << unload_status;
                  }
                }
              }
            }

            tensorcast::daemon::v2::SealAssemblyResult result_msg;
            auto* artifact = result_msg.mutable_artifact();
            artifact->set_artifact_id(sealed_artifact_id);
            if (!seal_or->index_multihash.empty()) {
              artifact->set_index_multihash(seal_or->index_multihash);
            }
            if (!seal_or->data_multihash.empty()) {
              artifact->set_data_multihash(seal_or->data_multihash);
            }
            if (!seal_or->schema_version.empty()) {
              artifact->set_schema_version(seal_or->schema_version);
            }
            if (!seal_or->encoding.empty()) {
              artifact->set_encoding(seal_or->encoding);
            }
            if (seal_or->total_size > 0) {
              artifact->set_total_size(seal_or->total_size);
            }

            tensorcast::operation::v1::UpdateOperationRequest success;
            success.set_operation_id(operation_id);
            success.set_lease_generation(lease_generation);
            auto* out = success.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
            out->set_message("sealed");
            out->set_progress(1.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            out->mutable_result()->PackFrom(result_msg);
            if (final_status.ok()) {
              final_status = client->update_operation(success);
            }
          }

          if (!final_status.ok()) {
            tensorcast::operation::v1::UpdateOperationRequest failed;
            failed.set_operation_id(operation_id);
            failed.set_lease_generation(lease_generation);
            auto* out = failed.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_FAILED);
            out->set_message("seal failed");
            out->set_progress(0.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            auto* err = out->mutable_error();
            err->set_status_code(absl::StatusCodeToString(final_status.code()));
            err->set_message(std::string(final_status.message()));
            err->set_retryable(retryable_status(final_status));
            absl::Status update_st = client->update_operation(failed);
            if (!update_st.ok()) {
              LOG(WARNING) << "update_operation(FAILED) failed for op=" << operation_id << ": " << update_st;
            }
          }

          keepalive_stop->store(true, std::memory_order_relaxed);
          tensorcast::operation::v1::ReleaseOperationLeaseRequest release;
          release.set_lease_token(lease_token);
          auto release_or = client->release_operation_lease(release);
          if (!release_or.ok()) {
            LOG(WARNING) << "release_operation_lease failed for op=" << operation_id << ": " << release_or.status();
          }
        });
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::get_operation(
    RpcContext& rctx,
    const tensorcast::operation::v1::GetOperationRequest& req,
    tensorcast::operation::v1::GetOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto op_or = d_.global_store_client->get_operation(req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp = std::move(*op_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const uint64_t timeout_ms = req.timeout_ms();
  const absl::Time start = absl::Now();
  const absl::Time deadline = timeout_ms > 0 ? start + absl::Milliseconds(timeout_ms) : absl::InfiniteFuture();

  absl::Duration sleep = absl::Milliseconds(50);
  tensorcast::operation::v1::GetOperationRequest op_req;
  op_req.set_operation_id(req.operation_id());

  while (absl::Now() < deadline) {
    auto op_or = d_.global_store_client->get_operation(op_req);
    if (!op_or.ok()) {
      return to_grpc_status(op_or.status());
    }
    const auto state = op_or->status().state();
    resp.mutable_operation()->Swap(&(*op_or));
    if (state == tensorcast::operation::v1::OPERATION_STATE_SUCCESS ||
        state == tensorcast::operation::v1::OPERATION_STATE_FAILED ||
        state == tensorcast::operation::v1::OPERATION_STATE_CANCELLED) {
      rctx.mark_success();
      return Status::OK;
    }
    absl::SleepFor(sleep);
    sleep = std::min(sleep * 12 / 10, absl::Milliseconds(500));
  }

  auto op_or = d_.global_store_client->get_operation(op_req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp.mutable_operation()->Swap(&(*op_or));
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::confirm(
    RpcContext& rctx,
    const v2::ConfirmReplicaRequest& req,
    v2::ConfirmReplicaResponse& resp) const {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", req.disk_path());
  }
  span->SetAttribute("tc.device.type", static_cast<int64_t>(req.target_device_type()));
  resp.set_disk_path(req.disk_path());

  if (req.replica_uuid().empty()) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  auto entry = d_.sessions.get(req.replica_uuid());
  if (!entry.has_value()) {
    // Parity: unknown replica_uuid → code=0 OK
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  // Wait bounded by gRPC deadline with a 30s hard cap (Confirm has no user timeout)
  using namespace std::chrono;
  const auto wait_ms = ClampToDeadline(rctx.server_context(), milliseconds(30000), milliseconds(30000));
  const absl::Status st = entry->wait_ready(wait_ms);
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, "confirm timeout"};
  }
  if (st.ok()) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_code(1);
  return to_grpc_status(st);
}

grpc::Status MaterializationController::unload(
    RpcContext& rctx,
    const v2::UnloadReplicaRequest& req,
    v2::UnloadReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.disk_path().empty())
      span->SetAttribute("tc.disk.path", req.disk_path());
    if (req.has_pid())
      span->SetAttribute("tc.pid", static_cast<int64_t>(req.pid()));
  }
  resp.set_disk_path(req.disk_path());

  if (req.target_device_type() == v2::DeviceType::DEVICE_TYPE_DISK) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  store::loading::ReplicaKey key;
  if (!req.replica_uuid().empty()) {
    auto entry = d_.sessions.get(req.replica_uuid());
    if (entry.has_value())
      key = entry->key;
  }
  if (key.artifact_id.empty()) {
    if (!req.disk_path().empty()) {
      key.artifact_id = req.disk_path();
      key.device = d_.devices.From(req.target_device_type(), /*uuid=*/"", /*ordinal_hint=*/std::nullopt);
      key.replica = 0;
    } else {
      resp.set_code(0);
      rctx.mark_success();
      return Status::OK;
    }
  }
  if (req.has_pid()) {
    bool lease_released = false;
    if (d_.lifecycle) {
      const auto status = d_.lifecycle->release_use_lease(key, req.pid());
      if (status.ok()) {
        lease_released = true;
      } else if (absl::IsNotFound(status)) {
        VLOG(1) << "release_use_lease not found: artifact_id=" << key.artifact_id << " dev=" << key.device.ordinal
                << " pid=" << req.pid() << " replica_uuid=" << req.replica_uuid();
      } else {
        LOG(WARNING) << "failed to release use lease: artifact_id=" << key.artifact_id << " dev=" << key.device.ordinal
                     << " pid=" << req.pid() << " replica_uuid=" << req.replica_uuid() << " status=" << status;
      }
    }
    if (!lease_released) {
      d_.refs.drop_ref(key, req.pid());
    }
    if (d_.refs.ref_count(key) > 0) {
      resp.set_code(0);
      rctx.mark_success();
      return Status::OK;
    }
  }
  const absl::Status unload_status = d_.engine.unload_replica_status(key);
  if (unload_status.ok()) {
    if (!req.replica_uuid().empty()) {
      const bool erased = d_.sessions.erase(req.replica_uuid());
      if (!erased) {
        VLOG(2) << "unload: session not found for replica_uuid=" << req.replica_uuid();
      }
    }
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_code(1);
  return to_grpc_status(unload_status);
}

grpc::Status MaterializationController::wait_verification(
    RpcContext& rctx,
    const v2::WaitReplicaVerificationRequest& req,
    v2::WaitReplicaVerificationResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.replica_uuid().empty())
      span->SetAttribute("tc.replica.id", req.replica_uuid());
  }

  // If known terminal state, return immediately
  // Access via SessionsService (VerificationTracker behind it)
  // There is no direct getter here; fallback to session lookup then wait
  // Check known terminal state first via tracker
  if (auto known = d_.sessions.get_known(req.replica_uuid()); known.has_value()) {
    resp.set_status(known->first);
    if (!known->second.empty())
      resp.set_err_msg(known->second);
    rctx.mark_success();
    return Status::OK;
  }
  auto entry = d_.sessions.get(req.replica_uuid());
  if (!entry.has_value()) {
    resp.set_status(v2::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
    rctx.mark_success();
    return Status::OK;
  }
  using namespace std::chrono;
  const auto user_ms = milliseconds(req.timeout_ms() > 0 ? req.timeout_ms() : 30000);
  const auto wait_ms = ClampToDeadline(rctx.server_context(), user_ms, milliseconds(30000));
  const absl::Status st = entry->wait_ready(wait_ms);
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, "verification wait timeout"};
  }
  if (st.ok()) {
    resp.set_status(v2::VerificationStatus::VERIFICATION_STATUS_PASSED);
    d_.sessions.update_verification_status(req.replica_uuid(), v2::VerificationStatus::VERIFICATION_STATUS_PASSED);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v2::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp.set_err_msg(std::string(st.message()));
  d_.sessions.update_verification_status(
      req.replica_uuid(), v2::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
  return to_grpc_status(st);
}

} // namespace tensorcast::daemon
