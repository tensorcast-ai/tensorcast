// Copyright (c) 2025, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "google/protobuf/util/time_util.h"
#include "gsl/pointers"
#include "opentelemetry/metrics/provider.h"

#include "core/common/artifact_hash.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "daemon/cuda_ipc_raii.h"
#include "daemon/deadline_utils.h"
#include "daemon/status_utils.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using store::loader::ViewOp;
using store::loader::ViewSpec;
using store::loading::MaterializationSource;
using store::loading::SourcePreference;

bool path_has_prefix(const std::filesystem::path& path, const std::filesystem::path& prefix) {
  auto path_it = path.begin();
  for (auto prefix_it = prefix.begin(); prefix_it != prefix.end(); ++prefix_it, ++path_it) {
    if (path_it == path.end() || *path_it != *prefix_it) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::filesystem::path> normalize_disk_path(std::string_view disk_path) {
  std::error_code ec;
  const auto normalized = std::filesystem::weakly_canonical(disk_path, ec);
  if (!ec) {
    return normalized;
  }
  const std::filesystem::path lex = std::filesystem::path(disk_path).lexically_normal();
  if (lex.empty()) {
    return absl::ErrnoToStatus(ec.value(), "Failed to canonicalize disk_path");
  }
  return lex;
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

struct DescriptorMetadata {
  bool found{false};
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
    v1::MaterializationSource source) {
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

SourcePreference to_hint_preference(v1::SourcePreference preference) {
  switch (preference) {
    case v1::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P:
      return SourcePreference::kPreferP2P;
    case v1::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK:
      return SourcePreference::kPreferDisk;
    case v1::SourcePreference::SOURCE_PREFERENCE_AUTO:
    case v1::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED:
    default:
      return SourcePreference::kAuto;
  }
}

v1::MaterializationSource to_proto_source(MaterializationSource source) {
  switch (source) {
    case MaterializationSource::kDisk:
      return v1::MaterializationSource::MATERIALIZATION_SOURCE_DISK;
    case MaterializationSource::kP2P:
      return v1::MaterializationSource::MATERIALIZATION_SOURCE_P2P;
    case MaterializationSource::kLocalReplica:
      return v1::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA;
    case MaterializationSource::kUnspecified:
    default:
      return v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED;
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

absl::StatusOr<ViewSpec> convert_view_spec(const v1::ViewSpec& proto) {
  ViewSpec spec;
  for (const auto& [tensor_name, ops_proto] : proto.tensors()) {
    store::loader::TensorViewOps ops;
    for (const auto& op_proto : ops_proto.ops()) {
      switch (op_proto.kind_case()) {
        case v1::Op::kNarrow: {
          const auto& narrow = op_proto.narrow();
          store::loader::NarrowOp op{
              .dim = static_cast<int32_t>(narrow.dim()),
              .start = narrow.start(),
              .length = narrow.length(),
          };
          ops.ops.push_back(ViewOp::Narrow(op));
          break;
        }
        case v1::Op::kTranspose: {
          const auto& transpose = op_proto.transpose();
          store::loader::TransposeOp op{
              .dim0 = static_cast<int32_t>(transpose.dim0()),
              .dim1 = static_cast<int32_t>(transpose.dim1()),
          };
          ops.ops.push_back(ViewOp::Transpose(op));
          break;
        }
        case v1::Op::KIND_NOT_SET:
          return absl::InvalidArgumentError("view op kind not set");
      }
    }
    spec.tensors.emplace(tensor_name, std::move(ops));
  }
  return spec;
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
    const v1::MaterializeReplicaRequest& req,
    const std::optional<ViewSpec>& spec) {
  switch (req.placement()) {
    case v1::TransformPlacement::TRANSFORM_PLACEMENT_SERVER:
      return store::loading::TransformPlacement::kServer;
    case v1::TransformPlacement::TRANSFORM_PLACEMENT_CLIENT:
      return store::loading::TransformPlacement::kClient;
    case v1::TransformPlacement::TRANSFORM_PLACEMENT_UNSPECIFIED:
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
    const v1::MaterializeReplicaResponse& v1_resp,
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
    const v1::MaterializeByKeyResponse& v1_resp,
    store::StoreEngine& engine) {
  if (!v1_resp.artifact_id().empty()) {
    return engine.get_canonical_index_by_id(v1_resp.artifact_id());
  }
  return absl::NotFoundError("canonical index JSON unavailable for key materialization");
}

} // namespace

MaterializationController::MaterializationController(Dep d) : d_(std::move(d)) {
  std::error_code ec;
  for (const auto& entry : d_.disk_path_whitelist) {
    if (entry.empty()) {
      continue;
    }
    auto normalized = std::filesystem::weakly_canonical(entry, ec);
    if (ec) {
      ec.clear();
      normalized = entry.lexically_normal();
    }
    if (!normalized.empty()) {
      disk_path_whitelist_.push_back(normalized);
    }
  }
  whitelist_enforced_ = !disk_path_whitelist_.empty();
}

grpc::Status MaterializationController::materialize_replica(
    RpcContext& rctx,
    const v1::MaterializeReplicaRequest& req,
    v1::MaterializeReplicaResponse& resp) {
  auto& span = rctx.span();
  const v1::SourcePreference preference = req.preference();
  const bool prefer_disk = preference == v1::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK;
  const bool prefer_p2p = preference == v1::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P;
  const bool verify_checksums = req.has_verify_checksums() ? req.verify_checksums() : true;

  span->SetAttribute("tc.store.verify_checksums", verify_checksums);

  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.device.uuid", req.device_uuid());
  }
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.size_bytes()));
  span->SetAttribute("tc.store.preference", static_cast<int64_t>(preference));

  using v1::MaterializeReplicaStatus;
  if (d_.is_shutting_down.load()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool request_has_artifact = req.has_artifact_id() && !req.artifact_id().empty();
  const bool request_has_disk = req.has_disk_path() && !req.disk_path().empty();
  if (!request_has_artifact && !request_has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id or disk_path is required"};
  }

  const auto dev = d_.devices.From(req.target_device_type(), req.device_uuid(), std::nullopt);

  std::optional<std::filesystem::path> normalized_disk_path;
  if (request_has_disk) {
    auto normalized_or = normalize_disk_path(req.disk_path());
    if (!normalized_or.ok()) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(normalized_or.status());
    }
    const auto& normalized = *normalized_or;
    if (whitelist_enforced_) {
      bool allowed = false;
      for (const auto& prefix : disk_path_whitelist_) {
        if (path_has_prefix(normalized, prefix)) {
          allowed = true;
          break;
        }
      }
      if (!allowed) {
        record_disk_path_denied();
        LOG(WARNING) << "disk_path not permitted by whitelist: " << normalized;
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return {StatusCode::INVALID_ARGUMENT, "disk_path not permitted by daemon whitelist"};
      }
    }
    normalized_disk_path = normalized;
    resp.set_disk_path(normalized.string());
    if (rctx.allow_high_card_attrs()) {
      span->SetAttribute("tc.disk.path", normalized.string());
    }
  }

  DescriptorMetadata descriptor_meta;
  std::optional<store::loader::IndexInfo> disk_index;
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
  const std::string resolved_artifact_id = has_artifact ? *artifact_id : std::string();
  if (has_artifact) {
    span->SetAttribute("tc.artifact.id", resolved_artifact_id);
    resp.set_artifact_id(resolved_artifact_id);
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

  switch (req.view_identity_case()) {
    case v1::MaterializeReplicaRequest::kView: {
      if (!has_artifact) {
        return {StatusCode::INVALID_ARGUMENT, "view spec requires artifact_id for canonical planning"};
      }
      auto spec_or = convert_view_spec(req.view());
      if (!spec_or.ok()) {
        return to_grpc_status(spec_or.status());
      }
      view_spec = std::move(*spec_or);
      auto index_or = d_.engine.get_canonical_index_by_id(resolved_artifact_id);
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
      } else {
        // Identity view collapses to canonical path
        view_spec.reset();
        view_plan.reset();
        canonical_index_json.reset();
      }
      break;
    }
    case v1::MaterializeReplicaRequest::kViewId: {
      if (!req.view_id().empty()) {
        if (!has_artifact) {
          return {StatusCode::INVALID_ARGUMENT, "view_id requires artifact_id for routing"};
        }
        request_view_id = req.view_id();
      }
      break;
    }
    case v1::MaterializeReplicaRequest::VIEW_IDENTITY_NOT_SET:
      break;
  }
  if (request_view_id.has_value()) {
    span->SetAttribute("tc.view.id", *request_view_id);
  }

  // Artifact LIP fast path: try cross-device consumption
  const bool view_requested = view_spec.has_value() || request_view_id.has_value();
  if (has_artifact && !view_requested) {
    auto satisfied = d_.lip.try_satisfy_from_lip(
        resolved_artifact_id,
        dev.ordinal,
        [&](const store::loading::ReplicaKey& rkey) {
          if (!req.replica_uuid().empty()) {
            d_.sessions.put_with_verification(req.replica_uuid(), rkey, nullptr);
          }
          if (req.pid() > 0) {
            d_.refs.add_ref(rkey, req.pid());
            if (d_.lifecycle && rkey.device.type == DeviceType::GPU) {
              SessionLifecycleManager::ReplicaSubject subj{
                  .artifact_id = rkey.artifact_id, .device_id = rkey.device.ordinal};
              auto lid_or = d_.lifecycle->create_use_lease(subj, req.pid());
              if (!lid_or.ok()) {
                LOG(WARNING) << "create_use_lease failed (LIP path): artifact_id=" << rkey.artifact_id
                             << " dev=" << rkey.device.ordinal << ": " << lid_or.status();
                try {
                  static auto meter =
                      opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
                  static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
                  ctr->Add(1.0);
                } catch (...) {
                }
              }
              // TTL prefetch pins via request flag removed; only UseLease is created.
            }
          }
        },
        resp.mutable_mem_handle());
    if (!satisfied.ok()) {
      // Same-device denial should fall back to the engine path just like MaterializeByKey.
      if (!absl::IsFailedPrecondition(satisfied.status())) {
        resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
        return to_grpc_status(satisfied.status());
      }
    } else if (*satisfied) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
      resp.set_source(v1::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA);
      span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
      rctx.mark_success();
      return Status::OK;
    }
  }

  // Engine-backed materialization
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.verify = verify_checksums ? store::loading::MaterializeHints::Verify::CHECKSUM
                                  : store::loading::MaterializeHints::Verify::NONE;
  hints.source_preference = to_hint_preference(preference);
  if (has_artifact)
    hints.artifact_id = resolved_artifact_id;
  if (has_disk)
    hints.disk_path = normalized_disk_path->string();
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
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  resp.set_source(to_proto_source(handle.source));
  span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
  if (!req.replica_uuid().empty()) {
    d_.sessions.put_with_verification(req.replica_uuid(), handle.replica_key, handle.ready_signal);
  }
  if (req.pid() > 0) {
    d_.refs.add_ref(handle.replica_key, req.pid());
    if (d_.lifecycle && handle.replica_key.device.type == DeviceType::GPU) {
      SessionLifecycleManager::ReplicaSubject subj{
          .artifact_id = handle.replica_key.artifact_id, .device_id = handle.replica_key.device.ordinal};
      auto lid_or = d_.lifecycle->create_use_lease(subj, req.pid());
      if (!lid_or.ok()) {
        LOG(WARNING) << "create_use_lease failed (engine path): artifact_id=" << handle.replica_key.artifact_id
                     << " dev=" << handle.replica_key.device.ordinal << ": " << lid_or.status();
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
          ctr->Add(1.0);
        } catch (...) {
        }
      }
      // TTL prefetch pins via request flag removed; only UseLease is created.
    }
  }
  if (has_disk)
    resp.set_disk_path(normalized_disk_path->string());
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  if (handle.cuda_ipc_handle.is_valid()) {
    resp.mutable_mem_handle()->set_cuda_ipc_handle(
        handle.cuda_ipc_handle.bytes.data(), handle.cuda_ipc_handle.bytes.size());
  }
  if (handle.view_index_json.has_value()) {
    resp.set_view_index_json(*handle.view_index_json);
  }
  if (resp.view_index_json().empty() && handle.source == store::loading::MaterializationSource::kDisk) {
    // Prefer disk-local canonical index to avoid Global Store dependency for disk loads.
    if (disk_index.has_value()) {
      resp.set_view_index_json(disk_index->canonical_index_json);
    } else if (normalized_disk_path.has_value()) {
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
    if (resp.view_index_json().empty()) {
      auto index_or = d_.engine.get_canonical_index_by_id(handle.replica_key.artifact_id);
      if (index_or.ok()) {
        resp.set_view_index_json(*index_or);
        VLOG(1) << "MaterializationController: filled view_index_json from engine for disk artifact_id="
                << handle.replica_key.artifact_id;
      } else {
        LOG(WARNING) << "Failed to fetch canonical index for disk materialization response: " << index_or.status();
      }
    }
  }
  if (handle.view_data_hash.has_value()) {
    resp.set_view_data_hash(*handle.view_data_hash);
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::materialize_by_key(
    RpcContext& rctx,
    const v1::MaterializeByKeyRequest& req,
    v1::MaterializeByKeyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());

  using v1::MaterializeReplicaStatus;
  if (d_.is_shutting_down.load()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.key().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "key is required"};
  }

  auto mapping_or = d_.engine.resolve_key_mapping(req.key());
  if (!mapping_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(mapping_or.status());
  }
  const auto& mapping = *mapping_or;
  span->SetAttribute("tc.artifact.id", mapping.artifact_id);

  // Try LIP fast path first
  {
    auto satisfied = d_.lip.try_satisfy_from_lip(
        mapping.artifact_id,
        req.device_id(),
        [&](const store::loading::ReplicaKey& rkey) {
          if (!req.replica_uuid().empty()) {
            d_.sessions.put_with_verification(req.replica_uuid(), rkey, nullptr);
          }
          if (req.pid() > 0) {
            d_.refs.add_ref(rkey, req.pid());
            if (d_.lifecycle && rkey.device.type == DeviceType::GPU) {
              SessionLifecycleManager::ReplicaSubject subj{
                  .artifact_id = rkey.artifact_id, .device_id = rkey.device.ordinal};
              auto lid_or = d_.lifecycle->create_use_lease(subj, req.pid());
              if (!lid_or.ok()) {
                LOG(WARNING) << "create_use_lease failed (LIP by-key): artifact_id=" << rkey.artifact_id
                             << " dev=" << rkey.device.ordinal << ": " << lid_or.status();
                try {
                  static auto meter =
                      opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
                  static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
                  ctr->Add(1.0);
                } catch (...) {
                }
              }
              // MaterializeByKey: no TTL prefetch; only UseLease is created.
            }
          }
        },
        resp.mutable_mem_handle());
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
      resp.set_used_disk_path(mapping.disk_path);
      resp.set_source(v1::MaterializationSource::MATERIALIZATION_SOURCE_LOCAL_REPLICA);
      span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
      rctx.mark_success();
      return Status::OK;
    }
  }

  // Engine path
  // Validate device_id
  if (req.device_id() < 0 || req.device_id() >= d_.engine.get_num_gpus()) {
    return {StatusCode::INVALID_ARGUMENT, "invalid device_id"};
  }
  const auto dev = store::DeviceRegistry::instance().gpu_key(req.device_id());
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.artifact_id = mapping.artifact_id;
  if (!mapping.disk_path.empty())
    hints.disk_path = mapping.disk_path;

  auto result = d_.engine.materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, hints);
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  if (!req.replica_uuid().empty()) {
    d_.sessions.put_with_verification(req.replica_uuid(), handle.replica_key, handle.ready_signal);
  }
  if (req.pid() > 0) {
    d_.refs.add_ref(handle.replica_key, req.pid());
    if (d_.lifecycle && handle.replica_key.device.type == DeviceType::GPU) {
      SessionLifecycleManager::ReplicaSubject subj{
          .artifact_id = handle.replica_key.artifact_id, .device_id = handle.replica_key.device.ordinal};
      auto lid_or = d_.lifecycle->create_use_lease(subj, req.pid());
      if (!lid_or.ok()) {
        LOG(WARNING) << "create_use_lease failed (engine by-key): artifact_id=" << handle.replica_key.artifact_id
                     << " dev=" << handle.replica_key.device.ordinal << ": " << lid_or.status();
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
          ctr->Add(1.0);
        } catch (...) {
        }
      }
      // MaterializeByKey: no TTL prefetch; only UseLease is created.
    }
  }
  if (handle.cuda_ipc_handle.is_valid()) {
    resp.mutable_mem_handle()->set_cuda_ipc_handle(
        handle.cuda_ipc_handle.bytes.data(), handle.cuda_ipc_handle.bytes.size());
  }
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_artifact_id(mapping.artifact_id);
  resp.set_used_disk_path(mapping.disk_path);
  resp.set_source(to_proto_source(handle.source));
  span->SetAttribute("tc.store.source", static_cast<int64_t>(resp.source()));
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::materialize_replica_v2(
    RpcContext& rctx,
    const v2::MaterializeReplicaRequest& req,
    v2::MaterializeReplicaResponse& resp) {
  v1::MaterializeReplicaRequest v1_req;
  v1_req.set_verify_checksums(true);
  if (req.has_artifact_id() && !req.artifact_id().empty()) {
    v1_req.set_artifact_id(req.artifact_id());
  }
  if (req.has_disk_fallback() && !req.disk_fallback().disk_path().empty()) {
    v1_req.set_disk_path(req.disk_fallback().disk_path());
    v1_req.set_verify_checksums(req.disk_fallback().verify_checksums());
  }
  v1_req.set_replica_uuid(req.replica_uuid());
  v1_req.set_device_uuid(req.device_uuid());
  v1_req.set_target_device_type(req.target_device_type());
  v1_req.set_pinned_allocation_timeout_ms(req.pinned_allocation_timeout_ms());
  v1_req.set_pid(req.pid());
  v1_req.set_size_bytes(req.size_bytes());
  v1_req.set_preference(req.preference());
  v1_req.set_placement(req.placement());
  switch (req.view_identity_case()) {
    case v2::MaterializeReplicaRequest::kView:
      v1_req.mutable_view()->CopyFrom(req.view());
      break;
    case v2::MaterializeReplicaRequest::kViewId:
      v1_req.set_view_id(req.view_id());
      break;
    case v2::MaterializeReplicaRequest::VIEW_IDENTITY_NOT_SET:
      break;
  }

  v1::MaterializeReplicaResponse v1_resp;
  auto status = materialize_replica(rctx, v1_req, v1_resp);
  if (!status.ok()) {
    return status;
  }

  if (!v1_resp.artifact_id().empty()) {
    resp.set_artifact_id(v1_resp.artifact_id());
  } else if (req.has_artifact_id()) {
    resp.set_artifact_id(req.artifact_id());
  }
  resp.set_disk_path(v1_resp.disk_path());
  resp.set_status(v1_resp.status());
  resp.set_source(v1_resp.source());
  if (v1_resp.has_mem_handle()) {
    resp.mutable_mem_handle()->CopyFrom(v1_resp.mem_handle());
  }
  if (!v1_resp.view_index_json().empty()) {
    resp.set_view_index_bytes(v1_resp.view_index_json());
  }

  auto layout_or = resolve_layout_json(v1_resp, req, d_.engine);
  if (!layout_or.ok()) {
    return to_grpc_status(layout_or.status());
  }
  const uint64_t generation = compute_generation_from_index(*layout_or);
  resp.set_canonical_index_bytes(*layout_or);
  resp.set_generation(generation);

  std::optional<DescriptorBuildResult> desc_result;
  if (req.view_identity_case() == v2::MaterializeReplicaRequest::kView && v1_resp.view_index_json().empty()) {
    auto spec_or = convert_view_spec(req.view());
    if (!spec_or.ok()) {
      return to_grpc_status(spec_or.status());
    }
    auto plan_or = store::StoreEngine::compute_view_plan(*layout_or, *spec_or);
    if (!plan_or.ok()) {
      return to_grpc_status(plan_or.status());
    }
    if (!plan_or->is_identity) {
      auto desc_or = build_descriptors_from_view_plan(*plan_or, req.tensor_names(), req.device_uuid());
      if (!desc_or.ok()) {
        return to_grpc_status(desc_or.status());
      }
      desc_result = std::move(*desc_or);
    }
  }
  if (!desc_result.has_value()) {
    auto desc_or = build_descriptors_from_index(*layout_or, req.tensor_names(), req.device_uuid());
    if (!desc_or.ok()) {
      return to_grpc_status(desc_or.status());
    }
    desc_result = std::move(*desc_or);
  }
  for (auto& desc : desc_result->descriptors) {
    *resp.add_payloads() = std::move(desc);
  }
  if (!req.view_subset_hash().empty() || !desc_result->included_names.empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!req.view_subset_hash().empty()) {
      subset->set_subset_hash(req.view_subset_hash());
    }
    for (const auto& name : desc_result->included_names) {
      subset->add_tensor_names(name);
    }
  }
  if (!req.wait_for_completion() && !req.replica_uuid().empty()) {
    auto* ticket = resp.mutable_ticket();
    ticket->set_replica_uuid(req.replica_uuid());
    ticket->set_status(v1_resp.status());
    *ticket->mutable_created_at() = google::protobuf::util::TimeUtil::GetCurrentTime();
  }
  return status;
}

grpc::Status MaterializationController::materialize_into_target(
    RpcContext& rctx,
    const v2::MaterializeIntoTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  auto& span = rctx.span();
  if (d_.is_shutting_down.load()) {
    record_materialize_into_target(
        "error", "unavailable", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool has_artifact_id = req.has_artifact_id() && !req.artifact_id().empty();
  const bool has_key = req.has_key() && !req.key().empty();
  if (has_key) {
    record_materialize_into_target(
        "error", "key_not_supported", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "key-based requests not supported for MaterializeIntoTarget"};
  }
  if (!has_artifact_id) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required for MaterializeIntoTarget"};
  }
  if (req.has_disk_fallback() && req.disk_fallback().disk_path().empty()) {
    record_materialize_into_target(
        "error", "disk_fallback_empty", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "disk_fallback.disk_path must not be empty"};
  }
  if (!req.has_target_layout()) {
    record_materialize_into_target(
        "error", "layout_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout is required"};
  }
  if (req.tensor_names_size() > 0 || !req.view_subset_hash().empty()) {
    record_materialize_into_target(
        "error", "subset_not_supported", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "tensor_names/view_subset_hash not supported for MaterializeIntoTarget"};
  }
  if (req.view_identity_case() != v2::MaterializeIntoTargetRequest::VIEW_IDENTITY_NOT_SET) {
    record_materialize_into_target(
        "error", "view_not_supported", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "view/view_id not supported for MaterializeIntoTarget"};
  }
  if (req.device_uuid().empty()) {
    record_materialize_into_target(
        "error", "device_uuid_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "device_uuid is required"};
  }
  if (req.pid() <= 0) {
    record_materialize_into_target(
        "error", "owner_pid_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "pid is required for MaterializeIntoTarget"};
  }

  const auto& layout = req.target_layout();
  if (layout.layout_kind() != v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "layout_kind_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Only LAYOUT_KIND_COALESCED_UNSPECIFIED is supported"};
  }
  if (layout.index_kind() != v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "index_kind_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Only INDEX_KIND_CANONICAL_UNSPECIFIED is supported"};
  }
  if (layout.storages_size() != 1) {
    record_materialize_into_target(
        "error", "multi_storage", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "COALESCED layouts must include exactly one storage entry"};
  }
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS &&
      layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "tensor_spec_kind_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Unsupported tensor_spec_kind for MaterializeIntoTarget"};
  }

  const auto& storage = layout.storages(0);
  if (storage.storage_source_case() != v1::StorageEntry::kVramRegionId) {
    record_materialize_into_target(
        "error", "storage_not_region", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Target storage must reference a vram_region_id"};
  }
  if (!storage.has_region_base_offset()) {
    record_materialize_into_target(
        "error", "region_base_offset_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "region_base_offset is required for region-backed targets"};
  }

  const auto device = d_.devices.From(v1::DeviceType::DEVICE_TYPE_GPU, req.device_uuid(), std::nullopt);
  if (storage.device_id() != device.ordinal) {
    record_materialize_into_target(
        "error", "device_uuid_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "storage.device_id does not match device_uuid"};
  }

  auto canonical_json_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!canonical_json_or.ok()) {
    record_materialize_into_target(
        "error", "index_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_json_or.status());
  }
  auto index_table_or = parse_canonical_index(*canonical_json_or);
  if (!index_table_or.ok()) {
    record_materialize_into_target(
        "error", "index_parse_failed", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(index_table_or.status());
  }
  const CanonicalIndexTable& index_table = *index_table_or;
  const uint64_t logical_total_size = index_table.logical_total_size;

  auto offsets_or = resolve_target_offsets(layout);
  if (!offsets_or.ok()) {
    record_materialize_into_target(
        "error", "offsets_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(offsets_or.status());
  }
  const auto& offsets = *offsets_or;
  if (offsets.empty()) {
    record_materialize_into_target(
        "error", "offsets_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout offsets are required"};
  }
  if (offsets.size() != index_table.entries.size()) {
    record_materialize_into_target(
        "error", "tensor_name_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include every canonical tensor"};
  }
  for (const auto& entry : offsets) {
    auto it = index_table.entries.find(entry.name);
    if (it == index_table.entries.end()) {
      record_materialize_into_target(
          "error", "tensor_name_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes unknown tensor name"};
    }
    if (entry.logical_length != it->second.logical_length) {
      record_materialize_into_target(
          "error", "layout_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout logical_length mismatch"};
    }
    if (entry.storage_offset != it->second.logical_offset) {
      record_materialize_into_target(
          "error", "offset_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout storage_offset must match logical offset"};
    }
    if (entry.storage_id != storage.storage_id()) {
      record_materialize_into_target(
          "error", "storage_id_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return {StatusCode::INVALID_ARGUMENT, "target_layout storage_id mismatch"};
    }
  }
  if (storage.storage_length() != logical_total_size) {
    record_materialize_into_target(
        "error", "storage_length_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "storage_length must cover full logical space"};
  }

  if (rctx.server_context().IsCancelled()) {
    record_materialize_into_target("error", "cancelled", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::CANCELLED, "request cancelled before transfer"};
  }

  auto region_desc_or = d_.regions.acquire(storage.vram_region_id(), req.pid());
  if (!region_desc_or.ok()) {
    const absl::Status& st = region_desc_or.status();
    const bool poisoned = absl::IsFailedPrecondition(st) && st.message() == "region is poisoned";
    record_materialize_into_target(
        "error",
        poisoned ? "region_poisoned" : "region_missing",
        v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(st);
  }
  auto region_desc = *region_desc_or;
  if (region_desc.device_id != storage.device_id()) {
    record_materialize_into_target(
        "error", "device_uuid_mismatch", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    d_.regions.release(storage.vram_region_id()).IgnoreError();
    return {StatusCode::FAILED_PRECONDITION, "region device does not match storage device"};
  }
  const uint64_t region_end = storage.region_base_offset() + storage.storage_length();
  if (region_end > region_desc.size_bytes) {
    record_materialize_into_target("error", "bounds", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    d_.regions.release(storage.vram_region_id()).IgnoreError();
    return {StatusCode::FAILED_PRECONDITION, "region-backed storage exceeds region bounds"};
  }

  struct RegionReleaseGuard {
    IpcRegionRegistry& registry;
    std::string region_id;
    bool active{true};

    ~RegionReleaseGuard() {
      if (active) {
        (void)registry.release(region_id);
      }
    }
  } guard{d_.regions, storage.vram_region_id(), true};

  auto handle_or = d_.regions.get_handle_bytes(storage.vram_region_id());
  if (!handle_or.ok()) {
    record_materialize_into_target(
        "error", "region_missing", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(handle_or.status());
  }
  auto map_or = CudaIpcMapping::open(*handle_or, cudaIpcMemLazyEnablePeerAccess);
  if (!map_or.ok()) {
    record_materialize_into_target(
        "error", "map_failed", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(map_or.status());
  }

  store::loading::MaterializeHints hints;
  hints.artifact_id = req.artifact_id();
  if (req.has_disk_fallback() && !req.disk_fallback().disk_path().empty()) {
    hints.disk_path = req.disk_fallback().disk_path();
  }
  hints.source_preference = to_hint_preference(req.preference());
  hints.verify = store::loading::MaterializeHints::Verify::NONE;

  record_materialize_into_target_verification_skipped();

  void* region_base_ptr = static_cast<uint8_t*>(map_or->get()) + static_cast<uint64_t>(storage.region_base_offset());
  const uint64_t generation = compute_generation_from_index(*canonical_json_or);
  auto result_or = d_.engine.materialize_into_target(
      device, gsl::not_null<void*>{region_base_ptr}, logical_total_size, *canonical_json_or, generation, hints);
  if (!result_or.ok()) {
    if (absl::IsDataLoss(result_or.status())) {
      d_.regions.mark_poisoned(storage.vram_region_id()).IgnoreError();
      record_materialize_into_target(
          "error", "transfer_failed", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    } else {
      record_materialize_into_target(
          "error", "transfer_error", v1::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    }
    return to_grpc_status(result_or.status());
  }

  resp.set_artifact_id(req.artifact_id());
  resp.set_status(v1::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_source(to_proto_source(result_or->source));
  resp.set_canonical_index_bytes(*canonical_json_or);
  resp.set_generation(generation);
  record_materialize_into_target("ok", "ok", resp.source());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::materialize_by_key_v2(
    RpcContext& rctx,
    const v2::MaterializeByKeyRequest& req,
    v2::MaterializeByKeyResponse& resp) {
  v1::MaterializeByKeyRequest v1_req;
  v1_req.set_key(req.key());
  v1_req.set_device_id(req.device_id());
  v1_req.set_pinned_allocation_timeout_ms(req.pinned_allocation_timeout_ms());
  v1_req.set_replica_uuid(req.replica_uuid());
  v1_req.set_pid(req.pid());

  v1::MaterializeByKeyResponse v1_resp;
  auto status = materialize_by_key(rctx, v1_req, v1_resp);
  if (!status.ok()) {
    return status;
  }

  resp.set_status(v1_resp.status());
  resp.set_artifact_id(v1_resp.artifact_id());
  resp.set_used_disk_path(v1_resp.used_disk_path());
  resp.set_source(v1_resp.source());
  if (v1_resp.has_mem_handle()) {
    resp.mutable_mem_handle()->CopyFrom(v1_resp.mem_handle());
  }

  auto layout_or = resolve_layout_json_by_key(v1_resp, d_.engine);
  if (!layout_or.ok()) {
    return to_grpc_status(layout_or.status());
  }
  const uint64_t generation = compute_generation_from_index(*layout_or);
  resp.set_canonical_index_bytes(*layout_or);
  resp.set_generation(generation);

  auto desc_or = build_descriptors_from_index(*layout_or, req.tensor_names(), /*device_uuid=*/"");
  if (!desc_or.ok()) {
    return to_grpc_status(desc_or.status());
  }
  for (auto& desc : desc_or->descriptors) {
    *resp.add_payloads() = std::move(desc);
  }
  if (!req.view_subset_hash().empty() || !desc_or->included_names.empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!req.view_subset_hash().empty()) {
      subset->set_subset_hash(req.view_subset_hash());
    }
    for (const auto& name : desc_or->included_names) {
      subset->add_tensor_names(name);
    }
  }
  if (!req.wait_for_completion() && !req.replica_uuid().empty()) {
    auto* ticket = resp.mutable_ticket();
    ticket->set_replica_uuid(req.replica_uuid());
    ticket->set_status(v1_resp.status());
    *ticket->mutable_created_at() = google::protobuf::util::TimeUtil::GetCurrentTime();
  }
  // Fill view index bytes with resolved canonical to aid clients that expect a layout hint.
  resp.set_view_index_bytes(*layout_or);
  return status;
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
  if (d_.is_shutting_down.load()) {
    record_disk_resolution_outcome("unavailable");
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto normalized_or = normalize_disk_path(req.disk_path());
  if (!normalized_or.ok()) {
    record_disk_resolution_outcome("invalid_argument");
    return to_grpc_status(normalized_or.status());
  }
  const auto& normalized = *normalized_or;
  if (whitelist_enforced_) {
    bool allowed = false;
    for (const auto& prefix : disk_path_whitelist_) {
      if (path_has_prefix(normalized, prefix)) {
        allowed = true;
        break;
      }
    }
    if (!allowed) {
      record_disk_path_denied();
      LOG(WARNING) << "disk_path not permitted by whitelist: " << normalized;
      record_disk_resolution_outcome("whitelist_denied");
      return {StatusCode::INVALID_ARGUMENT, "disk_path not permitted by daemon whitelist"};
    }
  }
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
    const v1::GetArtifactIndexByIdRequest& req,
    v1::GetArtifactIndexByIdResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());

  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (d_.is_shutting_down.load()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  auto bytes_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!bytes_or.ok())
    return to_grpc_status(bytes_or.status());
  resp.set_tensor_index_data(*bytes_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::confirm(
    RpcContext& rctx,
    const v1::ConfirmReplicaRequest& req,
    v1::ConfirmReplicaResponse& resp) const {
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
    const v1::UnloadReplicaRequest& req,
    v1::UnloadReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.disk_path().empty())
      span->SetAttribute("tc.disk.path", req.disk_path());
    if (req.has_pid())
      span->SetAttribute("tc.pid", static_cast<int64_t>(req.pid()));
  }
  resp.set_disk_path(req.disk_path());

  if (req.target_device_type() == v1::DeviceType::DEVICE_TYPE_DISK) {
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
    d_.refs.drop_ref(key, req.pid());
    if (d_.lifecycle && key.device.type == DeviceType::GPU) {
      SessionLifecycleManager::ReplicaSubject subj{.artifact_id = key.artifact_id, .device_id = key.device.ordinal};
      const auto status = d_.lifecycle->release_use_lease(subj, req.pid());
      if (!status.ok()) {
        LOG(ERROR) << "failed to release use lease: " << status;
      }
    }
    if (d_.refs.ref_count(key) > 0) {
      resp.set_code(0);
      rctx.mark_success();
      return Status::OK;
    }
  }
  const int rc = d_.engine.unload_replica(key);
  if (rc == 0) {
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
  return {StatusCode::INTERNAL, absl::StrFormat("unload_replica() returned %d", rc)};
}

grpc::Status MaterializationController::wait_verification(
    RpcContext& rctx,
    const v1::WaitReplicaVerificationRequest& req,
    v1::WaitReplicaVerificationResponse& resp) {
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
    resp.set_status(v1::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
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
    resp.set_status(v1::VerificationStatus::VERIFICATION_STATUS_PASSED);
    d_.sessions.update_verification_status(req.replica_uuid(), v1::VerificationStatus::VERIFICATION_STATUS_PASSED);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v1::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp.set_err_msg(std::string(st.message()));
  d_.sessions.update_verification_status(
      req.replica_uuid(), v1::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
  return to_grpc_status(st);
}

} // namespace tensorcast::daemon
