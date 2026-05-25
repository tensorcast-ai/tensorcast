// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_payload_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "google/protobuf/util/time_util.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon::materialization_payload {

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
  if (!v1_resp.view_index_bytes().empty()) {
    return v1_resp.view_index_bytes();
  }
  std::string artifact_id = v1_resp.artifact_id();
  if (v2_req.has_selection() && !v2_req.selection().artifact_id().empty()) {
    artifact_id = v2_req.selection().artifact_id();
  }
  const std::string disk_path = v1_resp.disk_path();

  auto read_from_disk = [&]() -> absl::StatusOr<std::string> {
    if (disk_path.empty()) {
      return absl::NotFoundError("disk_path is empty");
    }
    auto local_or = store::loader::read_from_artifact_dir(disk_path, /*target_device_id=*/0);
    if (!local_or.ok()) {
      return local_or.status();
    }
    return local_or->canonical_index_json;
  };

  if (artifact_id.empty()) {
    if (!disk_path.empty()) {
      return read_from_disk();
    }
    return absl::NotFoundError("canonical index JSON unavailable for materialization response");
  }

  auto index_or = engine.get_canonical_index_by_id(artifact_id);
  if (index_or.ok()) {
    return std::move(index_or).value();
  }
  if (!disk_path.empty()) {
    auto disk_or = read_from_disk();
    if (disk_or.ok()) {
      return std::move(disk_or).value();
    }
    return disk_or.status();
  }
  return index_or.status();
}

template <typename ResponseT>
absl::Status populate_materialize_payloads_impl(
    ResponseT& resp,
    std::string_view layout_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid,
    std::string_view view_subset_hash,
    bool wait_for_completion,
    std::string_view replica_uuid,
    const std::string* ticket_device_uuid,
    const std::optional<store::loader::ViewPlan>& view_plan,
    bool prefer_view_plan,
    bool fill_view_index_bytes) {
  const uint64_t generation = compute_generation_from_index(layout_json);
  resp.set_canonical_index_bytes(std::string(layout_json));
  resp.set_generation(generation);
  if (fill_view_index_bytes && resp.view_index_bytes().empty()) {
    resp.set_view_index_bytes(std::string(layout_json));
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
      subset->set_subset_hash(std::string(view_subset_hash));
    }
    for (const auto& name : desc_result->included_names) {
      subset->add_tensor_names(name);
    }
  }
  if (!wait_for_completion && !replica_uuid.empty()) {
    auto* ticket = resp.mutable_ticket();
    ticket->set_replica_uuid(std::string(replica_uuid));
    ticket->set_status(resp.status());
    if (ticket_device_uuid != nullptr && !ticket_device_uuid->empty()) {
      ticket->set_device_uuid(*ticket_device_uuid);
    }
    *ticket->mutable_created_at() = google::protobuf::util::TimeUtil::GetCurrentTime();
  }
  return absl::OkStatus();
}

absl::Status populate_materialize_payloads(
    v2::MaterializeReplicaResponse& resp,
    std::string_view layout_json,
    const google::protobuf::RepeatedPtrField<std::string>& tensor_names,
    std::string_view device_uuid,
    std::string_view view_subset_hash,
    bool wait_for_completion,
    std::string_view replica_uuid,
    const std::string* ticket_device_uuid,
    const std::optional<store::loader::ViewPlan>& view_plan,
    bool prefer_view_plan,
    bool fill_view_index_bytes) {
  return populate_materialize_payloads_impl(
      resp,
      layout_json,
      tensor_names,
      device_uuid,
      view_subset_hash,
      wait_for_completion,
      replica_uuid,
      ticket_device_uuid,
      view_plan,
      prefer_view_plan,
      fill_view_index_bytes);
}

} // namespace tensorcast::daemon::materialization_payload
