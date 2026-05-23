// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/target_materialization_service.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/common/selection_identity.h"
#include "core/store/components/endpoint_id.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/loaders/p2p_loader.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "core/store/runtime/ingestion/source_bound_strategy_planner.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/controllers/materialization_index_source_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_payload_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/materialization_request_common_utils.h"
#include "daemon/service/controllers/materialization_target_plan_utils.h"
#include "daemon/service/controllers/materialization_target_storage_utils.h"
#include "daemon/service/controllers/serving_artifact_manifest_utils.h"
#include "daemon/util/deadline_utils.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
namespace global_store = tensorcast::global_store::v1;
using status_utils::to_grpc_status;

namespace {

using materialization_index_source::compute_target_layout_multihash;
using materialization_index_source::load_canonical_index_with_disk_fallback;
using materialization_index_source::load_descriptor_metadata;
using materialization_index_source::parse_mi2_data_multihash;
using materialization_index_source::TargetLayoutSpan;
using materialization_layout::resolve_target_offsets;
using materialization_payload::compute_generation_from_index;
using materialization_policy::apply_group_realization_begin_context_to_transport_context;
using materialization_policy::apply_operation_transport_context;
using materialization_policy::apply_request_context_to_hints;
using materialization_policy::begin_or_join_group_realization_if_enabled;
using materialization_policy::default_collective_policy_for_mapped_target;
using materialization_policy::NormalizedMaterializationRequestContext;
using materialization_policy::OperationTransportContext;
using materialization_policy::resolve_group_realization_transport_context;
using materialization_policy::resolve_materialization_request_context;
using materialization_policy::resolve_transform_placement;
using materialization_policy::validate_group_realization_staged_publish_supported;
using materialization_request_common::resolve_artifact_and_disk_source;
using materialization_target_plan::build_mapped_target_materialization_plan;
using materialization_target_plan::build_resolved_mapped_materialization_plan;
using materialization_target_plan::build_target_materialization_plan;
using materialization_target_plan::MappedTargetMaterializationPlan;
using materialization_target_plan::TargetMaterializationPlan;
using store::loading::MaterializationSource;

constexpr std::string_view kProgressiveBytePrefixCoverageOrder{"tensorcast.progressive.byte_prefix.v1"};
constexpr uint32_t kProgressiveTargetMaxLocalAttempts{1024};

std::optional<std::string> progressive_mi2_index_multihash(std::string_view artifact_id) {
  constexpr std::string_view kPrefix{"mi2:"};
  if (!artifact_id.starts_with(kPrefix)) {
    return std::nullopt;
  }
  const std::string_view suffix = artifact_id.substr(kPrefix.size());
  const size_t separator = suffix.find(':');
  if (separator == std::string_view::npos || separator == 0) {
    return std::nullopt;
  }
  return std::string(suffix.substr(0, separator));
}

std::string digest_to_string(std::string_view payload) {
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::string byte_prefix_coverage_order_hash() {
  return digest_to_string(kProgressiveBytePrefixCoverageOrder);
}

void append_fingerprint_field(std::string& payload, std::string_view value) {
  const uint64_t size = value.size();
  for (int shift = 56; shift >= 0; shift -= 8) {
    payload.push_back(static_cast<char>((size >> shift) & 0xff));
  }
  payload.append(value.data(), value.size());
}

void append_fingerprint_uint64(std::string& payload, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    payload.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void append_fingerprint_uint32(std::string& payload, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    payload.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

std::optional<global_store::ProgressiveCoverageIdentity> build_progressive_target_identity(
    std::string_view artifact_id,
    const tensorcast::common::v1::ArtifactSelection& resolved_selection,
    const v2::TargetLayout& layout,
    const std::optional<std::string>& resolved_view_id,
    bool has_subset,
    bool has_view_transform,
    const std::optional<materialization_policy::GroupRealizationBeginContext>& group_context) {
  if (layout.index_kind() != v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED || resolved_view_id.has_value() ||
      has_subset || has_view_transform) {
    return std::nullopt;
  }
  auto index_multihash = progressive_mi2_index_multihash(artifact_id);
  if (!index_multihash.has_value() || resolved_selection.logical_layout_hash().empty() ||
      resolved_selection.selection_hash().empty()) {
    return std::nullopt;
  }

  global_store::ProgressiveCoverageIdentity identity;
  identity.set_artifact_id(std::string(artifact_id));
  identity.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  identity.set_selection_hash(resolved_selection.selection_hash());
  identity.set_logical_layout_hash(resolved_selection.logical_layout_hash());
  identity.mutable_hash_space()->mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  identity.mutable_hash_space()->set_canonical_index_multihash(*index_multihash);
  identity.set_coverage_order_hash(byte_prefix_coverage_order_hash());
  if (group_context.has_value() && !group_context->version_set.version_set_id().empty() &&
      !group_context->part_id.empty()) {
    identity.set_group_version_set_id(group_context->version_set.version_set_id());
    identity.set_group_part_id(group_context->part_id);
  }
  return identity;
}

uint64_t progressive_deadline_unix_nanos(std::chrono::milliseconds request_budget) {
  if (request_budget.count() <= 0) {
    return 0;
  }
  const int64_t nanos = absl::ToUnixNanos(absl::Now() + absl::Milliseconds(request_budget.count()));
  if (nanos <= 0) {
    return 0;
  }
  return static_cast<uint64_t>(nanos);
}

std::string progressive_request_fingerprint(
    const global_store::ProgressiveCoverageIdentity& identity,
    std::string_view materialization_attempt_id,
    uint64_t next_unit,
    uint32_t retry_attempt) {
  std::string payload;
  append_fingerprint_field(payload, identity.artifact_id());
  append_fingerprint_uint32(payload, static_cast<uint32_t>(identity.byte_space().kind()));
  append_fingerprint_field(payload, identity.byte_space().id());
  append_fingerprint_field(payload, identity.selection_hash());
  append_fingerprint_field(payload, identity.logical_layout_hash());
  append_fingerprint_uint32(payload, static_cast<uint32_t>(identity.hash_space().byte_space().kind()));
  append_fingerprint_field(payload, identity.hash_space().byte_space().id());
  append_fingerprint_field(payload, identity.hash_space().canonical_index_multihash());
  append_fingerprint_field(payload, identity.coverage_order_hash());
  append_fingerprint_field(payload, identity.group_version_set_id());
  append_fingerprint_field(payload, identity.group_part_id());
  append_fingerprint_field(payload, materialization_attempt_id);
  append_fingerprint_uint64(payload, next_unit);
  append_fingerprint_uint32(payload, retry_attempt);
  return digest_to_string(payload);
}

absl::StatusOr<store::loading::IntoTargetLayout> slice_target_layout(
    const store::loading::IntoTargetLayout& layout,
    uint64_t start_byte,
    uint64_t end_byte_exclusive) {
  if (start_byte >= end_byte_exclusive) {
    return absl::InvalidArgumentError("progressive target segment must be non-empty");
  }
  if (end_byte_exclusive > layout.total_size) {
    return absl::OutOfRangeError("progressive target segment exceeds target layout");
  }
  store::loading::IntoTargetLayout sliced;
  sliced.total_size = end_byte_exclusive - start_byte;

  uint64_t cursor = 0;
  for (const auto& storage : layout.storages) {
    const uint64_t storage_start = cursor;
    if (storage.length > std::numeric_limits<uint64_t>::max() - cursor) {
      return absl::OutOfRangeError("target layout storage offset overflow");
    }
    const uint64_t storage_end = cursor + storage.length;
    cursor = storage_end;
    const uint64_t overlap_start = std::max(start_byte, storage_start);
    const uint64_t overlap_end = std::min(end_byte_exclusive, storage_end);
    if (overlap_start >= overlap_end) {
      continue;
    }
    auto* base = static_cast<uint8_t*>(storage.base_ptr.get());
    auto* segment_base = base + (overlap_start - storage_start);
    sliced.storages.push_back(
        store::loading::IntoTargetStorage{
            .base_ptr = gsl::not_null<void*>{segment_base},
            .length = overlap_end - overlap_start,
            .stable_backing = storage.stable_backing,
            .stable_backing_keepalive = storage.stable_backing_keepalive,
            .keepalive = storage.keepalive,
        });
  }
  if (cursor != layout.total_size) {
    return absl::InvalidArgumentError("target layout total_size does not match storage lengths");
  }
  if (sliced.storages.empty()) {
    return absl::InvalidArgumentError("progressive target segment does not overlap target layout");
  }
  return sliced;
}

store::loader::ByteRangeMap build_progressive_segment_map(uint64_t source_offset, uint64_t length) {
  store::loader::ByteRangeMap map;
  map.total_bytes = length;
  map.num_sources = 1;
  map.segments.push_back(
      store::loader::ByteRangeSegment{
          .kind = store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = length,
          .src_offset = source_offset,
          .source_index = 0,
      });
  return map;
}

common::memory::MemoryLocation progressive_memory_type(tensorcast::common::v1::MemoryType type) {
  switch (type) {
    case tensorcast::common::v1::MEMORY_TYPE_GPU:
      return common::memory::MemoryLocation::GPU;
    case tensorcast::common::v1::MEMORY_TYPE_RAM:
      return common::memory::MemoryLocation::CPU;
    case tensorcast::common::v1::MEMORY_TYPE_UNSPECIFIED:
    default:
      return common::memory::MemoryLocation::NONE;
  }
}

absl::StatusOr<store::P2PSource> build_progressive_p2p_source(
    const global_store::ProgressiveSourceAssignment& assignment,
    const store::DeviceKey& target_device,
    std::string_view requester_node_id,
    std::string_view artifact_id,
    std::chrono::milliseconds request_budget,
    store::StoreEngine& engine) {
  if (!assignment.has_source_memory_info()) {
    return absl::FailedPreconditionError("progressive assignment is missing source memory info");
  }
  const auto& info = assignment.source_memory_info();
  const common::memory::MemoryLocation memory_type = progressive_memory_type(info.memory_type());
  if (memory_type == common::memory::MemoryLocation::NONE) {
    return absl::FailedPreconditionError("progressive assignment source memory type is unsupported");
  }
  if (info.node_address().empty() || info.node_port() == 0 || info.memory_size() == 0) {
    return absl::FailedPreconditionError("progressive assignment source memory endpoint is incomplete");
  }
  if (!info.has_transport()) {
    return absl::FailedPreconditionError("progressive assignment source memory info is missing transport metadata");
  }

  std::vector<std::string> memory_keys;
  const auto& transport = info.transport();
  memory_keys.assign(transport.remote_memory_keys().begin(), transport.remote_memory_keys().end());
  if (memory_keys.empty() || memory_keys.size() != static_cast<size_t>(transport.buffer_sizes_size())) {
    return absl::FailedPreconditionError("progressive assignment source memory keys and buffer sizes must match");
  }

  std::vector<size_t> buffer_sizes;
  buffer_sizes.reserve(static_cast<size_t>(transport.buffer_sizes_size()));
  uint64_t buffer_total = 0;
  for (const auto size : transport.buffer_sizes()) {
    if (size == 0) {
      return absl::FailedPreconditionError("progressive assignment source buffer size must be non-zero");
    }
    if (size > std::numeric_limits<size_t>::max()) {
      return absl::OutOfRangeError("progressive assignment source buffer size exceeds platform size_t");
    }
    if (buffer_total > std::numeric_limits<uint64_t>::max() - size) {
      return absl::OutOfRangeError("progressive assignment source buffer sizes overflow");
    }
    buffer_total += size;
    buffer_sizes.push_back(static_cast<size_t>(size));
  }
  if (buffer_total != info.memory_size()) {
    return absl::FailedPreconditionError("progressive assignment source buffer sizes do not match memory size");
  }
  std::string verification_json = transport.verification_json();

  auto comm_manager = engine.get_shared_comm_manager();
  if (!comm_manager->is_enabled()) {
    return absl::FailedPreconditionError("progressive target materialization requires communicator");
  }

  const common::memory::MemoryLocation target_memory_type =
      target_device.type == DeviceType::GPU ? common::memory::MemoryLocation::GPU : common::memory::MemoryLocation::CPU;
  const int target_device_id = target_device.type == DeviceType::GPU ? target_device.ordinal : 0;
  store::P2PSource source;
  source.size_bytes = info.memory_size();
  source.ip = info.node_address();
  source.port = static_cast<uint16_t>(info.node_port());
  source.local_endpoint_id =
      store::components::derive_endpoint_id(requester_node_id, target_memory_type, target_device_id);
  source.remote_endpoint_id =
      store::components::derive_endpoint_id(info.node_id(), memory_type, static_cast<int>(info.device_id()));
  source.memory_keys = std::move(memory_keys);
  source.buf_sizes = std::move(buffer_sizes);
  source.enable_checksum = true;
  source.location.type = memory_type;
  source.location.device_id = static_cast<int>(info.device_id());
  source.verification_json = std::move(verification_json);
  source.request_budget = request_budget;
  source.artifact_id = std::string(artifact_id);
  source.transport_request_id = assignment.assignment_id();
  source.comm_engine = comm_manager->get_shared_engine();
  source.routing_context = comm_manager->routing_context();
  return source;
}

void complete_progressive_assignment(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    std::string_view assignment_id,
    global_store::ProgressiveAssignmentState outcome,
    std::string_view detail = {}) {
  if (client == nullptr || !client->is_connected() || assignment_id.empty()) {
    return;
  }
  global_store::CompleteProgressiveAssignmentRequest request;
  request.set_assignment_id(std::string(assignment_id));
  request.set_outcome(outcome);
  if (!detail.empty()) {
    request.set_outcome_detail(std::string(detail));
  }
  auto response_or = client->complete_progressive_assignment(request);
  if (!response_or.ok()) {
    LOG(WARNING) << "CompleteProgressiveAssignment failed for assignment_id=" << assignment_id << ": "
                 << response_or.status();
    return;
  }
  if (response_or->status() != global_store::STATUS_OK && response_or->status() != global_store::STATUS_NOT_FOUND) {
    LOG(WARNING) << "CompleteProgressiveAssignment returned status=" << static_cast<int>(response_or->status())
                 << " assignment_id=" << assignment_id;
  }
}

void retire_progressive_coverage_after_read_failure(
    const std::shared_ptr<store::components::IGlobalStoreClient>& client,
    const global_store::ProgressiveSourceAssignment& assignment,
    const absl::Status& status) {
  if (client == nullptr || !client->is_connected() || assignment.coverage_id().empty()) {
    return;
  }
  global_store::RetireProgressiveCoverageRequest request;
  request.set_coverage_id(assignment.coverage_id());
  request.set_source_export_generation(assignment.source_export_generation());
  request.set_state(global_store::PROGRESSIVE_COVERAGE_STATE_FAILED);
  request.set_reason(status.ToString());
  auto response_or = client->retire_progressive_coverage(request);
  if (!response_or.ok()) {
    LOG(WARNING) << "RetireProgressiveCoverage failed for coverage_id=" << assignment.coverage_id() << ": "
                 << response_or.status();
  }
}

bool is_active_progressive_assignment_state(global_store::ProgressiveAssignmentState state) {
  return state == global_store::PROGRESSIVE_ASSIGNMENT_STATE_CLAIMED ||
      state == global_store::PROGRESSIVE_ASSIGNMENT_STATE_READING;
}

absl::StatusOr<std::optional<store::loading::MaterializeIntoTargetResult>> try_progressive_materialize_into_target(
    const TargetMaterializationService::Dep& dep,
    const store::DeviceKey& device,
    const store::loading::IntoTargetLayout& target_layout,
    const global_store::ProgressiveCoverageIdentity& identity,
    const store::loading::MaterializeHints& base_hints,
    std::string_view artifact_id,
    std::string_view daemon_id,
    std::string_view worker_id,
    std::string_view requester_source_domain,
    std::chrono::milliseconds request_budget,
    grpc::ServerContext& server_context) {
  if (!dep.progressive_replication.enabled || dep.global_store_client == nullptr ||
      !dep.global_store_client->is_connected()) {
    return std::nullopt;
  }
  if (daemon_id.empty() || worker_id.empty() || requester_source_domain.empty()) {
    return std::nullopt;
  }
  const std::string materialization_attempt_id =
      absl::StrCat(dep.daemon_session_id, ":", artifact_id, ":", device.ordinal, ":", absl::ToUnixNanos(absl::Now()));
  const uint64_t deadline_unix_nanos = progressive_deadline_unix_nanos(request_budget);
  const uint64_t total_size = target_layout.total_size;
  uint64_t next_unit = 0;
  uint32_t local_attempts = 0;
  bool progressive_attempted = false;
  bool target_may_be_dirty = false;
  store::loading::MaterializeIntoTargetResult aggregate;
  aggregate.source = MaterializationSource::kP2P;
  aggregate.requested_bytes = total_size;

  while (next_unit < total_size) {
    if (server_context.IsCancelled()) {
      return absl::CancelledError("request cancelled during progressive target materialization");
    }
    if (local_attempts >= kProgressiveTargetMaxLocalAttempts) {
      if (target_may_be_dirty) {
        return absl::DataLossError("progressive target materialization stopped after partial target writes");
      }
      return absl::ResourceExhaustedError("progressive target materialization exceeded local assignment attempts");
    }
    const std::string fingerprint =
        progressive_request_fingerprint(identity, materialization_attempt_id, next_unit, local_attempts);
    global_store::FindProgressiveSourceRequest request;
    request.mutable_identity()->CopyFrom(identity);
    request.set_next_unit(next_unit);
    request.set_max_units(0);
    request.set_requester_daemon_id(std::string(daemon_id));
    request.set_requester_worker_id(std::string(worker_id));
    request.set_requester_source_domain(std::string(requester_source_domain));
    request.set_request_fingerprint(fingerprint);
    request.set_deadline_unix_nanos(deadline_unix_nanos);
    request.set_requester_materialization_attempt_id(materialization_attempt_id);

    auto claim_or = dep.global_store_client->find_progressive_source(request);
    if (!claim_or.ok()) {
      return claim_or.status();
    }
    if (claim_or->status() == global_store::STATUS_NOT_FOUND || !claim_or->has_assignment() ||
        claim_or->assignment().assignment_id().empty()) {
      if (!progressive_attempted) {
        return std::nullopt;
      }
      if (target_may_be_dirty) {
        return absl::DataLossError(
            absl::StrCat(
                "progressive source unavailable after partial target writes; bytes=",
                next_unit,
                " reason=",
                claim_or->no_eligible_reason()));
      }
      if (next_unit == 0) {
        return std::nullopt;
      }
      return absl::UnavailableError(
          absl::StrCat(
              "progressive source unavailable after ", next_unit, " bytes; reason=", claim_or->no_eligible_reason()));
    }
    if (claim_or->status() != global_store::STATUS_OK) {
      return absl::UnavailableError(
          absl::StrCat("FindProgressiveSource returned status=", static_cast<int>(claim_or->status())));
    }

    const auto& assignment = claim_or->assignment();
    if (!is_active_progressive_assignment_state(assignment.state())) {
      const std::string detail = absl::StrCat(
          "FindProgressiveSource replayed terminal assignment state=",
          global_store::ProgressiveAssignmentState_Name(assignment.state()));
      if (target_may_be_dirty) {
        return absl::DataLossError(absl::StrCat(detail, " after partial target writes; bytes=", next_unit));
      }
      return std::nullopt;
    }
    progressive_attempted = true;
    if (assignment.start_unit() != next_unit || assignment.start_byte() != next_unit ||
        assignment.end_unit_exclusive() <= assignment.start_unit() ||
        assignment.end_byte_exclusive() <= assignment.start_byte() ||
        assignment.end_unit_exclusive() != assignment.end_byte_exclusive()) {
      complete_progressive_assignment(
          dep.global_store_client,
          assignment.assignment_id(),
          global_store::PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
          "assignment range does not match byte-prefix cursor");
      return absl::FailedPreconditionError("progressive assignment range does not match byte-prefix cursor");
    }
    if (assignment.end_byte_exclusive() > total_size) {
      complete_progressive_assignment(
          dep.global_store_client,
          assignment.assignment_id(),
          global_store::PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
          "assignment exceeds target size");
      return absl::OutOfRangeError("progressive assignment exceeds target size");
    }

    auto sliced_or = slice_target_layout(target_layout, assignment.start_byte(), assignment.end_byte_exclusive());
    if (!sliced_or.ok()) {
      complete_progressive_assignment(
          dep.global_store_client,
          assignment.assignment_id(),
          global_store::PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
          sliced_or.status().ToString());
      return sliced_or.status();
    }
    const uint64_t segment_len = assignment.end_byte_exclusive() - assignment.start_byte();
    auto p2p_source_or = build_progressive_p2p_source(
        assignment, device, requester_source_domain, artifact_id, request_budget, dep.engine);
    if (!p2p_source_or.ok()) {
      complete_progressive_assignment(
          dep.global_store_client,
          assignment.assignment_id(),
          global_store::PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
          p2p_source_or.status().ToString());
      retire_progressive_coverage_after_read_failure(dep.global_store_client, assignment, p2p_source_or.status());
      local_attempts += 1;
      continue;
    }

    auto hints = base_hints;
    hints.transport_request_id = assignment.assignment_id();
    auto loader = std::make_unique<store::P2PLoader>(std::move(*p2p_source_or));
    target_may_be_dirty = true;
    auto segment_map = build_progressive_segment_map(assignment.start_byte(), segment_len);
    auto sources_or = store::runtime::ingestion::open_single_loader_sources(
        std::move(loader), segment_map, "progressive_target_materialization");
    if (!sources_or.ok()) {
      const std::string detail = sources_or.status().ToString();
      complete_progressive_assignment(
          dep.global_store_client,
          assignment.assignment_id(),
          global_store::PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
          detail);
      retire_progressive_coverage_after_read_failure(dep.global_store_client, assignment, sources_or.status());
      local_attempts += 1;
      continue;
    }
    auto segment_result_or = dep.engine.materialize_mapped_sources_into_target(
        device, *sliced_or, std::move(*sources_or), segment_map, hints, MaterializationSource::kP2P);
    if (!segment_result_or.ok()) {
      const std::string detail = segment_result_or.status().ToString();
      complete_progressive_assignment(
          dep.global_store_client,
          assignment.assignment_id(),
          global_store::PROGRESSIVE_ASSIGNMENT_STATE_FAILED,
          detail);
      retire_progressive_coverage_after_read_failure(dep.global_store_client, assignment, segment_result_or.status());
      local_attempts += 1;
      continue;
    }

    complete_progressive_assignment(
        dep.global_store_client, assignment.assignment_id(), global_store::PROGRESSIVE_ASSIGNMENT_STATE_SUCCEEDED);
    aggregate.committed_bytes += segment_result_or->committed_bytes;
    aggregate.fallback_bytes += segment_result_or->fallback_bytes;
    aggregate.residual_bytes += segment_result_or->residual_bytes;
    aggregate.actual_collective_committed_bytes += segment_result_or->actual_collective_committed_bytes;
    aggregate.actual_local_typed_bytes += segment_result_or->actual_local_typed_bytes;
    aggregate.actual_generic_backend_bytes += segment_result_or->actual_generic_backend_bytes;
    aggregate.collective_unique_source_bytes += segment_result_or->collective_unique_source_bytes;
    aggregate.collective_peer_transfer_bytes += segment_result_or->collective_peer_transfer_bytes;
    aggregate.collective_peak_temporary_bytes =
        std::max(aggregate.collective_peak_temporary_bytes, segment_result_or->collective_peak_temporary_bytes);
    aggregate.collective_batch_count += segment_result_or->collective_batch_count;
    aggregate.collective_dedup_saving_bytes += segment_result_or->collective_dedup_saving_bytes;
    aggregate.collective_handled = aggregate.collective_handled || segment_result_or->collective_handled;
    aggregate.direct_write_supported = aggregate.direct_write_supported || segment_result_or->direct_write_supported;
    aggregate.source_ordered = aggregate.source_ordered || segment_result_or->source_ordered;
    if (aggregate.dominant_executor.empty()) {
      aggregate.dominant_executor = segment_result_or->dominant_executor;
    }
    next_unit = assignment.end_unit_exclusive();
    local_attempts += 1;
  }

  aggregate.committed_bytes = total_size;
  aggregate.selection_reason = "progressive_byte_prefix";
  return aggregate;
}

std::string mint_publication_id() {
  thread_local absl::BitGen bitgen;
  std::string raw;
  raw.resize(16);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen, 0u, 256u));
  }
  return absl::BytesToHexString(raw);
}

bool selection_requires_tensor_aware_metadata(const tensorcast::common::v1::ArtifactSelection& selection) {
  return selection.has_view_spec() || !selection.view_id().empty() || selection.tensor_names_size() > 0 ||
      !selection.view_subset_hash().empty();
}

bool is_byte_only_disk_metadata(const std::optional<store::loading::DiskMetadata>& disk_metadata) {
  return disk_metadata.has_value() && disk_metadata->tensor_aware.has_value() && !*disk_metadata->tensor_aware;
}

std::string artifact_id_kind_attr(std::string_view artifact_id) {
  switch (common::infer_artifact_id_kind(artifact_id)) {
    case common::ArtifactIdKind::kMi2:
      return "mi2";
    case common::ArtifactIdKind::kCgid:
      return "cgid";
    case common::ArtifactIdKind::kMsa1:
      return "msa1";
    case common::ArtifactIdKind::kUnspecified:
    default:
      return "unknown";
  }
}

std::string disk_metadata_capability_attr(const std::optional<store::loading::DiskMetadata>& disk_metadata) {
  if (!disk_metadata.has_value() || !disk_metadata->tensor_aware.has_value()) {
    return "unspecified";
  }
  return *disk_metadata->tensor_aware ? "tensor_aware" : "byte_only";
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

struct TargetMaterializationCommonContext {
  NormalizedMaterializationRequestContext request_context;
  OperationTransportContext transport_context;
  std::optional<materialization_policy::GroupRealizationBeginContext> group_begin_context;
  tensorcast::common::v1::ArtifactSelection effective_selection;
  std::string resolved_artifact_id;
  bool gs_connected{false};
  std::optional<std::filesystem::path> normalized_disk_path;
};

std::chrono::milliseconds resolve_target_request_budget(const grpc::ServerContext& server_context) {
  using clock = std::chrono::system_clock;
  constexpr std::chrono::milliseconds kDefaultBudget{600000};
  constexpr std::chrono::milliseconds kHardCap{1800000};
  const auto grpc_deadline = server_context.deadline();
  if (grpc_deadline != clock::time_point::max()) {
    const auto now = clock::now();
    if (grpc_deadline <= now) {
      return std::chrono::milliseconds(0);
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(grpc_deadline - now);
    if (remaining.count() <= 0) {
      return std::chrono::milliseconds(1);
    }
    return std::min(remaining, kHardCap);
  }
  return std::min(kDefaultBudget, kHardCap);
}

absl::StatusOr<std::optional<store::loading::DiskMetadata>> build_target_disk_metadata(
    const std::optional<std::filesystem::path>& normalized_disk_path,
    std::string_view resolved_artifact_id,
    int device_ordinal,
    ArtifactSourceRegistry& disk_imports) {
  (void)device_ordinal;
  std::optional<store::loading::DiskMetadata> disk_metadata;
  if (normalized_disk_path.has_value()) {
    auto descriptor_or = load_descriptor_metadata(*normalized_disk_path);
    if (!descriptor_or.ok()) {
      return descriptor_or.status();
    }
    auto mounted_metadata_or = materialization_disk_resolve::build_mounted_source_metadata(*normalized_disk_path);
    if (!mounted_metadata_or.ok()) {
      return mounted_metadata_or.status();
    }
    store::loading::DiskMetadata metadata;
    metadata.descriptor_present = descriptor_or->found;
    metadata.schema_version = descriptor_or->schema_version;
    metadata.index_multihash = descriptor_or->index_multihash;
    metadata.data_multihash = descriptor_or->data_multihash;
    metadata.canonical_index_json = mounted_metadata_or->index_info.canonical_index_json;
    if (mounted_metadata_or->index_info.source_index_json.has_value()) {
      metadata.source_index_json = mounted_metadata_or->index_info.source_index_json;
    }
    if (!mounted_metadata_or->canonical_index_multihash.empty()) {
      metadata.index_multihash = mounted_metadata_or->canonical_index_multihash;
    }
    if (mounted_metadata_or->exact_size_bytes > 0) {
      metadata.logical_total_size = mounted_metadata_or->exact_size_bytes;
    }
    if (mounted_metadata_or->index_info.source_total_size_bytes > 0) {
      metadata.source_total_size_bytes = mounted_metadata_or->index_info.source_total_size_bytes;
    } else if (mounted_metadata_or->exact_size_bytes > 0) {
      metadata.source_total_size_bytes = mounted_metadata_or->exact_size_bytes;
    }
    metadata.is_safetensors =
        mounted_metadata_or->format_kind == materialization_disk_resolve::MountedSourceFormatKind::kSafetensors;
    metadata.tensor_aware = mounted_metadata_or->metadata_capability ==
        materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware;
    disk_metadata = std::move(metadata);
  }
  if (auto local_import = disk_imports.lookup_binding(resolved_artifact_id); local_import.has_value()) {
    if (!disk_metadata.has_value()) {
      disk_metadata = store::loading::DiskMetadata{};
    }
    auto& metadata = *disk_metadata;
    metadata.descriptor_present = metadata.descriptor_present || local_import->descriptor_present;
    if (!metadata.canonical_index_json.has_value() && !local_import->canonical_index_json.empty()) {
      metadata.canonical_index_json = local_import->canonical_index_json;
    }
    if (!metadata.source_index_json.has_value() && local_import->source_index_json.has_value()) {
      metadata.source_index_json = *local_import->source_index_json;
    }
    if (!metadata.index_multihash.has_value() && local_import->index_multihash.has_value()) {
      metadata.index_multihash = *local_import->index_multihash;
    }
    if (!metadata.data_multihash.has_value() && local_import->data_multihash.has_value()) {
      metadata.data_multihash = *local_import->data_multihash;
    }
    if (!metadata.tensor_aware.has_value()) {
      metadata.tensor_aware = local_import->tensor_aware_metadata;
    }
  }
  return disk_metadata;
}

template <typename RequestT>
absl::StatusOr<TargetMaterializationCommonContext> prepare_target_materialization_common(
    const RequestT& req,
    std::string_view peer,
    std::string_view rpc_name,
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    std::string_view daemon_id,
    std::string_view daemon_session_id,
    std::string_view worker_id,
    ArtifactSourceRegistry& disk_imports,
    const std::filesystem::path& storage_path) {
  TargetMaterializationCommonContext context;
  std::string_view operation_id;
  if constexpr (requires {
                  req.has_operation_id();
                  req.operation_id();
                }) {
    if (req.has_operation_id() && !req.operation_id().empty()) {
      operation_id = req.operation_id();
    }
  }
  const v2::GroupRealizationOptions* group_realization = nullptr;
  if constexpr (requires {
                  req.has_group_realization();
                  req.group_realization();
                }) {
    if (req.has_group_realization()) {
      group_realization = &req.group_realization();
    }
  }
  auto transport_context_or = resolve_group_realization_transport_context(operation_id, group_realization);
  if (!transport_context_or.ok()) {
    record_materialize_into_target(
        "error", "group_realization_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return transport_context_or.status();
  }
  auto staged_publish_status = validate_group_realization_staged_publish_supported(
      group_realization,
      /*staged_publish_supported=*/false);
  if (!staged_publish_status.ok()) {
    record_materialize_into_target(
        "error",
        "group_realization_staging_unsupported",
        v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return staged_publish_status;
  }
  context.transport_context = std::move(*transport_context_or);
  auto request_context_or = resolve_materialization_request_context(
      req.has_source_policy() ? &req.source_policy() : nullptr, context.transport_context.execution_topology);
  if (!request_context_or.ok()) {
    record_materialize_into_target(
        "error", "policy_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return request_context_or.status();
  }
  context.request_context = std::move(*request_context_or);
  if (!is_loopback_grpc_peer(peer)) {
    record_materialize_into_target(
        "error", "non_loopback_peer", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::PermissionDeniedError(std::format("{} is local-only (loopback/UDS)", rpc_name));
  }

  const bool group_realization_enabled = group_realization != nullptr && group_realization->enabled();
  if (!req.has_selection() && !group_realization_enabled) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("selection is required for {}", rpc_name));
  }
  if (req.has_selection()) {
    context.effective_selection = req.selection();
  }
  auto begin_context_or = begin_or_join_group_realization_if_enabled(
      global_store_client, group_realization, daemon_id, daemon_session_id, worker_id);
  if (!begin_context_or.ok()) {
    record_materialize_into_target(
        "error", "group_realization_begin_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return begin_context_or.status();
  }
  if (begin_context_or->has_value()) {
    context.group_begin_context = **begin_context_or;
    context.effective_selection = context.group_begin_context->part_selection;
    apply_group_realization_begin_context_to_transport_context(
        *context.group_begin_context, &context.transport_context);
  }
  if (context.effective_selection.artifact_id().empty()) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("selection.artifact_id is required for {}", rpc_name));
  }

  auto resolution_or = resolve_artifact_and_disk_source(
      global_store_client,
      &disk_imports,
      storage_path,
      context.effective_selection.artifact_id(),
      context.request_context.retrieval_policy.allow_disk,
      /*allow_local_import_fallback=*/true,
      /*loopback_peer=*/true);
  if (!resolution_or.ok()) {
    record_materialize_into_target(
        "error", "binding_error", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return resolution_or.status();
  }
  context.resolved_artifact_id = std::move(resolution_or->resolved_artifact_id);
  context.gs_connected = resolution_or->gs_connected;
  context.normalized_disk_path = std::move(resolution_or->normalized_disk_path);
  if (!req.has_target_layout()) {
    record_materialize_into_target(
        "error", "layout_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError("target_layout is required");
  }
  if (req.device_uuid().empty()) {
    record_materialize_into_target(
        "error", "device_uuid_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError("device_uuid is required");
  }
  if (req.pid() <= 0) {
    record_materialize_into_target(
        "error", "owner_pid_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return absl::InvalidArgumentError(std::format("pid is required for {}", rpc_name));
  }
  return context;
}

} // namespace

TargetMaterializationService::TargetMaterializationService(Dep d)
    : d_(std::move(d)),
      capability_tokens_(d_.capability_tokens),
      target_publish_service_(
          TargetPublishService::Dep{
              .lip_manager = d_.lip_manager,
              .bindings = d_.bindings,
              .devices = d_.devices,
              .identity = d_.identity,
              .lifecycle = d_.lifecycle,
              .lifecycle_kernel = d_.lifecycle_kernel,
              .async_runtime = d_.async_runtime,
              .shutdown_signal = d_.shutdown_signal,
              .global_store_client = d_.global_store_client,
              .capability_tokens = d_.capability_tokens,
              .max_concurrency = d_.max_concurrency,
              .await_state_sync_barrier = d_.await_state_sync_barrier,
              .progressive_replication = d_.progressive_replication,
              .daemon_id = d_.daemon_id,
              .daemon_session_id = d_.daemon_session_id,
          }) {
  if (!d_.storage_path.empty()) {
    std::error_code ec;
    storage_path_ = std::filesystem::weakly_canonical(d_.storage_path, ec);
    if (ec) {
      ec.clear();
      storage_path_ = d_.storage_path.lexically_normal();
    }
  }
}

absl::StatusOr<TargetPublicationRegistry::Record> TargetMaterializationService::insert_target_publication_for_testing(
    TargetPublicationRegistry::Record record) {
  if (!record.binding_id.empty()) {
    auto binding = std::make_shared<BindingRegistry::Record>();
    binding->binding_id = record.binding_id;
    binding->binding_layout_id = record.binding_layout_id;
    binding->owner_pid = record.owner_pid;
    binding->creator_pid = record.owner_pid;
    binding->device_uuid = record.device_uuid;
    binding->state = v2::BINDING_STATE_READY_ARTIFACT;
    binding->current_artifact_id = record.selection.artifact_id();
    binding->current_selection = record.selection;
    binding->current_binding_value_id = record.binding_value_id;
    binding->seal_generation = record.seal_generation;
    binding->target_layout_hash = record.target_layout_hash;
    binding->daemon_id = record.daemon_id;
    binding->daemon_session_id = record.daemon_session_id;
    auto insert_status = d_.bindings.insert(binding);
    if (!insert_status.ok() && !absl::IsAlreadyExists(insert_status)) {
      return insert_status;
    }
  }
  return target_publish_service_.remember_target_publication(std::move(record));
}

absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext> TargetMaterializationService::
    inspect_target_publication_context_for_testing(const v2::PublishTargetReplicaRequest& req, absl::Time now) {
  return target_publish_service_.inspect_target_publication_context(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> TargetMaterializationService::
    build_target_publication_workflow_routed_request_for_testing(
        const v2::PublishTargetReplicaRequest& req,
        absl::Time now) const {
  return target_publish_service_.build_target_publication_workflow_routed_request(req, now);
}

absl::StatusOr<RoutedAuthorityRequest> TargetMaterializationService::
    build_target_publication_workflow_continuation_request_for_testing(
        const RoutedAuthorityRequest& routed_request,
        const OwnerStageReply& workflow_gate_reply) const {
  return target_publish_service_.build_target_publication_workflow_continuation_request(
      routed_request, workflow_gate_reply);
}

absl::StatusOr<std::optional<OwnerStageReply>> TargetMaterializationService::maybe_route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  return target_publish_service_.maybe_route_authority_stage(routed_request, now);
}

grpc::Status TargetMaterializationService::materialize_into_target(
    RpcContext& rctx,
    const v2::MaterializeIntoTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  auto& span = rctx.span();
  if (d_.shutdown_signal.is_shutting_down()) {
    record_materialize_into_target(
        "error", "unavailable", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool group_realization_enabled = req.has_group_realization() && req.group_realization().enabled();
  if ((!req.has_selection() || req.selection().artifact_id().empty()) && !group_realization_enabled) {
    record_materialize_into_target(
        "error", "missing_artifact_id", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "selection.artifact_id is required for MaterializeIntoTarget"};
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

  auto validated_target_or = d_.external_target_access_service.validate_local_target_layout(
      rctx.server_context().peer(), "MaterializeIntoTarget", layout, req.pid(), req.device_uuid());
  if (!validated_target_or.ok()) {
    record_materialize_into_target(
        "error", "target_access_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(validated_target_or.status());
  }
  auto validated_target = std::move(*validated_target_or);
  const auto device = validated_target.device;
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    record_materialize_into_target(
        "error", "target_kind_unsupported", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "HOST_SHARED target_layout is not supported for MaterializeIntoTarget"};
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

  std::string daemon_id = d_.identity.daemon_id();
  if (daemon_id.empty()) {
    daemon_id = d_.daemon_id;
  }
  const std::string worker_id = d_.identity.worker_id();
  auto common_or = prepare_target_materialization_common(
      req,
      rctx.server_context().peer(),
      "MaterializeIntoTarget",
      d_.global_store_client,
      daemon_id,
      d_.daemon_session_id,
      worker_id,
      d_.disk_imports,
      storage_path_);
  if (!common_or.ok()) {
    return to_grpc_status(common_or.status());
  }
  auto common = std::move(*common_or);
  auto effective_req = req;
  effective_req.mutable_selection()->CopyFrom(common.effective_selection);
  const auto request_context = common.request_context;
  std::string resolved_artifact_id = std::move(common.resolved_artifact_id);
  const bool gs_connected = common.gs_connected;
  std::optional<std::filesystem::path> normalized_disk_path = std::move(common.normalized_disk_path);
  span->SetAttribute("tc.artifact.id_kind", artifact_id_kind_attr(resolved_artifact_id));
  auto disk_metadata_or =
      build_target_disk_metadata(normalized_disk_path, resolved_artifact_id, device.ordinal, d_.disk_imports);
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  auto disk_metadata = std::move(*disk_metadata_or);
  span->SetAttribute("tc.disk.metadata_capability", disk_metadata_capability_attr(disk_metadata));
  if (is_byte_only_disk_metadata(disk_metadata) &&
      selection_requires_tensor_aware_metadata(effective_req.selection())) {
    return {StatusCode::INVALID_ARGUMENT, "selection/view requires tensor-aware mounted-source metadata"};
  }

  auto canonical_json_or = load_canonical_index_with_disk_fallback(
      d_.engine, resolved_artifact_id, normalized_disk_path, device.ordinal, gs_connected);
  if (!canonical_json_or.ok()) {
    record_materialize_into_target(
        "error", "index_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_json_or.status());
  }

  TargetMaterializationPlan plan;
  auto build_plan_status = build_target_materialization_plan(
      d_.engine,
      resolved_artifact_id,
      effective_req,
      layout,
      offsets,
      std::move(*canonical_json_or),
      record_materialize_into_target,
      plan);
  if (!build_plan_status.ok()) {
    return build_plan_status;
  }

  const auto& resolved_selection = plan.resolved_selection;
  const bool has_subset = resolved_selection.tensor_names_size() > 0;
  auto& view_spec = plan.view_spec;
  auto& view_plan = plan.view_plan;
  auto& resolved_view_id = plan.resolved_view_id;
  auto& view_data_hash = plan.view_data_hash;
  const bool has_view_transform = plan.has_view_transform;
  const std::string& canonical_index_json = plan.canonical_index_json;
  const uint64_t logical_total_size = plan.logical_total_size;

  if (d_.external_target_verification_enabled && resolved_view_id.has_value() && !view_data_hash.has_value()) {
    auto view_meta_or = d_.engine.get_view_metadata(resolved_artifact_id, *resolved_view_id);
    if (view_meta_or.ok()) {
      view_data_hash = view_meta_or->view_data_hash;
    } else {
      VLOG(1) << "MaterializeIntoTarget: view metadata unavailable for verification: " << view_meta_or.status();
    }
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

  auto storage_lease = std::move(validated_target.storage_lease);

  std::vector<TargetLayoutSpan> verification_spans;
  if (verify_external_target) {
    verification_spans.reserve(storage_lease.storages().size());
    uint64_t cursor = 0;
    for (const auto& storage : storage_lease.storages()) {
      verification_spans.push_back(
          TargetLayoutSpan{
              .base_ptr = storage.base_ptr,
              .offset = cursor,
              .length = storage.length,
          });
      cursor += storage.length;
    }
  }

  std::optional<store::loading::DiskSource> disk_source;
  if (normalized_disk_path.has_value()) {
    disk_source = store::loading::DiskSource{
        .path = *normalized_disk_path,
        .expected_size = logical_total_size,
        .require_descriptor = common::is_mi2_artifact_id(resolved_artifact_id),
    };
  }

  store::loading::MaterializeHints hints;
  const std::chrono::milliseconds request_budget = resolve_target_request_budget(rctx.server_context());
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.artifact_id = resolved_artifact_id;
  const std::string requester_worker_id = d_.identity.worker_id();
  if (!requester_worker_id.empty()) {
    hints.transport_requester_worker_id = requester_worker_id;
  }
  apply_operation_transport_context(common.transport_context, &hints);
  apply_request_context_to_hints(request_context, &hints);
  const bool prefer_direct_disk_for_source_layout =
      disk_source.has_value() && disk_metadata.has_value() && disk_metadata->source_index_json.has_value();
  if (prefer_direct_disk_for_source_layout) {
    hints.set_retrieval_policy(
        store::loading::RetrievalPolicy{
            .preference = store::loading::SourcePreference::kPreferDisk,
            .allow_p2p = false,
            .allow_disk = request_context.retrieval_policy.allow_disk,
        });
  }
  hints.verify = store::loading::MaterializeHints::Verify::NONE;
  // bind/swap target materialization should become reusable sources so
  // subsequent peers can diffuse fan-out instead of contending on the
  // original publisher. This is still policy-gated by daemon promotion
  // controls and GS routing limits.
  hints.export_policy = store::loading::ExportPolicy::kForce;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }

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
    variant.canonical_index_json = canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    hints.variant = std::move(variant);
  }

  const uint64_t generation = compute_generation_from_index(canonical_index_json);
  store::loading::IntoTargetLayout target_layout;
  target_layout.storages.assign(storage_lease.storages().begin(), storage_lease.storages().end());
  target_layout.total_size = logical_total_size;
  auto result_or = [&]() -> absl::StatusOr<store::loading::MaterializeIntoTargetResult> {
    const std::string requester_node_id = d_.identity.node_id();
    const std::string requester_source_domain = !requester_node_id.empty() ? requester_node_id : daemon_id;
    auto progressive_identity = build_progressive_target_identity(
        resolved_artifact_id,
        resolved_selection,
        layout,
        resolved_view_id,
        has_subset,
        has_view_transform,
        common.group_begin_context);
    if (progressive_identity.has_value()) {
      auto progressive_or = try_progressive_materialize_into_target(
          d_,
          device,
          target_layout,
          *progressive_identity,
          hints,
          resolved_artifact_id,
          daemon_id,
          requester_worker_id,
          requester_source_domain,
          request_budget,
          rctx.server_context());
      if (!progressive_or.ok()) {
        return progressive_or.status();
      }
      if (progressive_or->has_value()) {
        return **progressive_or;
      }
    }
    return d_.engine.materialize_into_target(
        device, target_layout, canonical_index_json, generation, hints, disk_source);
  }();
  if (!result_or.ok()) {
    if (absl::IsDataLoss(result_or.status())) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
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
  auto preflight_or = serving_artifact_manifest::preflight_serving_artifact(
      &d_.engine,
      serving_artifact_manifest::build_preflight_request(
          resolved_artifact_id,
          canonical_index_json,
          disk_source,
          disk_metadata,
          req.has_serving_artifact_policy() ? &req.serving_artifact_policy() : nullptr));
  if (!preflight_or.ok()) {
    for (const auto& region_id : storage_lease.acquired_region_ids()) {
      d_.regions.mark_poisoned(region_id).IgnoreError();
    }
    record_materialize_into_target(
        "error", "serving_manifest_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(preflight_or.status());
  }

  if (verify_external_target) {
    auto actual_hash_or =
        compute_target_layout_multihash(std::move(verification_spans), logical_total_size, device.ordinal);
    if (!actual_hash_or.ok()) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
        d_.regions.mark_poisoned(region_id).IgnoreError();
      }
      record_materialize_into_target(
          "error", "verification_failed", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
      return to_grpc_status(
          absl::DataLossError(
              absl::StrCat("external target verification failed: ", actual_hash_or.status().message())));
    }
    if (expected_data_hash.has_value() && *expected_data_hash != *actual_hash_or) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
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
  if (!result_or->selected_source_replica_id.empty()) {
    resp.set_selected_source_replica_id(result_or->selected_source_replica_id);
  }
  if (!result_or->selected_source_transport_id.empty()) {
    resp.set_selected_source_transport_id(result_or->selected_source_transport_id);
  }
  resp.set_canonical_index_bytes(canonical_index_json);
  if (view_plan.has_value() && layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    resp.set_view_index_bytes(view_plan->view_index_json);
  }
  if (resolved_selection.tensor_names_size() > 0 || !resolved_selection.view_subset_hash().empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!resolved_selection.view_subset_hash().empty()) {
      subset->set_subset_hash(resolved_selection.view_subset_hash());
    }
    for (const auto& name : resolved_selection.tensor_names()) {
      subset->add_tensor_names(name);
    }
  }
  resp.mutable_resolved_selection()->CopyFrom(resolved_selection);
  resp.set_generation(generation);
  record_materialize_into_target("ok", "ok", resp.source());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TargetMaterializationService::materialize_into_mapped_target(
    RpcContext& rctx,
    const v2::MaterializeIntoMappedTargetRequest& req,
    v2::MaterializeIntoTargetResponse& resp) {
  const auto total_start = std::chrono::steady_clock::now();
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs() && req.has_operation_id()) {
    span->SetAttribute("tc.operation.id", req.operation_id());
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    record_materialize_into_target(
        "error", "unavailable", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  std::string daemon_id = d_.identity.daemon_id();
  if (daemon_id.empty()) {
    daemon_id = d_.daemon_id;
  }
  const std::string worker_id = d_.identity.worker_id();
  auto common_or = prepare_target_materialization_common(
      req,
      rctx.server_context().peer(),
      "MaterializeIntoMappedTarget",
      d_.global_store_client,
      daemon_id,
      d_.daemon_session_id,
      worker_id,
      d_.disk_imports,
      storage_path_);
  const auto common_done = std::chrono::steady_clock::now();
  if (!common_or.ok()) {
    return to_grpc_status(common_or.status());
  }
  auto common = std::move(*common_or);
  auto effective_req = req;
  effective_req.mutable_selection()->CopyFrom(common.effective_selection);
  const auto request_context = common.request_context;
  std::string resolved_artifact_id = std::move(common.resolved_artifact_id);
  const bool gs_connected = common.gs_connected;
  std::optional<std::filesystem::path> normalized_disk_path = std::move(common.normalized_disk_path);
  span->SetAttribute("tc.artifact.id_kind", artifact_id_kind_attr(resolved_artifact_id));
  auto disk_metadata_or =
      build_target_disk_metadata(normalized_disk_path, resolved_artifact_id, /*device_ordinal=*/0, d_.disk_imports);
  const auto disk_metadata_ready = std::chrono::steady_clock::now();
  if (!disk_metadata_or.ok()) {
    return to_grpc_status(disk_metadata_or.status());
  }
  auto disk_metadata = std::move(*disk_metadata_or);
  span->SetAttribute("tc.disk.metadata_capability", disk_metadata_capability_attr(disk_metadata));
  if (is_byte_only_disk_metadata(disk_metadata) &&
      selection_requires_tensor_aware_metadata(effective_req.selection())) {
    return {StatusCode::INVALID_ARGUMENT, "selection/view requires tensor-aware mounted-source metadata"};
  }

  if (!req.has_copy_plan() || req.copy_plan().entries_size() == 0) {
    record_materialize_into_target(
        "error", "mapping_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "copy_plan is required for mapped binding"};
  }
  if (req.copy_plan().version() != 1) {
    record_materialize_into_target(
        "error", "mapping_version", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "unsupported copy_plan version"};
  }

  const auto& layout = req.target_layout();
  if (layout.layout_kind() != v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED) {
    record_materialize_into_target(
        "error", "layout_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Only LAYOUT_KIND_COALESCED_UNSPECIFIED is supported"};
  }
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS) {
    record_materialize_into_target(
        "error", "tensor_spec_kind_mismatch", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "Mapped binding requires TENSOR_SPEC_KIND_OFFSETS"};
  }
  if (layout.storages_size() == 0) {
    record_materialize_into_target(
        "error", "storage_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include at least one storage entry"};
  }

  auto validated_target_or = d_.external_target_access_service.validate_local_target_layout(
      rctx.server_context().peer(), "MaterializeIntoMappedTarget", layout, req.pid(), req.device_uuid());
  const auto target_validate_done = std::chrono::steady_clock::now();
  if (!validated_target_or.ok()) {
    record_materialize_into_target(
        "error", "target_access_invalid", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(validated_target_or.status());
  }
  auto validated_target = std::move(*validated_target_or);
  const auto device = validated_target.device;

  auto offsets_or = resolve_target_offsets(layout);
  const auto offsets_done = std::chrono::steady_clock::now();
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

  auto canonical_json_or = load_canonical_index_with_disk_fallback(
      d_.engine, resolved_artifact_id, normalized_disk_path, device.ordinal, gs_connected);
  const auto canonical_done = std::chrono::steady_clock::now();
  if (!canonical_json_or.ok()) {
    record_materialize_into_target(
        "error", "index_missing", v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
    return to_grpc_status(canonical_json_or.status());
  }

  MappedTargetMaterializationPlan mapped_plan;
  auto build_mapped_plan_status = build_mapped_target_materialization_plan(
      d_.engine,
      effective_req,
      resolved_artifact_id,
      offsets,
      std::move(*canonical_json_or),
      record_materialize_into_target,
      mapped_plan);
  const auto mapped_plan_done = std::chrono::steady_clock::now();
  if (!build_mapped_plan_status.ok()) {
    return build_mapped_plan_status;
  }
  const auto& resolved_selection = mapped_plan.resolved_selection;
  const uint64_t logical_total_size = mapped_plan.logical_total_size;
  const std::string& canonical_index_json = mapped_plan.canonical_index_json;
  auto& view_spec = mapped_plan.view_spec;
  auto& view_plan = mapped_plan.view_plan;

  auto storage_lease = std::move(validated_target.storage_lease);

  std::optional<store::loading::DiskSource> disk_source;
  if (normalized_disk_path.has_value()) {
    disk_source = store::loading::DiskSource{
        .path = *normalized_disk_path,
        .expected_size = logical_total_size,
        .require_descriptor = common::is_mi2_artifact_id(resolved_artifact_id),
    };
  }
  const auto disk_metadata_done = disk_metadata_ready;

  store::loading::MaterializeHints hints;
  const std::chrono::milliseconds request_budget = resolve_target_request_budget(rctx.server_context());
  hints.request_budget = request_budget;
  hints.transport_wait_timeout = request_budget;
  hints.artifact_id = resolved_artifact_id;
  const std::string requester_worker_id = d_.identity.worker_id();
  if (!requester_worker_id.empty()) {
    hints.transport_requester_worker_id = requester_worker_id;
  }
  apply_operation_transport_context(common.transport_context, &hints);
  apply_request_context_to_hints(request_context, &hints);
  const bool prefer_direct_disk_for_source_layout =
      disk_source.has_value() && disk_metadata.has_value() && disk_metadata->source_index_json.has_value();
  if (prefer_direct_disk_for_source_layout) {
    hints.set_retrieval_policy(
        store::loading::RetrievalPolicy{
            .preference = store::loading::SourcePreference::kPreferDisk,
            .allow_p2p = false,
            .allow_disk = request_context.retrieval_policy.allow_disk,
        });
  }
  hints.verify = store::loading::MaterializeHints::Verify::NONE;
  // bind/swap target materialization should become reusable sources so
  // subsequent peers can diffuse fan-out instead of contending on the
  // original publisher. This is still policy-gated by daemon promotion
  // controls and GS routing limits.
  hints.export_policy = store::loading::ExportPolicy::kForce;
  if (disk_metadata.has_value()) {
    hints.disk_metadata = *disk_metadata;
  }
  if (disk_source.has_value()) {
    hints.source_mutation_policy = store::loading::SourceMutationPolicy::kReadOnly;
  }
  if (view_plan.has_value() && view_plan->transform.requires_materialization) {
    return {StatusCode::INVALID_ARGUMENT, "mapped binding does not support view transforms"};
  }
  if (!resolved_selection.view_id().empty() || view_spec.has_value() || view_plan.has_value()) {
    store::loading::VariantIdentity variant;
    variant.canonical_artifact_id = resolved_artifact_id;
    if (!resolved_selection.view_id().empty()) {
      variant.view_id = resolved_selection.view_id();
    }
    if (view_spec.has_value()) {
      variant.view_spec = view_spec;
    }
    if (view_plan.has_value()) {
      variant.cached_plan = view_plan;
    }
    variant.canonical_index_json = canonical_index_json;
    variant.placement = resolve_transform_placement(req.placement(), view_spec);
    hints.variant = std::move(variant);
  }
  store::loading::IntoTargetLayout target_layout;
  target_layout.storages.assign(storage_lease.storages().begin(), storage_lease.storages().end());
  target_layout.total_size = logical_total_size;
  const uint64_t generation = compute_generation_from_index(canonical_index_json);
  auto prepared_execution_or = build_resolved_mapped_materialization_plan(
      resolved_artifact_id,
      generation,
      target_layout,
      materialization_target_plan::MappedTargetMaterializationPlan{
          .view_spec = view_spec,
          .view_plan = view_plan,
          .resolved_selection = resolved_selection,
          .representation = mapped_plan.representation,
          .canonical_index_json = canonical_index_json,
          .selected_index_json = {},
          .publish_storages = {},
          .publish_segments = {},
          .logical_total_size = logical_total_size,
      },
      hints.variant,
      disk_metadata.has_value() && disk_metadata->source_index_json.has_value()
          ? std::optional<std::string_view>(*disk_metadata->source_index_json)
          : std::nullopt);
  if (!prepared_execution_or.ok()) {
    return to_grpc_status(prepared_execution_or.status());
  }
  auto prepared_execution = std::move(*prepared_execution_or);
  // Mapped-target RPCs only carry collective topology via operation metadata, so the
  // best-effort default must stay collective-first rather than source-bound strict mode.
  const auto collective_policy = default_collective_policy_for_mapped_target(request_context.execution_topology);
  const auto source_facts = store::runtime::ingestion::strategy::SourceBoundSourceFacts{
      .disk_source_available = disk_source.has_value(),
      .disk_source_is_safetensors = disk_metadata.has_value() && disk_metadata->is_safetensors.value_or(false),
  };
  auto strategy_plan_or = store::runtime::ingestion::strategy::build_source_bound_execution_strategy_plan(
      prepared_execution.resolved_plan,
      prepared_execution.lowering_artifacts,
      collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE
          ? store::runtime::ingestion::strategy::SourceBoundPolicy::kRequirePureCollective
          : collective_policy == v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE
          ? store::runtime::ingestion::strategy::SourceBoundPolicy::kDisableCollective
          : store::runtime::ingestion::strategy::SourceBoundPolicy::kCollectiveFirst,
      d_.engine.options().materialization_strategy,
      request_context.execution_topology,
      source_facts);
  if (!strategy_plan_or.ok()) {
    return to_grpc_status(strategy_plan_or.status());
  }
  prepared_execution.strategy_plan = *strategy_plan_or;
  prepared_execution.lowering_artifacts.reset();
  const auto materialize_start = std::chrono::steady_clock::now();
  auto result_or = d_.engine.materialize_mapped_into_target(device, prepared_execution, hints, disk_source);
  const auto engine_done = std::chrono::steady_clock::now();
  if (!result_or.ok()) {
    LOG(ERROR) << "MaterializeIntoMappedTarget engine failure"
               << " artifact_id=" << resolved_artifact_id << " copy_entries=" << req.copy_plan().entries_size()
               << " dst_tensors=" << req.dst_tensors_size() << " storages=" << layout.storages_size()
               << " status=" << result_or.status();
    if (absl::IsDataLoss(result_or.status())) {
      for (const auto& region_id : storage_lease.acquired_region_ids()) {
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

  resp.set_artifact_id(resolved_artifact_id);
  resp.set_status(v2::MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_source(to_proto_source(result_or->source));
  if (!result_or->selected_source_replica_id.empty()) {
    resp.set_selected_source_replica_id(result_or->selected_source_replica_id);
  }
  if (!result_or->selected_source_transport_id.empty()) {
    resp.set_selected_source_transport_id(result_or->selected_source_transport_id);
  }
  resp.set_canonical_index_bytes(canonical_index_json);
  if (view_plan.has_value() && !view_plan->is_identity) {
    resp.set_view_index_bytes(view_plan->view_index_json);
  }
  if (resolved_selection.tensor_names_size() > 0 || !resolved_selection.view_subset_hash().empty()) {
    auto* subset = resp.mutable_view_subset();
    if (!resolved_selection.view_subset_hash().empty()) {
      subset->set_subset_hash(resolved_selection.view_subset_hash());
    }
    for (const auto& name : resolved_selection.tensor_names()) {
      subset->add_tensor_names(name);
    }
  }
  resp.mutable_resolved_selection()->CopyFrom(resolved_selection);
  resp.set_generation(generation);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.mapped.entries", static_cast<int64_t>(req.copy_plan().entries_size()));
    span->SetAttribute("tc.mapped.bytes", static_cast<int64_t>(mapped_plan.representation.total_bytes_copied));
  }
  const double materialize_sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - materialize_start).count();
  const double total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  LOG(INFO) << "MaterializeIntoMappedTarget controller timings"
            << " artifact_id=" << resolved_artifact_id << " copy_entries=" << req.copy_plan().entries_size()
            << " dst_tensors=" << req.dst_tensors_size() << " storages=" << layout.storages_size()
            << " common_sec=" << std::chrono::duration<double>(common_done - total_start).count()
            << " target_validate_sec=" << std::chrono::duration<double>(target_validate_done - common_done).count()
            << " offsets_sec=" << std::chrono::duration<double>(offsets_done - target_validate_done).count()
            << " canonical_sec=" << std::chrono::duration<double>(canonical_done - offsets_done).count()
            << " mapped_plan_sec=" << std::chrono::duration<double>(mapped_plan_done - canonical_done).count()
            << " disk_metadata_sec=" << std::chrono::duration<double>(disk_metadata_done - mapped_plan_done).count()
            << " engine_sec=" << std::chrono::duration<double>(engine_done - disk_metadata_done).count()
            << " materialize_sec=" << materialize_sec << " total_sec=" << total_sec;
  LOG(INFO) << "MaterializeIntoMappedTarget completed"
            << " artifact_id=" << resolved_artifact_id
            << " source_layout=" << (disk_metadata.has_value() && disk_metadata->source_index_json.has_value())
            << " collective=" << hints.collective_load_group.has_value()
            << " source=" << v2::MaterializationSource_Name(resp.source()) << " target_bytes=" << logical_total_size
            << " materialize_sec=" << materialize_sec << " total_sec=" << total_sec;
  record_materialize_into_target("ok", "ok", resp.source());
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TargetMaterializationService::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  return target_publish_service_.publish_target_replica(rctx, req, resp);
}

grpc::Status TargetMaterializationService::start_publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::StartPublishTargetReplicaResponse& resp) {
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  return target_publish_service_.start_publish_target_replica(rctx, req, resp);
}

absl::StatusOr<TargetPublicationRegistry::Record> TargetMaterializationService::remember_target_publication(
    TargetPublicationRegistry::Record record) {
  return target_publish_service_.remember_target_publication(std::move(record));
}

absl::Status TargetMaterializationService::terminalize_target_publication(
    std::string_view publication_id,
    std::string_view reason,
    bool release_published_lifecycle_lease) {
  return target_publish_service_.terminalize_publication(publication_id, reason, release_published_lifecycle_lease);
}

absl::Status TargetMaterializationService::admit_public_operation(
    const tensorcast::operation::v1::OperationRef& operation_ref,
    absl::Time now) const {
  return target_publish_service_.admit_public_operation(operation_ref, now);
}

} // namespace tensorcast::daemon
