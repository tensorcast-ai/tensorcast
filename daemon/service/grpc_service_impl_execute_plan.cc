// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/grpc_service_impl.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/selection_identity.h"
#include "core/store/device_registry.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/timestamp.pb.h"
#include "tensorcast/node_agent/v1/node_agent.pb.h"

namespace tensorcast::daemon {

namespace {

using ::grpc::Status;
using ::grpc::StatusCode;

constexpr std::string_view kArtifactSetDigestPrefix = "tensorcast.artifact_set.v1\n";
constexpr std::string_view kArtifactSetCarrierInline = "inline";
constexpr std::string_view kArtifactSetCarrierManifestBacked = "manifest_backed";
constexpr std::string_view kManifestBridgeSchema = "tensorcast.manifest_artifact_set_bridge";
constexpr uint32_t kManifestBridgeVersion = 1;
constexpr int32_t kCpuDeviceId = -1;
constexpr size_t kMaxInlineArtifactSetItems = 1024;

struct CanonicalResolvedItem {
  common::SelectionIdentity identity;
  tensorcast::common::v1::ArtifactSelection selection;
};

struct IdentityLess {
  bool operator()(const common::SelectionIdentity& lhs, const common::SelectionIdentity& rhs) const {
    if (lhs.artifact_id != rhs.artifact_id) {
      return lhs.artifact_id < rhs.artifact_id;
    }
    if (lhs.logical_layout_hash != rhs.logical_layout_hash) {
      return lhs.logical_layout_hash < rhs.logical_layout_hash;
    }
    return lhs.selection_hash < rhs.selection_hash;
  }
};

absl::StatusOr<std::string> serialize_deterministic(const google::protobuf::Message& message) {
  const size_t size = message.ByteSizeLong();
  std::string out;
  out.resize(size);
  google::protobuf::io::ArrayOutputStream aos(out.data(), static_cast<int>(out.size()));
  google::protobuf::io::CodedOutputStream cos(&aos);
  cos.SetSerializationDeterministic(true);
  if (!message.SerializeToCodedStream(&cos)) {
    return absl::InternalError("failed to serialize protobuf deterministically");
  }
  return out;
}

std::string hex_encode(std::string_view bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.resize(bytes.size() * 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    const uint8_t value = static_cast<uint8_t>(bytes[i]);
    out[2 * i] = kHex[value >> 4];
    out[2 * i + 1] = kHex[value & 0x0f];
  }
  return out;
}

bool is_lower_hex(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(
      value.begin(), value.end(), [](const char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); });
}

std::string grpc_status_code_name(const StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      return "OK";
    case StatusCode::CANCELLED:
      return "CANCELLED";
    case StatusCode::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case StatusCode::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case StatusCode::NOT_FOUND:
      return "NOT_FOUND";
    case StatusCode::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case StatusCode::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case StatusCode::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case StatusCode::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case StatusCode::ABORTED:
      return "ABORTED";
    case StatusCode::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case StatusCode::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case StatusCode::INTERNAL:
      return "INTERNAL";
    case StatusCode::UNAVAILABLE:
      return "UNAVAILABLE";
    case StatusCode::DATA_LOSS:
      return "DATA_LOSS";
    case StatusCode::UNAUTHENTICATED:
      return "UNAUTHENTICATED";
    default:
      return "UNKNOWN";
  }
}

bool grpc_status_retryable(const StatusCode code) {
  switch (code) {
    case StatusCode::CANCELLED:
    case StatusCode::DEADLINE_EXCEEDED:
    case StatusCode::ABORTED:
    case StatusCode::RESOURCE_EXHAUSTED:
    case StatusCode::UNAVAILABLE:
      return true;
    default:
      return false;
  }
}

void set_timestamp_ms(google::protobuf::Timestamp* timestamp, const int64_t epoch_ms) {
  timestamp->set_seconds(epoch_ms / 1000);
  timestamp->set_nanos(static_cast<int32_t>((epoch_ms % 1000) * 1000000));
}

void fill_status(
    tensorcast::node_agent::v1::OperationStatus* status,
    tensorcast::node_agent::v1::OperationState state,
    std::string_view message,
    std::optional<StatusCode> code,
    bool retryable) {
  status->set_state(state);
  status->set_message(std::string(message));
  set_timestamp_ms(status->mutable_as_of(), absl::ToUnixMillis(absl::Now()));
  if (code.has_value()) {
    auto* error = status->mutable_error();
    error->set_status_code(grpc_status_code_name(*code));
    error->set_message(std::string(message));
    error->set_retryable(retryable);
  } else {
    status->clear_error();
  }
}

void fill_status_success(tensorcast::node_agent::v1::OperationStatus* status, std::string_view message) {
  fill_status(status, tensorcast::node_agent::v1::OPERATION_STATE_SUCCESS, message, std::nullopt, false);
}

void fill_status_failed(
    tensorcast::node_agent::v1::OperationStatus* status,
    std::string_view message,
    const StatusCode code) {
  fill_status(status, tensorcast::node_agent::v1::OPERATION_STATE_FAILED, message, code, grpc_status_retryable(code));
}

void fill_status_cancelled(tensorcast::node_agent::v1::OperationStatus* status, std::string_view message) {
  fill_status(status, tensorcast::node_agent::v1::OPERATION_STATE_CANCELLED, message, StatusCode::CANCELLED, true);
}

void fill_status_degraded(
    tensorcast::node_agent::v1::OperationStatus* status,
    std::string_view message,
    const StatusCode code) {
  fill_status(status, tensorcast::node_agent::v1::OPERATION_STATE_DEGRADED, message, code, grpc_status_retryable(code));
}

std::string request_id_from_plan(const tensorcast::plan::v1::PlanSpec& plan) {
  if (!plan.context().request_id().empty()) {
    return plan.context().request_id();
  }
  return absl::StrCat("req-", absl::ToUnixNanos(absl::Now()));
}

std::optional<int64_t> remaining_deadline_ms(
    const tensorcast::plan::v1::CallContext& call_context,
    const int64_t execution_started_ms) {
  if (!call_context.has_deadline_ms()) {
    return std::nullopt;
  }
  const int64_t elapsed_ms = absl::ToUnixMillis(absl::Now()) - execution_started_ms;
  return static_cast<int64_t>(call_context.deadline_ms()) - elapsed_ms;
}

std::string derive_action_idempotency_key(
    std::string_view base_key,
    std::string_view action,
    std::string_view target_id,
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::optional<int32_t> device_id,
    std::optional<uint64_t> ttl_ms,
    std::optional<std::string_view> extra) {
  std::string payload;
  payload.reserve(
      base_key.size() + action.size() + target_id.size() + selection.artifact_id().size() +
      selection.logical_layout_hash().size() + selection.selection_hash().size() + 64);
  payload.append(base_key);
  payload.append("|");
  payload.append(action);
  payload.append("|target=");
  payload.append(target_id);
  payload.append("|artifact=");
  payload.append(selection.artifact_id());
  if (device_id.has_value()) {
    payload.append("|device=");
    payload.append(std::to_string(*device_id));
  }
  if (ttl_ms.has_value()) {
    payload.append("|ttl=");
    payload.append(std::to_string(*ttl_ms));
  }
  payload.append("|logical=");
  payload.append(selection.logical_layout_hash());
  payload.append("|selection=");
  payload.append(selection.selection_hash());
  if (extra.has_value()) {
    payload.append("|extra=");
    payload.append(*extra);
  }
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::StrCat(
      "tc.plan.action.v1:", hex_encode(std::string_view(reinterpret_cast<const char*>(digest.data()), digest.size())));
}

std::string replica_uuid_for_action(
    const tensorcast::plan::v1::CallContext& call_context,
    std::string_view action,
    std::string_view target_id,
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::optional<int32_t> device_id,
    std::optional<uint64_t> ttl_ms,
    std::optional<std::string_view> extra,
    std::string_view fallback_seed) {
  if (!call_context.idempotency_key().empty()) {
    const std::string action_key = derive_action_idempotency_key(
        call_context.idempotency_key(), action, target_id, selection, device_id, ttl_ms, extra);
    const size_t pos = action_key.find(':');
    return pos == std::string::npos ? action_key : action_key.substr(pos + 1);
  }
  return absl::StrCat("ingress-", fallback_seed);
}

std::string action_name(const tensorcast::plan::v1::PlanAction& action) {
  switch (action.kind_case()) {
    case tensorcast::plan::v1::PlanAction::kPrefetch:
      return "prefetch";
    case tensorcast::plan::v1::PlanAction::kPrefetchSet:
      return "prefetch_set";
    case tensorcast::plan::v1::PlanAction::kPinDeviceResidency:
      return "pin_device_residency";
    case tensorcast::plan::v1::PlanAction::kUnpinDeviceResidency:
      return "unpin_device_residency";
    case tensorcast::plan::v1::PlanAction::kTransformInto:
      return "transform_into";
    case tensorcast::plan::v1::PlanAction::kTransformRegister:
      return "transform_register";
    case tensorcast::plan::v1::PlanAction::kManifest:
      return "manifest";
    case tensorcast::plan::v1::PlanAction::kPublish:
      return "publish";
    case tensorcast::plan::v1::PlanAction::kHydrate:
      return "hydrate";
    case tensorcast::plan::v1::PlanAction::kEvictLocal:
      return "evict_local";
    case tensorcast::plan::v1::PlanAction::kClusterAction:
      return "cluster_action";
    case tensorcast::plan::v1::PlanAction::KIND_NOT_SET:
      return "unknown";
  }
  return "unknown";
}

absl::StatusOr<std::vector<CanonicalResolvedItem>> canonicalize_artifact_selections(
    const google::protobuf::RepeatedPtrField<tensorcast::common::v1::ArtifactSelection>& selections,
    std::optional<size_t> max_items = std::nullopt) {
  if (max_items.has_value() && static_cast<size_t>(selections.size()) > *max_items) {
    return absl::InvalidArgumentError("inline ArtifactSetRef exceeds the explicit inline limit");
  }

  std::map<common::SelectionIdentity, std::pair<std::string, tensorcast::common::v1::ArtifactSelection>, IdentityLess>
      canonical_items;
  for (const auto& selection : selections) {
    auto identity_or = common::build_selection_identity(selection);
    if (!identity_or.ok()) {
      return identity_or.status();
    }
    auto stable_bytes_or = serialize_deterministic(selection);
    if (!stable_bytes_or.ok()) {
      return stable_bytes_or.status();
    }
    auto it = canonical_items.find(*identity_or);
    if (it == canonical_items.end() || *stable_bytes_or < it->second.first) {
      canonical_items[*identity_or] = {*stable_bytes_or, selection};
    }
  }

  std::vector<CanonicalResolvedItem> resolved;
  resolved.reserve(canonical_items.size());
  for (const auto& [identity, payload] : canonical_items) {
    resolved.push_back(
        CanonicalResolvedItem{
            .identity = identity,
            .selection = payload.second,
        });
  }
  return resolved;
}

std::string compute_artifact_set_digest_hex(const std::vector<CanonicalResolvedItem>& resolved) {
  std::string payload;
  payload.append(kArtifactSetDigestPrefix);
  for (const auto& item : resolved) {
    const uint64_t artifact_id_size = item.identity.artifact_id.size();
    for (int shift = 56; shift >= 0; shift -= 8) {
      payload.push_back(static_cast<char>((artifact_id_size >> shift) & 0xff));
    }
    payload.append(item.identity.artifact_id);

    const uint32_t logical_size = static_cast<uint32_t>(item.identity.logical_layout_hash.size());
    for (int shift = 24; shift >= 0; shift -= 8) {
      payload.push_back(static_cast<char>((logical_size >> shift) & 0xff));
    }
    payload.append(item.identity.logical_layout_hash);

    const uint32_t selection_size = static_cast<uint32_t>(item.identity.selection_hash.size());
    for (int shift = 24; shift >= 0; shift -= 8) {
      payload.push_back(static_cast<char>((selection_size >> shift) & 0xff));
    }
    payload.append(item.identity.selection_hash);
  }

  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return hex_encode(std::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
}

absl::Status validate_artifact_set_ref_shape(const tensorcast::plan::v1::ArtifactSetRef& artifact_set) {
  if (!is_lower_hex(artifact_set.set_digest_hex())) {
    return absl::InvalidArgumentError("ArtifactSetRef.set_digest_hex must be non-empty lowercase hex");
  }
  if (artifact_set.carrier_form() == kArtifactSetCarrierInline) {
    if (artifact_set.has_manifest_selection()) {
      return absl::InvalidArgumentError("ArtifactSetRef inline carrier must not set manifest_selection");
    }
    if (static_cast<size_t>(artifact_set.inline_items_size()) > kMaxInlineArtifactSetItems) {
      return absl::InvalidArgumentError("ArtifactSetRef inline carrier exceeds the explicit inline limit");
    }
    return absl::OkStatus();
  }
  if (artifact_set.carrier_form() == kArtifactSetCarrierManifestBacked) {
    if (!artifact_set.has_manifest_selection()) {
      return absl::InvalidArgumentError("ArtifactSetRef manifest_backed carrier requires manifest_selection");
    }
    if (artifact_set.inline_items_size() != 0) {
      return absl::InvalidArgumentError("ArtifactSetRef manifest_backed carrier must not set inline_items");
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError("ArtifactSetRef.carrier_form must be 'inline' or 'manifest_backed'");
}

absl::StatusOr<std::vector<CanonicalResolvedItem>> resolve_artifact_set(
    const tensorcast::plan::v1::PrefetchSetAction& action) {
  const auto& artifact_set = action.artifact_set();
  if (const absl::Status shape_status = validate_artifact_set_ref_shape(artifact_set); !shape_status.ok()) {
    return shape_status;
  }

  std::vector<CanonicalResolvedItem> resolved;
  if (artifact_set.carrier_form() == kArtifactSetCarrierInline) {
    auto resolved_or = canonicalize_artifact_selections(artifact_set.inline_items(), kMaxInlineArtifactSetItems);
    if (!resolved_or.ok()) {
      return resolved_or.status();
    }
    resolved = std::move(*resolved_or);
  } else {
    if (!action.has_manifest_bridge()) {
      return absl::FailedPreconditionError(
          "manifest_backed ArtifactSetRef resolution requires an explicit owner-provided resolver");
    }
    const auto& bridge = action.manifest_bridge();
    if (bridge.bridge_schema() != kManifestBridgeSchema) {
      return absl::FailedPreconditionError("ManifestArtifactSetBridge schema is unsupported");
    }
    if (bridge.bridge_version() != kManifestBridgeVersion) {
      return absl::FailedPreconditionError("ManifestArtifactSetBridge version is unsupported");
    }
    auto bridge_ref_bytes_or = serialize_deterministic(bridge.artifact_set_ref());
    if (!bridge_ref_bytes_or.ok()) {
      return bridge_ref_bytes_or.status();
    }
    auto request_ref_bytes_or = serialize_deterministic(artifact_set);
    if (!request_ref_bytes_or.ok()) {
      return request_ref_bytes_or.status();
    }
    if (*bridge_ref_bytes_or != *request_ref_bytes_or) {
      return absl::FailedPreconditionError(
          "ManifestArtifactSetBridge artifact_set_ref does not match requested ArtifactSetRef");
    }
    auto resolved_or = canonicalize_artifact_selections(bridge.resolved_items());
    if (!resolved_or.ok()) {
      return resolved_or.status();
    }
    resolved = std::move(*resolved_or);
  }

  if (compute_artifact_set_digest_hex(resolved) != artifact_set.set_digest_hex()) {
    return absl::FailedPreconditionError("ArtifactSetRef digest does not match resolved canonical item set");
  }
  if (resolved.size() != artifact_set.item_count()) {
    return absl::FailedPreconditionError("ArtifactSetRef item_count does not match resolved canonical item set");
  }
  return resolved;
}

void fill_item_identity(tensorcast::common::v1::SelectionIdentity* identity, const common::SelectionIdentity& source) {
  identity->set_artifact_id(source.artifact_id);
  identity->set_logical_layout_hash(source.logical_layout_hash);
  identity->set_selection_hash(source.selection_hash);
}

tensorcast::node_agent::v1::OperationState summarize_artifact_set_outcomes(
    const tensorcast::node_agent::v1::ArtifactSetResult& result,
    tensorcast::node_agent::v1::OperationStatus* summary) {
  size_t failed_count = 0;
  const tensorcast::node_agent::v1::OperationStatus* first_failed = nullptr;
  for (const auto& outcome : result.outcomes()) {
    if (outcome.status().state() != tensorcast::node_agent::v1::OPERATION_STATE_SUCCESS) {
      ++failed_count;
      if (first_failed == nullptr) {
        first_failed = &outcome.status();
      }
    }
  }
  const size_t success_count = static_cast<size_t>(result.outcomes_size()) - failed_count;
  const std::string message = absl::StrCat(
      "prefetch_set guarantees local_replica_ready only: ",
      success_count,
      "/",
      result.outcomes_size(),
      " items reached that floor");
  if (failed_count == 0) {
    fill_status_success(summary, message);
    return tensorcast::node_agent::v1::OPERATION_STATE_SUCCESS;
  }

  StatusCode code = StatusCode::FAILED_PRECONDITION;
  bool retryable = false;
  if (first_failed != nullptr && first_failed->has_error()) {
    retryable = first_failed->error().retryable();
    const std::string_view code_name = first_failed->error().status_code();
    if (code_name == "DEADLINE_EXCEEDED") {
      code = StatusCode::DEADLINE_EXCEEDED;
    } else if (code_name == "UNAVAILABLE") {
      code = StatusCode::UNAVAILABLE;
    } else if (code_name == "INTERNAL") {
      code = StatusCode::INTERNAL;
    } else if (code_name == "INVALID_ARGUMENT") {
      code = StatusCode::INVALID_ARGUMENT;
    }
  }
  if (success_count == 0) {
    fill_status(summary, tensorcast::node_agent::v1::OPERATION_STATE_FAILED, message, code, retryable);
    return tensorcast::node_agent::v1::OPERATION_STATE_FAILED;
  }
  fill_status(summary, tensorcast::node_agent::v1::OPERATION_STATE_DEGRADED, message, code, retryable);
  return tensorcast::node_agent::v1::OPERATION_STATE_DEGRADED;
}

} // namespace

Status StoreDaemonServiceImpl::ExecutePlan(
    grpc::ServerContext* ctx,
    const v2::ExecutePlanRequest* req,
    v2::ExecutePlanResponse* resp) {
  if (auto startup_status = block_if_startup_pending(); !startup_status.ok()) {
    return startup_status;
  }
  if (!opts_.gateway_ingress_enabled) {
    return {StatusCode::FAILED_PRECONDITION, "gateway ingress is not enabled on this daemon"};
  }
  if (req->execution_class() != v2::PLAN_EXECUTION_CLASS_TERMINAL_ONLY) {
    return {StatusCode::FAILED_PRECONDITION, "daemon ingress currently supports terminal_only execution only"};
  }

  const auto& plan = req->plan();
  const std::string request_id = request_id_from_plan(plan);
  const int64_t execution_started_ms = absl::ToUnixMillis(absl::Now());

  absl::flat_hash_map<std::string, const tensorcast::plan::v1::PlanStep*> steps_by_id;
  steps_by_id.reserve(static_cast<size_t>(plan.steps_size()));
  for (const auto& step : plan.steps()) {
    const auto [it, inserted] = steps_by_id.emplace(step.step_id(), &step);
    if (!inserted) {
      return {StatusCode::INVALID_ARGUMENT, absl::StrCat("Duplicate step id: ", step.step_id())};
    }
  }

  absl::flat_hash_map<std::string, std::vector<std::string>> dependents;
  absl::flat_hash_map<std::string, int> remaining_deps;
  dependents.reserve(steps_by_id.size());
  remaining_deps.reserve(steps_by_id.size());
  for (const auto& step : plan.steps()) {
    remaining_deps.emplace(step.step_id(), step.depends_on_size());
  }
  for (const auto& step : plan.steps()) {
    for (const auto& dep : step.depends_on()) {
      if (!steps_by_id.contains(dep)) {
        return {StatusCode::INVALID_ARGUMENT, absl::StrCat("Unknown dependency step id: ", dep)};
      }
      dependents[dep].push_back(step.step_id());
    }
  }

  std::vector<std::string> ready;
  for (const auto& step : plan.steps()) {
    if (remaining_deps[step.step_id()] == 0) {
      ready.push_back(step.step_id());
    }
  }
  if (!plan.steps().empty() && ready.empty()) {
    return {StatusCode::FAILED_PRECONDITION, "Plan has no runnable steps (cycle detected)"};
  }

  tensorcast::node_agent::v1::ExecutePlanResponse terminal;
  terminal.set_request_id(request_id);

  bool prior_failure = false;
  absl::flat_hash_map<std::string, bool> completed;
  completed.reserve(steps_by_id.size());

  auto add_cancelled = [&](const tensorcast::plan::v1::PlanStep& step, std::string_view message) {
    auto* step_result = terminal.add_steps();
    step_result->set_step_id(step.step_id());
    step_result->set_target_id(step.target().target_id());
    step_result->set_action(action_name(step.action()));
    fill_status_cancelled(step_result->mutable_status(), message);
  };

  auto fill_worker_prefetch = [&](const tensorcast::plan::v1::PlanStep& step,
                                  tensorcast::node_agent::v1::StepResult* step_result) {
    const auto& action = step.action().prefetch();
    const auto budget_or = remaining_deadline_ms(plan.context(), execution_started_ms);
    if (budget_or.has_value() && *budget_or <= 0) {
      fill_status_failed(step_result->mutable_status(), "CallContext deadline exceeded", StatusCode::DEADLINE_EXCEEDED);
      return;
    }

    const int32_t device_id = action.device_id();
    v2::MaterializeReplicaRequest materialize_req;
    materialize_req.mutable_selection()->CopyFrom(action.selection());
    materialize_req.set_pid(::getpid());
    materialize_req.set_lease_mode(v2::LEASE_MODE_NO_LEASE);
    materialize_req.set_wait_for_completion(false);
    if (device_id == kCpuDeviceId) {
      materialize_req.set_target_device_type(v2::DEVICE_TYPE_CPU);
    } else {
      const auto device_key = store::DeviceRegistry::instance().gpu_key(device_id);
      if (device_key.uuid.empty()) {
        fill_status_failed(
            step_result->mutable_status(),
            absl::StrCat("prefetch failed: Device ordinal ", device_id, " not found in daemon device map"),
            StatusCode::FAILED_PRECONDITION);
        return;
      }
      materialize_req.set_target_device_type(v2::DEVICE_TYPE_GPU);
      materialize_req.set_device_uuid(device_key.uuid);
    }
    materialize_req.set_replica_uuid(replica_uuid_for_action(
        plan.context(),
        "prefetch",
        step.target().target_id(),
        action.selection(),
        device_id,
        std::nullopt,
        std::nullopt,
        absl::StrCat(request_id, "-", step.step_id())));

    RpcContext step_rctx{"ExecutePlanPrefetch", *ctx, opts_.allow_high_card_attrs};
    v2::MaterializeReplicaResponse materialize_resp;
    const Status status =
        materialization_controller_->materialize_replica(step_rctx, materialize_req, materialize_resp);
    if (!status.ok()) {
      fill_status_failed(
          step_result->mutable_status(),
          absl::StrCat("prefetch failed: ", status.error_message()),
          status.error_code());
      return;
    }
    fill_status_success(step_result->mutable_status(), "prefetch issued");
  };

  auto fill_worker_prefetch_set = [&](const tensorcast::plan::v1::PlanStep& step,
                                      tensorcast::node_agent::v1::StepResult* step_result) {
    const auto resolved_or = resolve_artifact_set(step.action().prefetch_set());
    if (!resolved_or.ok()) {
      fill_status_failed(
          step_result->mutable_status(),
          absl::StrCat("prefetch_set failed: ", resolved_or.status().message()),
          resolved_or.status().code() == absl::StatusCode::kInvalidArgument ? StatusCode::INVALID_ARGUMENT
                                                                            : StatusCode::FAILED_PRECONDITION);
      return;
    }

    auto* artifact_set_result = step_result->mutable_artifact_set_result();
    artifact_set_result->set_set_digest_hex(step.action().prefetch_set().artifact_set().set_digest_hex());

    const int32_t device_id = step.action().prefetch_set().device_id();
    std::optional<std::string> device_uuid;
    if (device_id != kCpuDeviceId) {
      const auto device_key = store::DeviceRegistry::instance().gpu_key(device_id);
      if (device_key.uuid.empty()) {
        fill_status_failed(
            step_result->mutable_status(),
            absl::StrCat("prefetch_set failed: Device ordinal ", device_id, " not found in daemon device map"),
            StatusCode::FAILED_PRECONDITION);
        return;
      }
      device_uuid = device_key.uuid;
    }

    for (size_t index = 0; index < resolved_or->size(); ++index) {
      const auto& item = resolved_or->at(index);
      auto* outcome = artifact_set_result->add_outcomes();
      fill_item_identity(outcome->mutable_item_identity(), item.identity);
      outcome->set_artifact_id(item.identity.artifact_id);

      const auto budget_or = remaining_deadline_ms(plan.context(), execution_started_ms);
      if (budget_or.has_value() && *budget_or <= 0) {
        fill_status_failed(outcome->mutable_status(), "CallContext deadline exceeded", StatusCode::DEADLINE_EXCEEDED);
        continue;
      }

      v2::MaterializeReplicaRequest materialize_req;
      materialize_req.mutable_selection()->CopyFrom(item.selection);
      materialize_req.set_pid(::getpid());
      materialize_req.set_lease_mode(v2::LEASE_MODE_NO_LEASE);
      materialize_req.set_wait_for_completion(false);
      if (device_id == kCpuDeviceId) {
        materialize_req.set_target_device_type(v2::DEVICE_TYPE_CPU);
      } else {
        materialize_req.set_target_device_type(v2::DEVICE_TYPE_GPU);
        materialize_req.set_device_uuid(*device_uuid);
      }
      materialize_req.set_replica_uuid(replica_uuid_for_action(
          plan.context(),
          "prefetch_set",
          step.target().target_id(),
          item.selection,
          device_id,
          std::nullopt,
          step.action().prefetch_set().artifact_set().set_digest_hex(),
          absl::StrCat(request_id, "-", step.step_id(), "-", index)));

      RpcContext step_rctx{"ExecutePlanPrefetchSet", *ctx, opts_.allow_high_card_attrs};
      v2::MaterializeReplicaResponse materialize_resp;
      const Status status =
          materialization_controller_->materialize_replica(step_rctx, materialize_req, materialize_resp);
      if (!status.ok()) {
        fill_status_failed(
            outcome->mutable_status(),
            absl::StrCat("prefetch_set failed: ", status.error_message()),
            status.error_code());
        continue;
      }
      fill_status_success(outcome->mutable_status(), "local_replica_ready");
    }

    summarize_artifact_set_outcomes(*artifact_set_result, step_result->mutable_status());
  };

  auto fill_worker_pin = [&](const tensorcast::plan::v1::PlanStep& step,
                             tensorcast::node_agent::v1::StepResult* step_result) {
    const auto& action = step.action().pin_device_residency();
    const auto budget_or = remaining_deadline_ms(plan.context(), execution_started_ms);
    if (budget_or.has_value() && *budget_or <= 0) {
      fill_status_failed(step_result->mutable_status(), "CallContext deadline exceeded", StatusCode::DEADLINE_EXCEEDED);
      return;
    }

    v2::CreatePlacementLeaseRequest lease_req;
    lease_req.set_artifact_id(action.selection().artifact_id());
    lease_req.set_view_id(action.selection().view_id());
    lease_req.set_device_id(action.device_id());
    if (action.has_ttl_ms()) {
      lease_req.set_ttl_ms(static_cast<uint32_t>(action.ttl_ms()));
    }
    RpcContext step_rctx{"ExecutePlanPin", *ctx, opts_.allow_high_card_attrs};
    v2::CreatePlacementLeaseResponse lease_resp;
    const Status status = lease_controller_->create_placement_lease(step_rctx, lease_req, lease_resp);
    if (!status.ok()) {
      fill_status_failed(
          step_result->mutable_status(), absl::StrCat("pin failed: ", status.error_message()), status.error_code());
      return;
    }
    fill_status_success(step_result->mutable_status(), "placement pin created");
  };

  auto fill_worker_unpin = [&](const tensorcast::plan::v1::PlanStep& step,
                               tensorcast::node_agent::v1::StepResult* step_result) {
    const auto budget_or = remaining_deadline_ms(plan.context(), execution_started_ms);
    if (budget_or.has_value() && *budget_or <= 0) {
      fill_status_failed(step_result->mutable_status(), "CallContext deadline exceeded", StatusCode::DEADLINE_EXCEEDED);
      return;
    }

    v2::ReleasePlacementLeaseRequest lease_req;
    lease_req.set_lease_token(step.action().unpin_device_residency().capability_token());
    RpcContext step_rctx{"ExecutePlanUnpin", *ctx, opts_.allow_high_card_attrs};
    v2::ReleasePlacementLeaseResponse lease_resp;
    const Status status = lease_controller_->release_placement_lease(step_rctx, lease_req, lease_resp);
    if (!status.ok()) {
      fill_status_failed(
          step_result->mutable_status(), absl::StrCat("unpin failed: ", status.error_message()), status.error_code());
      return;
    }
    if (!lease_resp.released()) {
      fill_status_failed(
          step_result->mutable_status(), "placement pin release failed", StatusCode::FAILED_PRECONDITION);
      return;
    }
    fill_status_success(step_result->mutable_status(), "placement pin released");
  };

  while (!ready.empty()) {
    const std::string step_id = ready.front();
    ready.erase(ready.begin());
    const auto* step = steps_by_id[step_id];
    completed[step_id] = true;

    if (prior_failure) {
      add_cancelled(*step, "skipped due to prior failure");
    } else {
      auto* step_result = terminal.add_steps();
      step_result->set_step_id(step->step_id());
      step_result->set_target_id(step->target().target_id());
      step_result->set_action(action_name(step->action()));

      switch (step->target().target_type()) {
        case tensorcast::plan::v1::TARGET_TYPE_WORKER:
          if (step->target().target_id() != identity_store_->daemon_id()) {
            fill_status_failed(
                step_result->mutable_status(),
                "worker target does not match this agent",
                StatusCode::FAILED_PRECONDITION);
            break;
          }
          if (req->dry_run()) {
            fill_status_success(step_result->mutable_status(), "dry-run");
            break;
          }
          switch (step->action().kind_case()) {
            case tensorcast::plan::v1::PlanAction::kPrefetch:
              fill_worker_prefetch(*step, step_result);
              break;
            case tensorcast::plan::v1::PlanAction::kPrefetchSet:
              fill_worker_prefetch_set(*step, step_result);
              break;
            case tensorcast::plan::v1::PlanAction::kPinDeviceResidency:
              fill_worker_pin(*step, step_result);
              break;
            case tensorcast::plan::v1::PlanAction::kUnpinDeviceResidency:
              fill_worker_unpin(*step, step_result);
              break;
            default:
              fill_status_failed(step_result->mutable_status(), "unknown action", StatusCode::INVALID_ARGUMENT);
              break;
          }
          break;
        case tensorcast::plan::v1::TARGET_TYPE_INSTANCE:
          fill_status_failed(
              step_result->mutable_status(),
              "instance target does not match this agent",
              StatusCode::FAILED_PRECONDITION);
          break;
        case tensorcast::plan::v1::TARGET_TYPE_CLUSTER:
          fill_status_failed(
              step_result->mutable_status(),
              "cluster targets are not executable via daemon ingress",
              StatusCode::FAILED_PRECONDITION);
          break;
        default:
          fill_status_failed(step_result->mutable_status(), "unknown target type", StatusCode::INVALID_ARGUMENT);
          break;
      }

      if (step_result->status().state() != tensorcast::node_agent::v1::OPERATION_STATE_SUCCESS) {
        prior_failure = true;
      }
    }

    for (const auto& child : dependents[step_id]) {
      auto it = remaining_deps.find(child);
      if (it == remaining_deps.end()) {
        continue;
      }
      --it->second;
      if (it->second == 0 && !completed.contains(child)) {
        ready.push_back(child);
      }
    }
  }

  for (const auto& step : plan.steps()) {
    if (!completed.contains(step.step_id())) {
      add_cancelled(step, "skipped due to prior failure");
    }
  }

  bool ok = true;
  for (const auto& step_result : terminal.steps()) {
    if (step_result.status().state() != tensorcast::node_agent::v1::OPERATION_STATE_SUCCESS) {
      ok = false;
      break;
    }
  }
  terminal.set_ok(ok);

  auto terminal_bytes_or = serialize_deterministic(terminal);
  if (!terminal_bytes_or.ok()) {
    return {StatusCode::INTERNAL, std::string(terminal_bytes_or.status().message())};
  }
  resp->set_request_id(request_id);
  resp->set_ok(ok);
  resp->set_terminal_result(*terminal_bytes_or);
  return Status::OK;
}

} // namespace tensorcast::daemon
