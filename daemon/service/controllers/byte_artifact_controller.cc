// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/byte_artifact_controller.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
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
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/components/endpoint_id.h"
#include "core/store/materialization/dataplane/contracts/inline_buffer_loader.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"
#include "core/store/runtime/ingestion/artifact_lowering_plan.h"
#include "daemon/service/artifact_profile_registry.h"
#include "daemon/util/grpc_daemon_transport.h"
#include "daemon/util/grpc_peer_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/executors/thread_factory/NamedThreadFactory.h"
#include "folly/futures/Future.h"
#include "grpcpp/grpcpp.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

const ArtifactProfileRuntime& byte_artifact_runtime() {
  return ArtifactProfileRegistry::runtime_for_profile(ArtifactProfileRegistry::Profile::kByteArtifact);
}

absl::Status validate_batch_selection(const tensorcast::common::v1::ArtifactSelection& selection) {
  return byte_artifact_runtime().validate_batch_selection(selection);
}

struct BatchItemSlotToken {
  std::optional<std::uint64_t> slot_index;
  std::optional<std::uint64_t> slot_generation;
};

absl::flat_hash_map<std::string, BatchItemSlotToken> collect_batch_item_slot_tokens(const v2::TargetLayout& layout) {
  absl::flat_hash_map<std::string, BatchItemSlotToken> tokens;
  tokens.reserve(static_cast<std::size_t>(layout.offsets_size()));
  for (const auto& offset : layout.offsets()) {
    BatchItemSlotToken token;
    if (offset.has_slot_index()) {
      token.slot_index = offset.slot_index();
    }
    if (offset.has_slot_generation()) {
      token.slot_generation = offset.slot_generation();
    }
    if (token.slot_index.has_value() || token.slot_generation.has_value()) {
      tokens.emplace(offset.name(), std::move(token));
    }
  }
  return tokens;
}

void attach_slot_tokens_to_outcomes(
    const absl::flat_hash_map<std::string, BatchItemSlotToken>& slot_tokens,
    google::protobuf::RepeatedPtrField<v2::BatchItemOutcome>* outcomes) {
  if (outcomes == nullptr || slot_tokens.empty()) {
    return;
  }
  for (auto& outcome : *outcomes) {
    const auto it = slot_tokens.find(outcome.artifact_id());
    if (it == slot_tokens.end()) {
      continue;
    }
    if (it->second.slot_index.has_value()) {
      outcome.set_slot_index(*it->second.slot_index);
    }
    if (it->second.slot_generation.has_value()) {
      outcome.set_slot_generation(*it->second.slot_generation);
    }
  }
}

std::string_view host_region_class_label(IpcRegionRegistry::HostRegionClass host_region_class) {
  switch (host_region_class) {
    case IpcRegionRegistry::HostRegionClass::kScratch:
      return "scratch";
    case IpcRegionRegistry::HostRegionClass::kAllocator:
      return "allocator";
    case IpcRegionRegistry::HostRegionClass::kNone:
    default:
      return "none";
  }
}

v2::BatchItemOutcome make_outcome(
    std::string_view artifact_id,
    v2::BatchItemStatus status,
    std::string_view message = "",
    const BatchItemSlotToken* slot_token = nullptr) {
  v2::BatchItemOutcome outcome;
  outcome.set_artifact_id(std::string(artifact_id));
  outcome.set_status(status);
  if (!message.empty()) {
    outcome.set_message(std::string(message));
  }
  if (slot_token != nullptr) {
    if (slot_token->slot_index.has_value()) {
      outcome.set_slot_index(*slot_token->slot_index);
    }
    if (slot_token->slot_generation.has_value()) {
      outcome.set_slot_generation(*slot_token->slot_generation);
    }
  }
  return outcome;
}

v2::HomeBatchGetItem make_home_get_item(
    std::string_view artifact_id,
    v2::BatchItemStatus status,
    std::string_view message = "") {
  v2::HomeBatchGetItem item;
  item.set_artifact_id(std::string(artifact_id));
  item.set_status(status);
  if (!message.empty()) {
    item.set_message(std::string(message));
  }
  return item;
}

std::chrono::milliseconds inter_daemon_home_rpc_timeout(const ByteArtifactController::Options& options) {
  return std::max(options.routing.route_staleness_budget, options.routing.lease_ttl);
}

v2::BatchItemStatus batch_item_status_from_absl_status(const absl::Status& status) {
  if (status.ok()) {
    return v2::BATCH_ITEM_STATUS_OK;
  }
  if (absl::IsInvalidArgument(status)) {
    return v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT;
  }
  if (absl::IsFailedPrecondition(status) || absl::IsPermissionDenied(status)) {
    return v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION;
  }
  if (absl::IsNotFound(status)) {
    return v2::BATCH_ITEM_STATUS_MISS;
  }
  if (absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status)) {
    return v2::BATCH_ITEM_STATUS_UNAVAILABLE;
  }
  return v2::BATCH_ITEM_STATUS_INTERNAL_ERROR;
}

bool is_non_actionable_policy_path_error(const absl::Status& status) {
  return absl::IsNotFound(status) || absl::IsFailedPrecondition(status) || absl::IsDataLoss(status) ||
      absl::IsInvalidArgument(status);
}

PolicyVisibilityPathKind policy_visibility_path_kind_from_source(PersistenceManager::PolicySourceKind kind) {
  switch (kind) {
    case PersistenceManager::PolicySourceKind::kSharedDisk:
      return PolicyVisibilityPathKind::kSharedDisk;
    case PersistenceManager::PolicySourceKind::kUnspecified:
    default:
      return PolicyVisibilityPathKind::kUnspecified;
  }
}

class SeekableSourceLoader final : public store::IArtifactLoader {
 public:
  SeekableSourceLoader(std::shared_ptr<store::loader::SeekableSource> source, std::uint64_t size_bytes)
      : source_(std::move(source)), size_bytes_(size_bytes) {}

  absl::Status initialize() override {
    initialized_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<std::uint64_t> get_artifact_size() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("SeekableSourceLoader not initialized");
    }
    return size_bytes_;
  }

  absl::StatusOr<std::unique_ptr<store::loader::SeekableSource>> open_source() override {
    if (!initialized_) {
      return absl::FailedPreconditionError("SeekableSourceLoader not initialized");
    }
    if (!source_) {
      return absl::FailedPreconditionError("SeekableSourceLoader requires source");
    }

    class SourceRef final : public store::loader::SeekableSource {
     public:
      explicit SourceRef(std::shared_ptr<store::loader::SeekableSource> source) : source_(std::move(source)) {}

      [[nodiscard]] uint64_t total_bytes() const override {
        return source_->total_bytes();
      }

      absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
        return source_->read(dst, max_bytes);
      }

      absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
        return source_->read_at(offset, dst, bytes);
      }

      [[nodiscard]] bool supports_direct_write_at() const override {
        return source_->supports_direct_write_at();
      }

      [[nodiscard]] bool supports_batched_direct_write_at() const override {
        return source_->supports_batched_direct_write_at();
      }

      absl::StatusOr<size_t> read_into_at(
          uint64_t src_offset,
          uint64_t dest_va_offset,
          size_t bytes,
          const store::DirectWriteGrant& grant) override {
        return source_->read_into_at(src_offset, dest_va_offset, bytes, grant);
      }

      absl::StatusOr<size_t> readv_into_at(
          absl::Span<const store::loader::DirectWriteOp> ops,
          const store::DirectWriteGrant& grant) override {
        return source_->readv_into_at(ops, grant);
      }

     private:
      std::shared_ptr<store::loader::SeekableSource> source_;
    };

    return std::unique_ptr<store::loader::SeekableSource>(std::make_unique<SourceRef>(source_));
  }

 private:
  bool initialized_{false};
  std::shared_ptr<store::loader::SeekableSource> source_;
  std::uint64_t size_bytes_{0};
};

store::loading::MaterializeHints build_lowering_hints(std::string_view artifact_id, std::string_view operation_id) {
  store::loading::MaterializeHints hints;
  hints.artifact_id = std::string(artifact_id);
  if (!operation_id.empty()) {
    hints.transport_request_id = std::string(operation_id);
  }
  return hints;
}

absl::StatusOr<store::runtime::ingestion::ArtifactLoweringPlan> build_into_target_lowering_plan(
    std::string_view artifact_id,
    const store::DeviceKey& target_device,
    const store::loading::IntoTargetLayout& target_layout,
    std::unique_ptr<store::IArtifactLoader> loader,
    std::uint64_t payload_bytes,
    store::loading::MaterializationSource source_kind,
    std::string_view operation_id) {
  if (loader == nullptr) {
    return absl::InvalidArgumentError("into-target lowering requires loader");
  }
  return store::runtime::ingestion::lower_to_artifact_plan(
      store::runtime::ingestion::LowerToArtifactPlanRequest{
          .identity =
              store::runtime::ingestion::ArtifactLoweringIdentity{
                  .logical_artifact_id = std::string(artifact_id),
                  .request_id = std::string(operation_id),
              },
          .target_device = target_device,
          .source_loader = std::move(loader),
          .selection_identity =
              tensorcast::common::SelectionIdentity{
                  .artifact_id = std::string(artifact_id),
                  .logical_layout_hash = tensorcast::common::compute_byte_artifact_logical_layout_hash_bytes(),
                  .selection_hash = tensorcast::common::compute_byte_artifact_selection_hash_bytes(),
              },
          .expected_size_bytes = payload_bytes,
          .generation = 1,
          .hints = build_lowering_hints(artifact_id, operation_id),
          .source_kind = source_kind,
          .into_target = target_layout,
      });
}

struct LoaderSourceResolution {
  std::unique_ptr<store::IArtifactLoader> loader;
  store::loading::MaterializationSource source_kind{store::loading::MaterializationSource::kUnspecified};
  std::uint64_t payload_bytes{0};
};

struct BatchPayloadPackEntry {
  std::string artifact_id;
  std::uint64_t payload_size_bytes{0};
  std::shared_ptr<const std::string> inline_payload;
  std::optional<BodyHandle> body_handle;
  std::optional<BodyDescriptor> body_descriptor;
  std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
  std::uint64_t backing_instance_generation{0};
  std::string digest_alg;
  std::string digest_hex;
  absl::Time capability_expires_at{absl::InfiniteFuture()};
};

struct SourceLayoutBatchPayloadEntry {
  std::string artifact_id;
  std::uint64_t payload_size_bytes{0};
  ByteArtifactRegionLayout::HostSharedSourceSpan source_span;
  std::string digest_alg;
  std::string digest_hex;
  absl::Time capability_expires_at{absl::InfiniteFuture()};
};

struct PlannedBatchPayload {
  v2::BatchPayloadManifest manifest;
  std::vector<std::size_t> source_indices;
  std::vector<v2::BatchPayloadSlice> slices;
  absl::Time capability_expires_at{absl::InfiniteFuture()};
};

struct PackedBatchPayload {
  v2::BatchPayloadManifest manifest;
  std::shared_ptr<const std::string> payload;
  std::vector<std::size_t> source_indices;
  std::vector<v2::BatchPayloadSlice> slices;
  absl::Time capability_expires_at{absl::InfiniteFuture()};
};

struct HomeBatchGetResponseShape {
  std::size_t items{0};
  std::size_t ok_items{0};
  std::size_t inline_items{0};
  std::size_t payload_ref_items{0};
  std::size_t batch_slice_items{0};
  std::size_t transports{0};
  std::size_t communicator_transports{0};
  std::size_t grpc_chunk_transports{0};
};

HomeBatchGetResponseShape inspect_home_batch_get_response_shape(const v2::HomeBatchGetResponse& resp) {
  HomeBatchGetResponseShape shape;
  shape.items = static_cast<std::size_t>(resp.items_size());
  shape.transports = static_cast<std::size_t>(resp.batch_transports_size());
  for (const auto& item : resp.items()) {
    if (item.status() == v2::BATCH_ITEM_STATUS_OK) {
      ++shape.ok_items;
    }
    if (!item.inline_payload().empty()) {
      ++shape.inline_items;
    }
    if (!item.payload_ref().empty()) {
      ++shape.payload_ref_items;
    }
    if (item.has_batch_payload_slice()) {
      ++shape.batch_slice_items;
    }
  }
  for (const auto& transport : resp.batch_transports()) {
    if (transport.has_communicator_source()) {
      ++shape.communicator_transports;
    } else if (transport.has_grpc_chunk_ref()) {
      ++shape.grpc_chunk_transports;
    }
  }
  return shape;
}

class SourceSlice final : public store::loader::SeekableSource {
 public:
  SourceSlice(
      std::shared_ptr<store::loader::SeekableSource> source,
      std::uint64_t base_offset,
      std::uint64_t length,
      std::shared_ptr<std::mutex> source_mutex = nullptr)
      : source_(std::move(source)),
        source_mutex_(std::move(source_mutex)),
        base_offset_(base_offset),
        length_(length) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return length_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or.status();
    }
    cursor_ += *read_or;
    return *read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= length_ || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t bounded_bytes = static_cast<size_t>(std::min<uint64_t>(bytes, length_ - offset));
    if (source_mutex_ != nullptr) {
      std::lock_guard<std::mutex> lock(*source_mutex_);
      return source_->read_at(base_offset_ + offset, dst, bounded_bytes);
    }
    return source_->read_at(base_offset_ + offset, dst, bounded_bytes);
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return source_->supports_direct_write_at();
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const store::DirectWriteGrant& grant) override {
    if (src_offset >= length_ || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t bounded_bytes = static_cast<size_t>(std::min<uint64_t>(bytes, length_ - src_offset));
    if (source_mutex_ != nullptr) {
      std::lock_guard<std::mutex> lock(*source_mutex_);
      return source_->read_into_at(base_offset_ + src_offset, dest_va_offset, bounded_bytes, grant);
    }
    return source_->read_into_at(base_offset_ + src_offset, dest_va_offset, bounded_bytes, grant);
  }

 private:
  std::shared_ptr<store::loader::SeekableSource> source_;
  std::shared_ptr<std::mutex> source_mutex_;
  std::uint64_t base_offset_{0};
  std::uint64_t length_{0};
  std::uint64_t cursor_{0};
};

std::unique_ptr<store::IArtifactLoader> make_loader_from_payload_slice(
    const std::shared_ptr<const std::string>& payload,
    std::uint64_t offset,
    std::uint64_t length) {
  return std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
      .data = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data() + offset)),
      .size_bytes = length,
  });
}

std::unique_ptr<store::IArtifactLoader> make_loader_from_source_slice(
    std::shared_ptr<store::loader::SeekableSource> source,
    std::uint64_t offset,
    std::uint64_t length,
    std::shared_ptr<std::mutex> source_mutex = nullptr) {
  return std::make_unique<SeekableSourceLoader>(
      std::make_shared<SourceSlice>(std::move(source), offset, length, std::move(source_mutex)), length);
}

struct PutRemoteCommunicatorSourceEligibility {
  bool remote{false};
  bool source_supports_direct_write{false};
  bool source_supports_batched_direct_write{false};
  bool direct_remote_slice{false};
};

PutRemoteCommunicatorSourceEligibility classify_put_remote_communicator_source(
    const PayloadTransportBroker::BatchPayloadSource& source) {
  PutRemoteCommunicatorSourceEligibility eligibility;
  eligibility.remote = source.remote;
  eligibility.source_supports_direct_write = source.source != nullptr && source.source->supports_direct_write_at();
  eligibility.source_supports_batched_direct_write =
      source.source != nullptr && source.source->supports_batched_direct_write_at();
  eligibility.direct_remote_slice = eligibility.remote && eligibility.source_supports_direct_write;
  return eligibility;
}

absl::StatusOr<std::shared_ptr<const std::string>> mirror_seekable_source_payload(
    const std::shared_ptr<store::loader::SeekableSource>& source,
    std::uint64_t total_bytes) {
  if (source == nullptr) {
    return absl::InvalidArgumentError("mirror_seekable_source_payload requires source");
  }
  if (total_bytes == 0) {
    return absl::InvalidArgumentError("mirror_seekable_source_payload requires non-empty payload");
  }
  if (total_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return absl::OutOfRangeError("mirror_seekable_source_payload exceeds host memory limits");
  }
  auto payload = std::make_shared<std::string>();
  payload->resize(static_cast<std::size_t>(total_bytes));
  std::size_t copied = 0;
  while (copied < payload->size()) {
    auto read_or = source->read_at(copied, payload->data() + copied, payload->size() - copied);
    if (!read_or.ok()) {
      return read_or.status();
    }
    if (*read_or == 0) {
      return absl::DataLossError("mirror_seekable_source_payload terminated before expected size");
    }
    copied += *read_or;
  }
  return std::shared_ptr<const std::string>(std::move(payload));
}

absl::Status validate_batch_payload_pack_entry(const BatchPayloadPackEntry& entry) {
  const bool has_inline_payload = static_cast<bool>(entry.inline_payload);
  const bool has_body_handle = entry.body_handle.has_value();
  if (has_inline_payload == has_body_handle) {
    return absl::InvalidArgumentError("batch payload entry requires exactly one source");
  }
  if (entry.payload_size_bytes == 0) {
    return absl::InvalidArgumentError("batch payload entry payload_size_bytes must be > 0");
  }
  if (has_inline_payload && entry.inline_payload->size() != entry.payload_size_bytes) {
    return absl::InvalidArgumentError("batch payload entry inline payload size mismatch");
  }
  if (has_body_handle && !entry.body_descriptor.has_value()) {
    return absl::InvalidArgumentError("batch payload entry body descriptor is required");
  }
  if (has_body_handle && entry.body_descriptor->size_bytes != entry.payload_size_bytes) {
    return absl::InvalidArgumentError("batch payload entry body descriptor size mismatch");
  }
  if (entry.digest_alg.empty() != entry.digest_hex.empty()) {
    return absl::InvalidArgumentError("batch payload entry digest_alg and digest_hex must both be set");
  }
  return absl::OkStatus();
}

absl::Status validate_source_layout_batch_payload_entry(const SourceLayoutBatchPayloadEntry& entry) {
  if (entry.artifact_id.empty()) {
    return absl::InvalidArgumentError("source-layout batch payload entry requires artifact_id");
  }
  if (entry.payload_size_bytes == 0) {
    return absl::InvalidArgumentError("source-layout batch payload entry payload_size_bytes must be > 0");
  }
  if (entry.source_span.data == nullptr) {
    return absl::InvalidArgumentError("source-layout batch payload entry requires source span data");
  }
  if (entry.source_span.length != entry.payload_size_bytes) {
    return absl::InvalidArgumentError("source-layout batch payload entry source span size mismatch");
  }
  if (entry.source_span.keepalive == nullptr) {
    return absl::InvalidArgumentError("source-layout batch payload entry requires source span keepalive");
  }
  if (entry.digest_alg.empty() != entry.digest_hex.empty()) {
    return absl::InvalidArgumentError("source-layout batch payload entry digest_alg and digest_hex must both be set");
  }
  return absl::OkStatus();
}

absl::Status fill_batch_payload_pack_entry(const BatchPayloadPackEntry& entry, char* dst) {
  auto validate_status = validate_batch_payload_pack_entry(entry);
  if (!validate_status.ok()) {
    return validate_status;
  }
  if (entry.inline_payload) {
    std::memcpy(dst, entry.inline_payload->data(), static_cast<std::size_t>(entry.payload_size_bytes));
    return absl::OkStatus();
  }
  return entry.body_handle->read_into_range(
      /*offset=*/0, dst, static_cast<std::size_t>(entry.payload_size_bytes));
}

absl::StatusOr<std::string> issue_payload_ref_for_batch_payload_pack_entry(
    PayloadTransportBroker& payload_transport_broker,
    const BatchPayloadPackEntry& entry,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id) {
  auto validate_status = validate_batch_payload_pack_entry(entry);
  if (!validate_status.ok()) {
    return validate_status;
  }
  if (entry.inline_payload) {
    return payload_transport_broker.issue_payload_ref(
        entry.artifact_id, entry.inline_payload, direction, operation_id, entry.capability_expires_at);
  }
  if (entry.body_descriptor.has_value() &&
      (!entry.body_descriptor->payload_digest_alg.empty() != !entry.body_descriptor->payload_digest_hex.empty())) {
    return absl::InvalidArgumentError("body descriptor digest metadata must be fully set or fully empty");
  }
  if (entry.body_descriptor.has_value() && entry.body_descriptor->payload_digest_alg.empty()) {
    auto payload_or = entry.body_handle->read_all_bytes();
    if (!payload_or.ok()) {
      return payload_or.status();
    }
    return payload_transport_broker.issue_payload_ref(
        entry.artifact_id,
        std::make_shared<const std::string>(std::move(*payload_or)),
        direction,
        operation_id,
        entry.capability_expires_at);
  }
  return payload_transport_broker.issue_payload_ref(
      entry.artifact_id,
      *entry.body_handle,
      *entry.body_descriptor,
      entry.backing_identity,
      entry.backing_instance_generation == 0 ? entry.body_handle->binding_generation()
                                             : entry.backing_instance_generation,
      direction,
      operation_id,
      entry.capability_expires_at);
}

const v2::BatchPayloadEntry* find_batch_payload_entry(
    const v2::BatchPayloadManifest& manifest,
    std::string_view artifact_id,
    std::uint64_t offset,
    std::uint64_t length) {
  for (const auto& entry : manifest.entries()) {
    if (entry.artifact_id() == artifact_id && entry.offset() == offset && entry.length() == length) {
      return &entry;
    }
  }
  return nullptr;
}

absl::Duration peer_batch_transport_support_cache_ttl() {
  return absl::Seconds(30);
}

absl::StatusOr<std::vector<PlannedBatchPayload>> plan_batch_payload_entries(
    const std::vector<BatchPayloadPackEntry>& entries,
    std::uint64_t max_payload_bytes,
    std::uint32_t max_items) {
  std::vector<PlannedBatchPayload> packs;
  if (entries.empty()) {
    return packs;
  }

  struct PendingPack {
    std::vector<std::size_t> entry_indices;
    std::uint64_t total_bytes{0};
    absl::Time capability_expires_at{absl::InfiniteFuture()};
  };

  const auto flush_pack = [&](const PendingPack& pending) -> absl::StatusOr<PlannedBatchPayload> {
    PlannedBatchPayload packed;
    packed.source_indices = pending.entry_indices;
    packed.capability_expires_at = pending.capability_expires_at;

    std::uint64_t offset = 0;
    for (const auto entry_index : pending.entry_indices) {
      const auto& entry = entries[entry_index];
      auto validate_status = validate_batch_payload_pack_entry(entry);
      if (!validate_status.ok()) {
        return validate_status;
      }

      auto* manifest_entry = packed.manifest.add_entries();
      manifest_entry->set_artifact_id(entry.artifact_id);
      manifest_entry->set_offset(offset);
      manifest_entry->set_length(entry.payload_size_bytes);
      manifest_entry->set_digest_alg(entry.digest_alg);
      manifest_entry->set_digest_hex(entry.digest_hex);

      v2::BatchPayloadSlice slice;
      slice.set_offset(offset);
      slice.set_length(entry.payload_size_bytes);
      packed.slices.push_back(std::move(slice));
      offset += entry.payload_size_bytes;
    }
    packed.manifest.set_total_size(offset);
    return packed;
  };

  PendingPack pending;
  for (std::size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
    const auto& entry = entries[entry_index];
    auto validate_status = validate_batch_payload_pack_entry(entry);
    if (!validate_status.ok()) {
      return validate_status;
    }
    const std::uint64_t entry_bytes = entry.payload_size_bytes;
    if (max_payload_bytes != 0 && entry_bytes > max_payload_bytes) {
      return absl::InvalidArgumentError("batch payload entry exceeds max_payload_bytes");
    }

    const bool reaches_item_limit = max_items != 0 && pending.entry_indices.size() >= max_items;
    const bool reaches_byte_limit =
        max_payload_bytes != 0 && pending.total_bytes != 0 && pending.total_bytes + entry_bytes > max_payload_bytes;
    if (!pending.entry_indices.empty() && (reaches_item_limit || reaches_byte_limit)) {
      auto packed_or = flush_pack(pending);
      if (!packed_or.ok()) {
        return packed_or.status();
      }
      packs.push_back(std::move(*packed_or));
      pending = PendingPack{};
    }

    pending.entry_indices.push_back(entry_index);
    pending.total_bytes += entry_bytes;
    pending.capability_expires_at = std::min(pending.capability_expires_at, entry.capability_expires_at);
  }

  if (!pending.entry_indices.empty()) {
    auto packed_or = flush_pack(pending);
    if (!packed_or.ok()) {
      return packed_or.status();
    }
    packs.push_back(std::move(*packed_or));
  }
  return packs;
}

absl::StatusOr<std::vector<PlannedBatchPayload>> plan_source_layout_batch_payload_entries(
    const std::vector<SourceLayoutBatchPayloadEntry>& entries,
    std::uint64_t max_payload_bytes,
    std::uint32_t max_items) {
  std::vector<PlannedBatchPayload> packs;
  if (entries.empty()) {
    return packs;
  }

  struct PendingPack {
    std::vector<std::size_t> entry_indices;
    std::uint64_t total_bytes{0};
    absl::Time capability_expires_at{absl::InfiniteFuture()};
  };

  const auto flush_pack = [&](const PendingPack& pending) -> absl::StatusOr<PlannedBatchPayload> {
    PlannedBatchPayload packed;
    packed.source_indices = pending.entry_indices;
    packed.capability_expires_at = pending.capability_expires_at;

    std::uint64_t offset = 0;
    for (const auto entry_index : pending.entry_indices) {
      const auto& entry = entries[entry_index];
      auto validate_status = validate_source_layout_batch_payload_entry(entry);
      if (!validate_status.ok()) {
        return validate_status;
      }

      auto* manifest_entry = packed.manifest.add_entries();
      manifest_entry->set_artifact_id(entry.artifact_id);
      manifest_entry->set_offset(offset);
      manifest_entry->set_length(entry.payload_size_bytes);
      manifest_entry->set_digest_alg(entry.digest_alg);
      manifest_entry->set_digest_hex(entry.digest_hex);

      v2::BatchPayloadSlice slice;
      slice.set_offset(offset);
      slice.set_length(entry.payload_size_bytes);
      packed.slices.push_back(std::move(slice));
      offset += entry.payload_size_bytes;
    }
    packed.manifest.set_total_size(offset);
    return packed;
  };

  PendingPack pending;
  for (std::size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
    const auto& entry = entries[entry_index];
    auto validate_status = validate_source_layout_batch_payload_entry(entry);
    if (!validate_status.ok()) {
      return validate_status;
    }
    const std::uint64_t entry_bytes = entry.payload_size_bytes;
    if (max_payload_bytes != 0 && entry_bytes > max_payload_bytes) {
      return absl::InvalidArgumentError("source-layout batch payload entry exceeds max_payload_bytes");
    }

    const bool reaches_item_limit = max_items != 0 && pending.entry_indices.size() >= max_items;
    const bool reaches_byte_limit =
        max_payload_bytes != 0 && pending.total_bytes != 0 && pending.total_bytes + entry_bytes > max_payload_bytes;
    if (!pending.entry_indices.empty() && (reaches_item_limit || reaches_byte_limit)) {
      auto packed_or = flush_pack(pending);
      if (!packed_or.ok()) {
        return packed_or.status();
      }
      packs.push_back(std::move(*packed_or));
      pending = PendingPack{};
    }

    pending.entry_indices.push_back(entry_index);
    pending.total_bytes += entry_bytes;
    pending.capability_expires_at = std::min(pending.capability_expires_at, entry.capability_expires_at);
  }

  if (!pending.entry_indices.empty()) {
    auto packed_or = flush_pack(pending);
    if (!packed_or.ok()) {
      return packed_or.status();
    }
    packs.push_back(std::move(*packed_or));
  }
  return packs;
}

absl::StatusOr<std::shared_ptr<const std::string>> realize_staged_batch_payload(
    const std::vector<BatchPayloadPackEntry>& entries,
    const PlannedBatchPayload& plan) {
  if (plan.manifest.total_size() == 0) {
    return absl::InvalidArgumentError("planned batch payload must be non-empty");
  }
  if (plan.manifest.total_size() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return absl::OutOfRangeError("planned batch payload exceeds host memory limits");
  }
  auto slab = std::make_shared<std::string>();
  slab->resize(static_cast<std::size_t>(plan.manifest.total_size()));
  std::uint64_t offset = 0;
  for (const auto entry_index : plan.source_indices) {
    const auto& entry = entries[entry_index];
    auto fill_status = fill_batch_payload_pack_entry(entry, slab->data() + static_cast<std::size_t>(offset));
    if (!fill_status.ok()) {
      return fill_status;
    }
    offset += entry.payload_size_bytes;
  }
  if (offset != plan.manifest.total_size()) {
    return absl::FailedPreconditionError("planned batch payload realization size mismatch");
  }
  return std::shared_ptr<const std::string>(std::move(slab));
}

absl::StatusOr<std::vector<PackedBatchPayload>> pack_batch_payload_entries(
    const std::vector<BatchPayloadPackEntry>& entries,
    std::uint64_t max_payload_bytes,
    std::uint32_t max_items) {
  auto plans_or = plan_batch_payload_entries(entries, max_payload_bytes, max_items);
  if (!plans_or.ok()) {
    return plans_or.status();
  }
  std::vector<PackedBatchPayload> packs;
  packs.reserve(plans_or->size());
  for (const auto& plan : *plans_or) {
    auto payload_or = realize_staged_batch_payload(entries, plan);
    if (!payload_or.ok()) {
      return payload_or.status();
    }
    packs.push_back(
        PackedBatchPayload{
            .manifest = plan.manifest,
            .payload = std::move(*payload_or),
            .source_indices = plan.source_indices,
            .slices = plan.slices,
            .capability_expires_at = plan.capability_expires_at,
        });
  }
  return packs;
}

absl::StatusOr<std::vector<PayloadTransportBroker::BatchCommunicatorSourceSegment>>
acquire_segmented_batch_payload_source_segments(
    const std::vector<BatchPayloadPackEntry>& entries,
    const PlannedBatchPayload& plan) {
  std::vector<PayloadTransportBroker::BatchCommunicatorSourceSegment> source_segments;
  source_segments.reserve(plan.source_indices.size());
  for (std::size_t plan_index = 0; plan_index < plan.source_indices.size(); ++plan_index) {
    const auto entry_index = plan.source_indices[plan_index];
    const auto& entry = entries[entry_index];
    auto validate_status = validate_batch_payload_pack_entry(entry);
    if (!validate_status.ok()) {
      return validate_status;
    }
    if (!entry.body_handle.has_value() || !entry.body_descriptor.has_value()) {
      return absl::FailedPreconditionError("segmented batch communicator source requires retained bodies");
    }
    auto export_view_or = entry.body_handle->acquire_export_view(
        BodyExportRequest{
            .preferred_location = common::memory::MemoryLocation::CPU,
            .require_remote_source = true,
            .allow_segmented_export = true,
        });
    if (!export_view_or.ok()) {
      return export_view_or.status();
    }
    if (!export_view_or->communicator_export.has_value()) {
      return absl::FailedPreconditionError("segmented batch communicator source requires communicator export");
    }
    std::uint64_t exported_bytes = 0;
    for (const auto buffer_size : export_view_or->communicator_export->buffer_sizes) {
      if (buffer_size > static_cast<size_t>(std::numeric_limits<std::uint64_t>::max() - exported_bytes)) {
        return absl::OutOfRangeError("segmented batch communicator source exceeds uint64 range");
      }
      exported_bytes += static_cast<std::uint64_t>(buffer_size);
    }
    if (exported_bytes != entry.payload_size_bytes || exported_bytes != plan.slices[plan_index].length()) {
      return absl::FailedPreconditionError("segmented batch communicator source size mismatch");
    }
    source_segments.push_back(
        PayloadTransportBroker::BatchCommunicatorSourceSegment{
            .export_view = std::move(*export_view_or),
        });
  }
  return source_segments;
}

absl::StatusOr<std::vector<PayloadTransportBroker::BatchCommunicatorRegionSourceSegment>>
acquire_segmented_region_source_segments(
    const std::vector<SourceLayoutBatchPayloadEntry>& entries,
    const PlannedBatchPayload& plan) {
  std::vector<PayloadTransportBroker::BatchCommunicatorRegionSourceSegment> source_segments;
  source_segments.reserve(plan.source_indices.size());
  for (std::size_t plan_index = 0; plan_index < plan.source_indices.size(); ++plan_index) {
    const auto entry_index = plan.source_indices[plan_index];
    const auto& entry = entries[entry_index];
    auto validate_status = validate_source_layout_batch_payload_entry(entry);
    if (!validate_status.ok()) {
      return validate_status;
    }
    if (entry.payload_size_bytes != plan.slices[plan_index].length()) {
      return absl::FailedPreconditionError("segmented region source size mismatch");
    }
    source_segments.push_back(
        PayloadTransportBroker::BatchCommunicatorRegionSourceSegment{
            .data = entry.source_span.data,
            .size_bytes = entry.source_span.length,
            .stable_backing = entry.source_span.stable_backing,
            .stable_backing_keepalive = entry.source_span.stable_backing_keepalive,
            .keepalive = entry.source_span.keepalive,
        });
  }
  return source_segments;
}

absl::StatusOr<LoaderSourceResolution> open_loader_from_resolved_source_capability(
    PayloadTransportBroker& payload_transport_broker,
    WorkerDirectoryCache& worker_directory_cache,
    const ResolvedSourceCapability& source_capability,
    absl::Time now,
    absl::Duration worker_directory_staleness_budget,
    std::string_view local_daemon_id,
    tensorcast::common::v1::PayloadRefDirection direction,
    std::string_view operation_id) {
  auto source_status = validate_resolved_source_capability(source_capability);
  if (!source_status.ok()) {
    return source_status;
  }
  if (source_capability.body_capability.has_value()) {
    auto loader_or = source_capability.body_capability->body_handle.make_loader();
    if (!loader_or.ok()) {
      return loader_or.status();
    }
    return LoaderSourceResolution{
        .loader = std::move(*loader_or),
        .source_kind = source_capability.source_kind,
        .payload_bytes = source_capability.body_capability->descriptor.size_bytes,
    };
  }
  if (source_capability.inline_payload) {
    return LoaderSourceResolution{
        .loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
            .data = std::shared_ptr<const void>(
                source_capability.inline_payload, static_cast<const void*>(source_capability.inline_payload->data())),
            .size_bytes = source_capability.inline_payload->size(),
        }),
        .source_kind = source_capability.source_kind,
        .payload_bytes = source_capability.inline_payload->size(),
    };
  }
  if (!source_capability.payload_ref.empty()) {
    auto loader_or = payload_transport_broker.open_payload_ref_loader(
        worker_directory_cache,
        now,
        worker_directory_staleness_budget,
        local_daemon_id,
        source_capability.payload_ref,
        source_capability.selection_identity.artifact_id,
        direction,
        operation_id);
    if (!loader_or.ok()) {
      return loader_or.status();
    }
    return LoaderSourceResolution{
        .loader = std::move(loader_or->loader),
        .source_kind = loader_or->remote ? store::loading::MaterializationSource::kP2P
                                         : store::loading::MaterializationSource::kLocalReplica,
        .payload_bytes = loader_or->metadata.payload_size,
    };
  }
  return absl::FailedPreconditionError("ResolvedSourceCapability does not provide a loader-backed source");
}

} // namespace

ByteArtifactController::ByteArtifactController(Dep d, Options options)
    : d_(std::move(d)),
      authority_service_(d_.body_store),
      body_backing_manager_(d_.engine),
      options_(std::move(options)),
      batch_get_apply_threads_(
          std::max<std::uint32_t>(
              1,
              options_.batch_get_apply_threads != 0
                  ? options_.batch_get_apply_threads
                  : static_cast<std::uint32_t>(std::max(1, d_.engine.options().num_thread)))) {
  batch_get_apply_pool_ = std::make_unique<folly::CPUThreadPoolExecutor>(
      batch_get_apply_threads_, std::make_shared<folly::NamedThreadFactory>("tensorcast-byte-apply"));
}

ByteArtifactController::PeerBatchTransportSupport ByteArtifactController::local_batch_transport_support() const {
  return PeerBatchTransportSupport{
      .protocol_version = d_.payload_transport_broker.batch_transport_protocol_version(),
      .grpc_chunk_ref_enabled = d_.payload_transport_broker.batch_transport_enabled(),
      .communicator_source_enabled = d_.payload_transport_broker.batch_transport_communicator_enabled(),
      .host_memory_export_enabled = d_.payload_transport_broker.batch_transport_communicator_enabled(),
      .segmented_communicator_export_enabled =
          d_.payload_transport_broker.batch_transport_segmented_communicator_export_enabled(),
  };
}

bool ByteArtifactController::is_peer_batch_transport_support_refresh_complete(
    const PeerBatchTransportSupportAwaitContext* ctx) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (ctx == nullptr || ctx->self == nullptr || ctx->daemon_id == nullptr) {
    return true;
  }
  const auto it = ctx->self->peer_batch_transport_support_cache_.find(*ctx->daemon_id);
  return it == ctx->self->peer_batch_transport_support_cache_.end() || !it->second.refresh_in_flight;
}

ByteArtifactController::PeerBatchTransportSupport ByteArtifactController::resolve_peer_batch_transport_support(
    std::string_view daemon_id,
    absl::Time now,
    bool* cache_hit) const {
  if (cache_hit != nullptr) {
    *cache_hit = false;
  }

  PeerBatchTransportSupport unsupported;
  if (daemon_id.empty() || !d_.payload_transport_broker.batch_transport_enabled()) {
    return unsupported;
  }

  const std::string daemon_id_key(daemon_id);
  for (;;) {
    bool should_refresh = false;
    {
      absl::MutexLock lock(&peer_batch_transport_support_cache_mu_);
      auto it = peer_batch_transport_support_cache_.find(daemon_id_key);
      if (it == peer_batch_transport_support_cache_.end()) {
        it = peer_batch_transport_support_cache_
                 .emplace(daemon_id_key, PeerBatchTransportSupportCacheEntry{.refresh_in_flight = true})
                 .first;
        should_refresh = true;
      } else if (it->second.refresh_in_flight) {
        const PeerBatchTransportSupportAwaitContext await_ctx{
            .self = this,
            .daemon_id = &daemon_id_key,
        };
        peer_batch_transport_support_cache_mu_.Await(
            absl::Condition(&ByteArtifactController::is_peer_batch_transport_support_refresh_complete, &await_ctx));
        now = absl::Now();
        continue;
      } else if (it->second.expires_at > now) {
        if (cache_hit != nullptr) {
          *cache_hit = true;
        }
        return it->second.support;
      } else {
        it->second.refresh_in_flight = true;
        should_refresh = true;
      }
    }

    if (!should_refresh) {
      continue;
    }

    PeerBatchTransportSupport support;
    auto address_or = d_.worker_directory_cache.resolve_daemon_address(
        daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    if (address_or.ok()) {
      auto channel = create_inter_daemon_channel(*address_or, d_.inter_daemon_channel_credentials);
      auto stub = v2::StoreDaemonService::NewStub(channel);
      grpc::ClientContext client_ctx;
      client_ctx.set_deadline(std::chrono::system_clock::now() + inter_daemon_home_rpc_timeout(options_));
      v2::GetServerConfigRequest config_req;
      v2::GetServerConfigResponse config_resp;
      const auto status = stub->GetServerConfig(&client_ctx, config_req, &config_resp);
      if (status.ok()) {
        support.protocol_version = config_resp.batch_transport_protocol_version();
        support.grpc_chunk_ref_enabled = config_resp.batch_payload_grpc_chunk_ref_enabled();
        support.communicator_source_enabled = config_resp.batch_payload_communicator_source_enabled();
        support.host_memory_export_enabled = config_resp.batch_payload_host_memory_export_enabled();
        support.segmented_communicator_export_enabled =
            config_resp.batch_payload_segmented_communicator_export_enabled();
      }
    }

    {
      absl::MutexLock lock(&peer_batch_transport_support_cache_mu_);
      auto& entry = peer_batch_transport_support_cache_[daemon_id_key];
      entry.support = support;
      entry.expires_at = absl::Now() + peer_batch_transport_support_cache_ttl();
      entry.refresh_in_flight = false;
    }
    return support;
  }
}

void ByteArtifactController::publish_preregistered_export(
    std::string_view artifact_id,
    const BodyHandle& body_handle,
    absl::Time now) {
  if (!options_.publish_prereg.enabled || options_.publish_prereg.ttl <= absl::ZeroDuration()) {
    return;
  }
  auto comm_manager = d_.engine.get_shared_comm_manager();
  if (!comm_manager->is_enabled() || !comm_manager->get_engine().is_rdma_enabled()) {
    return;
  }
  if (body_handle.empty()) {
    VLOG(2) << "byte_artifact.publish_prereg.skip"
            << " artifact_id=" << artifact_id << " published_at_ms=" << absl::ToUnixMillis(now)
            << " reason=empty_handle";
    return;
  }
  if (body_handle.location() != common::memory::MemoryLocation::CPU) {
    VLOG(2) << "byte_artifact.publish_prereg.skip"
            << " artifact_id=" << artifact_id << " published_at_ms=" << absl::ToUnixMillis(now)
            << " location=" << static_cast<int>(body_handle.location());
    return;
  }

  const absl::Time acquire_started_at = absl::Now();
  auto export_view_or = body_handle.acquire_export_view(
      BodyExportRequest{
          .preferred_location = common::memory::MemoryLocation::CPU,
          .require_remote_source = true,
          .allow_segmented_export = true,
      });
  const absl::Duration acquire_elapsed = absl::Now() - acquire_started_at;
  if (!export_view_or.ok()) {
    VLOG(2) << "byte_artifact.publish_prereg.acquire_failed"
            << " artifact_id=" << artifact_id << " acquire_ms=" << absl::ToDoubleMilliseconds(acquire_elapsed)
            << " error=" << export_view_or.status();
    return;
  }
  const absl::Status pin_status = body_handle.pin_export_keepalive(
      common::memory::MemoryLocation::CPU, export_view_or->keepalive, now + options_.publish_prereg.ttl);
  if (!pin_status.ok()) {
    VLOG(2) << "byte_artifact.publish_prereg.pin_failed"
            << " artifact_id=" << artifact_id << " acquire_ms=" << absl::ToDoubleMilliseconds(acquire_elapsed)
            << " error=" << pin_status;
    return;
  }
  const std::size_t export_key_count = export_view_or->communicator_export.has_value()
      ? export_view_or->communicator_export->remote_memory_keys.size()
      : 0;
  VLOG(2) << "byte_artifact.publish_prereg.result"
          << " artifact_id=" << artifact_id << " acquire_ms=" << absl::ToDoubleMilliseconds(acquire_elapsed)
          << " export_key_count=" << export_key_count
          << " expires_in_ms=" << absl::ToDoubleMilliseconds(options_.publish_prereg.ttl) << " pinned=true";
}

void ByteArtifactController::reconcile_policy_visibility(
    const std::vector<std::string>& artifact_ids,
    const ByteArtifactAuthorityService::Context& context) const {
  for (const auto& artifact_id : artifact_ids) {
    auto authority = d_.body_store.inspect_authority(
        artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now);
    if (!authority.has_value()) {
      continue;
    }
    if (authority->authority_record.claim_state == AuthorityClaimState::kClaimDeleted ||
        authority->authority_record.claim_state == AuthorityClaimState::kUnclaimed) {
      continue;
    }
    if (authority->authority_record.visibility_kind == AuthorityVisibilityKind::kReadyBacking &&
        authority->authority_record.claim_state == AuthorityClaimState::kClaimedVisible) {
      continue;
    }

    auto desired_ref = resolve_policy_visibility_ref(*authority);
    if (desired_ref.has_value()) {
      (void)d_.body_store.install_policy_visibility(
          artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now, *desired_ref);
      continue;
    }
    (void)d_.body_store.clear_policy_visibility(
        artifact_id,
        context.shard_id,
        context.lease_generation,
        context.routing_epoch,
        context.now,
        "policy_path_not_actionable");
  }
}

std::optional<PolicyVisibilityRef> ByteArtifactController::resolve_policy_visibility_ref(
    const ByteArtifactBodyStore::AuthoritySnapshot& authority_snapshot) const {
  if (d_.persistence_manager == nullptr) {
    return std::nullopt;
  }
  auto policy_source = d_.persistence_manager->resolve_policy_source(authority_snapshot.authority_record.artifact_id);
  if (!policy_source.has_value()) {
    return std::nullopt;
  }
  if (policy_source->path_kind != PersistenceManager::PolicySourceKind::kSharedDisk) {
    return std::nullopt;
  }
  if (!policy_source->verified_content_descriptor.has_value() ||
      *policy_source->verified_content_descriptor != authority_snapshot.verified_content_descriptor) {
    return std::nullopt;
  }
  return PolicyVisibilityRef{
      .path_id = policy_source->path_id,
      .path_kind = policy_visibility_path_kind_from_source(policy_source->path_kind),
      .verified_content_descriptor = authority_snapshot.verified_content_descriptor,
      .control_ref = policy_source->control_ref,
      .expires_at = authority_snapshot.expires_at,
  };
}

absl::StatusOr<ResolvedSourceCapability> ByteArtifactController::restore_backing_from_policy_visibility(
    ResolvedSourceCapability source_capability,
    const ByteArtifactAuthorityService::Context& context,
    std::string_view operation_id) const {
  const std::string_view artifact_id = source_capability.selection_identity.artifact_id;
  if (d_.persistence_manager == nullptr || !source_capability.policy_source_ref.has_value()) {
    return absl::FailedPreconditionError("policy-backed visibility requires persistence proof");
  }
  if (source_capability.policy_source_ref->path_kind != PolicyVisibilityPathKind::kSharedDisk) {
    return absl::FailedPreconditionError("unsupported policy-backed path kind");
  }
  const auto& content_identity = source_capability.verified_content_descriptor.content_identity;
  if (content_identity.semantic_layout_identity.kind != store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId ||
      content_identity.semantic_layout_identity.value.empty()) {
    return absl::FailedPreconditionError(
        "policy-backed visibility requires a named-layout verified content descriptor");
  }
  BodyDescriptor expected_descriptor;
  expected_descriptor.layout_id = content_identity.semantic_layout_identity.value;
  expected_descriptor.size_bytes = content_identity.logical_size_bytes;
  expected_descriptor.payload_digest_alg = normalize_body_digest_value(content_identity.digest_alg);
  expected_descriptor.payload_digest_hex = normalize_body_digest_value(
      store::runtime::ingestion::content_digest_bytes_to_hex(content_identity.digest_bytes));
  expected_descriptor.verification_mode =
      expected_descriptor.payload_digest_alg.empty() && expected_descriptor.payload_digest_hex.empty()
      ? v2::BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY
      : v2::BYTE_ARTIFACT_VERIFICATION_MODE_STRICT_SHA256;
  auto policy_source =
      d_.persistence_manager->resolve_policy_source(artifact_id, source_capability.policy_source_ref->control_ref);
  if (!policy_source.has_value()) {
    (void)d_.body_store.clear_policy_visibility(
        artifact_id,
        context.shard_id,
        context.lease_generation,
        context.routing_epoch,
        context.now,
        "policy_path_not_found");
    return absl::NotFoundError("policy-backed shared-disk path is no longer actionable");
  }
  if (policy_source->path_kind != PersistenceManager::PolicySourceKind::kSharedDisk) {
    return absl::FailedPreconditionError("policy-backed source kind mismatch");
  }

  auto staged_body_or = body_backing_manager_.stage_body(
      BodyBackingManager::StageRequest{
          .artifact_id = std::string(artifact_id),
          .invariant = body_descriptor_to_invariant(expected_descriptor),
          .loader = std::make_unique<store::DiskLoader>(store::loading::DiskSource{
              .path = policy_source->local_path,
              .expected_size = expected_descriptor.size_bytes,
              .require_descriptor = true,
          }),
          .source_kind = store::loading::MaterializationSource::kDisk,
          .operation_id = std::string(operation_id),
          .access_class = BodyAccessClass::kHomeDefault,
          .route_role = BodyRouteRole::kHomeAuthority,
      });
  if (!staged_body_or.ok()) {
    if (is_non_actionable_policy_path_error(staged_body_or.status())) {
      (void)d_.body_store.clear_policy_visibility(
          artifact_id,
          context.shard_id,
          context.lease_generation,
          context.routing_epoch,
          context.now,
          "policy_materialization_failed");
    }
    return staged_body_or.status();
  }

  auto put_result = d_.body_store.put_if_absent(
      artifact_id,
      body_descriptor_to_invariant(expected_descriptor),
      staged_body_or->descriptor,
      staged_body_or->verified_content_descriptor,
      staged_body_or->verification_record,
      staged_body_or->backing_identity,
      staged_body_or->observation,
      staged_body_or->body_handle,
      context.shard_id,
      context.lease_generation,
      context.routing_epoch,
      context.now,
      std::nullopt);
  if (put_result.outcome == ByteArtifactBodyStore::PutOutcome::kConflict) {
    (void)staged_body_or->body_handle.retire();
    return absl::FailedPreconditionError("policy-backed restore conflicted with current claim descriptor");
  }

  auto entry =
      d_.body_store.get(artifact_id, context.shard_id, context.lease_generation, context.routing_epoch, context.now);
  if (!entry.has_value()) {
    return absl::InternalError("policy-backed restore did not produce a visible backing");
  }
  source_capability.verified_content_descriptor = entry->verified_content_descriptor;
  source_capability.backing_identity = entry->backing_record.identity;
  source_capability.source_kind = store::loading::MaterializationSource::kLocalReplica;
  source_capability.body_capability = ResolvedBodyCapability{
      .mode = BodyCapabilityResolutionMode::kLocalBodyHandle,
      .local = true,
      .body_handle = entry->backing_record.retained_body_handle,
      .descriptor = entry->descriptor,
  };
  source_capability.inline_payload.reset();
  source_capability.payload_ref.clear();
  source_capability.policy_source_ref.reset();
  auto source_status = validate_resolved_source_capability(source_capability);
  if (!source_status.ok()) {
    return source_status;
  }
  return source_capability;
}

grpc::Status ByteArtifactController::home_batch_exists(
    RpcContext& rctx,
    const v2::HomeBatchExistsRequest& req,
    v2::HomeBatchExistsResponse& resp) {
  const absl::Time total_started_at = absl::Now();
  const absl::Time now = absl::Now();
  const absl::Time ensure_home_lease_started_at = absl::Now();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  const absl::Duration ensure_home_lease_elapsed = absl::Now() - ensure_home_lease_started_at;
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& artifact_id : req.artifact_ids()) {
      auto* outcome = resp.add_outcomes();
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::string> artifact_ids;
  artifact_ids.reserve(req.artifact_ids_size());
  for (const auto& artifact_id : req.artifact_ids()) {
    artifact_ids.push_back(artifact_id);
  }
  ByteArtifactAuthorityService::Context authority_context{
      .shard_id = req.fence().shard_id(),
      .lease_generation = home_lease_or->lease_generation,
      .routing_epoch = options_.routing.routing_epoch,
      .shard_count = options_.routing.shard_count,
      .now = now,
  };
  const absl::Time reconcile_started_at = absl::Now();
  reconcile_policy_visibility(artifact_ids, authority_context);
  const absl::Duration reconcile_elapsed = absl::Now() - reconcile_started_at;
  const absl::Time authority_started_at = absl::Now();
  auto outcomes = authority_service_.batch_exists(artifact_ids, authority_context);
  const absl::Duration authority_elapsed = absl::Now() - authority_started_at;
  for (auto outcome : outcomes) {
    *resp.add_outcomes() = std::move(outcome);
  }
  VLOG(2) << "byte_artifact.home_batch_exists_timing_summary"
          << " shard_id=" << req.fence().shard_id() << " requested_artifacts=" << req.artifact_ids_size()
          << " first_artifact_id=" << (req.artifact_ids_size() == 0 ? "" : req.artifact_ids(0))
          << " ensure_home_lease_ms=" << absl::ToDoubleMilliseconds(ensure_home_lease_elapsed)
          << " reconcile_policy_ms=" << absl::ToDoubleMilliseconds(reconcile_elapsed)
          << " authority_batch_exists_ms=" << absl::ToDoubleMilliseconds(authority_elapsed)
          << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::home_batch_get(
    RpcContext& rctx,
    const v2::HomeBatchGetRequest& req,
    v2::HomeBatchGetResponse& resp) {
  const absl::Time total_started_at = absl::Now();
  const absl::Time now = absl::Now();
  const std::string& local_daemon_id = d_.route_resolver.local_daemon_id();

  struct HomeBatchGetTimingStats {
    absl::Duration ensure_home_lease_elapsed{absl::ZeroDuration()};
    absl::Duration reconcile_policy_elapsed{absl::ZeroDuration()};
    absl::Duration requester_config_elapsed{absl::ZeroDuration()};
    bool requester_config_cache_hit{false};
    absl::Duration authority_batch_get_elapsed{absl::ZeroDuration()};
    absl::Duration candidate_build_elapsed{absl::ZeroDuration()};
    absl::Duration requester_entry_lookup_elapsed{absl::ZeroDuration()};
    absl::Duration producer_entry_lookup_elapsed{absl::ZeroDuration()};
    absl::Duration policy_restore_elapsed{absl::ZeroDuration()};
    absl::Duration payload_read_elapsed{absl::ZeroDuration()};
    absl::Duration payload_ref_issue_elapsed{absl::ZeroDuration()};
    absl::Duration pack_build_elapsed{absl::ZeroDuration()};
    absl::Duration staged_pack_realization_elapsed{absl::ZeroDuration()};
    absl::Duration communicator_export_elapsed{absl::ZeroDuration()};
    absl::Duration batch_payload_ref_issue_elapsed{absl::ZeroDuration()};
    absl::Duration transport_response_emit_elapsed{absl::ZeroDuration()};
    absl::Duration response_shape_elapsed{absl::ZeroDuration()};
    std::size_t payload_read_count{0};
    std::uint64_t payload_read_bytes{0};
    std::size_t batch_candidate_count{0};
    std::uint64_t batch_candidate_bytes{0};
    std::size_t pack_count{0};
    std::uint64_t pack_bytes{0};
    std::size_t segmented_pack_count{0};
    std::uint64_t segmented_pack_bytes{0};
    std::size_t staged_pack_count{0};
    std::uint64_t staged_pack_bytes{0};
  } timing_stats;

  const absl::Time home_batch_get_ensure_home_lease_started_at = absl::Now();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  timing_stats.ensure_home_lease_elapsed += absl::Now() - home_batch_get_ensure_home_lease_started_at;
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& artifact_id : req.artifact_ids()) {
      auto* item = resp.add_items();
      *item = make_home_get_item(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::string> artifact_ids;
  artifact_ids.reserve(req.artifact_ids_size());
  for (const auto& artifact_id : req.artifact_ids()) {
    artifact_ids.push_back(artifact_id);
  }
  ByteArtifactAuthorityService::Context authority_context{
      .shard_id = req.fence().shard_id(),
      .lease_generation = home_lease_or->lease_generation,
      .routing_epoch = options_.routing.routing_epoch,
      .shard_count = options_.routing.shard_count,
      .now = now,
  };
  const absl::Time home_batch_get_reconcile_policy_started_at = absl::Now();
  reconcile_policy_visibility(artifact_ids, authority_context);
  timing_stats.reconcile_policy_elapsed += absl::Now() - home_batch_get_reconcile_policy_started_at;
  const std::string_view operation_id =
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view("");
  const bool batch_transport_enabled = d_.payload_transport_broker.batch_transport_enabled();
  const std::uint64_t max_batch_payload_bytes = d_.payload_transport_broker.max_batch_payload_bytes();
  PeerBatchTransportSupport requester_batch_transport_support = local_batch_transport_support();
  if (req.has_requester_daemon_id() && !req.requester_daemon_id().empty() &&
      req.requester_daemon_id() != local_daemon_id) {
    const absl::Time requester_config_started_at = absl::Now();
    requester_batch_transport_support =
        resolve_peer_batch_transport_support(req.requester_daemon_id(), now, &timing_stats.requester_config_cache_hit);
    timing_stats.requester_config_elapsed += absl::Now() - requester_config_started_at;
  }

  struct BatchGetTransportCandidate {
    int item_index{0};
    BatchPayloadPackEntry pack_entry;
  };

  std::vector<BatchGetTransportCandidate> batch_candidates;
  batch_candidates.reserve(req.artifact_ids_size());

  const absl::Time authority_batch_get_started_at = absl::Now();
  auto authority_results = authority_service_.batch_get(artifact_ids, authority_context);
  timing_stats.authority_batch_get_elapsed += absl::Now() - authority_batch_get_started_at;
  const absl::Time candidate_build_started_at = absl::Now();
  for (auto& result : authority_results) {
    auto* item = resp.add_items();
    const int item_index = resp.items_size() - 1;
    item->set_artifact_id(result.artifact_id);
    item->set_status(result.status);
    if (!result.message.empty()) {
      item->set_message(result.message);
    }
    if (result.status != v2::BATCH_ITEM_STATUS_OK) {
      continue;
    }
    if (!result.source_capability.has_value()) {
      *item = make_home_get_item(result.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing source capability");
      continue;
    }

    ResolvedSourceCapability source_capability = *result.source_capability;
    if (source_capability.policy_source_ref.has_value()) {
      const absl::Time policy_restore_started_at = absl::Now();
      auto restored_or =
          restore_backing_from_policy_visibility(std::move(source_capability), authority_context, operation_id);
      timing_stats.policy_restore_elapsed += absl::Now() - policy_restore_started_at;
      if (!restored_or.ok()) {
        if (is_non_actionable_policy_path_error(restored_or.status())) {
          *item = make_home_get_item(result.artifact_id, v2::BATCH_ITEM_STATUS_MISS, "artifact is no longer visible");
        } else {
          *item = make_home_get_item(
              result.artifact_id,
              batch_item_status_from_absl_status(restored_or.status()),
              restored_or.status().message());
        }
        continue;
      }
      source_capability = std::move(*restored_or);
    }

    if (source_capability.body_capability.has_value()) {
      const auto& body_source = *source_capability.body_capability;
      if (body_source.descriptor.size_bytes > options_.routing.inline_payload_threshold_bytes) {
        if (batch_transport_enabled &&
            (max_batch_payload_bytes == 0 || body_source.descriptor.size_bytes <= max_batch_payload_bytes)) {
          ++timing_stats.batch_candidate_count;
          timing_stats.batch_candidate_bytes += body_source.descriptor.size_bytes;
          batch_candidates.push_back(
              BatchGetTransportCandidate{
                  .item_index = item_index,
                  .pack_entry =
                      BatchPayloadPackEntry{
                          .artifact_id = result.artifact_id,
                          .payload_size_bytes = body_source.descriptor.size_bytes,
                          .body_handle = body_source.body_handle,
                          .body_descriptor = body_source.descriptor,
                          .backing_identity = source_capability.backing_identity,
                          .backing_instance_generation =
                              source_capability.serving_capability.backing_instance_generation == 0
                              ? body_source.body_handle.binding_generation()
                              : source_capability.serving_capability.backing_instance_generation,
                          .digest_alg = body_source.descriptor.payload_digest_alg,
                          .digest_hex = body_source.descriptor.payload_digest_hex,
                          .capability_expires_at = source_capability.serving_capability.expires_at,
                      },
              });
          continue;
        }
        const absl::Time payload_ref_issue_started_at = absl::Now();
        auto payload_ref_or = d_.payload_transport_broker.issue_payload_ref(
            result.artifact_id,
            body_source.body_handle,
            body_source.descriptor,
            source_capability.backing_identity,
            source_capability.serving_capability.backing_instance_generation == 0
                ? body_source.body_handle.binding_generation()
                : source_capability.serving_capability.backing_instance_generation,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
            operation_id,
            source_capability.serving_capability.expires_at);
        timing_stats.payload_ref_issue_elapsed += absl::Now() - payload_ref_issue_started_at;
        if (!payload_ref_or.ok()) {
          *item = make_home_get_item(
              result.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, payload_ref_or.status().message());
          continue;
        }
        item->set_payload_ref(*payload_ref_or);
        continue;
      }
      const absl::Time payload_read_started_at = absl::Now();
      auto payload_or = body_source.body_handle.read_all_bytes();
      timing_stats.payload_read_elapsed += absl::Now() - payload_read_started_at;
      if (!payload_or.ok()) {
        d_.body_store.invalidate_artifact_visibility(result.artifact_id, now, "serve_read_failed");
        *item = make_home_get_item(result.artifact_id, v2::BATCH_ITEM_STATUS_MISS, "artifact is no longer visible");
        continue;
      }
      ++timing_stats.payload_read_count;
      timing_stats.payload_read_bytes += payload_or->size();
      item->set_inline_payload(*payload_or);
      continue;
    }

    if (source_capability.inline_payload) {
      if (source_capability.inline_payload->size() > options_.routing.inline_payload_threshold_bytes) {
        if (batch_transport_enabled &&
            (max_batch_payload_bytes == 0 || source_capability.inline_payload->size() <= max_batch_payload_bytes)) {
          ++timing_stats.batch_candidate_count;
          timing_stats.batch_candidate_bytes += source_capability.inline_payload->size();
          batch_candidates.push_back(
              BatchGetTransportCandidate{
                  .item_index = item_index,
                  .pack_entry =
                      BatchPayloadPackEntry{
                          .artifact_id = result.artifact_id,
                          .payload_size_bytes = source_capability.inline_payload->size(),
                          .inline_payload = source_capability.inline_payload,
                          .digest_alg = source_capability.verified_content_descriptor.content_identity.digest_alg,
                          .digest_hex = store::runtime::ingestion::content_digest_bytes_to_hex(
                              source_capability.verified_content_descriptor.content_identity.digest_bytes),
                          .capability_expires_at = source_capability.serving_capability.expires_at,
                      },
              });
          continue;
        }
        const absl::Time payload_ref_issue_started_at = absl::Now();
        auto payload_ref_or = d_.payload_transport_broker.issue_payload_ref(
            result.artifact_id,
            source_capability.inline_payload,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
            operation_id,
            source_capability.serving_capability.expires_at);
        timing_stats.payload_ref_issue_elapsed += absl::Now() - payload_ref_issue_started_at;
        if (!payload_ref_or.ok()) {
          *item = make_home_get_item(
              result.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, payload_ref_or.status().message());
          continue;
        }
        item->set_payload_ref(*payload_ref_or);
        continue;
      }
      item->set_inline_payload(*source_capability.inline_payload);
      continue;
    }

    if (!source_capability.payload_ref.empty()) {
      item->set_payload_ref(source_capability.payload_ref);
      continue;
    }

    *item = make_home_get_item(
        result.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "resolved source capability is not readable");
  }
  timing_stats.candidate_build_elapsed += absl::Now() - candidate_build_started_at;

  if (!batch_candidates.empty()) {
    std::vector<BatchPayloadPackEntry> pack_entries;
    pack_entries.reserve(batch_candidates.size());
    for (const auto& candidate : batch_candidates) {
      pack_entries.push_back(candidate.pack_entry);
    }
    const absl::Time pack_build_started_at = absl::Now();
    auto packs_or = plan_batch_payload_entries(
        pack_entries,
        d_.payload_transport_broker.max_batch_payload_bytes(),
        d_.payload_transport_broker.max_batch_items());
    timing_stats.pack_build_elapsed += absl::Now() - pack_build_started_at;
    if (!packs_or.ok()) {
      LOG(WARNING) << "home_batch_get batch transport fallback to payload_ref: " << packs_or.status();
    } else {
      std::size_t transport_count = 0;
      std::size_t transport_items = 0;
      std::uint64_t transport_bytes = 0;
      std::size_t communicator_transport_count = 0;
      for (auto& pack : *packs_or) {
        ++timing_stats.pack_count;
        timing_stats.pack_bytes += pack.manifest.total_size();
        const bool use_communicator_transport = requester_batch_transport_support.supports_v2() &&
            d_.payload_transport_broker.batch_transport_communicator_enabled();
        std::optional<WorkerDirectoryCache::Entry> producer_entry;
        if (use_communicator_transport) {
          const absl::Time producer_entry_lookup_started_at = absl::Now();
          auto producer_entry_or = d_.worker_directory_cache.resolve_daemon_entry(
              local_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
          timing_stats.producer_entry_lookup_elapsed += absl::Now() - producer_entry_lookup_started_at;
          if (producer_entry_or.ok() && producer_entry_or->p2p_port != 0) {
            producer_entry = *producer_entry_or;
          }
        }
        const bool emit_communicator_source = producer_entry.has_value();
        const bool use_segmented_communicator_transport = emit_communicator_source &&
            requester_batch_transport_support.supports_segmented_communicator_export() &&
            d_.payload_transport_broker.batch_transport_segmented_communicator_export_enabled();
        absl::Status transport_issue_status;
        std::optional<std::string> batch_payload_ref;
        std::optional<store::ExportRegistration> communicator_export;
        std::string pack_realization_mode;
        std::shared_ptr<const std::string> staged_payload;
        const auto ensure_staged_payload = [&]() -> absl::StatusOr<std::shared_ptr<const std::string>> {
          if (staged_payload) {
            return staged_payload;
          }
          const absl::Time staged_pack_started_at = absl::Now();
          auto staged_payload_or = realize_staged_batch_payload(pack_entries, pack);
          timing_stats.staged_pack_realization_elapsed += absl::Now() - staged_pack_started_at;
          if (!staged_payload_or.ok()) {
            return staged_payload_or.status();
          }
          staged_payload = *staged_payload_or;
          ++timing_stats.staged_pack_count;
          timing_stats.staged_pack_bytes += pack.manifest.total_size();
          return staged_payload;
        };
        if (use_segmented_communicator_transport) {
          const absl::Time communicator_export_started_at = absl::Now();
          auto source_segments_or = acquire_segmented_batch_payload_source_segments(pack_entries, pack);
          if (source_segments_or.ok()) {
            auto communicator_export_or = d_.payload_transport_broker.issue_batch_payload_communicator_export(
                pack.manifest,
                absl::MakeSpan(*source_segments_or),
                tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
                operation_id,
                pack.capability_expires_at,
                req.has_requester_daemon_id() ? std::string_view(req.requester_daemon_id()) : std::string_view(""));
            timing_stats.communicator_export_elapsed += absl::Now() - communicator_export_started_at;
            if (communicator_export_or.ok()) {
              batch_payload_ref = communicator_export_or->batch_payload_ref;
              communicator_export = communicator_export_or->export_registration;
              pack_realization_mode = "segmented_communicator_export";
              ++timing_stats.segmented_pack_count;
              timing_stats.segmented_pack_bytes += pack.manifest.total_size();
            } else {
              transport_issue_status = communicator_export_or.status();
            }
          } else {
            timing_stats.communicator_export_elapsed += absl::Now() - communicator_export_started_at;
            transport_issue_status = source_segments_or.status();
          }
        }
        if (emit_communicator_source && !batch_payload_ref.has_value()) {
          auto staged_payload_or = ensure_staged_payload();
          if (!staged_payload_or.ok()) {
            transport_issue_status = staged_payload_or.status();
          } else {
            const absl::Time communicator_export_started_at = absl::Now();
            auto communicator_export_or = d_.payload_transport_broker.issue_batch_payload_communicator_export(
                pack.manifest,
                *staged_payload_or,
                tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
                operation_id,
                pack.capability_expires_at,
                req.has_requester_daemon_id() ? std::string_view(req.requester_daemon_id()) : std::string_view(""));
            timing_stats.communicator_export_elapsed += absl::Now() - communicator_export_started_at;
            if (communicator_export_or.ok()) {
              batch_payload_ref = communicator_export_or->batch_payload_ref;
              communicator_export = communicator_export_or->export_registration;
              pack_realization_mode = "staged_slab";
            } else {
              transport_issue_status = communicator_export_or.status();
            }
          }
        }
        if (!batch_payload_ref.has_value()) {
          auto staged_payload_or = ensure_staged_payload();
          if (!staged_payload_or.ok()) {
            transport_issue_status = staged_payload_or.status();
          } else {
            const absl::Time batch_payload_ref_issue_started_at = absl::Now();
            auto batch_payload_ref_or = d_.payload_transport_broker.issue_batch_payload_ref(
                pack.manifest,
                *staged_payload_or,
                tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
                operation_id,
                pack.capability_expires_at,
                req.has_requester_daemon_id() ? std::string_view(req.requester_daemon_id()) : std::string_view(""));
            timing_stats.batch_payload_ref_issue_elapsed += absl::Now() - batch_payload_ref_issue_started_at;
            if (batch_payload_ref_or.ok()) {
              batch_payload_ref = *batch_payload_ref_or;
              if (pack_realization_mode.empty()) {
                pack_realization_mode = "staged_slab";
              }
            } else {
              transport_issue_status = batch_payload_ref_or.status();
            }
          }
        }
        if (!batch_payload_ref.has_value()) {
          LOG(WARNING) << "home_batch_get batch transport pack fallback to payload_ref: " << transport_issue_status;
          for (const auto source_index : pack.source_indices) {
            const auto& candidate = batch_candidates[source_index];
            const absl::Time payload_ref_issue_started_at = absl::Now();
            auto payload_ref_or = issue_payload_ref_for_batch_payload_pack_entry(
                d_.payload_transport_broker,
                candidate.pack_entry,
                tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
                operation_id);
            timing_stats.payload_ref_issue_elapsed += absl::Now() - payload_ref_issue_started_at;
            if (!payload_ref_or.ok()) {
              *resp.mutable_items(candidate.item_index) = make_home_get_item(
                  candidate.pack_entry.artifact_id,
                  v2::BATCH_ITEM_STATUS_UNAVAILABLE,
                  payload_ref_or.status().message());
              continue;
            }
            resp.mutable_items(candidate.item_index)->set_payload_ref(*payload_ref_or);
          }
          continue;
        }
        LOG(INFO) << "byte_artifact.home_batch_get_pack_realization"
                  << " operation_id=" << operation_id
                  << " requester_daemon_id=" << (req.has_requester_daemon_id() ? req.requester_daemon_id() : "")
                  << " mode=" << pack_realization_mode << " item_count=" << pack.source_indices.size()
                  << " payload_bytes=" << pack.manifest.total_size()
                  << " communicator_transport=" << (emit_communicator_source && communicator_export.has_value());

        auto* transport = resp.add_batch_transports();
        const std::string transport_id = absl::StrCat("batch-transport-", resp.batch_transports_size());
        const absl::Time transport_response_emit_started_at = absl::Now();
        transport->set_transport_id(transport_id);
        transport->mutable_manifest()->CopyFrom(pack.manifest);
        if (emit_communicator_source && communicator_export.has_value()) {
          auto* communicator_source = transport->mutable_communicator_source();
          communicator_source->set_batch_payload_ref(*batch_payload_ref);
          communicator_source->set_protocol_version(d_.payload_transport_broker.batch_transport_protocol_version());
          communicator_source->set_producer_daemon_id(local_daemon_id);
          if (req.has_requester_daemon_id()) {
            communicator_source->set_consumer_daemon_id(req.requester_daemon_id());
          }
          communicator_source->set_producer_host(producer_entry->node_address);
          communicator_source->set_producer_port(producer_entry->p2p_port);
          for (const auto& remote_memory_key : communicator_export->remote_memory_keys) {
            communicator_source->add_remote_memory_keys(remote_memory_key);
          }
          for (const auto buffer_size : communicator_export->buffer_sizes) {
            communicator_source->add_buffer_sizes(buffer_size);
          }
          if (!producer_entry->node_id.empty()) {
            communicator_source->set_remote_endpoint_id(
                store::components::derive_endpoint_id(
                    producer_entry->node_id, common::memory::MemoryLocation::CPU, /*device_id=*/0));
          }
          communicator_source->set_memory_location(v2::BATCH_PAYLOAD_MEMORY_LOCATION_HOST);
          communicator_source->set_total_payload_bytes(pack.manifest.total_size());
          ++communicator_transport_count;
        } else {
          auto* grpc_chunk_ref = transport->mutable_grpc_chunk_ref();
          grpc_chunk_ref->set_batch_payload_ref(*batch_payload_ref);
          grpc_chunk_ref->set_protocol_version(
              std::min<std::uint32_t>(1, d_.payload_transport_broker.batch_transport_protocol_version()));
        }
        ++transport_count;
        transport_items += pack.source_indices.size();
        transport_bytes += pack.manifest.total_size();

        for (std::size_t pack_index = 0; pack_index < pack.source_indices.size(); ++pack_index) {
          const auto source_index = pack.source_indices[pack_index];
          const auto& candidate = batch_candidates[source_index];
          auto slice = pack.slices[pack_index];
          slice.set_transport_id(transport_id);
          resp.mutable_items(candidate.item_index)->mutable_batch_payload_slice()->CopyFrom(slice);
        }
        timing_stats.transport_response_emit_elapsed += absl::Now() - transport_response_emit_started_at;
      }
      if (transport_count != 0) {
        VLOG(2) << "byte_artifact.home_batch_get_batch_transport_summary"
                << " operation_id=" << operation_id << " transport_count=" << transport_count
                << " communicator_transport_count=" << communicator_transport_count
                << " segmented_pack_count=" << timing_stats.segmented_pack_count
                << " staged_pack_count=" << timing_stats.staged_pack_count << " item_count=" << transport_items
                << " payload_bytes=" << transport_bytes;
      }
    }
  }
  const absl::Time response_shape_started_at = absl::Now();
  const auto response_shape = inspect_home_batch_get_response_shape(resp);
  timing_stats.response_shape_elapsed += absl::Now() - response_shape_started_at;
  LOG(INFO) << "byte_artifact.home_batch_get_response_shape"
            << " operation_id=" << operation_id
            << " requester_daemon_id=" << (req.has_requester_daemon_id() ? req.requester_daemon_id() : "")
            << " requested_artifacts=" << req.artifact_ids_size() << " batch_candidates=" << batch_candidates.size()
            << " requester_support_v1=" << requester_batch_transport_support.supports_v1()
            << " requester_support_v2=" << requester_batch_transport_support.supports_v2()
            << " items=" << response_shape.items << " ok_items=" << response_shape.ok_items
            << " inline_items=" << response_shape.inline_items
            << " payload_ref_items=" << response_shape.payload_ref_items
            << " batch_slice_items=" << response_shape.batch_slice_items << " transports=" << response_shape.transports
            << " communicator_transports=" << response_shape.communicator_transports
            << " grpc_chunk_transports=" << response_shape.grpc_chunk_transports;
  {
    std::ostringstream log;
    log << "byte_artifact.home_batch_get_timing_summary"
        << " operation_id=" << operation_id << " shard_id=" << req.fence().shard_id()
        << " requested_artifacts=" << req.artifact_ids_size()
        << " batch_candidates=" << timing_stats.batch_candidate_count
        << " batch_candidate_bytes=" << timing_stats.batch_candidate_bytes
        << " payload_read_count=" << timing_stats.payload_read_count
        << " payload_read_bytes=" << timing_stats.payload_read_bytes << " pack_count=" << timing_stats.pack_count
        << " pack_bytes=" << timing_stats.pack_bytes
        << " ensure_home_lease_ms=" << absl::ToDoubleMilliseconds(timing_stats.ensure_home_lease_elapsed)
        << " reconcile_policy_ms=" << absl::ToDoubleMilliseconds(timing_stats.reconcile_policy_elapsed)
        << " requester_config_ms=" << absl::ToDoubleMilliseconds(timing_stats.requester_config_elapsed)
        << " requester_config_cache_hit=" << timing_stats.requester_config_cache_hit
        << " authority_batch_get_ms=" << absl::ToDoubleMilliseconds(timing_stats.authority_batch_get_elapsed)
        << " candidate_build_ms=" << absl::ToDoubleMilliseconds(timing_stats.candidate_build_elapsed)
        << " requester_entry_lookup_ms=" << absl::ToDoubleMilliseconds(timing_stats.requester_entry_lookup_elapsed)
        << " producer_entry_lookup_ms=" << absl::ToDoubleMilliseconds(timing_stats.producer_entry_lookup_elapsed)
        << " policy_restore_ms=" << absl::ToDoubleMilliseconds(timing_stats.policy_restore_elapsed)
        << " payload_read_ms=" << absl::ToDoubleMilliseconds(timing_stats.payload_read_elapsed)
        << " payload_ref_issue_ms=" << absl::ToDoubleMilliseconds(timing_stats.payload_ref_issue_elapsed)
        << " pack_build_ms=" << absl::ToDoubleMilliseconds(timing_stats.pack_build_elapsed)
        << " staged_pack_realization_ms=" << absl::ToDoubleMilliseconds(timing_stats.staged_pack_realization_elapsed)
        << " communicator_export_ms=" << absl::ToDoubleMilliseconds(timing_stats.communicator_export_elapsed)
        << " batch_payload_ref_issue_ms=" << absl::ToDoubleMilliseconds(timing_stats.batch_payload_ref_issue_elapsed)
        << " transport_response_emit_ms=" << absl::ToDoubleMilliseconds(timing_stats.transport_response_emit_elapsed)
        << " response_shape_ms=" << absl::ToDoubleMilliseconds(timing_stats.response_shape_elapsed)
        << " segmented_pack_count=" << timing_stats.segmented_pack_count
        << " segmented_pack_bytes=" << timing_stats.segmented_pack_bytes
        << " staged_pack_count=" << timing_stats.staged_pack_count
        << " staged_pack_bytes=" << timing_stats.staged_pack_bytes
        << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at);
    if (!operation_id.empty()) {
      LOG(INFO) << log.str();
    } else {
      VLOG(2) << log.str();
    }
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::home_batch_put_if_absent(
    RpcContext& rctx,
    const v2::HomeBatchPutIfAbsentRequest& req,
    v2::HomeBatchPutIfAbsentResponse& resp) {
  struct HomeBatchPutTimingStats {
    std::size_t batch_payload_slice_items{0};
    std::size_t inline_payload_items{0};
    std::size_t payload_ref_items{0};
    std::size_t communicator_open_count{0};
    std::size_t grpc_fetch_count{0};
    std::size_t remote_mirror_count{0};
    std::uint64_t remote_mirror_bytes{0};
    std::size_t remote_communicator_source_count{0};
    std::size_t remote_communicator_source_direct_write_count{0};
    std::size_t remote_communicator_source_batched_direct_write_count{0};
    std::size_t remote_direct_slice_transport_count{0};
    std::size_t remote_direct_slice_items{0};
    std::uint64_t remote_direct_slice_bytes{0};
    std::size_t remote_composite_stage_transport_count{0};
    std::size_t remote_composite_stage_items{0};
    std::uint64_t remote_composite_stage_bytes{0};
    std::size_t remote_composite_materialize_calls{0};
    std::size_t remote_composite_batched_direct_write_count{0};
    std::size_t remote_composite_fallback_count{0};
    std::size_t remote_composite_fallback_items{0};
    std::size_t remote_full_pack_mirror_items{0};
    std::size_t stage_body_count{0};
    std::size_t fast_cpu_stage_count{0};
    std::size_t stage_loader_count{0};
    std::size_t stage_local_source_count{0};
    std::size_t stage_p2p_count{0};
    std::size_t stage_local_replica_count{0};
    std::size_t reuse_attempt_count{0};
    std::size_t reuse_hit_count{0};
    std::size_t authority_item_count{0};
    absl::Duration communicator_open_elapsed{absl::ZeroDuration()};
    absl::Duration grpc_fetch_elapsed{absl::ZeroDuration()};
    absl::Duration remote_mirror_elapsed{absl::ZeroDuration()};
    absl::Duration remote_composite_stage_elapsed{absl::ZeroDuration()};
    absl::Duration stage_body_elapsed{absl::ZeroDuration()};
    absl::Duration fast_cpu_stage_elapsed{absl::ZeroDuration()};
    absl::Duration reuse_elapsed{absl::ZeroDuration()};
    absl::Duration authority_apply_elapsed{absl::ZeroDuration()};
  };

  const absl::Time total_started_at = absl::Now();
  HomeBatchPutTimingStats timing_stats;
  const absl::Time now = absl::Now();
  const std::optional<std::uint64_t> ttl_ms =
      req.has_ttl_ms() ? std::optional<std::uint64_t>(req.ttl_ms()) : std::nullopt;
  const std::string& local_daemon_id = d_.route_resolver.local_daemon_id();
  const std::string_view operation_id =
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view("");
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& item : req.items()) {
      auto* outcome = resp.add_outcomes();
      *outcome = make_outcome(item.artifact_id(), v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  absl::flat_hash_map<std::string, const v2::BatchPayloadTransport*> batch_transports_by_id;
  batch_transports_by_id.reserve(req.batch_transports_size());
  for (const auto& transport : req.batch_transports()) {
    if (!transport.transport_id().empty()) {
      batch_transports_by_id.emplace(transport.transport_id(), &transport);
    }
  }
  absl::flat_hash_map<std::string, PayloadTransportBroker::ResolvedBatchPayload> resolved_batch_payloads;
  absl::flat_hash_map<std::string, PayloadTransportBroker::BatchPayloadSource> resolved_batch_sources;
  absl::flat_hash_map<std::string, std::shared_ptr<const std::string>> mirrored_remote_batch_payloads;
  absl::flat_hash_map<std::string, std::shared_ptr<std::mutex>> remote_direct_source_mutexes;
  absl::flat_hash_set<std::string> direct_remote_batch_payloads;

  struct CompositeStageCandidate {
    std::string transport_id;
    std::uint64_t source_offset{0};
    std::uint64_t length{0};
  };

  struct PreparedHomeBatchPutItem {
    const v2::HomeBatchPutIfAbsentItem* item{nullptr};
    std::string artifact_id;
    std::unique_ptr<store::IArtifactLoader> loader;
    std::optional<BodyBackingManager::LocalByteSpan> local_source;
    std::optional<CompositeStageCandidate> composite_candidate;
    store::loading::MaterializationSource source_kind = store::loading::MaterializationSource::kLocalReplica;
    std::optional<BodyBackingManager::StageResult> staged_body;
  };

  struct StageWorkItem {
    int index{0};
    std::string artifact_id;
    std::unique_ptr<store::IArtifactLoader> loader;
    std::optional<BodyBackingManager::LocalByteSpan> local_source;
    store::loading::MaterializationSource source_kind = store::loading::MaterializationSource::kLocalReplica;
  };

  struct StageWorkItemResult {
    int index{0};
    std::string artifact_id;
    absl::Status status;
    std::optional<BodyBackingManager::StageResult> staged_body;
    absl::Duration elapsed{absl::ZeroDuration()};
    bool used_fast_cpu_stage{false};
  };

  struct ScheduledStageWorkItem {
    int index{0};
    std::string artifact_id;
    folly::SemiFuture<StageWorkItemResult> future;
  };

  std::vector<std::optional<v2::BatchItemOutcome>> deferred_outcomes(req.items_size());
  std::vector<PreparedHomeBatchPutItem> prepared_items(static_cast<std::size_t>(req.items_size()));
  std::vector<ByteArtifactAuthorityService::PutItem> authority_items;
  std::vector<int> authority_item_indices;
  authority_items.reserve(req.items_size());
  authority_item_indices.reserve(req.items_size());

  for (int index = 0; index < req.items_size(); ++index) {
    const auto& item = req.items(index);
    const std::string artifact_id = item.artifact_id();
    const bool has_batch_payload_slice =
        item.has_batch_payload_slice() && !item.batch_payload_slice().transport_id().empty();
    if (has_batch_payload_slice) {
      ++timing_stats.batch_payload_slice_items;
    } else if (!item.inline_payload().empty()) {
      ++timing_stats.inline_payload_items;
    } else if (!item.payload_ref().empty()) {
      ++timing_stats.payload_ref_items;
    }
    const int source_count = (!item.inline_payload().empty() ? 1 : 0) + (!item.payload_ref().empty() ? 1 : 0) +
        (has_batch_payload_slice ? 1 : 0);
    if (source_count > 1) {
      deferred_outcomes[index] = make_outcome(
          artifact_id,
          v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT,
          "inline_payload, payload_ref, and batch_payload_slice are mutually exclusive");
      continue;
    }

    std::unique_ptr<store::IArtifactLoader> loader;
    std::optional<BodyBackingManager::LocalByteSpan> local_source;
    std::optional<CompositeStageCandidate> composite_candidate;
    store::loading::MaterializationSource source_kind = store::loading::MaterializationSource::kLocalReplica;
    std::optional<BodyBackingManager::StageResult> staged_body;
    if (has_batch_payload_slice) {
      const auto transport_it = batch_transports_by_id.find(item.batch_payload_slice().transport_id());
      if (transport_it == batch_transports_by_id.end()) {
        deferred_outcomes[index] =
            make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "batch transport is missing");
        continue;
      }

      const auto* manifest_entry = find_batch_payload_entry(
          transport_it->second->manifest(),
          artifact_id,
          item.batch_payload_slice().offset(),
          item.batch_payload_slice().length());
      if (manifest_entry == nullptr) {
        deferred_outcomes[index] = make_outcome(
            artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "batch transport manifest entry is missing");
        continue;
      }
      if (transport_it->second->has_communicator_source()) {
        const std::string transport_id = item.batch_payload_slice().transport_id();
        auto resolved_it = resolved_batch_sources.find(transport_id);
        if (resolved_it == resolved_batch_sources.end()) {
          const absl::Time resolve_started_at = absl::Now();
          auto resolved_or = d_.payload_transport_broker.open_batch_payload_communicator_source(
              d_.worker_directory_cache,
              now,
              absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
              local_daemon_id,
              *transport_it->second,
              tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
              operation_id);
          if (!resolved_or.ok()) {
            deferred_outcomes[index] = make_outcome(
                artifact_id,
                batch_item_status_from_absl_status(resolved_or.status()),
                std::string(resolved_or.status().message()));
            continue;
          }
          timing_stats.communicator_open_elapsed += absl::Now() - resolve_started_at;
          ++timing_stats.communicator_open_count;
          resolved_it = resolved_batch_sources.emplace(transport_id, std::move(*resolved_or)).first;
          const auto eligibility = classify_put_remote_communicator_source(resolved_it->second);
          if (eligibility.remote) {
            ++timing_stats.remote_communicator_source_count;
            if (eligibility.source_supports_direct_write) {
              ++timing_stats.remote_communicator_source_direct_write_count;
            }
            if (eligibility.source_supports_batched_direct_write) {
              ++timing_stats.remote_communicator_source_batched_direct_write_count;
            }
          }
          LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_open"
                    << " operation_id=" << operation_id << " transport_id=" << transport_id
                    << " kind=communicator_source"
                    << " remote=" << eligibility.remote
                    << " item_count=" << transport_it->second->manifest().entries_size()
                    << " payload_bytes=" << transport_it->second->manifest().total_size()
                    << " source_direct_write_at=" << eligibility.source_supports_direct_write
                    << " source_batched_direct_write_at=" << eligibility.source_supports_batched_direct_write
                    << " resolve_ms=" << absl::ToDoubleMilliseconds(absl::Now() - resolve_started_at);
        }
        source_kind = resolved_it->second.remote ? store::loading::MaterializationSource::kP2P
                                                 : store::loading::MaterializationSource::kLocalReplica;
        if (resolved_it->second.remote) {
          const auto eligibility = classify_put_remote_communicator_source(resolved_it->second);
          const bool composite_stage_eligible = eligibility.direct_remote_slice &&
              eligibility.source_supports_batched_direct_write &&
              !verification_mode_requires_payload_digest(invariant_verification_mode(item.invariant()));
          if (composite_stage_eligible) {
            composite_candidate = CompositeStageCandidate{
                .transport_id = transport_id,
                .source_offset = item.batch_payload_slice().offset(),
                .length = item.batch_payload_slice().length(),
            };
          } else if (eligibility.direct_remote_slice) {
            if (direct_remote_batch_payloads.emplace(transport_id).second) {
              ++timing_stats.remote_direct_slice_transport_count;
              timing_stats.remote_direct_slice_bytes += transport_it->second->manifest().total_size();
              timing_stats.remote_direct_slice_items += transport_it->second->manifest().entries_size();
              LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_read_mode"
                        << " operation_id=" << operation_id << " transport_id=" << transport_id
                        << " kind=communicator_source"
                        << " remote=true"
                        << " read_mode=direct_remote_slice"
                        << " realization=source_slice_loader"
                        << " payload_bytes=" << transport_it->second->manifest().total_size()
                        << " item_count=" << transport_it->second->manifest().entries_size()
                        << " source_direct_write_at=" << eligibility.source_supports_direct_write
                        << " source_batched_direct_write_at=" << eligibility.source_supports_batched_direct_write
                        << " mirror_ms=0"
                        << " subsequent_item_slices_local=false";
            }
            auto& source_mutex = remote_direct_source_mutexes[transport_id];
            if (source_mutex == nullptr) {
              source_mutex = std::make_shared<std::mutex>();
            }
            loader = make_loader_from_source_slice(
                resolved_it->second.source,
                item.batch_payload_slice().offset(),
                item.batch_payload_slice().length(),
                source_mutex);
          } else {
            auto mirrored_it = mirrored_remote_batch_payloads.find(transport_id);
            if (mirrored_it == mirrored_remote_batch_payloads.end()) {
              const absl::Time mirror_started_at = absl::Now();
              auto mirrored_or = mirror_seekable_source_payload(
                  resolved_it->second.source, transport_it->second->manifest().total_size());
              if (!mirrored_or.ok()) {
                deferred_outcomes[index] = make_outcome(
                    artifact_id,
                    batch_item_status_from_absl_status(mirrored_or.status()),
                    std::string(mirrored_or.status().message()));
                continue;
              }
              const absl::Duration mirror_elapsed = absl::Now() - mirror_started_at;
              timing_stats.remote_mirror_elapsed += mirror_elapsed;
              ++timing_stats.remote_mirror_count;
              timing_stats.remote_mirror_bytes += transport_it->second->manifest().total_size();
              timing_stats.remote_full_pack_mirror_items += transport_it->second->manifest().entries_size();
              mirrored_it = mirrored_remote_batch_payloads.emplace(transport_id, std::move(*mirrored_or)).first;
              LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_mirror"
                        << " operation_id=" << operation_id << " transport_id=" << transport_id
                        << " kind=communicator_source"
                        << " remote=true"
                        << " read_mode=full_pack"
                        << " realization=full_pack_mirror"
                        << " payload_bytes=" << transport_it->second->manifest().total_size()
                        << " item_count=" << transport_it->second->manifest().entries_size()
                        << " source_direct_write_at=" << eligibility.source_supports_direct_write
                        << " source_batched_direct_write_at=" << eligibility.source_supports_batched_direct_write
                        << " mirror_ms=" << absl::ToDoubleMilliseconds(mirror_elapsed)
                        << " subsequent_item_slices_local=true";
            }
            if (item.batch_payload_slice().offset() + item.batch_payload_slice().length() >
                mirrored_it->second->size()) {
              deferred_outcomes[index] = make_outcome(
                  artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "batch transport payload is truncated");
              continue;
            }
            local_source = BodyBackingManager::LocalByteSpan{
                .owner = std::shared_ptr<const void>(
                    mirrored_it->second, static_cast<const void*>(mirrored_it->second->data())),
                .data = reinterpret_cast<const std::uint8_t*>(mirrored_it->second->data()) +
                    item.batch_payload_slice().offset(),
                .size_bytes = item.batch_payload_slice().length(),
            };
          }
        } else {
          loader = make_loader_from_source_slice(
              resolved_it->second.source, item.batch_payload_slice().offset(), item.batch_payload_slice().length());
        }
      } else if (transport_it->second->has_grpc_chunk_ref()) {
        auto resolved_it = resolved_batch_payloads.find(item.batch_payload_slice().transport_id());
        if (resolved_it == resolved_batch_payloads.end()) {
          const absl::Time fetch_started_at = absl::Now();
          auto resolved_or = d_.payload_transport_broker.fetch_batch_payload_ref(
              d_.worker_directory_cache,
              now,
              absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
              local_daemon_id,
              transport_it->second->grpc_chunk_ref().batch_payload_ref(),
              tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
              operation_id);
          if (!resolved_or.ok()) {
            deferred_outcomes[index] = make_outcome(
                artifact_id,
                batch_item_status_from_absl_status(resolved_or.status()),
                std::string(resolved_or.status().message()));
            continue;
          }
          timing_stats.grpc_fetch_elapsed += absl::Now() - fetch_started_at;
          ++timing_stats.grpc_fetch_count;
          resolved_it =
              resolved_batch_payloads.emplace(item.batch_payload_slice().transport_id(), std::move(*resolved_or)).first;
        }
        if (!resolved_it->second.payload ||
            item.batch_payload_slice().offset() + item.batch_payload_slice().length() >
                resolved_it->second.payload->size()) {
          deferred_outcomes[index] = make_outcome(
              artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "batch transport payload is truncated");
          continue;
        }
        local_source = BodyBackingManager::LocalByteSpan{
            .owner = std::shared_ptr<const void>(
                resolved_it->second.payload, static_cast<const void*>(resolved_it->second.payload->data())),
            .data = reinterpret_cast<const std::uint8_t*>(resolved_it->second.payload->data()) +
                item.batch_payload_slice().offset(),
            .size_bytes = item.batch_payload_slice().length(),
        };
        source_kind = resolved_it->second.remote ? store::loading::MaterializationSource::kP2P
                                                 : store::loading::MaterializationSource::kLocalReplica;
      } else {
        deferred_outcomes[index] =
            make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "batch transport kind is unsupported");
        continue;
      }
    }
    if (loader != nullptr || local_source.has_value() || composite_candidate.has_value()) {
      // Batch transport already provided a readable source.
    } else if (!item.inline_payload().empty()) {
      auto payload = std::make_shared<std::string>(item.inline_payload());
      local_source = BodyBackingManager::LocalByteSpan{
          .owner = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data())),
          .data = reinterpret_cast<const std::uint8_t*>(payload->data()),
          .size_bytes = payload->size(),
      };
    } else if (!item.payload_ref().empty()) {
      auto source_capability_or = d_.payload_transport_broker.resolve_payload_ref_capability(
          d_.worker_directory_cache,
          item.payload_ref(),
          artifact_id,
          now,
          absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
          local_daemon_id,
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
          operation_id);
      if (!source_capability_or.ok()) {
        deferred_outcomes[index] = make_outcome(
            artifact_id,
            batch_item_status_from_absl_status(source_capability_or.status()),
            std::string(source_capability_or.status().message()));
        continue;
      }
      if (source_capability_or->body_capability.has_value()) {
        const absl::Time reuse_started_at = absl::Now();
        auto reused_or = body_backing_manager_.try_reuse_body(
            BodyBackingManager::ReuseRequest{
                .artifact_id = artifact_id,
                .invariant = item.invariant(),
                .descriptor = source_capability_or->body_capability->descriptor,
                .body_handle = source_capability_or->body_capability->body_handle,
                .operation_id = std::string(operation_id),
                .access_class = BodyAccessClass::kHomeDefault,
                .route_role = BodyRouteRole::kHomeAuthority,
            });
        timing_stats.reuse_elapsed += absl::Now() - reuse_started_at;
        ++timing_stats.reuse_attempt_count;
        if (!reused_or.ok()) {
          deferred_outcomes[index] = make_outcome(
              artifact_id,
              batch_item_status_from_absl_status(reused_or.status()),
              std::string(reused_or.status().message()));
          continue;
        }
        if (reused_or->has_value()) {
          ++timing_stats.reuse_hit_count;
          staged_body = std::move(**reused_or);
        }
      }
      if (!staged_body.has_value()) {
        auto loader_or = open_loader_from_resolved_source_capability(
            d_.payload_transport_broker,
            d_.worker_directory_cache,
            *source_capability_or,
            now,
            absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
            local_daemon_id,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
            operation_id);
        if (!loader_or.ok()) {
          deferred_outcomes[index] = make_outcome(
              artifact_id,
              batch_item_status_from_absl_status(loader_or.status()),
              std::string(loader_or.status().message()));
          continue;
        }
        source_kind = loader_or->source_kind;
        loader = std::move(loader_or->loader);
      }
    } else {
      deferred_outcomes[index] = make_outcome(
          artifact_id,
          v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT,
          "inline_payload, payload_ref, or batch_payload_slice is required");
      continue;
    }
    prepared_items[static_cast<std::size_t>(index)] = PreparedHomeBatchPutItem{
        .item = &item,
        .artifact_id = artifact_id,
        .loader = std::move(loader),
        .local_source = std::move(local_source),
        .composite_candidate = std::move(composite_candidate),
        .source_kind = source_kind,
        .staged_body = std::move(staged_body),
    };
  }

  const auto fallback_composite_group_to_direct_slice =
      [&](const std::string& transport_id, const std::vector<int>& indices, std::string_view reason) {
        ++timing_stats.remote_composite_fallback_count;
        timing_stats.remote_composite_fallback_items += indices.size();
        const auto transport_it = batch_transports_by_id.find(transport_id);
        const auto resolved_it = resolved_batch_sources.find(transport_id);
        if (transport_it == batch_transports_by_id.end() || resolved_it == resolved_batch_sources.end() ||
            resolved_it->second.source == nullptr || !resolved_it->second.source->supports_direct_write_at()) {
          for (const int index : indices) {
            deferred_outcomes[index] = make_outcome(
                req.items(index).artifact_id(),
                v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
                absl::StrCat("composite fallback unavailable: ", reason));
          }
          return;
        }
        const auto eligibility = classify_put_remote_communicator_source(resolved_it->second);
        LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_apply_summary"
                  << " operation_id=" << operation_id << " transport_id=" << transport_id << " kind=communicator_source"
                  << " remote=true"
                  << " read_mode=direct_remote_slice"
                  << " materialize_mode=per_item"
                  << " stage_mode=source_slice_loader"
                  << " batched_direct_write=false"
                  << " source_count=1"
                  << " mapping_segments=0"
                  << " item_count=" << indices.size() << " item_bytes=0"
                  << " transport_payload_bytes=" << transport_it->second->manifest().total_size() << " mirror_ms=0"
                  << " fallback_reason=" << reason;
        if (direct_remote_batch_payloads.emplace(transport_id).second) {
          ++timing_stats.remote_direct_slice_transport_count;
          timing_stats.remote_direct_slice_bytes += transport_it->second->manifest().total_size();
          timing_stats.remote_direct_slice_items += transport_it->second->manifest().entries_size();
          LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_read_mode"
                    << " operation_id=" << operation_id << " transport_id=" << transport_id
                    << " kind=communicator_source"
                    << " remote=true"
                    << " read_mode=direct_remote_slice"
                    << " realization=source_slice_loader"
                    << " payload_bytes=" << transport_it->second->manifest().total_size()
                    << " item_count=" << transport_it->second->manifest().entries_size()
                    << " source_direct_write_at=" << eligibility.source_supports_direct_write
                    << " source_batched_direct_write_at=" << eligibility.source_supports_batched_direct_write
                    << " mirror_ms=0"
                    << " subsequent_item_slices_local=false";
        }
        auto& source_mutex = remote_direct_source_mutexes[transport_id];
        if (source_mutex == nullptr) {
          source_mutex = std::make_shared<std::mutex>();
        }
        for (const int index : indices) {
          if (deferred_outcomes[index].has_value()) {
            continue;
          }
          auto& prepared_item = prepared_items[static_cast<std::size_t>(index)];
          if (!prepared_item.composite_candidate.has_value()) {
            continue;
          }
          prepared_item.loader = make_loader_from_source_slice(
              resolved_it->second.source,
              prepared_item.composite_candidate->source_offset,
              prepared_item.composite_candidate->length,
              source_mutex);
          prepared_item.composite_candidate.reset();
        }
      };

  absl::flat_hash_map<std::string, std::vector<int>> composite_groups;
  for (int index = 0; index < req.items_size(); ++index) {
    if (deferred_outcomes[index].has_value()) {
      continue;
    }
    const auto& prepared_item = prepared_items[static_cast<std::size_t>(index)];
    if (prepared_item.composite_candidate.has_value()) {
      composite_groups[prepared_item.composite_candidate->transport_id].push_back(index);
    }
  }
  for (const auto& [transport_id, indices] : composite_groups) {
    if (indices.empty()) {
      continue;
    }
    const auto transport_it = batch_transports_by_id.find(transport_id);
    const auto resolved_it = resolved_batch_sources.find(transport_id);
    if (transport_it == batch_transports_by_id.end() || resolved_it == resolved_batch_sources.end() ||
        resolved_it->second.source == nullptr) {
      for (const int index : indices) {
        deferred_outcomes[index] = make_outcome(
            req.items(index).artifact_id(),
            v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
            "composite transport source is missing");
      }
      continue;
    }
    absl::flat_hash_set<std::string> seen_artifacts;
    bool has_duplicate = false;
    for (const int index : indices) {
      if (!seen_artifacts.emplace(prepared_items[static_cast<std::size_t>(index)].artifact_id).second) {
        has_duplicate = true;
        break;
      }
    }
    if (has_duplicate) {
      fallback_composite_group_to_direct_slice(transport_id, indices, "duplicate_key");
      continue;
    }

    std::vector<BodyBackingManager::CompositeStageItem> composite_items;
    composite_items.reserve(indices.size());
    std::uint64_t item_bytes = 0;
    bool invalid_group = false;
    for (const int index : indices) {
      const auto& prepared_item = prepared_items[static_cast<std::size_t>(index)];
      if (!prepared_item.composite_candidate.has_value() || prepared_item.item == nullptr) {
        invalid_group = true;
        break;
      }
      if (item_bytes > std::numeric_limits<std::uint64_t>::max() - prepared_item.composite_candidate->length) {
        invalid_group = true;
        break;
      }
      item_bytes += prepared_item.composite_candidate->length;
      composite_items.push_back(
          BodyBackingManager::CompositeStageItem{
              .artifact_id = prepared_item.artifact_id,
              .invariant = prepared_item.item->invariant(),
              .source_offset = prepared_item.composite_candidate->source_offset,
              .length = prepared_item.composite_candidate->length,
              .access_class = BodyAccessClass::kHomeDefault,
              .route_role = BodyRouteRole::kHomeAuthority,
          });
    }
    if (invalid_group) {
      fallback_composite_group_to_direct_slice(transport_id, indices, "mapping_invalid");
      continue;
    }

    const absl::Time composite_started_at = absl::Now();
    auto composite_or = body_backing_manager_.stage_bodies_composite(
        BodyBackingManager::StageBodiesCompositeRequest{
            .source = resolved_it->second.source,
            .items = std::move(composite_items),
            .source_kind = store::loading::MaterializationSource::kP2P,
            .operation_id = std::string(operation_id),
            .transport_id = transport_id,
        });
    const absl::Duration composite_elapsed = absl::Now() - composite_started_at;
    timing_stats.remote_composite_stage_elapsed += composite_elapsed;
    if (!composite_or.ok()) {
      LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_apply_summary"
                << " operation_id=" << operation_id << " transport_id=" << transport_id << " kind=communicator_source"
                << " remote=true"
                << " read_mode=batched_direct_write"
                << " materialize_mode=single_source_composite"
                << " stage_mode=composite_final_body"
                << " batched_direct_write=true"
                << " source_count=1"
                << " mapping_segments=" << indices.size() << " item_count=" << indices.size()
                << " item_bytes=" << item_bytes
                << " transport_payload_bytes=" << transport_it->second->manifest().total_size() << " mirror_ms=0"
                << " materialize_ms=" << absl::ToDoubleMilliseconds(composite_elapsed) << " outcome=failed"
                << " status=" << composite_or.status();
      for (const int index : indices) {
        deferred_outcomes[index] = make_outcome(
            prepared_items[static_cast<std::size_t>(index)].artifact_id,
            batch_item_status_from_absl_status(composite_or.status()),
            std::string(composite_or.status().message()));
      }
      continue;
    }
    if (composite_or->staged_bodies.size() != indices.size()) {
      for (auto& staged_body : composite_or->staged_bodies) {
        (void)staged_body.body_handle.retire();
      }
      for (const int index : indices) {
        deferred_outcomes[index] = make_outcome(
            prepared_items[static_cast<std::size_t>(index)].artifact_id,
            v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
            "composite stage returned unexpected item count");
      }
      continue;
    }
    ++timing_stats.remote_composite_stage_transport_count;
    timing_stats.remote_composite_stage_items += indices.size();
    timing_stats.remote_composite_stage_bytes += item_bytes;
    ++timing_stats.remote_composite_materialize_calls;
    if (composite_or->materialize_result.direct_write_supported) {
      ++timing_stats.remote_composite_batched_direct_write_count;
    }
    LOG(INFO) << "byte_artifact.home_batch_put_if_absent_transport_apply_summary"
              << " operation_id=" << operation_id << " transport_id=" << transport_id << " kind=communicator_source"
              << " remote=true"
              << " read_mode=batched_direct_write"
              << " materialize_mode=single_source_composite"
              << " stage_mode=composite_final_body"
              << " batched_direct_write=true"
              << " source_count=1"
              << " mapping_segments=" << indices.size() << " item_count=" << indices.size()
              << " item_bytes=" << item_bytes
              << " transport_payload_bytes=" << transport_it->second->manifest().total_size() << " mirror_ms=0"
              << " materialize_ms=" << absl::ToDoubleMilliseconds(composite_elapsed)
              << " direct_write_supported=" << composite_or->materialize_result.direct_write_supported
              << " fallback_reason=none";
    for (std::size_t local_index = 0; local_index < indices.size(); ++local_index) {
      auto& prepared_item = prepared_items[static_cast<std::size_t>(indices[local_index])];
      prepared_item.staged_body = std::move(composite_or->staged_bodies[local_index]);
      prepared_item.loader.reset();
      prepared_item.local_source.reset();
      prepared_item.composite_candidate.reset();
    }
  }

  std::vector<StageWorkItem> stage_work_items;
  stage_work_items.reserve(prepared_items.size());
  for (int index = 0; index < req.items_size(); ++index) {
    if (deferred_outcomes[index].has_value()) {
      continue;
    }
    auto& prepared_item = prepared_items[static_cast<std::size_t>(index)];
    if (prepared_item.item == nullptr || prepared_item.staged_body.has_value()) {
      continue;
    }
    if (prepared_item.loader == nullptr && !prepared_item.local_source.has_value()) {
      deferred_outcomes[index] =
          make_outcome(req.items(index).artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing stage loader");
      continue;
    }
    stage_work_items.push_back(
        StageWorkItem{
            .index = index,
            .artifact_id = prepared_item.artifact_id,
            .loader = std::move(prepared_item.loader),
            .local_source = std::move(prepared_item.local_source),
            .source_kind = prepared_item.source_kind,
        });
    if (stage_work_items.back().local_source.has_value()) {
      ++timing_stats.stage_local_source_count;
    } else {
      ++timing_stats.stage_loader_count;
    }
    if (stage_work_items.back().source_kind == store::loading::MaterializationSource::kP2P) {
      ++timing_stats.stage_p2p_count;
    } else {
      ++timing_stats.stage_local_replica_count;
    }
  }

  const auto run_stage_work_item = [&](StageWorkItem stage_work_item) -> StageWorkItemResult {
    StageWorkItemResult result;
    result.index = stage_work_item.index;
    result.artifact_id = std::move(stage_work_item.artifact_id);
    const absl::Time stage_started_at = absl::Now();
    absl::StatusOr<BodyBackingManager::StageResult> staged_body_or = [&]() {
      if (stage_work_item.local_source.has_value()) {
        result.used_fast_cpu_stage = true;
        return body_backing_manager_.stage_body_fast_cpu_verified(
            result.artifact_id,
            req.items(result.index).invariant(),
            std::move(*stage_work_item.local_source),
            stage_work_item.source_kind,
            std::string(operation_id),
            BodyAccessClass::kHomeDefault,
            BodyRouteRole::kHomeAuthority);
      }
      return body_backing_manager_.stage_body(
          BodyBackingManager::StageRequest{
              .artifact_id = result.artifact_id,
              .invariant = req.items(result.index).invariant(),
              .loader = std::move(stage_work_item.loader),
              .source_kind = stage_work_item.source_kind,
              .operation_id = std::string(operation_id),
              .access_class = BodyAccessClass::kHomeDefault,
              .route_role = BodyRouteRole::kHomeAuthority,
          });
    }();
    result.elapsed = absl::Now() - stage_started_at;
    if (!staged_body_or.ok()) {
      result.status = staged_body_or.status();
      return result;
    }
    result.staged_body = std::move(*staged_body_or);
    return result;
  };
  const auto apply_stage_result = [&](StageWorkItemResult result) {
    timing_stats.stage_body_elapsed += result.elapsed;
    ++timing_stats.stage_body_count;
    if (result.used_fast_cpu_stage) {
      timing_stats.fast_cpu_stage_elapsed += result.elapsed;
      ++timing_stats.fast_cpu_stage_count;
    }
    if (!result.status.ok()) {
      deferred_outcomes[result.index] = make_outcome(
          result.artifact_id, batch_item_status_from_absl_status(result.status), std::string(result.status.message()));
      return;
    }
    if (!result.staged_body.has_value()) {
      deferred_outcomes[result.index] =
          make_outcome(result.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing staged body");
      return;
    }
    prepared_items[static_cast<std::size_t>(result.index)].staged_body = std::move(result.staged_body);
  };

  const bool parallel_stage_enabled = batch_get_apply_pool_ != nullptr && batch_get_apply_threads_ > 1;
  const bool parallel_stage_active = parallel_stage_enabled && stage_work_items.size() > 1;
  LOG(INFO) << "byte_artifact.home_batch_put_if_absent_stage_plan"
            << " operation_id=" << operation_id << " stage_work_items=" << stage_work_items.size()
            << " stage_executor_threads=" << batch_get_apply_threads_ << " parallel_enabled=" << parallel_stage_enabled
            << " parallel_active=" << parallel_stage_active;
  const absl::Time stage_wall_started_at = absl::Now();
  if (!parallel_stage_active) {
    for (auto& stage_work_item : stage_work_items) {
      apply_stage_result(run_stage_work_item(std::move(stage_work_item)));
    }
  } else {
    std::vector<ScheduledStageWorkItem> scheduled_stage_items;
    scheduled_stage_items.reserve(stage_work_items.size());
    for (auto& stage_work_item : stage_work_items) {
      scheduled_stage_items.push_back(
          ScheduledStageWorkItem{
              .index = stage_work_item.index,
              .artifact_id = stage_work_item.artifact_id,
              .future = folly::via(
                            folly::getKeepAliveToken(*batch_get_apply_pool_),
                            [run_stage_work_item, stage_work_item = std::move(stage_work_item)]() mutable
                                -> StageWorkItemResult { return run_stage_work_item(std::move(stage_work_item)); })
                            .semi(),
          });
    }
    for (auto& scheduled_stage_item : scheduled_stage_items) {
      try {
        apply_stage_result(std::move(scheduled_stage_item.future).get());
      } catch (const std::exception& ex) {
        deferred_outcomes[scheduled_stage_item.index] =
            make_outcome(scheduled_stage_item.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, ex.what());
      } catch (...) {
        deferred_outcomes[scheduled_stage_item.index] = make_outcome(
            scheduled_stage_item.artifact_id,
            v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
            "HomeBatchPutIfAbsent stage task failed");
      }
    }
  }
  const absl::Duration stage_wall_elapsed = absl::Now() - stage_wall_started_at;

  for (int index = 0; index < req.items_size(); ++index) {
    if (deferred_outcomes[index].has_value()) {
      continue;
    }
    auto& prepared_item = prepared_items[static_cast<std::size_t>(index)];
    if (prepared_item.item == nullptr || !prepared_item.staged_body.has_value()) {
      deferred_outcomes[index] =
          make_outcome(req.items(index).artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing staged body");
      continue;
    }

    auto& staged_body = prepared_item.staged_body;
    const auto invariant_st = byte_artifact_runtime().validate_invariant_body_descriptor(
        prepared_item.item->invariant(), staged_body->descriptor);
    if (!invariant_st.ok()) {
      auto retire_status = staged_body->body_handle.retire();
      if (!retire_status.ok()) {
        deferred_outcomes[index] = make_outcome(
            prepared_item.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(retire_status.message()));
        continue;
      }
      deferred_outcomes[index] = make_outcome(
          prepared_item.artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(invariant_st.message()));
      continue;
    }

    authority_items.push_back(
        ByteArtifactAuthorityService::PutItem{
            .artifact_id = prepared_item.artifact_id,
            .invariant = prepared_item.item->invariant(),
            .descriptor = staged_body->descriptor,
            .verified_content_descriptor = staged_body->verified_content_descriptor,
            .verification_record = staged_body->verification_record,
            .backing_identity = staged_body->backing_identity,
            .observation = staged_body->observation,
            .body_handle = staged_body->body_handle,
        });
    authority_item_indices.push_back(index);
  }

  timing_stats.authority_item_count = authority_items.size();
  const absl::Time authority_apply_started_at = absl::Now();
  const auto authority_outcomes = authority_service_.batch_put_if_absent(
      authority_items,
      ByteArtifactAuthorityService::Context{
          .shard_id = req.fence().shard_id(),
          .lease_generation = home_lease_or->lease_generation,
          .routing_epoch = options_.routing.routing_epoch,
          .shard_count = options_.routing.shard_count,
          .now = now,
      },
      ttl_ms);
  timing_stats.authority_apply_elapsed = absl::Now() - authority_apply_started_at;
  for (size_t index = 0; index < authority_outcomes.size(); ++index) {
    deferred_outcomes[authority_item_indices[index]] = authority_outcomes[index];
  }
  if (options_.publish_prereg.enabled) {
    for (size_t index = 0; index < authority_outcomes.size(); ++index) {
      if (authority_outcomes[index].status() != v2::BATCH_ITEM_STATUS_OK) {
        continue;
      }
      auto entry = d_.body_store.get(
          authority_items[index].artifact_id,
          req.fence().shard_id(),
          home_lease_or->lease_generation,
          options_.routing.routing_epoch,
          now);
      if (!entry.has_value()) {
        VLOG(2) << "byte_artifact.publish_prereg.skip"
                << " artifact_id=" << authority_items[index].artifact_id
                << " published_at_ms=" << absl::ToUnixMillis(now) << " reason=canonical_backing_missing";
        continue;
      }
      publish_preregistered_export(authority_items[index].artifact_id, entry->backing_record.retained_body_handle, now);
    }
  }

  for (int index = 0; index < req.items_size(); ++index) {
    if (deferred_outcomes[index].has_value()) {
      *resp.add_outcomes() = std::move(*deferred_outcomes[index]);
      continue;
    }
    *resp.add_outcomes() =
        make_outcome(req.items(index).artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing authority outcome");
  }
  VLOG(2) << "byte_artifact.home_batch_put_if_absent_summary"
          << " operation_id=" << operation_id << " requested_items=" << req.items_size()
          << " batch_payload_slice_items=" << timing_stats.batch_payload_slice_items
          << " inline_payload_items=" << timing_stats.inline_payload_items
          << " payload_ref_items=" << timing_stats.payload_ref_items
          << " communicator_open_count=" << timing_stats.communicator_open_count
          << " grpc_fetch_count=" << timing_stats.grpc_fetch_count
          << " remote_communicator_source_count=" << timing_stats.remote_communicator_source_count
          << " remote_communicator_source_direct_write_count="
          << timing_stats.remote_communicator_source_direct_write_count
          << " remote_communicator_source_batched_direct_write_count="
          << timing_stats.remote_communicator_source_batched_direct_write_count
          << " remote_direct_slice_transport_count=" << timing_stats.remote_direct_slice_transport_count
          << " remote_direct_slice_items=" << timing_stats.remote_direct_slice_items
          << " remote_direct_slice_bytes=" << timing_stats.remote_direct_slice_bytes
          << " remote_composite_stage_transport_count=" << timing_stats.remote_composite_stage_transport_count
          << " remote_composite_stage_items=" << timing_stats.remote_composite_stage_items
          << " remote_composite_stage_bytes=" << timing_stats.remote_composite_stage_bytes
          << " remote_composite_materialize_calls=" << timing_stats.remote_composite_materialize_calls
          << " remote_composite_batched_direct_write_count=" << timing_stats.remote_composite_batched_direct_write_count
          << " remote_composite_fallback_count=" << timing_stats.remote_composite_fallback_count
          << " remote_composite_fallback_items=" << timing_stats.remote_composite_fallback_items
          << " remote_mirror_count=" << timing_stats.remote_mirror_count
          << " remote_mirror_bytes=" << timing_stats.remote_mirror_bytes
          << " remote_full_pack_mirror_items=" << timing_stats.remote_full_pack_mirror_items
          << " stage_body_count=" << timing_stats.stage_body_count
          << " fast_cpu_stage_count=" << timing_stats.fast_cpu_stage_count
          << " stage_loader_count=" << timing_stats.stage_loader_count
          << " stage_local_source_count=" << timing_stats.stage_local_source_count
          << " stage_p2p_count=" << timing_stats.stage_p2p_count
          << " stage_local_replica_count=" << timing_stats.stage_local_replica_count
          << " stage_body_wall_ms=" << absl::ToDoubleMilliseconds(stage_wall_elapsed)
          << " reuse_attempt_count=" << timing_stats.reuse_attempt_count
          << " reuse_hit_count=" << timing_stats.reuse_hit_count
          << " authority_item_count=" << timing_stats.authority_item_count
          << " communicator_open_ms=" << absl::ToDoubleMilliseconds(timing_stats.communicator_open_elapsed)
          << " grpc_fetch_ms=" << absl::ToDoubleMilliseconds(timing_stats.grpc_fetch_elapsed)
          << " remote_mirror_ms=" << absl::ToDoubleMilliseconds(timing_stats.remote_mirror_elapsed)
          << " remote_composite_stage_ms=" << absl::ToDoubleMilliseconds(timing_stats.remote_composite_stage_elapsed)
          << " stage_body_ms=" << absl::ToDoubleMilliseconds(timing_stats.stage_body_elapsed)
          << " fast_cpu_stage_ms=" << absl::ToDoubleMilliseconds(timing_stats.fast_cpu_stage_elapsed)
          << " reuse_ms=" << absl::ToDoubleMilliseconds(timing_stats.reuse_elapsed)
          << " authority_apply_ms=" << absl::ToDoubleMilliseconds(timing_stats.authority_apply_elapsed)
          << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::home_batch_touch_ttl(
    RpcContext& rctx,
    const v2::HomeBatchTouchTtlRequest& req,
    v2::HomeBatchTouchTtlResponse& resp) {
  if (req.ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms must be > 0"};
  }
  const absl::Time now = absl::Now();
  auto home_lease_or = d_.route_resolver.ensure_home_lease(req.fence(), now);
  if (!home_lease_or.ok()) {
    return to_grpc_status(home_lease_or.status());
  }
  if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
    if (home_lease_or->redirect.lease_generation() != 0 && !home_lease_or->redirect.holder_daemon_id().empty()) {
      resp.mutable_redirect()->CopyFrom(home_lease_or->redirect);
    }
    for (const auto& artifact_id : req.artifact_ids()) {
      auto* outcome = resp.add_outcomes();
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, home_lease_or->message);
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::string> artifact_ids;
  artifact_ids.reserve(req.artifact_ids_size());
  for (const auto& artifact_id : req.artifact_ids()) {
    artifact_ids.push_back(artifact_id);
  }
  for (auto outcome : authority_service_.batch_touch_ttl(
           artifact_ids,
           ByteArtifactAuthorityService::Context{
               .shard_id = req.fence().shard_id(),
               .lease_generation = home_lease_or->lease_generation,
               .routing_epoch = options_.routing.routing_epoch,
               .shard_count = options_.routing.shard_count,
               .now = now,
           },
           req.ttl_ms())) {
    *resp.add_outcomes() = std::move(outcome);
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_exists(
    RpcContext& rctx,
    const v2::BatchExistsRequest& req,
    v2::BatchExistsResponse& resp) {
  const absl::Time total_started_at = absl::Now();
  const bool local_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!local_peer && !options_.gateway_ingress_enabled) {
    return {StatusCode::PERMISSION_DENIED, "BatchExists requires gateway ingress on non-local peers"};
  }

  const absl::Time now = absl::Now();
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();
  const bool allow_high_card_attrs = rctx.allow_high_card_attrs();
  const std::string first_artifact_id = req.selections_size() == 0 ? std::string() : req.selections(0).artifact_id();

  struct ShardRequest {
    std::vector<std::string> artifact_ids;
    std::vector<int> outcome_indices;
  };

  absl::flat_hash_map<std::uint64_t, ShardRequest> shard_requests;
  shard_requests.reserve(static_cast<size_t>(req.selections_size()));

  for (int i = 0; i < req.selections_size(); ++i) {
    const auto& selection = req.selections(i);
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(selection.artifact_id());

    const auto selection_st = validate_batch_selection(selection);
    if (!selection_st.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(selection_st.message()));
      continue;
    }
    auto shard_or =
        byte_artifact_runtime().shard_id_for_artifact(selection.artifact_id(), options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }

    auto& entry = shard_requests[*shard_or];
    entry.artifact_ids.push_back(selection.artifact_id());
    entry.outcome_indices.push_back(i);
  }

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    for (const auto& [shard_id, batch] : shard_requests) {
      const auto outcomes = authority_service_.batch_exists(
          batch.artifact_ids,
          ByteArtifactAuthorityService::Context{
              .shard_id = shard_id,
              .lease_generation = 1,
              .routing_epoch = options_.routing.routing_epoch,
              .shard_count = options_.routing.shard_count,
              .now = now,
          });
      for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        *resp.mutable_outcomes(batch.outcome_indices[idx]) = outcomes.at(idx);
      }
    }
    VLOG(2) << "byte_artifact.batch_exists_summary"
            << " selections=" << req.selections_size() << " shard_count=" << shard_requests.size()
            << " first_artifact_id=" << first_artifact_id << " local_home_batches=" << shard_requests.size()
            << " remote_home_batches=0"
            << " local_home_items=" << req.selections_size() << " remote_home_items=0"
            << " local_home_rpc_ms=0 remote_home_rpc_ms=0"
            << " local_home_rpc_max_ms=0 remote_home_rpc_max_ms=0"
            << " route_resolve_ms=0 worker_directory_warm_ms=0 total_wall_ms="
            << absl::ToDoubleMilliseconds(absl::Now() - total_started_at)
            << " parallel_active=" << (shard_requests.size() > 1);
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_requests.size());
  for (const auto& [shard_id, /*batch*/ _] : shard_requests) {
    shard_ids.push_back(shard_id);
  }
  const absl::Time route_resolve_started_at = absl::Now();
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);
  const absl::Duration route_resolve_elapsed = absl::Now() - route_resolve_started_at;

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (!route.ok || route.holder_daemon_id == local_daemon_id) {
      continue;
    }
    remote_daemon_ids.push_back(route.holder_daemon_id);
  }
  absl::Duration worker_directory_warm_elapsed = absl::ZeroDuration();
  if (!remote_daemon_ids.empty()) {
    const absl::Time worker_directory_warm_started_at = absl::Now();
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    worker_directory_warm_elapsed = absl::Now() - worker_directory_warm_started_at;
  }

  struct PendingHomeBatchExists {
    std::uint64_t shard_id{0};
    const ShardRequest* batch{nullptr};
    std::string holder_daemon_id;
    std::uint64_t lease_generation{0};
    bool remote_home{false};
    int attempt{0};
  };

  struct HomeBatchExistsRpcResult {
    PendingHomeBatchExists request;
    grpc::Status status;
    v2::HomeBatchExistsResponse home_resp;
    absl::Duration rpc_elapsed{absl::ZeroDuration()};
  };

  struct ScheduledHomeBatchExists {
    PendingHomeBatchExists request;
    folly::SemiFuture<HomeBatchExistsRpcResult> future;
  };

  struct BatchExistsStats {
    std::size_t local_home_batch_count{0};
    std::size_t remote_home_batch_count{0};
    std::size_t local_home_item_count{0};
    std::size_t remote_home_item_count{0};
    absl::Duration local_home_rpc_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_rpc_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_rpc_max_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_rpc_max_elapsed{absl::ZeroDuration()};
  } stats;

  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>> remote_channels;
  remote_channels.reserve(remote_daemon_ids.size());
  const auto get_or_create_remote_channel =
      [&](std::string_view daemon_id) -> absl::StatusOr<std::shared_ptr<grpc::Channel>> {
    const auto cached_it = remote_channels.find(std::string(daemon_id));
    if (cached_it != remote_channels.end()) {
      return cached_it->second;
    }
    auto address_or = d_.worker_directory_cache.resolve_daemon_address(
        daemon_id, absl::Now(), absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    if (!address_or.ok()) {
      return address_or.status();
    }
    auto channel = create_inter_daemon_channel(*address_or, d_.inter_daemon_channel_credentials);
    remote_channels.emplace(std::string(daemon_id), channel);
    return channel;
  };

  const auto make_home_batch_exists_result = [&](PendingHomeBatchExists request,
                                                 grpc::Status status,
                                                 std::string_view message = "") -> HomeBatchExistsRpcResult {
    if (!status.ok() && !message.empty()) {
      status = grpc::Status(status.error_code(), std::string(message));
    }
    return HomeBatchExistsRpcResult{
        .request = std::move(request),
        .status = std::move(status),
    };
  };

  std::vector<PendingHomeBatchExists> pending_batches;
  pending_batches.reserve(shard_requests.size());
  for (const auto& [shard_id, batch] : shard_requests) {
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "routing lease unavailable"
          : route_it->second.message;
      for (size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        *resp.mutable_outcomes(batch.outcome_indices[idx]) =
            make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    const std::string holder_daemon_id = route_it->second.holder_daemon_id;
    const bool remote_home = holder_daemon_id != local_daemon_id;
    if (remote_home) {
      ++stats.remote_home_batch_count;
      stats.remote_home_item_count += batch.artifact_ids.size();
    } else {
      ++stats.local_home_batch_count;
      stats.local_home_item_count += batch.artifact_ids.size();
    }

    pending_batches.push_back(
        PendingHomeBatchExists{
            .shard_id = shard_id,
            .batch = &batch,
            .holder_daemon_id = holder_daemon_id,
            .lease_generation = route_it->second.lease_generation,
            .remote_home = remote_home,
            .attempt = 0,
        });
  }

  const auto dispatch_home_batch_exists_wave =
      [&](const std::vector<PendingHomeBatchExists>& wave) -> std::vector<HomeBatchExistsRpcResult> {
    std::vector<HomeBatchExistsRpcResult> completed_results;
    completed_results.reserve(wave.size());
    std::vector<ScheduledHomeBatchExists> scheduled;
    scheduled.reserve(wave.size());

    for (const auto& pending : wave) {
      std::shared_ptr<grpc::Channel> remote_channel;
      if (pending.remote_home) {
        auto channel_or = get_or_create_remote_channel(pending.holder_daemon_id);
        if (!channel_or.ok()) {
          completed_results.push_back(make_home_batch_exists_result(
              pending,
              grpc::Status(StatusCode::UNAVAILABLE, "home daemon address unavailable"),
              "home daemon address unavailable"));
          continue;
        }
        remote_channel = *channel_or;
      }

      scheduled.push_back(
          ScheduledHomeBatchExists{
              .request = pending,
              .future = folly::via(
                            d_.async_runtime.blocking_executor(),
                            [this, pending, remote_channel = std::move(remote_channel), allow_high_card_attrs]() mutable
                                -> HomeBatchExistsRpcResult {
                              HomeBatchExistsRpcResult result;
                              result.request = pending;

                              v2::HomeBatchExistsRequest home_req;
                              home_req.mutable_fence()->set_shard_id(pending.shard_id);
                              home_req.mutable_fence()->set_lease_generation(pending.lease_generation);
                              home_req.mutable_fence()->set_holder_daemon_id(pending.holder_daemon_id);
                              home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
                              for (const auto& artifact_id : pending.batch->artifact_ids) {
                                home_req.add_artifact_ids(artifact_id);
                              }

                              const absl::Time home_rpc_started_at = absl::Now();
                              if (!pending.remote_home) {
                                grpc::ServerContext home_ctx;
                                RpcContext home_rctx{"HomeBatchExists", home_ctx, allow_high_card_attrs};
                                result.status = home_batch_exists(home_rctx, home_req, result.home_resp);
                              } else {
                                auto stub = v2::StoreDaemonService::NewStub(remote_channel);
                                grpc::ClientContext client_ctx;
                                client_ctx.set_deadline(
                                    std::chrono::system_clock::now() + inter_daemon_home_rpc_timeout(options_));
                                result.status = stub->HomeBatchExists(&client_ctx, home_req, &result.home_resp);
                              }
                              result.rpc_elapsed = absl::Now() - home_rpc_started_at;
                              return result;
                            })
                            .semi(),
          });
    }

    for (auto& pending : scheduled) {
      try {
        completed_results.push_back(std::move(pending.future).get());
      } catch (const std::exception& ex) {
        completed_results.push_back(
            make_home_batch_exists_result(pending.request, grpc::Status(StatusCode::INTERNAL, ex.what()), ex.what()));
      } catch (...) {
        completed_results.push_back(make_home_batch_exists_result(
            pending.request,
            grpc::Status(StatusCode::INTERNAL, "HomeBatchExists fanout task failed"),
            "HomeBatchExists fanout task failed"));
      }
    }
    return completed_results;
  };

  std::function<void(std::vector<HomeBatchExistsRpcResult>, bool)> process_home_batch_exists_results;
  process_home_batch_exists_results = [&](std::vector<HomeBatchExistsRpcResult> results, bool allow_retry) {
    std::vector<PendingHomeBatchExists> retry_batches;
    retry_batches.reserve(results.size());

    for (auto& result : results) {
      if (result.request.remote_home) {
        stats.remote_home_rpc_elapsed += result.rpc_elapsed;
        stats.remote_home_rpc_max_elapsed = std::max(stats.remote_home_rpc_max_elapsed, result.rpc_elapsed);
      } else {
        stats.local_home_rpc_elapsed += result.rpc_elapsed;
        stats.local_home_rpc_max_elapsed = std::max(stats.local_home_rpc_max_elapsed, result.rpc_elapsed);
      }

      VLOG(2) << "byte_artifact.batch_exists_home_rpc_result"
              << " shard_id=" << result.request.shard_id << " remote_home=" << result.request.remote_home
              << " holder_daemon_id=" << result.request.holder_daemon_id
              << " requested_artifacts=" << result.request.batch->artifact_ids.size()
              << " attempt=" << result.request.attempt << " rpc_ms=" << absl::ToDoubleMilliseconds(result.rpc_elapsed)
              << " status_ok=" << result.status.ok() << " status_code=" << result.status.error_code()
              << " status_message=" << result.status.error_message();

      if (!result.status.ok()) {
        for (std::size_t idx = 0; idx < result.request.batch->artifact_ids.size(); ++idx) {
          *resp.mutable_outcomes(result.request.batch->outcome_indices[idx]) = make_outcome(
              result.request.batch->artifact_ids[idx],
              v2::BATCH_ITEM_STATUS_UNAVAILABLE,
              result.status.error_message());
        }
        continue;
      }

      bool needs_redirect_retry = false;
      if (result.home_resp.has_redirect() && result.home_resp.redirect().shard_id() == result.request.shard_id &&
          result.home_resp.redirect().lease_generation() != 0 &&
          !result.home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& outcome : result.home_resp.outcomes()) {
          if (outcome.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && allow_retry && result.request.attempt == 0) {
        const auto refreshed =
            d_.route_resolver.refresh_route_from_redirect(result.request.shard_id, result.home_resp.redirect(), now);
        if (!refreshed.ok) {
          for (std::size_t idx = 0; idx < result.request.batch->artifact_ids.size(); ++idx) {
            *resp.mutable_outcomes(result.request.batch->outcome_indices[idx]) = make_outcome(
                result.request.batch->artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          continue;
        }
        retry_batches.push_back(
            PendingHomeBatchExists{
                .shard_id = result.request.shard_id,
                .batch = result.request.batch,
                .holder_daemon_id = refreshed.holder_daemon_id,
                .lease_generation = refreshed.lease_generation,
                .remote_home = refreshed.holder_daemon_id != local_daemon_id,
                .attempt = 1,
            });
        continue;
      }

      absl::flat_hash_map<std::string, int> index_by_artifact;
      index_by_artifact.reserve(result.request.batch->artifact_ids.size());
      for (size_t idx = 0; idx < result.request.batch->artifact_ids.size(); ++idx) {
        index_by_artifact.emplace(result.request.batch->artifact_ids[idx], static_cast<int>(idx));
      }
      for (const auto& item : result.home_resp.outcomes()) {
        const auto idx_it = index_by_artifact.find(item.artifact_id());
        if (idx_it == index_by_artifact.end()) {
          continue;
        }
        v2::BatchItemStatus status = item.status();
        if (status == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
          status = v2::BATCH_ITEM_STATUS_UNAVAILABLE;
        }
        *resp.mutable_outcomes(result.request.batch->outcome_indices[static_cast<size_t>(idx_it->second)]) =
            make_outcome(item.artifact_id(), status, item.message());
      }
    }

    if (!retry_batches.empty()) {
      process_home_batch_exists_results(dispatch_home_batch_exists_wave(retry_batches), false);
    }
  };

  process_home_batch_exists_results(dispatch_home_batch_exists_wave(pending_batches), true);

  VLOG(2) << "byte_artifact.batch_exists_summary"
          << " selections=" << req.selections_size() << " shard_count=" << shard_requests.size()
          << " first_artifact_id=" << first_artifact_id << " local_home_batches=" << stats.local_home_batch_count
          << " remote_home_batches=" << stats.remote_home_batch_count
          << " local_home_items=" << stats.local_home_item_count
          << " remote_home_items=" << stats.remote_home_item_count
          << " local_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.local_home_rpc_elapsed)
          << " remote_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_elapsed)
          << " local_home_rpc_max_ms=" << absl::ToDoubleMilliseconds(stats.local_home_rpc_max_elapsed)
          << " remote_home_rpc_max_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_max_elapsed)
          << " route_resolve_ms=" << absl::ToDoubleMilliseconds(route_resolve_elapsed)
          << " worker_directory_warm_ms=" << absl::ToDoubleMilliseconds(worker_directory_warm_elapsed)
          << " total_wall_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at)
          << " parallel_active=" << (pending_batches.size() > 1);

  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_get_into_region(
    RpcContext& rctx,
    const v2::BatchGetIntoRegionRequest& req,
    v2::BatchGetIntoRegionResponse& resp) {
  const absl::Time total_started_at = absl::Now();
  const absl::Time ensure_local_peer_started_at = total_started_at;
  auto local_peer_status =
      d_.external_target_access_service.ensure_local_region_peer(rctx.server_context().peer(), "BatchGetIntoRegion");
  const absl::Duration ensure_local_peer_elapsed = absl::Now() - ensure_local_peer_started_at;
  if (!local_peer_status.ok()) {
    return to_grpc_status(local_peer_status);
  }
  const absl::Time now = absl::Now();
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();
  const auto requested_slot_tokens = collect_batch_item_slot_tokens(req.target_layout());
  const absl::Time selection_build_started_at = absl::Now();

  struct ShardRequest {
    std::vector<std::string> artifact_ids;
    std::vector<int> outcome_indices;
  };

  absl::flat_hash_map<std::uint64_t, ShardRequest> shard_requests;
  shard_requests.reserve(static_cast<std::size_t>(req.selections_size()));
  absl::flat_hash_map<std::string, std::uint64_t> target_layout_lengths;
  target_layout_lengths.reserve(static_cast<std::size_t>(req.selections_size()));

  for (int i = 0; i < req.selections_size(); ++i) {
    const auto& selection = req.selections(i);
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(selection.artifact_id());

    const auto selection_st = validate_batch_selection(selection);
    if (!selection_st.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(selection_st.message()));
      continue;
    }
    auto shard_or =
        byte_artifact_runtime().shard_id_for_artifact(selection.artifact_id(), options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome = make_outcome(
          selection.artifact_id(), v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }
    auto& entry = shard_requests[*shard_or];
    entry.artifact_ids.push_back(selection.artifact_id());
    entry.outcome_indices.push_back(i);
    target_layout_lengths.emplace(selection.artifact_id(), /*wildcard=*/0);
  }
  const absl::Duration selection_build_elapsed = absl::Now() - selection_build_started_at;

  const absl::Time validate_target_layout_started_at = absl::Now();
  auto target_layout_or = d_.external_target_access_service.validate_local_source_layout(
      rctx.server_context().peer(),
      "BatchGetIntoRegion",
      req.target_layout(),
      req.pid(),
      req.device_uuid(),
      target_layout_lengths);
  if (!target_layout_or.ok()) {
    return to_grpc_status(target_layout_or.status());
  }
  const absl::Duration validate_target_layout_elapsed = absl::Now() - validate_target_layout_started_at;
  auto target_layout = std::move(target_layout_or->layout);

  struct BatchGetInstrumentation {
    std::size_t local_home_batch_count{0};
    std::size_t remote_home_batch_count{0};
    std::size_t local_home_item_count{0};
    std::size_t remote_home_item_count{0};
    std::size_t local_source_item_count{0};
    std::size_t remote_source_item_count{0};
    std::size_t local_batch_transport_item_count{0};
    std::size_t remote_batch_transport_item_count{0};
    std::size_t local_batch_transport_per_item_materialize_calls{0};
    std::size_t remote_batch_transport_per_item_materialize_calls{0};
    std::size_t local_batch_transport_composite_materialize_calls{0};
    std::size_t remote_batch_transport_composite_materialize_calls{0};
    std::size_t remote_batch_transport_count{0};
    std::size_t remote_batch_transport_communicator_count{0};
    std::size_t remote_batch_transport_grpc_count{0};
    std::size_t remote_payload_ref_item_count{0};
    std::uint64_t local_source_bytes{0};
    std::uint64_t remote_source_bytes{0};
    std::uint64_t remote_batch_transport_bytes{0};
    absl::Duration local_home_rpc_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_rpc_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_rpc_max_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_rpc_max_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_queue_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_queue_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_queue_max_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_queue_max_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_task_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_task_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_task_max_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_task_max_elapsed{absl::ZeroDuration()};
    absl::Duration local_source_apply_elapsed{absl::ZeroDuration()};
    absl::Duration remote_source_apply_elapsed{absl::ZeroDuration()};
  } stats;

  struct BatchTransportApplyStats {
    std::uint64_t shard_id{0};
    std::string transport_id;
    std::string kind;
    std::string read_mode;
    std::string materialize_mode{"per_item"};
    bool remote{false};
    bool batched_direct_write{false};
    std::uint64_t payload_bytes{0};
    std::uint64_t staged_bytes{0};
    std::uint64_t gpu_write_bytes{0};
    std::uint32_t source_count{0};
    std::uint32_t mapping_segments{0};
    std::size_t items_applied{0};
    std::size_t materialize_calls{0};
    std::size_t per_item_materialize_calls{0};
    std::size_t composite_materialize_calls{0};
    absl::Duration resolve_elapsed{absl::ZeroDuration()};
    absl::Duration mirror_elapsed{absl::ZeroDuration()};
    absl::Duration source_stage_copy_elapsed{absl::ZeroDuration()};
    absl::Duration gpu_write_elapsed{absl::ZeroDuration()};
    absl::Duration lowering_elapsed{absl::ZeroDuration()};
    absl::Duration apply_elapsed{absl::ZeroDuration()};
  };

  struct BatchGetInstrumentationDelta {
    std::size_t local_source_item_count{0};
    std::size_t remote_source_item_count{0};
    std::size_t local_batch_transport_item_count{0};
    std::size_t remote_batch_transport_item_count{0};
    std::size_t local_batch_transport_per_item_materialize_calls{0};
    std::size_t remote_batch_transport_per_item_materialize_calls{0};
    std::size_t local_batch_transport_composite_materialize_calls{0};
    std::size_t remote_batch_transport_composite_materialize_calls{0};
    std::size_t remote_batch_transport_count{0};
    std::size_t remote_batch_transport_communicator_count{0};
    std::size_t remote_batch_transport_grpc_count{0};
    std::size_t remote_payload_ref_item_count{0};
    std::uint64_t local_source_bytes{0};
    std::uint64_t remote_source_bytes{0};
    std::uint64_t remote_batch_transport_bytes{0};
    absl::Duration local_source_apply_elapsed{absl::ZeroDuration()};
    absl::Duration remote_source_apply_elapsed{absl::ZeroDuration()};
  };

  struct ApplyOutcomeUpdate {
    int outcome_index{0};
    v2::BatchItemOutcome outcome;
  };

  struct TransportApplyLogRecord {
    BatchTransportApplyStats stats;
    std::size_t manifest_items{0};
    std::uint64_t manifest_bytes{0};
  };

  struct ApplyWorkUnitResult {
    std::vector<ApplyOutcomeUpdate> outcome_updates;
    BatchGetInstrumentationDelta stats_delta;
    std::vector<TransportApplyLogRecord> transport_logs;
  };

  struct ApplyItemRef {
    int outcome_index{0};
    std::string artifact_id;
    const v2::HomeBatchGetItem* item{nullptr};
  };

  struct ApplyWorkUnit {
    std::uint64_t shard_id{0};
    std::string transport_id;
    const v2::BatchPayloadTransport* batch_transport{nullptr};
    std::vector<ApplyItemRef> items;
  };

  struct ScheduledApplyWorkUnit {
    ApplyWorkUnit work_unit;
    folly::SemiFuture<ApplyWorkUnitResult> future;
  };

  std::vector<TransportApplyLogRecord> transport_apply_logs;
  const std::string_view operation_id =
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view("");
  const std::string operation_id_owned(operation_id);
  const std::string first_artifact_id = req.selections_size() == 0 ? std::string() : req.selections(0).artifact_id();
  const bool allow_high_card_attrs = rctx.allow_high_card_attrs();
  const bool host_target_layout = target_layout.device_id() < 0;
  absl::Duration route_resolve_elapsed{absl::ZeroDuration()};
  absl::Duration worker_directory_warm_elapsed{absl::ZeroDuration()};
  absl::Duration remote_channel_resolve_elapsed{absl::ZeroDuration()};
  absl::Duration remote_channel_resolve_max_elapsed{absl::ZeroDuration()};
  std::size_t remote_channel_resolve_count{0};
  absl::Duration home_dispatch_elapsed{absl::ZeroDuration()};
  absl::Duration home_process_elapsed{absl::ZeroDuration()};
  absl::Duration build_apply_work_elapsed{absl::ZeroDuration()};
  const auto log_transport_apply_stats = [&]() {
    for (const auto& record : transport_apply_logs) {
      std::ostringstream log;
      log << "byte_artifact.batch_get_into_region_transport_apply_summary"
          << " operation_id=" << operation_id << " shard_id=" << record.stats.shard_id
          << " transport_id=" << record.stats.transport_id << " kind=" << record.stats.kind
          << " read_mode=" << record.stats.read_mode << " materialize_mode=" << record.stats.materialize_mode
          << " batched_direct_write=" << record.stats.batched_direct_write << " remote=" << record.stats.remote
          << " manifest_items=" << record.manifest_items << " manifest_bytes=" << record.manifest_bytes
          << " payload_bytes=" << record.stats.payload_bytes << " staged_bytes=" << record.stats.staged_bytes
          << " gpu_write_bytes=" << record.stats.gpu_write_bytes << " source_count=" << record.stats.source_count
          << " mapping_segments=" << record.stats.mapping_segments << " items_applied=" << record.stats.items_applied
          << " materialize_calls=" << record.stats.materialize_calls
          << " per_item_materialize_calls=" << record.stats.per_item_materialize_calls
          << " composite_materialize_calls=" << record.stats.composite_materialize_calls
          << " resolve_ms=" << absl::ToDoubleMilliseconds(record.stats.resolve_elapsed)
          << " mirror_ms=" << absl::ToDoubleMilliseconds(record.stats.mirror_elapsed)
          << " source_to_pinned_ms=" << absl::ToDoubleMilliseconds(record.stats.source_stage_copy_elapsed)
          << " pinned_to_gpu_ms=" << absl::ToDoubleMilliseconds(record.stats.gpu_write_elapsed)
          << " lowering_ms=" << absl::ToDoubleMilliseconds(record.stats.lowering_elapsed)
          << " apply_ms=" << absl::ToDoubleMilliseconds(record.stats.apply_elapsed);
      if (!operation_id.empty() && record.stats.remote) {
        LOG(INFO) << log.str();
      } else {
        VLOG(2) << log.str();
      }
    }
  };
  const auto merge_stats_delta = [&](const BatchGetInstrumentationDelta& delta) {
    stats.local_source_item_count += delta.local_source_item_count;
    stats.remote_source_item_count += delta.remote_source_item_count;
    stats.local_batch_transport_item_count += delta.local_batch_transport_item_count;
    stats.remote_batch_transport_item_count += delta.remote_batch_transport_item_count;
    stats.local_batch_transport_per_item_materialize_calls += delta.local_batch_transport_per_item_materialize_calls;
    stats.remote_batch_transport_per_item_materialize_calls += delta.remote_batch_transport_per_item_materialize_calls;
    stats.local_batch_transport_composite_materialize_calls += delta.local_batch_transport_composite_materialize_calls;
    stats.remote_batch_transport_composite_materialize_calls +=
        delta.remote_batch_transport_composite_materialize_calls;
    stats.remote_batch_transport_count += delta.remote_batch_transport_count;
    stats.remote_batch_transport_communicator_count += delta.remote_batch_transport_communicator_count;
    stats.remote_batch_transport_grpc_count += delta.remote_batch_transport_grpc_count;
    stats.remote_payload_ref_item_count += delta.remote_payload_ref_item_count;
    stats.local_source_bytes += delta.local_source_bytes;
    stats.remote_source_bytes += delta.remote_source_bytes;
    stats.remote_batch_transport_bytes += delta.remote_batch_transport_bytes;
    stats.local_source_apply_elapsed += delta.local_source_apply_elapsed;
    stats.remote_source_apply_elapsed += delta.remote_source_apply_elapsed;
  };
  const auto merge_apply_work_unit_result = [&](ApplyWorkUnitResult result) {
    for (auto& update : result.outcome_updates) {
      *resp.mutable_outcomes(update.outcome_index) = std::move(update.outcome);
    }
    merge_stats_delta(result.stats_delta);
    for (auto& log_record : result.transport_logs) {
      transport_apply_logs.push_back(std::move(log_record));
    }
  };
  const auto run_apply_work_unit = [&](const ApplyWorkUnit& work_unit) -> ApplyWorkUnitResult {
    ApplyWorkUnitResult result;
    absl::flat_hash_map<std::string, PayloadTransportBroker::ResolvedBatchPayload> resolved_batch_payloads;
    absl::flat_hash_map<std::string, PayloadTransportBroker::BatchPayloadSource> resolved_batch_sources;
    absl::flat_hash_map<std::string, std::shared_ptr<const std::string>> mirrored_remote_batch_payloads;
    absl::flat_hash_map<const v2::BatchPayloadTransport*, BatchTransportApplyStats> transport_apply_stats;
    absl::flat_hash_set<std::string> opened_remote_batch_transports;
    absl::flat_hash_map<std::string, const v2::BatchPayloadTransport*> batch_transports_by_id;
    if (work_unit.batch_transport != nullptr) {
      batch_transports_by_id.emplace(work_unit.transport_id, work_unit.batch_transport);
    }
    const auto finalize_transport_logs =
        [&](const absl::flat_hash_map<const v2::BatchPayloadTransport*, BatchTransportApplyStats>& local_stats) {
          std::vector<TransportApplyLogRecord> logs;
          logs.reserve(local_stats.size());
          for (const auto& [transport, transport_stats] : local_stats) {
            if (transport == nullptr) {
              continue;
            }
            logs.push_back(
                TransportApplyLogRecord{
                    .stats = transport_stats,
                    .manifest_items = static_cast<std::size_t>(transport->manifest().entries_size()),
                    .manifest_bytes = transport->manifest().total_size(),
                });
          }
          return logs;
        };
    const auto try_apply_transport_composite = [&]() -> bool {
      if (work_unit.batch_transport == nullptr || !host_target_layout || work_unit.items.size() <= 1) {
        return false;
      }
      const auto* transport = work_unit.batch_transport;
      if (!transport->has_communicator_source()) {
        return false;
      }

      const std::string transport_id =
          work_unit.transport_id.empty() ? transport->transport_id() : work_unit.transport_id;
      auto resolved_it = resolved_batch_sources.find(transport_id);
      absl::Duration source_resolve_elapsed{absl::ZeroDuration()};
      if (resolved_it == resolved_batch_sources.end()) {
        const absl::Time resolve_started_at = absl::Now();
        auto resolved_or = d_.payload_transport_broker.open_batch_payload_communicator_source(
            d_.worker_directory_cache,
            now,
            absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
            local_daemon_id,
            *transport,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
            operation_id_owned);
        if (!resolved_or.ok()) {
          for (const auto& item_ref : work_unit.items) {
            result.outcome_updates.push_back(
                ApplyOutcomeUpdate{
                    .outcome_index = item_ref.outcome_index,
                    .outcome = make_outcome(
                        item_ref.artifact_id,
                        batch_item_status_from_absl_status(resolved_or.status()),
                        std::string(resolved_or.status().message())),
                });
          }
          return true;
        }
        source_resolve_elapsed += absl::Now() - resolve_started_at;
        resolved_it = resolved_batch_sources.emplace(transport_id, std::move(*resolved_or)).first;
        if (opened_remote_batch_transports.emplace(transport_id).second && resolved_it->second.remote) {
          ++result.stats_delta.remote_batch_transport_count;
          ++result.stats_delta.remote_batch_transport_communicator_count;
          result.stats_delta.remote_batch_transport_bytes += transport->manifest().total_size();
          LOG(INFO) << "byte_artifact.batch_get_into_region_transport_open"
                    << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                    << " transport_id=" << transport_id << " kind=communicator_source"
                    << " remote=" << resolved_it->second.remote
                    << " item_count=" << transport->manifest().entries_size()
                    << " payload_bytes=" << transport->manifest().total_size()
                    << " resolve_ms=" << absl::ToDoubleMilliseconds(source_resolve_elapsed);
        }
      }
      if (!resolved_it->second.remote || !resolved_it->second.source->supports_batched_direct_write_at()) {
        if (resolved_it->second.remote) {
          VLOG(2) << "byte_artifact.batch_get_into_region_transport_composite_skip"
                  << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                  << " transport_id=" << transport_id << " reason=batched_direct_write_unsupported";
        }
        return false;
      }

      store::loading::IntoTargetLayout composite_target_layout;
      composite_target_layout.storages.reserve(work_unit.items.size());
      store::loader::ByteRangeMap composite_mapping;
      composite_mapping.num_sources = 1;
      composite_mapping.segments.reserve(work_unit.items.size());
      std::uint64_t composite_cursor = 0;
      std::uint64_t payload_bytes = 0;
      for (const auto& item_ref : work_unit.items) {
        const v2::HomeBatchGetItem* item = item_ref.item;
        if (item == nullptr || item->status() != v2::BATCH_ITEM_STATUS_OK || !item->has_batch_payload_slice() ||
            item->batch_payload_slice().transport_id() != transport_id) {
          return false;
        }
        const auto* manifest_entry = find_batch_payload_entry(
            transport->manifest(),
            item_ref.artifact_id,
            item->batch_payload_slice().offset(),
            item->batch_payload_slice().length());
        if (manifest_entry == nullptr) {
          return false;
        }
        auto item_target_layout_or = target_layout.build_item_target_layout(item_ref.artifact_id);
        if (!item_target_layout_or.ok()) {
          return false;
        }
        if (item->batch_payload_slice().length() != item_target_layout_or->total_size) {
          return false;
        }
        if (composite_cursor > std::numeric_limits<std::uint64_t>::max() - item_target_layout_or->total_size) {
          return false;
        }
        if (payload_bytes > std::numeric_limits<std::uint64_t>::max() - item->batch_payload_slice().length()) {
          return false;
        }
        composite_mapping.segments.push_back(
            store::loader::ByteRangeSegment{
                .kind = store::loader::ByteRangeSegment::Kind::kData,
                .dst_offset = composite_cursor,
                .length = item->batch_payload_slice().length(),
                .src_offset = manifest_entry->offset(),
                .source_index = 0,
            });
        composite_cursor += item_target_layout_or->total_size;
        payload_bytes += item->batch_payload_slice().length();
        for (const auto& storage : item_target_layout_or->storages) {
          composite_target_layout.storages.push_back(storage);
        }
      }
      if (composite_target_layout.storages.empty() || composite_cursor == 0) {
        return false;
      }
      composite_target_layout.total_size = composite_cursor;
      composite_mapping.total_bytes = composite_cursor;

      auto& transport_stats = transport_apply_stats[transport];
      if (transport_stats.transport_id.empty()) {
        transport_stats.shard_id = work_unit.shard_id;
        transport_stats.transport_id = transport_id;
        transport_stats.kind = "communicator_source";
        transport_stats.remote = true;
        transport_stats.payload_bytes = transport->manifest().total_size();
      }
      transport_stats.resolve_elapsed += source_resolve_elapsed;
      transport_stats.read_mode = "batched_direct_write";
      transport_stats.materialize_mode = "single_source_composite";
      transport_stats.batched_direct_write = true;
      transport_stats.source_count = 1;
      transport_stats.mapping_segments = static_cast<std::uint32_t>(composite_mapping.segments.size());
      transport_stats.materialize_calls += 1;
      transport_stats.composite_materialize_calls += 1;
      result.stats_delta.remote_batch_transport_item_count += work_unit.items.size();
      result.stats_delta.remote_batch_transport_composite_materialize_calls += 1;

      LOG(INFO) << "byte_artifact.batch_get_into_region_transport_materialize_mode"
                << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                << " transport_id=" << transport_id << " kind=communicator_source"
                << " remote=true"
                << " materialize_mode=single_source_composite"
                << " batched_direct_write=true"
                << " source_count=1"
                << " mapping_segments=" << composite_mapping.segments.size() << " item_count=" << work_unit.items.size()
                << " payload_bytes=" << payload_bytes;

      const absl::Time apply_started_at = absl::Now();
      auto materialize_or = d_.engine.materialize_mapped_loader_into_target(
          store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1, .uuid = ""},
          composite_target_layout,
          std::make_unique<SeekableSourceLoader>(resolved_it->second.source, transport->manifest().total_size()),
          composite_mapping,
          build_lowering_hints(transport_id, operation_id_owned),
          store::loading::MaterializationSource::kP2P);
      const absl::Duration lowering_elapsed = absl::Now() - apply_started_at;
      transport_stats.lowering_elapsed += lowering_elapsed;
      transport_stats.apply_elapsed += lowering_elapsed;
      if (!materialize_or.ok()) {
        for (const auto& item_ref : work_unit.items) {
          result.outcome_updates.push_back(
              ApplyOutcomeUpdate{
                  .outcome_index = item_ref.outcome_index,
                  .outcome = make_outcome(
                      item_ref.artifact_id,
                      batch_item_status_from_absl_status(materialize_or.status()),
                      std::string(materialize_or.status().message())),
              });
        }
        return true;
      }
      if (materialize_or->debug_stats.has_value()) {
        const auto& debug_stats = *materialize_or->debug_stats;
        transport_stats.staged_bytes += debug_stats.produced_bytes == 0 ? payload_bytes : debug_stats.produced_bytes;
        transport_stats.gpu_write_bytes += debug_stats.gpu_write_bytes_total;
        transport_stats.source_stage_copy_elapsed += absl::Microseconds(debug_stats.source_read_at_us_total);
        transport_stats.gpu_write_elapsed += absl::Microseconds(debug_stats.gpu_write_wait_us_total);
      } else {
        transport_stats.staged_bytes += payload_bytes;
      }
      transport_stats.items_applied += work_unit.items.size();
      result.stats_delta.remote_source_item_count += work_unit.items.size();
      result.stats_delta.remote_source_bytes += payload_bytes;
      result.stats_delta.remote_source_apply_elapsed += lowering_elapsed;
      result.outcome_updates.reserve(work_unit.items.size());
      for (const auto& item_ref : work_unit.items) {
        result.outcome_updates.push_back(
            ApplyOutcomeUpdate{
                .outcome_index = item_ref.outcome_index,
                .outcome = make_outcome(item_ref.artifact_id, v2::BATCH_ITEM_STATUS_OK),
            });
      }
      return true;
    };
    const auto apply_item = [&](const ApplyItemRef& item_ref) {
      const absl::Time apply_started_at = absl::Now();
      auto push_outcome = [&](v2::BatchItemOutcome outcome) {
        result.outcome_updates.push_back(
            ApplyOutcomeUpdate{
                .outcome_index = item_ref.outcome_index,
                .outcome = std::move(outcome),
            });
      };
      const v2::HomeBatchGetItem* item = item_ref.item;
      if (item == nullptr) {
        push_outcome(make_outcome(item_ref.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing home item"));
        return;
      }
      if (item->status() != v2::BATCH_ITEM_STATUS_OK) {
        push_outcome(make_outcome(item_ref.artifact_id, item->status(), item->message()));
        return;
      }

      auto item_target_layout_or = target_layout.build_item_target_layout(item_ref.artifact_id);
      if (!item_target_layout_or.ok()) {
        push_outcome(make_outcome(
            item_ref.artifact_id,
            batch_item_status_from_absl_status(item_target_layout_or.status()),
            std::string(item_target_layout_or.status().message())));
        return;
      }

      std::unique_ptr<store::IArtifactLoader> loader;
      store::loading::MaterializationSource source_kind = store::loading::MaterializationSource::kLocalReplica;
      uint64_t payload_bytes = 0;
      bool source_is_remote = false;
      BatchTransportApplyStats* applied_transport_stats = nullptr;
      absl::Duration source_resolve_elapsed{absl::ZeroDuration()};
      if (item->has_batch_payload_slice()) {
        const auto transport_it = batch_transports_by_id.find(item->batch_payload_slice().transport_id());
        if (transport_it == batch_transports_by_id.end()) {
          push_outcome(make_outcome(
              item_ref.artifact_id, v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION, "batch transport is unavailable"));
          return;
        }
        const auto* manifest_entry = find_batch_payload_entry(
            transport_it->second->manifest(),
            item_ref.artifact_id,
            item->batch_payload_slice().offset(),
            item->batch_payload_slice().length());
        if (manifest_entry == nullptr) {
          push_outcome(make_outcome(
              item_ref.artifact_id,
              v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
              "batch transport slice does not match manifest"));
          return;
        }
        payload_bytes = item->batch_payload_slice().length();
        const std::string transport_id = item->batch_payload_slice().transport_id();
        if (transport_it->second->has_communicator_source()) {
          auto resolved_it = resolved_batch_sources.find(transport_id);
          if (resolved_it == resolved_batch_sources.end()) {
            const absl::Time resolve_started_at = absl::Now();
            auto resolved_or = d_.payload_transport_broker.open_batch_payload_communicator_source(
                d_.worker_directory_cache,
                now,
                absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
                local_daemon_id,
                *transport_it->second,
                tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
                operation_id_owned);
            if (!resolved_or.ok()) {
              push_outcome(make_outcome(
                  item_ref.artifact_id,
                  batch_item_status_from_absl_status(resolved_or.status()),
                  std::string(resolved_or.status().message())));
              return;
            }
            source_resolve_elapsed += absl::Now() - resolve_started_at;
            resolved_it = resolved_batch_sources.emplace(transport_id, std::move(*resolved_or)).first;
            if (opened_remote_batch_transports.emplace(transport_id).second && resolved_it->second.remote) {
              ++result.stats_delta.remote_batch_transport_count;
              ++result.stats_delta.remote_batch_transport_communicator_count;
              result.stats_delta.remote_batch_transport_bytes += transport_it->second->manifest().total_size();
              LOG(INFO) << "byte_artifact.batch_get_into_region_transport_open"
                        << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                        << " transport_id=" << transport_id << " kind=communicator_source"
                        << " remote=" << resolved_it->second.remote
                        << " item_count=" << transport_it->second->manifest().entries_size()
                        << " payload_bytes=" << transport_it->second->manifest().total_size()
                        << " resolve_ms=" << absl::ToDoubleMilliseconds(source_resolve_elapsed);
            }
          }
          source_is_remote = resolved_it->second.remote;
          auto& transport_stats = transport_apply_stats[transport_it->second];
          if (transport_stats.transport_id.empty()) {
            transport_stats.shard_id = work_unit.shard_id;
            transport_stats.transport_id = transport_id;
            transport_stats.kind = "communicator_source";
            transport_stats.remote = resolved_it->second.remote;
            transport_stats.payload_bytes = transport_it->second->manifest().total_size();
            transport_stats.source_count = 1;
          }
          transport_stats.resolve_elapsed += source_resolve_elapsed;
          applied_transport_stats = &transport_stats;
          source_kind = source_is_remote ? store::loading::MaterializationSource::kP2P
                                         : store::loading::MaterializationSource::kLocalReplica;
          if (source_is_remote) {
            if (resolved_it->second.source->supports_direct_write_at()) {
              if (transport_stats.read_mode.empty()) {
                transport_stats.read_mode = "direct_remote_slice";
                LOG(INFO) << "byte_artifact.batch_get_into_region_transport_read_mode"
                          << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                          << " transport_id=" << transport_id << " kind=communicator_source"
                          << " remote=true"
                          << " read_mode=direct_remote_slice"
                          << " payload_bytes=" << transport_it->second->manifest().total_size()
                          << " item_count=" << transport_it->second->manifest().entries_size() << " mirror_ms=0"
                          << " subsequent_item_slices_local=false";
              }
              loader = make_loader_from_source_slice(
                  resolved_it->second.source,
                  item->batch_payload_slice().offset(),
                  item->batch_payload_slice().length());
            } else {
              auto mirrored_it = mirrored_remote_batch_payloads.find(transport_id);
              if (mirrored_it == mirrored_remote_batch_payloads.end()) {
                const absl::Time mirror_started_at = absl::Now();
                auto mirrored_or = mirror_seekable_source_payload(
                    resolved_it->second.source, transport_it->second->manifest().total_size());
                if (!mirrored_or.ok()) {
                  push_outcome(make_outcome(
                      item_ref.artifact_id,
                      batch_item_status_from_absl_status(mirrored_or.status()),
                      std::string(mirrored_or.status().message())));
                  return;
                }
                const absl::Duration mirror_elapsed = absl::Now() - mirror_started_at;
                mirrored_it = mirrored_remote_batch_payloads.emplace(transport_id, std::move(*mirrored_or)).first;
                auto& transport_stats = transport_apply_stats[transport_it->second];
                if (transport_stats.transport_id.empty()) {
                  transport_stats.shard_id = work_unit.shard_id;
                  transport_stats.transport_id = transport_id;
                  transport_stats.kind = "communicator_source";
                  transport_stats.remote = true;
                  transport_stats.payload_bytes = transport_it->second->manifest().total_size();
                }
                transport_stats.read_mode = "full_pack_mirror";
                transport_stats.mirror_elapsed += mirror_elapsed;
                LOG(INFO) << "byte_artifact.batch_get_into_region_transport_mirror"
                          << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                          << " transport_id=" << transport_id << " kind=communicator_source"
                          << " remote=true"
                          << " read_mode=full_pack_mirror"
                          << " payload_bytes=" << transport_it->second->manifest().total_size()
                          << " item_count=" << transport_it->second->manifest().entries_size()
                          << " mirror_ms=" << absl::ToDoubleMilliseconds(mirror_elapsed)
                          << " subsequent_item_slices_local=true";
              }
              loader = make_loader_from_payload_slice(
                  mirrored_it->second, item->batch_payload_slice().offset(), item->batch_payload_slice().length());
            }
          } else {
            if (transport_stats.read_mode.empty()) {
              transport_stats.read_mode = "local_source_slice";
            }
            loader = make_loader_from_source_slice(
                resolved_it->second.source, item->batch_payload_slice().offset(), item->batch_payload_slice().length());
          }
        } else if (transport_it->second->has_grpc_chunk_ref()) {
          auto resolved_it = resolved_batch_payloads.find(transport_id);
          if (resolved_it == resolved_batch_payloads.end()) {
            const absl::Time resolve_started_at = absl::Now();
            auto resolved_or = d_.payload_transport_broker.fetch_batch_payload_ref(
                d_.worker_directory_cache,
                now,
                absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
                local_daemon_id,
                transport_it->second->grpc_chunk_ref().batch_payload_ref(),
                tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
                operation_id_owned);
            if (!resolved_or.ok()) {
              push_outcome(make_outcome(
                  item_ref.artifact_id,
                  batch_item_status_from_absl_status(resolved_or.status()),
                  std::string(resolved_or.status().message())));
              return;
            }
            source_resolve_elapsed += absl::Now() - resolve_started_at;
            resolved_it = resolved_batch_payloads.emplace(transport_id, std::move(*resolved_or)).first;
            if (opened_remote_batch_transports.emplace(transport_id).second && resolved_it->second.remote) {
              ++result.stats_delta.remote_batch_transport_count;
              ++result.stats_delta.remote_batch_transport_grpc_count;
              result.stats_delta.remote_batch_transport_bytes += transport_it->second->manifest().total_size();
              LOG(INFO) << "byte_artifact.batch_get_into_region_transport_open"
                        << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                        << " transport_id=" << transport_id << " kind=grpc_chunk_ref"
                        << " remote=" << resolved_it->second.remote
                        << " item_count=" << transport_it->second->manifest().entries_size()
                        << " payload_bytes=" << transport_it->second->manifest().total_size()
                        << " resolve_ms=" << absl::ToDoubleMilliseconds(source_resolve_elapsed);
            }
          }
          auto& transport_stats = transport_apply_stats[transport_it->second];
          if (transport_stats.transport_id.empty()) {
            transport_stats.shard_id = work_unit.shard_id;
            transport_stats.transport_id = transport_id;
            transport_stats.kind = "grpc_chunk_ref";
            transport_stats.remote = resolved_it->second.remote;
            transport_stats.payload_bytes = transport_it->second->manifest().total_size();
            transport_stats.source_count = 1;
          }
          transport_stats.resolve_elapsed += source_resolve_elapsed;
          applied_transport_stats = &transport_stats;
          if (transport_stats.read_mode.empty()) {
            transport_stats.read_mode =
                resolved_it->second.remote ? "grpc_chunk_full_pack_remote" : "grpc_chunk_full_pack_local";
          }
          if (!resolved_it->second.payload ||
              item->batch_payload_slice().offset() + item->batch_payload_slice().length() >
                  resolved_it->second.payload->size()) {
            push_outcome(make_outcome(
                item_ref.artifact_id,
                v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
                "batch transport payload is truncated"));
            return;
          }
          source_is_remote = resolved_it->second.remote;
          source_kind = source_is_remote ? store::loading::MaterializationSource::kP2P
                                         : store::loading::MaterializationSource::kLocalReplica;
          loader = make_loader_from_payload_slice(
              resolved_it->second.payload, item->batch_payload_slice().offset(), item->batch_payload_slice().length());
        } else {
          push_outcome(make_outcome(
              item_ref.artifact_id,
              v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
              "batch transport has no supported transport kind"));
          return;
        }
      }
      if (loader != nullptr) {
        // Batch transport already produced a loader.
      } else if (!item->inline_payload().empty()) {
        auto payload = std::make_shared<std::string>(item->inline_payload());
        auto payload_view = std::shared_ptr<const void>(payload, static_cast<const void*>(payload->data()));
        loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
            .data = std::move(payload_view),
            .size_bytes = payload->size(),
        });
        payload_bytes = payload->size();
      } else if (!item->payload_ref().empty()) {
        auto loader_or = d_.payload_transport_broker.open_payload_ref_loader(
            d_.worker_directory_cache,
            now,
            absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
            local_daemon_id,
            item->payload_ref(),
            item_ref.artifact_id,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_GET,
            operation_id_owned);
        if (!loader_or.ok()) {
          push_outcome(make_outcome(
              item_ref.artifact_id,
              batch_item_status_from_absl_status(loader_or.status()),
              std::string(loader_or.status().message())));
          return;
        }
        source_is_remote = loader_or->remote;
        source_kind = source_is_remote ? store::loading::MaterializationSource::kP2P
                                       : store::loading::MaterializationSource::kLocalReplica;
        payload_bytes = loader_or->metadata.payload_size;
        loader = std::move(loader_or->loader);
        if (source_is_remote) {
          ++result.stats_delta.remote_payload_ref_item_count;
        }
      } else {
        push_outcome(
            make_outcome(item_ref.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "home get returned no payload"));
        return;
      }

      if (payload_bytes != item_target_layout_or->total_size) {
        push_outcome(make_outcome(
            item_ref.artifact_id,
            v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
            "payload size does not match target layout length"));
        return;
      }

      const absl::Time lowering_started_at = absl::Now();
      if (host_target_layout) {
        if (applied_transport_stats != nullptr) {
          applied_transport_stats->materialize_calls += 1;
          applied_transport_stats->per_item_materialize_calls += 1;
          if (source_is_remote) {
            result.stats_delta.remote_batch_transport_item_count += 1;
            result.stats_delta.remote_batch_transport_per_item_materialize_calls += 1;
          } else {
            result.stats_delta.local_batch_transport_item_count += 1;
            result.stats_delta.local_batch_transport_per_item_materialize_calls += 1;
          }
        }
        auto materialize_or = d_.engine.materialize_mapped_loader_into_target(
            store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1, .uuid = ""},
            *item_target_layout_or,
            std::move(loader),
            store::loading::build_identity_byte_range_map(payload_bytes),
            build_lowering_hints(item_ref.artifact_id, operation_id_owned),
            source_kind);
        if (!materialize_or.ok()) {
          push_outcome(make_outcome(
              item_ref.artifact_id,
              batch_item_status_from_absl_status(materialize_or.status()),
              std::string(materialize_or.status().message())));
          return;
        }
        if (applied_transport_stats != nullptr) {
          if (materialize_or->debug_stats.has_value()) {
            const auto& debug_stats = *materialize_or->debug_stats;
            applied_transport_stats->staged_bytes +=
                debug_stats.produced_bytes == 0 ? payload_bytes : debug_stats.produced_bytes;
            applied_transport_stats->gpu_write_bytes += debug_stats.gpu_write_bytes_total;
            applied_transport_stats->source_stage_copy_elapsed +=
                absl::Microseconds(debug_stats.source_read_at_us_total);
            applied_transport_stats->gpu_write_elapsed += absl::Microseconds(debug_stats.gpu_write_wait_us_total);
          } else {
            applied_transport_stats->staged_bytes += payload_bytes;
          }
        }
      } else {
        if (applied_transport_stats != nullptr) {
          applied_transport_stats->materialize_calls += 1;
          applied_transport_stats->per_item_materialize_calls += 1;
          if (source_is_remote) {
            result.stats_delta.remote_batch_transport_item_count += 1;
            result.stats_delta.remote_batch_transport_per_item_materialize_calls += 1;
          } else {
            result.stats_delta.local_batch_transport_item_count += 1;
            result.stats_delta.local_batch_transport_per_item_materialize_calls += 1;
          }
        }
        const auto target_device = store::DeviceRegistry::instance().gpu_key(target_layout.device_id());
        auto lowering_or = build_into_target_lowering_plan(
            item_ref.artifact_id,
            target_device,
            *item_target_layout_or,
            std::move(loader),
            payload_bytes,
            source_kind,
            operation_id_owned);
        if (!lowering_or.ok()) {
          push_outcome(make_outcome(
              item_ref.artifact_id,
              batch_item_status_from_absl_status(lowering_or.status()),
              std::string(lowering_or.status().message())));
          return;
        }
        auto materialize_or = d_.engine.execute_artifact_lowering_plan(std::move(*lowering_or));
        if (!materialize_or.ok()) {
          push_outcome(make_outcome(
              item_ref.artifact_id,
              batch_item_status_from_absl_status(materialize_or.status()),
              std::string(materialize_or.status().message())));
          return;
        }
        if (applied_transport_stats != nullptr && materialize_or->into_target_result.has_value() &&
            materialize_or->into_target_result->debug_stats.has_value()) {
          const auto& debug_stats = *materialize_or->into_target_result->debug_stats;
          applied_transport_stats->staged_bytes += debug_stats.produced_bytes;
          applied_transport_stats->gpu_write_bytes += debug_stats.gpu_write_bytes_total;
          applied_transport_stats->source_stage_copy_elapsed += absl::Microseconds(debug_stats.source_read_at_us_total);
          applied_transport_stats->gpu_write_elapsed += absl::Microseconds(debug_stats.gpu_write_wait_us_total);
        }
      }
      push_outcome(make_outcome(item_ref.artifact_id, v2::BATCH_ITEM_STATUS_OK));
      const absl::Duration lowering_elapsed = absl::Now() - lowering_started_at;
      const absl::Duration apply_elapsed = absl::Now() - apply_started_at;
      if (applied_transport_stats != nullptr) {
        ++applied_transport_stats->items_applied;
        applied_transport_stats->lowering_elapsed += lowering_elapsed;
        applied_transport_stats->apply_elapsed += apply_elapsed;
      }
      if (source_is_remote) {
        ++result.stats_delta.remote_source_item_count;
        result.stats_delta.remote_source_bytes += payload_bytes;
        result.stats_delta.remote_source_apply_elapsed += apply_elapsed;
      } else {
        ++result.stats_delta.local_source_item_count;
        result.stats_delta.local_source_bytes += payload_bytes;
        result.stats_delta.local_source_apply_elapsed += apply_elapsed;
      }
      if (source_is_remote && apply_elapsed >= absl::Milliseconds(5)) {
        LOG(INFO) << "byte_artifact.batch_get_into_region_item_apply"
                  << " operation_id=" << operation_id << " shard_id=" << work_unit.shard_id
                  << " artifact_id=" << item_ref.artifact_id << " transport_id="
                  << (applied_transport_stats == nullptr ? "" : applied_transport_stats->transport_id)
                  << " transport_kind=" << (applied_transport_stats == nullptr ? "" : applied_transport_stats->kind)
                  << " payload_bytes=" << payload_bytes
                  << " source_resolve_ms=" << absl::ToDoubleMilliseconds(source_resolve_elapsed)
                  << " lowering_ms=" << absl::ToDoubleMilliseconds(lowering_elapsed)
                  << " apply_ms=" << absl::ToDoubleMilliseconds(apply_elapsed);
      }
    };
    result.outcome_updates.reserve(work_unit.items.size());
    if (try_apply_transport_composite()) {
      result.transport_logs = finalize_transport_logs(transport_apply_stats);
      return result;
    }
    for (const auto& item_ref : work_unit.items) {
      apply_item(item_ref);
    }
    result.transport_logs = finalize_transport_logs(transport_apply_stats);
    return result;
  };
  const auto can_parallelize_target_apply = [&]() {
    if (req.target_layout().offsets_size() <= 1) {
      return true;
    }
    struct TargetRange {
      std::string storage_id;
      std::uint64_t begin{0};
      std::uint64_t end{0};
      std::string artifact_id;
    };
    std::vector<TargetRange> ranges;
    ranges.reserve(static_cast<std::size_t>(req.target_layout().offsets_size()));
    for (const auto& offset : req.target_layout().offsets()) {
      ranges.push_back(
          TargetRange{
              .storage_id = offset.storage_id(),
              .begin = offset.storage_offset(),
              .end = offset.storage_offset() + offset.logical_length(),
              .artifact_id = offset.name(),
          });
    }
    std::sort(ranges.begin(), ranges.end(), [](const TargetRange& lhs, const TargetRange& rhs) {
      if (lhs.storage_id != rhs.storage_id) {
        return lhs.storage_id < rhs.storage_id;
      }
      if (lhs.begin != rhs.begin) {
        return lhs.begin < rhs.begin;
      }
      return lhs.end < rhs.end;
    });
    for (std::size_t idx = 1; idx < ranges.size(); ++idx) {
      const auto& prev = ranges[idx - 1];
      const auto& curr = ranges[idx];
      if (prev.storage_id == curr.storage_id && prev.end > curr.begin) {
        LOG(WARNING) << "byte_artifact.batch_get_into_region_parallel_apply_disabled"
                     << " operation_id=" << operation_id << " reason=overlapping_target_ranges"
                     << " storage_id=" << curr.storage_id << " prev_artifact_id=" << prev.artifact_id
                     << " curr_artifact_id=" << curr.artifact_id;
        return false;
      }
    }
    return true;
  };
  const auto build_apply_work_units =
      [&](std::uint64_t shard_id, const ShardRequest& batch, const v2::HomeBatchGetResponse& home_resp) {
        std::vector<ApplyWorkUnit> work_units;
        absl::flat_hash_map<std::string, const v2::HomeBatchGetItem*> by_id;
        by_id.reserve(home_resp.items_size());
        for (const auto& item : home_resp.items()) {
          by_id.emplace(item.artifact_id(), &item);
        }
        absl::flat_hash_map<std::string, const v2::BatchPayloadTransport*> batch_transports_by_id;
        batch_transports_by_id.reserve(home_resp.batch_transports_size());
        for (const auto& transport : home_resp.batch_transports()) {
          batch_transports_by_id.emplace(transport.transport_id(), &transport);
        }
        absl::flat_hash_map<std::string, std::size_t> transport_work_unit_indices;
        std::optional<std::size_t> residual_work_unit_index;
        const auto ensure_residual_work_unit = [&]() -> ApplyWorkUnit& {
          if (!residual_work_unit_index.has_value()) {
            residual_work_unit_index = work_units.size();
            work_units.push_back(
                ApplyWorkUnit{
                    .shard_id = shard_id,
                });
          }
          return work_units[*residual_work_unit_index];
        };
        for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
          const auto it = by_id.find(batch.artifact_ids[idx]);
          const v2::HomeBatchGetItem* item = (it == by_id.end()) ? nullptr : it->second;
          const v2::BatchPayloadTransport* batch_transport = nullptr;
          std::string transport_id;
          if (item != nullptr && item->status() == v2::BATCH_ITEM_STATUS_OK && item->has_batch_payload_slice()) {
            transport_id = item->batch_payload_slice().transport_id();
            const auto transport_it = batch_transports_by_id.find(transport_id);
            if (transport_it != batch_transports_by_id.end()) {
              batch_transport = transport_it->second;
            }
          }
          ApplyWorkUnit* work_unit = nullptr;
          if (batch_transport != nullptr) {
            auto [transport_it, inserted] = transport_work_unit_indices.emplace(transport_id, work_units.size());
            if (inserted) {
              work_units.push_back(
                  ApplyWorkUnit{
                      .shard_id = shard_id,
                      .transport_id = transport_id,
                      .batch_transport = batch_transport,
                  });
            }
            work_unit = &work_units[transport_it->second];
          } else {
            work_unit = &ensure_residual_work_unit();
          }
          work_unit->items.push_back(
              ApplyItemRef{
                  .outcome_index = batch.outcome_indices[idx],
                  .artifact_id = batch.artifact_ids[idx],
                  .item = item,
              });
        }
        return work_units;
      };
  const bool parallel_apply_enabled = can_parallelize_target_apply();
  absl::Duration apply_wall_elapsed = absl::ZeroDuration();
  const auto execute_apply_work_units = [&](std::vector<ApplyWorkUnit> work_units) {
    const absl::Time apply_started_at = absl::Now();
    if (work_units.empty()) {
      apply_wall_elapsed += absl::Now() - apply_started_at;
      return;
    }
    std::size_t batch_work_units = 0;
    std::size_t residual_work_units = 0;
    for (const auto& work_unit : work_units) {
      if (work_unit.batch_transport != nullptr) {
        ++batch_work_units;
      } else {
        ++residual_work_units;
      }
    }
    LOG(INFO) << "byte_artifact.batch_get_into_region_apply_plan"
              << " operation_id=" << operation_id << " work_units=" << work_units.size()
              << " batch_work_units=" << batch_work_units << " residual_work_units=" << residual_work_units
              << " apply_executor_threads=" << batch_get_apply_threads_
              << " parallel_enabled=" << parallel_apply_enabled
              << " parallel_active=" << (parallel_apply_enabled && work_units.size() > 1);
    if (!parallel_apply_enabled || work_units.size() <= 1) {
      for (const auto& work_unit : work_units) {
        merge_apply_work_unit_result(run_apply_work_unit(work_unit));
      }
      apply_wall_elapsed += absl::Now() - apply_started_at;
      return;
    }
    std::vector<ScheduledApplyWorkUnit> scheduled;
    scheduled.reserve(work_units.size());
    for (auto& work_unit : work_units) {
      scheduled.push_back(
          ScheduledApplyWorkUnit{
              .work_unit =
                  ApplyWorkUnit{
                      .shard_id = work_unit.shard_id,
                      .transport_id = work_unit.transport_id,
                      .batch_transport = work_unit.batch_transport,
                      .items = work_unit.items,
                  },
              .future = folly::via(
                            folly::getKeepAliveToken(*batch_get_apply_pool_),
                            [run_apply_work_unit, work_unit = std::move(work_unit)]() mutable -> ApplyWorkUnitResult {
                              return run_apply_work_unit(work_unit);
                            })
                            .semi(),
          });
    }
    for (auto& scheduled_work_unit : scheduled) {
      try {
        merge_apply_work_unit_result(std::move(scheduled_work_unit.future).get());
      } catch (const std::exception& ex) {
        for (const auto& item_ref : scheduled_work_unit.work_unit.items) {
          *resp.mutable_outcomes(item_ref.outcome_index) =
              make_outcome(item_ref.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, ex.what());
        }
      } catch (...) {
        for (const auto& item_ref : scheduled_work_unit.work_unit.items) {
          *resp.mutable_outcomes(item_ref.outcome_index) = make_outcome(
              item_ref.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "BatchGetIntoRegion apply task failed");
        }
      }
    }
    apply_wall_elapsed += absl::Now() - apply_started_at;
  };
  const auto log_apply_breakdown = [&]() {
    LOG(INFO) << "byte_artifact.batch_get_into_region_apply_breakdown"
              << " operation_id=" << operation_id
              << " local_batch_transport_items=" << stats.local_batch_transport_item_count
              << " remote_batch_transport_items=" << stats.remote_batch_transport_item_count
              << " local_batch_transport_per_item_materialize_calls="
              << stats.local_batch_transport_per_item_materialize_calls
              << " remote_batch_transport_per_item_materialize_calls="
              << stats.remote_batch_transport_per_item_materialize_calls
              << " local_batch_transport_composite_materialize_calls="
              << stats.local_batch_transport_composite_materialize_calls
              << " remote_batch_transport_composite_materialize_calls="
              << stats.remote_batch_transport_composite_materialize_calls
              << " local_source_items=" << stats.local_source_item_count
              << " remote_source_items=" << stats.remote_source_item_count
              << " apply_wall_ms=" << absl::ToDoubleMilliseconds(apply_wall_elapsed);
  };

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    absl::flat_hash_map<std::uint64_t, v2::HomeBatchGetResponse> completed_local_home_batches;
    completed_local_home_batches.reserve(shard_requests.size());
    for (const auto& [shard_id, batch] : shard_requests) {
      if (batch.artifact_ids.empty()) {
        continue;
      }
      ++stats.local_home_batch_count;
      stats.local_home_item_count += batch.artifact_ids.size();
      v2::HomeBatchGetRequest home_req;
      home_req.mutable_fence()->set_shard_id(shard_id);
      home_req.mutable_fence()->set_lease_generation(1);
      home_req.mutable_fence()->set_holder_daemon_id(local_daemon_id);
      home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
      if (req.has_operation_id()) {
        home_req.set_operation_id(req.operation_id());
      }
      home_req.set_requester_daemon_id(local_daemon_id);
      for (const auto& artifact_id : batch.artifact_ids) {
        home_req.add_artifact_ids(artifact_id);
      }
      v2::HomeBatchGetResponse home_resp;
      grpc::ServerContext home_ctx;
      RpcContext home_rctx{"HomeBatchGet", home_ctx, allow_high_card_attrs};
      const absl::Time home_rpc_started_at = absl::Now();
      const auto home_status = home_batch_get(home_rctx, home_req, home_resp);
      const absl::Duration home_rpc_elapsed = absl::Now() - home_rpc_started_at;
      stats.local_home_rpc_elapsed += home_rpc_elapsed;
      stats.local_home_rpc_max_elapsed = std::max(stats.local_home_rpc_max_elapsed, home_rpc_elapsed);
      if (!home_status.ok()) {
        for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
          *resp.mutable_outcomes(batch.outcome_indices[idx]) =
              make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_status.error_message());
        }
        continue;
      }
      const auto home_resp_shape = inspect_home_batch_get_response_shape(home_resp);
      LOG(INFO) << "byte_artifact.batch_get_into_region_home_response"
                << " operation_id=" << operation_id << " holder_daemon_id=" << local_daemon_id << " remote_home=false"
                << " shard_id=" << shard_id << " requested_artifacts=" << batch.artifact_ids.size()
                << " items=" << home_resp_shape.items << " ok_items=" << home_resp_shape.ok_items
                << " inline_items=" << home_resp_shape.inline_items
                << " payload_ref_items=" << home_resp_shape.payload_ref_items
                << " batch_slice_items=" << home_resp_shape.batch_slice_items
                << " transports=" << home_resp_shape.transports
                << " communicator_transports=" << home_resp_shape.communicator_transports
                << " grpc_chunk_transports=" << home_resp_shape.grpc_chunk_transports;
      completed_local_home_batches.emplace(shard_id, std::move(home_resp));
    }
    std::vector<ApplyWorkUnit> all_work_units;
    for (const auto& [shard_id, batch] : shard_requests) {
      const auto completed_it = completed_local_home_batches.find(shard_id);
      if (completed_it == completed_local_home_batches.end()) {
        continue;
      }
      auto shard_work_units = build_apply_work_units(shard_id, batch, completed_it->second);
      all_work_units.insert(
          all_work_units.end(),
          std::make_move_iterator(shard_work_units.begin()),
          std::make_move_iterator(shard_work_units.end()));
    }
    execute_apply_work_units(std::move(all_work_units));
    log_transport_apply_stats();
    log_apply_breakdown();
    attach_slot_tokens_to_outcomes(requested_slot_tokens, resp.mutable_outcomes());
    {
      std::ostringstream log;
      log << "byte_artifact.batch_get_into_region_summary"
          << " operation_id=" << (req.has_operation_id() ? req.operation_id() : "")
          << " first_artifact_id=" << first_artifact_id << " selections=" << req.selections_size()
          << " shard_count=" << shard_requests.size() << " local_home_batches=" << stats.local_home_batch_count
          << " remote_home_batches=" << stats.remote_home_batch_count
          << " local_home_items=" << stats.local_home_item_count
          << " remote_home_items=" << stats.remote_home_item_count
          << " local_source_items=" << stats.local_source_item_count
          << " remote_source_items=" << stats.remote_source_item_count
          << " local_batch_transport_items=" << stats.local_batch_transport_item_count
          << " remote_batch_transport_items=" << stats.remote_batch_transport_item_count
          << " local_batch_transport_per_item_materialize_calls="
          << stats.local_batch_transport_per_item_materialize_calls
          << " remote_batch_transport_per_item_materialize_calls="
          << stats.remote_batch_transport_per_item_materialize_calls
          << " local_batch_transport_composite_materialize_calls="
          << stats.local_batch_transport_composite_materialize_calls
          << " remote_batch_transport_composite_materialize_calls="
          << stats.remote_batch_transport_composite_materialize_calls
          << " remote_batch_transports=" << stats.remote_batch_transport_count
          << " remote_batch_transport_communicator=" << stats.remote_batch_transport_communicator_count
          << " remote_batch_transport_grpc=" << stats.remote_batch_transport_grpc_count
          << " remote_payload_ref_items=" << stats.remote_payload_ref_item_count
          << " local_source_bytes=" << stats.local_source_bytes << " remote_source_bytes=" << stats.remote_source_bytes
          << " remote_batch_transport_bytes=" << stats.remote_batch_transport_bytes
          << " local_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.local_home_rpc_elapsed)
          << " remote_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_elapsed)
          << " local_home_rpc_max_ms=" << absl::ToDoubleMilliseconds(stats.local_home_rpc_max_elapsed)
          << " remote_home_rpc_max_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_max_elapsed)
          << " local_source_apply_ms=" << absl::ToDoubleMilliseconds(stats.local_source_apply_elapsed)
          << " remote_source_apply_ms=" << absl::ToDoubleMilliseconds(stats.remote_source_apply_elapsed)
          << " apply_wall_ms=" << absl::ToDoubleMilliseconds(apply_wall_elapsed)
          << " total_wall_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at);
      if (req.has_operation_id() && !req.operation_id().empty()) {
        LOG(INFO) << log.str();
      } else {
        VLOG(2) << log.str();
      }
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_requests.size());
  for (const auto& [shard_id, /*batch*/ _] : shard_requests) {
    shard_ids.push_back(shard_id);
  }
  const absl::Time route_resolve_started_at = absl::Now();
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);
  route_resolve_elapsed = absl::Now() - route_resolve_started_at;

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (route.ok && route.holder_daemon_id != local_daemon_id) {
      remote_daemon_ids.push_back(route.holder_daemon_id);
    }
  }
  if (!remote_daemon_ids.empty()) {
    const absl::Time worker_directory_warm_started_at = absl::Now();
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    worker_directory_warm_elapsed += absl::Now() - worker_directory_warm_started_at;
  }

  struct PendingHomeBatchGet {
    std::uint64_t shard_id{0};
    const ShardRequest* batch{nullptr};
    std::string holder_daemon_id;
    std::uint64_t lease_generation{0};
    bool remote_home{false};
    int attempt{0};
  };

  struct HomeBatchGetRpcResult {
    PendingHomeBatchGet request;
    grpc::Status status;
    v2::HomeBatchGetResponse home_resp;
    absl::Duration queue_elapsed{absl::ZeroDuration()};
    absl::Duration rpc_elapsed{absl::ZeroDuration()};
    absl::Duration task_elapsed{absl::ZeroDuration()};
  };

  struct ScheduledHomeBatchGet {
    PendingHomeBatchGet request;
    folly::SemiFuture<HomeBatchGetRpcResult> future;
  };

  struct CompletedHomeBatchGet {
    std::string holder_daemon_id;
    bool remote_home{false};
    v2::HomeBatchGetResponse home_resp;
  };

  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>> remote_channels;
  remote_channels.reserve(remote_daemon_ids.size());
  const auto get_or_create_remote_channel =
      [&](std::string_view daemon_id) -> absl::StatusOr<std::shared_ptr<grpc::Channel>> {
    const auto cached_it = remote_channels.find(std::string(daemon_id));
    if (cached_it != remote_channels.end()) {
      return cached_it->second;
    }
    const absl::Time resolve_started_at = absl::Now();
    auto address_or = d_.worker_directory_cache.resolve_daemon_address(
        daemon_id, absl::Now(), absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    const absl::Duration resolve_elapsed = absl::Now() - resolve_started_at;
    remote_channel_resolve_elapsed += resolve_elapsed;
    remote_channel_resolve_max_elapsed = std::max(remote_channel_resolve_max_elapsed, resolve_elapsed);
    ++remote_channel_resolve_count;
    if (!address_or.ok()) {
      return address_or.status();
    }
    auto channel = create_inter_daemon_channel(*address_or, d_.inter_daemon_channel_credentials);
    remote_channels.emplace(std::string(daemon_id), channel);
    return channel;
  };

  const auto make_home_batch_get_result =
      [&](PendingHomeBatchGet request, grpc::Status status, std::string_view message = "") -> HomeBatchGetRpcResult {
    if (!status.ok() && !message.empty()) {
      status = grpc::Status(status.error_code(), std::string(message));
    }
    return HomeBatchGetRpcResult{
        .request = std::move(request),
        .status = std::move(status),
    };
  };

  std::vector<PendingHomeBatchGet> pending_batches;
  pending_batches.reserve(shard_requests.size());
  for (const auto& [shard_id, batch] : shard_requests) {
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "shard home route unavailable"
          : route_it->second.message;
      for (std::size_t idx = 0; idx < batch.artifact_ids.size(); ++idx) {
        *resp.mutable_outcomes(batch.outcome_indices[idx]) =
            make_outcome(batch.artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    const std::string holder_daemon_id = route_it->second.holder_daemon_id;
    const bool remote_home = holder_daemon_id != local_daemon_id;
    if (remote_home) {
      ++stats.remote_home_batch_count;
      stats.remote_home_item_count += batch.artifact_ids.size();
    } else {
      ++stats.local_home_batch_count;
      stats.local_home_item_count += batch.artifact_ids.size();
    }

    pending_batches.push_back(
        PendingHomeBatchGet{
            .shard_id = shard_id,
            .batch = &batch,
            .holder_daemon_id = holder_daemon_id,
            .lease_generation = route_it->second.lease_generation,
            .remote_home = remote_home,
            .attempt = 0,
        });
  }

  const auto dispatch_home_batch_get_wave =
      [&](const std::vector<PendingHomeBatchGet>& wave) -> std::vector<HomeBatchGetRpcResult> {
    const absl::Time dispatch_started_at = absl::Now();
    std::vector<HomeBatchGetRpcResult> completed_results;
    completed_results.reserve(wave.size());
    std::vector<ScheduledHomeBatchGet> scheduled;
    scheduled.reserve(wave.size());

    for (const auto& pending : wave) {
      std::shared_ptr<grpc::Channel> remote_channel;
      if (pending.remote_home) {
        auto channel_or = get_or_create_remote_channel(pending.holder_daemon_id);
        if (!channel_or.ok()) {
          completed_results.push_back(make_home_batch_get_result(
              pending,
              grpc::Status(StatusCode::UNAVAILABLE, "home daemon address unavailable"),
              "home daemon address unavailable"));
          continue;
        }
        remote_channel = *channel_or;
      }
      const absl::Time enqueued_at = absl::Now();

      scheduled.push_back(
          ScheduledHomeBatchGet{
              .request = pending,
              .future = folly::via(
                            d_.async_runtime.blocking_executor(),
                            [this,
                             pending,
                             remote_channel = std::move(remote_channel),
                             allow_high_card_attrs,
                             local_daemon_id,
                             operation_id_owned,
                             enqueued_at]() mutable -> HomeBatchGetRpcResult {
                              HomeBatchGetRpcResult result;
                              result.request = pending;
                              const absl::Time task_started_at = absl::Now();
                              result.queue_elapsed = task_started_at - enqueued_at;

                              v2::HomeBatchGetRequest home_req;
                              home_req.mutable_fence()->set_shard_id(pending.shard_id);
                              home_req.mutable_fence()->set_lease_generation(pending.lease_generation);
                              home_req.mutable_fence()->set_holder_daemon_id(pending.holder_daemon_id);
                              home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
                              if (!operation_id_owned.empty()) {
                                home_req.set_operation_id(operation_id_owned);
                              }
                              home_req.set_requester_daemon_id(local_daemon_id);
                              for (const auto& artifact_id : pending.batch->artifact_ids) {
                                home_req.add_artifact_ids(artifact_id);
                              }

                              const absl::Time home_rpc_started_at = absl::Now();
                              if (!pending.remote_home) {
                                grpc::ServerContext home_ctx;
                                RpcContext home_rctx{"HomeBatchGet", home_ctx, allow_high_card_attrs};
                                result.status = home_batch_get(home_rctx, home_req, result.home_resp);
                              } else {
                                auto stub = v2::StoreDaemonService::NewStub(remote_channel);
                                grpc::ClientContext client_ctx;
                                client_ctx.set_deadline(
                                    std::chrono::system_clock::now() + inter_daemon_home_rpc_timeout(options_));
                                result.status = stub->HomeBatchGet(&client_ctx, home_req, &result.home_resp);
                              }
                              result.rpc_elapsed = absl::Now() - home_rpc_started_at;
                              result.task_elapsed = absl::Now() - enqueued_at;
                              return result;
                            })
                            .semi(),
          });
    }

    for (auto& pending : scheduled) {
      try {
        completed_results.push_back(std::move(pending.future).get());
      } catch (const std::exception& ex) {
        completed_results.push_back(
            make_home_batch_get_result(pending.request, grpc::Status(StatusCode::INTERNAL, ex.what()), ex.what()));
      } catch (...) {
        completed_results.push_back(make_home_batch_get_result(
            pending.request,
            grpc::Status(StatusCode::INTERNAL, "HomeBatchGet fanout task failed"),
            "HomeBatchGet fanout task failed"));
      }
    }
    home_dispatch_elapsed += absl::Now() - dispatch_started_at;
    return completed_results;
  };

  absl::flat_hash_map<std::uint64_t, CompletedHomeBatchGet> completed_home_batches;
  completed_home_batches.reserve(pending_batches.size());
  std::function<void(std::vector<HomeBatchGetRpcResult>, bool)> process_home_batch_get_results;
  process_home_batch_get_results = [&](std::vector<HomeBatchGetRpcResult> results, bool allow_retry) {
    const absl::Time process_started_at = absl::Now();
    std::vector<PendingHomeBatchGet> retry_batches;
    retry_batches.reserve(results.size());

    for (auto& result : results) {
      if (result.request.remote_home) {
        stats.remote_home_rpc_elapsed += result.rpc_elapsed;
        stats.remote_home_rpc_max_elapsed = std::max(stats.remote_home_rpc_max_elapsed, result.rpc_elapsed);
        stats.remote_home_queue_elapsed += result.queue_elapsed;
        stats.remote_home_queue_max_elapsed = std::max(stats.remote_home_queue_max_elapsed, result.queue_elapsed);
        stats.remote_home_task_elapsed += result.task_elapsed;
        stats.remote_home_task_max_elapsed = std::max(stats.remote_home_task_max_elapsed, result.task_elapsed);
      } else {
        stats.local_home_rpc_elapsed += result.rpc_elapsed;
        stats.local_home_rpc_max_elapsed = std::max(stats.local_home_rpc_max_elapsed, result.rpc_elapsed);
        stats.local_home_queue_elapsed += result.queue_elapsed;
        stats.local_home_queue_max_elapsed = std::max(stats.local_home_queue_max_elapsed, result.queue_elapsed);
        stats.local_home_task_elapsed += result.task_elapsed;
        stats.local_home_task_max_elapsed = std::max(stats.local_home_task_max_elapsed, result.task_elapsed);
      }
      LOG(INFO) << "byte_artifact.batch_get_into_region_home_rpc_result"
                << " operation_id=" << operation_id << " shard_id=" << result.request.shard_id
                << " remote_home=" << result.request.remote_home
                << " holder_daemon_id=" << result.request.holder_daemon_id
                << " requested_artifacts=" << result.request.batch->artifact_ids.size()
                << " attempt=" << result.request.attempt
                << " queue_ms=" << absl::ToDoubleMilliseconds(result.queue_elapsed)
                << " rpc_ms=" << absl::ToDoubleMilliseconds(result.rpc_elapsed)
                << " task_wall_ms=" << absl::ToDoubleMilliseconds(result.task_elapsed)
                << " status_ok=" << result.status.ok() << " status_code=" << result.status.error_code()
                << " status_message=" << result.status.error_message();

      if (!result.status.ok()) {
        for (std::size_t idx = 0; idx < result.request.batch->artifact_ids.size(); ++idx) {
          *resp.mutable_outcomes(result.request.batch->outcome_indices[idx]) = make_outcome(
              result.request.batch->artifact_ids[idx],
              v2::BATCH_ITEM_STATUS_UNAVAILABLE,
              result.status.error_message());
        }
        continue;
      }

      bool needs_redirect_retry = false;
      if (result.home_resp.has_redirect() && result.home_resp.redirect().shard_id() == result.request.shard_id &&
          result.home_resp.redirect().lease_generation() != 0 &&
          !result.home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& item : result.home_resp.items()) {
          if (item.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && allow_retry && result.request.attempt == 0) {
        const auto refreshed =
            d_.route_resolver.refresh_route_from_redirect(result.request.shard_id, result.home_resp.redirect(), now);
        if (!refreshed.ok) {
          for (std::size_t idx = 0; idx < result.request.batch->artifact_ids.size(); ++idx) {
            *resp.mutable_outcomes(result.request.batch->outcome_indices[idx]) = make_outcome(
                result.request.batch->artifact_ids[idx], v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          continue;
        }
        retry_batches.push_back(
            PendingHomeBatchGet{
                .shard_id = result.request.shard_id,
                .batch = result.request.batch,
                .holder_daemon_id = refreshed.holder_daemon_id,
                .lease_generation = refreshed.lease_generation,
                .remote_home = refreshed.holder_daemon_id != local_daemon_id,
                .attempt = 1,
            });
        continue;
      }

      const auto home_resp_shape = inspect_home_batch_get_response_shape(result.home_resp);
      LOG(INFO) << "byte_artifact.batch_get_into_region_home_response"
                << " operation_id=" << operation_id << " holder_daemon_id=" << result.request.holder_daemon_id
                << " remote_home=" << result.request.remote_home << " shard_id=" << result.request.shard_id
                << " requested_artifacts=" << result.request.batch->artifact_ids.size()
                << " items=" << home_resp_shape.items << " ok_items=" << home_resp_shape.ok_items
                << " inline_items=" << home_resp_shape.inline_items
                << " payload_ref_items=" << home_resp_shape.payload_ref_items
                << " batch_slice_items=" << home_resp_shape.batch_slice_items
                << " transports=" << home_resp_shape.transports
                << " communicator_transports=" << home_resp_shape.communicator_transports
                << " grpc_chunk_transports=" << home_resp_shape.grpc_chunk_transports;
      completed_home_batches.emplace(
          result.request.shard_id,
          CompletedHomeBatchGet{
              .holder_daemon_id = result.request.holder_daemon_id,
              .remote_home = result.request.remote_home,
              .home_resp = std::move(result.home_resp),
          });
    }

    if (retry_batches.empty()) {
      home_process_elapsed += absl::Now() - process_started_at;
      return;
    }

    std::vector<std::string> retry_remote_daemon_ids;
    retry_remote_daemon_ids.reserve(retry_batches.size());
    for (const auto& pending : retry_batches) {
      if (pending.remote_home) {
        retry_remote_daemon_ids.push_back(pending.holder_daemon_id);
      }
    }
    if (!retry_remote_daemon_ids.empty()) {
      const absl::Time retry_worker_directory_warm_started_at = absl::Now();
      (void)d_.worker_directory_cache.warm_for_daemons(
          retry_remote_daemon_ids,
          absl::Now(),
          absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
      worker_directory_warm_elapsed += absl::Now() - retry_worker_directory_warm_started_at;
    }
    home_process_elapsed += absl::Now() - process_started_at;
    auto retry_results = dispatch_home_batch_get_wave(retry_batches);
    process_home_batch_get_results(std::move(retry_results), /*allow_retry=*/false);
  };

  auto first_wave_results = dispatch_home_batch_get_wave(pending_batches);
  process_home_batch_get_results(std::move(first_wave_results), /*allow_retry=*/true);

  const absl::Time build_apply_work_started_at = absl::Now();
  std::vector<ApplyWorkUnit> all_work_units;
  for (const auto& [shard_id, batch] : shard_requests) {
    const auto completed_it = completed_home_batches.find(shard_id);
    if (completed_it == completed_home_batches.end()) {
      continue;
    }
    auto shard_work_units = build_apply_work_units(shard_id, batch, completed_it->second.home_resp);
    all_work_units.insert(
        all_work_units.end(),
        std::make_move_iterator(shard_work_units.begin()),
        std::make_move_iterator(shard_work_units.end()));
  }
  build_apply_work_elapsed = absl::Now() - build_apply_work_started_at;
  execute_apply_work_units(std::move(all_work_units));

  log_transport_apply_stats();
  log_apply_breakdown();
  attach_slot_tokens_to_outcomes(requested_slot_tokens, resp.mutable_outcomes());
  std::ostringstream summary;
  summary << "byte_artifact.batch_get_into_region_summary"
          << " operation_id=" << (req.has_operation_id() ? req.operation_id() : "")
          << " first_artifact_id=" << first_artifact_id << " selections=" << req.selections_size()
          << " shard_count=" << shard_requests.size() << " local_home_batches=" << stats.local_home_batch_count
          << " remote_home_batches=" << stats.remote_home_batch_count
          << " local_home_items=" << stats.local_home_item_count
          << " remote_home_items=" << stats.remote_home_item_count
          << " local_batch_transport_items=" << stats.local_batch_transport_item_count
          << " remote_batch_transport_items=" << stats.remote_batch_transport_item_count
          << " local_batch_transport_per_item_materialize_calls="
          << stats.local_batch_transport_per_item_materialize_calls
          << " remote_batch_transport_per_item_materialize_calls="
          << stats.remote_batch_transport_per_item_materialize_calls
          << " local_batch_transport_composite_materialize_calls="
          << stats.local_batch_transport_composite_materialize_calls
          << " remote_batch_transport_composite_materialize_calls="
          << stats.remote_batch_transport_composite_materialize_calls
          << " local_source_items=" << stats.local_source_item_count
          << " remote_source_items=" << stats.remote_source_item_count
          << " local_source_bytes=" << stats.local_source_bytes << " remote_source_bytes=" << stats.remote_source_bytes
          << " ensure_local_peer_ms=" << absl::ToDoubleMilliseconds(ensure_local_peer_elapsed)
          << " selection_build_ms=" << absl::ToDoubleMilliseconds(selection_build_elapsed)
          << " validate_target_layout_ms=" << absl::ToDoubleMilliseconds(validate_target_layout_elapsed)
          << " route_resolve_ms=" << absl::ToDoubleMilliseconds(route_resolve_elapsed)
          << " worker_directory_warm_ms=" << absl::ToDoubleMilliseconds(worker_directory_warm_elapsed)
          << " remote_channel_resolve_ms=" << absl::ToDoubleMilliseconds(remote_channel_resolve_elapsed)
          << " remote_channel_resolve_max_ms=" << absl::ToDoubleMilliseconds(remote_channel_resolve_max_elapsed)
          << " remote_channel_resolve_count=" << remote_channel_resolve_count
          << " home_dispatch_ms=" << absl::ToDoubleMilliseconds(home_dispatch_elapsed)
          << " home_process_ms=" << absl::ToDoubleMilliseconds(home_process_elapsed)
          << " build_apply_work_ms=" << absl::ToDoubleMilliseconds(build_apply_work_elapsed)
          << " local_home_queue_ms=" << absl::ToDoubleMilliseconds(stats.local_home_queue_elapsed)
          << " remote_home_queue_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_queue_elapsed)
          << " local_home_queue_max_ms=" << absl::ToDoubleMilliseconds(stats.local_home_queue_max_elapsed)
          << " remote_home_queue_max_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_queue_max_elapsed)
          << " local_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.local_home_rpc_elapsed)
          << " remote_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_elapsed)
          << " local_home_rpc_max_ms=" << absl::ToDoubleMilliseconds(stats.local_home_rpc_max_elapsed)
          << " remote_home_rpc_max_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_max_elapsed)
          << " local_home_task_ms=" << absl::ToDoubleMilliseconds(stats.local_home_task_elapsed)
          << " remote_home_task_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_task_elapsed)
          << " local_home_task_max_ms=" << absl::ToDoubleMilliseconds(stats.local_home_task_max_elapsed)
          << " remote_home_task_max_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_task_max_elapsed)
          << " local_source_apply_ms=" << absl::ToDoubleMilliseconds(stats.local_source_apply_elapsed)
          << " remote_source_apply_ms=" << absl::ToDoubleMilliseconds(stats.remote_source_apply_elapsed)
          << " apply_wall_ms=" << absl::ToDoubleMilliseconds(apply_wall_elapsed)
          << " total_wall_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at);
  if (req.has_operation_id() && !req.operation_id().empty()) {
    LOG(INFO) << summary.str();
  } else {
    VLOG(2) << summary.str();
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_put_if_absent_from_region(
    RpcContext& rctx,
    const v2::BatchPutIfAbsentFromRegionRequest& req,
    v2::BatchPutIfAbsentFromRegionResponse& resp) {
  auto local_peer_status = d_.external_target_access_service.ensure_local_region_peer(
      rctx.server_context().peer(), "BatchPutIfAbsentFromRegion");
  if (!local_peer_status.ok()) {
    return to_grpc_status(local_peer_status);
  }
  const absl::Time now = absl::Now();
  const std::optional<std::uint64_t> ttl_ms =
      req.has_ttl_ms() ? std::optional<std::uint64_t>(req.ttl_ms()) : std::nullopt;
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();
  const std::string operation_id = req.has_operation_id() ? req.operation_id() : "";
  const auto requested_slot_tokens = collect_batch_item_slot_tokens(req.source_layout());

  struct PendingPut {
    std::string artifact_id;
    v2::PutIfAbsentInvariant invariant;
    std::optional<BodyHandle> body_handle;
    std::optional<BodyDescriptor> descriptor;
    std::optional<store::runtime::ingestion::VerifiedContentDescriptor> verified_content_descriptor;
    std::optional<store::runtime::ingestion::VerificationRecord> verification_record;
    std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
    std::optional<BodyBackingObservation> observation;
    std::shared_ptr<const std::string> inline_payload;
    std::string payload_ref;
    bool needs_source_layout{false};
    int outcome_index{0};
  };

  struct ShardPutBatch {
    std::vector<PendingPut> items;
  };

  absl::flat_hash_map<std::uint64_t, ShardPutBatch> shard_batches;
  shard_batches.reserve(static_cast<size_t>(req.items_size()));
  absl::flat_hash_map<std::string, std::uint64_t> source_layout_lengths;
  source_layout_lengths.reserve(static_cast<std::size_t>(req.items_size()));

  for (int i = 0; i < req.items_size(); ++i) {
    const auto& item = req.items(i);
    const std::string artifact_id = item.selection().artifact_id();
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(artifact_id);

    const auto selection_st = validate_batch_selection(item.selection());
    if (!selection_st.ok()) {
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(selection_st.message()));
      continue;
    }

    auto shard_or = byte_artifact_runtime().shard_id_for_artifact(artifact_id, options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }

    auto& batch = shard_batches[*shard_or];
    PendingPut pending{
        .artifact_id = artifact_id,
        .invariant = item.invariant(),
        .outcome_index = i,
    };
    if (!item.inline_payload().empty() && !item.payload_ref().empty()) {
      *outcome = make_outcome(
          artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, "inline_payload and payload_ref are mutually exclusive");
      continue;
    }
    if (!item.inline_payload().empty()) {
      pending.inline_payload = std::make_shared<const std::string>(item.inline_payload());
    } else if (!item.payload_ref().empty()) {
      pending.payload_ref = item.payload_ref();
    } else {
      pending.needs_source_layout = true;
      source_layout_lengths[artifact_id] = item.invariant().byte_length();
    }
    batch.items.push_back(std::move(pending));
  }

  std::optional<ByteArtifactRegionLayout> source_layout;
  if (!source_layout_lengths.empty()) {
    auto source_layout_or = d_.external_target_access_service.validate_local_source_layout(
        rctx.server_context().peer(),
        "BatchPutIfAbsentFromRegion",
        req.source_layout(),
        req.pid(),
        req.device_uuid(),
        source_layout_lengths);
    if (!source_layout_or.ok()) {
      const auto status = batch_item_status_from_absl_status(source_layout_or.status());
      for (auto& [_, batch] : shard_batches) {
        for (auto& pending : batch.items) {
          if (!pending.needs_source_layout) {
            continue;
          }
          *resp.mutable_outcomes(pending.outcome_index) =
              make_outcome(pending.artifact_id, status, std::string(source_layout_or.status().message()));
        }
      }
    } else {
      source_layout = std::move(source_layout_or->layout);
    }
  }

  struct BatchPutInstrumentation {
    std::size_t local_home_batch_count{0};
    std::size_t remote_home_batch_count{0};
    std::size_t local_home_item_count{0};
    std::size_t remote_home_item_count{0};
    std::size_t local_stage_item_count{0};
    std::size_t remote_stage_item_count{0};
    std::size_t remote_payload_ref_count{0};
    std::size_t remote_batch_transport_count{0};
    std::size_t remote_batch_transport_communicator_count{0};
    std::size_t remote_batch_transport_grpc_count{0};
    std::size_t remote_batch_transport_item_count{0};
    std::uint64_t remote_batch_transport_bytes{0};
    std::size_t remote_batch_pack_count{0};
    std::size_t remote_batch_pack_item_count{0};
    std::uint64_t remote_batch_pack_bytes{0};
    std::size_t remote_batch_segmented_region_export_count{0};
    std::size_t remote_batch_segmented_region_export_item_count{0};
    std::uint64_t remote_batch_segmented_region_export_bytes{0};
    absl::Duration local_stage_elapsed{absl::ZeroDuration()};
    absl::Duration remote_stage_elapsed{absl::ZeroDuration()};
    absl::Duration remote_batch_pack_elapsed{absl::ZeroDuration()};
    absl::Duration remote_batch_segmented_region_export_elapsed{absl::ZeroDuration()};
    absl::Duration local_home_apply_elapsed{absl::ZeroDuration()};
    absl::Duration remote_home_rpc_elapsed{absl::ZeroDuration()};
  } stats;

  struct RemoteHomePutOutcomeSlot {
    std::string artifact_id;
    int outcome_index{0};
  };

  struct PreparedRemoteHomeBatchPut {
    std::uint64_t shard_id{0};
    std::string holder_daemon_id;
    std::uint64_t lease_generation{0};
    int attempt{0};
    v2::HomeBatchPutIfAbsentRequest home_req;
    std::vector<RemoteHomePutOutcomeSlot> outcome_slots;
    std::vector<BodyHandle> retire_handles;
  };

  struct HomeBatchPutRpcResult {
    PreparedRemoteHomeBatchPut request;
    grpc::Status status;
    v2::HomeBatchPutIfAbsentResponse home_resp;
    absl::Duration rpc_elapsed{absl::ZeroDuration()};
  };

  struct PutShardTaskInput {
    std::uint64_t shard_id{0};
    ByteArtifactRouteResolver::RouteDecision route;
    ShardPutBatch batch;
  };

  struct PutShardTaskResult {
    BatchPutInstrumentation stats;
    std::vector<std::pair<int, v2::BatchItemOutcome>> outcomes;
    absl::Duration total_elapsed{absl::ZeroDuration()};
  };

  const auto retire_remote_home_batch_handles = [&](PreparedRemoteHomeBatchPut* request) {
    if (request == nullptr) {
      return;
    }
    for (const auto& body_handle : request->retire_handles) {
      (void)body_handle.retire();
    }
    request->retire_handles.clear();
  };

  const auto make_home_batch_put_result = [&](PreparedRemoteHomeBatchPut request,
                                              grpc::Status status,
                                              std::string_view message = "") -> HomeBatchPutRpcResult {
    if (!status.ok() && !message.empty()) {
      status = grpc::Status(status.error_code(), std::string(message));
    }
    return HomeBatchPutRpcResult{
        .request = std::move(request),
        .status = std::move(status),
    };
  };

  const bool local_only = (d_.global_store_client == nullptr);
  absl::flat_hash_map<std::uint64_t, ByteArtifactRouteResolver::RouteDecision> routes;
  routes.reserve(shard_batches.size());

  if (local_only) {
    for (const auto& [shard_id, /*batch*/ _] : shard_batches) {
      routes.emplace(
          shard_id,
          ByteArtifactRouteResolver::RouteDecision{
              .ok = true,
              .lease_generation = 1,
              .holder_daemon_id = local_daemon_id,
          });
    }
  } else {
    std::vector<std::uint64_t> shard_ids;
    shard_ids.reserve(shard_batches.size());
    for (const auto& [shard_id, /*batch*/ _] : shard_batches) {
      shard_ids.push_back(shard_id);
    }
    routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);
  }

  absl::flat_hash_map<std::string, bool> remote_daemon_id_set;
  remote_daemon_id_set.reserve(routes.size());
  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    (void)shard_id;
    if (!route.ok || route.holder_daemon_id.empty() || route.holder_daemon_id == local_daemon_id) {
      continue;
    }
    const auto [_, inserted] = remote_daemon_id_set.emplace(route.holder_daemon_id, true);
    if (inserted) {
      remote_daemon_ids.push_back(route.holder_daemon_id);
    }
  }

  if (!remote_daemon_ids.empty()) {
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  }

  absl::flat_hash_map<std::string, std::shared_ptr<grpc::Channel>> prebuilt_remote_channels;
  prebuilt_remote_channels.reserve(remote_daemon_ids.size());
  absl::flat_hash_map<std::string, PeerBatchTransportSupport> prebuilt_peer_transport_support;
  prebuilt_peer_transport_support.reserve(remote_daemon_ids.size());
  absl::flat_hash_map<std::string, std::string> prebuilt_remote_cpu_endpoint_ids;
  prebuilt_remote_cpu_endpoint_ids.reserve(remote_daemon_ids.size());
  for (const auto& daemon_id : remote_daemon_ids) {
    auto address_or = d_.worker_directory_cache.resolve_daemon_address(
        daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    if (address_or.ok()) {
      prebuilt_remote_channels.emplace(
          daemon_id, create_inter_daemon_channel(*address_or, d_.inter_daemon_channel_credentials));
    }
    prebuilt_peer_transport_support.emplace(daemon_id, resolve_peer_batch_transport_support(daemon_id, now));
    auto entry_or = d_.worker_directory_cache.resolve_daemon_entry(
        daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    if (entry_or.ok() && !entry_or->node_id.empty()) {
      prebuilt_remote_cpu_endpoint_ids.emplace(
          daemon_id,
          store::components::derive_endpoint_id(
              entry_or->node_id, common::memory::MemoryLocation::CPU, /*device_id=*/0));
    }
  }

  struct LocalProducerEndpoint {
    bool available{false};
    std::string node_address;
    std::uint32_t p2p_port{0};
    std::string node_id;
  } local_producer_endpoint;

  auto producer_entry_or = d_.worker_directory_cache.resolve_daemon_entry(
      local_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  if (producer_entry_or.ok()) {
    local_producer_endpoint.available = producer_entry_or->p2p_port != 0;
    local_producer_endpoint.node_address = producer_entry_or->node_address;
    local_producer_endpoint.p2p_port = producer_entry_or->p2p_port;
    local_producer_endpoint.node_id = producer_entry_or->node_id;
  }

  std::vector<PutShardTaskInput> shard_tasks;
  shard_tasks.reserve(shard_batches.size());
  for (auto& [shard_id, batch] : shard_batches) {
    if (batch.items.empty()) {
      continue;
    }
    const auto route_it = routes.find(shard_id);
    shard_tasks.push_back(
        PutShardTaskInput{
            .shard_id = shard_id,
            .route = route_it != routes.end()
                ? route_it->second
                : ByteArtifactRouteResolver::RouteDecision{
                      .ok = false,
                      .message = "routing lease unavailable",
                  },
            .batch = std::move(batch),
        });
  }

  const auto resolve_remote_channel_for_task =
      [&](std::string_view daemon_id) -> absl::StatusOr<std::shared_ptr<grpc::Channel>> {
    const auto cached_it = prebuilt_remote_channels.find(std::string(daemon_id));
    if (cached_it != prebuilt_remote_channels.end()) {
      return cached_it->second;
    }
    auto address_or = d_.worker_directory_cache.resolve_daemon_address(
        daemon_id, absl::Now(), absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
    if (!address_or.ok()) {
      return address_or.status();
    }
    return create_inter_daemon_channel(*address_or, d_.inter_daemon_channel_credentials);
  };

  const auto resolve_peer_batch_transport_support_for_task = [&](std::string_view daemon_id) {
    const auto cached_it = prebuilt_peer_transport_support.find(std::string(daemon_id));
    if (cached_it != prebuilt_peer_transport_support.end()) {
      return cached_it->second;
    }
    return resolve_peer_batch_transport_support(daemon_id, absl::Now());
  };

  const auto run_put_shard_task = [&](PutShardTaskInput task) -> PutShardTaskResult {
    const absl::Time task_started_at = absl::Now();
    PutShardTaskResult task_result;
    auto& task_stats = task_result.stats;

    std::vector<v2::BatchItemOutcome> task_outcomes(static_cast<std::size_t>(req.items_size()));
    std::vector<bool> task_has_outcome(static_cast<std::size_t>(req.items_size()), false);
    std::vector<int> touched_outcomes;
    touched_outcomes.reserve(task.batch.items.size());

    const auto record_outcome = [&](int outcome_index, v2::BatchItemOutcome outcome) {
      const auto cast_index = static_cast<std::size_t>(outcome_index);
      if (!task_has_outcome[cast_index]) {
        task_has_outcome[cast_index] = true;
        touched_outcomes.push_back(outcome_index);
      }
      task_outcomes[cast_index] = std::move(outcome);
    };

    const auto has_outcome = [&](int outcome_index) {
      return task_has_outcome[static_cast<std::size_t>(outcome_index)];
    };

    const auto stage_pending_body = [&](PendingPut* pending, BodyAccessClass access_class) {
      if (pending == nullptr || pending->body_handle.has_value() || has_outcome(pending->outcome_index)) {
        return;
      }
      const absl::Time stage_started_at = absl::Now();

      std::unique_ptr<store::IArtifactLoader> loader;
      store::loading::MaterializationSource source_kind{store::loading::MaterializationSource::kLocalReplica};
      std::optional<BodyBackingManager::StageResult> staged_body;
      if (pending->inline_payload) {
        loader = std::make_unique<store::InlineBufferLoader>(store::loading::InlineBufferSource{
            .data = std::shared_ptr<const void>(
                pending->inline_payload, static_cast<const void*>(pending->inline_payload->data())),
            .size_bytes = pending->inline_payload->size(),
        });
      } else if (!pending->payload_ref.empty()) {
        auto source_capability_or = d_.payload_transport_broker.resolve_payload_ref_capability(
            d_.worker_directory_cache,
            pending->payload_ref,
            pending->artifact_id,
            now,
            absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
            local_daemon_id,
            tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
            operation_id);
        if (!source_capability_or.ok()) {
          record_outcome(
              pending->outcome_index,
              make_outcome(
                  pending->artifact_id,
                  batch_item_status_from_absl_status(source_capability_or.status()),
                  std::string(source_capability_or.status().message())));
          return;
        }
        const bool allow_reuse = access_class == BodyAccessClass::kHomeDefault;
        if (allow_reuse && source_capability_or->body_capability.has_value()) {
          auto reused_or = body_backing_manager_.try_reuse_body(
              BodyBackingManager::ReuseRequest{
                  .artifact_id = pending->artifact_id,
                  .invariant = pending->invariant,
                  .descriptor = source_capability_or->body_capability->descriptor,
                  .body_handle = source_capability_or->body_capability->body_handle,
                  .operation_id = operation_id,
                  .access_class = access_class,
                  .route_role = access_class == BodyAccessClass::kTransientForward ? BodyRouteRole::kTransientForwarder
                                                                                   : BodyRouteRole::kHomeAuthority,
              });
          if (!reused_or.ok()) {
            record_outcome(
                pending->outcome_index,
                make_outcome(
                    pending->artifact_id,
                    batch_item_status_from_absl_status(reused_or.status()),
                    std::string(reused_or.status().message())));
            return;
          }
          if (reused_or->has_value()) {
            staged_body = std::move(**reused_or);
          }
        }
        if (!staged_body.has_value()) {
          auto loader_or = open_loader_from_resolved_source_capability(
              d_.payload_transport_broker,
              d_.worker_directory_cache,
              *source_capability_or,
              now,
              absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()),
              local_daemon_id,
              tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
              operation_id);
          if (!loader_or.ok()) {
            record_outcome(
                pending->outcome_index,
                make_outcome(
                    pending->artifact_id,
                    batch_item_status_from_absl_status(loader_or.status()),
                    std::string(loader_or.status().message())));
            return;
          }
          source_kind = loader_or->source_kind;
          loader = std::move(loader_or->loader);
        }
      } else if (pending->needs_source_layout) {
        if (!source_layout.has_value()) {
          record_outcome(
              pending->outcome_index,
              make_outcome(
                  pending->artifact_id,
                  v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION,
                  "source_layout validation did not produce a readable layout"));
          return;
        }
        auto source_or = source_layout->open_item_source(pending->artifact_id);
        if (!source_or.ok()) {
          record_outcome(
              pending->outcome_index,
              make_outcome(
                  pending->artifact_id,
                  batch_item_status_from_absl_status(source_or.status()),
                  std::string(source_or.status().message())));
          return;
        }
        loader = std::make_unique<SeekableSourceLoader>(*source_or, (*source_or)->total_bytes());
      } else {
        record_outcome(
            pending->outcome_index,
            make_outcome(
                pending->artifact_id,
                v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT,
                "inline_payload, payload_ref, or source_layout entry is required"));
        return;
      }

      if (!staged_body.has_value()) {
        auto staged_body_or = body_backing_manager_.stage_body(
            BodyBackingManager::StageRequest{
                .artifact_id = pending->artifact_id,
                .invariant = pending->invariant,
                .loader = std::move(loader),
                .source_kind = source_kind,
                .operation_id = operation_id,
                .access_class = access_class,
                .route_role = access_class == BodyAccessClass::kTransientForward ? BodyRouteRole::kTransientForwarder
                                                                                 : BodyRouteRole::kHomeAuthority,
            });
        if (!staged_body_or.ok()) {
          record_outcome(
              pending->outcome_index,
              make_outcome(
                  pending->artifact_id,
                  batch_item_status_from_absl_status(staged_body_or.status()),
                  std::string(staged_body_or.status().message())));
          return;
        }
        staged_body = std::move(*staged_body_or);
      }

      auto invariant_st =
          byte_artifact_runtime().validate_invariant_body_descriptor(pending->invariant, staged_body->descriptor);
      if (!invariant_st.ok()) {
        auto retire_status = staged_body->body_handle.retire();
        if (!retire_status.ok()) {
          record_outcome(
              pending->outcome_index,
              make_outcome(
                  pending->artifact_id,
                  batch_item_status_from_absl_status(retire_status),
                  std::string(retire_status.message())));
          return;
        }
        record_outcome(
            pending->outcome_index,
            make_outcome(
                pending->artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(invariant_st.message())));
        return;
      }

      pending->body_handle = staged_body->body_handle;
      pending->descriptor = staged_body->descriptor;
      pending->verified_content_descriptor = staged_body->verified_content_descriptor;
      pending->verification_record = staged_body->verification_record;
      pending->backing_identity = staged_body->backing_identity;
      pending->observation = staged_body->observation;
      pending->needs_source_layout = false;

      const absl::Duration stage_elapsed = absl::Now() - stage_started_at;
      if (access_class == BodyAccessClass::kTransientForward) {
        ++task_stats.remote_stage_item_count;
        task_stats.remote_stage_elapsed += stage_elapsed;
      } else {
        ++task_stats.local_stage_item_count;
        task_stats.local_stage_elapsed += stage_elapsed;
      }
    };

    const auto apply_local_home_put = [&](std::uint64_t lease_generation) {
      const absl::Time apply_started_at = absl::Now();
      v2::RouteFence fence;
      fence.set_shard_id(task.shard_id);
      fence.set_lease_generation(lease_generation);
      fence.set_holder_daemon_id(local_daemon_id);
      fence.set_routing_epoch(options_.routing.routing_epoch);
      auto home_lease_or = d_.route_resolver.ensure_home_lease(fence, now);
      if (!home_lease_or.ok()) {
        for (const auto& pending : task.batch.items) {
          if (!pending.body_handle.has_value()) {
            continue;
          }
          record_outcome(
              pending.outcome_index,
              make_outcome(
                  pending.artifact_id,
                  batch_item_status_from_absl_status(home_lease_or.status()),
                  std::string(home_lease_or.status().message())));
        }
        return;
      }
      if (home_lease_or->kind != ByteArtifactRouteResolver::HomeLeaseDecision::Kind::kOwned) {
        for (const auto& pending : task.batch.items) {
          if (!pending.body_handle.has_value()) {
            continue;
          }
          record_outcome(
              pending.outcome_index,
              make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_lease_or->message));
        }
        return;
      }

      std::vector<ByteArtifactAuthorityService::PutItem> authority_items;
      std::vector<int> authority_item_indices;
      authority_items.reserve(task.batch.items.size());
      authority_item_indices.reserve(task.batch.items.size());
      for (const auto& pending : task.batch.items) {
        if (!pending.body_handle.has_value()) {
          continue;
        }
        if (!pending.descriptor.has_value() || !pending.verified_content_descriptor.has_value() ||
            !pending.verification_record.has_value() || !pending.backing_identity.has_value() ||
            !pending.observation.has_value()) {
          record_outcome(
              pending.outcome_index,
              make_outcome(
                  pending.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "staged body metadata is missing"));
          continue;
        }
        authority_items.push_back(
            ByteArtifactAuthorityService::PutItem{
                .artifact_id = pending.artifact_id,
                .invariant = pending.invariant,
                .descriptor = *pending.descriptor,
                .verified_content_descriptor = *pending.verified_content_descriptor,
                .verification_record = *pending.verification_record,
                .backing_identity = *pending.backing_identity,
                .observation = *pending.observation,
                .body_handle = *pending.body_handle,
            });
        authority_item_indices.push_back(pending.outcome_index);
      }
      const auto authority_outcomes = authority_service_.batch_put_if_absent(
          authority_items,
          ByteArtifactAuthorityService::Context{
              .shard_id = task.shard_id,
              .lease_generation = home_lease_or->lease_generation,
              .routing_epoch = options_.routing.routing_epoch,
              .shard_count = options_.routing.shard_count,
              .now = now,
          },
          ttl_ms);
      for (std::size_t index = 0; index < authority_outcomes.size(); ++index) {
        record_outcome(authority_item_indices[index], authority_outcomes[index]);
      }
      if (options_.publish_prereg.enabled) {
        for (std::size_t index = 0; index < authority_outcomes.size(); ++index) {
          if (authority_outcomes[index].status() != v2::BATCH_ITEM_STATUS_OK) {
            continue;
          }
          auto entry = d_.body_store.get(
              authority_items[index].artifact_id,
              task.shard_id,
              home_lease_or->lease_generation,
              options_.routing.routing_epoch,
              now);
          if (!entry.has_value()) {
            VLOG(2) << "byte_artifact.publish_prereg.skip"
                    << " artifact_id=" << authority_items[index].artifact_id
                    << " published_at_ms=" << absl::ToUnixMillis(now) << " reason=canonical_backing_missing";
            continue;
          }
          publish_preregistered_export(
              authority_items[index].artifact_id, entry->backing_record.retained_body_handle, now);
        }
      }
      task_stats.local_home_apply_elapsed += absl::Now() - apply_started_at;
    };

    const auto finalize_task = [&]() -> PutShardTaskResult {
      for (const auto& pending : task.batch.items) {
        if (!has_outcome(pending.outcome_index)) {
          record_outcome(
              pending.outcome_index,
              make_outcome(
                  pending.artifact_id,
                  v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
                  "PutShardTask completed without producing an outcome"));
        }
      }
      task_result.outcomes.reserve(touched_outcomes.size());
      for (const auto outcome_index : touched_outcomes) {
        task_result.outcomes.emplace_back(
            outcome_index, std::move(task_outcomes[static_cast<std::size_t>(outcome_index)]));
      }
      task_result.total_elapsed = absl::Now() - task_started_at;
      VLOG(2) << "byte_artifact.put_shard_task_summary"
              << " operation_id=" << operation_id << " shard_id=" << task.shard_id
              << " holder_daemon_id=" << task.route.holder_daemon_id
              << " remote=" << (task.route.holder_daemon_id != local_daemon_id)
              << " requested_items=" << task.batch.items.size()
              << " local_stage_ms=" << absl::ToDoubleMilliseconds(task_stats.local_stage_elapsed)
              << " remote_stage_ms=" << absl::ToDoubleMilliseconds(task_stats.remote_stage_elapsed)
              << " remote_batch_pack_count=" << task_stats.remote_batch_pack_count
              << " remote_batch_pack_items=" << task_stats.remote_batch_pack_item_count
              << " remote_batch_pack_bytes=" << task_stats.remote_batch_pack_bytes
              << " remote_batch_pack_ms=" << absl::ToDoubleMilliseconds(task_stats.remote_batch_pack_elapsed)
              << " remote_batch_segmented_region_export_count=" << task_stats.remote_batch_segmented_region_export_count
              << " remote_batch_segmented_region_export_items="
              << task_stats.remote_batch_segmented_region_export_item_count
              << " remote_batch_segmented_region_export_bytes=" << task_stats.remote_batch_segmented_region_export_bytes
              << " remote_batch_segmented_region_export_ms="
              << absl::ToDoubleMilliseconds(task_stats.remote_batch_segmented_region_export_elapsed)
              << " local_home_apply_ms=" << absl::ToDoubleMilliseconds(task_stats.local_home_apply_elapsed)
              << " remote_home_rpc_ms=" << absl::ToDoubleMilliseconds(task_stats.remote_home_rpc_elapsed)
              << " total_ms=" << absl::ToDoubleMilliseconds(task_result.total_elapsed);
      return task_result;
    };

    if (!task.route.ok) {
      const std::string message = task.route.message.empty() ? "routing lease unavailable" : task.route.message;
      for (const auto& pending : task.batch.items) {
        record_outcome(
            pending.outcome_index, make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, message));
      }
      return finalize_task();
    }

    const bool is_remote = task.route.holder_daemon_id != local_daemon_id;
    if (!is_remote) {
      ++task_stats.local_home_batch_count;
      task_stats.local_home_item_count += task.batch.items.size();
      for (auto& pending : task.batch.items) {
        stage_pending_body(&pending, BodyAccessClass::kHomeDefault);
      }
      apply_local_home_put(task.route.lease_generation != 0 ? task.route.lease_generation : 1);
      return finalize_task();
    }

    ++task_stats.remote_home_batch_count;
    task_stats.remote_home_item_count += task.batch.items.size();
    const PeerBatchTransportSupport peer_transport_support =
        resolve_peer_batch_transport_support_for_task(task.route.holder_daemon_id);
    std::vector<v2::BatchPayloadTransport> batch_transports;
    absl::flat_hash_map<int, v2::BatchPayloadSlice> batch_slice_by_outcome_index;

    const auto log_source_no_pack_fallback =
        [&](const PendingPut& pending, std::string_view reason, std::string_view message = "") {
          VLOG(2) << "byte_artifact.batch_put_if_absent_from_region_segmented_export_fallback"
                  << " operation_id=" << operation_id << " shard_id=" << task.shard_id
                  << " holder_daemon_id=" << task.route.holder_daemon_id << " artifact_id=" << pending.artifact_id
                  << " reason=" << reason << " message=" << message;
        };

    const auto stage_source_layout_fallback_entries =
        [&](absl::Span<PendingPut*> pending_items, std::string_view reason, std::string_view message = "") {
          for (PendingPut* pending : pending_items) {
            if (pending == nullptr || has_outcome(pending->outcome_index) || pending->body_handle.has_value()) {
              continue;
            }
            log_source_no_pack_fallback(*pending, reason, message);
            stage_pending_body(pending, BodyAccessClass::kTransientForward);
          }
        };

    const bool source_layout_no_pack_possible = peer_transport_support.supports_segmented_communicator_export() &&
        d_.payload_transport_broker.batch_transport_segmented_communicator_export_enabled() &&
        local_producer_endpoint.available && local_producer_endpoint.p2p_port != 0;

    std::vector<SourceLayoutBatchPayloadEntry> source_layout_entries;
    std::vector<PendingPut*> source_layout_pending;
    source_layout_entries.reserve(task.batch.items.size());
    source_layout_pending.reserve(task.batch.items.size());
    for (auto& pending : task.batch.items) {
      if (has_outcome(pending.outcome_index)) {
        continue;
      }
      bool admitted_source_layout_no_pack = false;
      if (pending.needs_source_layout) {
        if (!source_layout_no_pack_possible) {
          std::string_view reason = "peer_lacks_segmented_export";
          if (peer_transport_support.supports_segmented_communicator_export() &&
              !d_.payload_transport_broker.batch_transport_segmented_communicator_export_enabled()) {
            reason = "local_segmented_export_disabled";
          } else if (peer_transport_support.supports_segmented_communicator_export()) {
            reason = "producer_endpoint_unavailable";
          }
          log_source_no_pack_fallback(pending, reason);
        } else if (!source_layout.has_value()) {
          log_source_no_pack_fallback(pending, "source_layout_missing");
        } else if (verification_mode_requires_payload_digest(invariant_verification_mode(pending.invariant))) {
          log_source_no_pack_fallback(pending, "strict_digest");
        } else {
          auto source_span_or = source_layout->open_host_shared_source_span(pending.artifact_id);
          if (!source_span_or.ok()) {
            log_source_no_pack_fallback(pending, "not_host_shared", std::string(source_span_or.status().message()));
          } else if (source_span_or->length != pending.invariant.byte_length()) {
            log_source_no_pack_fallback(pending, "source_span_length_mismatch");
          } else {
            source_layout_entries.push_back(
                SourceLayoutBatchPayloadEntry{
                    .artifact_id = pending.artifact_id,
                    .payload_size_bytes = source_span_or->length,
                    .source_span = std::move(*source_span_or),
                    .digest_alg = normalize_body_digest_value(pending.invariant.payload_digest_alg()),
                    .digest_hex = normalize_body_digest_value(pending.invariant.payload_digest_hex()),
                    .capability_expires_at = absl::InfiniteFuture(),
                });
            source_layout_pending.push_back(&pending);
            admitted_source_layout_no_pack = true;
          }
        }
      }
      if (!admitted_source_layout_no_pack) {
        stage_pending_body(&pending, BodyAccessClass::kTransientForward);
      }
    }

    if (!source_layout_entries.empty()) {
      const absl::Time export_started_at = absl::Now();
      auto plans_or = plan_source_layout_batch_payload_entries(
          source_layout_entries,
          d_.payload_transport_broker.max_batch_payload_bytes(),
          d_.payload_transport_broker.max_batch_items());
      if (!plans_or.ok()) {
        stage_source_layout_fallback_entries(
            absl::MakeSpan(source_layout_pending), "segment_budget_exceeded", std::string(plans_or.status().message()));
      } else {
        for (const auto& pack : *plans_or) {
          std::vector<PendingPut*> pack_pending;
          pack_pending.reserve(pack.source_indices.size());
          std::string host_region_class = "mixed";
          for (const auto entry_index : pack.source_indices) {
            pack_pending.push_back(source_layout_pending[entry_index]);
            const auto current_label =
                host_region_class_label(source_layout_entries[entry_index].source_span.host_region_class);
            if (host_region_class == "mixed") {
              host_region_class = std::string(current_label);
            } else if (host_region_class != current_label) {
              host_region_class = "mixed";
            }
          }

          auto source_segments_or = acquire_segmented_region_source_segments(source_layout_entries, pack);
          if (!source_segments_or.ok()) {
            stage_source_layout_fallback_entries(
                absl::MakeSpan(pack_pending),
                "export_registration_failed",
                std::string(source_segments_or.status().message()));
            continue;
          }
          auto communicator_export_or = d_.payload_transport_broker.issue_batch_payload_communicator_export(
              pack.manifest,
              absl::MakeSpan(*source_segments_or),
              tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
              operation_id,
              pack.capability_expires_at,
              task.route.holder_daemon_id);
          if (!communicator_export_or.ok()) {
            stage_source_layout_fallback_entries(
                absl::MakeSpan(pack_pending),
                "export_registration_failed",
                std::string(communicator_export_or.status().message()));
            continue;
          }

          v2::BatchPayloadTransport transport;
          const std::string transport_id = absl::StrCat("batch-transport-", batch_transports.size() + 1);
          transport.set_transport_id(transport_id);
          transport.mutable_manifest()->CopyFrom(pack.manifest);
          auto* communicator_source = transport.mutable_communicator_source();
          communicator_source->set_batch_payload_ref(communicator_export_or->batch_payload_ref);
          communicator_source->set_protocol_version(d_.payload_transport_broker.batch_transport_protocol_version());
          communicator_source->set_producer_daemon_id(local_daemon_id);
          communicator_source->set_consumer_daemon_id(task.route.holder_daemon_id);
          communicator_source->set_producer_host(local_producer_endpoint.node_address);
          communicator_source->set_producer_port(local_producer_endpoint.p2p_port);
          for (const auto& remote_memory_key : communicator_export_or->export_registration.remote_memory_keys) {
            communicator_source->add_remote_memory_keys(remote_memory_key);
          }
          for (const auto buffer_size : communicator_export_or->export_registration.buffer_sizes) {
            communicator_source->add_buffer_sizes(buffer_size);
          }
          if (!local_producer_endpoint.node_id.empty()) {
            communicator_source->set_remote_endpoint_id(
                store::components::derive_endpoint_id(
                    local_producer_endpoint.node_id, common::memory::MemoryLocation::CPU, /*device_id=*/0));
          }
          const auto consumer_endpoint_it = prebuilt_remote_cpu_endpoint_ids.find(task.route.holder_daemon_id);
          if (consumer_endpoint_it != prebuilt_remote_cpu_endpoint_ids.end()) {
            communicator_source->set_local_endpoint_id_hint(consumer_endpoint_it->second);
          }
          communicator_source->set_memory_location(v2::BATCH_PAYLOAD_MEMORY_LOCATION_HOST);
          communicator_source->set_total_payload_bytes(pack.manifest.total_size());

          ++task_stats.remote_batch_transport_count;
          ++task_stats.remote_batch_transport_communicator_count;
          task_stats.remote_batch_transport_item_count += pack.source_indices.size();
          task_stats.remote_batch_transport_bytes += pack.manifest.total_size();
          ++task_stats.remote_batch_segmented_region_export_count;
          task_stats.remote_batch_segmented_region_export_item_count += pack.source_indices.size();
          task_stats.remote_batch_segmented_region_export_bytes += pack.manifest.total_size();

          LOG(INFO) << "byte_artifact.batch_put_if_absent_from_region_pack_realization"
                    << " operation_id=" << operation_id << " shard_id=" << task.shard_id
                    << " holder_daemon_id=" << task.route.holder_daemon_id << " remote=true"
                    << " mode=segmented_region_export"
                    << " staged_slab=false"
                    << " source_realization_mode="
                    << (communicator_export_or->broker_owned_register ? "source_layout_host_shared"
                                                                      : "source_layout_host_shared_stable_backing")
                    << " host_region_class=" << host_region_class << " pack_count=1"
                    << " item_count=" << pack.source_indices.size() << " payload_bytes=" << pack.manifest.total_size()
                    << " source_segments=" << source_segments_or->size()
                    << " remote_keys=" << communicator_export_or->export_registration.remote_memory_keys.size()
                    << " registration_ownership=" << communicator_export_or->registration_ownership
                    << " mr_ownership=" << communicator_export_or->mr_ownership
                    << " broker_owned_register=" << communicator_export_or->broker_owned_register;

          LOG(INFO) << "byte_artifact.batch_put_if_absent_from_region_transport_emit"
                    << " operation_id=" << operation_id << " shard_id=" << task.shard_id
                    << " holder_daemon_id=" << task.route.holder_daemon_id << " transport_id=" << transport_id
                    << " kind=communicator_source"
                    << " source_realization_mode=segmented_region_export"
                    << " item_count=" << pack.source_indices.size() << " payload_bytes=" << pack.manifest.total_size();

          for (std::size_t pack_index = 0; pack_index < pack.source_indices.size(); ++pack_index) {
            auto slice = pack.slices[pack_index];
            slice.set_transport_id(transport_id);
            batch_slice_by_outcome_index.emplace(
                source_layout_pending[pack.source_indices[pack_index]]->outcome_index, std::move(slice));
          }
          batch_transports.push_back(std::move(transport));
        }
      }
      task_stats.remote_batch_segmented_region_export_elapsed += absl::Now() - export_started_at;
    }

    if (peer_transport_support.supports_v1()) {
      std::vector<BatchPayloadPackEntry> batch_entries;
      std::vector<int> batch_entry_outcome_indices;
      batch_entries.reserve(task.batch.items.size());
      batch_entry_outcome_indices.reserve(task.batch.items.size());
      for (const auto& pending : task.batch.items) {
        if (!pending.body_handle.has_value() || !pending.descriptor.has_value() || has_outcome(pending.outcome_index)) {
          continue;
        }
        batch_entries.push_back(
            BatchPayloadPackEntry{
                .artifact_id = pending.artifact_id,
                .payload_size_bytes = pending.descriptor->size_bytes,
                .body_handle = *pending.body_handle,
                .body_descriptor = *pending.descriptor,
                .backing_identity = pending.backing_identity,
                .backing_instance_generation = pending.body_handle->binding_generation(),
                .digest_alg = pending.descriptor->payload_digest_alg,
                .digest_hex = pending.descriptor->payload_digest_hex,
                .capability_expires_at = absl::InfiniteFuture(),
            });
        batch_entry_outcome_indices.push_back(pending.outcome_index);
      }
      if (!batch_entries.empty()) {
        const absl::Time pack_started_at = absl::Now();
        auto packs_or = pack_batch_payload_entries(
            batch_entries,
            d_.payload_transport_broker.max_batch_payload_bytes(),
            d_.payload_transport_broker.max_batch_items());
        const absl::Duration pack_elapsed = absl::Now() - pack_started_at;
        task_stats.remote_batch_pack_elapsed += pack_elapsed;
        if (!packs_or.ok()) {
          LOG(WARNING) << "batch_put_if_absent_from_region batch transport fallback to payload_ref: "
                       << packs_or.status();
        } else {
          std::uint64_t packed_bytes = 0;
          std::size_t packed_items = 0;
          for (const auto& pack : *packs_or) {
            packed_bytes += pack.manifest.total_size();
            packed_items += pack.source_indices.size();
          }
          task_stats.remote_batch_pack_count += packs_or->size();
          task_stats.remote_batch_pack_item_count += packed_items;
          task_stats.remote_batch_pack_bytes += packed_bytes;
          LOG(INFO) << "byte_artifact.batch_put_if_absent_from_region_pack_realization"
                    << " operation_id=" << operation_id << " shard_id=" << task.shard_id
                    << " holder_daemon_id=" << task.route.holder_daemon_id << " remote=true"
                    << " mode=staged_slab"
                    << " pack_count=" << packs_or->size() << " item_count=" << packed_items
                    << " payload_bytes=" << packed_bytes << " pack_ms=" << absl::ToDoubleMilliseconds(pack_elapsed);
          for (auto& pack : *packs_or) {
            const bool use_communicator_transport = peer_transport_support.supports_v2() &&
                d_.payload_transport_broker.batch_transport_communicator_enabled();
            absl::Status transport_issue_status;
            std::optional<std::string> batch_payload_ref;
            std::optional<store::ExportRegistration> communicator_export;
            bool emitted_communicator_transport = false;
            if (use_communicator_transport) {
              auto communicator_export_or = d_.payload_transport_broker.issue_batch_payload_communicator_export(
                  pack.manifest,
                  pack.payload,
                  tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
                  operation_id,
                  pack.capability_expires_at,
                  task.route.holder_daemon_id);
              if (communicator_export_or.ok()) {
                batch_payload_ref = communicator_export_or->batch_payload_ref;
                communicator_export = communicator_export_or->export_registration;
                emitted_communicator_transport = true;
              } else {
                transport_issue_status = communicator_export_or.status();
              }
            }
            if (!batch_payload_ref.has_value()) {
              auto batch_payload_ref_or = d_.payload_transport_broker.issue_batch_payload_ref(
                  pack.manifest,
                  pack.payload,
                  tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
                  operation_id,
                  pack.capability_expires_at,
                  task.route.holder_daemon_id);
              if (batch_payload_ref_or.ok()) {
                batch_payload_ref = *batch_payload_ref_or;
              } else {
                transport_issue_status = batch_payload_ref_or.status();
              }
            }
            if (!batch_payload_ref.has_value()) {
              LOG(WARNING) << "batch_put_if_absent_from_region pack fallback to payload_ref: "
                           << transport_issue_status;
              continue;
            }

            v2::BatchPayloadTransport transport;
            const std::string transport_id = absl::StrCat("batch-transport-", batch_transports.size() + 1);
            transport.set_transport_id(transport_id);
            transport.mutable_manifest()->CopyFrom(pack.manifest);
            if (use_communicator_transport && communicator_export.has_value()) {
              if (local_producer_endpoint.available && local_producer_endpoint.p2p_port != 0) {
                auto* communicator_source = transport.mutable_communicator_source();
                communicator_source->set_batch_payload_ref(*batch_payload_ref);
                communicator_source->set_protocol_version(
                    d_.payload_transport_broker.batch_transport_protocol_version());
                communicator_source->set_producer_daemon_id(local_daemon_id);
                communicator_source->set_consumer_daemon_id(task.route.holder_daemon_id);
                communicator_source->set_producer_host(local_producer_endpoint.node_address);
                communicator_source->set_producer_port(local_producer_endpoint.p2p_port);
                for (const auto& remote_memory_key : communicator_export->remote_memory_keys) {
                  communicator_source->add_remote_memory_keys(remote_memory_key);
                }
                for (const auto buffer_size : communicator_export->buffer_sizes) {
                  communicator_source->add_buffer_sizes(buffer_size);
                }
                if (!local_producer_endpoint.node_id.empty()) {
                  communicator_source->set_remote_endpoint_id(
                      store::components::derive_endpoint_id(
                          local_producer_endpoint.node_id, common::memory::MemoryLocation::CPU, /*device_id=*/0));
                }
                const auto consumer_endpoint_it = prebuilt_remote_cpu_endpoint_ids.find(task.route.holder_daemon_id);
                if (consumer_endpoint_it != prebuilt_remote_cpu_endpoint_ids.end()) {
                  communicator_source->set_local_endpoint_id_hint(consumer_endpoint_it->second);
                }
                communicator_source->set_memory_location(v2::BATCH_PAYLOAD_MEMORY_LOCATION_HOST);
                communicator_source->set_total_payload_bytes(pack.manifest.total_size());
              } else {
                auto* grpc_chunk_ref = transport.mutable_grpc_chunk_ref();
                grpc_chunk_ref->set_batch_payload_ref(*batch_payload_ref);
                grpc_chunk_ref->set_protocol_version(1);
                emitted_communicator_transport = false;
              }
            } else {
              auto* grpc_chunk_ref = transport.mutable_grpc_chunk_ref();
              grpc_chunk_ref->set_batch_payload_ref(*batch_payload_ref);
              grpc_chunk_ref->set_protocol_version(
                  std::min<std::uint32_t>(1, d_.payload_transport_broker.batch_transport_protocol_version()));
            }

            ++task_stats.remote_batch_transport_count;
            if (emitted_communicator_transport) {
              ++task_stats.remote_batch_transport_communicator_count;
            } else {
              ++task_stats.remote_batch_transport_grpc_count;
            }
            task_stats.remote_batch_transport_item_count += pack.source_indices.size();
            task_stats.remote_batch_transport_bytes += pack.manifest.total_size();

            LOG(INFO) << "byte_artifact.batch_put_if_absent_from_region_transport_emit"
                      << " operation_id=" << operation_id << " shard_id=" << task.shard_id
                      << " holder_daemon_id=" << task.route.holder_daemon_id << " transport_id=" << transport_id
                      << " kind=" << (emitted_communicator_transport ? "communicator_source" : "grpc_chunk_ref")
                      << " source_realization_mode=staged_slab"
                      << " item_count=" << pack.source_indices.size()
                      << " payload_bytes=" << pack.manifest.total_size();

            for (std::size_t pack_index = 0; pack_index < pack.source_indices.size(); ++pack_index) {
              auto slice = pack.slices[pack_index];
              slice.set_transport_id(transport_id);
              batch_slice_by_outcome_index.emplace(
                  batch_entry_outcome_indices[pack.source_indices[pack_index]], std::move(slice));
            }
            batch_transports.push_back(std::move(transport));
          }
        }
      }
    }

    absl::flat_hash_map<std::string, std::string> payload_ref_by_artifact;
    payload_ref_by_artifact.reserve(task.batch.items.size());
    std::optional<std::string> payload_ref_error_message;
    for (const auto& pending : task.batch.items) {
      if (!pending.body_handle.has_value()) {
        continue;
      }
      if (batch_slice_by_outcome_index.find(pending.outcome_index) != batch_slice_by_outcome_index.end()) {
        continue;
      }
      if (!pending.descriptor.has_value()) {
        record_outcome(
            pending.outcome_index,
            make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "staged descriptor is missing"));
        payload_ref_error_message = "staged descriptor is missing";
        break;
      }
      auto payload_ref_or = d_.payload_transport_broker.issue_payload_ref(
          pending.artifact_id,
          *pending.body_handle,
          *pending.descriptor,
          pending.backing_identity,
          pending.body_handle->binding_generation(),
          tensorcast::common::v1::PAYLOAD_REF_DIRECTION_PUT,
          operation_id);
      if (!payload_ref_or.ok()) {
        record_outcome(
            pending.outcome_index,
            make_outcome(
                pending.artifact_id,
                batch_item_status_from_absl_status(payload_ref_or.status()),
                std::string(payload_ref_or.status().message())));
        payload_ref_error_message = std::string(payload_ref_or.status().message());
        break;
      }
      payload_ref_by_artifact.emplace(pending.artifact_id, *payload_ref_or);
      ++task_stats.remote_payload_ref_count;
    }
    if (payload_ref_error_message.has_value()) {
      for (const auto& pending : task.batch.items) {
        if (pending.body_handle.has_value() && !has_outcome(pending.outcome_index)) {
          record_outcome(
              pending.outcome_index,
              make_outcome(pending.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, *payload_ref_error_message));
        }
      }
    }

    if (payload_ref_by_artifact.empty() && batch_slice_by_outcome_index.empty()) {
      for (const auto& pending : task.batch.items) {
        if (pending.body_handle.has_value()) {
          (void)pending.body_handle->retire();
        }
      }
      return finalize_task();
    }

    PreparedRemoteHomeBatchPut prepared_remote_batch;
    prepared_remote_batch.shard_id = task.shard_id;
    prepared_remote_batch.holder_daemon_id = task.route.holder_daemon_id;
    prepared_remote_batch.lease_generation = task.route.lease_generation;
    prepared_remote_batch.home_req.mutable_fence()->set_shard_id(task.shard_id);
    prepared_remote_batch.home_req.mutable_fence()->set_lease_generation(task.route.lease_generation);
    prepared_remote_batch.home_req.mutable_fence()->set_holder_daemon_id(task.route.holder_daemon_id);
    prepared_remote_batch.home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
    if (ttl_ms.has_value()) {
      prepared_remote_batch.home_req.set_ttl_ms(*ttl_ms);
    }
    if (req.has_operation_id()) {
      prepared_remote_batch.home_req.set_operation_id(req.operation_id());
    }
    prepared_remote_batch.home_req.set_requester_daemon_id(local_daemon_id);
    for (const auto& transport : batch_transports) {
      prepared_remote_batch.home_req.add_batch_transports()->CopyFrom(transport);
    }
    prepared_remote_batch.outcome_slots.reserve(task.batch.items.size());
    prepared_remote_batch.retire_handles.reserve(task.batch.items.size());
    for (const auto& pending : task.batch.items) {
      const auto batch_slice_it = batch_slice_by_outcome_index.find(pending.outcome_index);
      const bool has_batch_slice = batch_slice_it != batch_slice_by_outcome_index.end();
      if (!pending.body_handle.has_value() && !has_batch_slice) {
        continue;
      }
      if (pending.body_handle.has_value()) {
        prepared_remote_batch.retire_handles.push_back(*pending.body_handle);
      }
      auto* dst = prepared_remote_batch.home_req.add_items();
      dst->set_artifact_id(pending.artifact_id);
      dst->mutable_invariant()->CopyFrom(pending.invariant);
      if (has_batch_slice) {
        dst->mutable_batch_payload_slice()->CopyFrom(batch_slice_it->second);
      }
      const auto payload_ref_it = payload_ref_by_artifact.find(pending.artifact_id);
      if (payload_ref_it != payload_ref_by_artifact.end()) {
        dst->set_payload_ref(payload_ref_it->second);
      } else if (!has_batch_slice) {
        prepared_remote_batch.home_req.mutable_items()->RemoveLast();
        continue;
      }
      prepared_remote_batch.outcome_slots.push_back(
          RemoteHomePutOutcomeSlot{
              .artifact_id = pending.artifact_id,
              .outcome_index = pending.outcome_index,
          });
    }
    if (prepared_remote_batch.home_req.items_size() == 0) {
      retire_remote_home_batch_handles(&prepared_remote_batch);
      return finalize_task();
    }

    const auto dispatch_remote_home_batch = [&](PreparedRemoteHomeBatchPut request) -> HomeBatchPutRpcResult {
      auto channel_or = resolve_remote_channel_for_task(request.holder_daemon_id);
      if (!channel_or.ok()) {
        return make_home_batch_put_result(
            std::move(request),
            grpc::Status(StatusCode::UNAVAILABLE, "home daemon address unavailable"),
            "home daemon address unavailable");
      }
      HomeBatchPutRpcResult result;
      result.request = std::move(request);
      auto stub = v2::StoreDaemonService::NewStub(*channel_or);
      grpc::ClientContext client_ctx;
      client_ctx.set_deadline(std::chrono::system_clock::now() + inter_daemon_home_rpc_timeout(options_));
      const absl::Time rpc_started_at = absl::Now();
      result.status = stub->HomeBatchPutIfAbsent(&client_ctx, result.request.home_req, &result.home_resp);
      result.rpc_elapsed = absl::Now() - rpc_started_at;
      return result;
    };

    for (;;) {
      auto rpc_result = dispatch_remote_home_batch(std::move(prepared_remote_batch));
      task_stats.remote_home_rpc_elapsed += rpc_result.rpc_elapsed;
      LOG(INFO) << "byte_artifact.batch_put_if_absent_from_region_home_rpc_result"
                << " operation_id=" << operation_id << " shard_id=" << rpc_result.request.shard_id
                << " holder_daemon_id=" << rpc_result.request.holder_daemon_id
                << " requested_items=" << rpc_result.request.home_req.items_size()
                << " attempt=" << rpc_result.request.attempt
                << " rpc_ms=" << absl::ToDoubleMilliseconds(rpc_result.rpc_elapsed)
                << " status_ok=" << rpc_result.status.ok() << " status_code=" << rpc_result.status.error_code()
                << " status_message=" << rpc_result.status.error_message();

      if (!rpc_result.status.ok()) {
        for (const auto& slot : rpc_result.request.outcome_slots) {
          record_outcome(
              slot.outcome_index,
              make_outcome(slot.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, rpc_result.status.error_message()));
        }
        retire_remote_home_batch_handles(&rpc_result.request);
        return finalize_task();
      }

      bool needs_redirect_retry = false;
      if (rpc_result.home_resp.has_redirect() &&
          rpc_result.home_resp.redirect().shard_id() == rpc_result.request.shard_id &&
          rpc_result.home_resp.redirect().lease_generation() != 0 &&
          !rpc_result.home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& outcome : rpc_result.home_resp.outcomes()) {
          if (outcome.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }

      if (needs_redirect_retry && rpc_result.request.attempt == 0) {
        const auto refreshed = d_.route_resolver.refresh_route_from_redirect(
            rpc_result.request.shard_id, rpc_result.home_resp.redirect(), absl::Now());
        if (!refreshed.ok) {
          for (const auto& slot : rpc_result.request.outcome_slots) {
            record_outcome(
                slot.outcome_index,
                make_outcome(slot.artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message));
          }
          retire_remote_home_batch_handles(&rpc_result.request);
          return finalize_task();
        }
        rpc_result.request.holder_daemon_id = refreshed.holder_daemon_id;
        rpc_result.request.lease_generation = refreshed.lease_generation;
        rpc_result.request.attempt = 1;
        rpc_result.request.home_req.mutable_fence()->set_holder_daemon_id(refreshed.holder_daemon_id);
        rpc_result.request.home_req.mutable_fence()->set_lease_generation(refreshed.lease_generation);
        prepared_remote_batch = std::move(rpc_result.request);
        continue;
      }

      absl::flat_hash_map<std::string, const v2::BatchItemOutcome*> by_id;
      by_id.reserve(rpc_result.home_resp.outcomes_size());
      for (const auto& out : rpc_result.home_resp.outcomes()) {
        by_id.emplace(out.artifact_id(), &out);
      }
      for (const auto& slot : rpc_result.request.outcome_slots) {
        const auto out_it = by_id.find(slot.artifact_id);
        if (out_it == by_id.end()) {
          record_outcome(
              slot.outcome_index,
              make_outcome(slot.artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing home outcome"));
          continue;
        }
        v2::BatchItemStatus status = out_it->second->status();
        if (status == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
          status = v2::BATCH_ITEM_STATUS_UNAVAILABLE;
        }
        record_outcome(slot.outcome_index, make_outcome(slot.artifact_id, status, out_it->second->message()));
      }
      retire_remote_home_batch_handles(&rpc_result.request);
      return finalize_task();
    }
  };

  const auto merge_stats = [&](const BatchPutInstrumentation& delta) {
    stats.local_home_batch_count += delta.local_home_batch_count;
    stats.remote_home_batch_count += delta.remote_home_batch_count;
    stats.local_home_item_count += delta.local_home_item_count;
    stats.remote_home_item_count += delta.remote_home_item_count;
    stats.local_stage_item_count += delta.local_stage_item_count;
    stats.remote_stage_item_count += delta.remote_stage_item_count;
    stats.remote_payload_ref_count += delta.remote_payload_ref_count;
    stats.remote_batch_transport_count += delta.remote_batch_transport_count;
    stats.remote_batch_transport_communicator_count += delta.remote_batch_transport_communicator_count;
    stats.remote_batch_transport_grpc_count += delta.remote_batch_transport_grpc_count;
    stats.remote_batch_transport_item_count += delta.remote_batch_transport_item_count;
    stats.remote_batch_transport_bytes += delta.remote_batch_transport_bytes;
    stats.remote_batch_pack_count += delta.remote_batch_pack_count;
    stats.remote_batch_pack_item_count += delta.remote_batch_pack_item_count;
    stats.remote_batch_pack_bytes += delta.remote_batch_pack_bytes;
    stats.remote_batch_segmented_region_export_count += delta.remote_batch_segmented_region_export_count;
    stats.remote_batch_segmented_region_export_item_count += delta.remote_batch_segmented_region_export_item_count;
    stats.remote_batch_segmented_region_export_bytes += delta.remote_batch_segmented_region_export_bytes;
    stats.local_stage_elapsed += delta.local_stage_elapsed;
    stats.remote_stage_elapsed += delta.remote_stage_elapsed;
    stats.remote_batch_pack_elapsed += delta.remote_batch_pack_elapsed;
    stats.remote_batch_segmented_region_export_elapsed += delta.remote_batch_segmented_region_export_elapsed;
    stats.local_home_apply_elapsed += delta.local_home_apply_elapsed;
    stats.remote_home_rpc_elapsed += delta.remote_home_rpc_elapsed;
  };

  const std::size_t max_parallel_put_shards = std::max<std::size_t>(
      1, std::min<std::size_t>(shard_tasks.size(), static_cast<std::size_t>(batch_get_apply_threads_)));

  for (std::size_t offset = 0; offset < shard_tasks.size(); offset += max_parallel_put_shards) {
    const std::size_t limit = std::min<std::size_t>(offset + max_parallel_put_shards, shard_tasks.size());
    std::vector<folly::SemiFuture<PutShardTaskResult>> futures;
    futures.reserve(limit - offset);
    for (std::size_t index = offset; index < limit; ++index) {
      futures.push_back(
          folly::via(
              folly::getKeepAliveToken(*batch_get_apply_pool_),
              [task = std::move(shard_tasks[index]), &run_put_shard_task]() mutable {
                return run_put_shard_task(std::move(task));
              })
              .semi());
    }

    for (auto& future : futures) {
      try {
        auto shard_result = std::move(future).get();
        merge_stats(shard_result.stats);
        for (auto& [outcome_index, outcome] : shard_result.outcomes) {
          *resp.mutable_outcomes(outcome_index) = std::move(outcome);
        }
      } catch (const std::exception& ex) {
        LOG(ERROR) << "PutShardTask failed: " << ex.what();
      } catch (...) {
        LOG(ERROR) << "PutShardTask failed with unknown exception";
      }
    }
  }

  for (int index = 0; index < resp.outcomes_size(); ++index) {
    if (resp.outcomes(index).status() != v2::BATCH_ITEM_STATUS_UNSPECIFIED) {
      continue;
    }
    *resp.mutable_outcomes(index) = make_outcome(
        resp.outcomes(index).artifact_id(),
        v2::BATCH_ITEM_STATUS_INTERNAL_ERROR,
        "PutShardTask failed before producing an outcome");
  }

  attach_slot_tokens_to_outcomes(requested_slot_tokens, resp.mutable_outcomes());
  VLOG(2) << "byte_artifact.batch_put_if_absent_from_region_summary"
          << " operation_id=" << operation_id << " items=" << req.items_size() << " shard_count=" << shard_tasks.size()
          << " max_parallel_put_shards=" << max_parallel_put_shards
          << " local_home_batches=" << stats.local_home_batch_count
          << " remote_home_batches=" << stats.remote_home_batch_count
          << " local_home_items=" << stats.local_home_item_count
          << " remote_home_items=" << stats.remote_home_item_count
          << " local_stage_items=" << stats.local_stage_item_count
          << " remote_stage_items=" << stats.remote_stage_item_count
          << " remote_payload_refs=" << stats.remote_payload_ref_count
          << " remote_batch_transports=" << stats.remote_batch_transport_count
          << " remote_batch_transport_communicator=" << stats.remote_batch_transport_communicator_count
          << " remote_batch_transport_grpc=" << stats.remote_batch_transport_grpc_count
          << " remote_batch_transport_items=" << stats.remote_batch_transport_item_count
          << " remote_batch_transport_bytes=" << stats.remote_batch_transport_bytes
          << " remote_batch_pack_count=" << stats.remote_batch_pack_count
          << " remote_batch_pack_items=" << stats.remote_batch_pack_item_count
          << " remote_batch_pack_bytes=" << stats.remote_batch_pack_bytes
          << " remote_batch_segmented_region_export_count=" << stats.remote_batch_segmented_region_export_count
          << " remote_batch_segmented_region_export_items=" << stats.remote_batch_segmented_region_export_item_count
          << " remote_batch_segmented_region_export_bytes=" << stats.remote_batch_segmented_region_export_bytes
          << " local_stage_ms=" << absl::ToDoubleMilliseconds(stats.local_stage_elapsed)
          << " remote_stage_ms=" << absl::ToDoubleMilliseconds(stats.remote_stage_elapsed)
          << " remote_batch_pack_ms=" << absl::ToDoubleMilliseconds(stats.remote_batch_pack_elapsed)
          << " remote_batch_segmented_region_export_ms="
          << absl::ToDoubleMilliseconds(stats.remote_batch_segmented_region_export_elapsed)
          << " local_home_apply_ms=" << absl::ToDoubleMilliseconds(stats.local_home_apply_elapsed)
          << " remote_home_rpc_ms=" << absl::ToDoubleMilliseconds(stats.remote_home_rpc_elapsed);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status ByteArtifactController::batch_touch_ttl(
    RpcContext& rctx,
    const v2::BatchTouchTtlRequest& req,
    v2::BatchTouchTtlResponse& resp) {
  const bool local_peer = is_loopback_grpc_peer(rctx.server_context().peer());
  if (!local_peer && !options_.gateway_ingress_enabled) {
    return {StatusCode::PERMISSION_DENIED, "BatchTouchTtl requires gateway ingress on non-local peers"};
  }
  if (req.ttl_ms() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "ttl_ms must be > 0"};
  }

  const absl::Time now = absl::Now();
  const std::string local_daemon_id = d_.route_resolver.local_daemon_id();
  absl::flat_hash_map<std::uint64_t, std::vector<std::pair<std::string, int>>> shard_batches;
  shard_batches.reserve(static_cast<size_t>(req.artifact_ids_size()));

  for (int i = 0; i < req.artifact_ids_size(); ++i) {
    const auto& artifact_id = req.artifact_ids(i);
    auto* outcome = resp.add_outcomes();
    outcome->set_artifact_id(artifact_id);
    if (artifact_id.empty()) {
      *outcome = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, "artifact_id is required");
      continue;
    }
    const auto artifact_id_st = byte_artifact_runtime().validate_artifact_id_for_field(artifact_id, "artifact_id");
    if (!artifact_id_st.ok()) {
      *outcome =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT, std::string(artifact_id_st.message()));
      continue;
    }
    auto shard_or = byte_artifact_runtime().shard_id_for_artifact(artifact_id, options_.routing.shard_count);
    if (!shard_or.ok()) {
      *outcome =
          make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, std::string(shard_or.status().message()));
      continue;
    }
    shard_batches[*shard_or].push_back({artifact_id, i});
  }

  const bool local_only = (d_.global_store_client == nullptr);
  if (local_only) {
    for (const auto& [shard_id, items] : shard_batches) {
      std::vector<std::string> artifact_ids;
      artifact_ids.reserve(items.size());
      for (const auto& [artifact_id, _] : items) {
        artifact_ids.push_back(artifact_id);
      }
      const auto outcomes = authority_service_.batch_touch_ttl(
          artifact_ids,
          ByteArtifactAuthorityService::Context{
              .shard_id = shard_id,
              .lease_generation = 1,
              .routing_epoch = options_.routing.routing_epoch,
              .shard_count = options_.routing.shard_count,
              .now = now,
          },
          req.ttl_ms());
      for (size_t idx = 0; idx < items.size(); ++idx) {
        *resp.mutable_outcomes(items[idx].second) = outcomes.at(idx);
      }
    }
    rctx.mark_success();
    return Status::OK;
  }

  std::vector<std::uint64_t> shard_ids;
  shard_ids.reserve(shard_batches.size());
  for (const auto& [shard_id, /*items*/ _] : shard_batches) {
    shard_ids.push_back(shard_id);
  }
  auto routes = d_.route_resolver.resolve_routes(absl::MakeSpan(shard_ids), now);

  std::vector<std::string> remote_daemon_ids;
  remote_daemon_ids.reserve(routes.size());
  for (const auto& [shard_id, route] : routes) {
    if (route.ok && route.holder_daemon_id != local_daemon_id) {
      remote_daemon_ids.push_back(route.holder_daemon_id);
    }
  }
  if (!remote_daemon_ids.empty()) {
    (void)d_.worker_directory_cache.warm_for_daemons(
        remote_daemon_ids, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
  }

  for (const auto& [shard_id, items] : shard_batches) {
    const auto route_it = routes.find(shard_id);
    if (route_it == routes.end() || !route_it->second.ok) {
      const std::string message = (route_it == routes.end() || route_it->second.message.empty())
          ? "routing lease unavailable"
          : route_it->second.message;
      for (const auto& [artifact_id, idx] : items) {
        *resp.mutable_outcomes(idx) = make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, message);
      }
      continue;
    }

    std::uint64_t lease_generation = route_it->second.lease_generation;
    std::string holder_daemon_id = route_it->second.holder_daemon_id;

    for (int attempt = 0; attempt < 2; ++attempt) {
      v2::HomeBatchTouchTtlResponse home_resp;
      grpc::Status home_status;
      if (holder_daemon_id == local_daemon_id) {
        v2::HomeBatchTouchTtlRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(local_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        home_req.set_ttl_ms(req.ttl_ms());
        for (const auto& [artifact_id, /*idx*/ _] : items) {
          home_req.add_artifact_ids(artifact_id);
        }
        grpc::ServerContext home_ctx;
        RpcContext home_rctx{"HomeBatchTouchTtl", home_ctx, rctx.allow_high_card_attrs()};
        home_status = home_batch_touch_ttl(home_rctx, home_req, home_resp);
      } else {
        auto address_or = d_.worker_directory_cache.resolve_daemon_address(
            holder_daemon_id, now, absl::Milliseconds(options_.routing.worker_directory_staleness_budget.count()));
        if (!address_or.ok()) {
          for (const auto& [artifact_id, idx] : items) {
            *resp.mutable_outcomes(idx) =
                make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, "home daemon address unavailable");
          }
          break;
        }
        auto channel = create_inter_daemon_channel(*address_or, d_.inter_daemon_channel_credentials);
        auto stub = v2::StoreDaemonService::NewStub(channel);
        grpc::ClientContext client_ctx;
        client_ctx.set_deadline(std::chrono::system_clock::now() + inter_daemon_home_rpc_timeout(options_));
        v2::HomeBatchTouchTtlRequest home_req;
        home_req.mutable_fence()->set_shard_id(shard_id);
        home_req.mutable_fence()->set_lease_generation(lease_generation);
        home_req.mutable_fence()->set_holder_daemon_id(holder_daemon_id);
        home_req.mutable_fence()->set_routing_epoch(options_.routing.routing_epoch);
        home_req.set_ttl_ms(req.ttl_ms());
        for (const auto& [artifact_id, /*idx*/ _] : items) {
          home_req.add_artifact_ids(artifact_id);
        }
        home_status = stub->HomeBatchTouchTtl(&client_ctx, home_req, &home_resp);
      }

      if (!home_status.ok()) {
        for (const auto& [artifact_id, idx] : items) {
          *resp.mutable_outcomes(idx) =
              make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, home_status.error_message());
        }
        break;
      }

      bool needs_redirect_retry = false;
      if (home_resp.has_redirect() && home_resp.redirect().shard_id() == shard_id &&
          home_resp.redirect().lease_generation() != 0 && !home_resp.redirect().holder_daemon_id().empty()) {
        for (const auto& outcome : home_resp.outcomes()) {
          if (outcome.status() == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
            needs_redirect_retry = true;
            break;
          }
        }
      }
      if (needs_redirect_retry && attempt == 0) {
        const auto& redirect = home_resp.redirect();
        const auto refreshed = d_.route_resolver.refresh_route_from_redirect(shard_id, redirect, now);
        if (!refreshed.ok) {
          for (const auto& [artifact_id, idx] : items) {
            *resp.mutable_outcomes(idx) =
                make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_UNAVAILABLE, refreshed.message);
          }
          break;
        }
        holder_daemon_id = refreshed.holder_daemon_id;
        lease_generation = refreshed.lease_generation;
        continue;
      }

      absl::flat_hash_map<std::string, const v2::BatchItemOutcome*> by_id;
      by_id.reserve(home_resp.outcomes_size());
      for (const auto& out : home_resp.outcomes()) {
        by_id.emplace(out.artifact_id(), &out);
      }
      for (const auto& [artifact_id, idx] : items) {
        const auto out_it = by_id.find(artifact_id);
        if (out_it == by_id.end()) {
          *resp.mutable_outcomes(idx) =
              make_outcome(artifact_id, v2::BATCH_ITEM_STATUS_INTERNAL_ERROR, "missing home outcome");
          continue;
        }
        v2::BatchItemStatus status = out_it->second->status();
        if (status == v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION) {
          status = v2::BATCH_ITEM_STATUS_UNAVAILABLE;
        }
        *resp.mutable_outcomes(idx) = make_outcome(artifact_id, status, out_it->second->message());
      }
      break;
    }
  }

  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
