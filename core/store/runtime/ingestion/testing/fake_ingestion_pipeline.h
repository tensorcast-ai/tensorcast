// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/runtime/pipeline/ingestion_pipeline.h"
#include "core/store/runtime/ingestion_events.h"

namespace tensorcast::store::runtime::ingestion::testing {

class FakeIngestionPipeline final : public materialization::runtime::pipeline::IngestionPipeline {
 public:
  using Pipeline = materialization::runtime::pipeline::IngestionPipeline;

  explicit FakeIngestionPipeline(const Pipeline::Config& config) : Pipeline(config) {}

  struct Invocation {
    std::string artifact_identifier;
    loading::ReplicaTarget target;
    loading::MaterializeHints hints;
    bool publish_to_global_store;
    std::string request_id;
    std::string publish_context_id;
  };

  void set_next_disk_result(
      const absl::Status& status,
      std::optional<IngestionResultEvent> event_override = std::nullopt) {
    configure_next_result(next_disk_result_, absl::StatusOr<loading::ReplicaHandle>(status), std::move(event_override));
  }

  void set_next_disk_result(
      loading::ReplicaHandle handle,
      std::optional<IngestionResultEvent> event_override = std::nullopt) {
    configure_next_result(
        next_disk_result_, absl::StatusOr<loading::ReplicaHandle>(std::move(handle)), std::move(event_override));
  }

  void set_next_p2p_result(
      const absl::Status& status,
      std::optional<IngestionResultEvent> event_override = std::nullopt) {
    configure_next_result(next_p2p_result_, absl::StatusOr<loading::ReplicaHandle>(status), std::move(event_override));
  }

  void set_next_p2p_result(
      loading::ReplicaHandle handle,
      std::optional<IngestionResultEvent> event_override = std::nullopt) {
    configure_next_result(
        next_p2p_result_, absl::StatusOr<loading::ReplicaHandle>(std::move(handle)), std::move(event_override));
  }

  absl::StatusOr<loading::ReplicaHandle> ingest_from_disk(
      const std::string& artifact_identifier,
      const loading::DiskSource& /*source*/,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store,
      IngestionResultEvent* event_out,
      std::string request_id,
      std::string publish_context_id) override {
    disk_invocations_.push_back(
        Invocation{
            .artifact_identifier = artifact_identifier,
            .target = target,
            .hints = hints,
            .publish_to_global_store = publish_to_global_store,
            .request_id = std::move(request_id),
            .publish_context_id = std::move(publish_context_id),
        });
    return consume_result(next_disk_result_, event_out);
  }

  absl::StatusOr<loading::ReplicaHandle> ingest_from_p2p(
      const std::string& artifact_identifier,
      const P2PSource& /*source*/,
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints,
      bool publish_to_global_store,
      IngestionResultEvent* event_out,
      std::string request_id,
      std::string publish_context_id) override {
    p2p_invocations_.push_back(
        Invocation{
            .artifact_identifier = artifact_identifier,
            .target = target,
            .hints = hints,
            .publish_to_global_store = publish_to_global_store,
            .request_id = std::move(request_id),
            .publish_context_id = std::move(publish_context_id),
        });
    return consume_result(next_p2p_result_, event_out);
  }

  const std::vector<Invocation>& disk_invocations() const {
    return disk_invocations_;
  }

  const std::vector<Invocation>& p2p_invocations() const {
    return p2p_invocations_;
  }

 private:
  struct ResultSlot {
    absl::StatusOr<loading::ReplicaHandle> result;
    std::optional<IngestionResultEvent> event_override;
  };

  using OptionalResultSlot = std::optional<ResultSlot>;

  static absl::Status failed_configuration_status() {
    return absl::FailedPreconditionError("fake ingestion pipeline result not configured");
  }

  void configure_next_result(
      OptionalResultSlot& slot,
      absl::StatusOr<loading::ReplicaHandle> result,
      std::optional<IngestionResultEvent> event_override) {
    ResultSlot bucket{
        .result = std::move(result),
        .event_override = std::move(event_override),
    };
    slot = std::move(bucket);
  }

  absl::StatusOr<loading::ReplicaHandle> consume_result(OptionalResultSlot& slot, IngestionResultEvent* event_out) {
    if (!slot.has_value()) {
      return failed_configuration_status();
    }
    if (event_out != nullptr && slot->event_override.has_value()) {
      *event_out = *slot->event_override;
    }
    absl::StatusOr<loading::ReplicaHandle> result = std::move(slot->result);
    slot.reset();
    return result;
  }

  OptionalResultSlot next_disk_result_;
  OptionalResultSlot next_p2p_result_;
  std::vector<Invocation> disk_invocations_;
  std::vector<Invocation> p2p_invocations_;
};

} // namespace tensorcast::store::runtime::ingestion::testing
