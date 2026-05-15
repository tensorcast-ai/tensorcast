// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/types.h"
#include "tensorcast/common/v1/common.pb.h"

namespace tensorcast::daemon {

struct PublicationSubjectKey {
  std::string value;

  bool operator==(const PublicationSubjectKey&) const = default;
};

template <typename H>
H AbslHashValue(H h, const PublicationSubjectKey& key) {
  return H::combine(std::move(h), key.value);
}

struct PublicationInstanceId {
  std::string value;

  bool operator==(const PublicationInstanceId&) const = default;
};

template <typename H>
H AbslHashValue(H h, const PublicationInstanceId& id) {
  return H::combine(std::move(h), id.value);
}

[[nodiscard]] PublicationSubjectKey build_publication_subject_key(
    const tensorcast::common::v1::ArtifactSelection& selection,
    const tensorcast::common::v1::ByteSpaceRef& byte_space,
    std::string_view target_layout_hash,
    std::string_view device_uuid);

class TargetPublicationRegistry {
 public:
  struct Options {
    size_t capacity{4096};
    absl::Duration ttl{absl::Minutes(5)};
  };

  struct Record {
    PublicationInstanceId publication_id;
    PublicationSubjectKey publication_subject_key;
    std::uint64_t subject_generation{0};
    std::string target_layout_hash;
    tensorcast::common::v1::ArtifactSelection selection;
    tensorcast::common::v1::ByteSpaceRef byte_space;
    std::string canonical_index_json;
    std::string index_key_hex;
    std::string device_uuid;
    int owner_pid{0};
    std::string request_operation_id;
    absl::Time expires_at{absl::InfinitePast()};
    std::string capability_id;
    std::uint64_t lease_id{0};
    std::optional<WorkflowBindingProjection> workflow_binding_projection;
    std::optional<WorkflowOutcomeProjection> replay_outcome_projection;
    WorkflowRecoveryClass workflow_recovery_class{WorkflowRecoveryClass::kEphemeralProcessLocal};
    std::vector<LeaseSegMeta> segments;
    std::vector<RegisterStorageMeta> storages;
  };

  struct GroupRealizationStaging {
    std::string transaction_id;
    std::string version_set_id;
    std::string part_id;
    std::string staging_token;
    std::uint64_t staging_epoch{0};
    bool publish_admitted{false};
  };

  struct StagedRecord {
    Record publication;
    GroupRealizationStaging staging;
  };

  explicit TargetPublicationRegistry(Options opts);

  [[nodiscard]] Record insert(Record record);
  [[nodiscard]] StagedRecord insert_staged(StagedRecord record);
  [[nodiscard]] std::optional<StagedRecord> lookup_staged(std::string_view publication_id, absl::Time now) const;
  [[nodiscard]] std::optional<StagedRecord> publish_staged(std::string_view publication_id);
  [[nodiscard]] size_t erase_staged_for_transaction(std::string_view transaction_id, size_t max_to_erase = 0);
  [[nodiscard]] std::optional<Record> lookup(std::string_view publication_id, absl::Time now, bool require_not_expired)
      const;
  [[nodiscard]] std::optional<Record> lookup_current_for_subject(
      std::string_view publication_subject_key,
      absl::Time now,
      bool require_not_expired) const;
  [[nodiscard]] bool is_current_for_subject(std::string_view publication_subject_key, std::string_view publication_id)
      const;
  void erase(std::string_view publication_id);
  void prune(absl::Time now);

 private:
  [[nodiscard]] std::uint64_t assign_subject_generation_locked(const Record& record) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void prune_locked(absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Options opts_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Record> records_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, StagedRecord> staged_records_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> latest_by_subject_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
