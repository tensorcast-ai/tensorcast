// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "daemon/state/types.h"
#include "tensorcast/common/v1/common.pb.h"

namespace tensorcast::daemon {

class TargetPublicationRegistry {
 public:
  struct Options {
    size_t capacity{4096};
    absl::Duration ttl{absl::Minutes(5)};
  };

  struct Record {
    std::string publication_id;
    std::string publication_key;
    std::string target_layout_hash;
    tensorcast::common::v1::ArtifactSelection selection;
    tensorcast::common::v1::ByteSpaceRef byte_space;
    std::string canonical_index_json;
    std::string index_key_hex;
    std::string device_uuid;
    int owner_pid{0};
    std::string operation_id;
    absl::Time expires_at{absl::InfinitePast()};
    std::string capability_id;
    std::uint64_t lease_id{0};
    std::vector<LeaseSegMeta> segments;
    std::vector<RegisterStorageMeta> storages;
  };

  explicit TargetPublicationRegistry(Options opts);

  [[nodiscard]] Record insert(Record record);
  [[nodiscard]] std::optional<Record> lookup(std::string_view publication_id, absl::Time now, bool require_not_expired)
      const;
  [[nodiscard]] bool is_current_for_target(std::string_view publication_key, std::string_view publication_id) const;
  void erase(std::string_view publication_id);
  void prune(absl::Time now);

 private:
  void prune_locked(absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Options opts_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, Record> records_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> latest_by_target_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
