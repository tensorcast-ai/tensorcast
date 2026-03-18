// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_LLDP_SOURCE_H_
#define CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_LLDP_SOURCE_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

namespace tensorcast::communicator::topology::discovery {

struct LldpNicRecord {
  std::string if_name;
  std::string pci_bdf;
  std::string nic_name;
  int rail_id = 0;

  bool operator==(const LldpNicRecord& other) const = default;
};

struct LldpParseOptions {
  // strict=true: malformed lines fail-fast.
  // strict=false: malformed lines are skipped with warnings.
  bool strict = false;
};

absl::StatusOr<std::vector<LldpNicRecord>> load_lldp_records(
    const std::string& file_path,
    LldpParseOptions options = {});

absl::StatusOr<absl::flat_hash_map<std::string, LldpNicRecord>> load_lldp_records_by_nic(
    const std::string& file_path,
    LldpParseOptions options = {});

} // namespace tensorcast::communicator::topology::discovery

#endif // CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_LLDP_SOURCE_H_
