// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "communicator_config.pb.h"

namespace tensorcast::communicator {

// Load communicator config from a YAML or JSON file and normalize defaults.
absl::StatusOr<tensorcast::communicator::CommunicatorConfig> LoadCommunicatorConfigFromFile(const std::string& path);

// Apply runtime defaults to a partially-filled proto (proto3 has no field defaults).
// Note: Only non-boolean numeric/time fields are defaulted here to avoid
// overriding explicit false booleans (proto3 lacks presence for scalars).
void NormalizeDefaults(tensorcast::communicator::CommunicatorConfig* cfg);

} // namespace tensorcast::communicator
