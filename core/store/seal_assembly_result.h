// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

namespace tensorcast::store {

struct SealAssemblyResult {
  std::string assembly_id;
  std::string sealed_artifact_id;
  std::string index_multihash;
  std::string data_multihash;
  std::string schema_version;
  std::string encoding;
  uint64_t total_size{0};
  bool already_sealed{false};
};

} // namespace tensorcast::store
