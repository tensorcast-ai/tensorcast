// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/components/global_store_client.h"

namespace tensorcast::daemon::materialization_post_seal {

std::vector<uint8_t> compute_view_meta_digest(const store::components::ViewInfo& view);

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id);

} // namespace tensorcast::daemon::materialization_post_seal
