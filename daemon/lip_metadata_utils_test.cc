// Copyright (c) 2025, TensorCast Team.

#include "daemon/lip_metadata_utils.h"

#include <catch2/catch_test_macros.hpp>

using tensorcast::daemon::build_canonical_index_from_metadata;
using tensorcast::daemon::LeaseSegMeta;
using tensorcast::daemon::RegisterStorageMeta;
using tensorcast::daemon::RegisterTensorAliasMeta;

TEST_CASE("BuildCanonicalIndexFromMetadata generates stable JSON", "[daemon][lip]") {
  std::vector<LeaseSegMeta> segments = {
      LeaseSegMeta{.device_id = 0, .handle_bytes = "handle-a", .base_offset = 0, .length = 1024, .dst_offset = 0},
  };
  std::vector<RegisterStorageMeta> storages = {
      RegisterStorageMeta{.storage_id = "s0", .device_id = 0, .handle_bytes = "handle-a", .storage_length = 1024},
  };
  std::vector<RegisterTensorAliasMeta> aliases;
  RegisterTensorAliasMeta a;
  a.name = "a_tensor";
  a.storage_id = "s0";
  a.storage_offset = 0;
  a.logical_length = 512;
  a.shape = {128, 1};
  a.stride = {1, 1};
  a.dtype = "torch.float32";
  aliases.push_back(a);

  RegisterTensorAliasMeta b;
  b.name = "b_view";
  b.storage_id = "s0";
  b.storage_offset = 256;
  b.logical_length = 256;
  b.shape = {128, 1};
  b.stride = {1, 1};
  b.dtype = "torch.float32";
  aliases.push_back(b);

  auto canon_or = build_canonical_index_from_metadata(segments, storages, aliases, /*device_id=*/0);
  REQUIRE(canon_or.ok());
  REQUIRE(
      *canon_or ==
      R"({"a_tensor":[0,512,[128,1],[1,1],"torch.float32",0],"b_view":[256,256,[128,1],[1,1],"torch.float32",256]})");
}

TEST_CASE("BuildCanonicalIndexFromMetadata detects mismatched storage", "[daemon][lip]") {
  std::vector<LeaseSegMeta> segments = {
      LeaseSegMeta{.device_id = 0, .handle_bytes = "seg", .base_offset = 0, .length = 256, .dst_offset = 0},
  };
  std::vector<RegisterStorageMeta> storages = {
      RegisterStorageMeta{.storage_id = "s0", .device_id = 0, .handle_bytes = "other", .storage_length = 256},
  };
  std::vector<RegisterTensorAliasMeta> aliases = {
      RegisterTensorAliasMeta{
          .name = "tensor",
          .storage_id = "s0",
          .storage_offset = 0,
          .logical_length = 256,
          .shape = {64, 1},
          .stride = {1, 1},
          .dtype = "torch.float32",
      },
  };
  auto canon_or = build_canonical_index_from_metadata(segments, storages, aliases, /*device_id=*/0);
  REQUIRE_FALSE(canon_or.ok());
}
