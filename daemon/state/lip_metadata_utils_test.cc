// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/lip_metadata_utils.h"

#include <catch2/catch_test_macros.hpp>

using tensorcast::daemon::build_canonical_index_from_metadata;
using tensorcast::daemon::LeaseSegMeta;
using tensorcast::daemon::RegisterStorageMeta;
using tensorcast::daemon::RegisterTensorAliasMeta;

TEST_CASE("BuildCanonicalIndexFromMetadata generates stable JSON", "[daemon][lip]") {
  std::vector<LeaseSegMeta> segments = {
      LeaseSegMeta{.storage_id = "s0", .storage_offset = 0, .artifact_offset = 0, .length = 1024},
  };
  std::vector<RegisterStorageMeta> storages = {
      RegisterStorageMeta{
          .storage_id = "s0",
          .device_id = 0,
          .handle_bytes = "handle-a",
          .storage_length = 1024,
          .mapping_base_offset = 0,
      },
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
  // Both aliases share a storage, so offsets/sizes mirror the storage itself.
  REQUIRE(
      *canon_or ==
      R"({"a_tensor":[0,1024,[128,1],[1,1],"torch.float32",0],"b_view":[0,1024,[128,1],[1,1],"torch.float32",256]})");
}

TEST_CASE("BuildCanonicalIndexFromMetadata handles multiple storages and views", "[daemon][lip]") {
  std::vector<LeaseSegMeta> segments = {
      LeaseSegMeta{.storage_id = "s0", .storage_offset = 0, .artifact_offset = 0, .length = 1024},
      LeaseSegMeta{.storage_id = "s1", .storage_offset = 0, .artifact_offset = 4096, .length = 2048},
  };
  std::vector<RegisterStorageMeta> storages = {
      RegisterStorageMeta{
          .storage_id = "s0",
          .device_id = 0,
          .handle_bytes = "handle-a",
          .storage_length = 1024,
          .mapping_base_offset = 0,
      },
      RegisterStorageMeta{
          .storage_id = "s1",
          .device_id = 0,
          .handle_bytes = "handle-b",
          .storage_length = 2048,
          .mapping_base_offset = 0,
      },
  };

  std::vector<RegisterTensorAliasMeta> aliases;
  RegisterTensorAliasMeta a_full;
  a_full.name = "a_full";
  a_full.storage_id = "s0";
  a_full.storage_offset = 0;
  a_full.logical_length = 1024;
  a_full.shape = {256};
  a_full.stride = {1};
  a_full.dtype = "torch.float32";
  aliases.push_back(a_full);

  RegisterTensorAliasMeta a_tail;
  a_tail.name = "a_tail";
  a_tail.storage_id = "s0";
  a_tail.storage_offset = 256;
  a_tail.logical_length = 512;
  a_tail.shape = {128};
  a_tail.stride = {1};
  a_tail.dtype = "torch.float32";
  aliases.push_back(a_tail);

  RegisterTensorAliasMeta b_full;
  b_full.name = "b_full";
  b_full.storage_id = "s1";
  b_full.storage_offset = 0;
  b_full.logical_length = 2048;
  b_full.shape = {512};
  b_full.stride = {1};
  b_full.dtype = "torch.float16";
  aliases.push_back(b_full);

  RegisterTensorAliasMeta b_view;
  b_view.name = "b_view";
  b_view.storage_id = "s1";
  b_view.storage_offset = 512;
  b_view.logical_length = 1024;
  b_view.shape = {256};
  b_view.stride = {1};
  b_view.dtype = "torch.float16";
  aliases.push_back(b_view);

  auto canon_or = build_canonical_index_from_metadata(segments, storages, aliases, /*device_id=*/0);
  REQUIRE(canon_or.ok());
  REQUIRE(
      *canon_or ==
      R"({"a_full":[0,1024,[256],[1],"torch.float32",0],"a_tail":[0,1024,[128],[1],"torch.float32",256],"b_full":[4096,2048,[512],[1],"torch.float16",0],"b_view":[4096,2048,[256],[1],"torch.float16",512]})");
}

TEST_CASE("BuildCanonicalIndexFromMetadata detects mismatched storage", "[daemon][lip]") {
  std::vector<LeaseSegMeta> segments = {
      LeaseSegMeta{.storage_id = "s1", .storage_offset = 0, .artifact_offset = 0, .length = 256},
  };
  std::vector<RegisterStorageMeta> storages = {
      RegisterStorageMeta{
          .storage_id = "s0",
          .device_id = 0,
          .handle_bytes = "other",
          .storage_length = 256,
          .mapping_base_offset = 0,
      },
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
