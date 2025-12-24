// Copyright (c) 2025, TensorCast Team.

#include "daemon/ipc_region_registry.h"

#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/time/clock.h"

using tensorcast::daemon::IpcRegionRegistry;

TEST_CASE("IpcRegionRegistry basic register/describe/unregister", "[daemon][region]") {
  IpcRegionRegistry reg(
      IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });

  // Invalid args rejected
  {
    IpcRegionRegistry::RegisterParams p;
    p.device_id = -1;
    p.owner_pid = 1234;
    p.size_bytes = 4096;
    p.ttl_ms = 1000;
    p.handle_bytes = "handle";
    auto st = reg.register_region(p);
    REQUIRE_FALSE(st.ok());
  }

  // Happy path
  IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = 4242;
  params.size_bytes = 1 << 20; // 1 MiB
  params.ttl_ms = 1000;
  params.session_id = "s1";
  params.region_name = "kv-slab";
  params.handle_bytes = std::string("fake-cuda-ipc-handle");

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());
  auto desc = *desc_or;
  REQUIRE_FALSE(desc.region_id.empty());
  REQUIRE(desc.device_id == 0);
  REQUIRE(desc.owner_pid == 4242);
  REQUIRE(desc.size_bytes == (1u << 20));
  REQUIRE(desc.ttl_ms > 0);

  // Describe returns same
  auto desc2_or = reg.describe(desc.region_id);
  REQUIRE(desc2_or.ok());
  REQUIRE(desc2_or->region_id == desc.region_id);

  // Acquire/Release updates refcount and blocks unregister while held
  auto acq_or = reg.acquire(desc.region_id, params.owner_pid);
  REQUIRE(acq_or.ok());
  auto unreg_or = reg.unregister_region(desc.region_id, params.owner_pid, /*force=*/false);
  REQUIRE_FALSE(unreg_or.ok()); // active references prevent normal unregister
  REQUIRE(reg.release(desc.region_id).ok());

  // Now unregister succeeds
  auto unreg2_or = reg.unregister_region(desc.region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unreg2_or.ok());
  REQUIRE(*unreg2_or);
}

TEST_CASE("IpcRegionRegistry TTL refresh and sweep", "[daemon][region]") {
  IpcRegionRegistry reg(
      IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(200),
      });

  IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = 11;
  params.size_bytes = 4096;
  params.ttl_ms = 100; // clamped by max_ttl but valid
  params.handle_bytes = "h";

  auto d_or = reg.register_region(params);
  REQUIRE(d_or.ok());
  const std::string region_id = d_or->region_id;

  // Refresh pushes expiry out
  REQUIRE(reg.refresh_ttl(region_id, 150));

  // Sweep before expiry -> nothing
  auto expired = reg.sweep_expired(absl::Now());
  REQUIRE(expired.empty());

  // Advance time well beyond TTL and sweep -> should expire
  absl::SleepFor(absl::Milliseconds(210));
  expired = reg.sweep_expired(absl::Now());
  REQUIRE(expired.size() == 1);
  REQUIRE(expired[0].region_id == region_id);
}

TEST_CASE("IpcRegionRegistry poison blocks acquire", "[daemon][region]") {
  IpcRegionRegistry reg(IpcRegionRegistry::Options{});
  IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = 7;
  params.size_bytes = 4096;
  params.ttl_ms = 1000;
  params.handle_bytes = "h";

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());
  const std::string region_id = desc_or->region_id;

  REQUIRE(reg.mark_poisoned(region_id).ok());
  REQUIRE(reg.is_poisoned(region_id));
  auto acq_or = reg.acquire(region_id, params.owner_pid);
  REQUIRE_FALSE(acq_or.ok());
}
