// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/ipc_region_registry.h"

#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
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

TEST_CASE("IpcRegionRegistry host shared register/describe/unregister", "[daemon][region]") {
  IpcRegionRegistry reg(
      IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });

  IpcRegionRegistry::RegisterParams params;
  params.memory_kind = IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = 5150;
  params.size_bytes = 2 << 20;
  params.ttl_ms = 1000;
  params.session_id = "host-session";
  params.region_name = "host-slab";
  params.daemon_managed = true;
  params.host_region_class = IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());
  REQUIRE(desc_or->memory_kind == IpcRegionRegistry::MemoryKind::kHostShared);
  REQUIRE(desc_or->device_id == -1);
  REQUIRE(desc_or->daemon_managed);
  REQUIRE(desc_or->host_region_class == IpcRegionRegistry::HostRegionClass::kScratch);

  auto desc2_or = reg.describe(desc_or->region_id);
  REQUIRE(desc2_or.ok());
  REQUIRE(desc2_or->memory_kind == IpcRegionRegistry::MemoryKind::kHostShared);

  auto unreg_or = reg.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unreg_or.ok());
  REQUIRE(*unreg_or);
}

TEST_CASE("IpcRegionRegistry pre-cleanup callback observes retiring host shared regions", "[daemon][region]") {
  IpcRegionRegistry reg(
      IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });

  IpcRegionRegistry::RegisterParams params;
  params.memory_kind = IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = 9123;
  params.size_bytes = 1 << 20;
  params.ttl_ms = 0;
  params.daemon_managed = true;
  params.host_region_class = IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());

  int callback_count = 0;
  reg.set_pre_cleanup_callback([&](const IpcRegionRegistry::RegionDescriptor& desc) {
    ++callback_count;
    CHECK(desc.region_id == desc_or->region_id);
    auto acquire_or = reg.acquire(desc.region_id, params.owner_pid);
    REQUIRE_FALSE(acquire_or.ok());
    CHECK(acquire_or.status().message() == "region is retiring");
  });

  auto unreg_or = reg.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true);
  REQUIRE(unreg_or.ok());
  REQUIRE(*unreg_or);
  CHECK(callback_count == 1);
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

TEST_CASE("IpcRegionRegistry ttl_ms=0 disables expiry", "[daemon][region]") {
  IpcRegionRegistry reg(
      IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(200),
      });

  IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = 22;
  params.size_bytes = 4096;
  params.ttl_ms = 0;
  params.handle_bytes = "h";

  auto d_or = reg.register_region(params);
  REQUIRE(d_or.ok());
  const std::string region_id = d_or->region_id;
  REQUIRE(d_or->ttl_ms == 0);
  REQUIRE(d_or->expires_at == absl::InfiniteFuture());

  // Keepalive bumps should not enable expiry when ttl_ms=0.
  REQUIRE(reg.refresh_ttl(region_id, 150));
  auto d2_or = reg.describe(region_id);
  REQUIRE(d2_or.ok());
  REQUIRE(d2_or->ttl_ms == 0);
  REQUIRE(d2_or->expires_at == absl::InfiniteFuture());

  // Acquire/release should preserve infinite expiry.
  auto acq_or = reg.acquire(region_id, params.owner_pid);
  REQUIRE(acq_or.ok());
  REQUIRE(acq_or->ttl_ms == 0);
  REQUIRE(acq_or->expires_at == absl::InfiniteFuture());
  REQUIRE(reg.release(region_id).ok());

  // Sweeps should never expire the region by time.
  auto expired = reg.sweep_expired(absl::Now() + absl::Hours(24));
  REQUIRE(expired.empty());
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

TEST_CASE(
    "IpcRegionRegistry daemon-managed HOST_SHARED attachment keeps lease alive until release",
    "[daemon][region]") {
  IpcRegionRegistry reg(
      IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(500),
      });

  IpcRegionRegistry::RegisterParams params;
  params.memory_kind = IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = 31337;
  params.size_bytes = 1 << 20;
  params.ttl_ms = 100;
  params.daemon_managed = true;
  params.host_region_class = IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());
  REQUIRE_FALSE(desc_or->attach_token.empty());

  auto attachment_or = reg.acquire_host_shared_attachment(desc_or->attach_token, params.owner_pid);
  REQUIRE(attachment_or.ok());
  REQUIRE(attachment_or->fd >= 0);
  REQUIRE(attachment_or->size_bytes == params.size_bytes);

  absl::SleepFor(absl::Milliseconds(150));
  auto expired = reg.sweep_expired(absl::Now());
  REQUIRE(expired.empty());

  REQUIRE(reg.release_host_shared_attachment(desc_or->attach_token, params.owner_pid).ok());

  absl::SleepFor(absl::Milliseconds(150));
  expired = reg.sweep_expired(absl::Now());
  REQUIRE(expired.size() == 1);
  REQUIRE(expired[0].region_id == desc_or->region_id);
}

TEST_CASE("IpcRegionRegistry HOST_SHARED attachment release does not consume local mapping holds", "[daemon][region]") {
  IpcRegionRegistry reg(IpcRegionRegistry::Options{});

  IpcRegionRegistry::RegisterParams params;
  params.memory_kind = IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = 31337;
  params.size_bytes = 1 << 20;
  params.ttl_ms = 1000;
  params.daemon_managed = true;
  params.host_region_class = IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());
  REQUIRE_FALSE(desc_or->attach_token.empty());

  auto attachment_or = reg.acquire_host_shared_attachment(desc_or->attach_token, params.owner_pid);
  REQUIRE(attachment_or.ok());
  auto mapping_or = reg.acquire_host_shared_local_mapping(desc_or->region_id, params.owner_pid);
  REQUIRE(mapping_or.ok());

  REQUIRE(reg.release_host_shared_attachment(desc_or->attach_token, params.owner_pid).ok());

  auto duplicate_release = reg.release_host_shared_attachment(desc_or->attach_token, params.owner_pid);
  REQUIRE_FALSE(duplicate_release.ok());
  REQUIRE(absl::IsFailedPrecondition(duplicate_release));

  auto unregister_busy_or = reg.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE_FALSE(unregister_busy_or.ok());

  REQUIRE(reg.release(desc_or->region_id).ok());

  auto unregister_or = reg.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unregister_or.ok());
  REQUIRE(*unregister_or);
}

TEST_CASE("IpcRegionRegistry daemon-managed HOST_SHARED cleanup on pid exit", "[daemon][region]") {
  IpcRegionRegistry reg(IpcRegionRegistry::Options{});

  IpcRegionRegistry::RegisterParams params;
  params.memory_kind = IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = 42424;
  params.size_bytes = 1 << 20;
  params.ttl_ms = 0;
  params.daemon_managed = true;
  params.host_region_class = IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = reg.register_region(params);
  REQUIRE(desc_or.ok());

  auto removed = reg.handle_pid_exit(params.owner_pid);
  REQUIRE(removed.size() == 1);
  REQUIRE(removed[0].region_id == desc_or->region_id);

  auto attachment_or = reg.acquire_host_shared_attachment(desc_or->attach_token, params.owner_pid);
  REQUIRE_FALSE(attachment_or.ok());
}
