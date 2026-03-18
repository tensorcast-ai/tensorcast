// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/staging_flow_controller.h"

#include <atomic>
#include <thread>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"

#include "core/communicator/base/constants.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::engine {

namespace {

class DummyStager : public MemoryStager {
 public:
  absl::StatusOr<void*> stage(
      const std::shared_ptr<transport::PartitionTensor>& /*tensor*/,
      uint64_t /*offset*/,
      uint64_t /*bytes*/,
      StageMode /*mode*/ = StageMode::kBlocking) override {
    return absl::UnimplementedError("DummyStager stage not used");
  }

  absl::Status release_staged_buffer(gsl::not_null<void*> exposed_ptr) override {
    released_ptrs_.push_back(exposed_ptr);
    return absl::OkStatus();
  }

  size_t get_chunk_size() const override {
    return 1;
  }

  size_t get_num_buffers() const override {
    return 1;
  }

  std::vector<void*> released_ptrs_;
};

struct DummyStage {
  std::shared_ptr<DummyStager> stager = std::make_shared<DummyStager>();
  std::atomic<int> credit_released{0};

  StageLease make(void* ptr, FlowCreditLedger& ledger, StageTransport transport) {
    StageLease::Metadata meta;
    meta.transport = transport;
    return StageLease(
        stager,
        &ledger,
        ptr,
        /*bytes=*/8,
        /*mr=*/nullptr,
        /*deregister_mr=*/false,
        meta,
        [this]() { credit_released.fetch_add(1); });
  }
};

} // namespace

TEST_CASE("FlowCreditLedger acquires and releases credit") {
  FlowCreditLedger ledger(/*total_credit=*/4);

  auto lease_or = ledger.acquire(2);
  REQUIRE(lease_or.ok());
  FlowCreditLedger::Lease lease = std::move(lease_or.value());
  CHECK(ledger.outstanding_credit() == 2);

  lease.mark_consumed(2);
  CHECK(ledger.outstanding_credit() == 2);

  // Releasing used credit happens through StageLease; unused credit returns via destructor.
  lease.release_unused();
  CHECK(ledger.outstanding_credit() == 2);
  ledger.release(2);
  CHECK(ledger.outstanding_credit() == 0);
}

TEST_CASE("FlowCreditLedger blocks until credit available") {
  FlowCreditLedger ledger(/*total_credit=*/2);

  auto lease_or = ledger.acquire(2);
  REQUIRE(lease_or.ok());
  FlowCreditLedger::Lease lease = std::move(lease_or.value());
  lease.mark_consumed(2);

  std::atomic<bool> blocked{false};
  std::atomic<bool> acquired{false};

  std::thread waiter([&]() {
    blocked = true;
    auto lease2_or = ledger.acquire(2);
    REQUIRE(lease2_or.ok());
    acquired = true;
    FlowCreditLedger::Lease lease2 = std::move(lease2_or.value());
    lease2.mark_consumed(2);
    // Release when done
    lease2.release_unused();
  });

  // Ensure waiter is blocked
  while (!blocked.load()) {
    std::this_thread::yield();
  }
  CHECK_FALSE(acquired.load());

  // Simulate ACK returning credit
  ledger.release(2);
  waiter.join();
  CHECK(acquired.load());
  CHECK(ledger.outstanding_credit() == 2);
  ledger.release(2);
  CHECK(ledger.outstanding_credit() == 0);
}

TEST_CASE("FlowCreditLedger try_acquire returns available credit without blocking") {
  FlowCreditLedger ledger(/*total_credit=*/3);

  auto first = ledger.try_acquire(2);
  REQUIRE(first.ok());
  CHECK(first->granted_segments() == 2);
  first->mark_consumed(2);

  auto second = ledger.try_acquire(2);
  REQUIRE(second.ok());
  CHECK(second->granted_segments() == 1);
  second->mark_consumed();

  // With all credit outstanding, the next try should fail immediately.
  auto third = ledger.try_acquire(1);
  CHECK_FALSE(third.ok());
  CHECK(third.status().code() == absl::StatusCode::kUnavailable);

  // Release staged credit to restore availability.
  ledger.release(3);
}

TEST_CASE("StageLease releases credit and buffers exactly once") {
  FlowCreditLedger ledger(/*total_credit=*/3);
  DummyStage helper;

  auto lease_or = ledger.acquire(1);
  REQUIRE(lease_or.ok());
  FlowCreditLedger::Lease credit_lease = std::move(lease_or.value());
  credit_lease.mark_consumed();

  int value = 42;
  StageLease lease = helper.make(&value, ledger, StageTransport::kRdma);
  REQUIRE(lease.valid());

  lease.release();
  lease.release(); // second release is a no-op

  CHECK(helper.credit_released.load() == 1);
  CHECK(helper.stager->released_ptrs_.size() == 1);
  CHECK(helper.stager->released_ptrs_.front() == &value);
  CHECK(ledger.outstanding_credit() == 0);
  credit_lease.release_unused();
}

TEST_CASE("StageLease concurrent release is idempotent", "[staging][concurrency]") {
  FlowCreditLedger ledger(/*total_credit=*/2);
  DummyStage helper;

  auto lease_or = ledger.acquire(1);
  REQUIRE(lease_or.ok());
  FlowCreditLedger::Lease credit_lease = std::move(lease_or.value());
  credit_lease.mark_consumed();

  int value = 7;
  StageLease lease = helper.make(&value, ledger, StageTransport::kRdma);
  REQUIRE(lease.valid());

  std::vector<std::thread> workers;
  workers.reserve(8);
  for (int i = 0; i < 8; ++i) {
    workers.emplace_back([lease]() mutable { lease.release(); });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  CHECK(helper.credit_released.load() == 1);
  CHECK(helper.stager->released_ptrs_.size() == 1);
  CHECK(helper.stager->released_ptrs_.front() == &value);
  CHECK(ledger.outstanding_credit() == 0);
  credit_lease.release_unused();
}

TEST_CASE("StageLeaseRegistry stores and retrieves leases") {
  FlowCreditLedger ledger(/*total_credit=*/2);
  DummyStage helper;

  auto lease_or = ledger.acquire(1);
  REQUIRE(lease_or.ok());
  lease_or->mark_consumed();
  StageLease lease = helper.make(reinterpret_cast<void*>(0x1), ledger, StageTransport::kMtcp);

  StageLeaseRegistry registry;
  StageLeaseKey key{.request_key = "tensor:0", .window_seq = 7, .segment_idx = 2};
  registry.put(key, lease);
  CHECK(registry.size() == 1);

  auto take_or = registry.take(key);
  REQUIRE(take_or.ok());
  StageLease taken = take_or.value();
  CHECK(registry.size() == 0);
  taken.release();
}

TEST_CASE("StagingWindow stages windows respecting credit") {
  FlowCreditLedger ledger(/*total_credit=*/3);
  DummyStage helper;

  std::atomic<int> offsets{0};

  StagingWindow window(
      ledger,
      [&](uint64_t offset, uint32_t bytes, uint32_t segment_idx) -> absl::StatusOr<StageLease> {
        offsets.fetch_add(static_cast<int>(offset + bytes + segment_idx));
        return helper.make(reinterpret_cast<void*>(offset), ledger, StageTransport::kRdma);
      },
      /*total_bytes=*/64,
      /*chunk_size=*/16,
      /*initial_offset=*/0,
      /*max_window_segments=*/2);

  auto first = window.stage_next();
  REQUIRE(first.ok());
  CHECK(first->segments.size() == 2);
  CHECK(first->more_segments);

  auto second = window.stage_next();
  REQUIRE(second.ok());
  CHECK(second->segments.size() == 1);
  CHECK(second->more_segments);

  for (auto& seg : first->segments) {
    seg.lease.release();
  }
  for (auto& seg : second->segments) {
    seg.lease.release();
  }

  auto third = window.stage_next();
  REQUIRE(third.ok());
  CHECK(third->segments.size() == 1);
  CHECK_FALSE(third->more_segments);

  for (auto& seg : third->segments) {
    seg.lease.release();
  }

  auto done = window.stage_next();
  CHECK_FALSE(done.ok());
  CHECK(done.status().code() == absl::StatusCode::kOutOfRange);
}

TEST_CASE("StagingWindow surfaces unavailable while credit inflight") {
  FlowCreditLedger ledger(/*total_credit=*/2);
  DummyStage helper;

  std::vector<StageLease> inflight;
  inflight.reserve(2);

  StagingWindow window(
      ledger,
      [&](uint64_t offset, uint32_t /*bytes*/, uint32_t segment_idx) -> absl::StatusOr<StageLease> {
        StageLease::Metadata meta;
        meta.transport = StageTransport::kMtcp;
        meta.offset = offset;
        meta.segment_idx = segment_idx;
        return helper.make(reinterpret_cast<void*>(offset + segment_idx), ledger, StageTransport::kMtcp);
      },
      /*total_bytes=*/48,
      /*chunk_size=*/16,
      /*initial_offset=*/0,
      /*max_window_segments=*/2);

  auto first = window.stage_next();
  REQUIRE(first.ok());
  CHECK(first->segments.size() == 2);
  CHECK(first->more_segments);

  for (auto& seg : first->segments) {
    inflight.push_back(std::move(seg.lease));
  }

  auto second = window.stage_next();
  CHECK_FALSE(second.ok());
  CHECK(second.status().code() == absl::StatusCode::kUnavailable);

  for (auto& lease : inflight) {
    lease.release();
  }
  inflight.clear();

  auto third = window.stage_next();
  REQUIRE(third.ok());
  CHECK_FALSE(third->segments.empty());
  for (auto& seg : third->segments) {
    seg.lease.release();
  }
}

TEST_CASE("PartitionTensor direct RDMA flag toggles") {
  auto tensor = std::make_shared<transport::PartitionTensor>(
      "tensor",
      /*addr=*/0x1000,
      /*bytes=*/1024,
      communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
      nullptr);
  CHECK_FALSE(tensor->direct_rdma_enabled());
  tensor->set_direct_rdma_enabled(true);
  CHECK(tensor->direct_rdma_enabled());
  CHECK_FALSE(tensor->has_registered_mr());
}

TEST_CASE("StageLease metadata preserves zero_copy flag") {
  FlowCreditLedger ledger(/*total_credit=*/1);
  DummyStage helper;

  auto lease_or = ledger.acquire(1);
  REQUIRE(lease_or.ok());
  lease_or->mark_consumed();
  StageLease lease = helper.make(reinterpret_cast<void*>(0x1), ledger, StageTransport::kRdma);
  REQUIRE_FALSE(lease.metadata().zero_copy);

  auto metadata = lease.metadata();
  metadata.zero_copy = true;
  lease.set_metadata(metadata);
  CHECK(lease.metadata().zero_copy);

  lease.release();
  lease_or->release_unused();
}

TEST_CASE("StagingWindow supports request-scoped direct credit expansion", "[staging][direct]") {
  FlowCreditLedger direct_ledger(/*total_credit=*/16);
  DummyStage helper;

  StagingWindow window(
      direct_ledger,
      [&](uint64_t offset, uint32_t bytes, uint32_t segment_idx) -> absl::StatusOr<StageLease> {
        StageLease::Metadata meta;
        meta.transport = StageTransport::kRdma;
        meta.offset = offset;
        meta.bytes = bytes;
        meta.segment_idx = segment_idx;
        meta.zero_copy = true;
        return helper.make(reinterpret_cast<void*>(offset + segment_idx), direct_ledger, StageTransport::kRdma);
      },
      /*total_bytes=*/64,
      /*chunk_size=*/4,
      /*initial_offset=*/0,
      /*max_window_segments=*/16);

  auto first = window.stage_next();
  REQUIRE(first.ok());
  CHECK(first->segments.size() == 16);
  CHECK_FALSE(first->more_segments);
  CHECK(first->granted_credit == 16);
  CHECK(direct_ledger.outstanding_credit() == 16);

  for (auto& seg : first->segments) {
    seg.lease.release();
  }
  CHECK(direct_ledger.outstanding_credit() == 0);
}

} // namespace tensorcast::communicator::engine
