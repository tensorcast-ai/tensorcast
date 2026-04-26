// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "core/communicator/engine/mr_cache.h"
#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/transport/net_dev.h"
#include "core/communicator/transport/partition_tensor.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

struct ibv_mr;

namespace tensorcast::communicator::engine {

struct RdmaSourceStageProfile;

StagingWindow::StageFn MakeStageFunction(
    const std::shared_ptr<transport::PartitionTensor>& tensor,
    FlowCreditLedger* ledger,
    const std::shared_ptr<MemoryStager>& stager,
    const transport::net_dev_t& dev,
    MrCache* mr_cache,
    std::string tensor_key,
    std::string request_key,
    v1::RdmaConfig::StagedRdmaBackend staged_backend,
    bool use_direct,
    ::ibv_mr* direct_mr,
    std::shared_ptr<RdmaSourceStageProfile> source_stage_profile);

TEST_CASE("MakeStageFunction guards null stager", "[communicator][rdma]") {
  transport::net_dev_t dev = nullptr;

  auto stage_fn = MakeStageFunction(
      /*tensor=*/nullptr,
      /*ledger=*/nullptr,
      /*stager=*/nullptr,
      dev,
      /*mr_cache=*/nullptr,
      /*tensor_key=*/"missing_stager",
      /*request_key=*/"missing_stager:0#1",
      /*staged_backend=*/v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED,
      /*use_direct=*/false,
      /*direct_mr=*/nullptr,
      /*source_stage_profile=*/nullptr);

  auto lease_or = stage_fn(/*offset=*/0, /*bytes=*/1, /*segment_idx=*/0);
  REQUIRE_FALSE(lease_or.ok());
  REQUIRE(lease_or.status().code() == absl::StatusCode::kFailedPrecondition);
  REQUIRE(lease_or.status().message().find("missing_stager") != std::string::npos);
}

} // namespace tensorcast::communicator::engine
