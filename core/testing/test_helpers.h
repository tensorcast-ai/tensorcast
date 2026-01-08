// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/communicator/engine/engine.h"
#include "core/cuda/cuda_api.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

namespace tensorcast::testing {

// Helper to create test data pattern on GPU
std::vector<uint8_t> create_test_pattern(std::size_t size, uint8_t seed);

// Helper to verify data pattern
bool verify_pattern(const void* data, std::size_t size, uint8_t seed);

// Helper to find an available TCP port. The search randomizes candidate order
// even when a base_port hint is supplied, expanding into nearby ephemeral
// ranges to reduce collisions across concurrent tests. Returns the first
// available port, or -1 if none found.
int find_available_port(int base_port = 50000, int max_attempts = 1000);

// Configure TCP communicator staging defaults for tests. Ensures the GPU/CPU
// stagers and buffer pipeline satisfy Communicator constructor invariants and
// forces TCP listeners to disable SO_REUSEPORT for deterministic binding.
void configure_tcp_stager_defaults(
    tensorcast::communicator::v1::CommunicatorConfig* cfg,
    uint32_t buffers_per_flow = 4);

// Create a CommunicatorConfig with RDMA disabled and staging defaults applied.
// Callers can further mutate the returned proto (transport params, pool sizes).
tensorcast::communicator::v1::CommunicatorConfig make_tcp_communicator_config(
    bool enable_rdma = false,
    uint32_t buffers_per_flow = 4);

tensorcast::communicator::engine::Communicator::PinnedStagingPools make_test_pinned_staging_pools(
    uint32_t buffers_per_flow = 4,
    int tcp_conn_count = 8,
    size_t gpu_slice_bytes = 16ULL * 1024 * 1024,
    size_t cpu_slice_bytes = 4ULL * 1024 * 1024,
    bool enable_rdma = false);

// Check if CUDA is available and skip test if not
#define SKIP_IF_NO_CUDA()                                                    \
  do {                                                                       \
    int device_count;                                                        \
    absl::Status status = tensorcast::cuda::get_device_count(&device_count); \
    if (!status.ok() || device_count == 0) {                                 \
      SKIP("No CUDA devices available");                                     \
    }                                                                        \
  } while (0)

} // namespace tensorcast::testing
