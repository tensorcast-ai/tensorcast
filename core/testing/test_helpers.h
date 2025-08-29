// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/common/cuda_api.h"

namespace tensorcast::communicator::test {

// Helper to create test data pattern on GPU
std::vector<uint8_t> create_test_pattern(std::size_t size, uint8_t seed);

// Helper to verify data pattern
bool verify_pattern(const void* data, std::size_t size, uint8_t seed);

// Helper to find an available port starting from base_port
// Returns the first available port, or -1 if none found
int find_available_port(int base_port = 50000, int max_attempts = 1000);

// Check if CUDA is available and skip test if not
#define SKIP_IF_NO_CUDA()                                                    \
  do {                                                                       \
    int device_count;                                                        \
    absl::Status status = tensorcast::cuda::get_device_count(&device_count); \
    if (!status.ok() || device_count == 0) {                                 \
      SKIP("No CUDA devices available");                                     \
    }                                                                        \
  } while (0)

} // namespace tensorcast::communicator::test