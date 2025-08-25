// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef TESTS_COMMON_H_
#define TESTS_COMMON_H_

#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include "absl/status/status.h"

// Command line arguments
extern std::string g_actor;
extern std::string g_ip;
extern uint16_t g_port;
extern uint32_t g_count;
extern uint32_t g_gpu;
extern uint32_t g_chunk;
extern uint32_t g_rdma;

int parse_options(int argc, char* argv[]);

// File operations
namespace stepcast::tests {

// Helper function to create a dummy file with patterned content.
// Returns true on success, false otherwise.
bool create_dummy_file(const std::filesystem::path& path, size_t size, char start_char = 'A');

// Helper function to read a file completely into a std::vector<char>.
std::vector<char> read_file_content(const std::filesystem::path& path);

// Helper utility to query CUDA device availability at runtime.
bool is_cuda_available();

// RFC-0007 test helper: generate minimal canonical `tensor_index.json` and
// `artifact_descriptor.json` for a standard partition replica directory that
// contains `tensor.data` or `tensor.data_*` files. This enables DiskLoader's
// descriptor gate in unit tests without requiring full Python save flows.
// Returns OK on success.
absl::Status write_rfc0007_descriptor_for_standard_artifact_dir(const std::filesystem::path& artifact_dir);

} // namespace stepcast::tests

#endif // TESTS_COMMON_H_
