// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "common.h"
#include <getopt.h>
#include <fstream>
#include <string>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/common/cuda_api.h"

struct option longOpts[] = {
    {"actor", required_argument, 0, 'a'},
    {"ip", required_argument, 0, 'i'},
    {"port", required_argument, 0, 'p'},
    {"count", required_argument, 0, 'c'},
    {"gpu", required_argument, 0, 'g'},
    {"chunk", required_argument, 0, 'k'},
    {"rdma", no_argument, 0, 'r'},
    {"help", no_argument, 0, 'h'},
};

static void printHelp(char* program_name) {
  printf(
      "USAGE: %s \n\t"
      "[-a,--actor server and client] \n\t"
      "[-i,--ip <ip address>] \n\t"
      "[-p,--port  <tcp count>\n\t"
      "[-c,--count <data count>] \n\t"
      "[-g,--gpu <gpu count>] \n\t"
      "[-r,--rdma <tcp or rdma>] \n\t"
      "[-h,--help]\n",
      program_name);
}

std::string g_actor = "server";
std::string g_ip = "127.0.0.1";
uint16_t g_port = 6606;
uint32_t g_count = 256 * 1024 * 1024;
uint32_t g_gpu = 1;
uint32_t g_chunk = 1;
uint32_t g_rdma = 0;

int parse_options(int argc, char* argv[]) {
  int opt = -1;
  int longIndex = 0;
  while (-1 != (opt = getopt_long(argc, argv, "a:i:p:g:k:c:ht", longOpts, &longIndex))) {
    switch (opt) {
      case 'a':
        g_actor = std::string(optarg);
        break;
      case 'i':
        g_ip = std::string(optarg);
        break;
      case 'p':
        g_port = atoi(optarg);
        break;
      case 'c':
        g_count = atoi(optarg);
        break;
      case 'g':
        g_gpu = atoi(optarg);
        break;
      case 'k':
        g_chunk = atoi(optarg);
        break;
      case 't':
        g_rdma = 1;
        break;
      case 'h':
        printHelp(argv[0]);
        exit(0);
      default:
        printf("invalid option \n");
        printf("Try [-h --help] for more information.\n");
        exit(-1);
    }
  }
  return 0;
}

namespace stepcast::tests {

bool create_dummy_file(const std::filesystem::path& path, size_t size, char start_char) {
  LOG(INFO) << "Creating dummy file at " << path << " with size " << size;
  std::ofstream outfile(path, std::ios::binary);
  if (!outfile) {
    LOG(ERROR) << "Failed to open file for writing: " << path;
    return false;
  }
  std::vector<char> buffer(size);
  for (size_t i = 0; i < size; ++i) {
    buffer[i] = static_cast<char>(start_char + (i % 26));
  }
  outfile.write(buffer.data(), static_cast<std::streamsize>(size));
  return outfile.good();
}

std::vector<char> read_file_content(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file for reading: " << path;
    return {};
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  if (!file.read(buffer.data(), size)) {
    LOG(ERROR) << "Failed to read data from file: " << path;
    return {};
  }
  return buffer;
}

bool is_cuda_available() {
  int device_count = 0;
  absl::Status status = stepcast::cuda::get_device_count(&device_count);
  if (!status.ok()) {
    LOG(WARNING) << "CUDA device count query failed: " << status.message();
    return false;
  }
  if (device_count == 0) {
    LOG(WARNING) << "No CUDA-capable devices found.";
    return false;
  }
  LOG(INFO) << "CUDA available. Found " << device_count << " device(s).";
  return true;
}

} // namespace stepcast::tests
