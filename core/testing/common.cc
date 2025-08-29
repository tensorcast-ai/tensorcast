// Copyright (c) 2025, TensorCast Team.

#include "common.h"
#include <getopt.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/common/artifact_hash.h"
#include "core/common/cuda_api.h"
#include "core/store/loader/disk_dir_hash.h"

struct option longOpts[] = {
    {"actor", required_argument, nullptr, 'a'},
    {"ip", required_argument, nullptr, 'i'},
    {"port", required_argument, nullptr, 'p'},
    {"count", required_argument, nullptr, 'c'},
    {"gpu", required_argument, nullptr, 'g'},
    {"chunk", required_argument, nullptr, 'k'},
    {"rdma", no_argument, nullptr, 'r'},
    {"help", no_argument, nullptr, 'h'},
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

namespace tensorcast::tests {

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
  absl::Status status = tensorcast::cuda::get_device_count(&device_count);
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

absl::Status write_rfc0007_descriptor_for_standard_artifact_dir(const std::filesystem::path& artifact_dir) {
  using nlohmann::json;
  // Validate directory exists
  if (!std::filesystem::exists(artifact_dir) || !std::filesystem::is_directory(artifact_dir)) {
    return absl::NotFoundError("artifact_dir does not exist or is not a directory");
  }
  // Enumerate partition files and compute total size as sum of file sizes.
  std::vector<std::filesystem::path> parts;
  uint64_t total_size = 0;
  for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name == "tensor.data" || name.rfind("tensor.data_", 0) == 0) {
      parts.push_back(entry.path());
    }
  }
  std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
  if (parts.empty()) {
    return absl::NotFoundError("no partition files (tensor.data or tensor.data_*) found");
  }
  for (const auto& p : parts) {
    total_size += static_cast<uint64_t>(std::filesystem::file_size(p));
  }

  // 1) Write minimal canonical tensor_index.json covering [0, total_size)
  json idx = json::object();
  json entry = json::array();
  entry.push_back(0); // offset
  entry.push_back(total_size); // size
  entry.push_back(json::array()); // shape
  entry.push_back(json::array()); // stride
  entry.push_back("torch.uint8"); // dtype
  entry.push_back(0); // storage_offset
  idx["__dummy__"] = std::move(entry);

  const auto index_json_path = artifact_dir / "tensor_index.json";
  {
    std::ofstream of(index_json_path);
    if (!of.is_open()) {
      return absl::InternalError("failed to open tensor_index.json for writing");
    }
    of << idx.dump();
  }

  // 2) Compute multihashes via core/common/artifact_hash
  auto index_mh_or =
      tensorcast::store::artifact_hash::compute_index_multihash(std::optional<std::string>(idx.dump()), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  auto data_mh_or = tensorcast::store::loader::compute_data_multihash_from_disk_dir(artifact_dir.string());
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }

  // 3) Persist artifact_descriptor.json
  json desc;
  desc["artifact_id"] = std::string("mi2:") + *index_mh_or + ":" + *data_mh_or;
  desc["index_multihash"] = *index_mh_or;
  desc["data_multihash"] = *data_mh_or;
  desc["schema_version"] = "v2";
  desc["encoding"] = "json";
  desc["total_size"] = total_size;
  json hp;
  hp["chunk_size"] = 4 * 1024 * 1024;
  hp["fanout"] = 2;
  hp["algorithm"] = "sha2-256";
  desc["hash_params"] = hp;
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  {
    std::ofstream of(descriptor_path);
    if (!of.is_open()) {
      return absl::InternalError("failed to open artifact_descriptor.json for writing");
    }
    of << desc.dump(2);
  }

  return absl::OkStatus();
}

} // namespace tensorcast::tests
