
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <torch/extension.h>
#include "scstore/csrc/logging.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/checkpoint/checkpoint.h"
#include "core/checkpoint/checkpoint_streaming.h"
#include "core/common/logging_init.h"
#include "core/common/model_hash.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include "core/common/model_verification.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/source_hash.h"

namespace py = pybind11;

using namespace stepcast::store;
using stepcast::store::model_hash::compute_index_multihash;

// Helper function to convert ModelVerificationInfo to Python dictionary
py::dict verification_info_to_dict(const ModelVerificationInfo& info) {
  py::dict result;
  result["model_size"] = info.model_size;
  result["full_hash"] = info.full_hash;

  py::list segment_hashes;
  for (const auto& hash : info.segment_hashes) {
    segment_hashes.append(hash);
  }
  result["segment_hashes"] = segment_hashes;

  py::list sample_values;
  for (const auto& value : info.sample_values) {
    sample_values.append(value);
  }
  result["sample_values"] = sample_values;

  py::list key_values;
  for (const auto& value : info.key_values) {
    key_values.append(value);
  }
  result["key_values"] = key_values;

  return result;
}

// Helper function to convert Python dictionary to ModelVerificationInfo
ModelVerificationInfo dict_to_verification_info(const py::dict& dict) {
  ModelVerificationInfo info;

  if (dict.contains("model_size")) {
    info.model_size = dict["model_size"].cast<uint64_t>();
  }
  if (dict.contains("full_hash")) {
    info.full_hash = dict["full_hash"].cast<uint64_t>();
  }

  if (dict.contains("segment_hashes")) {
    py::list seg_hashes = dict["segment_hashes"];
    for (size_t i = 0; i < std::min(seg_hashes.size(), info.segment_hashes.size()); ++i) {
      info.segment_hashes[i] = seg_hashes[i].cast<uint64_t>();
    }
  }

  if (dict.contains("sample_values")) {
    py::list sample_vals = dict["sample_values"];
    for (size_t i = 0; i < std::min(sample_vals.size(), info.sample_values.size()); ++i) {
      info.sample_values[i] = sample_vals[i].cast<uint64_t>();
    }
  }

  if (dict.contains("key_values")) {
    py::list key_vals = dict["key_values"];
    for (size_t i = 0; i < std::min(key_vals.size(), info.key_values.size()); ++i) {
      info.key_values[i] = key_vals[i].cast<uint64_t>();
    }
  }

  return info;
}

// Wrapper function for generate_model_verification_info_from_disk
py::dict generate_model_verification_info_wrapper(const std::string& model_path, int verification_level = 1) {
  try {
    ModelVerificationInfo info;
    {
      py::gil_scoped_release release;
      auto level = static_cast<VerificationLevel>(verification_level);
      info = generate_model_verification_info_from_disk(model_path, level);
    }
    return verification_info_to_dict(info);
  } catch (const std::exception& e) {
    const auto str = std::string("Failed to generate verification info: ") + e.what();
    LOG(ERROR) << str;
    PyErr_SetString(PyExc_RuntimeError, str.c_str());
    throw py::error_already_set();
  }
}

// Wrapper function for GPU verification
bool verify_model_data_from_gpu_wrapper(
    int device_id,
    std::uint64_t cuda_memory_ptr,
    size_t memory_size,
    const py::dict& expected_verification,
    int verification_level) {
  try {
    ModelVerificationInfo expected_info = dict_to_verification_info(expected_verification);

    // Create data pointers and sizes for verification
    std::vector<void*> data_ptrs = {reinterpret_cast<void*>(cuda_memory_ptr)};
    std::vector<size_t> data_sizes = {memory_size};

    auto level = static_cast<VerificationLevel>(verification_level);
    absl::Status result;
    {
      py::gil_scoped_release release;
      result = ModelVerifier::verify_model_data(data_ptrs, data_sizes, expected_info, level, device_id);
    }

    if (result.ok()) {
      return true;
    }
    LOG(ERROR) << "GPU verification failed: " << result.message();
    return false;

  } catch (const std::exception& e) {
    const auto str = std::string("GPU verification failed: ") + e.what();
    LOG(ERROR) << str;
    PyErr_SetString(PyExc_RuntimeError, str.c_str());
    throw py::error_already_set();
  }
}

// Wrapper for save_tensors_streaming with optional config
std::unordered_map<std::string, uint64_t> save_tensors_streaming_wrapper(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path,
    const py::dict& config = py::dict()) {
  StreamingTensorWriter::Config writer_config;

  // Parse config from Python dict
  if (config.contains("num_buffers")) {
    writer_config.num_buffers = config["num_buffers"].cast<size_t>();
  }
  if (config.contains("buffer_size_mb")) {
    writer_config.buffer_size_mb = config["buffer_size_mb"].cast<size_t>();
  }
  if (config.contains("enable_async_write")) {
    writer_config.enable_async_write = config["enable_async_write"].cast<bool>();
  }

  return save_tensors_streaming(tensor_names, tensor_data, path, writer_config);
}

// Helper wrapper to get CUDA memory pointer with proper error handling
static uint64_t get_cuda_memory_ptr_wrapper(int device_id, const py::bytes& cuda_ipc_handle_py) {
  // Convert Python bytes to std::string handle
  const auto handle = cuda_ipc_handle_py.cast<std::string>();

  absl::StatusOr<std::uint64_t> ptr_or = get_cuda_memory_ptr(device_id, handle);
  if (!ptr_or.ok()) {
    const auto str = ptr_or.status().ToString();
    LOG(ERROR) << "Failed to get CUDA memory pointer: " << str;
    PyErr_SetString(PyExc_ValueError, str.c_str());
    throw py::error_already_set();
  }
  return ptr_or.value();
}

// Helper wrapper to close CUDA memory handle with error conversion
static bool close_cuda_memory_handle_wrapper(int device_id, std::uint64_t cuda_memory_ptr) {
  absl::Status status = close_cuda_memory_handle(device_id, cuda_memory_ptr);
  if (!status.ok()) {
    const auto str = status.ToString();
    LOG(ERROR) << "Failed to close CUDA memory handle: " << str;
    PyErr_SetString(PyExc_ValueError, str.c_str());
    throw py::error_already_set();
  }
  return true;
}

// ------------------------------------------------------------------
// Unified Save: write data partitions, tensor_index.json, model_descriptor.json
// ------------------------------------------------------------------
static py::dict save_model_to_disk_wrapper(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const py::dict& meta_state_dict,
    const std::string& path,
    const py::dict& config = py::dict()) {
  // Write data partitions using streaming writer
  StreamingTensorWriter::Config writer_config;
  if (config.contains("num_buffers")) {
    writer_config.num_buffers = config["num_buffers"].cast<size_t>();
  }
  if (config.contains("buffer_size_mb")) {
    writer_config.buffer_size_mb = config["buffer_size_mb"].cast<size_t>();
  }
  if (config.contains("enable_async_write")) {
    writer_config.enable_async_write = config["enable_async_write"].cast<bool>();
  }

  const auto offsets = save_tensors_streaming(tensor_names, tensor_data, path, writer_config);

  // Compute canonical storage size per offset (max over aliases)
  std::unordered_map<uint64_t, uint64_t> offset_max_size;
  for (const auto& name : tensor_names) {
    const auto off_it = offsets.find(name);
    if (off_it == offsets.end()) {
      continue;
    }
    const uint64_t off = off_it->second;
    const uint64_t sz = tensor_data.at(name).second;
    auto it = offset_max_size.find(off);
    if (it == offset_max_size.end()) {
      offset_max_size.emplace(off, sz);
    } else if (sz > it->second) {
      it->second = sz;
    }
  }

  // Build v2 tensor_index JSON (deterministic order: tensor name ascending)
  nlohmann::json j;
  std::vector<std::string> sorted_names = tensor_names;
  std::ranges::sort(sorted_names);
  for (const auto& name : sorted_names) {
    const auto off_it = offsets.find(name);
    if (off_it == offsets.end()) {
      continue;
    }
    const uint64_t off = off_it->second;
    const uint64_t sz = offset_max_size.at(off);

    // meta_state_dict[name] = (shape, stride, dtype, storage_offset)
    if (!meta_state_dict.contains(name.c_str())) {
      throw std::runtime_error("meta_state_dict missing tensor: " + name);
    }
    auto tpl = meta_state_dict[name.c_str()].cast<py::tuple>();
    if (tpl.size() != 4) {
      throw std::runtime_error("meta_state_dict entry must be a 4-tuple");
    }
    auto shape = tpl[0].cast<std::vector<int64_t>>();
    auto stride = tpl[1].cast<std::vector<int64_t>>();
    auto dtype = tpl[2].cast<std::string>();
    auto storage_offset = tpl[3].cast<uint64_t>();

    // v2 tuple: [offset, size, shape[], stride[], dtype, storage_offset]
    nlohmann::json arr = nlohmann::json::array();
    arr.push_back(off);
    arr.push_back(sz);
    // shape and stride as arrays
    nlohmann::json j_shape = nlohmann::json::array();
    for (auto v : shape) {
      j_shape.push_back(static_cast<uint64_t>(v));
    }
    nlohmann::json j_stride = nlohmann::json::array();
    for (auto v : stride) {
      j_stride.push_back(static_cast<uint64_t>(v));
    }
    arr.push_back(j_shape);
    arr.push_back(j_stride);
    arr.push_back(dtype);
    arr.push_back(storage_offset);
    j[name] = arr;
  }

  // Serialize canonical JSON with sorted keys and minimal separators
  const std::string index_json = j.dump();
  const std::filesystem::path dir(path);
  std::filesystem::create_directories(dir);
  {
    std::ofstream out(dir / "tensor_index.json");
    if (!out.is_open()) {
      throw std::runtime_error("Failed to write tensor_index.json");
    }
    out << index_json;
    out.close();
  }
  // Additionally write CBOR encoding (preferred for canonical index persistence)
  {
    std::vector<std::uint8_t> cbor = nlohmann::json::to_cbor(j);
    std::ofstream oc(dir / "tensor_index.cbor", std::ios::binary);
    if (!oc.is_open()) {
      throw std::runtime_error("Failed to write tensor_index.cbor");
    }
    oc.write(reinterpret_cast<const char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
    oc.close();
  }

  // Compute descriptor fields
  absl::StatusOr<std::string> idx_mh_or =
      compute_index_multihash(std::optional<std::string>(index_json), /*index_key_hex=*/"");
  if (!idx_mh_or.ok()) {
    throw std::runtime_error(idx_mh_or.status().ToString());
  }
  // Compute data multihash via unified SeekableSource pipeline
  auto compute_mh_via_source = [&](const std::string& dir_path) -> absl::StatusOr<std::string> {
    namespace fs = std::filesystem;
    fs::path dir(dir_path);
    // Collect partition files deterministically
    std::vector<fs::path> parts;
    for (const auto& entry : fs::directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto name = entry.path().filename().string();
      if (name.rfind("tensor.data_", 0) == 0) {
        parts.push_back(entry.path());
      }
    }
    std::ranges::sort(parts, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    if (parts.empty()) {
      fs::path single = dir / "tensor.data";
      if (fs::exists(single)) {
        parts.push_back(single);
      }
    }
    if (parts.empty()) {
      return absl::NotFoundError("No tensor.data partitions found");
    }
    // Determine logical total size from canonical index (we just wrote j)
    uint64_t total_size = 0;
    for (auto it = j.begin(); it != j.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      uint64_t off = arr[0].get<uint64_t>();
      uint64_t sz = arr[1].get<uint64_t>();
      total_size = std::max<uint64_t>(total_size, off + sz);
    }
    stepcast::store::loader::FilePartitionSource::Options opts;
    for (const auto& p : parts) {
      opts.partition_paths.push_back(p);
      opts.partition_sizes.push_back(static_cast<size_t>(fs::file_size(p)));
    }
    opts.total_size = total_size;
    opts.chunk_size = 128 * 1024 * 1024;
    opts.use_direct_io = (total_size > 5ULL * 1024 * 1024 * 1024);
    stepcast::store::loader::FilePartitionSource src(std::move(opts));
    return stepcast::store::loader::compute_data_multihash_from_seekable_source(src, total_size);
  };
  absl::StatusOr<std::string> data_mh_or = compute_mh_via_source(path);
  if (!data_mh_or.ok()) {
    throw std::runtime_error(data_mh_or.status().ToString());
  }

  // Compute total size: max(offset+size)
  uint64_t total_size = 0;
  for (const auto& [off, sz] : offset_max_size) {
    total_size = std::max<uint64_t>(total_size, off + sz);
  }

  nlohmann::json desc;
  desc["model_id"] = std::string("mi2:") + *idx_mh_or + std::string(":") + *data_mh_or;
  desc["index_multihash"] = *idx_mh_or;
  desc["data_multihash"] = *data_mh_or;
  desc["schema_version"] = "v2";
  desc["encoding"] = "json";
  desc["total_size"] = total_size;
  nlohmann::json hash_params;
  hash_params["chunk_size"] = 4 * 1024 * 1024;
  hash_params["fanout"] = 2;
  desc["hash_params"] = hash_params;

  {
    std::ofstream out(dir / "model_descriptor.json");
    if (!out.is_open()) {
      throw std::runtime_error("Failed to write model_descriptor.json");
    }
    out << desc.dump(2);
    out.close();
  }

  // Return descriptor to Python
  py::dict result;
  result["model_id"] = desc["model_id"].get<std::string>();
  result["index_multihash"] = desc["index_multihash"].get<std::string>();
  result["data_multihash"] = desc["data_multihash"].get<std::string>();
  result["schema_version"] = desc["schema_version"].get<std::string>();
  result["encoding"] = desc["encoding"].get<std::string>();
  result["total_size"] = total_size;
  return result;
}

static py::dict inspect_or_generate_descriptor_wrapper(const std::string& path) {
  const std::filesystem::path dir(path);
  const auto desc_path = dir / "model_descriptor.json";
  if (std::filesystem::exists(desc_path)) {
    std::ifstream in(desc_path);
    if (!in.is_open()) {
      throw std::runtime_error("Failed to open model_descriptor.json");
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    nlohmann::json j = nlohmann::json::parse(buffer.str());
    py::dict result;
    result["model_id"] = j["model_id"].get<std::string>();
    result["index_multihash"] = j["index_multihash"].get<std::string>();
    result["data_multihash"] = j["data_multihash"].get<std::string>();
    result["schema_version"] = j["schema_version"].get<std::string>();
    result["encoding"] = j["encoding"].get<std::string>();
    result["total_size"] = j["total_size"].get<uint64_t>();
    return result;
  }

  // Compute index multihash from tensor_index.json
  std::ifstream idx_in(dir / "tensor_index.json");
  if (!idx_in.is_open()) {
    throw std::runtime_error("tensor_index.json not found");
  }
  std::stringstream idx_ss;
  idx_ss << idx_in.rdbuf();
  idx_in.close();
  const std::string index_json = idx_ss.str();

  absl::StatusOr<std::string> idx_mh_or =
      compute_index_multihash(std::optional<std::string>(index_json), /*index_key_hex=*/"");
  if (!idx_mh_or.ok()) {
    throw std::runtime_error(idx_mh_or.status().ToString());
  }
  // Compute data multihash via unified SeekableSource pipeline
  auto compute_mh_via_source2 = [&](const std::string& dir_path) -> absl::StatusOr<std::string> {
    namespace fs = std::filesystem;
    fs::path dir2(dir_path);
    std::vector<fs::path> parts;
    for (const auto& entry : fs::directory_iterator(dir2)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto name = entry.path().filename().string();
      if (name.rfind("tensor.data_", 0) == 0) {
        parts.push_back(entry.path());
      }
    }
    std::ranges::sort(parts, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    if (parts.empty()) {
      fs::path single = dir2 / "tensor.data";
      if (fs::exists(single)) {
        parts.push_back(single);
      }
    }
    if (parts.empty()) {
      return absl::NotFoundError("No tensor.data partitions found");
    }
    // Determine logical total size from index_json
    uint64_t total_size2 = 0;
    nlohmann::json idx_j2 = nlohmann::json::parse(index_json);
    for (auto it = idx_j2.begin(); it != idx_j2.end(); ++it) {
      const auto& arr = it.value();
      if (!arr.is_array() || arr.size() < 2) {
        continue;
      }
      uint64_t off = arr[0].get<uint64_t>();
      uint64_t sz = arr[1].get<uint64_t>();
      total_size2 = std::max<uint64_t>(total_size2, off + sz);
    }
    stepcast::store::loader::FilePartitionSource::Options opts2;
    for (const auto& p : parts) {
      opts2.partition_paths.push_back(p);
      opts2.partition_sizes.push_back(static_cast<size_t>(fs::file_size(p)));
    }
    opts2.total_size = total_size2;
    opts2.chunk_size = 128 * 1024 * 1024;
    opts2.use_direct_io = (total_size2 > 5ULL * 1024 * 1024 * 1024);
    stepcast::store::loader::FilePartitionSource src2(std::move(opts2));
    return stepcast::store::loader::compute_data_multihash_from_seekable_source(src2, total_size2);
  };
  absl::StatusOr<std::string> data_mh_or = compute_mh_via_source2(path);
  if (!data_mh_or.ok()) {
    throw std::runtime_error(data_mh_or.status().ToString());
  }

  // Compute total size from index JSON
  nlohmann::json idx_j = nlohmann::json::parse(index_json);
  uint64_t total_size = 0;
  for (auto it = idx_j.begin(); it != idx_j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    uint64_t off = arr[0].get<uint64_t>();
    uint64_t sz = arr[1].get<uint64_t>();
    total_size = std::max<uint64_t>(total_size, off + sz);
  }

  nlohmann::json desc;
  desc["model_id"] = std::string("mi2:") + *idx_mh_or + std::string(":") + *data_mh_or;
  desc["index_multihash"] = *idx_mh_or;
  desc["data_multihash"] = *data_mh_or;
  desc["schema_version"] = "v2";
  desc["encoding"] = "json";
  desc["total_size"] = total_size;
  nlohmann::json hash_params;
  hash_params["chunk_size"] = 4 * 1024 * 1024;
  hash_params["fanout"] = 2;
  desc["hash_params"] = hash_params;

  std::ofstream out(desc_path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to write model_descriptor.json");
  }
  out << desc.dump(2);
  out.close();

  py::dict result;
  result["model_id"] = desc["model_id"].get<std::string>();
  result["index_multihash"] = desc["index_multihash"].get<std::string>();
  result["data_multihash"] = desc["data_multihash"].get<std::string>();
  result["schema_version"] = desc["schema_version"].get<std::string>();
  result["encoding"] = desc["encoding"].get<std::string>();
  result["total_size"] = total_size;
  return result;
}

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Initialize logging only once across all modules
  ensure_logging_initialized();

#ifdef USE_FAKE_CUDA
  py::module_::import("warnings")
      .attr("warn")("CUDA not detected, running with FakeCuda backend. Only logical correctness is guaranteed.");
#endif
  m.def("save_tensors", &save_tensors, "Save a state dict")
      .def(
          "save_tensors_streaming",
          &save_tensors_streaming_wrapper,
          "Save a state dict using streaming approach",
          py::arg("tensor_names"),
          py::arg("tensor_data"),
          py::arg("path"),
          py::arg("config") = py::dict())
      .def(
          "save_model_to_disk",
          &save_model_to_disk_wrapper,
          py::arg("tensor_names"),
          py::arg("tensor_data"),
          py::arg("meta_state_dict"),
          py::arg("path"),
          py::arg("config") = py::dict(),
          "Unified save: writes data, tensor_index.json, model_descriptor.json and returns descriptor")
      .def("restore_tensors", &restore_tensors, "Restore a state dict")
      .def(
          "restore_tensors_from_model_path",
          &restore_tensors_from_model_path,
          py::arg("meta_state_dict"),
          py::arg("model_path"),
          py::arg("tensor_device_offsets"),
          py::arg("device_id") = -1,
          "Restore a state dict from model path, device_id=-1 means load to CPU, otherwise CUDA device id")
      .def(
          "allocate_cuda_memory",
          &allocate_cuda_memory,
          py::arg("device_id"),
          py::arg("size"),
          "Allocate CUDA memory on a single device")
      .def(
          "get_cuda_memory_handle",
          [](int device_id, std::uint64_t memory_ptr_int) {
            const std::string handle = get_cuda_memory_handle(device_id, memory_ptr_int);
            return py::bytes(handle);
          },
          py::arg("device_id"),
          py::arg("memory_ptr"),
          "Get a CUDA IPC memory handle for a single allocation")
      .def("get_device_uuid_map", &get_device_uuid_map, "Get device uuid map")
      .def(
          "get_cuda_memory_ptr",
          &get_cuda_memory_ptr_wrapper,
          py::arg("device_id"),
          py::arg("cuda_ipc_handle"),
          "Get CUDA memory pointer from IPC handle (raises Python exception on failure)")
      .def(
          "close_cuda_memory_handle",
          &close_cuda_memory_handle_wrapper,
          py::arg("device_id"),
          py::arg("cuda_memory_ptr"),
          "Close CUDA memory handle (raises Python exception on failure)")
      .def(
          "generate_model_verification_info",
          &generate_model_verification_info_wrapper,
          py::arg("model_path"),
          py::arg("verification_level") = 1,
          "Generate model verification information from saved model files")
      .def(
          "verify_model_data_from_gpu",
          &verify_model_data_from_gpu_wrapper,
          py::arg("device_id"),
          py::arg("cuda_memory_ptr"),
          py::arg("memory_size"),
          py::arg("expected_verification"),
          py::arg("verification_level"),
          "Verify model data integrity from GPU memory");

  m.def(
      "inspect_or_generate_descriptor",
      &inspect_or_generate_descriptor_wrapper,
      py::arg("model_path"),
      "Return model descriptor if present; otherwise compute multihashes and write model_descriptor.json");
}