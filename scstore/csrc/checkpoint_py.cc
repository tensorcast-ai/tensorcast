
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <torch/extension.h>

#include <cstdint>

#ifdef LOG
#undef LOG
#endif
#ifdef DLOG
#undef DLOG
#endif
#ifdef VLOG
#undef VLOG
#endif
#ifdef LOG_IF
#undef LOG_IF
#endif
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/checkpoint/checkpoint.h"
#include "core/checkpoint/checkpoint_streaming.h"
#include "core/common/logging_init.h"
#include "core/common/model_verification.h"

namespace py = pybind11;

using namespace stepcast::store;

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
    py::dict config = py::dict()) {
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
}