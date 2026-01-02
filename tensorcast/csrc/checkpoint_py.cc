
// Copyright (c) 2025-2026, TensorCast Team.

#include <torch/extension.h>
#include "tensorcast/csrc/logging.h"
#include "tensorcast/csrc/py_error_utils.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ranges>

#include <pybind11/stl.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/checkpoint/checkpoint.h"
#include "core/checkpoint/streaming_tensor_writer.h"
#include "core/common/artifact_hash.h"
#include "core/common/const/granularity.h"
#include "core/common/cuda_api.h"
#include "core/common/logging_init.h"
#include "core/common/memory/pinned_buffer_pool.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "core/common/artifact_verification.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"
#include "core/store/materialization/dataplane/view/view_planner.h"

namespace py = pybind11;

using tensorcast::checkpoint::close_cuda_memory_handle;
using tensorcast::checkpoint::generate_verification_info_from_disk;
using tensorcast::checkpoint::get_cuda_memory_ptr;
using tensorcast::checkpoint::StreamingTensorWriter;
using tensorcast::common::ArtifactVerificationInfo;
using tensorcast::common::ArtifactVerifier;
using tensorcast::common::compute_index_multihash;
using tensorcast::common::VerificationLevel;
using tensorcast::store::loader::BidirectionalViewPlan;
using tensorcast::store::loader::TensorTransformPlan;
using tensorcast::store::loader::ViewOp;
using tensorcast::store::loader::ViewPlanner;
using tensorcast::store::loader::ViewSpec;

// Helper function to convert ArtifactVerificationInfo to Python dictionary
py::dict verification_info_to_dict(const ArtifactVerificationInfo& info) {
  py::dict result;
  if (!info.byte_space_id.empty()) {
    result["byte_space_id"] = info.byte_space_id;
  }
  result["artifact_size"] = info.artifact_size;
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

// Helper function to convert Python dictionary to ArtifactVerificationInfo
ArtifactVerificationInfo dict_to_verification_info(const py::dict& dict) {
  ArtifactVerificationInfo info;

  if (dict.contains("byte_space_id")) {
    info.byte_space_id = dict["byte_space_id"].cast<std::string>();
  }
  if (dict.contains("artifact_size")) {
    info.artifact_size = dict["artifact_size"].cast<uint64_t>();
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

static absl::StatusOr<std::unordered_map<std::string, uint64_t>> write_tensor_data_partitions(
    const std::vector<std::string>& tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path,
    const StreamingTensorWriter::Config& config) {
  std::filesystem::create_directories(path);
  const auto tensor_filename = (std::filesystem::path(path) / "tensor.data").string();

  const size_t pool_size_bytes = config.pool_size_gb * (static_cast<size_t>(1) << 30);
  const size_t chunk_size_bytes = config.buffer_size_mb * (static_cast<size_t>(1) << 20);
  auto pinned_pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(pool_size_bytes, chunk_size_bytes);

  StreamingTensorWriter writer(tensor_filename, config, pinned_pool);
  absl::Status init_status = writer.initialize();
  if (!init_status.ok()) {
    return init_status;
  }

  struct StorageMeta {
    uint64_t max_size{0};
  };

  std::unordered_map<const void*, StorageMeta> storage_meta;
  storage_meta.reserve(tensor_names.size());
  for (const auto& name : tensor_names) {
    const auto it = tensor_data.find(name);
    if (it == tensor_data.end()) {
      return absl::InvalidArgumentError(std::string("tensor_data missing tensor: ") + name);
    }
    const auto& [base, size] = it->second;
    const void* ptr = reinterpret_cast<const void*>(base);
    auto& meta = storage_meta[ptr];
    if (size > meta.max_size) {
      meta.max_size = size;
    }
  }

  std::unordered_map<const void*, uint64_t> storage_offsets;
  storage_offsets.reserve(storage_meta.size());

  cudaStream_t stream = nullptr;
  absl::Status stream_status = tensorcast::cuda::stream_create(&stream);
  if (!stream_status.ok()) {
    stream = nullptr;
  }

  std::unordered_map<std::string, uint64_t> tensor_offsets;
  tensor_offsets.reserve(tensor_names.size());

  for (const auto& name : tensor_names) {
    const uint64_t base = tensor_data.at(name).first;
    const void* ptr = reinterpret_cast<const void*>(base);

    const auto off_it = storage_offsets.find(ptr);
    if (off_it != storage_offsets.end()) {
      tensor_offsets[name] = off_it->second;
      continue;
    }

    cudaPointerAttributes attr;
    const absl::Status attr_status = tensorcast::cuda::pointer_get_attributes_full(ptr, &attr);
    if (!attr_status.ok()) {
      if (stream) {
        (void)tensorcast::cuda::stream_destroy(stream);
      }
      (void)writer.finalize();
      return attr_status;
    }
    const bool is_device_ptr = (attr.type == cudaMemoryTypeDevice);

    const uint64_t size_to_write = storage_meta.at(ptr).max_size;
    absl::StatusOr<uint64_t> offset_or = writer.write_tensor(ptr, size_to_write, is_device_ptr, stream);
    if (!offset_or.ok()) {
      if (stream) {
        (void)tensorcast::cuda::stream_destroy(stream);
      }
      (void)writer.finalize();
      return offset_or.status();
    }

    storage_offsets.emplace(ptr, offset_or.value());
    tensor_offsets[name] = offset_or.value();
  }

  if (stream) {
    absl::Status destroy_status = tensorcast::cuda::stream_destroy(stream);
    if (!destroy_status.ok()) {
      (void)writer.finalize();
      return destroy_status;
    }
  }

  absl::Status finalize_status = writer.finalize();
  if (!finalize_status.ok()) {
    return finalize_status;
  }

  return tensor_offsets;
}

namespace {

ViewSpec build_view_spec_from_py(const py::dict& spec_dict) {
  ViewSpec spec;
  for (const auto& item : spec_dict) {
    const std::string tensor_name = item.first.cast<std::string>();
    py::list ops_list = item.second.cast<py::list>();
    tensorcast::store::loader::TensorViewOps ops;
    for (py::handle handle : ops_list) {
      if (!py::isinstance<py::dict>(handle)) {
        PY_THROW_WITH_LOG(PyExc_TypeError, "View operation specification must be dictionaries with 'type' field");
      }
      py::dict op_dict = handle.cast<py::dict>();
      if (!op_dict.contains("type")) {
        PY_THROW_WITH_LOG(PyExc_KeyError, "View operation missing 'type'");
      }
      const std::string kind = op_dict["type"].cast<std::string>();
      if (kind == "narrow") {
        if (!op_dict.contains("dim") || !op_dict.contains("start") || !op_dict.contains("length")) {
          PY_THROW_WITH_LOG(PyExc_KeyError, "Narrow operation requires 'dim', 'start', and 'length'");
        }
        tensorcast::store::loader::NarrowOp narrow;
        narrow.dim = op_dict["dim"].cast<int32_t>();
        narrow.start = op_dict["start"].cast<int64_t>();
        narrow.length = op_dict["length"].cast<uint64_t>();
        ops.ops.push_back(ViewOp::Narrow(narrow));
      } else if (kind == "transpose") {
        if (!op_dict.contains("dim0") || !op_dict.contains("dim1")) {
          PY_THROW_WITH_LOG(PyExc_KeyError, "Transpose operation requires 'dim0' and 'dim1'");
        }
        tensorcast::store::loader::TransposeOp transpose;
        transpose.dim0 = op_dict["dim0"].cast<int32_t>();
        transpose.dim1 = op_dict["dim1"].cast<int32_t>();
        ops.ops.push_back(ViewOp::Transpose(transpose));
      } else {
        PY_THROW_WITH_LOG(PyExc_ValueError, absl::StrCat("Unsupported view operation: ", kind));
      }
    }
    spec.tensors.emplace(tensor_name, std::move(ops));
  }
  return spec;
}

py::dict tensor_transform_plan_to_dict(const TensorTransformPlan& plan) {
  py::dict out;
  out["tensor_name"] = plan.tensor_name;
  out["dst_offset"] = py::int_(plan.dst_offset);
  out["canonical_offset"] = py::int_(plan.canonical_offset);
  out["storage_offset_elements"] = py::int_(plan.storage_offset_elements);
  py::list canonical_shape;
  for (int64_t v : plan.canonical_shape) {
    canonical_shape.append(py::int_(v));
  }
  out["canonical_shape"] = canonical_shape;
  py::list canonical_stride;
  for (int64_t v : plan.canonical_stride) {
    canonical_stride.append(py::int_(v));
  }
  out["canonical_stride"] = canonical_stride;
  py::list view_shape;
  for (int64_t v : plan.view_shape) {
    view_shape.append(py::int_(v));
  }
  out["view_shape"] = view_shape;
  py::list view_stride;
  for (int64_t v : plan.view_stride) {
    view_stride.append(py::int_(v));
  }
  out["view_stride"] = view_stride;
  py::list permutation;
  for (int64_t v : plan.permutation) {
    permutation.append(py::int_(v));
  }
  out["permutation"] = permutation;
  out["dtype"] = plan.dtype;
  out["element_size_bytes"] = py::int_(plan.element_size_bytes);
  return out;
}

py::dict selection_range_to_dict(const tensorcast::store::loader::SelectionPlan::Range& range) {
  py::dict out;
  out["kind"] = range.kind == tensorcast::store::loader::SelectionPlan::Range::Kind::kData ? "data" : "pad";
  out["src_offset"] = py::int_(range.src_offset);
  out["dst_offset"] = py::int_(range.dst_offset);
  out["length"] = py::int_(range.length);
  return out;
}

py::dict compute_view_registration_plan_wrapper(py::bytes canonical_index_bytes, const py::dict& spec_dict) {
  const std::string canonical_index_json = canonical_index_bytes.cast<std::string>();
  ViewSpec spec = build_view_spec_from_py(spec_dict);

  absl::StatusOr<BidirectionalViewPlan> plan_or =
      ViewPlanner::compute_bidirectional_view_plan(canonical_index_json, spec);
  if (!plan_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, plan_or.status().ToString());
  }
  const BidirectionalViewPlan& plan = *plan_or;

  py::dict result;
  py::dict forward;
  forward["is_identity"] = plan.forward.is_identity;
  forward["view_size_bytes"] = py::int_(plan.forward.view_size_bytes);
  forward["view_index_json"] = py::bytes(plan.forward.view_index_json);
  forward["selection_total_bytes"] = py::int_(plan.forward.selection.total_bytes);
  forward["selection_is_contiguous"] = plan.forward.selection.is_contiguous;
  forward["selection_is_segment_aligned"] = plan.forward.selection.is_segment_aligned;
  py::list selection_ranges;
  for (const auto& range : plan.forward.selection.ranges) {
    selection_ranges.append(selection_range_to_dict(range));
  }
  forward["selection_ranges"] = selection_ranges;
  result["forward"] = forward;

  py::list write_chunks;
  for (const auto& chunk : plan.write.chunks) {
    py::dict chunk_dict;
    chunk_dict["canonical_offset"] = py::int_(chunk.canonical_offset);
    chunk_dict["view_offset"] = py::int_(chunk.view_offset);
    chunk_dict["length"] = py::int_(chunk.length);
    chunk_dict["segment_aligned"] = chunk.segment_aligned;
    write_chunks.append(chunk_dict);
  }
  result["write_chunks"] = write_chunks;

  py::dict inverse;
  inverse["requires_materialization"] = plan.inverse_transform.requires_materialization;
  py::list inverse_tensors;
  for (const TensorTransformPlan& tensor_plan : plan.inverse_transform.tensors) {
    inverse_tensors.append(tensor_transform_plan_to_dict(tensor_plan));
  }
  inverse["tensors"] = inverse_tensors;
  result["inverse_transform"] = inverse;

  return result;
}

} // namespace

// Wrapper function for generate_verification_info_from_disk
py::dict generate_artifact_verification_info_wrapper(const std::string& disk_path, int verification_level = 1) {
  try {
    ArtifactVerificationInfo info;
    {
      py::gil_scoped_release release;
      auto level = static_cast<VerificationLevel>(verification_level);
      info = generate_verification_info_from_disk(disk_path, level);
    }
    return verification_info_to_dict(info);
  } catch (const std::exception& e) {
    const auto str = std::string("Failed to generate verification info: ") + e.what();
    PY_THROW_WITH_LOG(PyExc_RuntimeError, str);
  }
}

// Wrapper function for GPU verification
bool verify_artifact_data_from_gpu_wrapper(
    int device_id,
    std::uint64_t cuda_memory_ptr,
    size_t memory_size,
    const py::dict& expected_verification,
    int verification_level) {
  try {
    ArtifactVerificationInfo expected_info = dict_to_verification_info(expected_verification);

    // Create data pointers and sizes for verification
    std::vector<void*> data_ptrs = {reinterpret_cast<void*>(cuda_memory_ptr)};
    std::vector<size_t> data_sizes = {memory_size};

    auto level = static_cast<VerificationLevel>(verification_level);
    absl::Status result;
    {
      py::gil_scoped_release release;
      result = ArtifactVerifier::verify_artifact_data(data_ptrs, data_sizes, expected_info, level, device_id);
    }

    if (result.ok()) {
      return true;
    }
    LOG(ERROR) << "GPU verification failed: " << result.message();
    return false;

  } catch (const std::exception& e) {
    const auto str = std::string("GPU verification failed: ") + e.what();
    PY_THROW_WITH_LOG(PyExc_RuntimeError, str);
  }
}

// Helper wrapper to get CUDA memory pointer with proper error handling
static uint64_t get_cuda_memory_ptr_wrapper(int device_id, const py::bytes& cuda_ipc_handle_py) {
  // Convert Python bytes to std::string handle
  const auto handle = cuda_ipc_handle_py.cast<std::string>();

  absl::StatusOr<std::uint64_t> ptr_or = get_cuda_memory_ptr(device_id, handle);
  if (!ptr_or.ok()) {
    const auto str = std::string("Failed to get CUDA memory pointer: ") + ptr_or.status().ToString();
    PY_THROW_WITH_LOG(PyExc_ValueError, str);
  }
  return ptr_or.value();
}

// Helper wrapper to close CUDA memory handle with error conversion
static bool close_cuda_memory_handle_wrapper(int device_id, std::uint64_t cuda_memory_ptr) {
  absl::Status status = close_cuda_memory_handle(device_id, cuda_memory_ptr);
  if (!status.ok()) {
    const auto str = std::string("Failed to close CUDA memory handle: ") + status.ToString();
    PY_THROW_WITH_LOG(PyExc_ValueError, str);
  }
  return true;
}

// ------------------------------------------------------------------
// Unified Save: write data partitions, tensor_index.json, artifact_descriptor.json
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

  absl::StatusOr<std::unordered_map<std::string, uint64_t>> offsets_or =
      write_tensor_data_partitions(tensor_names, tensor_data, path, writer_config);
  if (!offsets_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, offsets_or.status().ToString());
  }
  const auto& offsets = offsets_or.value();

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

  // Build Canonical Index JSON via C++ authority (stable grouping + 8B alignment invariant already respected by writer)
  std::vector<std::string> ordered_names = tensor_names;
  std::sort(ordered_names.begin(), ordered_names.end());
  std::unordered_map<std::string, tensorcast::store::loader::CanonicalTensorMeta> metas;
  metas.reserve(ordered_names.size());
  for (const auto& name : ordered_names) {
    if (!meta_state_dict.contains(name.c_str())) {
      const auto msg = std::string("meta_state_dict missing tensor: ") + name;
      PY_THROW_WITH_LOG(PyExc_RuntimeError, msg);
    }
    auto tpl = meta_state_dict[name.c_str()].cast<py::tuple>();
    if (tpl.size() != 4) {
      PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("meta_state_dict entry must be a 4-tuple"));
    }
    tensorcast::store::loader::CanonicalTensorMeta m;
    m.shape = tpl[0].cast<std::vector<int64_t>>();
    m.stride = tpl[1].cast<std::vector<int64_t>>();
    m.dtype = tpl[2].cast<std::string>();
    m.storage_offset = tpl[3].cast<uint64_t>();
    metas.emplace(name, std::move(m));
  }
  std::unordered_map<std::string, uint64_t> logical_sizes;
  logical_sizes.reserve(offset_max_size.size());
  for (const auto& name : ordered_names) {
    const auto off_it = offsets.find(name);
    if (off_it == offsets.end()) {
      continue;
    }
    logical_sizes[name] = offset_max_size.at(off_it->second);
  }
  absl::StatusOr<std::string> index_json_or =
      tensorcast::store::loader::build_canonical_index_json(ordered_names, offsets, logical_sizes, metas);
  if (!index_json_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, index_json_or.status().ToString());
  }
  const std::string& index_json = *index_json_or;
  nlohmann::json j = nlohmann::json::parse(index_json);
  const std::filesystem::path dir(path);
  std::filesystem::create_directories(dir);
  {
    std::ofstream out(dir / "tensor_index.json");
    if (!out.is_open()) {
      PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("Failed to write tensor_index.json"));
    }
    out << index_json;
    out.close();
  }
  // Additionally write CBOR encoding (preferred for canonical index persistence)
  {
    std::vector<std::uint8_t> cbor = nlohmann::json::to_cbor(j);
    std::ofstream oc(dir / "tensor_index.cbor", std::ios::binary);
    if (!oc.is_open()) {
      PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("Failed to write tensor_index.cbor"));
    }
    oc.write(reinterpret_cast<const char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
    oc.close();
  }

  // Compute descriptor fields
  absl::StatusOr<std::string> idx_mh_or =
      compute_index_multihash(std::optional<std::string>(index_json), /*index_key_hex=*/"");
  if (!idx_mh_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, idx_mh_or.status().ToString());
  }
  // Compute data multihash via unified SeekableSource pipeline, with empty-artifact handling
  auto compute_mh_via_source = [&](const std::string& dir_path) -> absl::StatusOr<std::string> {
    namespace fs = std::filesystem;
    fs::path dir(dir_path);

    // Determine logical total size from canonical index JSON first
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

    // Empty artifact: no bytes to hash → define data_multihash deterministically
    if (total_size == 0) {
      const std::vector<std::vector<uint8_t>> empty_leaves;
      std::vector<uint8_t> root = tensorcast::common::compute_tree_hash_root_sha256(empty_leaves);
      return tensorcast::common::multibase_multihash_sha256(root);
    }

    // Collect partition files deterministically
    std::vector<fs::path> parts;
    for (const auto& entry : fs::directory_iterator(dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto name = entry.path().filename().string();
      if (name.starts_with("tensor.data_")) {
        parts.push_back(entry.path());
      }
    }
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    if (parts.empty()) {
      fs::path single = dir / "tensor.data";
      if (fs::exists(single)) {
        parts.push_back(single);
      }
    }
    if (parts.empty()) {
      return absl::NotFoundError("No tensor.data partitions found");
    }

    tensorcast::store::loader::FilePartitionSource::Options opts;
    for (const auto& p : parts) {
      opts.partition_paths.push_back(p);
      opts.partition_sizes.push_back(static_cast<size_t>(fs::file_size(p)));
    }
    opts.total_size = total_size;
    opts.io_batch_bytes = 128 * 1024 * 1024;
    tensorcast::store::loader::FilePartitionSource src(std::move(opts));
    return tensorcast::store::loader::compute_data_multihash_from_seekable_source(src, total_size);
  };
  absl::StatusOr<std::string> data_mh_or = compute_mh_via_source(path);
  if (!data_mh_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, data_mh_or.status().ToString());
  }

  // Compute total size: max(offset+size)
  uint64_t total_size = 0;
  for (const auto& [off, sz] : offset_max_size) {
    total_size = std::max<uint64_t>(total_size, off + sz);
  }

  nlohmann::json desc;
  desc["artifact_id"] = std::string("mi2:") + *idx_mh_or + std::string(":") + *data_mh_or;
  desc["index_multihash"] = *idx_mh_or;
  desc["data_multihash"] = *data_mh_or;
  desc["schema_version"] = "v3";
  desc["encoding"] = "json";
  desc["total_size"] = total_size;
  nlohmann::json hash_params;
  // Hash leaf size (protocol constant)
  hash_params["chunk_size"] = tensorcast::common::consts::kHashLeafBytes;
  hash_params["fanout"] = 2;
  desc["hash_params"] = hash_params;

  {
    std::ofstream out(dir / "artifact_descriptor.json");
    if (!out.is_open()) {
      PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("Failed to write artifact_descriptor.json"));
    }
    out << desc.dump(2);
    out.close();
  }

  // Return descriptor to Python
  py::dict result;
  result["artifact_id"] = desc["artifact_id"].get<std::string>();
  result["index_multihash"] = desc["index_multihash"].get<std::string>();
  result["data_multihash"] = desc["data_multihash"].get<std::string>();
  result["schema_version"] = desc["schema_version"].get<std::string>();
  result["encoding"] = desc["encoding"].get<std::string>();
  result["total_size"] = total_size;
  return result;
}

static py::dict inspect_or_generate_descriptor_wrapper(const std::string& path) {
  const std::filesystem::path dir(path);
  const auto desc_path = dir / "artifact_descriptor.json";
  if (std::filesystem::exists(desc_path)) {
    std::ifstream in(desc_path);
    if (!in.is_open()) {
      PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("Failed to open artifact_descriptor.json"));
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    nlohmann::json j = nlohmann::json::parse(buffer.str());
    py::dict result;
    result["artifact_id"] = j["artifact_id"].get<std::string>();
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
    PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("tensor_index.json not found"));
  }
  std::stringstream idx_ss;
  idx_ss << idx_in.rdbuf();
  idx_in.close();
  const std::string index_json = idx_ss.str();

  absl::StatusOr<std::string> idx_mh_or =
      compute_index_multihash(std::optional<std::string>(index_json), /*index_key_hex=*/"");
  if (!idx_mh_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, idx_mh_or.status().ToString());
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
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    if (parts.empty()) {
      fs::path single = dir2 / "tensor.data";
      if (fs::exists(single)) {
        parts.push_back(single);
      }
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
    // Empty artifact: produce deterministic data multihash without requiring partitions
    if (total_size2 == 0) {
      const std::vector<std::vector<uint8_t>> empty_leaves;
      std::vector<uint8_t> root = tensorcast::common::compute_tree_hash_root_sha256(empty_leaves);
      return tensorcast::common::multibase_multihash_sha256(root);
    }
    if (parts.empty()) {
      return absl::NotFoundError("No tensor.data partitions found");
    }
    tensorcast::store::loader::FilePartitionSource::Options opts2;
    for (const auto& p : parts) {
      opts2.partition_paths.push_back(p);
      opts2.partition_sizes.push_back(static_cast<size_t>(fs::file_size(p)));
    }
    opts2.total_size = total_size2;
    opts2.io_batch_bytes = 128 * 1024 * 1024;
    tensorcast::store::loader::FilePartitionSource src2(std::move(opts2));
    return tensorcast::store::loader::compute_data_multihash_from_seekable_source(src2, total_size2);
  };
  absl::StatusOr<std::string> data_mh_or = compute_mh_via_source2(path);
  if (!data_mh_or.ok()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, data_mh_or.status().ToString());
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
  desc["artifact_id"] = std::string("mi2:") + *idx_mh_or + std::string(":") + *data_mh_or;
  desc["index_multihash"] = *idx_mh_or;
  desc["data_multihash"] = *data_mh_or;
  desc["schema_version"] = "v3";
  desc["encoding"] = "json";
  desc["total_size"] = total_size;
  nlohmann::json hash_params;
  // Hash leaf size (protocol constant)
  hash_params["chunk_size"] = tensorcast::common::consts::kHashLeafBytes;
  hash_params["fanout"] = 2;
  desc["hash_params"] = hash_params;

  std::ofstream out(desc_path);
  if (!out.is_open()) {
    PY_THROW_WITH_LOG(PyExc_RuntimeError, std::string("Failed to write artifact_descriptor.json"));
  }
  out << desc.dump(2);
  out.close();

  py::dict result;
  result["artifact_id"] = desc["artifact_id"].get<std::string>();
  result["index_multihash"] = desc["index_multihash"].get<std::string>();
  result["data_multihash"] = desc["data_multihash"].get<std::string>();
  result["schema_version"] = desc["schema_version"].get<std::string>();
  result["encoding"] = desc["encoding"].get<std::string>();
  result["total_size"] = total_size;
  return result;
}

namespace {

struct StorageAccumulator {
  int device_id;
  uint64_t base_ptr;
  uint64_t size_bytes;
};

std::string make_storage_id(int device_id, uint64_t base_ptr, uint64_t size_bytes) {
  return absl::StrFormat("%d:%016x:%016x", device_id, base_ptr, size_bytes);
}

} // namespace

static py::dict collect_tensor_storage_graph(const py::dict& tensors) {
  absl::flat_hash_map<std::string, StorageAccumulator> storage_map;
  py::dict aliases;
  py::dict tensor_meta_index;
  py::dict tensor_source_index;

  for (auto item : tensors) {
    const std::string name = py::cast<std::string>(item.first);
    py::handle tensor_obj = item.second;
    at::Tensor tensor = tensor_obj.cast<at::Tensor>();
    if (!tensor.defined()) {
      PY_THROW_WITH_LOG(PyExc_ValueError, absl::StrCat("Tensor for '", name, "' is not defined"));
    }

    int device_id = -1;
    if (tensor.is_cuda()) {
      device_id = tensor.get_device();
    } else if (!tensor.device().is_cpu()) {
      PY_THROW_WITH_LOG(
          PyExc_ValueError, absl::StrCat("Unsupported device for tensor '", name, "': ", tensor.device().str()));
    }

    const uint64_t storage_offset = static_cast<uint64_t>(tensor.storage_offset());
    auto storage = tensor.storage();
    auto* storage_impl = storage.unsafeGetStorageImpl();
    if (storage_impl == nullptr) {
      PY_THROW_WITH_LOG(PyExc_RuntimeError, absl::StrCat("Missing storage for tensor '", name, "'"));
    }
    const uint64_t base_ptr = reinterpret_cast<uint64_t>(storage_impl->data_ptr().get());
    const uint64_t storage_size_bytes = static_cast<uint64_t>(storage_impl->nbytes());
    const uint64_t logical_length = static_cast<uint64_t>(tensor.nbytes());
    const std::string storage_id = make_storage_id(device_id, base_ptr, storage_size_bytes);

    storage_map.try_emplace(
        storage_id, StorageAccumulator{.device_id = device_id, .base_ptr = base_ptr, .size_bytes = storage_size_bytes});

    std::vector<int64_t> shape_vec(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int64_t> stride_vec(tensor.strides().begin(), tensor.strides().end());
    py::list shape_py;
    py::list stride_py;
    for (auto v : shape_vec) {
      shape_py.append(static_cast<int64_t>(v));
    }
    for (auto v : stride_vec) {
      stride_py.append(static_cast<int64_t>(v));
    }
    std::string dtype_str = py::str(tensor_obj.attr("dtype")).cast<std::string>();

    py::dict alias_entry;
    alias_entry["name"] = name;
    alias_entry["storage_id"] = storage_id;
    alias_entry["storage_offset"] = storage_offset;
    alias_entry["logical_length"] = logical_length;
    alias_entry["shape"] = shape_py;
    alias_entry["stride"] = stride_py;
    alias_entry["dtype"] = dtype_str;
    aliases[name.c_str()] = alias_entry;

    tensor_meta_index[name.c_str()] = py::make_tuple(shape_py, stride_py, dtype_str, storage_offset);
    tensor_source_index[name.c_str()] = py::make_tuple(static_cast<uint64_t>(base_ptr), storage_size_bytes);
  }

  py::dict storages;
  for (const auto& [storage_id, entry] : storage_map) {
    py::dict storage_entry;
    storage_entry["storage_id"] = storage_id;
    storage_entry["device_id"] = entry.device_id;
    storage_entry["base_ptr"] = py::int_(entry.base_ptr);
    storage_entry["size_bytes"] = py::int_(entry.size_bytes);
    storages[storage_id.c_str()] = storage_entry;
  }

  py::dict result;
  result["storages"] = storages;
  result["aliases"] = aliases;
  result["tensor_meta_index"] = tensor_meta_index;
  result["tensor_source_index"] = tensor_source_index;
  return result;
}

// ------------------------------------------------------------------
// Expose canonical index builder for safetensors directories
// ------------------------------------------------------------------
static py::bytes build_canonical_index_from_safetensors_wrapper(const std::string& dir_path) {
  namespace fs = std::filesystem;
  fs::path dir(dir_path);
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    const auto msg = std::string("Invalid artifact directory for safetensors: ") + dir_path;
    PY_THROW_WITH_LOG(PyExc_RuntimeError, msg);
  }
  std::vector<fs::path> st_files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto name = entry.path().filename().string();
    if (name.size() > 12 && name.substr(name.size() - 12) == ".safetensors") {
      st_files.push_back(entry.path());
    }
  }
  std::sort(st_files.begin(), st_files.end());
  if (st_files.empty()) {
    const auto msg = std::string("No .safetensors files found under: ") + dir_path;
    PY_THROW_WITH_LOG(PyExc_RuntimeError, msg);
  }

  absl::StatusOr<std::string> idx_bytes_or = tensorcast::store::loader::BuildCanonicalIndexFromSafetensors(st_files);
  if (!idx_bytes_or.ok()) {
    PY_THROW_WITH_LOG(
        PyExc_RuntimeError,
        std::string("BuildCanonicalIndexFromSafetensors failed: ") + idx_bytes_or.status().ToString());
  }
  const std::string& bytes = idx_bytes_or.value();
  return py::bytes(bytes);
}

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Initialize logging only once across all modules
  tensorcast::common::ensure_logging_initialized();

  // Expose build configuration for runtime validation
  m.def(
      "is_fake_cuda",
      []() -> bool {
        const bool fake = tensorcast::cuda::is_fake();
        if (fake) {
          static std::once_flag warn_once;
          std::call_once(warn_once, []() {
            py::module_::import("warnings")
                .attr("warn")("TensorCast Fake CUDA backend active (TEST ONLY). GPU operations are simulated.");
          });
        }
        return fake;
      },
      "Returns True if the runtime Fake CUDA backend is active");

  m.def("save_tensors", &tensorcast::checkpoint::save_tensors, "Save a state dict")
      .def(
          "save_model_to_disk",
          &save_model_to_disk_wrapper,
          py::arg("tensor_names"),
          py::arg("tensor_data"),
          py::arg("meta_state_dict"),
          py::arg("path"),
          py::arg("config") = py::dict(),
          "Unified save: writes data, tensor_index.json, artifact_descriptor.json and returns descriptor")
      .def(
          "compute_view_registration_plan",
          &compute_view_registration_plan_wrapper,
          py::arg("canonical_index_json"),
          py::arg("view_ops"),
          "Compute bidirectional view registration plan using the core ViewPlanner")
      .def("restore_tensors", &tensorcast::checkpoint::restore_tensors, "Restore a state dict")
      .def(
          "restore_tensors_from_disk",
          &tensorcast::checkpoint::restore_tensors_from_disk,
          py::arg("meta_state_dict"),
          py::arg("disk_path"),
          py::arg("tensor_device_offsets"),
          py::arg("device_id") = -1,
          "Restore a state dict from artifact path, device_id=-1 means load to CPU, otherwise CUDA device id")
      .def(
          "allocate_cuda_memory",
          &tensorcast::checkpoint::allocate_cuda_memory,
          py::arg("device_id"),
          py::arg("size"),
          "Allocate CUDA memory on a single device")
      .def(
          "get_cuda_memory_handle",
          [](int device_id, std::uint64_t memory_ptr_int) {
            const std::string handle = tensorcast::checkpoint::get_cuda_memory_handle(device_id, memory_ptr_int);
            return py::bytes(handle);
          },
          py::arg("device_id"),
          py::arg("memory_ptr"),
          "Get a CUDA IPC memory handle for a single allocation")
      .def("get_device_uuid_map", &tensorcast::checkpoint::get_device_uuid_map, "Get device uuid map")
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
          "generate_artifact_verification_info",
          &generate_artifact_verification_info_wrapper,
          py::arg("disk_path"),
          py::arg("verification_level") = 1,
          "Generate artifact verification information from saved artifact files")
      .def(
          "verify_artifact_data_from_gpu",
          &verify_artifact_data_from_gpu_wrapper,
          py::arg("device_id"),
          py::arg("cuda_memory_ptr"),
          py::arg("memory_size"),
          py::arg("expected_verification"),
          py::arg("verification_level"),
          "Verify artifact data integrity from GPU memory")
      .def(
          "collect_tensor_storage_graph",
          &collect_tensor_storage_graph,
          py::arg("tensors"),
          "Collect deduplicated storage metadata and tensor alias information");

  m.def(
      "inspect_or_generate_descriptor",
      &inspect_or_generate_descriptor_wrapper,
      py::arg("disk_path"),
      "Return artifact descriptor if present; otherwise compute multihashes and write artifact_descriptor.json");

  m.def(
      "build_canonical_index_from_safetensors",
      &build_canonical_index_from_safetensors_wrapper,
      py::arg("artifact_dir"),
      "Build canonical RFC-0007 index JSON bytes from a directory of .safetensors files");
}
