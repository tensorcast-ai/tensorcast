// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_ingest_executor.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "core/cuda/cuda_api.h"
#include "torch/torch.h"

namespace tensorcast::store::loader {
namespace {

absl::StatusOr<at::ScalarType> to_scalar_type(std::string_view dtype) {
  if (dtype == "torch.float16") {
    return at::kHalf;
  }
  if (dtype == "torch.bfloat16") {
    return at::kBFloat16;
  }
  if (dtype == "torch.float32" || dtype == "torch.float") {
    return at::kFloat;
  }
  if (dtype == "torch.float64" || dtype == "torch.double") {
    return at::kDouble;
  }
  if (dtype == "torch.int8") {
    return at::kChar;
  }
  if (dtype == "torch.uint8") {
    return at::kByte;
  }
  if (dtype == "torch.int16") {
    return at::kShort;
  }
  if (dtype == "torch.int32") {
    return at::kInt;
  }
  if (dtype == "torch.int64" || dtype == "torch.long") {
    return at::kLong;
  }
  if (dtype == "torch.bool") {
    return at::kBool;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unsupported dtype for view ingestion: ", dtype));
}

torch::TensorOptions create_tensor_options_for_location(
    at::ScalarType scalar_type,
    common::memory::MemoryLocation location,
    int device_id) {
  if (location == common::memory::MemoryLocation::GPU) {
    if (cuda::is_fake()) {
      return torch::TensorOptions().device(torch::kCPU).dtype(scalar_type);
    }
    return torch::TensorOptions().device(torch::kCUDA, device_id).dtype(scalar_type);
  }
  return torch::TensorOptions().device(torch::kCPU).dtype(scalar_type);
}

absl::Status execute_inverse_transform(
    const TransformPlan& plan,
    common::memory::MemoryLocation location,
    void* base_ptr,
    int device_id) {
  if (plan.empty() || !plan.requires_materialization) {
    return absl::OkStatus();
  }
  if (base_ptr == nullptr) {
    return absl::InvalidArgumentError("execute_inverse_transform requires non-null base pointer");
  }
  const bool is_gpu = (location == common::memory::MemoryLocation::GPU);
  if (is_gpu && !cuda::is_fake() && device_id < 0) {
    return absl::InvalidArgumentError("execute_inverse_transform (GPU) requires a valid CUDA device id");
  }

  torch::NoGradGuard no_grad;
  auto* byte_base = static_cast<std::uint8_t*>(base_ptr);

  for (const TensorTransformPlan& tensor_plan : plan.tensors) {
    if (tensor_plan.permutation.empty()) {
      continue;
    }
    auto scalar_or = to_scalar_type(tensor_plan.dtype);
    if (!scalar_or.ok()) {
      return scalar_or.status();
    }
    const at::ScalarType scalar_type = *scalar_or;
    torch::TensorOptions options = create_tensor_options_for_location(scalar_type, location, device_id);

    std::uint8_t* tensor_base = byte_base + tensor_plan.dst_offset;
    std::uint8_t* canonical_ptr = tensor_base + tensor_plan.storage_offset_elements * tensor_plan.element_size_bytes;

    auto view_tensor = torch::from_blob(
        canonical_ptr,
        torch::IntArrayRef(tensor_plan.view_shape),
        torch::IntArrayRef(tensor_plan.view_stride),
        [](void*) {},
        options);

    auto permutation_ref = torch::IntArrayRef(tensor_plan.permutation);
    torch::Tensor permuted = view_tensor.permute(permutation_ref);
    torch::Tensor contiguous = permuted.contiguous();

    auto destination_tensor = torch::from_blob(
        canonical_ptr,
        torch::IntArrayRef(tensor_plan.canonical_shape),
        torch::IntArrayRef(tensor_plan.canonical_stride),
        [](void*) {},
        options);

    destination_tensor.copy_(contiguous);
  }

  return absl::OkStatus();
}

} // namespace

ViewIngestExecutor::ViewIngestExecutor(ViewWritePlan write_plan, TransformPlan inverse_transform)
    : inverse_transform_(std::move(inverse_transform)) {
  chunks_.reserve(write_plan.chunks.size());
  for (auto& chunk : write_plan.chunks) {
    total_view_bytes_ += chunk.length;
    ChunkState state;
    state.chunk = std::move(chunk);
    state.written = 0;
    chunks_.push_back(std::move(state));
  }
  if (chunks_.empty()) {
    current_chunk_idx_ = 0;
  }
}

absl::Status ViewIngestExecutor::copy_into_canonical(
    ChunkState& chunk,
    uint64_t chunk_offset_bytes,
    absl::Span<const std::byte> data,
    common::memory::MemoryLocation location,
    void* canonical_base_ptr,
    int device_id) {
  if (data.empty()) {
    return absl::OkStatus();
  }

  const uint64_t canonical_offset = chunk.chunk.canonical_offset + chunk_offset_bytes;
  auto* dst = static_cast<std::byte*>(canonical_base_ptr) + static_cast<std::ptrdiff_t>(canonical_offset);

  switch (location) {
    case common::memory::MemoryLocation::CPU:
      std::memcpy(dst, data.data(), data.size());
      return absl::OkStatus();
    case common::memory::MemoryLocation::GPU: {
      auto copy_status = cuda::memcpy(dst, data.data(), data.size(), cudaMemcpyHostToDevice);
      return copy_status;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat(
              "unsupported memory location for view ingestion: ", common::memory::location_to_string(location)));
  }
}

absl::Status ViewIngestExecutor::ingest_chunk(
    uint64_t view_offset,
    absl::Span<const std::byte> data,
    common::memory::MemoryLocation location,
    void* canonical_base_ptr,
    int device_id) {
  if (finalized_) {
    return absl::FailedPreconditionError("view ingestion already finalized");
  }
  if (canonical_base_ptr == nullptr) {
    return absl::InvalidArgumentError("canonical_base_ptr must not be null");
  }
  if (data.empty()) {
    return absl::OkStatus();
  }
  if (location == common::memory::MemoryLocation::GPU) {
    if (!cuda::is_fake() && device_id < 0) {
      return absl::InvalidArgumentError("device_id must be >= 0 for GPU ingestion");
    }
    auto set_status = cuda::set_device(device_id);
    if (!set_status.ok()) {
      return set_status;
    }
  }

  const size_t total = data.size();
  if (current_chunk_idx_ >= chunks_.size()) {
    return absl::InvalidArgumentError("ingested bytes exceed planned view size");
  }

  size_t idx = current_chunk_idx_;
  size_t processed = 0;

  while (processed < total) {
    if (idx >= chunks_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("view chunk exceeds planned ranges: offset=", view_offset, " length=", total));
    }
    ChunkState& chunk = chunks_[idx];
    const uint64_t expected_view_offset = chunk.chunk.view_offset + chunk.written;
    const uint64_t absolute_offset = view_offset + processed;
    if (absolute_offset < expected_view_offset) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "view chunk overlaps already ingested data: offset=",
              absolute_offset,
              " expected=",
              expected_view_offset));
    }
    if (absolute_offset > expected_view_offset) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "view chunk violates contiguity requirement: offset=",
              absolute_offset,
              " expected=",
              expected_view_offset));
    }

    const uint64_t chunk_remaining = chunk.chunk.length - chunk.written;
    const uint64_t data_remaining = static_cast<uint64_t>(total - processed);
    const uint64_t step = std::min(chunk_remaining, data_remaining);
    if (step > std::numeric_limits<size_t>::max()) {
      return absl::OutOfRangeError("view chunk size exceeds platform limits");
    }
    const size_t to_copy = static_cast<size_t>(step);
    absl::Span<const std::byte> slice = data.subspan(processed, to_copy);

    auto copy_status = copy_into_canonical(chunk, chunk.written, slice, location, canonical_base_ptr, device_id);
    if (!copy_status.ok()) {
      return copy_status;
    }

    chunk.written += step;
    processed += to_copy;
    ingested_bytes_ += step;
    if (ingested_bytes_ > total_view_bytes_) {
      return absl::InvalidArgumentError("ingested bytes exceed planned view size");
    }

    if (chunk.written == chunk.chunk.length) {
      ++idx;
    }
  }

  current_chunk_idx_ = idx;
  return absl::OkStatus();
}

absl::Status ViewIngestExecutor::finalize(
    common::memory::MemoryLocation location,
    void* canonical_base_ptr,
    int device_id) {
  if (finalized_) {
    return absl::FailedPreconditionError("view ingestion already finalized");
  }
  if (!is_complete()) {
    return absl::FailedPreconditionError("view bytes incomplete; cannot finalize");
  }
  if (canonical_base_ptr == nullptr) {
    return absl::InvalidArgumentError("canonical_base_ptr must not be null");
  }
  if (location == common::memory::MemoryLocation::GPU) {
    if (!cuda::is_fake() && device_id < 0) {
      return absl::InvalidArgumentError("device_id must be >= 0 for GPU ingestion");
    }
    auto set_status = cuda::set_device(device_id);
    if (!set_status.ok()) {
      return set_status;
    }
  }

  auto transform_status = execute_inverse_transform(inverse_transform_, location, canonical_base_ptr, device_id);
  if (!transform_status.ok()) {
    return transform_status;
  }

  finalized_ = true;
  return absl::OkStatus();
}

} // namespace tensorcast::store::loader
