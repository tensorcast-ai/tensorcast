// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_transform_executor.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "core/common/memory/memory_location.h"
#include "core/cuda/cuda_api.h"

#include <torch/torch.h>

namespace tensorcast::store::loader {
namespace {

absl::StatusOr<at::ScalarType> to_scalar_type(std::string_view dtype) {
  if (dtype == "torch.float16") {
    return at::kHalf;
  }
  if (dtype == "torch.bfloat16") {
    return at::kBFloat16;
  }
  if (dtype == "torch.float8_e4m3fn") {
    return at::kFloat8_e4m3fn;
  }
  if (dtype == "torch.float8_e5m2") {
    return at::kFloat8_e5m2;
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
  return absl::InvalidArgumentError(absl::StrCat("Unsupported dtype for view transform: ", dtype));
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

} // namespace

absl::Status execute_transform(
    const TransformPlan& plan,
    common::memory::MemoryLocation location,
    void* base_ptr,
    int device_id) {
  if (plan.empty() || !plan.requires_materialization) {
    return absl::OkStatus();
  }

  if (base_ptr == nullptr) {
    return absl::InvalidArgumentError("execute_transform requires non-null base pointer");
  }

  const bool is_gpu = (location == common::memory::MemoryLocation::GPU);
  if (is_gpu && !cuda::is_fake() && device_id < 0) {
    return absl::InvalidArgumentError("execute_transform (GPU) requires a valid CUDA device id");
  }

  torch::NoGradGuard no_grad;

  auto* byte_base = static_cast<std::uint8_t*>(base_ptr);

  for (const TensorTransformPlan& tensor_plan : plan.tensors) {
    if (tensor_plan.permutation.empty()) {
      continue; // Defensive: nothing to permute
    }

    auto scalar_or = to_scalar_type(tensor_plan.dtype);
    if (!scalar_or.ok()) {
      return scalar_or.status();
    }
    const at::ScalarType scalar_type = *scalar_or;

    torch::TensorOptions options = create_tensor_options_for_location(scalar_type, location, device_id);

    std::uint8_t* tensor_base = byte_base + tensor_plan.dst_offset;
    std::uint8_t* canonical_ptr = tensor_base + tensor_plan.storage_offset_elements * tensor_plan.element_size_bytes;

    auto canonical_tensor = torch::from_blob(
        canonical_ptr,
        torch::IntArrayRef(tensor_plan.canonical_shape),
        torch::IntArrayRef(tensor_plan.canonical_stride),
        [](void*) {},
        options);

    auto permutation_ref = torch::IntArrayRef(tensor_plan.permutation);
    torch::Tensor permuted = canonical_tensor.permute(permutation_ref);
    torch::Tensor contiguous = permuted.contiguous();

    auto destination_tensor = torch::from_blob(
        tensor_base,
        torch::IntArrayRef(tensor_plan.view_shape),
        torch::IntArrayRef(tensor_plan.view_stride),
        [](void*) {},
        options);

    destination_tensor.copy_(contiguous);
  }

  return absl::OkStatus();
}

} // namespace tensorcast::store::loader
