// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::cuda {

struct IpcHandleBytes {
  static constexpr size_t kHandleSize = sizeof(cudaIpcMemHandle_t);
  std::array<std::uint8_t, kHandleSize> bytes{};

  [[nodiscard]] bool is_valid() const;
  [[nodiscard]] cudaIpcMemHandle_t to_native() const;
  static IpcHandleBytes from_native(const cudaIpcMemHandle_t& handle);
  [[nodiscard]] absl::Span<const std::uint8_t> as_bytes() const;
  [[nodiscard]] absl::string_view as_string_view() const;
};

struct OpenOptions {
  unsigned int flags = cudaIpcMemLazyEnablePeerAccess;
};

class IpcMapping {
 public:
  IpcMapping() = default;
  explicit IpcMapping(void* ptr);
  IpcMapping(IpcMapping&&) noexcept;
  IpcMapping& operator=(IpcMapping&&) noexcept;
  ~IpcMapping();

  IpcMapping(const IpcMapping&) = delete;
  IpcMapping& operator=(const IpcMapping&) = delete;

  [[nodiscard]] void* get() const;
  void reset();

  static absl::StatusOr<IpcMapping> open(const IpcHandleBytes& handle, OpenOptions opts = {});
  static absl::StatusOr<IpcMapping> open(cudaIpcMemHandle_t handle, OpenOptions opts = {});
  static absl::StatusOr<IpcMapping> open(absl::Span<const std::uint8_t> bytes, OpenOptions opts = {});
  static absl::StatusOr<IpcMapping> open(absl::string_view bytes, OpenOptions opts = {});

 private:
  void* ptr_{nullptr};
};

} // namespace tensorcast::cuda
