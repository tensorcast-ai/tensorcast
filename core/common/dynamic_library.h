// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "absl/status/statusor.h"

namespace tensorcast::common {

class DynamicLibrary {
 public:
  static absl::StatusOr<DynamicLibrary> Open(std::string_view library_name);

  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary();

  absl::StatusOr<void*> ResolveSymbol(std::string_view symbol_name) const;

  bool is_loaded() const {
    return handle_ != nullptr;
  }

  const std::string& name() const {
    return name_;
  }

 private:
  explicit DynamicLibrary(void* handle, std::string name);

  void* handle_ = nullptr;
  std::string name_;
};

} // namespace tensorcast::common
