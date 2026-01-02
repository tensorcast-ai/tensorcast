// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/dynamic_library.h"

#include <dlfcn.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::common {
namespace {

absl::StatusOr<void*> resolve_symbol(void* handle, std::string_view symbol_name) {
  if (handle == nullptr) {
    return absl::FailedPreconditionError("dynamic library handle is null");
  }

  dlerror();
  void* symbol = dlsym(handle, std::string(symbol_name).c_str());
  const char* error = dlerror();
  if (symbol == nullptr || error != nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Failed to resolve symbol ", symbol_name, ": ", error != nullptr ? error : "symbol not found"));
  }
  return symbol;
}

} // namespace

DynamicLibrary::DynamicLibrary(void* handle, std::string name) : handle_(handle), name_(std::move(name)) {}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept {
  handle_ = other.handle_;
  name_ = std::move(other.name_);
  other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
  handle_ = other.handle_;
  name_ = std::move(other.name_);
  other.handle_ = nullptr;
  return *this;
}

DynamicLibrary::~DynamicLibrary() {
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
}

absl::StatusOr<DynamicLibrary> DynamicLibrary::Open(std::string_view library_name) {
  dlerror();
  void* handle = dlopen(std::string(library_name).c_str(), RTLD_NOW | RTLD_LOCAL);
  const char* error = dlerror();
  if (handle == nullptr || error != nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Failed to load ", library_name, ": ", error != nullptr ? error : "dlopen failed"));
  }
  return DynamicLibrary(handle, std::string(library_name));
}

absl::StatusOr<void*> DynamicLibrary::ResolveSymbol(std::string_view symbol_name) const {
  return resolve_symbol(handle_, symbol_name);
}

} // namespace tensorcast::common
