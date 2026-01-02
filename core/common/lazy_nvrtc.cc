// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/lazy_nvrtc.h"

#include <array>

#include "absl/base/no_destructor.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::cuda {
namespace {

constexpr std::array<const char*, 2> kNvrtcLibraryNames = {"libnvrtc.so.12", "libnvrtc.so"};

} // namespace

LazyNvrtc& LazyNvrtc::get() {
  static absl::NoDestructor<LazyNvrtc> nvrtc;
  return *nvrtc;
}

absl::Status LazyNvrtc::ensure_loaded() {
  std::call_once(load_once_, [&]() {
    absl::Status status = open_library();
    absl::MutexLock lock(&mutex_);
    library_status_ = status;
  });

  absl::MutexLock lock(&mutex_);
  return library_status_;
}

absl::Status LazyNvrtc::open_library() {
  std::string errors;
  for (const char* library_name : kNvrtcLibraryNames) {
    absl::StatusOr<common::DynamicLibrary> lib_or = common::DynamicLibrary::open(library_name);
    if (lib_or.ok()) {
      absl::MutexLock lock(&mutex_);
      library_ = std::move(lib_or.value());
      return absl::OkStatus();
    }
    if (!errors.empty()) {
      absl::StrAppend(&errors, "; ");
    }
    absl::StrAppend(&errors, lib_or.status().message());
  }
  return absl::UnavailableError(
      absl::StrCat("Failed to load NVRTC library (tried libnvrtc.so.12, libnvrtc.so): ", errors));
}

#define TENSORCAST_DEFINE_NVRTC_ACCESSOR(name)          \
  absl::StatusOr<decltype(&::name)> LazyNvrtc::name() { \
    return resolve_symbol(#name, &api_.name);           \
  }

TENSORCAST_NVRTC_SYMBOLS(TENSORCAST_DEFINE_NVRTC_ACCESSOR)

#undef TENSORCAST_DEFINE_NVRTC_ACCESSOR

} // namespace tensorcast::cuda
