// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_backend.h"

#include <cstdlib>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::cuda {
namespace {

constexpr char kCudaBackendEnv[] = "TENSORCAST_CUDA_BACKEND";

struct TestEnvSignals {
  bool test_srcdir = false;
  bool test_tmpdir = false;
  bool pytest_current = false;

  bool is_test_environment() const {
    return test_srcdir || test_tmpdir || pytest_current;
  }
};

TestEnvSignals detect_test_env_signals() {
  TestEnvSignals signals;
  signals.test_srcdir = std::getenv("TEST_SRCDIR") != nullptr;
  signals.test_tmpdir = std::getenv("TEST_TMPDIR") != nullptr;
  signals.pytest_current = std::getenv("PYTEST_CURRENT_TEST") != nullptr;
  return signals;
}

std::string format_test_env_banner(const TestEnvSignals& signals) {
  return absl::StrCat(
      "Detected test env: ",
      signals.is_test_environment() ? "yes" : "no",
      " (TEST_SRCDIR=",
      signals.test_srcdir ? "set" : "unset",
      ", TEST_TMPDIR=",
      signals.test_tmpdir ? "set" : "unset",
      ", PYTEST_CURRENT_TEST=",
      signals.pytest_current ? "set" : "unset",
      ")");
}

absl::Status select_cuda_backend_from_env(CudaBackend** backend_out) {
  const char* env_value = std::getenv(kCudaBackendEnv);
  if (env_value == nullptr || env_value[0] == '\0' || std::string_view(env_value) == "real") {
    static absl::NoDestructor<RealCudaBackend> real_backend;
    *backend_out = real_backend.get();
    return absl::OkStatus();
  }

  if (std::string_view(env_value) == "fake") {
    const TestEnvSignals signals = detect_test_env_signals();
    if (!signals.is_test_environment()) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "TENSORCAST_CUDA_BACKEND=fake is restricted to test environments. ", format_test_env_banner(signals)));
    }

    LOG(ERROR) << "==================== TENSORCAST FAKE CUDA ENABLED ====================\n"
               << "This mode is TEST ONLY. GPU operations are simulated and may be incorrect\n"
               << "for performance, concurrency, and driver/IPC semantics.\n"
               << "TENSORCAST_CUDA_BACKEND=fake\n"
               << format_test_env_banner(signals) << "\n"
               << "======================================================================";

    static absl::NoDestructor<FakeCudaBackend> fake_backend;
    *backend_out = fake_backend.get();
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Invalid value for ", kCudaBackendEnv, ": ", env_value, " (expected 'real' or 'fake')"));
}

absl::StatusOr<CudaBackend*> get_backend() {
  static std::once_flag select_once;
  static absl::Status selection_status = absl::OkStatus();
  static CudaBackend* selected_backend = nullptr;

  std::call_once(select_once, [&]() { selection_status = select_cuda_backend_from_env(&selected_backend); });

  if (!selection_status.ok()) {
    return selection_status;
  }
  return selected_backend;
}

template <typename T>
struct IsStatusOr : std::false_type {};

template <typename T>
struct IsStatusOr<absl::StatusOr<T>> : std::true_type {};

template <typename Method, typename... Args>
auto Dispatch(Method method, Args&&... args)
    -> decltype((std::declval<CudaBackend>().*method)(std::forward<Args>(args)...)) {
  using ReturnType = decltype((std::declval<CudaBackend>().*method)(std::forward<Args>(args)...));

  absl::StatusOr<CudaBackend*> backend_or = get_backend();
  if constexpr (std::is_same_v<ReturnType, absl::Status> || IsStatusOr<ReturnType>::value) {
    if (!backend_or.ok()) {
      return backend_or.status();
    }
  } else {
    ABSL_CHECK_OK(backend_or.status());
  }

  CudaBackend* backend = backend_or.value();
  if constexpr (std::is_same_v<ReturnType, void>) {
    (backend->*method)(std::forward<Args>(args)...);
  } else {
    return (backend->*method)(std::forward<Args>(args)...);
  }
}

} // namespace

#define TENSORCAST_DISPATCH_BACKEND(return_type, name, args, ...)   \
  return_type name args {                                           \
    return Dispatch(&CudaBackend::name __VA_OPT__(, ) __VA_ARGS__); \
  }

TENSORCAST_CUDA_BACKEND_FUNCTIONS(TENSORCAST_DISPATCH_BACKEND)

#undef TENSORCAST_DISPATCH_BACKEND

} // namespace tensorcast::cuda
