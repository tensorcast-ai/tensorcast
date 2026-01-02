// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <initializer_list>

#include "absl/status/status.h"
#include "core/common/cuda_api.h"
#include "core/common/cuda_driver_api.h"

namespace {

struct EnvOverride {
  const char* name;
  const char* value;
  bool set;
};

int RunChild(std::initializer_list<EnvOverride> overrides, int (*child_fn)()) {
  const pid_t pid = fork();
  if (pid == 0) {
    for (const auto& env : overrides) {
      if (env.set) {
        setenv(env.name, env.value, 1);
      } else {
        unsetenv(env.name);
      }
    }
    const int code = child_fn();
    _exit(code);
  }
  if (pid < 0) {
    return -1;
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return -1;
}

int ExpectRealBackend() {
  return tensorcast::cuda::is_fake() ? 1 : 0;
}

int ExpectFakeBackend() {
  return tensorcast::cuda::is_fake() ? 0 : 1;
}

int ExpectNoDriverLoadInFake() {
  if (!tensorcast::cuda::is_fake()) {
    return 1;
  }
  return tensorcast::cuda::DriverApi::load_attempted_for_testing() ? 1 : 0;
}

int ExpectFakeRejected() {
  const absl::Status status = tensorcast::cuda::set_device(0);
  return status.code() == absl::StatusCode::kFailedPrecondition ? 0 : 1;
}

int ExpectInvalidEnvRejected() {
  const absl::Status status = tensorcast::cuda::set_device(0);
  return status.code() == absl::StatusCode::kInvalidArgument ? 0 : 1;
}

} // namespace

TEST_CASE("CUDA backend selection honors env and test gating", "[cuda]") {
  // Backend selection is process-global, so run each scenario in a child process.
  int exit_code = RunChild({{"TENSORCAST_CUDA_BACKEND", nullptr, false}}, &ExpectRealBackend);
  REQUIRE(exit_code == 0);

  exit_code = RunChild(
      {{"TENSORCAST_CUDA_BACKEND", "fake", true},
       {"TEST_SRCDIR", "/tmp", true},
       {"TEST_TMPDIR", "/tmp", true},
       {"PYTEST_CURRENT_TEST", "cuda_backend_selection_test", true}},
      &ExpectFakeBackend);
  REQUIRE(exit_code == 0);

  exit_code = RunChild(
      {{"TENSORCAST_CUDA_BACKEND", "fake", true},
       {"TEST_SRCDIR", "/tmp", true},
       {"TEST_TMPDIR", "/tmp", true},
       {"PYTEST_CURRENT_TEST", "cuda_backend_selection_test", true}},
      &ExpectNoDriverLoadInFake);
  REQUIRE(exit_code == 0);

  exit_code = RunChild(
      {{"TENSORCAST_CUDA_BACKEND", "fake", true},
       {"TEST_SRCDIR", nullptr, false},
       {"TEST_TMPDIR", nullptr, false},
       {"PYTEST_CURRENT_TEST", nullptr, false}},
      &ExpectFakeRejected);
  REQUIRE(exit_code == 0);

  exit_code = RunChild(
      {{"TENSORCAST_CUDA_BACKEND", "bogus", true},
       {"TEST_SRCDIR", nullptr, false},
       {"TEST_TMPDIR", nullptr, false},
       {"PYTEST_CURRENT_TEST", nullptr, false}},
      &ExpectInvalidEnvRejected);
  REQUIRE(exit_code == 0);
}
