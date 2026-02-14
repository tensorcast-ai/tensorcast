// Copyright (c) 2026, TensorCast Team.

#include "daemon/app/daemon_app.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/store/store_engine.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_app_import_root_test";
}

} // namespace

namespace tensorcast::daemon {

TEST_CASE("DaemonApp create fails when import root cannot be initialized", "[daemon][app][import_root]") {
  const auto root = test_tmpdir() / "daemon_app_import_root_unavailable";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage";
  std::filesystem::create_directories(storage_root);

  const auto import_root_file = root / "import_root_file";
  {
    std::ofstream out(import_root_file, std::ios::binary | std::ios::trunc);
    REQUIRE(out.is_open());
    out << "not-a-directory";
  }

  store::StoreEngineOptions engine_opts;
  engine_opts.storage_path = storage_root.string();
  engine_opts.p2p_port = 0;
  engine_opts.memory_pool_size = 64ULL << 20;
  engine_opts.tx_slice_bytes = 1ULL << 20;
  engine_opts.num_thread = 2;
  auto engine = std::make_shared<store::StoreEngine>(engine_opts);

  DaemonApp::Options options;
  options.engine = engine;
  options.daemon_options.storage_path = storage_root;
  options.daemon_options.import_root = import_root_file;
  options.grpc.listen_addr = "127.0.0.1:0";

  auto app_or = DaemonApp::create(std::move(options));
  REQUIRE_FALSE(app_or.ok());
  REQUIRE(absl::IsFailedPrecondition(app_or.status()));
  REQUIRE(std::string(app_or.status().message()).find("IMPORT_ROOT_UNAVAILABLE") != std::string::npos);
}

} // namespace tensorcast::daemon
