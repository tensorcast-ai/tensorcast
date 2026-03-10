// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/store/store_engine.h"
#include "core/testing/common.h"
#include "daemon/state/artifact_source_registry.h"
#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

using tensorcast::daemon::v2::ImportArtifactFromPathRequest;
using tensorcast::daemon::v2::ImportArtifactFromPathResponse;
using tensorcast::daemon::v2::ImportArtifactFromPathStreamEvent;
using tensorcast::daemon::v2::MaterializeReplicaRequest;
using tensorcast::daemon::v2::MaterializeReplicaResponse;
using tensorcast::daemon::v2::StoreDaemonService;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_import_artifact_from_path_test";
}

std::filesystem::path make_clean_dir(std::string_view name) {
  const auto path = test_tmpdir() / std::string(name);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::filesystem::path normalize_for_compare(const std::filesystem::path& path) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(path, ec);
  if (!ec) {
    return normalized;
  }
  return path.lexically_normal();
}

void create_safetensors_file(
    const std::filesystem::path& path,
    const std::string& tensor_name,
    std::uint64_t size_bytes) {
  const std::string header_json =
      nlohmann::json({{tensor_name, {{"dtype", "U8"}, {"shape", {size_bytes}}, {"data_offsets", {0, size_bytes}}}}})
          .dump();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  const std::uint64_t header_size = header_json.size();
  for (int i = 0; i < 8; ++i) {
    const auto byte = static_cast<unsigned char>((header_size >> (8 * i)) & 0xFF);
    out.put(static_cast<char>(byte));
  }
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  std::vector<char> payload(static_cast<size_t>(size_bytes), '\0');
  if (!payload.empty()) {
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
}

tensorcast::store::StoreEngineOptions make_engine_opts(const std::filesystem::path& storage_root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.p2p_port = 0;
  opts.memory_pool_size = 64ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, std::string value) : name_(name) {
    if (const char* prev = std::getenv(name_); prev != nullptr) {
      had_prev_ = true;
      prev_ = prev;
    }
#if defined(_WIN32)
    _putenv_s(name_, value.c_str());
#else
    ::setenv(name_, value.c_str(), /*overwrite=*/1);
#endif
  }

  ~ScopedEnvVar() {
#if defined(_WIN32)
    if (had_prev_) {
      _putenv_s(name_, prev_.c_str());
    } else {
      _putenv_s(name_, "");
    }
#else
    if (had_prev_) {
      ::setenv(name_, prev_.c_str(), /*overwrite=*/1);
    } else {
      ::unsetenv(name_);
    }
#endif
  }

 private:
  const char* name_;
  bool had_prev_{false};
  std::string prev_;
};

struct HarnessFixture {
  explicit HarnessFixture(const std::filesystem::path& storage_root) {
    std::filesystem::create_directories(storage_root);
    engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));

    tensorcast::daemon::DaemonOptions daemon_opts;
    daemon_opts.storage_path = storage_root;

    auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
    REQUIRE(harness_or.ok());
    harness = std::move(*harness_or);
    REQUIRE(harness->start().ok());
  }

  tensorcast::daemon::StoreDaemonServiceImpl& service() const {
    return harness->service();
  }

  std::shared_ptr<tensorcast::store::StoreEngine> engine;
  std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> harness;
};

struct GrpcStreamFixture {
  explicit GrpcStreamFixture(const std::filesystem::path& storage_root) : harness(storage_root) {
    grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&harness.service());
    server = builder.BuildAndStart();
    REQUIRE(server != nullptr);
    REQUIRE(selected_port != 0);

    address = "127.0.0.1:" + std::to_string(selected_port);
    auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    stub = StoreDaemonService::NewStub(channel);
  }

  ~GrpcStreamFixture() {
    if (server != nullptr) {
      server->Shutdown();
    }
  }

  HarnessFixture harness;
  std::unique_ptr<grpc::Server> server;
  std::string address;
  std::unique_ptr<StoreDaemonService::Stub> stub;
};

TEST_CASE("ImportArtifactFromPath returns ready response and records local source binding", "[daemon][disk][import]") {
  const auto storage_root = make_clean_dir("import_unary_storage");
  const auto artifact_dir = storage_root / "artifact";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  HarnessFixture fix(storage_root);

  ImportArtifactFromPathRequest req;
  req.set_path(artifact_dir.string());
  req.set_verify_checksums(true);
  grpc::ServerContext ctx;
  ImportArtifactFromPathResponse resp;
  const auto status = fix.service().ImportArtifactFromPath(&ctx, &req, &resp);

  REQUIRE(status.ok());
  REQUIRE(resp.import_state() == tensorcast::daemon::v2::IMPORT_ARTIFACT_STATE_READY);
  REQUIRE(resp.artifact_id().starts_with("mi2:"));
  REQUIRE_FALSE(resp.canonical_index_bytes().empty());
  REQUIRE(resp.generation() != 0);

  const auto binding = fix.harness->kernel().source_registry().lookup_binding(resp.artifact_id());
  REQUIRE(binding.has_value());
  REQUIRE(binding->source_kind == tensorcast::daemon::ArtifactSourceRegistry::SourceKind::kLocalImport);
  REQUIRE(binding->canonical_source_path == normalize_for_compare(artifact_dir).string());
  REQUIRE_FALSE(binding->file_fingerprints.empty());
}

TEST_CASE("ImportArtifactFromPath does not mutate source directories", "[daemon][disk][import][readonly]") {
  const auto storage_root = make_clean_dir("import_readonly_storage");
  const auto artifact_dir = storage_root / "artifact_safetensors";
  std::filesystem::create_directories(artifact_dir);
  create_safetensors_file(artifact_dir / "part0.safetensors", "weights", /*size_bytes=*/32);

  HarnessFixture fix(storage_root);

  ImportArtifactFromPathRequest req;
  req.set_path(artifact_dir.string());
  req.set_verify_checksums(false);
  grpc::ServerContext ctx;
  ImportArtifactFromPathResponse resp;
  const auto status = fix.service().ImportArtifactFromPath(&ctx, &req, &resp);

  REQUIRE(status.ok());
  REQUIRE(resp.import_state() == tensorcast::daemon::v2::IMPORT_ARTIFACT_STATE_READY);
  REQUIRE_FALSE(resp.artifact_id().empty());
  REQUIRE_FALSE(std::filesystem::exists(artifact_dir / "artifact_descriptor.json"));
  REQUIRE_FALSE(std::filesystem::exists(artifact_dir / "tensor_index.json"));
  REQUIRE_FALSE(std::filesystem::exists(artifact_dir / "tensor_index.cbor"));
}

TEST_CASE(
    "ImportArtifactFromPath cache uses import env knobs and serves repeated requests",
    "[daemon][disk][import][cache]") {
  ScopedEnvVar cache_ttl("TENSORCAST_IMPORT_ARTIFACT_CACHE_TTL_SECONDS", "600");
  ScopedEnvVar cache_max_entries("TENSORCAST_IMPORT_ARTIFACT_CACHE_MAX_ENTRIES", "32");

  const auto storage_root = make_clean_dir("import_cache_storage");
  const auto artifact_dir = storage_root / "artifact_cache";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  HarnessFixture fix(storage_root);

  ImportArtifactFromPathRequest first_req;
  first_req.set_path(artifact_dir.string());
  first_req.set_verify_checksums(false);
  grpc::ServerContext first_ctx;
  ImportArtifactFromPathResponse first_resp;
  const auto first_status = fix.service().ImportArtifactFromPath(&first_ctx, &first_req, &first_resp);
  REQUIRE(first_status.ok());

  REQUIRE(std::filesystem::remove(artifact_dir / "tensor.data"));
  REQUIRE(std::filesystem::remove(artifact_dir / "artifact_descriptor.json"));

  ImportArtifactFromPathRequest second_req;
  second_req.set_path(artifact_dir.string());
  second_req.set_verify_checksums(false);
  grpc::ServerContext second_ctx;
  ImportArtifactFromPathResponse second_resp;
  const auto second_status = fix.service().ImportArtifactFromPath(&second_ctx, &second_req, &second_resp);

  REQUIRE(second_status.ok());
  REQUIRE(second_resp.artifact_id() == first_resp.artifact_id());
  REQUIRE(second_resp.canonical_index_bytes() == first_resp.canonical_index_bytes());
}

TEST_CASE(
    "ImportArtifactFromPathStream emits monotonic progress and one terminal success",
    "[daemon][disk][import][stream]") {
  const auto storage_root = make_clean_dir("import_stream_storage");
  const auto artifact_dir = storage_root / "artifact_stream";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  GrpcStreamFixture fix(storage_root);

  ImportArtifactFromPathRequest req;
  req.set_path(artifact_dir.string());
  req.set_verify_checksums(true);

  grpc::ClientContext client_ctx;
  auto reader = fix.stub->ImportArtifactFromPathStream(&client_ctx, req);
  REQUIRE(reader != nullptr);

  std::vector<ImportArtifactFromPathStreamEvent> events;
  ImportArtifactFromPathStreamEvent event;
  while (reader->Read(&event)) {
    events.push_back(event);
  }
  const auto status = reader->Finish();

  REQUIRE(status.ok());
  REQUIRE_FALSE(events.empty());

  std::uint64_t last_seq = 0;
  std::size_t terminal_count = 0;
  for (const auto& item : events) {
    REQUIRE(item.seq() > last_seq);
    last_seq = item.seq();
    if (item.done()) {
      terminal_count += 1;
    }
  }
  REQUIRE(terminal_count == 1);

  const auto& terminal = events.back();
  REQUIRE(terminal.done());
  REQUIRE_FALSE(terminal.error());
  REQUIRE(terminal.has_result());
  REQUIRE(terminal.result().import_state() == tensorcast::daemon::v2::IMPORT_ARTIFACT_STATE_READY);
}

TEST_CASE(
    "ImportArtifactFromPathStream returns machine-readable SOURCE_NOT_FOUND",
    "[daemon][disk][import][stream][error]") {
  const auto storage_root = make_clean_dir("import_stream_error_storage");
  GrpcStreamFixture fix(storage_root);

  ImportArtifactFromPathRequest req;
  req.set_path((storage_root / "missing_artifact").string());
  req.set_verify_checksums(true);

  grpc::ClientContext client_ctx;
  auto reader = fix.stub->ImportArtifactFromPathStream(&client_ctx, req);
  REQUIRE(reader != nullptr);

  std::vector<ImportArtifactFromPathStreamEvent> events;
  ImportArtifactFromPathStreamEvent event;
  while (reader->Read(&event)) {
    events.push_back(event);
  }
  const auto status = reader->Finish();

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::NOT_FOUND);
  REQUIRE_FALSE(events.empty());

  const auto& terminal = events.back();
  REQUIRE(terminal.done());
  REQUIRE(terminal.error());
  REQUIRE(terminal.error_code() == tensorcast::daemon::v2::IMPORT_ARTIFACT_ERROR_CODE_SOURCE_NOT_FOUND);
}

TEST_CASE(
    "MaterializeReplica fails with SOURCE_MUTATED after local import source changes",
    "[daemon][disk][import][mutation]") {
  const auto storage_root = make_clean_dir("import_mutation_storage");
  const auto artifact_dir = storage_root / "artifact_mutation";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  HarnessFixture fix(storage_root);

  ImportArtifactFromPathRequest import_req;
  import_req.set_path(artifact_dir.string());
  import_req.set_verify_checksums(true);
  grpc::ServerContext import_ctx;
  ImportArtifactFromPathResponse import_resp;
  const auto import_status = fix.service().ImportArtifactFromPath(&import_ctx, &import_req, &import_resp);
  REQUIRE(import_status.ok());
  REQUIRE(import_resp.artifact_id().starts_with("mi2:"));

  REQUIRE(std::filesystem::remove(artifact_dir / "tensor.data"));

  MaterializeReplicaRequest materialize_req;
  materialize_req.mutable_selection()->set_artifact_id(import_resp.artifact_id());
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  grpc::ServerContext materialize_ctx;
  MaterializeReplicaResponse materialize_resp;
  const auto materialize_status =
      fix.service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);

  REQUIRE_FALSE(materialize_status.ok());
  REQUIRE(materialize_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(materialize_status.error_message().find("SOURCE_MUTATED") != std::string::npos);
}

} // namespace
