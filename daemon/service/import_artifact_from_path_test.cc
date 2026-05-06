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

#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/store_engine.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
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
using tensorcast::daemon::v2::PromoteMountedSourceArtifactRequest;
using tensorcast::daemon::v2::PromoteMountedSourceArtifactResponse;
using tensorcast::daemon::v2::ResolvePublicDiskSourceRequest;
using tensorcast::daemon::v2::ResolvePublicDiskSourceResponse;
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

nlohmann::json read_json_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  REQUIRE(in.is_open());
  nlohmann::json payload;
  in >> payload;
  return payload;
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
  explicit HarnessFixture(
      const std::filesystem::path& storage_root,
      std::shared_ptr<tensorcast::store::components::IGlobalStoreClient> global_store_client = nullptr) {
    std::filesystem::create_directories(storage_root);
    engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));

    tensorcast::daemon::DaemonOptions daemon_opts;
    daemon_opts.storage_path = storage_root;

    auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(
        engine, daemon_opts, /*async_runtime=*/nullptr, std::move(global_store_client));
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

TEST_CASE(
    "ImportArtifactFromPath backfills descriptor metadata for safetensors sources",
    "[daemon][disk][import][descriptor]") {
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
  REQUIRE(std::filesystem::exists(artifact_dir / "artifact_descriptor.json"));
  REQUIRE(std::filesystem::exists(artifact_dir / "tensor_index.json"));
  REQUIRE_FALSE(std::filesystem::exists(artifact_dir / "tensor_index.cbor"));

  const auto descriptor = read_json_file(artifact_dir / "artifact_descriptor.json");
  REQUIRE(descriptor["artifact_id"].get<std::string>() == resp.artifact_id());
  REQUIRE(descriptor["schema_version"].get<std::string>() == "v3");
  REQUIRE_FALSE(descriptor["data_multihash"].get<std::string>().empty());
}

TEST_CASE(
    "Imported safetensors remain readable after descriptor backfill",
    "[daemon][disk][import][descriptor][materialize]") {
  const auto storage_root = make_clean_dir("import_descriptor_materialize_storage");
  const auto artifact_dir = storage_root / "artifact_safetensors_materialize";
  std::filesystem::create_directories(artifact_dir);
  create_safetensors_file(artifact_dir / "part0.safetensors", "weights", /*size_bytes=*/32);

  HarnessFixture fix(storage_root);

  ImportArtifactFromPathRequest import_req;
  import_req.set_path(artifact_dir.string());
  import_req.set_verify_checksums(false);
  grpc::ServerContext import_ctx;
  ImportArtifactFromPathResponse import_resp;
  const auto import_status = fix.service().ImportArtifactFromPath(&import_ctx, &import_req, &import_resp);

  REQUIRE(import_status.ok());
  REQUIRE(std::filesystem::exists(artifact_dir / "artifact_descriptor.json"));
  REQUIRE(std::filesystem::exists(artifact_dir / "tensor_index.json"));

  MaterializeReplicaRequest materialize_req;
  materialize_req.mutable_selection()->set_artifact_id(import_resp.artifact_id());
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.mutable_source_policy()->set_preference(
      tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  grpc::ServerContext materialize_ctx;
  MaterializeReplicaResponse materialize_resp;
  const auto materialize_status =
      fix.service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);

  REQUIRE(materialize_status.ok());
  REQUIRE(materialize_resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
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
    "ImportArtifactFromPath publishes artifact metadata to global store on fresh and cached imports",
    "[daemon][disk][import][metadata]") {
  const auto storage_root = make_clean_dir("import_metadata_publish_storage");
  const auto artifact_dir = storage_root / "artifact_metadata_publish";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  HarnessFixture fix(storage_root, gs_client);

  ImportArtifactFromPathRequest first_req;
  first_req.set_path(artifact_dir.string());
  first_req.set_verify_checksums(true);
  grpc::ServerContext first_ctx;
  ImportArtifactFromPathResponse first_resp;
  const auto first_status = fix.service().ImportArtifactFromPath(&first_ctx, &first_req, &first_resp);

  REQUIRE(first_status.ok());
  REQUIRE(gs_client->upserted_artifact_metadata_descriptors.size() == 1);
  REQUIRE(gs_client->upserted_artifact_metadata_indices.size() == 1);
  REQUIRE(gs_client->upserted_artifact_metadata_descriptors.front().artifact_id() == first_resp.artifact_id());
  REQUIRE(gs_client->upserted_artifact_metadata_descriptors.front().index_multihash().size() > 0);
  REQUIRE(gs_client->upserted_artifact_metadata_descriptors.front().data_multihash().size() > 0);
  REQUIRE(gs_client->upserted_artifact_metadata_indices.front() == first_resp.canonical_index_bytes());
  REQUIRE(gs_client->disk_locations.empty());

  gs_client->upserted_artifact_metadata_descriptors.clear();
  gs_client->upserted_artifact_metadata_indices.clear();

  ImportArtifactFromPathRequest second_req;
  second_req.set_path(artifact_dir.string());
  second_req.set_verify_checksums(true);
  grpc::ServerContext second_ctx;
  ImportArtifactFromPathResponse second_resp;
  const auto second_status = fix.service().ImportArtifactFromPath(&second_ctx, &second_req, &second_resp);

  REQUIRE(second_status.ok());
  REQUIRE(second_resp.artifact_id() == first_resp.artifact_id());
  REQUIRE(gs_client->upserted_artifact_metadata_descriptors.size() == 1);
  REQUIRE(gs_client->upserted_artifact_metadata_indices.size() == 1);
  REQUIRE(gs_client->upserted_artifact_metadata_descriptors.front().artifact_id() == second_resp.artifact_id());
  REQUIRE(gs_client->upserted_artifact_metadata_indices.front() == second_resp.canonical_index_bytes());
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
    "ImportArtifactFromPathStream reuses artifact_descriptor and skips hash phase on later imports",
    "[daemon][disk][import][stream][descriptor]") {
  const auto storage_root = make_clean_dir("import_stream_descriptor_storage");
  const auto artifact_dir = storage_root / "artifact_stream_descriptor";
  std::filesystem::create_directories(artifact_dir);
  create_safetensors_file(artifact_dir / "part0.safetensors", "weights", /*size_bytes=*/32);

  {
    HarnessFixture fix(storage_root);

    ImportArtifactFromPathRequest req;
    req.set_path(artifact_dir.string());
    req.set_verify_checksums(false);
    grpc::ServerContext ctx;
    ImportArtifactFromPathResponse resp;
    const auto status = fix.service().ImportArtifactFromPath(&ctx, &req, &resp);

    REQUIRE(status.ok());
    REQUIRE(std::filesystem::exists(artifact_dir / "artifact_descriptor.json"));
  }

  GrpcStreamFixture fix(storage_root);

  ImportArtifactFromPathRequest req;
  req.set_path(artifact_dir.string());
  req.set_verify_checksums(false);

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

  bool saw_hash_phase = false;
  for (const auto& item : events) {
    if (item.phase() == tensorcast::daemon::v2::IMPORT_ARTIFACT_PHASE_HASH_DATA) {
      saw_hash_phase = true;
      break;
    }
  }
  REQUIRE_FALSE(saw_hash_phase);
  REQUIRE(events.back().done());
  REQUIRE(events.back().has_result());
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
    "Mounted source metadata classifies no-index partitioned directories as byte-only",
    "[daemon][disk][resolve][metadata][byte_only]") {
  const auto storage_root = make_clean_dir("mounted_source_metadata_byte_only_storage");
  const auto artifact_dir = storage_root / "artifact_byte_only_metadata";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  auto metadata_or = tensorcast::daemon::materialization_disk_resolve::build_mounted_source_metadata(artifact_dir);

  REQUIRE(metadata_or.ok());
  REQUIRE(
      metadata_or->format_kind ==
      tensorcast::daemon::materialization_disk_resolve::MountedSourceFormatKind::kPartitioned);
  REQUIRE(
      metadata_or->metadata_capability ==
      tensorcast::daemon::materialization_disk_resolve::MountedSourceMetadataCapability::kByteOnly);
  REQUIRE(metadata_or->exact_size_bytes == 64);
  REQUIRE_FALSE(metadata_or->index_info.source_index_json.has_value());
  REQUIRE(
      metadata_or->index_info.canonical_index_json ==
      tensorcast::store::loading::build_synthetic_payload_canonical_index_json(64));
  REQUIRE(metadata_or->file_fingerprints.contains("tensor.data"));
}

TEST_CASE(
    "ResolvePublicDiskSource returns msa1 and same-daemon lookups consume the attested source",
    "[daemon][disk][resolve][msa1]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_storage");
  const auto artifact_dir = storage_root / "artifact_resolve";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  resolve_req.set_verify_checksums(true);
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  const auto resolve_status = fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp);

  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.has_source());
  const auto& source = resolve_resp.source();
  REQUIRE(source.artifact_id().starts_with("msa1:"));
  REQUIRE(source.trusted_content_artifact_id().starts_with("mi2:"));
  REQUIRE(source.format_kind() == tensorcast::daemon::v2::DISK_SOURCE_FORMAT_KIND_PARTITIONED);
  REQUIRE(source.metadata_capability() == tensorcast::daemon::v2::DISK_METADATA_CAPABILITY_TENSOR_AWARE);
  REQUIRE(source.validation_mode() == tensorcast::daemon::v2::DISK_VALIDATION_MODE_VALIDATE_BEFORE_READ);
  REQUIRE_FALSE(source.policy_id().empty());
  REQUIRE(source.exact_size_bytes() == 64);

  const auto binding = fix.harness->kernel().source_registry().lookup_binding(source.artifact_id());
  REQUIRE(binding.has_value());
  REQUIRE(binding->source_kind == tensorcast::daemon::ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact);
  REQUIRE(binding->trusted_content_artifact_id == source.trusted_content_artifact_id());

  grpc::ServerContext index_ctx;
  tensorcast::daemon::v2::GetArtifactIndexByIdRequest index_req;
  index_req.set_artifact_id(source.artifact_id());
  tensorcast::daemon::v2::GetArtifactIndexByIdResponse index_resp;
  const auto index_status = fix.service().GetArtifactIndexById(&index_ctx, &index_req, &index_resp);

  REQUIRE(index_status.ok());
  REQUIRE(index_resp.tensor_index_data() == source.canonical_index_bytes());

  MaterializeReplicaRequest materialize_req;
  materialize_req.mutable_selection()->set_artifact_id(source.artifact_id());
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.mutable_source_policy()->set_preference(
      tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  materialize_req.mutable_source_policy()->set_allow_p2p(false);
  materialize_req.mutable_source_policy()->set_allow_disk(true);
  grpc::ServerContext materialize_ctx;
  MaterializeReplicaResponse materialize_resp;
  const auto materialize_status =
      fix.service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);

  INFO("status=" << materialize_status.error_code() << " message=" << materialize_status.error_message());
  REQUIRE(materialize_status.ok());
  REQUIRE(materialize_resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
}

TEST_CASE(
    "ResolvePublicDiskSource rejects absolute paths outside configured trusted roots",
    "[daemon][disk][resolve][policy]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_policy_storage");
  const auto outside_root = make_clean_dir("resolve_public_disk_source_policy_outside");
  const auto artifact_dir = outside_root / "artifact_outside_root";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  const auto resolve_status = fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp);

  REQUIRE_FALSE(resolve_status.ok());
  REQUIRE(resolve_status.error_code() == grpc::StatusCode::PERMISSION_DENIED);
  REQUIRE(resolve_status.error_message().find("outside configured trusted roots") != std::string::npos);
}

TEST_CASE(
    "ResolvePublicDiskSource invalidates msa1 after mounted snapshot mutation",
    "[daemon][disk][resolve][msa1][mutation]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_mutation_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_mutation";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  REQUIRE(fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp).ok());

  REQUIRE(std::filesystem::remove(artifact_dir / "tensor.data"));

  grpc::ServerContext index_ctx;
  tensorcast::daemon::v2::GetArtifactIndexByIdRequest index_req;
  index_req.set_artifact_id(resolve_resp.source().artifact_id());
  tensorcast::daemon::v2::GetArtifactIndexByIdResponse index_resp;
  const auto index_status = fix.service().GetArtifactIndexById(&index_ctx, &index_req, &index_resp);

  REQUIRE_FALSE(index_status.ok());
  REQUIRE(index_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE_FALSE(
      fix.harness->kernel().source_registry().lookup_binding(resolve_resp.source().artifact_id()).has_value());
}

TEST_CASE(
    "ResolvePublicDiskSource attests descriptorless safetensors to msa1 without a trusted mi2 hint",
    "[daemon][disk][resolve][msa1][safetensors]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_safetensors_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_safetensors";
  std::filesystem::create_directories(artifact_dir);
  create_safetensors_file(artifact_dir / "weights.safetensors", "weights", /*size_bytes=*/32);

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  const auto resolve_status = fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp);

  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.has_source());
  REQUIRE(resolve_resp.source().artifact_id().starts_with("msa1:"));
  REQUIRE(resolve_resp.source().trusted_content_artifact_id().empty());
  REQUIRE(resolve_resp.source().format_kind() == tensorcast::daemon::v2::DISK_SOURCE_FORMAT_KIND_SAFETENSORS);
  REQUIRE(
      resolve_resp.source().resolution_strategy() == tensorcast::daemon::v2::DISK_RESOLUTION_STRATEGY_ATTESTED_ONLY);
}

TEST_CASE(
    "ResolvePublicDiskSource returns byte-only msa1 for no-index partitioned sources and allows full local loads",
    "[daemon][disk][resolve][msa1][byte_only]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_byte_only_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_byte_only";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  const auto resolve_status = fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp);

  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.has_source());
  const auto& source = resolve_resp.source();
  REQUIRE(source.artifact_id().starts_with("msa1:"));
  REQUIRE(source.trusted_content_artifact_id().empty());
  REQUIRE(source.format_kind() == tensorcast::daemon::v2::DISK_SOURCE_FORMAT_KIND_PARTITIONED);
  REQUIRE(source.metadata_capability() == tensorcast::daemon::v2::DISK_METADATA_CAPABILITY_BYTE_ONLY);
  REQUIRE(source.exact_size_bytes() == 64);
  REQUIRE(
      source.canonical_index_bytes() == tensorcast::store::loading::build_synthetic_payload_canonical_index_json(64));

  const auto binding = fix.harness->kernel().source_registry().lookup_binding(source.artifact_id());
  REQUIRE(binding.has_value());
  REQUIRE_FALSE(binding->tensor_aware_metadata);

  grpc::ServerContext index_ctx;
  tensorcast::daemon::v2::GetArtifactIndexByIdRequest index_req;
  index_req.set_artifact_id(source.artifact_id());
  tensorcast::daemon::v2::GetArtifactIndexByIdResponse index_resp;
  const auto index_status = fix.service().GetArtifactIndexById(&index_ctx, &index_req, &index_resp);

  REQUIRE(index_status.ok());
  REQUIRE(index_resp.tensor_index_data() == source.canonical_index_bytes());

  MaterializeReplicaRequest materialize_req;
  materialize_req.mutable_selection()->set_artifact_id(source.artifact_id());
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.mutable_source_policy()->set_preference(
      tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  grpc::ServerContext materialize_ctx;
  MaterializeReplicaResponse materialize_resp;
  const auto materialize_status =
      fix.service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);

  REQUIRE(materialize_status.ok());
  REQUIRE(materialize_resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
}

TEST_CASE(
    "ImportArtifactFromPath records mi2 promotion for an existing msa1 mounted source",
    "[daemon][disk][resolve][msa1][promotion]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_import_promotion_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_import_promotion";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  REQUIRE(fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp).ok());
  REQUIRE(resolve_resp.source().artifact_id().starts_with("msa1:"));
  REQUIRE(resolve_resp.source().trusted_content_artifact_id().empty());

  ImportArtifactFromPathRequest import_req;
  import_req.set_path(artifact_dir.string());
  import_req.set_verify_checksums(false);
  grpc::ServerContext import_ctx;
  ImportArtifactFromPathResponse import_resp;
  const auto import_status = fix.service().ImportArtifactFromPath(&import_ctx, &import_req, &import_resp);

  REQUIRE(import_status.ok());
  REQUIRE(import_resp.artifact_id().starts_with("mi2:"));

  const auto mounted_entry =
      fix.harness->kernel().source_registry().lookup_binding(resolve_resp.source().artifact_id());
  REQUIRE(mounted_entry.has_value());
  REQUIRE(mounted_entry->promoted_content_artifact_id == import_resp.artifact_id());
}

TEST_CASE(
    "PromoteMountedSourceArtifact explicitly promotes msa1 to mi2 without a path request",
    "[daemon][disk][resolve][msa1][verify_promotion]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_explicit_promotion_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_explicit_promotion";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  REQUIRE(fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp).ok());
  REQUIRE(resolve_resp.source().artifact_id().starts_with("msa1:"));

  PromoteMountedSourceArtifactRequest promote_req;
  promote_req.set_artifact_id(resolve_resp.source().artifact_id());
  promote_req.set_verify_checksums(false);
  grpc::ServerContext promote_ctx;
  PromoteMountedSourceArtifactResponse promote_resp;
  const auto promote_status = fix.service().PromoteMountedSourceArtifact(&promote_ctx, &promote_req, &promote_resp);

  REQUIRE(promote_status.ok());
  REQUIRE(promote_resp.artifact_id().starts_with("mi2:"));
  REQUIRE(promote_resp.source_artifact_id() == resolve_resp.source().artifact_id());
  REQUIRE(promote_resp.import_state() == tensorcast::daemon::v2::IMPORT_ARTIFACT_STATE_READY);

  const auto mounted_entry =
      fix.harness->kernel().source_registry().lookup_binding(resolve_resp.source().artifact_id());
  REQUIRE(mounted_entry.has_value());
  REQUIRE(mounted_entry->promoted_content_artifact_id == promote_resp.artifact_id());
  REQUIRE(
      mounted_entry->promoted_content_origin ==
      tensorcast::daemon::ArtifactSourceRegistry::PromotionOrigin::kMountedVerify);
}

TEST_CASE(
    "PromoteMountedSourceArtifact rejects non-msa1 artifact ids",
    "[daemon][disk][resolve][msa1][verify_promotion][validation]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_explicit_promotion_validation_storage");
  HarnessFixture fix(storage_root);

  PromoteMountedSourceArtifactRequest promote_req;
  promote_req.set_artifact_id("mi2:test:test");
  grpc::ServerContext promote_ctx;
  PromoteMountedSourceArtifactResponse promote_resp;
  const auto promote_status = fix.service().PromoteMountedSourceArtifact(&promote_ctx, &promote_req, &promote_resp);

  REQUIRE_FALSE(promote_status.ok());
  REQUIRE(promote_status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(promote_status.error_message().find("msa1") != std::string::npos);
}

TEST_CASE(
    "PromoteMountedSourceArtifact fails closed after mounted source mutation",
    "[daemon][disk][resolve][msa1][verify_promotion][mutation]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_explicit_promotion_mutation_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_explicit_promotion_mutation";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  REQUIRE(fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp).ok());

  REQUIRE(std::filesystem::remove(artifact_dir / "tensor.data"));

  PromoteMountedSourceArtifactRequest promote_req;
  promote_req.set_artifact_id(resolve_resp.source().artifact_id());
  promote_req.set_verify_checksums(false);
  grpc::ServerContext promote_ctx;
  PromoteMountedSourceArtifactResponse promote_resp;
  const auto promote_status = fix.service().PromoteMountedSourceArtifact(&promote_ctx, &promote_req, &promote_resp);

  REQUIRE_FALSE(promote_status.ok());
  REQUIRE(promote_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(promote_status.error_message().find("SOURCE_MUTATED") != std::string::npos);
}

TEST_CASE(
    "MaterializeReplica rejects tensor-aware subset selection from byte-only msa1 sources",
    "[daemon][disk][resolve][msa1][byte_only][selection]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_byte_only_subset_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_byte_only_subset";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));

  HarnessFixture fix(storage_root);

  ResolvePublicDiskSourceRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  ResolvePublicDiskSourceResponse resolve_resp;
  REQUIRE(fix.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp).ok());

  MaterializeReplicaRequest materialize_req;
  materialize_req.mutable_selection()->set_artifact_id(resolve_resp.source().artifact_id());
  materialize_req.mutable_selection()->add_tensor_names("payload");
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.mutable_source_policy()->set_preference(
      tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  materialize_req.mutable_source_policy()->set_allow_p2p(false);
  materialize_req.mutable_source_policy()->set_allow_disk(true);
  grpc::ServerContext materialize_ctx;
  MaterializeReplicaResponse materialize_resp;
  const auto materialize_status =
      fix.service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);

  INFO("status=" << materialize_status.error_code() << " message=" << materialize_status.error_message());
  REQUIRE_FALSE(materialize_status.ok());
  REQUIRE(materialize_status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(materialize_status.error_message().find("tensor-aware mounted-source metadata") != std::string::npos);
}

TEST_CASE("ResolvePublicDiskSource msa1 is rejected after daemon restart", "[daemon][disk][resolve][msa1][restart]") {
  const auto storage_root = make_clean_dir("resolve_public_disk_source_restart_storage");
  const auto artifact_dir = storage_root / "artifact_resolve_restart";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  std::string stale_msa1;
  {
    HarnessFixture first(storage_root);
    ResolvePublicDiskSourceRequest resolve_req;
    resolve_req.set_path(artifact_dir.string());
    grpc::ServerContext resolve_ctx;
    ResolvePublicDiskSourceResponse resolve_resp;
    REQUIRE(first.service().ResolvePublicDiskSource(&resolve_ctx, &resolve_req, &resolve_resp).ok());
    stale_msa1 = resolve_resp.source().artifact_id();
    REQUIRE(stale_msa1.starts_with("msa1:"));
  }

  HarnessFixture second(storage_root);
  grpc::ServerContext index_ctx;
  tensorcast::daemon::v2::GetArtifactIndexByIdRequest index_req;
  index_req.set_artifact_id(stale_msa1);
  tensorcast::daemon::v2::GetArtifactIndexByIdResponse index_resp;
  const auto index_status = second.service().GetArtifactIndexById(&index_ctx, &index_req, &index_resp);

  REQUIRE_FALSE(index_status.ok());
  REQUIRE(index_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
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
  materialize_req.mutable_source_policy()->set_preference(
      tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  grpc::ServerContext materialize_ctx;
  MaterializeReplicaResponse materialize_resp;
  const auto materialize_status =
      fix.service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);

  REQUIRE_FALSE(materialize_status.ok());
  REQUIRE(materialize_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(materialize_status.error_message().find("SOURCE_MUTATED") != std::string::npos);
}

} // namespace
