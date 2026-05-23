// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/serving_artifact_manifest_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"

namespace serving_manifest = tensorcast::daemon::serving_artifact_manifest;

namespace {

std::string hash_serialized_payload(std::string_view version, const std::string& serialized) {
  std::string payload;
  payload.reserve(version.size() + 1 + serialized.size());
  payload.append(version.data(), version.size());
  payload.push_back('\n');
  payload.append(serialized);
  const std::vector<std::uint8_t> digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()));
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::string tensor_schema_hash_for_weight() {
  nlohmann::json tensors_json = nlohmann::json::array();
  tensors_json.push_back(
      nlohmann::json{
          {"name", "weight"},
          {"dtype", "torch.float32"},
          {"shape", std::vector<int64_t>{2}},
          {"stride", std::vector<int64_t>{1}},
          {"element_size", 4},
      });
  const std::string serialized = nlohmann::json{{"tensors", std::move(tensors_json)}}.dump(-1, ' ', true);
  return hash_serialized_payload("tensorcast.representation.tensor_schema.v1", serialized);
}

std::string serving_manifest_payload() {
  return nlohmann::json{
      {"schema_version", 1},
      {"artifact_kind", "serving"},
      {"framework_name", "vllm"},
      {"adapter_version", "v1"},
      {"serving_abi_version", "v1"},
      {"representation_contract_hash", "bciqrepresentation"},
      {"serving_build_digest", "bciqservingbuild"},
      {"serving_build_digest_version", std::string(serving_manifest::kPhase1ServingBuildDigestVersion)},
      {"tensor_schema_hash", tensor_schema_hash_for_weight()},
      {"canonical_tensor_count", 1},
      {"builder_mode", "binding_finalize"},
      {"build_pipeline_version", "tensorcast-bootstrap-v1"},
      {"serving_manifest_ref", std::string(serving_manifest::kPhase1ServingManifestRef)},
      {"topology_admission_digest", "topology-digest"},
  }
      .dump();
}

std::string shared_storage_serving_index() {
  return nlohmann::json{
      {std::string(serving_manifest::kPhase1ServingManifestTensorName),
       nlohmann::json::array(
           {0, 4096, nlohmann::json::array({1024}), nlohmann::json::array({1}), "torch.uint8", 3000})},
      {"weight",
       nlohmann::json::array({0, 4096, nlohmann::json::array({2}), nlohmann::json::array({1}), "torch.float32", 0})},
  }
      .dump();
}

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path();
}

std::filesystem::path make_clean_dir(std::string_view name) {
  const auto path = test_tmpdir() / std::string(name);
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

void write_u64_le(std::ofstream& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    const auto byte = static_cast<char>((value >> (8 * i)) & 0xFF);
    out.put(byte);
  }
}

void write_serving_safetensors_file(const std::filesystem::path& path, const std::string& manifest_payload) {
  constexpr std::uint64_t kWeightBytes = 8;
  const auto manifest_bytes = static_cast<std::uint64_t>(manifest_payload.size());
  nlohmann::json header = nlohmann::json::object();
  header["weight"] = nlohmann::json{
      {"dtype", "F32"},
      {"shape", nlohmann::json::array({2})},
      {"data_offsets", nlohmann::json::array({0, kWeightBytes})},
  };
  header[std::string(serving_manifest::kPhase1ServingManifestTensorName)] = nlohmann::json{
      {"dtype", "U8"},
      {"shape", nlohmann::json::array({manifest_bytes})},
      {"data_offsets", nlohmann::json::array({kWeightBytes, kWeightBytes + manifest_bytes})},
  };

  const std::string header_json = header.dump();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  write_u64_le(out, static_cast<std::uint64_t>(header_json.size()));
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  const std::vector<char> weight_payload(static_cast<size_t>(kWeightBytes), '\0');
  out.write(weight_payload.data(), static_cast<std::streamsize>(weight_payload.size()));
  out.write(manifest_payload.data(), static_cast<std::streamsize>(manifest_payload.size()));
}

std::uint64_t total_size_from_index(std::string_view index_json) {
  const auto parsed = nlohmann::json::parse(index_json);
  std::uint64_t total_size = 0;
  for (const auto& value : parsed.items()) {
    const auto& entry = value.value();
    total_size = std::max(total_size, entry.at(0).get<std::uint64_t>() + entry.at(1).get<std::uint64_t>());
  }
  return total_size;
}

tensorcast::store::StoreEngine make_store_engine(const std::filesystem::path& storage_root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 64ULL << 10;
  opts.num_thread = 2;
  opts.p2p_port = 0;
  opts.global_store_address.clear();
  return tensorcast::store::StoreEngine(opts);
}

} // namespace

TEST_CASE("serving manifest preflight accepts tensor payload inside a larger shared storage", "[serving_manifest]") {
  serving_manifest::ServingManifestPayloadPreflightRequest request;
  request.canonical_index_json = shared_storage_serving_index();
  request.manifest_payload = serving_manifest_payload();
  request.serving_manifest_ref = std::string(serving_manifest::kPhase1ServingManifestRef);
  request.require_manifest = true;

  auto result_or = serving_manifest::preflight_serving_manifest_payload(request);
  REQUIRE(result_or.ok());
  CHECK(result_or->serving_manifest_present);
  CHECK(result_or->canonical_tensor_count == 1);
  CHECK(result_or->serving_manifest_ref == serving_manifest::kPhase1ServingManifestRef);
  CHECK(result_or->manifest.builder_mode == "binding_finalize");
  CHECK(result_or->manifest.topology_admission_digest.value() == "topology-digest");
}

TEST_CASE("serving manifest preflight rejects topology digest mismatch", "[serving_manifest]") {
  serving_manifest::ServingManifestPayloadPreflightRequest request;
  request.canonical_index_json = shared_storage_serving_index();
  request.manifest_payload = serving_manifest_payload();
  request.serving_manifest_ref = std::string(serving_manifest::kPhase1ServingManifestRef);
  request.expected_topology_admission_digest = "other-topology";
  request.require_manifest = true;

  auto result_or = serving_manifest::preflight_serving_manifest_payload(request);
  REQUIRE_FALSE(result_or.ok());
  CHECK(result_or.status().message().find("topology_admission_digest") != std::string::npos);
}

TEST_CASE(
    "serving manifest preflight reads safetensors disk sources through the disk loader",
    "[serving_manifest][safetensors]") {
  const auto artifact_dir = make_clean_dir("serving_manifest_safetensors_disk_source");
  const auto engine_dir = make_clean_dir("serving_manifest_safetensors_engine");
  const std::string manifest_payload = serving_manifest_payload();
  const auto safetensors_path = artifact_dir / "weights.safetensors";
  write_serving_safetensors_file(safetensors_path, manifest_payload);

  auto source_index_or = tensorcast::store::loader::BuildSourceIndexFromSafetensors({safetensors_path});
  REQUIRE(source_index_or.ok());
  auto canonical_index_or = tensorcast::store::loader::BuildCanonicalIndexFromSafetensors({safetensors_path});
  REQUIRE(canonical_index_or.ok());

  tensorcast::store::loading::DiskMetadata disk_metadata;
  disk_metadata.canonical_index_json = *canonical_index_or;
  disk_metadata.source_index_json = *source_index_or;
  disk_metadata.logical_total_size = total_size_from_index(*canonical_index_or);
  disk_metadata.source_total_size_bytes = total_size_from_index(*source_index_or);
  disk_metadata.is_safetensors = true;
  disk_metadata.tensor_aware = true;

  auto engine = make_store_engine(engine_dir);
  auto request = serving_manifest::build_preflight_request(
      "msa1:test-session~policy~safetensors~serving-manifest",
      *canonical_index_or,
      tensorcast::store::loading::DiskSource{
          .path = artifact_dir,
          .expected_size = std::nullopt,
          .require_descriptor = false,
      },
      disk_metadata,
      nullptr);
  request.serving_manifest_ref = std::string(serving_manifest::kPhase1ServingManifestRef);
  request.require_manifest = true;

  auto result_or = serving_manifest::preflight_serving_artifact(&engine, request);
  INFO("status=" << result_or.status());
  REQUIRE(result_or.ok());
  CHECK(result_or->serving_manifest_present);
  CHECK(result_or->manifest.builder_mode == "binding_finalize");
  CHECK(result_or->canonical_tensor_count == 1);
}
