// Copyright (c) 2026, TensorCast Team.

#include "core/common/artifact_identity.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

namespace tensorcast::common {
namespace {

TEST_CASE("Byte artifact cgid parse vectors", "[artifact_identity][byte_artifact]") {
  const std::string artifact_id =
      "cgid:byte_artifact~tenantA~sglang~"
      "b64u.bWV0YS1sbGFtYS9MbGFtYS0zLjEtOEItSW5zdHJ1Y3Q~"
      "b64u.ZGVmYXVsdA~"
      "layout_v1~b64u.cmVxdWVzdC0wMDAxOmJsay00Mg";
  REQUIRE(is_byte_artifact_id(artifact_id));
  auto parsed_or = parse_byte_artifact_cgid(artifact_id);
  REQUIRE(parsed_or.ok());
  const auto& parsed = *parsed_or;
  REQUIRE(parsed.namespace_name == "tenantA");
  REQUIRE(parsed.engine == "sglang");
  REQUIRE(parsed.layout_id == "layout_v1");
  REQUIRE(parsed.model_id_enc == "b64u.bWV0YS1sbGFtYS9MbGFtYS0zLjEtOEItSW5zdHJ1Y3Q");
  REQUIRE(parsed.model_version_enc == "b64u.ZGVmYXVsdA");
  REQUIRE(parsed.engine_key_enc == "b64u.cmVxdWVzdC0wMDAxOmJsay00Mg");
}

TEST_CASE("Byte artifact cgid rejects wrong segment shape", "[artifact_identity][byte_artifact]") {
  auto parsed_or = parse_byte_artifact_cgid("cgid:byte_artifact~tenant~engine~layout_only");
  REQUIRE_FALSE(parsed_or.ok());
}

TEST_CASE("Byte artifact cgid accepts legacy segment shape", "[artifact_identity][byte_artifact]") {
  const std::string artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azQ";
  auto parsed_or = parse_byte_artifact_cgid(artifact_id);
  REQUIRE(parsed_or.ok());
  REQUIRE(parsed_or->namespace_name == "tenant");
  REQUIRE(parsed_or->engine == "engine");
  REQUIRE(parsed_or->model_id_enc == "b64u.bQ");
  REQUIRE(parsed_or->model_version_enc.empty());
  REQUIRE(parsed_or->layout_id == "layout_v1");
  REQUIRE(parsed_or->engine_key_enc == "b64u.azQ");
}

TEST_CASE("CGID segment b64u encode/decode roundtrip", "[artifact_identity][byte_artifact]") {
  const std::string raw = "request-0001:blk-42";
  const std::string encoded = encode_cgid_segment(raw);
  REQUIRE(encoded == "b64u.cmVxdWVzdC0wMDAxOmJsay00Mg");
  auto decoded_or = decode_cgid_segment(encoded);
  REQUIRE(decoded_or.ok());
  REQUIRE(*decoded_or == raw);
}

} // namespace
} // namespace tensorcast::common
