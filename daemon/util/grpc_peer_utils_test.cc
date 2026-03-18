// Copyright (c) 2026, TensorCast Team.

#include "daemon/util/grpc_peer_utils.h"

#include <catch2/catch_test_macros.hpp>

namespace tensorcast::daemon {

TEST_CASE("is_loopback_grpc_peer recognizes loopback", "[daemon][grpc_peer]") {
  REQUIRE(is_loopback_grpc_peer(""));
  REQUIRE(is_loopback_grpc_peer("unknown"));
  REQUIRE(is_loopback_grpc_peer("unix:/tmp/tensorcast.sock"));
  REQUIRE(is_loopback_grpc_peer("inproc:abc"));
  REQUIRE(is_loopback_grpc_peer("ipv4:127.0.0.1:12345"));
  REQUIRE(is_loopback_grpc_peer("ipv4:127.1.2.3:9999"));
  REQUIRE(is_loopback_grpc_peer("ipv6:[::1]:5555"));
}

TEST_CASE("is_loopback_grpc_peer rejects non-loopback", "[daemon][grpc_peer]") {
  REQUIRE_FALSE(is_loopback_grpc_peer("ipv4:10.0.0.1:12345"));
  REQUIRE_FALSE(is_loopback_grpc_peer("ipv6:[fe80::1]:12345"));
  REQUIRE_FALSE(is_loopback_grpc_peer("dns:localhost:12345"));
}

TEST_CASE("grpc_peer_endpoint normalizes peer endpoint strings", "[daemon][grpc_peer]") {
  REQUIRE(grpc_peer_endpoint("ipv4:127.0.0.1:50051") == std::optional<std::string>("127.0.0.1:50051"));
  REQUIRE(grpc_peer_endpoint("ipv6:[::1]:50051") == std::optional<std::string>("[::1]:50051"));
  REQUIRE_FALSE(grpc_peer_endpoint("unix:/tmp/tensorcast.sock").has_value());
}

TEST_CASE("grpc_peer_matches_address matches normalized peer addresses", "[daemon][grpc_peer]") {
  REQUIRE(grpc_peer_matches_address("ipv4:127.0.0.1:50051", "127.0.0.1:50051"));
  REQUIRE_FALSE(grpc_peer_matches_address("ipv4:127.0.0.1:50051", "127.0.0.1:50052"));
  REQUIRE_FALSE(grpc_peer_matches_address("unix:/tmp/tensorcast.sock", "127.0.0.1:50051"));
}

} // namespace tensorcast::daemon
