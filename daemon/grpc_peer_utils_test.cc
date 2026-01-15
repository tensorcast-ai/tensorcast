// Copyright (c) 2026, TensorCast Team.

#include "daemon/grpc_peer_utils.h"

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

} // namespace tensorcast::daemon
