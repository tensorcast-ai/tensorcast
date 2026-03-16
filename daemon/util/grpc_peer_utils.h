// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace tensorcast::daemon {

// Returns true when the gRPC peer string represents a host-local caller.
// Intended for gating local-only capabilities such as handle leases.
//
// gRPC peer formats are typically:
//  - "ipv4:127.0.0.1:12345"
//  - "ipv6:[::1]:12345"
//  - "unix:/path/to/socket"
//
// For in-process/unit tests, the peer may be empty or "unknown"; treat those as local.
bool is_loopback_grpc_peer(std::string_view peer);

// Returns a normalized host:port endpoint when the peer encodes one; otherwise nullopt.
[[nodiscard]] std::optional<std::string> grpc_peer_endpoint(std::string_view peer);

// Returns true when the normalized gRPC peer endpoint matches the supplied host:port address.
[[nodiscard]] bool grpc_peer_matches_address(std::string_view peer, std::string_view address);

} // namespace tensorcast::daemon
