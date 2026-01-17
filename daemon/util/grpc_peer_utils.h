// Copyright (c) 2026, TensorCast Team.

#pragma once

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

} // namespace tensorcast::daemon
