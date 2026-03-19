// Copyright (c) 2026, TensorCast Team.

#include "daemon/util/grpc_peer_utils.h"

#include <optional>
#include <string>
#include <string_view>

namespace tensorcast::daemon {
namespace {

bool has_prefix(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::string_view strip_prefix(std::string_view s, std::string_view prefix) {
  if (!has_prefix(s, prefix)) {
    return s;
  }
  return s.substr(prefix.size());
}

} // namespace

bool is_loopback_grpc_peer(std::string_view peer) {
  if (peer.empty() || peer == "unknown") {
    // In-process/unit tests may not have a real transport peer; treat as local so
    // local-only code paths remain testable without a gRPC server.
    return true;
  }
  if (has_prefix(peer, "unix:") || has_prefix(peer, "inproc:")) {
    return true;
  }

  if (has_prefix(peer, "ipv4:")) {
    const std::string_view rest = strip_prefix(peer, "ipv4:");
    const size_t colon = rest.rfind(':');
    const std::string_view host = (colon == std::string_view::npos) ? rest : rest.substr(0, colon);
    return has_prefix(host, "127.");
  }

  if (has_prefix(peer, "ipv6:")) {
    const std::string_view rest = strip_prefix(peer, "ipv6:");
    const size_t colon = rest.rfind(':');
    std::string_view host = (colon == std::string_view::npos) ? rest : rest.substr(0, colon);
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
      host = host.substr(1, host.size() - 2);
    }
    return host == "::1";
  }

  return false;
}

std::optional<std::string> grpc_peer_endpoint(std::string_view peer) {
  if (has_prefix(peer, "ipv4:")) {
    return std::string(strip_prefix(peer, "ipv4:"));
  }
  if (has_prefix(peer, "ipv6:")) {
    std::string_view rest = strip_prefix(peer, "ipv6:");
    if (rest.size() >= 2 && rest.front() == '[') {
      return std::string(rest);
    }
  }
  return std::nullopt;
}

bool grpc_peer_matches_address(std::string_view peer, std::string_view address) {
  const auto endpoint = grpc_peer_endpoint(peer);
  return endpoint.has_value() && *endpoint == address;
}

} // namespace tensorcast::daemon
