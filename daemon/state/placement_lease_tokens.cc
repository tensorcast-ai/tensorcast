// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/placement_lease_tokens.h"

#include "absl/status/status.h"
#include "absl/strings/escaping.h"

namespace tensorcast::daemon {

PlacementLeaseTokens::PlacementLeaseTokens(Options opts) : opts_(std::move(opts)) {}

std::string PlacementLeaseTokens::mint_token_locked() {
  std::string bytes;
  bytes.resize(opts_.token_bytes);
  for (size_t i = 0; i < opts_.token_bytes; ++i) {
    bytes[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen_, 0u, 256u));
  }
  return absl::BytesToHexString(bytes);
}

absl::StatusOr<std::string> PlacementLeaseTokens::mint(
    SessionLifecycleManager::LeaseId lease_id,
    absl::Duration ttl_hint) {
  if (lease_id == 0) {
    return absl::InvalidArgumentError("lease_id is zero");
  }
  if (opts_.token_bytes == 0) {
    return absl::FailedPreconditionError("token_bytes is zero");
  }

  const absl::Time now = absl::Now();
  const absl::Duration ttl = ttl_hint > absl::ZeroDuration() ? ttl_hint : opts_.default_ttl;
  const absl::Time expires_at = now + ttl;

  prune();
  absl::MutexLock lock(&mu_);
  if (tokens_.size() >= opts_.capacity) {
    return absl::ResourceExhaustedError("placement lease token capacity exceeded");
  }

  for (int attempt = 0; attempt < 4; ++attempt) {
    std::string token = mint_token_locked();
    if (tokens_.contains(token)) {
      continue;
    }
    tokens_.emplace(token, Record{.lease_id = lease_id, .expires_at = expires_at});
    return token;
  }

  return absl::InternalError("failed to mint unique placement lease token");
}

absl::StatusOr<SessionLifecycleManager::LeaseId> PlacementLeaseTokens::resolve(const std::string& token) const {
  if (token.empty()) {
    return absl::InvalidArgumentError("lease_token is empty");
  }

  const absl::Time now = absl::Now();
  absl::MutexLock lock(&mu_);
  auto it = tokens_.find(token);
  if (it == tokens_.end()) {
    return absl::NotFoundError("lease_token not found");
  }
  if (it->second.expires_at <= now) {
    return absl::PermissionDeniedError("lease_token expired");
  }
  return it->second.lease_id;
}

absl::Status PlacementLeaseTokens::refresh(const std::string& token, absl::Duration ttl_hint) {
  if (token.empty()) {
    return absl::InvalidArgumentError("lease_token is empty");
  }
  const absl::Time now = absl::Now();
  const absl::Duration ttl = ttl_hint > absl::ZeroDuration() ? ttl_hint : opts_.default_ttl;
  const absl::Time expires_at = now + ttl;

  absl::MutexLock lock(&mu_);
  auto it = tokens_.find(token);
  if (it == tokens_.end()) {
    return absl::NotFoundError("lease_token not found");
  }
  if (it->second.expires_at <= now) {
    return absl::PermissionDeniedError("lease_token expired");
  }
  it->second.expires_at = expires_at;
  return absl::OkStatus();
}

bool PlacementLeaseTokens::erase(const std::string& token) {
  absl::MutexLock lock(&mu_);
  return tokens_.erase(token) > 0;
}

void PlacementLeaseTokens::prune() {
  const absl::Time now = absl::Now();
  absl::MutexLock lock(&mu_);
  for (auto it = tokens_.begin(); it != tokens_.end();) {
    if (it->second.expires_at <= now) {
      auto to_erase = it++;
      tokens_.erase(to_erase);
    } else {
      ++it;
    }
  }
}

size_t PlacementLeaseTokens::size() const {
  absl::MutexLock lock(&mu_);
  return tokens_.size();
}

} // namespace tensorcast::daemon
