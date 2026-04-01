// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_body_handle.h"

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"

namespace tensorcast::daemon {

namespace {

std::uint64_t next_body_handle_binding_generation() {
  static std::atomic<std::uint64_t> counter{1};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

std::string compute_sha256_hex_from_bytes(std::string_view payload) {
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()));
  std::string hex =
      absl::BytesToHexString(absl::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
  absl::AsciiStrToLower(&hex);
  return hex;
}

} // namespace

BodyHandle::BodyHandle(std::shared_ptr<CoreBacking> backing) : backing_(std::move(backing)) {}

absl::StatusOr<BodyHandle> BodyHandle::create(
    store::StoreEngine& engine,
    store::loading::ReplicaHandle replica_handle) {
  if (replica_handle.key().artifact_id.empty()) {
    return absl::InvalidArgumentError("BodyHandle requires a replica-backed artifact_id");
  }
  const auto size_or = engine.get_replica_size(replica_handle.key());
  if (!size_or.ok()) {
    return size_or.status();
  }
  auto backing = std::make_shared<CoreBacking>();
  backing->engine = &engine;
  backing->replica_handle = std::move(replica_handle);
  backing->size_bytes = *size_or;
  backing->binding_generation = next_body_handle_binding_generation();
  return BodyHandle(std::move(backing));
}

bool BodyHandle::empty() const {
  return !backing_ || backing_->engine == nullptr;
}

BodyHandle::Kind BodyHandle::kind() const {
  return Kind::kCoreReplica;
}

std::uint64_t BodyHandle::size_bytes() const {
  return backing_ ? backing_->size_bytes : 0;
}

std::uint64_t BodyHandle::binding_generation() const {
  return backing_ ? backing_->binding_generation : 0;
}

const store::loading::ReplicaHandle& BodyHandle::replica_handle() const {
  static const auto* empty_handle = new store::loading::ReplicaHandle();
  return backing_ ? backing_->replica_handle : *empty_handle;
}

common::memory::MemoryLocation BodyHandle::location() const {
  if (!backing_) {
    return common::memory::MemoryLocation::CPU;
  }
  return backing_->replica_handle.key().device.type == DeviceType::GPU ? common::memory::MemoryLocation::GPU
                                                                       : common::memory::MemoryLocation::CPU;
}

bool BodyHandle::unique_owner() const {
  return backing_ && backing_.use_count() == 1;
}

absl::StatusOr<std::unique_ptr<store::IArtifactLoader>> BodyHandle::make_loader() const {
  if (empty()) {
    return absl::FailedPreconditionError("BodyHandle is empty");
  }
  return backing_->engine->open_local_replica_loader(backing_->replica_handle.key(), location());
}

absl::Status BodyHandle::read_into_range(std::uint64_t offset, void* dst, std::size_t bytes) const {
  if (empty()) {
    return absl::FailedPreconditionError("BodyHandle is empty");
  }
  if (offset > backing_->size_bytes) {
    return absl::OutOfRangeError("BodyHandle offset exceeds payload size");
  }
  const std::size_t to_read = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, backing_->size_bytes - offset));
  if (to_read != bytes) {
    return absl::OutOfRangeError("BodyHandle read exceeds payload size");
  }
  if (to_read == 0) {
    return absl::OkStatus();
  }

  auto loader_or = make_loader();
  if (!loader_or.ok()) {
    return loader_or.status();
  }
  auto init_status = (*loader_or)->initialize();
  if (!init_status.ok()) {
    return init_status;
  }
  auto source_or = (*loader_or)->open_source();
  if (!source_or.ok()) {
    return source_or.status();
  }

  std::size_t copied = 0;
  while (copied < to_read) {
    auto read_or = (**source_or).read_at(offset + copied, static_cast<char*>(dst) + copied, to_read - copied);
    if (!read_or.ok()) {
      return read_or.status();
    }
    if (*read_or == 0) {
      return absl::DataLossError("BodyHandle source terminated before requested range");
    }
    copied += *read_or;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> BodyHandle::read_range(std::uint64_t offset, std::size_t bytes) const {
  if (empty()) {
    return absl::FailedPreconditionError("BodyHandle is empty");
  }
  if (offset > backing_->size_bytes) {
    return absl::OutOfRangeError("BodyHandle offset exceeds payload size");
  }
  const std::size_t to_read = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, backing_->size_bytes - offset));
  if (to_read == 0) {
    return std::string();
  }

  std::string payload;
  payload.resize(to_read);
  auto read_status = read_into_range(offset, payload.data(), payload.size());
  if (!read_status.ok()) {
    return read_status;
  }
  return payload;
}

absl::StatusOr<std::string> BodyHandle::read_all_bytes() const {
  return read_range(/*offset=*/0, static_cast<std::size_t>(size_bytes()));
}

absl::StatusOr<std::string> BodyHandle::compute_sha256_hex() const {
  auto payload_or = read_all_bytes();
  if (!payload_or.ok()) {
    return payload_or.status();
  }
  return compute_sha256_hex_from_bytes(*payload_or);
}

absl::Status BodyHandle::retire() const {
  if (empty()) {
    return absl::OkStatus();
  }
  if (backing_->retired.exchange(true)) {
    return absl::OkStatus();
  }
  return backing_->engine->retire_replica_status(backing_->replica_handle.key());
}

} // namespace tensorcast::daemon
