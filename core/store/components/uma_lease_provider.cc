// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/uma_lease_provider.h"

#include "absl/log/log.h"

namespace tensorcast::store {

std::shared_ptr<UmaLeaseProvider> UmaLeaseProvider::instance() {
  static auto inst = std::shared_ptr<UmaLeaseProvider>(new UmaLeaseProvider());
  return inst;
}

void UmaLeaseProvider::register_mapping(
    const std::string& tensor_key,
    const ReplicaKey& key,
    uint64_t base_va_off,
    gsl::not_null<std::shared_ptr<ReplicaMemoryCoordinator>> uma) {
  absl::MutexLock lk(&mu_);
  map_[tensor_key] =
      Entry{.key = key, .base_va_off = base_va_off, .uma = std::weak_ptr<ReplicaMemoryCoordinator>(uma.get())};
}

std::unique_ptr<communicator::DRAMStager::LeaseHandle> UmaLeaseProvider::acquire(
    const std::string& tensor_key,
    uint64_t offset,
    uint64_t bytes) {
  Entry entry;
  {
    absl::MutexLock lk(&mu_);
    auto it = map_.find(tensor_key);
    if (it == map_.end()) {
      // Unknown key – return no-op handle (no UMA pin)
      return std::make_unique<communicator::DRAMStager::LeaseHandle>();
    }
    entry = it->second;
  }

  auto uma = entry.uma.lock();
  if (!uma) {
    return std::make_unique<communicator::DRAMStager::LeaseHandle>();
  }

  const uint64_t va_off = entry.base_va_off + offset;
  VaRange range{.offset = va_off, .length = bytes};
  auto token_or = uma->create_direct_write_token(entry.key, absl::Span<const VaRange>(&range, 1));
  if (!token_or.ok()) {
    LOG(WARNING) << "UMA lease acquire failed for key=" << tensor_key << ": " << token_or.status();
    return std::make_unique<communicator::DRAMStager::LeaseHandle>();
  }
  auto token = std::move(*token_or);
  // Keepalive via LeaseHandle lifetime
  return std::make_unique<TokenLeaseHandle>(std::move(token.keepalive));
}

bool UmaLeaseProvider::is_range_hot(const std::string& tensor_key, uint64_t offset, uint64_t bytes) const {
  if (bytes == 0)
    return false;
  Entry entry;
  {
    absl::MutexLock lk(&mu_);
    auto it = map_.find(tensor_key);
    if (it == map_.end()) {
      return false;
    }
    entry = it->second;
  }

  auto uma = entry.uma.lock();
  if (!uma)
    return false;

  const uint64_t va_off = entry.base_va_off + offset;
  const uint64_t end_off = va_off + bytes;
  const size_t chunk_sz = uma->get_chunk_size();
  if (chunk_sz == 0)
    return false;
  const auto start_idx = static_cast<uint32_t>(va_off / chunk_sz);
  const auto end_idx = static_cast<uint32_t>((end_off + chunk_sz - 1) / chunk_sz);

  auto mappings = uma->get_chunk_mappings(entry.key);
  if (mappings.empty())
    return false;

  // Build a quick index from chunk_idx to state, assuming contiguous indices
  for (uint32_t i = start_idx; i < end_idx; ++i) {
    // Bounds check
    bool found = false;
    for (const auto& m : mappings) {
      if (m.chunk_idx == i) {
        found = true;
        if (m.cpu_state != ChunkState::HOT) {
          return false;
        }
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

} // namespace tensorcast::store
