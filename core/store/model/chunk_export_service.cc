// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/chunk_export_service.h"

#include <algorithm>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/communicator/engine/engine.h"
#include "core/store/model/transfer_constants.h"

namespace stepcast::store {

std::vector<std::pair<uint32_t, uint32_t>> ChunkExportService::coalesce_ranges(std::vector<uint32_t> chunks) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  if (chunks.empty())
    return out;
  // Remove duplicates and sort in one pass
  std::sort(chunks.begin(), chunks.end());
  chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
  uint32_t start = chunks.front();
  uint32_t prev = start;
  for (size_t i = 1; i < chunks.size(); ++i) {
    if (chunks[i] == prev + 1) {
      prev = chunks[i];
      continue;
    }
    out.emplace_back(start, prev);
    start = prev = chunks[i];
  }
  out.emplace_back(start, prev);
  return out;
}

absl::StatusOr<CommRegistrationInfo> ChunkExportService::export_chunks(
    const InstanceKey& key,
    ModelLocation location,
    absl::Span<const uint32_t> chunks,
    communicator::CommunicateEngine& comm_engine) {
  // Validate parameters
  if (chunks.empty()) {
    return absl::InvalidArgumentError("No chunks specified for export");
  }
  if (location != ModelLocation::PAGEABLE_CPU && location != ModelLocation::GPU) {
    return absl::InvalidArgumentError("Invalid location for export");
  }

  CommRegistrationInfo info;
  info.model_size = uma_ && uma_->get_model_size(key).ok() ? *uma_->get_model_size(key) : 0;
  info.location = location;

  ExportRecord rec;

  if (location == ModelLocation::PAGEABLE_CPU) {
    void* base = uma_ ? uma_->get_cpu_base_ptr(key) : nullptr;
    if (!base)
      return absl::FailedPreconditionError("CPU base not available");

    info.device_id = kCpuDeviceId;
    info.comm_dev_type = communicator::COMMUNICATE_ENGINE_DEV_CPU;

    // Move semantics to avoid copy
    std::vector<uint32_t> chunk_vec;
    chunk_vec.reserve(chunks.size());
    chunk_vec.assign(chunks.begin(), chunks.end());
    auto ranges = coalesce_ranges(std::move(chunk_vec));
    constexpr uint64_t kChunk = memory::DistributedVirtualMemoryPool::kDefaultChunkSize;
    size_t range_idx = 0;
    rec.cpu_tokens.reserve(ranges.size());
    for (const auto& [start, end] : ranges) {
      uint64_t va_off = static_cast<uint64_t>(start) * kChunk;
      uint64_t va_end = std::min<uint64_t>(info.model_size, (static_cast<uint64_t>(end) + 1) * kChunk);
      uint64_t length = (va_end > va_off) ? (va_end - va_off) : 0;
      if (length == 0)
        continue;

      // Acquire pin lease via UMA direct write token (encapsulates DVMP pin)
      std::array<VaRange, 1> one{{VaRange{va_off, length}}};
      auto token_or = uma_->create_direct_write_token(key, absl::MakeSpan(one));
      if (!token_or.ok())
        return token_or.status();
      rec.cpu_tokens.push_back(std::move(*token_or));

      // Bounds check before pointer arithmetic
      if (va_off >= info.model_size) {
        return absl::OutOfRangeError(absl::StrFormat("Offset %llu exceeds model size %llu", va_off, info.model_size));
      }
      const uint64_t addr = reinterpret_cast<uint64_t>(static_cast<char*>(base) + va_off);
      auto tensor_key = absl::StrFormat("%s_CPU_chunk_%zu", key.model_id, range_idx++);
      auto ret = comm_engine.register_tensor(tensor_key, addr, length, info.comm_dev_type, info.device_id);
      if (!ret.ok()) {
        return absl::InternalError("Failed to register CPU chunk-range tensor");
      }
      info.buffer_addresses.push_back(addr);
      info.buffer_sizes.push_back(static_cast<size_t>(length));
      info.remote_memory_keys.push_back(std::move(tensor_key));
    }

    // Cache record for precise unexport
    {
      std::lock_guard<std::mutex> lock(records_mu_);
      ExportKey rkey{.key = key, .location = location};
      rec.info = info;
      records_[rkey] = std::move(rec);
    }

    return info;
  }

  if (location == ModelLocation::GPU) {
    void* gpu_ptr = uma_ ? uma_->get_gpu_base_ptr(key, key.device.ordinal) : nullptr;
    if (!gpu_ptr)
      return absl::FailedPreconditionError("GPU base not available");

    info.device_id = key.device.ordinal;
    info.comm_dev_type = communicator::COMMUNICATE_ENGINE_DEV_GPU;

    // Move semantics to avoid copy
    std::vector<uint32_t> chunk_vec;
    chunk_vec.reserve(chunks.size());
    chunk_vec.assign(chunks.begin(), chunks.end());
    auto ranges = coalesce_ranges(std::move(chunk_vec));
    constexpr uint64_t kChunk = memory::DistributedVirtualMemoryPool::kDefaultChunkSize;
    size_t range_idx = 0;
    for (const auto& [start, end] : ranges) {
      uint64_t off = static_cast<uint64_t>(start) * kChunk;
      uint64_t va_end = std::min<uint64_t>(info.model_size, (static_cast<uint64_t>(end) + 1) * kChunk);
      uint64_t length = (va_end > off) ? (va_end - off) : 0;
      if (length == 0)
        continue;
      // Bounds check before pointer arithmetic
      if (off >= info.model_size) {
        return absl::OutOfRangeError(absl::StrFormat("Offset %llu exceeds model size %llu", off, info.model_size));
      }
      const uint64_t addr = reinterpret_cast<uint64_t>(static_cast<char*>(gpu_ptr) + off);
      auto tensor_key = absl::StrFormat("%s_GPU_chunk_%zu", key.model_id, range_idx++);
      auto ret = comm_engine.register_tensor(tensor_key, addr, length, info.comm_dev_type, info.device_id);
      if (!ret.ok()) {
        return absl::InternalError("Failed to register GPU chunk-range tensor");
      }
      info.buffer_addresses.push_back(addr);
      info.buffer_sizes.push_back(static_cast<size_t>(length));
      info.remote_memory_keys.push_back(std::move(tensor_key));
    }

    // Cache record for precise unexport
    {
      std::lock_guard<std::mutex> lock(records_mu_);
      ExportKey rkey{.key = key, .location = location};
      rec.info = info;
      records_[rkey] = std::move(rec);
    }

    return info;
  }

  return absl::InvalidArgumentError("Invalid location for export");
}

absl::Status ChunkExportService::unexport_chunks(
    const InstanceKey& key,
    const CommRegistrationInfo& info,
    communicator::CommunicateEngine& comm_engine) {
  // Validate parameters
  if (info.remote_memory_keys.empty()) {
    return absl::OkStatus(); // Nothing to unexport
  }

  // Use keys from provided info to unregister precisely
  absl::Status first_error;
  for (const auto& tensor_key : info.remote_memory_keys) {
    absl::Status st = comm_engine.unregister_tensor(tensor_key);
    if (!st.ok() && first_error.ok()) {
      first_error = st;
      // Continue to try unregistering remaining tensors for best-effort cleanup
    }
  }

  // Erase record and drop leases (by dropping tokens)
  {
    std::lock_guard<std::mutex> lock(records_mu_);
    ExportKey rkey{.key = key, .location = info.location};
    auto it = records_.find(rkey);
    if (it != records_.end()) {
      // Overwrite stored info to ensure leases are dropped after this function returns
      records_.erase(it);
    }
  }

  return first_error.ok() ? absl::OkStatus() : first_error;
}

} // namespace stepcast::store