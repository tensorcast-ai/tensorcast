// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/replica/memory_export_registry.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/communicator/engine/engine.h"
#include "core/store/replica/transfer_constants.h"
// OpenTelemetry Metrics API (impl)
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::replica {

namespace {
inline const char* loc_str(common::memory::MemoryLocation loc) {
  return (loc == common::memory::MemoryLocation::GPU) ? "gpu" : "cpu";
}

std::string format_tensor_key(const loading::ReplicaKey& key, std::string_view suffix) {
  if (!key.view_id.has_value() || key.view_id->empty()) {
    return absl::StrCat(key.artifact_id, "_", suffix);
  }
  return absl::StrCat(key.artifact_id, "_view_", *key.view_id, "_", suffix);
}

absl::StatusOr<bool> has_stable_lease_for_chunks(
    UnifiedMemoryAuthority& uma,
    const loading::ReplicaKey& key,
    absl::Span<const uint32_t> chunk_indices) {
  if (chunk_indices.empty()) {
    return false;
  }
  const auto snapshots = uma.snapshot_cpu_chunks(key);
  if (snapshots.empty()) {
    return false;
  }
  for (const uint32_t chunk_index : chunk_indices) {
    if (chunk_index >= snapshots.size()) {
      return absl::OutOfRangeError(
          absl::StrFormat("Chunk index %u out of range for snapshot size %zu", chunk_index, snapshots.size()));
    }
    if (snapshots[chunk_index].stable_lease_count == 0) {
      return false;
    }
  }
  return true;
}
} // namespace

// OTel gauge callback
void MemoryExportRegistry::keepalive_gauge_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MemoryExportRegistry*>(state);
  if (self == nullptr)
    return;
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs)
    return;
  double count = 0.0;
  {
    std::lock_guard<std::mutex> lock(self->records_mu_);
    for (const auto& kv : self->records_) {
      if (kv.second.uma_keepalive)
        count += 1.0;
    }
  }
  self->keepalive_count_snapshot_ = count;
  obs->Observe(count, {{"location", opentelemetry::common::AttributeValue("cpu")}});
}

MemoryExportRegistry::MemoryExportRegistry(gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>> uma)
    : uma_(std::move(uma)) {
  meter_ = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  ex_reg_total_ = meter_->CreateDoubleCounter("tc_ex_registrations_total");
  ex_keepalive_gauge_ = meter_->CreateDoubleObservableGauge("tc_ex_keepalive_gauge");
  ex_keepalive_gauge_->AddCallback(&MemoryExportRegistry::keepalive_gauge_callback, this);
}

std::vector<std::pair<uint32_t, uint32_t>> MemoryExportRegistry::coalesce_ranges(std::vector<uint32_t> chunks) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  if (chunks.empty()) {
    return out;
  }
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

absl::StatusOr<ExportRegistration> MemoryExportRegistry::export_chunks(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location,
    absl::Span<const uint32_t> chunks,
    tensorcast::communicator::engine::Communicator& comm_engine) {
  const absl::Time total_started_at = absl::Now();
  // Validate parameters
  if (chunks.empty()) {
    return absl::InvalidArgumentError("No chunks specified for export");
  }
  if (location != common::memory::MemoryLocation::CPU && location != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("Invalid location for export");
  }

  ExportRegistration info;
  std::vector<uint32_t> normalized_indices(chunks.begin(), chunks.end());
  std::ranges::sort(normalized_indices);
  normalized_indices.erase(std::unique(normalized_indices.begin(), normalized_indices.end()), normalized_indices.end());
  auto rollback_export = [&]() {
    if (normalized_indices.empty()) {
      return;
    }
    (void)uma_->set_exported(key, location, absl::MakeSpan(normalized_indices), /*on=*/false);
  };
  auto cleanup_registered_tensors = [&](std::string_view stage) {
    if (info.remote_memory_keys.empty()) {
      return;
    }
    for (const auto& tensor_key : info.remote_memory_keys) {
      absl::Status st = comm_engine.unregister_tensor(tensor_key);
      if (!st.ok()) {
        LOG(WARNING) << "unregister_tensor failed for " << tensor_key << " during " << stage
                     << " rollback: " << st.message();
      }
    }
    info.remote_memory_keys.clear();
    info.buffer_addresses.clear();
    info.buffer_sizes.clear();
  };
  {
    auto sz_or = uma_->get_artifact_size(key);
    info.artifact_size = sz_or.ok() ? *sz_or : 0;
  }
  info.location = location;
  ExportRecord rec;
  std::optional<UnifiedMemoryAuthority::StableLease> stable_lease;
  absl::Duration set_exported_elapsed = absl::ZeroDuration();
  absl::Duration stable_lease_check_elapsed = absl::ZeroDuration();
  absl::Duration stable_lease_acquire_elapsed = absl::ZeroDuration();
  absl::Duration register_tensor_total_elapsed = absl::ZeroDuration();
  bool reused_stable_lease = false;
  std::size_t registered_range_count = 0;
  std::uint64_t registered_bytes = 0;

  if (location == common::memory::MemoryLocation::CPU) {
    void* base_raw = uma_->get_cpu_base_ptr(key);
    if (!base_raw) {
      return absl::FailedPreconditionError("CPU base not available");
    }
    gsl::not_null<void*> base{base_raw};
    info.device_id = kCpuDeviceId;
    info.comm_dev_type = communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
    // Always use UMA ledger/export registration in V3 final state
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    const absl::Time set_exported_started_at = absl::Now();
    auto reg_or = uma_->set_exported(key, location, chunks, /*on=*/true);
    set_exported_elapsed = absl::Now() - set_exported_started_at;
    if (!reg_or.ok()) {
      return reg_or.status();
    }
    ranges = reg_or->chunk_ranges;
    rec.uma_keepalive = reg_or->keepalive; // Hold VS pin leases across registration lifetime
    const absl::Time stable_lease_check_started_at = absl::Now();
    auto has_lease_or = has_stable_lease_for_chunks(*uma_, key, absl::MakeSpan(normalized_indices));
    stable_lease_check_elapsed = absl::Now() - stable_lease_check_started_at;
    if (!has_lease_or.ok()) {
      rollback_export();
      return has_lease_or.status();
    }
    if (!*has_lease_or) {
      const absl::Time stable_lease_acquire_started_at = absl::Now();
      auto lease_or = uma_->acquire_stable_lease(key, absl::MakeSpan(normalized_indices));
      stable_lease_acquire_elapsed = absl::Now() - stable_lease_acquire_started_at;
      if (!lease_or.ok()) {
        rollback_export();
        return lease_or.status();
      }
      stable_lease = std::move(*lease_or);
    } else {
      reused_stable_lease = true;
      VLOG(1) << "export_chunks: reusing existing stable lease for artifact_id=" << key.artifact_id;
    }
    // Derive chunk size from UMA layout to ensure alignment across VS/UMA
    uint64_t kChunk = static_cast<uint64_t>(uma_->get_artifact_chunk_bytes());
    if (auto layout_or = uma_->get_layout(key); layout_or.ok() && layout_or->artifact_chunk_bytes > 0) {
      kChunk = static_cast<uint64_t>(layout_or->artifact_chunk_bytes);
    }
    size_t range_idx = 0;
    for (const auto& [start, end] : ranges) {
      const size_t current_range_idx = range_idx;
      uint64_t va_off = static_cast<uint64_t>(start) * kChunk;
      uint64_t va_end = std::min<uint64_t>(info.artifact_size, (static_cast<uint64_t>(end) + 1) * kChunk);
      uint64_t length = (va_end > va_off) ? (va_end - va_off) : 0;
      if (length == 0) {
        continue;
      }

      // Bounds check before pointer arithmetic
      if (va_off >= info.artifact_size) {
        return absl::OutOfRangeError(
            absl::StrFormat("Offset %llu exceeds artifact size %llu", va_off, info.artifact_size));
      }
      const uint64_t addr = reinterpret_cast<uint64_t>(static_cast<char*>(base.get()) + va_off);
      auto tensor_key = format_tensor_key(key, absl::StrCat("CPU_chunk_", range_idx++));
      tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
      // RDMA producer direct-source serving requires the exported CPU chunk to
      // carry a registered MR. TCP still uses staged transport and leaves MR
      // registration disabled.
      opts.register_mr = comm_engine.is_rdma_enabled();
      opts.direct_rdma_enabled = comm_engine.is_rdma_enabled();
      opts.needs_staging = false;
      opts.async = false;

      const absl::Time register_tensor_started_at = absl::Now();
      auto ret = comm_engine.register_tensor_ex(tensor_key, addr, length, info.comm_dev_type, info.device_id, opts);
      const absl::Duration register_tensor_elapsed = absl::Now() - register_tensor_started_at;
      register_tensor_total_elapsed += register_tensor_elapsed;
      VLOG(2) << "memory_export_registry.export_chunks.range"
              << " artifact_id=" << key.artifact_id << " location=cpu"
              << " range_index=" << current_range_idx << " chunk_start=" << start << " chunk_end=" << end
              << " bytes=" << length << " register_mr=" << opts.register_mr
              << " direct_rdma_enabled=" << opts.direct_rdma_enabled
              << " register_tensor_ex_ms=" << absl::ToDoubleMilliseconds(register_tensor_elapsed)
              << " tensor_key=" << tensor_key << " status=" << (ret.ok() ? "OK" : ret.ToString());
      if (!ret.ok()) {
        cleanup_registered_tensors("CPU export");
        if (stable_lease.has_value()) {
          (void)uma_->release_stable_lease(*stable_lease);
          stable_lease.reset();
        }
        rollback_export();
        return absl::InternalError("Failed to register CPU chunk-range tensor");
      }
      info.buffer_addresses.push_back(addr);
      info.buffer_sizes.push_back(static_cast<size_t>(length));
      info.remote_memory_keys.push_back(std::move(tensor_key));
      ++registered_range_count;
      registered_bytes += length;
    }

    // Cache record for precise unexport
    {
      std::lock_guard<std::mutex> lock(records_mu_);
      ExportKey rkey{.key = key, .location = location};
      rec.info = info;
      rec.ranges = ranges;
      rec.stable_lease = std::move(stable_lease);
      records_[rkey] = std::move(rec);
    }

    // Metrics and logs
    if (ex_reg_total_) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      attrs.emplace("location", opentelemetry::common::AttributeValue(loc_str(location)));
      ex_reg_total_->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
    }
    LOG(INFO) << "export_chunks: artifact_id=" << key.artifact_id << " location=" << loc_str(location)
              << " ranges=" << ranges.size() << " status=OK";
    VLOG(2) << "memory_export_registry.export_chunks.summary"
            << " artifact_id=" << key.artifact_id << " location=cpu"
            << " normalized_chunk_count=" << normalized_indices.size() << " range_count=" << ranges.size()
            << " registered_range_count=" << registered_range_count << " registered_bytes=" << registered_bytes
            << " set_exported_ms=" << absl::ToDoubleMilliseconds(set_exported_elapsed)
            << " stable_lease_check_ms=" << absl::ToDoubleMilliseconds(stable_lease_check_elapsed)
            << " stable_lease_acquire_ms=" << absl::ToDoubleMilliseconds(stable_lease_acquire_elapsed)
            << " reused_stable_lease=" << reused_stable_lease
            << " register_tensor_total_ms=" << absl::ToDoubleMilliseconds(register_tensor_total_elapsed)
            << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - total_started_at)
            << " remote_key_count=" << info.remote_memory_keys.size();

    return info;
  }

  if (location == common::memory::MemoryLocation::GPU) {
    void* gpu_ptr_raw = uma_->get_gpu_base_ptr(key, key.device.ordinal);
    if (!gpu_ptr_raw) {
      return absl::FailedPreconditionError("GPU base not available");
    }
    gsl::not_null<void*> gpu_ptr{gpu_ptr_raw};

    info.device_id = key.device.ordinal;
    info.comm_dev_type = communicator::base::COMMUNICATE_ENGINE_DEV_GPU;

    // Always use UMA ledger/export registration in V3 final state
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    auto reg_or = uma_->set_exported(key, location, chunks, /*on=*/true);
    if (!reg_or.ok()) {
      return reg_or.status();
    }
    ranges = reg_or->chunk_ranges;
    uint64_t kChunk = static_cast<uint64_t>(uma_->get_artifact_chunk_bytes());
    if (auto layout_or = uma_->get_layout(key); layout_or.ok() && layout_or->artifact_chunk_bytes > 0) {
      kChunk = static_cast<uint64_t>(layout_or->artifact_chunk_bytes);
    }
    size_t range_idx = 0;
    for (const auto& [start, end] : ranges) {
      uint64_t off = static_cast<uint64_t>(start) * kChunk;
      uint64_t va_end = std::min<uint64_t>(info.artifact_size, (static_cast<uint64_t>(end) + 1) * kChunk);
      uint64_t length = (va_end > off) ? (va_end - off) : 0;
      if (length == 0) {
        continue;
      }
      // Bounds check before pointer arithmetic
      if (off >= info.artifact_size) {
        return absl::OutOfRangeError(
            absl::StrFormat("Offset %llu exceeds artifact size %llu", off, info.artifact_size));
      }
      const uint64_t addr = reinterpret_cast<uint64_t>(static_cast<char*>(gpu_ptr.get()) + off);
      auto tensor_key = format_tensor_key(key, absl::StrCat("GPU_chunk_", range_idx++));
      LOG(INFO) << "export_chunks(gpu): artifact_id=" << key.artifact_id
                << " view_id=" << (key.view_id.has_value() ? *key.view_id : "") << " tensor_key=" << tensor_key
                << " artifact_size=" << info.artifact_size << " off=" << off << " length=" << length;
      tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
      opts.register_mr = comm_engine.is_rdma_enabled();
      opts.needs_staging =
          (!comm_engine.is_rdma_enabled() && info.comm_dev_type == communicator::base::COMMUNICATE_ENGINE_DEV_GPU);
      opts.async = false;
      opts.direct_rdma_enabled = comm_engine.is_rdma_enabled() && !opts.needs_staging;
      auto ret = comm_engine.register_tensor_ex(tensor_key, addr, length, info.comm_dev_type, info.device_id, opts);
      if (!ret.ok()) {
        cleanup_registered_tensors("GPU export");
        rollback_export();
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
      rec.ranges = ranges;
      records_[rkey] = std::move(rec);
    }

    // Metrics and logs
    if (ex_reg_total_) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      attrs.emplace("location", opentelemetry::common::AttributeValue(loc_str(location)));
      ex_reg_total_->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
    }
    LOG(INFO) << "export_chunks: artifact_id=" << key.artifact_id << " location=" << loc_str(location)
              << " ranges=" << ranges.size() << " status=OK";

    return info;
  }

  return absl::InvalidArgumentError("Invalid location for export");
}

absl::Status MemoryExportRegistry::unexport_chunks(
    const loading::ReplicaKey& key,
    const ExportRegistration& info,
    communicator::engine::Communicator& comm_engine) {
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

  if (!first_error.ok()) {
    LOG(WARNING) << "unexport_chunks: artifact_id=" << key.artifact_id << " location=" << loc_str(info.location)
                 << " preserve_export_record=1 error=" << first_error;
    return first_error;
  }

  // Erase record and drop leases (by dropping tokens)
  {
    std::lock_guard<std::mutex> lock(records_mu_);
    ExportKey rkey{.key = key, .location = info.location};
    auto it = records_.find(rkey);
    if (it != records_.end()) {
      if (it->second.stable_lease.has_value()) {
        (void)uma_->release_stable_lease(*(it->second.stable_lease));
      }
      // Update UMA ledger on unexport (always enabled)
      std::vector<uint32_t> idx;
      for (const auto& re : it->second.ranges) {
        for (uint32_t c = re.first; c <= re.second; ++c)
          idx.push_back(c);
      }
      (void)uma_->set_exported(key, info.location, absl::MakeSpan(idx), /*on=*/false);
      // Overwrite stored info to ensure leases are dropped after this function returns
      records_.erase(it);
    }
  }

  LOG(INFO) << "unexport_chunks: artifact_id=" << key.artifact_id << " location=" << loc_str(info.location)
            << " status=OK";

  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
