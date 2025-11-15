// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/memory_export_registry.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include "absl/strings/str_format.h"
#include "core/communicator/engine/engine.h"
#include "core/store/replica/transfer_constants.h"
// OpenTelemetry Metrics API (impl)
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::replica {

namespace {
inline const char* loc_str(common::memory::MemoryLocation loc) {
  return (loc == common::memory::MemoryLocation::GPU) ? "gpu" : "cpu";
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
  // Validate parameters
  if (chunks.empty()) {
    return absl::InvalidArgumentError("No chunks specified for export");
  }
  if (location != common::memory::MemoryLocation::CPU && location != common::memory::MemoryLocation::GPU) {
    return absl::InvalidArgumentError("Invalid location for export");
  }

  ExportRegistration info;
  {
    auto sz_or = uma_->get_artifact_size(key);
    info.artifact_size = sz_or.ok() ? *sz_or : 0;
  }
  info.location = location;
  ExportRecord rec;

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
    auto reg_or = uma_->set_exported(key, location, chunks, /*on=*/true);
    if (!reg_or.ok()) {
      return reg_or.status();
    }
    ranges = reg_or->chunk_ranges;
    rec.uma_keepalive = reg_or->keepalive; // Hold VS pin leases across registration lifetime
    // Derive chunk size from UMA layout to ensure alignment across VS/UMA
    uint64_t kChunk = static_cast<uint64_t>(uma_->get_artifact_chunk_bytes());
    if (auto layout_or = uma_->get_layout(key); layout_or.ok() && layout_or->artifact_chunk_bytes > 0) {
      kChunk = static_cast<uint64_t>(layout_or->artifact_chunk_bytes);
    }
    size_t range_idx = 0;
    for (const auto& [start, end] : ranges) {
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
      auto tensor_key = absl::StrFormat("%s_CPU_chunk_%zu", key.artifact_id, range_idx++);
      tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
      // Avoid registering an MR for VS logical windows; CPU path will be staged for TCP
      opts.register_mr = false;
      // Hint: CPU staged when policy requires. For Phase 1 (TCP), staging happens in transport.
      opts.needs_staging = false;
      opts.async = false;
      // UMA ledger manages CPU lease lifetime; no legacy staging mapping

      auto ret = comm_engine.register_tensor_ex(tensor_key, addr, length, info.comm_dev_type, info.device_id, opts);
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
      auto tensor_key = absl::StrFormat("%s_GPU_chunk_%zu", key.artifact_id, range_idx++);
      tensorcast::communicator::engine::Communicator::RegisterTensorOptions opts;
      opts.register_mr = comm_engine.is_rdma_enabled();
      opts.needs_staging =
          (!comm_engine.is_rdma_enabled() && info.comm_dev_type == communicator::base::COMMUNICATE_ENGINE_DEV_GPU);
      opts.async = false;
      opts.direct_rdma_enabled = comm_engine.is_rdma_enabled() && !opts.needs_staging;
      auto ret = comm_engine.register_tensor_ex(tensor_key, addr, length, info.comm_dev_type, info.device_id, opts);
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

  // Erase record and drop leases (by dropping tokens)
  {
    std::lock_guard<std::mutex> lock(records_mu_);
    ExportKey rkey{.key = key, .location = info.location};
    auto it = records_.find(rkey);
    if (it != records_.end()) {
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
            << " status=" << (first_error.ok() ? "OK" : first_error.message());

  return first_error.ok() ? absl::OkStatus() : first_error;
}

} // namespace tensorcast::store::replica
