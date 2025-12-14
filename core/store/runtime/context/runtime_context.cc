// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/context/runtime_context.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_copy_manager.h"
#include "core/common/memory/host_memory.h"

namespace {
constexpr char kDefaultP2PHost[] = "0.0.0.0";
} // namespace

namespace tensorcast::store::runtime {
namespace {
constexpr absl::Duration kDefaultRuntimeDrainTimeout = absl::Seconds(30);
} // namespace

RuntimeContext::RuntimeContext(const StoreEngineOptions& options)
    : options_(options),
      artifact_chunk_bytes_(
          options.artifact_chunk_bytes == 0 ? tensorcast::common::consts::kArtifactChunkDefault
                                            : options.artifact_chunk_bytes),
      memory_pool_(
          std::make_shared<common::memory::PinnedBufferPool>(options.memory_pool_size, options.tx_slice_bytes)),
      device_manager_(std::make_unique<components::DeviceManager>()),
      replica_registry_(std::make_unique<components::ReplicaRegistry>()),
      metrics_collector_(std::make_unique<components::MetricsCollector>()),
      view_hash_computer_(std::make_shared<ViewHashComputer>(ViewHashConfig{artifact_chunk_bytes_})),
      events_(std::make_unique<RuntimeContextEvents>()),
      ingestion_event_hub_(std::make_unique<ingestion::IngestionEventHub>(events_.get())) {
  if (options_.async_runtime) {
    async_runtime_ = options_.async_runtime;
    owns_async_runtime_ = false;
  } else {
    const size_t cpu_threads = static_cast<size_t>(std::max<int>(1, options_.num_thread));
    const size_t blocking_threads = static_cast<size_t>(std::max<int>(2, options_.num_thread));
    async_runtime_ = std::make_shared<common::AsyncRuntime>(common::AsyncRuntime::Options{
        .cpu_threads = cpu_threads,
        .blocking_threads = blocking_threads,
        .thread_name_prefix = "tensorcast-store",
    });
    owns_async_runtime_ = true;
  }
  common::AsyncCopyManager::instance().set_async_runtime(async_runtime_);
  if (options_.comm_manager) {
    comm_manager_ = options_.comm_manager;
  } else {
    comm_manager_ = std::make_shared<components::CommunicationManager>();
  }
}

RuntimeContext::~RuntimeContext() {
  shutdown();
}

absl::Status RuntimeContext::start() {
  if (started_) {
    return absl::OkStatus();
  }
  auto validate_status = validate_options();
  if (!validate_status.ok()) {
    return validate_status;
  }
  auto comm_status = initialize_communication_manager();
  if (!comm_status.ok()) {
    return comm_status;
  }
  auto device_status = initialize_device_manager();
  if (!device_status.ok()) {
    return device_status;
  }
  auto gs_status = initialize_global_store_client();
  if (!gs_status.ok()) {
    LOG(WARNING) << "RuntimeContext: GlobalStoreClient init failed: " << gs_status;
  }
  if (options_.memory_tier_config.has_value()) {
    auto host_cap_or = common::memory::detect_host_memory_capacity_bytes();
    if (!host_cap_or.ok()) {
      return host_cap_or.status();
    }
    auto budget_or =
        MemoryTierBudget::from_config(*options_.memory_tier_config, *host_cap_or, options_.memory_pool_size);
    if (!budget_or.ok()) {
      return budget_or.status();
    }
    memory_tier_budget_ = std::make_shared<MemoryTierBudget>(std::move(*budget_or));
  }
  metrics_collector_->update_all_metrics(*memory_pool_, *replica_registry_, *device_manager_);
  started_ = true;
  return absl::OkStatus();
}

void RuntimeContext::shutdown() {
  if (!started_) {
    return;
  }
  started_ = false;

  if (async_runtime_ && owns_async_runtime_) {
    async_runtime_->shutdown();
    const absl::Status st = async_runtime_->drain(absl::Now() + kDefaultRuntimeDrainTimeout);
    if (!st.ok()) {
      LOG(FATAL) << "RuntimeContext: AsyncRuntime drain failed during shutdown: " << st;
    }
  }
  if (events_) {
    events_->drain();
  }
  replica_registry_->clear_all();
  global_store_client_.reset();
  comm_manager_.reset();
}

components::DeviceManager& RuntimeContext::device_manager() {
  return *device_manager_;
}

components::ReplicaRegistry& RuntimeContext::replica_registry() {
  return *replica_registry_;
}

components::MetricsCollector& RuntimeContext::metrics_collector() {
  return *metrics_collector_;
}

void RuntimeContext::set_global_store_client_for_testing(std::shared_ptr<components::IGlobalStoreClient> client) {
  global_store_client_ = std::move(client);
  if (global_store_client_) {
    global_store_client_->update_local_endpoint(
        worker_identity_.node_id, worker_identity_.node_address, worker_identity_.grpc_port, worker_identity_.p2p_port);
  }
}

void RuntimeContext::set_worker_identity(components::WorkerIdentity identity) {
  worker_identity_ = std::move(identity);
  if (global_store_client_) {
    global_store_client_->update_local_endpoint(
        worker_identity_.node_id, worker_identity_.node_address, worker_identity_.grpc_port, worker_identity_.p2p_port);
  }
}

RuntimeContextEvents::Publisher RuntimeContext::event_publisher() {
  if (!events_) {
    return RuntimeContextEvents::Publisher();
  }
  return events_->publisher();
}

std::unique_ptr<RuntimeContextEvents::Subscription> RuntimeContext::subscribe_to_events(
    RuntimeContextEvents::Callback callback) {
  if (!events_) {
    return nullptr;
  }
  return events_->subscribe(std::move(callback));
}

void RuntimeContext::drain_events() {
  if (events_) {
    events_->drain();
  }
}

std::string RuntimeContext::mint_publish_context_id() {
  const uint64_t sequence = publish_context_counter_.fetch_add(1, std::memory_order_relaxed);
  const int64_t timestamp = absl::ToUnixNanos(absl::Now());
  const absl::string_view worker =
      worker_identity_.worker_id.empty() ? absl::string_view("local") : absl::string_view(worker_identity_.worker_id);
  return absl::StrCat("pubctx_", worker, "_", timestamp, "_", sequence);
}

ingestion::IngestionEventHub* RuntimeContext::ingestion_event_hub() {
  return ingestion_event_hub_.get();
}

const ingestion::IngestionEventHub* RuntimeContext::ingestion_event_hub() const {
  return ingestion_event_hub_.get();
}

absl::Status RuntimeContext::validate_options() const {
  if (options_.tx_slice_bytes == 0) {
    return absl::InvalidArgumentError("Pinned buffer slice size must be non-zero");
  }
  if (artifact_chunk_bytes_ % options_.tx_slice_bytes != 0) {
    return absl::InvalidArgumentError("artifact_chunk_bytes must be a multiple of tx_slice_bytes");
  }
  const size_t pool_block = memory_pool_->slice_bytes();
  if (pool_block % common::memory::PinnedBufferPool::kDirectIOAlignment != 0) {
    return absl::InvalidArgumentError("Pinned buffer block size must be aligned to DIRECT_IO");
  }
  if (pool_block % common::memory::PinnedBufferPool::kMemoryAlignment != 0) {
    return absl::InvalidArgumentError("Pinned buffer block size must be aligned to page size");
  }

  if (options_.memory_tier_config.has_value()) {
    const auto& tiers = *options_.memory_tier_config;
    if (tiers.stable_bytes == 0) {
      return absl::InvalidArgumentError("engine.memory_tiers.stable_bytes must be greater than 0 when configured");
    }
    if (tiers.preemptible_low_watermark_ratio <= 0.0 || tiers.preemptible_low_watermark_ratio >= 1.0) {
      return absl::InvalidArgumentError("engine.memory_tiers.preemptible_low_watermark_ratio must be between 0 and 1");
    }
    auto host_capacity_or = common::memory::detect_host_memory_capacity_bytes();
    if (!host_capacity_or.ok()) {
      return host_capacity_or.status();
    }
    const uint64_t host_bytes = *host_capacity_or;
    const uint64_t pinned_pool_bytes = options_.memory_pool_size;
    const uint64_t uma_capacity = (host_bytes > pinned_pool_bytes) ? (host_bytes - pinned_pool_bytes) : 0;
    if (tiers.stable_bytes > uma_capacity) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "engine.memory_tiers.stable_bytes=%llu exceeds UMA capacity %llu bytes (host=%llu pinned_pool=%llu)",
              static_cast<uint64_t>(tiers.stable_bytes),
              static_cast<uint64_t>(uma_capacity),
              static_cast<uint64_t>(host_bytes),
              static_cast<uint64_t>(pinned_pool_bytes)));
    }
  }

  return absl::OkStatus();
}

absl::Status RuntimeContext::initialize_device_manager() {
  return device_manager_->initialize();
}

absl::Status RuntimeContext::initialize_communication_manager() {
  if (comm_manager_) {
    if (comm_manager_->is_enabled()) {
      const uint16_t active_port = comm_manager_->listen_port();
      if (options_.p2p_port == 0 && active_port != 0) {
        options_.p2p_port = active_port;
      }
      return absl::OkStatus();
    }
  } else {
    comm_manager_ = std::make_shared<components::CommunicationManager>();
  }

  const std::string listen_host =
      options_.p2p_listen_host.empty() ? std::string{kDefaultP2PHost} : options_.p2p_listen_host;
  const uint16_t requested_port = options_.p2p_port;
  const bool enable_rdma = options_.enable_rdma;

  absl::Status status = comm_manager_->initialize(listen_host, requested_port, enable_rdma);
  if (!status.ok()) {
    return status;
  }

  const uint16_t active_port = comm_manager_->listen_port();
  if (active_port != 0) {
    options_.p2p_port = active_port;
  }

  LOG(INFO) << "RuntimeContext: communication manager listening on " << listen_host << ":" << active_port
            << (enable_rdma ? " (RDMA enabled)" : " (RDMA disabled)");

  return absl::OkStatus();
}

absl::Status RuntimeContext::initialize_global_store_client() {
  if (options_.global_store_address.empty()) {
    return absl::OkStatus();
  }
  components::GlobalStoreClientConfig cfg;
  cfg.global_store_address = options_.global_store_address;

  auto client = std::make_shared<components::GlobalStoreClient>(cfg);
  absl::Status st = client->initialize();
  if (!st.ok()) {
    return st;
  }
  LOG(INFO) << "RuntimeContext: connected to Global Store at " << cfg.global_store_address;
  client->update_local_endpoint(
      worker_identity_.node_id, worker_identity_.node_address, worker_identity_.grpc_port, worker_identity_.p2p_port);
  global_store_client_ = std::move(client);
  return absl::OkStatus();
}

} // namespace tensorcast::store::runtime
