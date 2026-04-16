// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/ha/worker_lifecycle_manager.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <utility>
#include <variant>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/crc/crc32c.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/communicator/misc/utils.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/worker_directory_cache.h"
#include "daemon/util/identity_utils.h"
#include "folly/futures/Future.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

namespace global_store = tensorcast::global_store::v1;
namespace commonpb = common::v1;

using namespace std::chrono_literals;

namespace {

constexpr uint32_t kReconcileRemoveRetireMissThreshold = 2;

struct ReplicaSelector {
  std::string artifact_id;
  std::string view_id;
  commonpb::MemoryType memory_type{commonpb::MEMORY_TYPE_UNSPECIFIED};
  int device_id{0};

  bool operator==(const ReplicaSelector&) const = default;
};

struct ReplicaSelectorHash {
  size_t operator()(const ReplicaSelector& selector) const {
    return absl::HashOf(
        selector.artifact_id, selector.view_id, static_cast<int>(selector.memory_type), selector.device_id);
  }
};

std::optional<ReplicaSelector> replica_selector_from_memory_info(
    std::string_view artifact_id,
    const commonpb::MemoryInfo& memory_info) {
  std::string view_id;
  if (memory_info.has_byte_space() && memory_info.byte_space().kind() == commonpb::BYTE_SPACE_KIND_VIEW) {
    view_id = memory_info.byte_space().id();
  }
  switch (memory_info.memory_type()) {
    case commonpb::MEMORY_TYPE_GPU:
      return ReplicaSelector{
          .artifact_id = std::string(artifact_id),
          .view_id = std::move(view_id),
          .memory_type = commonpb::MEMORY_TYPE_GPU,
          .device_id = static_cast<int>(memory_info.device_id())};
    case commonpb::MEMORY_TYPE_RAM:
      return ReplicaSelector{
          .artifact_id = std::string(artifact_id),
          .view_id = std::move(view_id),
          .memory_type = commonpb::MEMORY_TYPE_RAM,
          .device_id = 0};
    case commonpb::MEMORY_TYPE_DISK:
    case commonpb::MEMORY_TYPE_UNSPECIFIED:
    default:
      return std::nullopt;
  }
}

void append_local_replica_selectors(
    const store::StoreEngine::ReplicaInventoryEntry& entry,
    std::vector<ReplicaSelector>* out) {
  if (entry.memory_location == common::memory::MemoryLocation::GPU) {
    if (entry.key.device.ordinal < 0) {
      VLOG(1) << "Skipping GPU replica with unknown device id: artifact_id=" << entry.key.artifact_id;
    } else {
      const std::string view_id = entry.key.view_id.value_or(std::string());
      out->push_back(
          ReplicaSelector{
              .artifact_id = entry.key.artifact_id,
              .view_id = view_id,
              .memory_type = commonpb::MEMORY_TYPE_GPU,
              .device_id = entry.key.device.ordinal});
    }
    return;
  }
  if (entry.memory_location == common::memory::MemoryLocation::CPU) {
    const std::string view_id = entry.key.view_id.value_or(std::string());
    out->push_back(
        ReplicaSelector{
            .artifact_id = entry.key.artifact_id,
            .view_id = view_id,
            .memory_type = commonpb::MEMORY_TYPE_RAM,
            .device_id = 0});
  }
}

void append_replica_keys_for_selector(
    store::StoreEngine& engine,
    const ReplicaSelector& selector,
    std::vector<store::loading::ReplicaKey>* out) {
  const std::optional<std::string_view> view_id =
      selector.view_id.empty() ? std::nullopt : std::optional<std::string_view>(selector.view_id);
  const auto devices = engine.get_resident_devices(selector.artifact_id, view_id);
  for (const auto& dev : devices) {
    if (selector.memory_type == commonpb::MEMORY_TYPE_GPU) {
      if (dev.type != DeviceType::GPU || dev.ordinal != selector.device_id) {
        continue;
      }
    } else if (selector.memory_type == commonpb::MEMORY_TYPE_RAM) {
      if (dev.type != DeviceType::CPU) {
        continue;
      }
    } else {
      continue;
    }
    const auto device_replicas = engine.list_device_replicas(dev);
    for (const auto& key : device_replicas) {
      const std::string key_view_id = key.view_id.value_or(std::string());
      if (key.artifact_id == selector.artifact_id && key_view_id == selector.view_id) {
        out->push_back(key);
      }
    }
  }
}

opentelemetry::metrics::Counter<uint64_t>* hb_success_counter() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto ctr = meter->CreateUInt64Counter("tc_daemon_ha_heartbeat_success_total");
  return ctr.get();
}

opentelemetry::metrics::Counter<uint64_t>* hb_failure_counter() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto ctr = meter->CreateUInt64Counter("tc_daemon_ha_heartbeat_failure_total");
  return ctr.get();
}

opentelemetry::metrics::Counter<uint64_t>* sync_success_counter() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto ctr = meter->CreateUInt64Counter("tc_daemon_ha_sync_success_total");
  return ctr.get();
}

opentelemetry::metrics::Counter<uint64_t>* sync_failure_counter() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto ctr = meter->CreateUInt64Counter("tc_daemon_ha_sync_failure_total");
  return ctr.get();
}

opentelemetry::metrics::Counter<uint64_t>* reconcile_enqueue_suppressed_counter() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto ctr = meter->CreateUInt64Counter("tc_daemon_ha_reconcile_enqueue_suppressed_total");
  return ctr.get();
}

opentelemetry::metrics::UpDownCounter<int64_t>* outage_mode_active_counter() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto ctr = meter->CreateInt64UpDownCounter("tc_daemon_ha_outage_mode_active");
  return ctr.get();
}

opentelemetry::metrics::Histogram<double>* reconnect_latency_histogram_ms() {
  static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  static auto hist = meter->CreateDoubleHistogram("tc_daemon_ha_reconnect_latency_ms");
  return hist.get();
}

enum class HeartbeatFailureClass {
  kConflict,
  kIdentity,
  kConnectivity,
  kOther,
};

const char* heartbeat_failure_class_to_cstr(HeartbeatFailureClass klass) {
  switch (klass) {
    case HeartbeatFailureClass::kConflict:
      return "conflict";
    case HeartbeatFailureClass::kIdentity:
      return "identity";
    case HeartbeatFailureClass::kConnectivity:
      return "connectivity";
    case HeartbeatFailureClass::kOther:
    default:
      return "other";
  }
}

bool message_contains_marker(absl::string_view message, std::initializer_list<absl::string_view> markers) {
  const std::string lowered = absl::AsciiStrToLower(message);
  return std::any_of(markers.begin(), markers.end(), [&](absl::string_view marker) {
    return lowered.find(marker) != std::string::npos;
  });
}

HeartbeatFailureClass classify_heartbeat_failure(const absl::Status& status) {
  if (absl::IsAborted(status) ||
      message_contains_marker(
          status.message(),
          {"write-write conflict", "conflict on tuple deletion", "conflict on update", "serialization"})) {
    return HeartbeatFailureClass::kConflict;
  }
  if (absl::IsNotFound(status) || absl::IsFailedPrecondition(status) ||
      message_contains_marker(status.message(), {"status_not_found", "worker not found"})) {
    return HeartbeatFailureClass::kIdentity;
  }
  if (absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status) || absl::IsCancelled(status) ||
      absl::IsResourceExhausted(status)) {
    return HeartbeatFailureClass::kConnectivity;
  }
  return HeartbeatFailureClass::kOther;
}

std::chrono::milliseconds bounded_backoff(
    uint32_t failure_count,
    std::chrono::milliseconds base,
    std::chrono::milliseconds max_backoff) {
  if (failure_count == 0) {
    return base;
  }
  const uint32_t shift = std::min<uint32_t>(failure_count - 1, 8);
  const auto multiplier = static_cast<int64_t>(1ULL << shift);
  const auto candidate = std::chrono::milliseconds(base.count() * multiplier);
  return std::min(candidate, max_backoff);
}

enum class StateSyncFailureClass {
  kConflict,
  kConnectivity,
  kOther,
};

const char* state_sync_failure_class_to_cstr(StateSyncFailureClass klass) {
  switch (klass) {
    case StateSyncFailureClass::kConflict:
      return "conflict";
    case StateSyncFailureClass::kConnectivity:
      return "connectivity";
    case StateSyncFailureClass::kOther:
    default:
      return "other";
  }
}

StateSyncFailureClass classify_state_sync_failure(const absl::Status& status) {
  if (absl::IsAborted(status) ||
      message_contains_marker(
          status.message(),
          {"write-write conflict", "conflict on tuple deletion", "conflict on update", "serialization"})) {
    return StateSyncFailureClass::kConflict;
  }
  if (absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status) || absl::IsCancelled(status) ||
      absl::IsResourceExhausted(status) ||
      message_contains_marker(status.message(), {"connection refused", "failed to connect to all addresses"})) {
    return StateSyncFailureClass::kConnectivity;
  }
  return StateSyncFailureClass::kOther;
}

std::chrono::milliseconds state_sync_failure_backoff(StateSyncFailureClass failure_class, uint32_t failure_count) {
  switch (failure_class) {
    case StateSyncFailureClass::kConnectivity:
      return bounded_backoff(failure_count, 500ms, 30s);
    case StateSyncFailureClass::kConflict:
      return bounded_backoff(failure_count, 100ms, 3s);
    case StateSyncFailureClass::kOther:
    default:
      return bounded_backoff(failure_count, 200ms, 10s);
  }
}

bool is_loopback_or_unspecified(absl::string_view addr) {
  if (addr.empty())
    return true;
  if (addr == "localhost" || addr == "ip6-localhost" || addr == "*" || addr == "0.0.0.0")
    return true;

  absl::string_view trimmed = addr;
  if (trimmed.front() == '[' && trimmed.back() == ']') {
    trimmed.remove_prefix(1);
    trimmed.remove_suffix(1);
  }

  return trimmed == "127.0.0.1" || trimmed == "::" || trimmed == "::1" || trimmed == "0:0:0:0:0:0:0:1";
}

struct ParsedEndpoint {
  std::string host;
  uint16_t port{0};
};

std::optional<ParsedEndpoint> parse_endpoint(std::string_view address) {
  if (address.empty()) {
    return std::nullopt;
  }
  std::string_view host;
  std::string_view port_part;
  if (address.front() == '[') {
    const auto close = address.find(']');
    if (close == std::string_view::npos) {
      return std::nullopt;
    }
    host = address.substr(1, close - 1);
    if (close + 1 >= address.size() || address[close + 1] != ':') {
      return std::nullopt;
    }
    port_part = address.substr(close + 2);
  } else {
    const auto pos = address.rfind(':');
    if (pos == std::string_view::npos) {
      return std::nullopt;
    }
    host = address.substr(0, pos);
    port_part = address.substr(pos + 1);
  }
  if (host.empty() || port_part.empty()) {
    return std::nullopt;
  }
  uint32_t port = 0;
  try {
    port = static_cast<uint32_t>(std::stoi(std::string(port_part)));
  } catch (const std::exception&) {
    return std::nullopt;
  }
  if (port == 0 || port > 65535) {
    return std::nullopt;
  }
  return ParsedEndpoint{std::string(host), static_cast<uint16_t>(port)};
}

std::optional<std::string> resolve_route_ip(const ParsedEndpoint& endpoint) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;

  addrinfo* res = nullptr;
  const std::string port_str = std::to_string(endpoint.port);
  if (getaddrinfo(endpoint.host.c_str(), port_str.c_str(), &hints, &res) != 0) {
    return std::nullopt;
  }
  absl::Cleanup cleanup = [res] { freeaddrinfo(res); };

  for (addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
    const int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) {
      continue;
    }
    if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
      sockaddr_in local{};
      socklen_t len = sizeof(local);
      if (getsockname(fd, reinterpret_cast<sockaddr*>(&local), &len) == 0) {
        char buffer[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &local.sin_addr, buffer, sizeof(buffer)) != nullptr) {
          ::close(fd);
          return std::string(buffer);
        }
      }
    }
    ::close(fd);
  }
  return std::nullopt;
}

} // namespace

const char* WorkerLifecycleManager::advertised_source_to_cstr(AdvertisedAddressSource source) {
  switch (source) {
    case AdvertisedAddressSource::kExplicit:
      return "explicit";
    case AdvertisedAddressSource::kListen:
      return "listen";
    case AdvertisedAddressSource::kRoute:
      return "route";
    case AdvertisedAddressSource::kDefault:
      return "default";
  }
  return "unknown";
}

WorkerLifecycleManager::WorkerLifecycleManager(
    gsl::not_null<std::shared_ptr<store::StoreEngine>> engine,
    WorkerLifecyclePorts ports,
    Options opts)
    : engine_(std::move(engine)),
      ports_(ports),
      opts_(std::move(opts)),
      daemon_id_(ports.identity_store.daemon_id()),
      global_store_(make_global_store_client(opts_)),
      node_id_(derive_node_id()) {
  if (engine_->promotion_manager() != nullptr) {
    engine_->set_promotion_sync_hooks(
        store::runtime::PromotionSyncHooks{
            .request_state_sync = [this]() { request_state_sync(); },
        });
  }
}

std::vector<store::StoreEngine::ReplicaInventoryEntry> WorkerLifecycleManager::collect_ha_inventory() const {
  std::vector<store::StoreEngine::ReplicaInventoryEntry> inventory = engine_->get_ha_inventory();
  const auto routable_inventory = ports_.lip_manager.list_active_routable_inventory();
  if (routable_inventory.empty()) {
    return inventory;
  }

  absl::flat_hash_set<store::loading::ReplicaKey, store::loading::ReplicaKeyHash> seen_keys;
  seen_keys.reserve(inventory.size() + routable_inventory.size());
  for (const auto& entry : inventory) {
    seen_keys.insert(entry.key);
  }

  inventory.reserve(inventory.size() + routable_inventory.size());
  for (const auto& routable_entry : routable_inventory) {
    store::loading::ReplicaKey key;
    key.artifact_id = routable_entry.key.artifact_id;
    if (!routable_entry.key.view_id.empty()) {
      key.view_id = routable_entry.key.view_id;
    }
    key.device = store::DeviceRegistry::instance().gpu_key(routable_entry.key.device_id);
    if (!seen_keys.insert(key).second) {
      continue;
    }

    store::StoreEngine::ReplicaInventoryEntry inventory_entry;
    inventory_entry.key = std::move(key);
    inventory_entry.size_bytes = routable_entry.total_size;
    inventory_entry.memory_location = common::memory::MemoryLocation::GPU;
    inventory_entry.is_available = !routable_entry.remote_memory_keys.empty();
    inventory_entry.publish_state = store::StoreEngine::ReplicaPublishState::kPublished;
    inventory_entry.remote_memory_keys = routable_entry.remote_memory_keys;
    inventory_entry.buffer_sizes = routable_entry.buffer_sizes;
    inventory_entry.export_state = inventory_entry.is_available ? store::runtime::ReplicaExportState::kExportable
                                                                : store::runtime::ReplicaExportState::kPresenceOnly;
    inventory.push_back(std::move(inventory_entry));
  }

  return inventory;
}

gsl::not_null<std::shared_ptr<store::components::IGlobalStoreClient>> WorkerLifecycleManager::make_global_store_client(
    const Options& opts) {
  if (opts.global_store_client) {
    return gsl::not_null<std::shared_ptr<store::components::IGlobalStoreClient>>{opts.global_store_client};
  }
  ABSL_CHECK(!opts.global_store_addr.empty()) << "WorkerLifecycleManager requires a Global Store address";

  store::components::GlobalStoreClientConfig cfg;
  cfg.global_store_address = opts.global_store_addr;
  cfg.cluster_token = opts.cluster_token;
  std::shared_ptr<store::components::IGlobalStoreClient> client =
      std::make_shared<store::components::GlobalStoreClient>(std::move(cfg));
  return gsl::not_null<std::shared_ptr<store::components::IGlobalStoreClient>>{std::move(client)};
}

absl::StatusOr<WorkerLifecycleManager::ResolvedAdvertisedAddress> WorkerLifecycleManager::resolve_advertised_address(
    const WorkerLifecycleManager::Options& opts) {
  if (!opts.advertise_host.empty()) {
    if (is_loopback_or_unspecified(opts.advertise_host)) {
      return absl::InvalidArgumentError(
          "Global Store registration requires advertise_host to be routable when configured; "
          "loopback/unspecified values are not allowed.");
    }
    return ResolvedAdvertisedAddress{.host = opts.advertise_host, .source = AdvertisedAddressSource::kExplicit};
  }

  const std::string listen_host = WorkerLifecycleManager::host_from_listen(opts.listen_addr);
  if (!listen_host.empty() && !is_loopback_or_unspecified(listen_host)) {
    return ResolvedAdvertisedAddress{.host = listen_host, .source = AdvertisedAddressSource::kListen};
  }

  if (!opts.global_store_addr.empty()) {
    const auto endpoint = parse_endpoint(opts.global_store_addr);
    if (endpoint.has_value() && !is_loopback_or_unspecified(endpoint->host)) {
      const auto route_ip = resolve_route_ip(*endpoint);
      if (route_ip.has_value() && !is_loopback_or_unspecified(*route_ip)) {
        return ResolvedAdvertisedAddress{.host = *route_ip, .source = AdvertisedAddressSource::kRoute};
      }
    }
  }

  const std::string default_ip = communicator::misc::get_default_ip();
  if (!default_ip.empty() && !is_loopback_or_unspecified(default_ip)) {
    return ResolvedAdvertisedAddress{.host = default_ip, .source = AdvertisedAddressSource::kDefault};
  }

  if (is_loopback_or_unspecified(listen_host)) {
    return absl::InvalidArgumentError(
        "Global Store registration requires a routable advertise_host. Provide --advertise_host with a non-loopback "
        "address or configure listen_addr accordingly.");
  }

  return ResolvedAdvertisedAddress{.host = listen_host, .source = AdvertisedAddressSource::kListen};
}

absl::Status WorkerLifecycleManager::start() {
  auto st = global_store_->initialize();
  if (!st.ok()) {
    return st;
  }
  auto node_addr_or = resolve_advertised_address(opts_);
  if (!node_addr_or.ok()) {
    return node_addr_or.status();
  }
  const std::string node_addr = node_addr_or->host;
  const uint32_t grpc_port = port_from_listen(opts_.listen_addr);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    node_address_ = node_addr;
  }

  if (grpc_port == 0) {
    return absl::InvalidArgumentError(
        "WorkerLifecycleManager requires listen_addr to include a non-zero port for gRPC registration.");
  }

  if (opts_.p2p_port == 0) {
    return absl::InvalidArgumentError(
        "WorkerLifecycleManager requires a non-zero p2p_port when Global Store HA is enabled; configure --p2p_listen.");
  }

  LOG(INFO) << "Resolved advertised address for Global Store registration: " << node_addr
            << " (source=" << advertised_source_to_cstr(node_addr_or->source) << ")";
  LOG(INFO) << "Global Store registration endpoints: listen=" << opts_.listen_addr << " advertise=" << node_addr << ":"
            << grpc_port << " p2p_port=" << opts_.p2p_port;

  absl::StatusOr<store::components::WorkerRegistrationInfo> reg_or;
  {
    std::lock_guard<std::mutex> lock(worker_control_plane_rpc_mu_);
    reg_or = global_store_->register_worker(
        node_id_,
        node_addr,
        grpc_port,
        opts_.p2p_port,
        engine_->get_mem_pool_size(),
        engine_->get_available_memory(),
        /*is_recovery_registration=*/false,
        /*previous_worker_id=*/"",
        /*daemon_id=*/daemon_id_,
        /*capability_flags=*/opts_.capability_flags);
  }
  if (!reg_or.ok())
    return reg_or.status();
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    worker_id_ = reg_or->worker_id;
    state_version_ = reg_or->expected_state_version;
    reconcile_generation_.store(std::max<uint64_t>(1, reg_or->reconcile_generation));
  }
  ports_.identity_store.set_registered(reg_or->worker_id, node_id_);
  // Propagate worker identity into the engine so subsequent GS registrations
  // use the real worker_id instead of a placeholder.
  engine_->set_worker_identity(reg_or->worker_id, node_id_, node_addr, grpc_port, opts_.p2p_port);
  ports_.worker_directory_cache.update_local_entry(
      WorkerDirectoryCache::Entry{
          .daemon_id = daemon_id_,
          .worker_id = reg_or->worker_id,
          .node_id = node_id_,
          .node_address = node_addr,
          .grpc_port = grpc_port,
          .p2p_port = opts_.p2p_port,
          .address = absl::StrCat(node_addr, ":", grpc_port),
          .capability_flags = opts_.capability_flags,
      });
  const uint64_t epoch_seed = static_cast<uint64_t>(absl::ToUnixNanos(absl::Now()));
  state_sync_epoch_.store(epoch_seed);
  state_sync_request_id_.store(0);
  state_sync_enqueue_suppressed_.store(0);
  state_sync_consecutive_failures_.store(0);
  state_sync_next_retry_ns_.store(0);
  state_sync_outage_mode_active_.store(false);
  state_sync_outage_enter_ns_.store(0);
  last_reconnect_latency_ms_.store(0);
  state_sync_last_progress_ns_.store(0);
  state_sync_restart_pending_.store(false);

  runtime_event_subscription_ = engine_->subscribe_to_runtime_events([this](const store::runtime::RuntimeEvent& event) {
    if (stop_.load()) {
      return;
    }
    if (event.type != store::runtime::RuntimeEventType::kReplicaLoaded &&
        event.type != store::runtime::RuntimeEventType::kReplicaEvicted) {
      return;
    }
    const auto* payload = std::get_if<store::runtime::ReplicaLifecycleEvent>(&event.payload);
    if (!payload) {
      return;
    }
    const auto state = engine_->get_replica_publish_state(payload->key);
    if (state == store::StoreEngine::ReplicaPublishState::kLocalOnly) {
      return;
    }
    if (event.type == store::runtime::RuntimeEventType::kReplicaEvicted) {
      auto* promotion = engine_->promotion_manager();
      if (promotion != nullptr) {
        auto& async_runtime = ports_.async_runtime;
        if (!async_runtime.is_shutting_down()) {
          auto executor = async_runtime.blocking_executor();
          const store::loading::ReplicaKey key = payload->key;
          executor->add([promotion, key]() {
            auto st = promotion->finalize_replica_demotion(key);
            if (!st.ok()) {
              LOG(WARNING) << "finalize_replica_demotion failed for evicted " << key << ": " << st;
            }
          });
        }
      }
    }
    request_state_sync();
  });

  memory_tier_enabled_ = engine_->get_memory_tier_config().has_value();
  if (memory_tier_enabled_) {
    reconcile_memory_tier_leases_once();
  }

  stop_.store(false);
  hb_epoch_.store(0);
  hb_thread_ = std::thread(&WorkerLifecycleManager::heartbeat_loop, this);
  state_sync_thread_ = std::thread(&WorkerLifecycleManager::state_sync_loop, this, state_sync_epoch_.load());
  if (opts_.chunk_sync_interval_ms > 0) {
    sync_thread_ = std::thread(&WorkerLifecycleManager::chunk_sync_loop, this);
  }
  monitor_thread_ = std::thread(&WorkerLifecycleManager::monitor_loop, this);
  const uint64_t baseline = sync_success_.load();
  request_state_sync();
  const auto bootstrap_timeout = state_sync_stall_budget().value_or(std::chrono::seconds(10));
  if (!wait_for_state_sync_success(baseline, bootstrap_timeout)) {
    LOG(WARNING) << "Bootstrap reconcile did not complete before timeout";
  }
  return absl::OkStatus();
}

void WorkerLifecycleManager::stop() {
  // Idempotent stop: return if we've already executed shutdown sequence.
  bool expected = false;
  if (!stop_called_.compare_exchange_strong(expected, true)) {
    return;
  }
  stop_.store(true);
  if (state_sync_outage_mode_active_.exchange(false)) {
    if (auto* gauge = outage_mode_active_counter()) {
      gauge->Add(-1);
    }
  }
  state_sync_outage_enter_ns_.store(0);
  stop_cv_.notify_all();
  state_sync_cv_.notify_all();
  sync_success_cv_.notify_all();
  runtime_event_subscription_.reset();
  if (hb_thread_.joinable())
    hb_thread_.join();
  if (state_sync_thread_.joinable())
    state_sync_thread_.join();
  if (sync_thread_.joinable())
    sync_thread_.join();
  if (monitor_thread_.joinable())
    monitor_thread_.join();
  {
    std::vector<std::thread> retired;
    {
      std::lock_guard<std::mutex> lock(retired_threads_mu_);
      retired.swap(retired_threads_);
    }
    for (auto& t : retired) {
      if (t.joinable()) {
        t.join();
      }
    }
  }
  // Best-effort disable remote access before unregistering worker
  for (const auto& info : engine_->get_all_replicas_info()) {
    if (!info.is_registered_for_comm)
      continue;
    if (info.key.device.type != DeviceType::GPU) {
      continue;
    }
    auto st = engine_->disable_remote_replica_access(info.key, common::memory::MemoryLocation::GPU);
    if (!st.ok()) {
      LOG(WARNING) << "disable_remote_replica_access failed during stop: key=" << info.key << ": " << st;
    }
  }
  std::string worker_id_snapshot;
  {
    std::scoped_lock lock(state_mu_);
    worker_id_snapshot = worker_id_;
  }
  if (!worker_id_snapshot.empty()) {
    const std::string id = worker_id_snapshot;
    auto st = unregister_worker_single_flight(id, /*is_graceful_shutdown=*/true);
    if (!st.ok()) {
      LOG(WARNING) << "GlobalStore unregister_worker failed: " << st;
    } else {
      LOG(INFO) << "GlobalStore unregister_worker succeeded for worker_id=" << id;
      // Clear identity to prevent any subsequent attempts (e.g., destructor) from retrying.
      std::scoped_lock lock(state_mu_);
      worker_id_.clear();
    }
  }
}

absl::Status WorkerLifecycleManager::unregister_worker_single_flight(
    std::string_view worker_id,
    bool is_graceful_shutdown) {
  bool expected = false;
  if (!unregister_worker_submitted_.compare_exchange_strong(expected, true)) {
    VLOG(1) << "Skipping duplicate GlobalStore unregister_worker for worker_id=" << worker_id;
    return absl::OkStatus();
  }

  std::lock_guard<std::mutex> lock(worker_control_plane_rpc_mu_);
  return global_store_->unregister_worker_idempotent(worker_id, is_graceful_shutdown);
}

bool WorkerLifecycleManager::wait_for_stop(std::chrono::milliseconds interval) {
  std::unique_lock<std::mutex> lock(stop_mu_);
  return stop_cv_.wait_for(lock, interval, [this]() { return stop_.load(); });
}

store::components::RpcOptions WorkerLifecycleManager::build_rpc_options(
    int timeout_ms,
    std::optional<int32_t> max_retries) const {
  store::components::RpcOptions opts;
  if (timeout_ms > 0) {
    opts.timeout = absl::Milliseconds(timeout_ms);
  }
  if (max_retries.has_value()) {
    opts.max_retries = static_cast<uint32_t>(*max_retries);
  }
  return opts;
}

store::components::StateSyncToken WorkerLifecycleManager::next_state_sync_token() {
  return store::components::StateSyncToken{
      .generation = reconcile_generation_.load(),
      .request_seq = state_sync_request_id_.load() + 1,
  };
}

void WorkerLifecycleManager::mark_state_sync_progress() {
  const auto now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
  state_sync_last_progress_ns_.store(now_ns);
}

std::optional<std::chrono::milliseconds> WorkerLifecycleManager::state_sync_stall_budget() const {
  constexpr uint32_t kDefaultRpcMaxRetries = 3;
  const int base_timeout_ms = opts_.state_sync_rpc_timeout_ms;
  if (base_timeout_ms <= 0) {
    return std::nullopt;
  }
  const uint32_t state_sync_retries = opts_.state_sync_rpc_max_retries.value_or(kDefaultRpcMaxRetries);
  const uint32_t retries = state_sync_retries;
  const int64_t attempts = static_cast<int64_t>(retries) + 1;
  const int64_t budget_ms =
      static_cast<int64_t>(base_timeout_ms) * attempts + std::min<int64_t>(base_timeout_ms, 1000) * attempts;
  return std::chrono::milliseconds(budget_ms);
}

void WorkerLifecycleManager::request_state_sync() {
  uint64_t expected = 0;
  if (!state_sync_requests_.compare_exchange_strong(expected, 1)) {
    state_sync_enqueue_suppressed_.fetch_add(1);
    if (auto* counter = reconcile_enqueue_suppressed_counter()) {
      counter->Add(1);
    }
    return;
  }
  state_sync_cv_.notify_all();
}

bool WorkerLifecycleManager::wait_for_state_sync_success(uint64_t baseline, std::chrono::milliseconds timeout) {
  if (timeout.count() <= 0) {
    return sync_success_.load() > baseline;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::unique_lock<std::mutex> lock(sync_success_mu_);
  return sync_success_cv_.wait_until(
      lock, deadline, [this, baseline]() { return stop_.load() || sync_success_.load() > baseline; });
}

void WorkerLifecycleManager::queue_obsolete_replicas(std::vector<std::string> obsolete) {
  if (obsolete.empty()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(obsolete_mu_);
    pending_obsolete_replicas_.insert(pending_obsolete_replicas_.end(), obsolete.begin(), obsolete.end());
  }
  obsolete_pending_.store(true);
  state_sync_cv_.notify_all();
}

void WorkerLifecycleManager::retire_thread(std::thread* thread) {
  if (!thread || !thread->joinable()) {
    return;
  }
  std::lock_guard<std::mutex> lock(retired_threads_mu_);
  retired_threads_.push_back(std::move(*thread));
}

void WorkerLifecycleManager::heartbeat_loop() {
  hb_alive_.store(true);
  const uint64_t epoch = hb_epoch_.load();
  const auto interval = std::chrono::milliseconds(opts_.heartbeat_interval_ms);
  uint32_t consecutive_heartbeat_failures = 0;
  uint32_t consecutive_reregister_failures = 0;
  auto next_reregister_attempt_at = std::chrono::steady_clock::time_point::min();
  try {
    while (!stop_.load() && hb_epoch_.load() == epoch) {
      // Prepare enhanced heartbeat fields
      const bool accepting = !ports_.shutdown_signal.is_shutting_down();
      const auto inventory = collect_ha_inventory();
      absl::flat_hash_set<std::string> registered_set;
      registered_set.reserve(inventory.size());
      for (const auto& entry : inventory) {
        registered_set.insert(entry.key.artifact_id);
      }
      std::vector<std::string> registered_ids;
      registered_ids.reserve(registered_set.size());
      for (const auto& id : registered_set) {
        registered_ids.push_back(id);
      }
      // Compute simple checksum over current snapshot
      std::string worker_id;
      std::string node_address;
      uint64_t state_version = 0;
      int64_t last_sync_success_ts = 0;
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        worker_id = worker_id_;
        node_address = node_address_;
        state_version = state_version_;
        last_sync_success_ts = last_sync_success_ts_;
      }
      const std::string checksum = compute_state_checksum(node_id_, node_address, opts_.p2p_port, inventory);
      {
        std::lock_guard<std::mutex> lock(state_mu_);
        state_checksum_ = checksum;
      }
      // Heartbeat must remain independent from state-sync/reconcile RPC latency.
      // Those paths can issue long control-plane calls; sharing one serial lock
      // can stall heartbeats and trigger false liveness loss under load.
      absl::StatusOr<global_store::WorkerHeartbeatResponse> hb_or = global_store_->send_heartbeat_enhanced(
          worker_id,
          engine_->get_available_memory(),
          accepting,
          state_version,
          checksum,
          registered_ids,
          last_sync_success_ts,
          global_store::CONNECTION_STATUS_CONNECTED,
          build_rpc_options(opts_.heartbeat_rpc_timeout_ms, opts_.heartbeat_rpc_max_retries),
          daemon_id_,
          opts_.capability_flags);
      if (stop_.load() || hb_epoch_.load() != epoch) {
        break;
      }
      auto loop_delay = interval;
      if (!hb_or.ok()) {
        const absl::Status hb_status = hb_or.status();
        ++consecutive_heartbeat_failures;
        const HeartbeatFailureClass failure_class = classify_heartbeat_failure(hb_status);
        const auto heartbeat_backoff = bounded_backoff(consecutive_heartbeat_failures, std::max(interval, 100ms), 5s);
        loop_delay = std::max(loop_delay, heartbeat_backoff);
        LOG(WARNING) << "Enhanced heartbeat failed: " << hb_status
                     << " class=" << heartbeat_failure_class_to_cstr(failure_class)
                     << " consecutive_failures=" << consecutive_heartbeat_failures
                     << " backoff_ms=" << loop_delay.count();
        hb_failure_.fetch_add(1);
        if (auto* counter = hb_failure_counter()) {
          counter->Add(1);
        }
        // Re-registration is reserved for identity-loss failures (e.g., worker row
        // missing after control-plane restart). Conflict/connectivity classes stay
        // on backoff-only path to prevent re-registration storms.
        if (failure_class == HeartbeatFailureClass::kIdentity && global_store_->is_connected()) {
          const auto now = std::chrono::steady_clock::now();
          if (now >= next_reregister_attempt_at) {
            auto st_re = reregister_worker(/*preserve_identity=*/true);
            if (!st_re.ok()) {
              ++consecutive_reregister_failures;
              const auto reregister_backoff = bounded_backoff(consecutive_reregister_failures, 200ms, 10s);
              next_reregister_attempt_at = now + reregister_backoff;
              loop_delay = std::max(loop_delay, reregister_backoff);
              LOG(WARNING) << "Re-registration attempt failed: " << st_re
                           << " consecutive_reregister_failures=" << consecutive_reregister_failures
                           << " next_reregister_backoff_ms=" << reregister_backoff.count();
            } else {
              consecutive_reregister_failures = 0;
              next_reregister_attempt_at = std::chrono::steady_clock::time_point::min();
            }
          } else {
            const auto until_next =
                std::chrono::duration_cast<std::chrono::milliseconds>(next_reregister_attempt_at - now);
            if (until_next.count() > 0) {
              loop_delay = std::max(loop_delay, until_next);
              VLOG(1) << "Skipping re-registration due to cooldown for worker_id=" << worker_id
                      << " wait_ms=" << until_next.count();
            }
          }
        } else if (failure_class == HeartbeatFailureClass::kConflict) {
          VLOG(1) << "Suppressing re-registration for conflict-class heartbeat failure";
        }
      } else {
        const auto& hb = *hb_or;
        consecutive_heartbeat_failures = 0;
        consecutive_reregister_failures = 0;
        next_reregister_attempt_at = std::chrono::steady_clock::time_point::min();
        hb_success_.fetch_add(1);
        if (auto* counter = hb_success_counter()) {
          counter->Add(1);
        }
        last_hb_ts_s_.store(
            static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                    .count()));
        // Apply any explicit obsolete replicas immediately
        if (hb.obsolete_replicas_size() > 0) {
          std::vector<std::string> obs;
          obs.reserve(hb.obsolete_replicas_size());
          for (const auto& id : hb.obsolete_replicas())
            obs.push_back(id);
          queue_obsolete_replicas(std::move(obs));
        }
        const bool needs_sync = hb.state_sync_required() ||
            (hb.expected_state_version() > 0 && hb.expected_state_version() != state_version);
        if (needs_sync) {
          request_state_sync();
        }
      }
      if (wait_for_stop(loop_delay)) {
        break;
      }
      if (hb_epoch_.load() != epoch) {
        break;
      }
      hb_ticks_.fetch_add(1);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "heartbeat_loop crashed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "heartbeat_loop crashed: unknown exception";
  }
  // A retired epoch must not clobber liveness for a newer heartbeat thread.
  if (hb_epoch_.load() == epoch) {
    hb_alive_.store(false);
  }
}

void WorkerLifecycleManager::state_sync_loop(uint64_t epoch) {
  state_sync_alive_.store(true);
  mark_state_sync_progress();
  const auto maintenance_interval = std::chrono::milliseconds(opts_.heartbeat_interval_ms);
  auto next_maintenance = std::chrono::steady_clock::now() + maintenance_interval;
  const auto memory_tier_cfg = engine_->get_memory_tier_config();
  const bool publish_memory_tiers = memory_tier_cfg.has_value();
  const std::string memory_tier_config_json = memory_tier_cfg.has_value()
      ? absl::StrFormat(
            R"({"enable_preemptible_memory":%s,"stable_bytes":%llu,"preemptible_limit_bytes":%llu,"preemptible_low_watermark_ratio":%.3f})",
            memory_tier_cfg->enable_preemptible_memory ? "true" : "false",
            static_cast<uint64_t>(memory_tier_cfg->stable_bytes),
            static_cast<uint64_t>(memory_tier_cfg->preemptible_limit_bytes),
            memory_tier_cfg->preemptible_low_watermark_ratio)
      : "{}";
  try {
    while (!stop_.load() && state_sync_epoch_.load() == epoch) {
      {
        std::unique_lock<std::mutex> lock(state_sync_mu_);
        state_sync_cv_.wait_until(lock, next_maintenance, [this, epoch]() {
          return stop_.load() || state_sync_epoch_.load() != epoch || state_sync_requests_.load() > 0 ||
              obsolete_pending_.load() || retire_pending_.load();
        });
      }
      mark_state_sync_progress();
      if (stop_.load() || state_sync_epoch_.load() != epoch) {
        break;
      }

      std::vector<std::string> obsolete;
      if (obsolete_pending_.load()) {
        std::lock_guard<std::mutex> lock(obsolete_mu_);
        obsolete.swap(pending_obsolete_replicas_);
        obsolete_pending_.store(false);
      }
      if (!obsolete.empty()) {
        apply_obsolete_replicas(obsolete);
      }

      const uint64_t requests = state_sync_requests_.exchange(0);
      if (requests > 0) {
        perform_state_sync(epoch);
        state_sync_ticks_.fetch_add(1);
      }

      process_retire_queue();

      const auto now = std::chrono::steady_clock::now();
      if (now >= next_maintenance) {
        if (publish_memory_tiers) {
          auto snap_opt = engine_->get_memory_tier_snapshot();
          if (snap_opt.has_value()) {
            const auto snap = *snap_opt;
            std::string worker_id;
            {
              std::lock_guard<std::mutex> lock(state_mu_);
              worker_id = worker_id_;
            }
            store::components::MemoryTierStatusPayload payload;
            payload.node_id = node_id_;
            payload.worker_id = worker_id;
            payload.stable_total_bytes = snap.stable_total_bytes;
            payload.stable_used_bytes = snap.stable_used_bytes;
            payload.preemptible_total_bytes = snap.preemptible_total_bytes;
            payload.preemptible_marked_bytes = snap.preemptible_marked_bytes;
            payload.faults_per_sec = snap.faults_per_sec;
            payload.rehydrate_p99_ns = snap.rehydrate_p99_ns;
            payload.enable_preemptible = memory_tier_cfg->enable_preemptible_memory;
            payload.memory_tier_config_json = memory_tier_config_json;
            payload.epoch_ns = static_cast<uint64_t>(absl::ToUnixNanos(absl::Now()));
            auto mt_status = global_store_->publish_memory_tier_status(payload);
            if (!mt_status.ok()) {
              LOG(WARNING) << "PublishMemoryTierStatus failed: " << mt_status;
            }
          }
        }
        if (memory_tier_enabled_) {
          reconcile_memory_tier_leases_once();
        }
        next_maintenance = now + maintenance_interval;
      }
      state_sync_ticks_.fetch_add(1);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "state_sync_loop crashed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "state_sync_loop crashed: unknown exception";
  }
  // A retired epoch must not clobber liveness for a newer state-sync thread.
  if (state_sync_epoch_.load() == epoch) {
    state_sync_alive_.store(false);
  }
}

void WorkerLifecycleManager::perform_state_sync(uint64_t epoch) {
  if (stop_.load() || state_sync_epoch_.load() != epoch) {
    return;
  }
  state_sync_inflight_.store(true);
  absl::Cleanup inflight_guard([this]() { state_sync_inflight_.store(false); });
  mark_state_sync_progress();
  const int64_t next_retry_ns = state_sync_next_retry_ns_.load();
  if (next_retry_ns > 0) {
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (now_ns < next_retry_ns) {
      const auto wait_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(next_retry_ns - now_ns));
      if (wait_duration.count() > 0) {
        VLOG(1) << "State sync retry backoff active; delaying reconcile for " << wait_duration.count() << "ms";
        if (wait_for_stop(wait_duration)) {
          return;
        }
        if (stop_.load() || state_sync_epoch_.load() != epoch) {
          return;
        }
      }
    }
  }
  const auto inventory = collect_ha_inventory();
  std::string worker_id;
  std::string node_address;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    worker_id = worker_id_;
    node_address = node_address_;
  }
  if (worker_id.empty()) {
    VLOG(1) << "Skipping state sync without registered worker_id";
    return;
  }
  const std::string checksum = compute_state_checksum(node_id_, node_address, opts_.p2p_port, inventory);
  if (state_sync_epoch_.load() == epoch) {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (state_sync_epoch_.load() == epoch) {
      state_checksum_ = checksum;
    }
  }
  const int64_t reconcile_ts_s = static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  std::vector<commonpb::ReplicaInfo> inventory_proto;
  inventory_proto.reserve(inventory.size());
  for (const auto& entry : inventory) {
    if (entry.memory_location != common::memory::MemoryLocation::GPU &&
        entry.memory_location != common::memory::MemoryLocation::CPU) {
      continue;
    }
    commonpb::ReplicaInfo rep;
    rep.mutable_ref()->set_artifact_id(entry.key.artifact_id);
    rep.mutable_ref()->set_replica_id("");
    auto* mi = rep.mutable_memory_info();
    mi->set_node_id(node_id_);
    mi->set_node_address(node_address);
    mi->set_node_port(opts_.p2p_port);
    mi->set_memory_size(entry.size_bytes);
    if (entry.memory_location == common::memory::MemoryLocation::GPU) {
      mi->set_memory_type(commonpb::MEMORY_TYPE_GPU);
      mi->set_device_id(entry.key.device.ordinal);
    } else if (entry.memory_location == common::memory::MemoryLocation::CPU) {
      mi->set_memory_type(commonpb::MEMORY_TYPE_RAM);
      mi->set_device_id(0);
    }
    auto* byte_space = mi->mutable_byte_space();
    if (entry.key.view_id.has_value() && !entry.key.view_id->empty()) {
      byte_space->set_kind(commonpb::BYTE_SPACE_KIND_VIEW);
      byte_space->set_id(*entry.key.view_id);
    } else {
      byte_space->set_kind(commonpb::BYTE_SPACE_KIND_CANONICAL);
    }
    auto* transport = mi->mutable_transport();
    auto export_state = entry.export_state;
    bool keys_valid = true;
    if (!entry.remote_memory_keys.empty() || !entry.buffer_sizes.empty()) {
      if (entry.remote_memory_keys.size() != entry.buffer_sizes.size()) {
        keys_valid = false;
        LOG(WARNING) << "State sync clearing transport metadata for " << entry.key.artifact_id
                     << " due to mismatched sizes (keys=" << entry.remote_memory_keys.size()
                     << " buffers=" << entry.buffer_sizes.size() << ")";
      } else {
        uint64_t total = 0;
        for (const auto size : entry.buffer_sizes) {
          if (size == 0) {
            keys_valid = false;
            break;
          }
          total += size;
        }
        if (!keys_valid || (entry.size_bytes > 0 && total != entry.size_bytes)) {
          keys_valid = false;
          LOG(WARNING) << "State sync clearing transport metadata for " << entry.key.artifact_id
                       << " due to invalid size sum (sum=" << total << " expected=" << entry.size_bytes << ")";
        }
      }
    }

    if (!keys_valid) {
      export_state = store::runtime::ReplicaExportState::kPresenceOnly;
    }

    switch (export_state) {
      case store::runtime::ReplicaExportState::kExportable:
        transport->set_export_state(commonpb::ReplicaTransportMetadata::EXPORT_STATE_EXPORTABLE);
        break;
      case store::runtime::ReplicaExportState::kDraining:
        transport->set_export_state(commonpb::ReplicaTransportMetadata::EXPORT_STATE_DRAINING);
        break;
      case store::runtime::ReplicaExportState::kPresenceOnly:
      default:
        transport->set_export_state(commonpb::ReplicaTransportMetadata::EXPORT_STATE_PRESENCE_ONLY);
        break;
    }
    transport->set_export_generation(entry.export_generation);
    const bool include_keys = keys_valid && !entry.remote_memory_keys.empty() &&
        export_state != store::runtime::ReplicaExportState::kPresenceOnly;
    if (include_keys) {
      for (const auto& key : entry.remote_memory_keys) {
        transport->add_remote_memory_keys(key);
      }
      for (const auto size : entry.buffer_sizes) {
        transport->add_buffer_sizes(size);
      }
    }
    if (!entry.verification_json.empty()) {
      transport->set_verification_json(entry.verification_json);
    }
    rep.mutable_stats()->set_max_concurrency(std::max<uint32_t>(1, opts_.max_concurrency));
    // Reconcile current_requests with active PID refs tracked by the service
    rep.mutable_stats()->set_current_requests(static_cast<uint32_t>(ports_.retire_gates.ref_count_for(entry.key)));
    rep.mutable_stats()->set_is_available(entry.is_available);
    auto* registered_ts = rep.mutable_stats()->mutable_registered_ts();
    registered_ts->set_seconds(reconcile_ts_s);
    registered_ts->set_nanos(0);
    inventory_proto.push_back(std::move(rep));
  }

  absl::StatusOr<store::components::StateSyncResult> sync_or;
  const auto sync_token = next_state_sync_token();
  {
    std::scoped_lock lock(worker_control_plane_rpc_mu_);
    sync_or = global_store_->reconcile_worker_state(
        worker_id,
        daemon_id_,
        inventory_proto,
        /*snapshot_request=*/true,
        sync_token,
        build_rpc_options(opts_.state_sync_rpc_timeout_ms, opts_.state_sync_rpc_max_retries));
  }
  if (stop_.load() || state_sync_epoch_.load() != epoch) {
    return;
  }
  if (!sync_or.ok()) {
    const absl::Status sync_status = sync_or.status();
    const StateSyncFailureClass failure_class = classify_state_sync_failure(sync_status);
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (failure_class == StateSyncFailureClass::kConnectivity) {
      bool expected_outage = false;
      if (state_sync_outage_mode_active_.compare_exchange_strong(expected_outage, true)) {
        state_sync_outage_enter_ns_.store(now_ns);
        if (auto* gauge = outage_mode_active_counter()) {
          gauge->Add(1);
        }
      }
    }
    const uint32_t consecutive_failures = state_sync_consecutive_failures_.fetch_add(1) + 1;
    const auto retry_delay = state_sync_failure_backoff(failure_class, consecutive_failures);
    const int64_t retry_not_before_ns =
        now_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(retry_delay).count();
    state_sync_next_retry_ns_.store(retry_not_before_ns);
    LOG(WARNING) << "ReconcileWorkerState returned: " << sync_status
                 << " class=" << state_sync_failure_class_to_cstr(failure_class)
                 << " consecutive_failures=" << consecutive_failures
                 << " next_retry_backoff_ms=" << retry_delay.count();
    sync_failure_.fetch_add(1);
    if (auto* counter = sync_failure_counter()) {
      counter->Add(1);
    }
    request_state_sync();
    return;
  }
  if (state_sync_outage_mode_active_.exchange(false)) {
    if (auto* gauge = outage_mode_active_counter()) {
      gauge->Add(-1);
    }
    const int64_t outage_enter_ns = state_sync_outage_enter_ns_.exchange(0);
    if (outage_enter_ns > 0) {
      const int64_t now_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
              .count();
      const int64_t latency_ns = std::max<int64_t>(0, now_ns - outage_enter_ns);
      const int64_t latency_ms = latency_ns / 1000000;
      last_reconnect_latency_ms_.store(latency_ms);
      if (auto* hist = reconnect_latency_histogram_ms()) {
        hist->Record(static_cast<double>(latency_ms), opentelemetry::context::Context{});
      }
    }
  }
  state_sync_consecutive_failures_.store(0);
  state_sync_next_retry_ns_.store(0);
  mark_state_sync_progress();

  auto record_sync_success = [this, epoch, &sync_or](int64_t last_sync_success) {
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (state_sync_epoch_.load() == epoch) {
        state_version_ = sync_or->new_state_version;
        state_checksum_ = sync_or->new_state_checksum;
        last_sync_success_ts_ = last_sync_success;
      }
    }
    sync_success_.fetch_add(1);
    sync_success_cv_.notify_all();
    if (auto* counter = sync_success_counter()) {
      counter->Add(1);
    }
    last_sync_ts_s_.store(last_sync_success);
  };

  auto ack_state_sync_token = [this, sync_token]() {
    if (sync_token.request_seq == 0) {
      return;
    }
    const uint64_t expected_previous = sync_token.request_seq - 1;
    uint64_t observed_previous = expected_previous;
    if (!state_sync_request_id_.compare_exchange_strong(observed_previous, sync_token.request_seq)) {
      VLOG(1) << "Skipping reconcile request_seq ack due to concurrent cursor update: expected_prev="
              << expected_previous << " observed_prev=" << observed_previous << " ack_seq=" << sync_token.request_seq;
    }
  };

  auto apply_state_changes = [this, &inventory, &sync_or]() {
    // Apply server-suggested state changes.
    std::vector<store::loading::ReplicaKey> retire_keys;
    retire_keys.reserve(sync_or->state_changes.size());
    absl::flat_hash_set<store::loading::ReplicaKey, store::loading::ReplicaKeyHash> remove_seen_keys;
    const auto increment_missing_count = [this](const store::loading::ReplicaKey& key) -> uint32_t {
      std::lock_guard<std::mutex> lock(pending_missing_mu_);
      return ++pending_missing_reconcile_counts_[key];
    };
    const auto clear_unseen_missing_counts = [this, &remove_seen_keys]() {
      std::lock_guard<std::mutex> lock(pending_missing_mu_);
      for (auto it = pending_missing_reconcile_counts_.begin(); it != pending_missing_reconcile_counts_.end();) {
        if (remove_seen_keys.contains(it->first)) {
          ++it;
          continue;
        }
        pending_missing_reconcile_counts_.erase(it++);
      }
    };

    auto engine_for_unload = engine_.get();
    for (const auto& ch : sync_or->state_changes) {
      switch (ch.type()) {
        case global_store::StateChange::CHANGE_TYPE_REMOVE_REPLICA: {
          const auto& ri = ch.replica_info();
          auto selector_opt = replica_selector_from_memory_info(ri.ref().artifact_id(), ri.memory_info());
          if (selector_opt.has_value()) {
            std::vector<store::loading::ReplicaKey> matched_keys;
            append_replica_keys_for_selector(*engine_for_unload, *selector_opt, &matched_keys);
            if (!ri.ref().replica_id().empty()) {
              for (const auto& key : matched_keys) {
                engine_->set_replica_global_store_id(key, ri.ref().replica_id());
              }
            }
            for (const auto& key : matched_keys) {
              remove_seen_keys.insert(key);
              const uint32_t missing_count = increment_missing_count(key);
              if (missing_count < kReconcileRemoveRetireMissThreshold) {
                const auto publish_state = engine_->get_replica_publish_state(key);
                VLOG(1) << "reconcile_worker_state: defer retire for " << key << " after " << missing_count
                        << " missing snapshot(s)"
                        << " publish_state=" << static_cast<int>(publish_state)
                        << " has_replica_id=" << (!ri.ref().replica_id().empty() ? 1 : 0);
                continue;
              }
              if (missing_count == kReconcileRemoveRetireMissThreshold) {
                LOG(WARNING) << "reconcile_worker_state: scheduling retire after " << missing_count
                             << " consecutive missing snapshots: " << key;
              }
              retire_keys.push_back(key);
            }
          } else {
            VLOG(1) << "Skipping REMOVE for non-resident memory type: artifact_id=" << ri.ref().artifact_id()
                    << " memory_type=" << ri.memory_info().memory_type();
          }
          break;
        }
        case global_store::StateChange::CHANGE_TYPE_ADD_REPLICA: {
          // Proactively materialize the replica locally on the indicated memory.
          const auto& ri = ch.replica_info();
          store::DeviceKey dev{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
          if (ri.memory_info().memory_type() == commonpb::MEMORY_TYPE_GPU) {
            dev = store::DeviceRegistry::instance().gpu_key(static_cast<int>(ri.memory_info().device_id()));
          } else if (ri.memory_info().memory_type() == commonpb::MEMORY_TYPE_RAM) {
            dev = store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
          } else {
            // Ignore DISK-only add in daemon prefetch.
            break;
          }
          std::string artifact_id = ri.ref().artifact_id();
          auto engine = engine_.get();
          auto& async_runtime = ports_.async_runtime;
          if (async_runtime.is_shutting_down()) {
            break;
          }
          auto executor = async_runtime.blocking_executor();
          executor->add([engine = std::move(engine), dev, artifact_id = std::move(artifact_id)]() mutable {
            store::loading::MaterializeHints hints;
            hints.artifact_id = artifact_id;
            auto res = engine->materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, hints);
            if (!res.ok()) {
              VLOG(1) << "Prefetch materialize_replica failed: artifact_id=" << artifact_id
                      << " dev=" << dev.to_string() << ": " << res.status();
            }
          });
          break;
        }
        case global_store::StateChange::CHANGE_TYPE_UPDATE_REPLICA: {
          // Reconcile availability (enable/disable remote access) if applicable.
          const auto& ri = ch.replica_info();
          const auto artifact_id = ri.ref().artifact_id();
          for (const auto& li : engine_->get_all_replicas_info()) {
            if (li.artifact_id != artifact_id) {
              continue;
            }
            if (li.gpu_state == common::memory::MemoryLocation::NONE) {
              continue;
            }
            if (li.key.device.type != DeviceType::GPU) {
              continue;
            }
            if (!ri.stats().is_available() && li.is_registered_for_comm) {
              auto st = engine_->disable_remote_replica_access(li.key, common::memory::MemoryLocation::GPU);
              if (!st.ok()) {
                LOG(WARNING) << "disable_remote_replica_access failed: key=" << li.key << ": " << st;
                try {
                  static auto meter =
                      opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
                  static auto ctr = meter->CreateDoubleCounter("tc_remote_access_toggle_failed_total");
                  ctr->Add(1.0);
                } catch (...) {
                }
              }
            } else if (ri.stats().is_available() && !li.is_registered_for_comm) {
              auto info_or = engine_->enable_remote_replica_access(li.key, common::memory::MemoryLocation::GPU);
              if (!info_or.ok()) {
                LOG(WARNING) << "enable_remote_replica_access failed: key=" << li.key << ": " << info_or.status();
                try {
                  static auto meter =
                      opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
                  static auto ctr = meter->CreateDoubleCounter("tc_remote_access_toggle_failed_total");
                  ctr->Add(1.0);
                } catch (...) {
                }
              }
            }
          }
          break;
        }
        default:
          break;
      }
    }
    clear_unseen_missing_counts();
    if (!retire_keys.empty()) {
      enqueue_retire_keys(std::move(retire_keys), "reconcile_worker_state");
    }

    for (const auto& entry : inventory) {
      if (entry.publish_state == store::StoreEngine::ReplicaPublishState::kPublishPending) {
        engine_->set_replica_publish_state(entry.key, store::StoreEngine::ReplicaPublishState::kPublished);
      }
    }
  };

  switch (sync_or->result_kind) {
    case store::components::ReconcileResultKind::kApplied:
    case store::components::ReconcileResultKind::kNoop: {
      ack_state_sync_token();
      record_sync_success(reconcile_ts_s);
      if (stop_.load() || state_sync_epoch_.load() != epoch) {
        return;
      }
      apply_state_changes();
      return;
    }
    case store::components::ReconcileResultKind::kIgnoredStale:
      VLOG(1) << "ReconcileWorkerState returned IGNORED_STALE for worker_id=" << worker_id;
      ack_state_sync_token();
      record_sync_success(reconcile_ts_s);
      return;
    case store::components::ReconcileResultKind::kRebaseRequired:
      VLOG(1) << "ReconcileWorkerState requested REBASE for worker_id=" << worker_id
              << " expected_replicas=" << sync_or->expected_replicas.size();
      ack_state_sync_token();
      record_sync_success(reconcile_ts_s);
      if (stop_.load() || state_sync_epoch_.load() != epoch) {
        return;
      }
      apply_full_state(sync_or->expected_replicas);
      return;
    case store::components::ReconcileResultKind::kRetryLater: {
      const auto retry_after_ms = sync_or->retry_after_ms > 0 ? sync_or->retry_after_ms : 100;
      VLOG(1) << "ReconcileWorkerState asked to retry later for worker_id=" << worker_id
              << " retry_after_ms=" << retry_after_ms;
      if (wait_for_stop(std::chrono::milliseconds(retry_after_ms))) {
        return;
      }
      if (stop_.load() || state_sync_epoch_.load() != epoch) {
        return;
      }
      request_state_sync();
      return;
    }
    case store::components::ReconcileResultKind::kFatal:
    case store::components::ReconcileResultKind::kUnspecified:
    default:
      LOG(WARNING) << "ReconcileWorkerState returned terminal result for worker_id=" << worker_id
                   << " result_kind=" << static_cast<int>(sync_or->result_kind);
      sync_failure_.fetch_add(1);
      if (auto* counter = sync_failure_counter()) {
        counter->Add(1);
      }
      return;
  }
}

absl::Status WorkerLifecycleManager::reregister_worker(bool preserve_identity) {
  auto node_addr_or = resolve_advertised_address(opts_);
  if (!node_addr_or.ok()) {
    return node_addr_or.status();
  }
  const std::string node_addr = node_addr_or->host;
  const uint32_t grpc_port = port_from_listen(opts_.listen_addr);
  std::string previous_worker_id;
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    node_address_ = node_addr;
    if (preserve_identity) {
      previous_worker_id = worker_id_;
    }
  }
  const bool recovery = preserve_identity && !previous_worker_id.empty();
  LOG(INFO) << "Resolved advertised address for Global Store re-registration: " << node_addr
            << " (source=" << advertised_source_to_cstr(node_addr_or->source) << ")";
  absl::StatusOr<store::components::WorkerRegistrationInfo> reg_or;
  {
    std::lock_guard<std::mutex> lock(worker_control_plane_rpc_mu_);
    reg_or = global_store_->register_worker(
        node_id_,
        node_addr,
        grpc_port,
        opts_.p2p_port,
        engine_->get_mem_pool_size(),
        engine_->get_available_memory(),
        /*is_recovery_registration=*/recovery,
        /*previous_worker_id=*/recovery ? std::string_view(previous_worker_id) : std::string_view{},
        /*daemon_id=*/daemon_id_,
        /*capability_flags=*/opts_.capability_flags);
  }
  if (!reg_or.ok())
    return reg_or.status();
  const std::string& new_worker_id = reg_or->worker_id;
  if (recovery && new_worker_id != previous_worker_id) {
    LOG(INFO) << "Worker identity changed after recovery: old=" << previous_worker_id << " new=" << new_worker_id;
  }
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    worker_id_ = new_worker_id;
    state_version_ = reg_or->expected_state_version;
    reconcile_generation_.store(std::max<uint64_t>(1, reg_or->reconcile_generation));
    state_checksum_.clear();
    last_sync_success_ts_ = 0;
  }
  state_sync_request_id_.store(0);
  state_sync_enqueue_suppressed_.store(0);
  state_sync_consecutive_failures_.store(0);
  state_sync_next_retry_ns_.store(0);
  if (state_sync_outage_mode_active_.exchange(false)) {
    if (auto* gauge = outage_mode_active_counter()) {
      gauge->Add(-1);
    }
  }
  state_sync_outage_enter_ns_.store(0);
  last_reconnect_latency_ms_.store(0);
  ports_.identity_store.set_registered(new_worker_id, node_id_);
  engine_->set_worker_identity(new_worker_id, node_id_, node_addr, grpc_port, opts_.p2p_port);
  ports_.worker_directory_cache.update_local_entry(
      WorkerDirectoryCache::Entry{
          .daemon_id = daemon_id_,
          .worker_id = new_worker_id,
          .node_id = node_id_,
          .node_address = node_addr,
          .grpc_port = grpc_port,
          .p2p_port = opts_.p2p_port,
          .address = absl::StrCat(node_addr, ":", grpc_port),
          .capability_flags = opts_.capability_flags,
      });
  // Bootstrap reconcile once after re-registration.
  const uint64_t baseline = sync_success_.load();
  request_state_sync();
  const auto timeout = state_sync_stall_budget().value_or(std::chrono::seconds(10));
  if (!wait_for_state_sync_success(baseline, timeout)) {
    return absl::DeadlineExceededError("reconcile bootstrap timed out after worker re-registration");
  }
  return absl::OkStatus();
}

void WorkerLifecycleManager::reconcile_memory_tier_leases_once() {
  if (!memory_tier_enabled_) {
    return;
  }
  auto leases_or = global_store_->list_memory_tier_leases(node_id_);
  if (!leases_or.ok()) {
    VLOG(1) << "ListOutstandingLeases failed: " << leases_or.status();
    return;
  }
  const uint64_t now_ns = static_cast<uint64_t>(absl::ToUnixNanos(absl::Now()));
  for (auto lease : *leases_or) {
    if (lease.kind != store::components::MemoryTierLeaseKind::kStable) {
      VLOG(1) << "Skipping non-stable memory tier lease id=" << lease.lease_id;
      continue;
    }
    if (lease.artifact_id.empty()) {
      LOG(WARNING) << "Memory tier lease " << lease.lease_id << " missing artifact_id";
      continue;
    }
    lease.node_id = lease.node_id.empty() ? node_id_ : lease.node_id;

    // Active leases are reconciled idempotently but only ACKed when they were pending.
    const bool should_ack_acquired =
        lease.state == store::components::MemoryTierLeaseState::kPending || lease.ack_epoch_ns == 0;

    if (lease.state == store::components::MemoryTierLeaseState::kRevoking) {
      auto released_or = engine_->release_memory_tier_lease(lease);
      if (!released_or.ok()) {
        LOG(WARNING) << "Failed to release stable lease for " << lease.artifact_id << ": " << released_or.status();
        continue;
      }
      store::components::MemoryTierLeaseAckPayload ack;
      ack.lease_id = lease.lease_id;
      ack.node_id = node_id_;
      ack.action = store::components::MemoryTierAckAction::kReleased;
      ack.artifact_id = lease.artifact_id;
      ack.chunk_ids = released_or->chunk_ids;
      ack.chunk_start =
          lease.chunk_start != 0 ? lease.chunk_start : (ack.chunk_ids.empty() ? 0 : ack.chunk_ids.front());
      ack.chunk_count = lease.chunk_count != 0 ? lease.chunk_count : static_cast<uint32_t>(ack.chunk_ids.size());
      ack.ledger_version = released_or->ledger_version;
      ack.bytes = released_or->bytes;
      ack.request_id = lease.request_id.empty() ? absl::StrFormat("daemon-ack-%s", lease.lease_id) : lease.request_id;
      ack.ack_epoch_ns = now_ns;
      auto ack_or = global_store_->acknowledge_memory_tier_lease(ack);
      if (!ack_or.ok()) {
        LOG(WARNING) << "AcknowledgeMemoryTierLease(released) failed for lease_id=" << lease.lease_id << ": "
                     << ack_or.status();
      }
      continue;
    }

    if (lease.state == store::components::MemoryTierLeaseState::kPending ||
        lease.state == store::components::MemoryTierLeaseState::kActive) {
      auto acquired_or = engine_->acquire_memory_tier_lease(lease);
      if (!acquired_or.ok()) {
        LOG(WARNING) << "Failed to acquire stable lease for " << lease.artifact_id << ": " << acquired_or.status();
        continue;
      }
      if (!should_ack_acquired) {
        continue;
      }
      store::components::MemoryTierLeaseAckPayload ack;
      ack.lease_id = lease.lease_id;
      ack.node_id = node_id_;
      ack.action = store::components::MemoryTierAckAction::kAcquired;
      ack.artifact_id = lease.artifact_id;
      ack.chunk_ids = acquired_or->chunk_ids;
      ack.chunk_start =
          lease.chunk_start != 0 ? lease.chunk_start : (ack.chunk_ids.empty() ? 0 : ack.chunk_ids.front());
      ack.chunk_count = lease.chunk_count != 0 ? lease.chunk_count : static_cast<uint32_t>(ack.chunk_ids.size());
      ack.ledger_version = acquired_or->ledger_version;
      ack.bytes = acquired_or->bytes;
      ack.request_id = lease.request_id.empty() ? absl::StrFormat("daemon-ack-%s", lease.lease_id) : lease.request_id;
      ack.ack_epoch_ns = now_ns;
      auto ack_or = global_store_->acknowledge_memory_tier_lease(ack);
      if (!ack_or.ok()) {
        LOG(WARNING) << "AcknowledgeMemoryTierLease(acquired) failed for lease_id=" << lease.lease_id << ": "
                     << ack_or.status();
      }
    }
  }
}

void WorkerLifecycleManager::chunk_sync_loop() {
  sync_alive_.store(true);
  const auto interval = std::chrono::milliseconds(opts_.chunk_sync_interval_ms);
  try {
    while (!stop_.load()) {
      std::vector<store::components::ChunkStateUpdate> updates;
      absl::flat_hash_set<store::loading::ReplicaKey, store::loading::ReplicaKeyHash> seen_keys;
      for (const auto& info : engine_->get_all_replicas_info()) {
        if (info.gpu_state == common::memory::MemoryLocation::NONE)
          continue;
        // Use UMA-backed per-device states to reflect actual GPU residency
        auto states = engine_->get_chunk_states_for_device(info.artifact_id, info.gpu_device_id);
        seen_keys.insert(info.key);
        std::vector<int32_t> state_codes;
        state_codes.reserve(states.size());
        for (const auto state : states) {
          state_codes.push_back(static_cast<int32_t>(state));
        }
        bool should_publish = false;
        {
          std::lock_guard<std::mutex> lock(chunk_sync_state_mu_);
          auto it = chunk_sync_last_states_.find(info.key);
          if (it == chunk_sync_last_states_.end() || it->second != state_codes) {
            chunk_sync_last_states_[info.key] = std::move(state_codes);
            should_publish = true;
          }
        }
        if (!should_publish) {
          continue;
        }
        updates.reserve(updates.size() + states.size());
        for (size_t i = 0; i < states.size(); ++i) {
          store::components::ChunkStateUpdate u;
          u.artifact_id = info.artifact_id;
          u.chunk_idx = static_cast<uint32_t>(i);
          u.state = states[i];
          u.device_uuid = info.key.device.uuid;
          u.replica = 0;
          updates.push_back(std::move(u));
        }
      }
      {
        std::lock_guard<std::mutex> lock(chunk_sync_state_mu_);
        for (auto it = chunk_sync_last_states_.begin(); it != chunk_sync_last_states_.end();) {
          if (seen_keys.contains(it->first)) {
            ++it;
            continue;
          }
          auto erase_it = it;
          ++it;
          chunk_sync_last_states_.erase(erase_it);
        }
      }
      if (!updates.empty()) {
        std::string worker_id;
        {
          std::lock_guard<std::mutex> lock(state_mu_);
          worker_id = worker_id_;
        }
        auto st = global_store_->batch_update_chunk_states(worker_id, node_id_, updates);
        if (!st.ok()) {
          LOG(WARNING) << "batch_update_chunk_states failed: " << st;
          try {
            static auto meter =
                opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
            static auto ctr = meter->CreateDoubleCounter("tc_chunk_states_batch_update_failed_total");
            ctr->Add(1.0);
          } catch (...) {
          }
        }
      }
      if (wait_for_stop(interval)) {
        break;
      }
      sync_ticks_.fetch_add(1);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "chunk_sync_loop crashed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "chunk_sync_loop crashed: unknown exception";
  }
  sync_alive_.store(false);
}

void WorkerLifecycleManager::monitor_loop() {
  // Simple liveness/restart loop for lifecycle threads
  using namespace std::chrono_literals;
  constexpr int64_t kDefaultGlobalStoreRpcTimeoutMs = 60000;
  constexpr auto kHeartbeatRestartGrace = std::chrono::seconds(5);
  uint64_t last_hb_ticks = 0;
  uint64_t last_sync_ticks = 0;
  auto last_hb_change = std::chrono::steady_clock::now();
  auto last_sync_change = std::chrono::steady_clock::now();
  const auto state_sync_budget = state_sync_stall_budget();
  const int64_t heartbeat_rpc_timeout_ms =
      opts_.heartbeat_rpc_timeout_ms > 0 ? opts_.heartbeat_rpc_timeout_ms : kDefaultGlobalStoreRpcTimeoutMs;
  const auto heartbeat_stall_budget = std::max(
      std::chrono::milliseconds(opts_.heartbeat_interval_ms * 3),
      std::chrono::milliseconds(heartbeat_rpc_timeout_ms + opts_.heartbeat_interval_ms) + kHeartbeatRestartGrace);
  while (!stop_.load()) {
    // Detect stalled heartbeat: ticks not increasing for > 3 * heartbeat_interval
    if (hb_ticks_.load() != last_hb_ticks) {
      last_hb_ticks = hb_ticks_.load();
      last_hb_change = std::chrono::steady_clock::now();
    } else if (std::chrono::steady_clock::now() - last_hb_change > heartbeat_stall_budget) {
      LOG(WARNING) << "heartbeat_loop appears stalled; restarting";
      hb_alive_.store(false);
    }
    if (!hb_alive_.load() && !stop_.load()) {
      hb_epoch_.fetch_add(1);
      retire_thread(&hb_thread_);
      hb_restarts_.fetch_add(1);
      hb_thread_ = std::thread(&WorkerLifecycleManager::heartbeat_loop, this);
      // Give the restarted loop a fresh grace window so monitor does not
      // repeatedly trigger before the new thread can complete one heartbeat.
      last_hb_ticks = hb_ticks_.load();
      last_hb_change = std::chrono::steady_clock::now();
    }
    if (state_sync_budget.has_value() && state_sync_inflight_.load()) {
      const int64_t last_progress_ns = state_sync_last_progress_ns_.load();
      if (last_progress_ns > 0) {
        const auto now_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count();
        const auto elapsed = std::chrono::nanoseconds(now_ns - last_progress_ns);
        if (elapsed > *state_sync_budget && !state_sync_restart_pending_.exchange(true)) {
          const uint64_t new_epoch = state_sync_epoch_.fetch_add(1) + 1;
          state_sync_cv_.notify_all();
          LOG(WARNING) << "state_sync_loop exceeded budget; requesting cancel epoch=" << new_epoch
                       << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        }
      }
    }
    if (!state_sync_alive_.load() && !stop_.load()) {
      uint64_t epoch = state_sync_epoch_.load();
      if (!state_sync_restart_pending_.load()) {
        epoch = state_sync_epoch_.fetch_add(1) + 1;
      }
      state_sync_restart_pending_.store(false);
      state_sync_cv_.notify_all();
      retire_thread(&state_sync_thread_);
      state_sync_thread_ = std::thread(&WorkerLifecycleManager::state_sync_loop, this, epoch);
    }
    // Detect stalled sync: ticks not increasing for > 3 * chunk_sync_interval
    if (opts_.chunk_sync_interval_ms > 0) {
      if (sync_ticks_.load() != last_sync_ticks) {
        last_sync_ticks = sync_ticks_.load();
        last_sync_change = std::chrono::steady_clock::now();
      } else if (
          std::chrono::steady_clock::now() - last_sync_change >
          std::chrono::milliseconds(opts_.chunk_sync_interval_ms * 3)) {
        LOG(WARNING) << "chunk_sync_loop appears stalled; restarting";
        sync_alive_.store(false);
      }
    }
    if (opts_.chunk_sync_interval_ms > 0 && !sync_alive_.load() && !stop_.load()) {
      retire_thread(&sync_thread_);
      sync_restarts_.fetch_add(1);
      sync_thread_ = std::thread(&WorkerLifecycleManager::chunk_sync_loop, this);
    }
    if (wait_for_stop(5s)) {
      break;
    }
  }
}

void WorkerLifecycleManager::apply_obsolete_replicas(const std::vector<std::string>& artifact_ids) {
  if (artifact_ids.empty()) {
    return;
  }
  LOG(INFO) << "Heartbeat reported " << artifact_ids.size()
            << " obsolete artifacts; requesting state sync for confirmation.";
  request_state_sync();
}

void WorkerLifecycleManager::apply_full_state(const std::vector<commonpb::ReplicaInfo>& expected) {
  absl::flat_hash_set<ReplicaSelector, ReplicaSelectorHash> expected_keys;
  expected_keys.reserve(expected.size());
  for (const auto& r : expected) {
    auto selector_opt = replica_selector_from_memory_info(r.ref().artifact_id(), r.memory_info());
    if (selector_opt.has_value()) {
      expected_keys.insert(std::move(*selector_opt));
    }
  }

  absl::flat_hash_set<ReplicaSelector, ReplicaSelectorHash> obsolete_selectors;
  std::vector<store::loading::ReplicaKey> retire_keys;

  const auto clear_pending_missing_count = [this](const store::loading::ReplicaKey& key) {
    std::lock_guard<std::mutex> lock(pending_missing_mu_);
    pending_missing_reconcile_counts_.erase(key);
  };
  const auto increment_pending_missing_count = [this](const store::loading::ReplicaKey& key) -> uint32_t {
    std::lock_guard<std::mutex> lock(pending_missing_mu_);
    return ++pending_missing_reconcile_counts_[key];
  };

  for (const auto& entry : engine_->get_ha_inventory()) {
    std::vector<ReplicaSelector> local_selectors;
    append_local_replica_selectors(entry, &local_selectors);
    if (local_selectors.empty()) {
      clear_pending_missing_count(entry.key);
      continue;
    }
    std::vector<ReplicaSelector> missing_selectors;
    for (const auto& selector : local_selectors) {
      if (expected_keys.find(selector) != expected_keys.end()) {
        continue;
      }
      missing_selectors.push_back(selector);
    }
    if (missing_selectors.empty()) {
      clear_pending_missing_count(entry.key);
      continue;
    }
    const uint32_t missing_count = increment_pending_missing_count(entry.key);
    if (missing_count < kReconcileRemoveRetireMissThreshold) {
      VLOG(1) << "apply_full_state: defer retire after " << missing_count << " missing snapshot(s): " << entry.key;
      continue;
    }
    if (missing_count == kReconcileRemoveRetireMissThreshold) {
      LOG(WARNING) << "apply_full_state: replica still missing after " << missing_count
                   << " consecutive snapshots; scheduling retire: " << entry.key;
    }
    for (const auto& selector : missing_selectors) {
      obsolete_selectors.insert(selector);
    }
  }
  auto engine = engine_.get();
  for (const auto& selector : obsolete_selectors) {
    append_replica_keys_for_selector(*engine, selector, &retire_keys);
  }
  if (!retire_keys.empty()) {
    enqueue_retire_keys(std::move(retire_keys), "apply_full_state");
  }
}

void WorkerLifecycleManager::enqueue_retire_keys(
    std::vector<store::loading::ReplicaKey> keys,
    std::string_view context) {
  if (keys.empty()) {
    return;
  }
  absl::flat_hash_set<store::loading::ReplicaKey, store::loading::ReplicaKeyHash> unique;
  unique.reserve(keys.size());
  for (const auto& key : keys) {
    unique.insert(key);
  }

  std::vector<store::loading::ReplicaKey> newly_added;
  {
    std::lock_guard<std::mutex> lock(retire_mu_);
    for (const auto& key : unique) {
      if (retire_queue_.find(key) != retire_queue_.end()) {
        continue;
      }
      retire_queue_.emplace(key, RetireEntry{});
      newly_added.push_back(key);
    }
  }

  if (newly_added.empty()) {
    return;
  }

  for (const auto& key : newly_added) {
    engine_->set_replica_publish_state(key, store::StoreEngine::ReplicaPublishState::kRetiring);
  }
  retire_pending_.store(true);
  state_sync_cv_.notify_all();
  VLOG(1) << context << ": queued retire for " << newly_added.size() << " replicas";
}

void WorkerLifecycleManager::schedule_demotion_task(const store::loading::ReplicaKey& key, std::string_view context) {
  auto* promotion_manager = engine_->promotion_manager();
  if (promotion_manager == nullptr) {
    std::lock_guard<std::mutex> lock(retire_mu_);
    auto it = retire_queue_.find(key);
    if (it != retire_queue_.end()) {
      it->second.demotion_complete = true;
    }
    return;
  }

  auto& async_runtime = ports_.async_runtime;
  if (async_runtime.is_shutting_down()) {
    std::lock_guard<std::mutex> lock(retire_mu_);
    auto it = retire_queue_.find(key);
    if (it != retire_queue_.end()) {
      it->second.demotion_complete = true;
    }
    return;
  }

  auto executor = async_runtime.blocking_executor();
  executor->add([this, key, context = std::string(context)]() {
    if (stop_.load()) {
      return;
    }
    auto* promotion = engine_->promotion_manager();
    if (promotion == nullptr) {
      return;
    }

    const auto& promo_opts = engine_->options().promotion;
    const auto drain_timeout = promo_opts.demotion_drain_timeout;
    const auto deadline =
        drain_timeout.count() > 0 ? std::chrono::steady_clock::now() + drain_timeout : std::chrono::steady_clock::now();

    const uint64_t baseline = sync_success_.load();
    const auto drain_status = promotion->mark_replica_draining(key);
    if (!drain_status.ok()) {
      LOG(WARNING) << context << ": mark_replica_draining failed for " << key << ": " << drain_status;
    }
    request_state_sync();

    if (drain_timeout.count() > 0) {
      if (!wait_for_state_sync_success(baseline, drain_timeout)) {
        LOG(WARNING) << context << ": state sync barrier timed out for " << key;
      }
    }

    if (drain_timeout.count() > 0) {
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now < deadline ? std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
                                            : std::chrono::milliseconds(0);
      if (remaining.count() > 0) {
        auto replica_id = engine_->get_replica_global_store_id(key);
        if (replica_id.has_value()) {
          auto drain_or =
              global_store_->wait_replica_drain(*replica_id, static_cast<uint32_t>(remaining.count()), context);
          if (!drain_or.ok()) {
            LOG(WARNING) << context << ": WaitReplicaDrain failed for " << key << ": " << drain_or.status();
          } else if (!drain_or->drained) {
            LOG(WARNING) << context << ": WaitReplicaDrain timed out for " << key
                         << " current_requests=" << drain_or->current_requests;
          }
        } else {
          VLOG(1) << context << ": no replica_id available for WaitReplicaDrain " << key;
        }
      }
    }

    if (drain_timeout.count() > 0) {
      while (!stop_.load()) {
        RetireGateSnapshot snapshot = ports_.retire_gates.snapshot_for(key);
        if (snapshot.ready()) {
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          LOG(WARNING) << context << ": local drain timeout for " << key << " refs=" << snapshot.ref_count
                       << " uses=" << snapshot.use_count << " pins=" << snapshot.placement_pins
                       << " transport_lock=" << (snapshot.has_transport_lock ? 1 : 0);
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

    const auto finalize_status = promotion->finalize_replica_demotion(key);
    if (!finalize_status.ok()) {
      LOG(WARNING) << context << ": finalize_replica_demotion failed for " << key << ": " << finalize_status;
    }

    {
      std::lock_guard<std::mutex> lock(retire_mu_);
      auto it = retire_queue_.find(key);
      if (it != retire_queue_.end()) {
        it->second.demotion_complete = true;
      }
    }
    state_sync_cv_.notify_all();
  });
}

void WorkerLifecycleManager::process_retire_queue() {
  std::vector<store::loading::ReplicaKey> keys;
  {
    std::lock_guard<std::mutex> lock(retire_mu_);
    if (retire_queue_.empty()) {
      retire_pending_.store(false);
      return;
    }
    retire_pending_.store(false);
    keys.reserve(retire_queue_.size());
    for (const auto& entry : retire_queue_) {
      keys.push_back(entry.first);
    }
  }

  for (const auto& key : keys) {
    {
      std::lock_guard<std::mutex> lock(retire_mu_);
      auto it = retire_queue_.find(key);
      if (it == retire_queue_.end()) {
        continue;
      }
      if (!it->second.demotion_started) {
        it->second.demotion_started = true;
        schedule_demotion_task(key, "retire_queue");
        continue;
      }
      if (!it->second.demotion_complete) {
        continue;
      }
    }
    RetireGateSnapshot snapshot = ports_.retire_gates.snapshot_for(key);

    if (!snapshot.ready()) {
      bool should_log = false;
      const int64_t now_s = absl::ToUnixSeconds(absl::Now());
      {
        std::lock_guard<std::mutex> lock(retire_mu_);
        auto it = retire_queue_.find(key);
        if (it == retire_queue_.end()) {
          continue;
        }
        it->second.attempts += 1;
        if (now_s - it->second.last_log_ts_s >= 30) {
          it->second.last_log_ts_s = now_s;
          should_log = true;
        }
      }
      if (should_log) {
        LOG(WARNING) << "Retire blocked for " << key << " refs=" << snapshot.ref_count << " uses=" << snapshot.use_count
                     << " pins=" << snapshot.placement_pins
                     << " transport_lock=" << (snapshot.has_transport_lock ? 1 : 0);
      }
      continue;
    }

    int rc = 1;
    try {
      auto& async_runtime = ports_.async_runtime;
      if (!async_runtime.is_shutting_down()) {
        auto executor = async_runtime.blocking_executor();
        rc = folly::via(std::move(executor), [engine = engine_.get(), key]() {
               return engine->unload_replica(key);
             }).get();
      } else {
        rc = engine_->unload_replica(key);
      }
    } catch (const std::exception& ex) {
      LOG(WARNING) << "Retire unload threw exception for " << key << ": " << ex.what();
      rc = 1;
    } catch (...) {
      LOG(WARNING) << "Retire unload threw unknown exception for " << key;
      rc = 1;
    }

    bool retired = (rc == 0 || rc == -1);
    if (!retired) {
      const auto state = engine_->get_replica_state(key, key.device.type);
      if (state <= store::replica::MemoryState::UNALLOCATED) {
        retired = true;
      }
    }
    if (retired) {
      std::lock_guard<std::mutex> lock(retire_mu_);
      retire_queue_.erase(key);
      {
        std::lock_guard<std::mutex> pending_lock(pending_missing_mu_);
        pending_missing_reconcile_counts_.erase(key);
      }
    } else {
      LOG(WARNING) << "Retire unload failed rc=" << rc << " for " << key;
    }
  }

  {
    std::lock_guard<std::mutex> lock(retire_mu_);
    if (retire_queue_.empty()) {
      retire_pending_.store(false);
    }
  }
}

std::string WorkerLifecycleManager::compute_state_checksum(
    std::string_view node_id,
    std::string_view node_address,
    uint32_t node_port,
    const std::vector<store::StoreEngine::ReplicaInventoryEntry>& inventory) {
  // Keep format aligned with Global Store's RecoveryService._compute_state_checksum:
  // artifact_id:view_id:node_id:node_address:node_port:device_id:memory_type:available; sorted by
  // (artifact_id, view_id, memory_type, device_id).
  struct Entry {
    std::string artifact_id;
    std::string view_id;
    std::string memory_type;
    int device_id;
    bool available;
  };

  std::vector<Entry> entries;
  entries.reserve(inventory.size());

  for (const auto& entry : inventory) {
    std::string mem_type;
    int device_id = 0;
    if (entry.memory_location == common::memory::MemoryLocation::GPU) {
      if (entry.key.device.ordinal < 0) {
        continue;
      }
      mem_type = "GPU";
      device_id = entry.key.device.ordinal;
    } else if (entry.memory_location == common::memory::MemoryLocation::CPU) {
      mem_type = "RAM";
      device_id = 0;
    } else {
      continue;
    }
    entries.push_back(
        Entry{
            .artifact_id = entry.key.artifact_id,
            .view_id = entry.key.view_id.value_or(std::string()),
            .memory_type = std::move(mem_type),
            .device_id = device_id,
            .available = entry.is_available,
        });
  }

  std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
    if (lhs.artifact_id != rhs.artifact_id)
      return lhs.artifact_id < rhs.artifact_id;
    if (lhs.view_id != rhs.view_id)
      return lhs.view_id < rhs.view_id;
    if (lhs.memory_type != rhs.memory_type)
      return lhs.memory_type < rhs.memory_type;
    return lhs.device_id < rhs.device_id;
  });

  std::string state_str;
  for (const auto& e : entries) {
    absl::StrAppendFormat(
        &state_str,
        "%s:%s:%s:%s:%u:%d:%s:%d;",
        e.artifact_id,
        e.view_id,
        node_id,
        node_address,
        node_port,
        e.device_id,
        e.memory_type,
        e.available ? 1 : 0);
  }

  // FNV-1a 64-bit over the stable string; portable across languages.
  uint64_t hash = 14695981039346656037ull; // FNV offset basis
  constexpr uint64_t kFnvPrime = 1099511628211ull; // FNV prime
  for (char c : state_str) {
    hash ^= static_cast<uint8_t>(c);
    hash *= kFnvPrime;
  }

  return absl::StrFormat("%016x", hash);
}

} // namespace tensorcast::daemon
