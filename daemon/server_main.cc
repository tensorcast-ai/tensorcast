// Copyright (c) 2025, TensorCast Team.

#include <filesystem>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "core/common/config/daemon_config_io.h"
#include "core/common/cuda_api.h"
#include "core/common/logging_init.h"
#include "core/common/otel/init.h"
#include "core/common/trace/trace_manager.h"
#include "core/store/components/communication_manager.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/grpc_service_impl.h"
#include "daemon/worker_lifecycle_manager.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gsl/pointers"

#include <pthread.h>
#include <csignal>
#include <fstream>
#include <sstream>

ABSL_FLAG(std::string, config, "", "Path to unified daemon config (YAML/JSON)");
ABSL_FLAG(std::string, config_text, "", "Inline daemon config as YAML/JSON text (mutually exclusive with --config)");
ABSL_FLAG(bool, use_cursor_pagination, false, "Enable opaque cursor pagination for GetLoadedReplicasV2");
using namespace tensorcast;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  common::ensure_logging_initialized();
  // Avoid global using-directives per project guidelines
  // Note: config loading happens below; defer OTel/log-sink init until then.
  // Load unified config from either --config (file) or --config_text (inline)
  const std::string cfg_path = absl::GetFlag(FLAGS_config);
  const std::string cfg_text = absl::GetFlag(FLAGS_config_text);
  if (cfg_path.empty() == cfg_text.empty()) {
    LOG(ERROR) << "Exactly one of --config (path) or --config_text (inline YAML/JSON) must be provided";
    return 2;
  }
  absl::StatusOr<config::v1::DaemonConfig> cfg_or = cfg_text.empty()
      ? common::config::load_daemon_config_from_file(cfg_path)
      : common::config::load_daemon_config_from_text(cfg_text);
  if (!cfg_or.ok()) {
    LOG(ERROR) << "Failed to load config: " << cfg_or.status();
    return 2;
  }
  const auto& cfg = *cfg_or;

  // Block SIGINT/SIGTERM in this main thread BEFORE starting any threads so that
  // all subsequently created threads inherit the blocked mask. We'll handle
  // these signals via sigwait in a dedicated thread later to perform a
  // cooperative shutdown (gRPC shutdown + worker unregistration).
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &set, nullptr);

  // Configure CUDA debug toggles
  cuda::configure_same_process_ipc_fallback(cfg.debug().cuda().enable_same_process_ipc_fallback());

  LOG(INFO) << "Config: " << cfg.DebugString();

  // Map config to StoreEngineOptions
  store::StoreEngineOptions opts;
  opts.storage_path = cfg.server().storage_path();
  opts.num_thread = static_cast<int>(cfg.server().num_threads());
  opts.memory_pool_size = static_cast<size_t>(cfg.engine().mem_pool_size_bytes());
  opts.tx_slice_bytes = static_cast<size_t>(cfg.engine().tx_slice_bytes());
  opts.artifact_chunk_bytes = static_cast<size_t>(cfg.engine().artifact_chunk_bytes());
  opts.streaming_buffer_max_concurrent_sessions =
      static_cast<int>(cfg.engine().streaming_buffer_max_concurrent_sessions());
  if (cfg.engine().has_pinned_allocation_timeout()) {
    opts.pinned_memory_timeout = std::chrono::milliseconds(
        cfg.engine().pinned_allocation_timeout().seconds() * 1000 +
        cfg.engine().pinned_allocation_timeout().nanos() / 1000000);
  }
  opts.p2p_fallback_disk_dir = cfg.engine().p2p_fallback_disk_dir();
  if (cfg.engine().has_memory_tiers()) {
    store::MemoryTierConfig tiers;
    const auto& mt = cfg.engine().memory_tiers();
    tiers.enable_preemptible_memory = mt.enable_preemptible();
    tiers.stable_bytes = mt.stable_bytes();
    tiers.preemptible_limit_bytes = mt.preemptible_limit_bytes();
    tiers.preemptible_low_watermark_ratio = mt.preemptible_low_watermark_ratio();
    opts.memory_tier_config = tiers;
  }

  // Communicator setup (always create; RDMA enable is a config toggle inside engine)
  std::shared_ptr<store::components::CommunicationManager> comm_mgr;
  uint16_t p2p_port = 0;
  std::string p2p_host = cfg.server().listen().host();
  if (cfg.server().has_p2p_listen()) {
    p2p_host = cfg.server().p2p_listen().host().empty() ? p2p_host : cfg.server().p2p_listen().host();
    p2p_port = static_cast<uint16_t>(cfg.server().p2p_listen().port());
  }
  comm_mgr = std::make_shared<store::components::CommunicationManager>();
  {
    auto st = comm_mgr->initialize_with_config(p2p_host, p2p_port, cfg.communicator());
    if (!st.ok()) {
      LOG(WARNING) << "Failed to initialize communication engine: " << st.message();
    } else {
      const uint16_t actual_port = comm_mgr->listen_port();
      if (actual_port != 0) {
        p2p_port = actual_port;
      }
      opts.comm_manager = comm_mgr;
    }
  }
  opts.p2p_port = p2p_port;
  opts.p2p_listen_host = p2p_host;
  opts.enable_rdma = cfg.communicator().enable_rdma();

  if (cfg.high_availability().enabled() && p2p_port == 0) {
    LOG(ERROR) << "Global Store high availability requires server.p2p_listen.port to be non-zero;"
               << " configure a valid P2P port before starting the daemon.";
    return 2;
  }

  // Global Store HA - pick first endpoint if enabled
  std::string gs_addr;
  if (cfg.high_availability().enabled() && !cfg.high_availability().global_store_endpoints().empty()) {
    const auto& ep = cfg.high_availability().global_store_endpoints(0);
    gs_addr = absl::StrCat(ep.host(), ":", ep.port());
  }
  opts.global_store_address = gs_addr;

  // Configure logging level/VLOG and optional sinks, then initialize OTel
  common::initialize_logging_from_config(cfg.observability().logging());
  if (!common::otel::init_from_config(cfg.observability(), "store-daemon")) {
    LOG(WARNING) << "OpenTelemetry initialization failed; continuing without telemetry";
  }
  // Configure Chrome trace directory (optional)
  if (!cfg.observability().tracing().chrome_trace_dir().empty()) {
    common::trace::TraceManager::set_chrome_trace_dir(cfg.observability().tracing().chrome_trace_dir());
  }

  auto engine = std::make_shared<store::StoreEngine>(opts);

  // Map lifecycle options into service options
  daemon::StoreDaemonServiceImpl::Options svc_opts;
  if (cfg.lifecycle().has_sessions_ttl()) {
    const auto& d = cfg.lifecycle().sessions_ttl();
    svc_opts.sessions_ttl = std::chrono::seconds(d.seconds());
  }
  if (cfg.lifecycle().has_locks_ttl()) {
    const auto& d = cfg.lifecycle().locks_ttl();
    svc_opts.locks_ttl = std::chrono::seconds(d.seconds());
  }
  if (cfg.lifecycle().has_sessions_sweep_interval()) {
    const auto& d = cfg.lifecycle().sessions_sweep_interval();
    svc_opts.sessions_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_locks_sweep_interval()) {
    const auto& d = cfg.lifecycle().locks_sweep_interval();
    svc_opts.locks_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_verification_sweep_interval()) {
    const auto& d = cfg.lifecycle().verification_sweep_interval();
    svc_opts.verification_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_proc_check_interval()) {
    const auto& d = cfg.lifecycle().proc_check_interval();
    svc_opts.proc_check_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  svc_opts.enable_periodic_eviction = cfg.lifecycle().enable_periodic_eviction();
  svc_opts.gpu_memory_limit_fraction = cfg.lifecycle().gpu_memory_limit_fraction();
  if (cfg.lifecycle().has_eviction_loop_interval()) {
    const auto& d = cfg.lifecycle().eviction_loop_interval();
    svc_opts.eviction_check_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  // Observability high-cardinality attributes: default off (config hook TBD)
  svc_opts.allow_high_card_attrs = false;
  // Feature flags (override via flags for now)
  svc_opts.use_cursor_pagination = absl::GetFlag(FLAGS_use_cursor_pagination);
  for (const auto& prefix : cfg.engine().disk_path_whitelist()) {
    svc_opts.disk_path_whitelist.emplace_back(prefix);
  }

  daemon::StoreDaemonServiceImpl service(engine, svc_opts);

  // gRPC server
  const std::string listen_addr = absl::StrCat(cfg.server().listen().host(), ":", cfg.server().listen().port());
  grpc::ServerBuilder builder;
  // Do not override gRPC max receive size; defaults are sufficient for metadata-only RPCs
  if (cfg.server().grpc().max_concurrent_streams() > 0) {
    builder.AddChannelArgument("grpc.max_concurrent_streams", cfg.server().grpc().max_concurrent_streams());
  }
  std::shared_ptr<grpc::ServerCredentials> creds;
  if (cfg.server().grpc().tls().enabled()) {
    std::string cert;
    std::string key;
    std::string ca;
    auto read_all = [](const std::string& path) -> std::string {
      std::ifstream f(path);
      if (!f.is_open())
        return {};
      std::stringstream ss;
      ss << f.rdbuf();
      return ss.str();
    };
    cert = read_all(cfg.server().grpc().tls().cert_file());
    key = read_all(cfg.server().grpc().tls().key_file());
    if (!cfg.server().grpc().tls().client_ca_file().empty()) {
      ca = read_all(cfg.server().grpc().tls().client_ca_file());
    }
    grpc::SslServerCredentialsOptions ssl_opts;
    if (!ca.empty()) {
      ssl_opts.pem_root_certs = ca;
    }
    grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp{.private_key = key, .cert_chain = cert};
    ssl_opts.pem_key_cert_pairs.push_back(pkcp);
    creds = grpc::SslServerCredentials(ssl_opts);
  } else {
    creds = grpc::InsecureServerCredentials();
  }
  builder.AddListeningPort(listen_addr, creds);

  // Apply channel/server args for keepalive and connection lifetimes
  auto to_ms = [](const google::protobuf::Duration& d) -> int {
    return static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
  };
  if (cfg.server().grpc().has_keepalive_time()) {
    builder.AddChannelArgument("grpc.keepalive_time_ms", to_ms(cfg.server().grpc().keepalive_time()));
  }
  if (cfg.server().grpc().has_keepalive_timeout()) {
    builder.AddChannelArgument("grpc.keepalive_timeout_ms", to_ms(cfg.server().grpc().keepalive_timeout()));
  }
  if (cfg.server().grpc().has_max_connection_idle()) {
    builder.AddChannelArgument("grpc.max_connection_idle_ms", to_ms(cfg.server().grpc().max_connection_idle()));
  }
  if (cfg.server().grpc().has_max_connection_age()) {
    builder.AddChannelArgument("grpc.max_connection_age_ms", to_ms(cfg.server().grpc().max_connection_age()));
  }
  // tcp_nodelay / so_reuseport toggles
  builder.AddChannelArgument("grpc.tcp_nodelay", cfg.server().grpc().tcp_nodelay() ? 1 : 0);
  builder.AddChannelArgument("grpc.so_reuseport", cfg.server().grpc().so_reuseport() ? 1 : 0);
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "tensorcast-daemon listening on " << listen_addr;

  // Start worker lifecycle if configured
  std::unique_ptr<daemon::WorkerLifecycleManager> lifecycle;
  if (!gs_addr.empty() && cfg.high_availability().enabled()) {
    daemon::WorkerLifecycleManager::Options lopts;
    lopts.global_store_addr = gs_addr;
    lopts.listen_addr = listen_addr;
    // Prefer explicit advertise.host if provided
    if (cfg.server().has_advertise() && !cfg.server().advertise().host().empty()) {
      lopts.advertise_host = cfg.server().advertise().host();
    }
    lopts.p2p_port = p2p_port;
    if (cfg.high_availability().has_heartbeat_interval()) {
      const auto& d = cfg.high_availability().heartbeat_interval();
      lopts.heartbeat_interval_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_periodic_sync_interval()) {
      const auto& d = cfg.high_availability().periodic_sync_interval();
      lopts.chunk_sync_interval_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    lifecycle = std::make_unique<daemon::WorkerLifecycleManager>(
        gsl::not_null<std::shared_ptr<store::StoreEngine>>{engine},
        gsl::not_null<daemon::StoreDaemonServiceImpl*>{&service},
        lopts);
    auto st = lifecycle->start();
    if (!st.ok()) {
      LOG(WARNING) << "Worker lifecycle start failed: " << st.message();
    }
  }

  std::thread sig_thread([&server, &service, set]() mutable {
    int sig = 0;
    if (sigwait(&set, &sig) == 0) {
      LOG(INFO) << "Received signal " << sig << ", initiating gRPC shutdown...";
      service.begin_shutdown();
      server->Shutdown();
    }
  });

  server->Wait();
  if (sig_thread.joinable()) {
    sig_thread.join();
  }
  if (lifecycle) {
    lifecycle->stop();
  }

  return 0;
}
