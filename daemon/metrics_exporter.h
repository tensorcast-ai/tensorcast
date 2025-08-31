// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

#include "core/store/store_engine.h"

namespace tensorcast::daemon {
class StoreDaemonServiceImpl;
}

namespace tensorcast::daemon {

class MetricsExporter {
 public:
  MetricsExporter(std::shared_ptr<tensorcast::store::StoreEngine> engine, uint16_t port)
      : engine_(std::move(engine)), port_(port) {}

  void start();
  void stop();
  void set_service(class StoreDaemonServiceImpl* svc) {
    service_ = svc;
  }
  // Register an optional extra metrics provider that returns Prometheus text.
  void set_extra_metrics_provider(std::function<std::string()> fn) {
    extra_metrics_provider_ = std::move(fn);
  }

 private:
  void run();
  std::string collect_metrics();

  std::shared_ptr<tensorcast::store::StoreEngine> engine_;
  uint16_t port_;
  class StoreDaemonServiceImpl* service_{nullptr};
  std::function<std::string()> extra_metrics_provider_{};
  std::atomic<bool> stop_{false};
  std::thread th_;
};

} // namespace tensorcast::daemon
