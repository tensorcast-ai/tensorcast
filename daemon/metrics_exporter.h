// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <string>
#include <thread>

#include "core/store/store_engine.h"

namespace tensorcast::daemon {

class MetricsExporter {
 public:
  MetricsExporter(std::shared_ptr<tensorcast::store::StoreEngine> engine, uint16_t port)
      : engine_(std::move(engine)), port_(port) {}

  void start();
  void stop();

 private:
  void run();
  std::string collect_metrics();

  std::shared_ptr<tensorcast::store::StoreEngine> engine_;
  uint16_t port_;
  std::atomic<bool> stop_{false};
  std::thread th_;
};

} // namespace tensorcast::daemon
