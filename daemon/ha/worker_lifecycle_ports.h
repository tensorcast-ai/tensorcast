// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include "core/common/async_runtime.h"
#include "daemon/state/retire_gates.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/worker_identity_store.h"

namespace tensorcast::daemon {

class WorkerDirectoryCache;
class LipManager;

struct WorkerLifecyclePorts {
  WorkerIdentityStore& identity_store;
  LipManager& lip_manager;
  WorkerDirectoryCache& worker_directory_cache;
  RetireGates& retire_gates;
  ShutdownSignal& shutdown_signal;
  common::AsyncRuntime& async_runtime;
};

} // namespace tensorcast::daemon
