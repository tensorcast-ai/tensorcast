// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/mr_cache.h"

#include "absl/log/log.h"
#include "core/communicator/misc/ibv_wrap.h"

namespace tensorcast::communicator::engine {

namespace {
MrCache::RegisterFn DefaultRegisterFn() {
  return [](ibv_pd* pd, void* addr, size_t bytes, int access) -> ibv_mr* {
    struct ibv_mr* mr = nullptr;
    int rc = misc::wrap_ibv_reg_mr(&mr, pd, addr, bytes, access);
    if (rc != misc::SUCCESS) {
      return nullptr;
    }
    return mr;
  };
}

MrCache::DeregisterFn DefaultDeregisterFn() {
  return [](ibv_mr* mr) -> absl::Status {
    if (mr == nullptr) {
      return absl::OkStatus();
    }
    auto rc = misc::wrap_ibv_dereg_mr(mr);
    if (rc != misc::SUCCESS) {
      return absl::InternalError("Failed to dereg MR");
    }
    return absl::OkStatus();
  };
}
} // namespace

MrCache::MrCache(Options options) : options_(std::move(options)) {
  if (!options_.register_fn) {
    options_.register_fn = DefaultRegisterFn();
  }
  if (!options_.deregister_fn) {
    options_.deregister_fn = DefaultDeregisterFn();
  }
}

MrCache::~MrCache() {
  absl::MutexLock lk(&mu_);
  for (auto& kv : cache_) {
    if (!kv.second) {
      continue;
    }
    auto status = options_.deregister_fn(kv.second);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to dereg MR in MrCache dtor: " << status;
    }
  }
  cache_.clear();
}

MrCache::Result MrCache::get_or_register(ibv_pd* pd, gsl::not_null<void*> ptr, size_t bytes, int access) {
  Key key{.pd = pd, .ptr = ptr};
  {
    absl::MutexLock lk(&mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return Result{.mr = it->second, .registered = false};
    }
  }
  // Not found; register
  struct ibv_mr* mr = options_.register_fn(pd, ptr.get(), bytes, access);
  if (mr == nullptr) {
    return Result{.mr = nullptr, .registered = false};
  }
  absl::MutexLock lk(&mu_);
  cache_.emplace(key, mr);
  return Result{.mr = mr, .registered = true};
}

bool MrCache::contains(gsl::not_null<ibv_pd*> pd, gsl::not_null<void*> ptr) const {
  Key key{.pd = pd, .ptr = ptr};
  absl::MutexLock lk(&mu_);
  return cache_.find(key) != cache_.end();
}

size_t MrCache::size() const {
  absl::MutexLock lk(&mu_);
  return cache_.size();
}

} // namespace tensorcast::communicator::engine
