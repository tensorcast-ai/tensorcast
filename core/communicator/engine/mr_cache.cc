// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/engine/mr_cache.h"

#include "absl/log/log.h"
#include "core/communicator/misc/ibv_wrap.h"

namespace tensorcast::communicator::engine {

MrCache::~MrCache() {
  absl::MutexLock lk(&mu_);
  for (auto& kv : cache_) {
    if (kv.second) {
      auto rc = misc::wrap_ibv_dereg_mr(kv.second);
      if (rc != misc::SUCCESS) {
        LOG(WARNING) << "Failed to dereg MR in MrCache dtor";
      }
    }
  }
  cache_.clear();
}

struct ibv_mr* MrCache::get_or_register(ibv_pd* pd, gsl::not_null<void*> ptr, size_t bytes, int access) {
  Key key{pd, ptr};
  {
    absl::MutexLock lk(&mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
  }
  // Not found; register
  struct ibv_mr* mr = nullptr;
  int rc = misc::wrap_ibv_reg_mr(&mr, pd, ptr.get(), bytes, access);
  if (rc != misc::SUCCESS) {
    return nullptr;
  }
  absl::MutexLock lk(&mu_);
  cache_.emplace(key, mr);
  return mr;
}

} // namespace tensorcast::communicator::engine
