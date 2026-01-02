// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "gsl/pointers"

namespace tensorcast::communicator::engine {

// Simple per-PD MR cache keyed by (pd, buffer_ptr).
// Registers an MR once for a given (pd, ptr, length) and reuses it.
class MrCache {
 public:
  struct Result {
    ibv_mr* mr = nullptr;
    bool registered = false;
  };

  using RegisterFn = std::function<ibv_mr*(ibv_pd* pd, void* addr, size_t bytes, int access)>;
  using DeregisterFn = std::function<absl::Status(ibv_mr* mr)>;

  struct Options {
    RegisterFn register_fn;
    DeregisterFn deregister_fn;
  };

  explicit MrCache(Options options = {});
  ~MrCache();

  // Returns a cached or newly-registered MR for (pd, ptr, bytes).
  // If a cached MR exists with matching base ptr, returns it regardless
  // of requested bytes, assuming the cached MR length covers the staging chunk.
  // Access flags must be consistent for a given (pd, ptr) registration.
  Result get_or_register(ibv_pd* pd, gsl::not_null<void*> ptr, size_t bytes, int access) ABSL_LOCKS_EXCLUDED(mu_);

  // Preferred overload using not_null for pd.
  Result get_or_register(gsl::not_null<ibv_pd*> pd, gsl::not_null<void*> ptr, size_t bytes, int access)
      ABSL_LOCKS_EXCLUDED(mu_) {
    return get_or_register(pd.get(), ptr, bytes, access);
  }

  [[nodiscard]] bool contains(gsl::not_null<ibv_pd*> pd, gsl::not_null<void*> ptr) const ABSL_LOCKS_EXCLUDED(mu_);
  [[nodiscard]] size_t size() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  struct Key {
    gsl::not_null<ibv_pd*> pd;
    gsl::not_null<void*> ptr;

    bool operator==(const Key& o) const {
      return pd.get() == o.pd.get() && ptr.get() == o.ptr.get();
    }
  };

  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<void*>{}(k.pd.get()) ^ (std::hash<void*>{}(k.ptr.get()) << 1);
    }
  };

  mutable absl::Mutex mu_;
  std::unordered_map<Key, gsl::owner<ibv_mr*>, KeyHash> cache_ ABSL_GUARDED_BY(mu_);
  Options options_;
};

} // namespace tensorcast::communicator::engine
