// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>

namespace tensorcast::common {

/**
 * @brief Process-wide capabilities detected at startup.
 *
 * The detector probes mlock quota/permissions and MADV_* support once and
 * caches results to avoid repeated syscalls on hot paths. RDMA availability
 * can be recorded by higher layers (e.g., Communicator) and consulted by
 * loaders to decide on direct-write paths.
 */
class SystemCapabilities {
 public:
  static SystemCapabilities& instance();

  // --- Getters -------------------------------------------------------------
  [[nodiscard]] bool mlock_enabled() const noexcept {
    return mlock_enabled_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool madv_free_available() const noexcept {
    return madv_free_available_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool madv_pageout_available() const noexcept {
    return madv_pageout_available_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool madv_willneed_available() const noexcept {
    return madv_willneed_available_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool rdma_available() const noexcept {
    return rdma_available_.load(std::memory_order_acquire);
  }

  // --- Setters -------------------------------------------------------------
  // Record RDMA capability once the communication engine is initialized.
  void record_rdma_available(bool available) noexcept {
    rdma_available_.store(available, std::memory_order_release);
  }

  // Allow tests or higher layers to override mlock policy if needed.
  void set_mlock_enabled(bool enabled) noexcept {
    mlock_enabled_.store(enabled, std::memory_order_release);
  }

  SystemCapabilities(const SystemCapabilities&) = delete;
  SystemCapabilities& operator=(const SystemCapabilities&) = delete;

 private:
  SystemCapabilities();
  ~SystemCapabilities() = default;

  void detect_mlock_() noexcept;
  void detect_madv_() noexcept;

  std::atomic<bool> mlock_enabled_{false};
  std::atomic<bool> madv_free_available_{false};
  std::atomic<bool> madv_pageout_available_{false};
  std::atomic<bool> madv_willneed_available_{false};
  std::atomic<bool> rdma_available_{false};
};

} // namespace tensorcast::common
