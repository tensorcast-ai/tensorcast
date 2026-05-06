// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "core/common/async_runtime.h"
#include "core/common/const/granularity.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/pinned_memory_authority.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/memory_tier_config.h"

namespace tensorcast::store {

/**
 * @brief Options structure used to configure a StoreEngine instance.
 *
 * This replaces the long positional parameter list previously used by the
 * StoreEngine constructor, making the API safer and future-proof.  All
 * fields have sensible defaults so callers may omit values they do not need
 * to customise.
 */
struct StoreEngineOptions {
  // Path were replica sub-directories are stored.  Empty string means fully
  // qualified paths will be provided to the loading API.
  std::string storage_path;

  // Size of the pinned host memory pool in bytes.
  size_t memory_pool_size{8ULL << 30}; // 8 GiB

  // Number of I/O worker threads.
  int num_thread{10};

  // Transfer slice size in bytes for streaming transfers (pinned pool block size).
  // Canonical name: tx_slice_bytes
  size_t tx_slice_bytes{common::consts::kTxSliceDefault};

  // Optional external pinned buffer pool to use instead of constructing a
  // private pool from (memory_pool_size, tx_slice_bytes).
  std::shared_ptr<common::memory::PinnedBufferPool> pinned_buffer_pool_override{nullptr};

  // Optional pinned memory authority for daemon-wide pooling and metrics.
  std::shared_ptr<common::memory::PinnedMemoryAuthority> pinned_memory_authority{nullptr};

  // Total host-pinned reservation used for UMA budgeting. When set to 0, the
  // engine uses `memory_pool_size` as the pinned reservation.
  size_t pinned_total_bytes{0};

  // Timeout for pinned-memory allocations.
  std::chrono::milliseconds pinned_memory_timeout{std::chrono::milliseconds{30000}}; // 30 s

  // Host/interface used by the Communicator listener. Empty string falls back
  // to loopback, while "0.0.0.0" exposes the port on all interfaces.
  std::string p2p_listen_host{"127.0.0.1"};

  // Port used by Communicator for P2P transfers.
  // When set to 0, the engine binds an ephemeral port chosen by the OS.
  uint16_t p2p_port{0};

  // Toggle RDMA support for the Communicator. When true the communicator will
  // attempt to register RDMA transports; otherwise it operates in TCP-only
  // mode with staging buffers for GPU memory.
  bool enable_rdma{false};

  // Address of the Global Store gRPC service ("host:port").  Empty string means
  // the StoreEngine will operate in standalone mode without remote
  // coordination and P2P source discovery.
  std::string global_store_address;

  // Optional externally created Global Store client so callers can share a
  // single channel (e.g., daemon worker lifecycle + StoreEngine).
  std::shared_ptr<components::IGlobalStoreClient> global_store_client{nullptr};

  // (Phase-3) Optional externally created CommunicationManager so multiple
  // StoreEngine instances can share the same underlying Communicator
  // and listen socket.  When provided, the StoreEngine will reuse this
  // manager instead of creating its own internal instance.
  std::shared_ptr<components::CommunicationManager> comm_manager{nullptr};

  // Optional externally created AsyncRuntime so StoreEngine and daemon can
  // share a single executor runtime with unified shutdown/drain.
  std::shared_ptr<common::AsyncRuntime> async_runtime{nullptr};

  // Default depth for StreamingPinnedBuffer instances created by StoreEngine.
  // This bounds overlap and pinned pressure for streaming pipelines.
  uint32_t streaming_buffer_chunks{16};

  // UMA/VS artifact chunk size for CPU-side allocations (bytes).
  // Canonical name: artifact_chunk_bytes
  size_t artifact_chunk_bytes{common::consts::kArtifactChunkDefault};

  // RFC-0007: Enable strong verification (FULL_DIGEST) by default on load.
  // When true, the loader will compute the data_multihash from the loaded
  // GPU buffer to strongly validate content-addressed identity without the
  // caller needing to set MaterializeHints::verify = FULL_DIGEST.
  bool force_full_digest_on_load{false};

  // Stable/preemptible memory tier settings (optional). When unset, the
  // engine preserves legacy preemptible behavior.
  std::optional<MemoryTierConfig> memory_tier_config;

  // When true, UMA CPU allocations are backed by memfd + MAP_SHARED so they can
  // be exported cross-process for zero-copy CPU tensor materialization.
  bool cpu_shared_memory_enabled{true};

  struct ByteMappingConfig {
    bool enable_strided_execution{true};
    bool enable_direct_write_at{true};
    uint32_t program_cache_entries{256};
    uint32_t strided_run_min_ranges{128};
    uint64_t strided_min_row_len_bytes{4096};
    uint32_t strided_max_amplification{8};
    uint64_t strided_block_target_bytes{16ULL * 1024 * 1024};
    uint64_t strided_block_max_bytes{64ULL * 1024 * 1024};
    bool disk_source_ordered_read{true};
    uint64_t disk_source_merge_max_gap_bytes{256ULL * 1024};
    uint32_t disk_source_merge_max_amplification{4};
    uint32_t disk_source_prefetch_depth{2};
  };

  ByteMappingConfig byte_mapping{};

  struct MaterializationStrategyConfig {
    enum class ExecutorPreference : std::uint8_t {
      kAuto = 0,
      kGenericByteRange = 1,
      kTensorAwareLocal = 2,
      kOwnerFileCollective = 3,
    };

    enum class DiagnosticsVerbosity : std::uint8_t {
      kOff = 0,
      kBasic = 1,
      kVerbose = 2,
    };

    bool enable_tensor_aware_mapped_executor{true};
    bool enable_local_batched_disk_load{false};
    bool enable_owner_file_collective{false};
    bool allow_mixed_execution{true};
    bool prefer_local_canonical_for_mapped{false};
    bool allow_source_ordered_for_mapped{true};
    bool enable_mapped_dim0_tensor_jobs{true};
    bool enable_mapped_dim1_tensor_jobs{true};
    bool enable_mapped_concat_jobs{true};
    bool enable_mapped_concat_execution{true};
    bool enable_mapped_single_range_concat_jobs{true};
    bool enable_mapped_multirange_concat_jobs{true};
    bool sync_after_single_range_concat_job{false};
    bool use_dedicated_single_range_concat_stream{false};
    ExecutorPreference executor_preference{ExecutorPreference::kAuto};
    DiagnosticsVerbosity diagnostics_verbosity{DiagnosticsVerbosity::kBasic};
    uint64_t direct_write_batch_bytes{0};
    uint32_t direct_write_batch_ops{0};
    uint64_t owner_file_collective_peak_bytes_budget{8ULL * 1024ULL * 1024ULL * 1024ULL};
    uint64_t owner_file_collective_batch_bytes{512ULL * 1024ULL * 1024ULL};
    uint64_t owner_file_collective_dim1_staging_bytes{256ULL * 1024ULL * 1024ULL};
    uint32_t owner_file_collective_max_inflight_batches{1};
    bool owner_file_collective_shared_fs_only{true};
    double owner_file_collective_max_owner_skew_ratio{1.5};
    uint64_t owner_file_collective_min_dedup_saving_bytes{64ULL * 1024ULL * 1024ULL};
    std::chrono::milliseconds owner_file_collective_group_assemble_timeout{std::chrono::milliseconds(15000)};
    bool owner_file_collective_allow_mixed_residual{false};
    uint32_t owner_file_collective_planner_cache_entries{256};
  };

  MaterializationStrategyConfig materialization_strategy{};

  enum class PromotionPolicy : std::uint8_t {
    kUnspecified = 0,
    kNever = 1,
    kOnMaterialize = 2,
    kOnHotness = 3,
    kOnPolicy = 4,
  };

  struct PromotionOptions {
    PromotionPolicy policy{PromotionPolicy::kNever};
    bool require_verified{false};
    std::chrono::milliseconds demotion_drain_timeout{0};
    uint32_t max_concurrency{4};
  };

  PromotionOptions promotion{};
};

} // namespace tensorcast::store
