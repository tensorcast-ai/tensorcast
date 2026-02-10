-- TensorCast canonical schema (schema.sql)
--
-- This file is the single source of truth for persistent relational data
-- structures across the repository (see docs/designs/0001-docs-system-design.md).
--
-- Areas currently covered:
--   - Global Store (DuckDB-backed): workers, artifact_replicas, replica_counters,
--     artifact_transports, artifacts, artifact_indices, chunk_directory, key_mappings
--
-- Notes:
--   - SQL dialect strives to be DuckDB-compatible as the Global Store uses DuckDB.
--   - CREATE TABLE statements use IF NOT EXISTS to allow safe re-application.
--   - When you change schema here, update affected design docs (link from the
--     design’s frontmatter to this file) and ensure code is updated accordingly.

-- ===================== Global Store =====================

-- Cluster info (singleton row)
CREATE TABLE IF NOT EXISTS cluster_info (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    cluster_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Workers table
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    -- Stable daemon identity (required; from daemon config); preferred control-plane identity.
    daemon_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_address TEXT NOT NULL,
    grpc_port INTEGER NOT NULL,
    p2p_port INTEGER NOT NULL,
    mem_pool_total_size BIGINT NOT NULL,
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    inactive_at TIMESTAMP WITH TIME ZONE,

    -- Prevent duplicate registration for the same address:port
    UNIQUE(node_address, grpc_port)
);

CREATE TABLE IF NOT EXISTS worker_liveness (
    worker_id TEXT PRIMARY KEY,
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    mem_pool_available_size BIGINT NOT NULL,
    accepting_new_requests BOOLEAN NOT NULL DEFAULT TRUE,
    capability_flags BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS worker_reconcile_state (
    worker_id TEXT PRIMARY KEY,
    generation BIGINT NOT NULL DEFAULT 1,
    request_seq BIGINT NOT NULL DEFAULT 0,
    state_version BIGINT NOT NULL DEFAULT 1,
    state_checksum TEXT NOT NULL DEFAULT '',
    last_reconcile_result TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_workers_node_id ON workers (node_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workers_daemon_id_unique ON workers (daemon_id);
CREATE INDEX IF NOT EXISTS idx_workers_registered_at ON workers (registered_at);
CREATE INDEX IF NOT EXISTS idx_workers_inactive_at ON workers (inactive_at);
CREATE INDEX IF NOT EXISTS idx_worker_liveness_last_heartbeat ON worker_liveness (last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_worker_liveness_accepting ON worker_liveness (accepting_new_requests, last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_worker_reconcile_state_generation_seq
    ON worker_reconcile_state (worker_id, generation, request_seq);

-- Engine instance registry (node-local engine processes)
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    daemon_id TEXT NOT NULL,
    worker_id TEXT NULL,
    engine TEXT NOT NULL,
    signals_endpoint TEXT,
    labels_json TEXT NOT NULL DEFAULT '{}',
    capability_flags BIGINT NOT NULL DEFAULT 0,
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    inactive_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_instances_daemon_id ON instances (daemon_id);
CREATE INDEX IF NOT EXISTS idx_instances_worker_id ON instances (worker_id);
CREATE INDEX IF NOT EXISTS idx_instances_last_heartbeat ON instances (last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_instances_inactive_at ON instances (inactive_at);
-- Memory tier telemetry snapshots (short retention)
CREATE TABLE IF NOT EXISTS memory_tier_snapshots (
    node_id TEXT NOT NULL,
    epoch_ns BIGINT NOT NULL,
    stable_total_bytes BIGINT NOT NULL,
    stable_used_bytes BIGINT NOT NULL,
    preemptible_total_bytes BIGINT NOT NULL,
    preemptible_marked_bytes BIGINT NOT NULL,
    faults_per_sec REAL NOT NULL,
    rehydrate_p99_ns BIGINT NOT NULL,
    enable_preemptible BOOLEAN NOT NULL,
    memory_tier_config_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (node_id, epoch_ns)
);
CREATE INDEX IF NOT EXISTS idx_memory_tier_snapshots_node_epoch ON memory_tier_snapshots(node_id, epoch_ns);

-- Latest per-node memory tier state derived from telemetry snapshots
CREATE OR REPLACE VIEW node_memory_tier_latest AS
WITH ranked AS (
    SELECT
        node_id,
        stable_total_bytes,
        stable_used_bytes,
        preemptible_total_bytes,
        preemptible_marked_bytes,
        faults_per_sec,
        rehydrate_p99_ns,
        enable_preemptible,
        memory_tier_config_json,
        epoch_ns,
        ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY epoch_ns DESC) AS rn
    FROM memory_tier_snapshots
)
SELECT
    node_id,
    stable_total_bytes,
    stable_used_bytes,
    preemptible_total_bytes,
    preemptible_marked_bytes,
    faults_per_sec,
    rehydrate_p99_ns,
    enable_preemptible,
    memory_tier_config_json,
    epoch_ns AS snapshot_epoch_ns
FROM ranked
WHERE rn = 1;

-- Memory tier leases with UMA chunk ordinals for replay/audit
CREATE TABLE IF NOT EXISTS memory_tier_leases (
    lease_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    kind TEXT CHECK (kind IN ('stable','preemptible')) NOT NULL,
    artifact_id TEXT NOT NULL,
    chunk_range JSON NOT NULL,
    chunk_ids JSON NOT NULL,
    ledger_version BIGINT NOT NULL,
    bytes BIGINT NOT NULL,
    workload_id TEXT NOT NULL,
    state TEXT CHECK (state IN ('pending','active','revoking','expired')) NOT NULL DEFAULT 'pending',
    request_id TEXT NOT NULL,
    ack_epoch_ns BIGINT,
    issued_at_ns BIGINT NOT NULL,
    expires_at_ns BIGINT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_tier_leases_node_artifact ON memory_tier_leases(node_id, artifact_id, state);

-- Artifacts: content-addressed artifact IDs (design-0007)
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,          -- "mi2:<index_multihash>:<data_multihash>"
    index_multihash TEXT NULL,         -- Multibase over multihash (sha2-256), base32
    data_multihash TEXT NULL,          -- Multibase over multihash (sha2-256 root), base32
    schema_version TEXT NOT NULL DEFAULT 'v3',          -- canonical index schema version
    encoding TEXT NOT NULL,                -- e.g., "json" or future "cbor"
    hash_params_json TEXT NULL,            -- JSON string for hashing params (e.g., chunk_size, fanout)
    id_kind TEXT NOT NULL DEFAULT 'MI2',   -- Identity kind (MI2 or CGID)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifacts_index_mh ON artifacts(index_multihash);
CREATE INDEX IF NOT EXISTS idx_artifacts_data_mh ON artifacts(data_multihash);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at);

-- Artifact replicas (memory/disk/P2P)
CREATE TABLE IF NOT EXISTS artifact_replicas (
    replica_id UUID PRIMARY KEY,
    artifact_id TEXT NOT NULL,             -- FK logical: artifacts.artifact_id
    view_id TEXT NULL,
    disk_path TEXT NULL,                   -- Original on-disk path (optional)
    node_id TEXT NOT NULL,
    node_address VARCHAR NOT NULL,
    node_port INTEGER NOT NULL,
    memory_size BIGINT NOT NULL,
    memory_type VARCHAR NOT NULL,
    device_id INTEGER NOT NULL,
    max_concurrency INTEGER DEFAULT 5,
    is_available BOOLEAN DEFAULT TRUE,
    remote_memory_keys TEXT[] NULL,
    buffer_sizes BIGINT[] NULL,
    export_state TEXT NOT NULL DEFAULT 'PRESENCE_ONLY',
    export_generation BIGINT NOT NULL DEFAULT 0,
    verification_json TEXT NULL,
    -- relationship to workers
    worker_id TEXT,
    -- Memory replica fields
    is_memory_replica BOOLEAN DEFAULT FALSE,
    tensor_index_key TEXT NULL,
    source_process_id TEXT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Counters split out for high-frequency updates
CREATE TABLE IF NOT EXISTS replica_counters (
    replica_id UUID PRIMARY KEY,
    current_requests INTEGER NOT NULL DEFAULT 0,
    last_assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- Note: DuckDB foreign key constraints can cause issues with updates
    -- Manual cleanup is required when artifact_replicas entries are deleted
);

-- Indexes for replicas and counters
CREATE INDEX IF NOT EXISTS idx_replica_counters_current_requests ON replica_counters(current_requests);
CREATE INDEX IF NOT EXISTS idx_replica_counters_last_assigned ON replica_counters(last_assigned_at);

CREATE INDEX IF NOT EXISTS idx_artifact_replicas_artifact_id ON artifact_replicas(artifact_id);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_artifact_view ON artifact_replicas(artifact_id, view_id);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_disk_path ON artifact_replicas(disk_path);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_updated_at ON artifact_replicas(updated_at);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_node_id ON artifact_replicas(node_id);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_node_address ON artifact_replicas(node_address);
CREATE INDEX IF NOT EXISTS idx_replicas_worker ON artifact_replicas(worker_id);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_tensor_index_key ON artifact_replicas(tensor_index_key);
CREATE INDEX IF NOT EXISTS idx_artifact_replicas_memory_replica ON artifact_replicas(is_memory_replica, artifact_id);

-- Deduplicated tensor index storage
CREATE TABLE IF NOT EXISTS artifact_indices (
    index_key TEXT PRIMARY KEY,            -- SHA-256 hex of canonical index bytes
    schema_version TEXT NOT NULL DEFAULT 'v3',          -- canonical index schema version
    encoding TEXT NOT NULL,                -- e.g., "json" or "cbor"
    size_bytes BIGINT NOT NULL,
    index_data BLOB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifact_indices_created_at ON artifact_indices(created_at);
CREATE INDEX IF NOT EXISTS idx_artifact_indices_size ON artifact_indices(size_bytes);

-- In-flight artifact transports
CREATE TABLE IF NOT EXISTS artifact_transports (
    transport_id UUID PRIMARY KEY,
    replica_id UUID NOT NULL,
    artifact_id TEXT NOT NULL,
    disk_path TEXT NULL,
    source_node_id VARCHAR NOT NULL,
    source_address VARCHAR NOT NULL,
    source_port INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    status VARCHAR NOT NULL DEFAULT 'in_progress'
);

CREATE INDEX IF NOT EXISTS idx_artifact_transports_replica_id ON artifact_transports(replica_id);
CREATE INDEX IF NOT EXISTS idx_artifact_transports_source_node_id ON artifact_transports(source_node_id);
CREATE INDEX IF NOT EXISTS idx_artifact_transports_status ON artifact_transports(status);
CREATE INDEX IF NOT EXISTS idx_artifact_transports_created_at ON artifact_transports(created_at);
CREATE INDEX IF NOT EXISTS idx_artifact_transports_completed_at ON artifact_transports(completed_at);

-- Virtual Address Space (VS) chunk directory
CREATE TABLE IF NOT EXISTS chunk_directory (
    artifact_id TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    device_uuid TEXT NOT NULL,
    replica INTEGER NOT NULL DEFAULT 0,
    -- HOT=0, LOCKED_TX=1, COPIED_GPU=2, COLD=3, EVICTED=4
    chunk_state INTEGER NOT NULL DEFAULT 0,
    last_update_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- For intelligent source selection
    node_load_ratio FLOAT DEFAULT 0.0,
    -- PK uniquely identifies a replica instance for an artifact on a device
    PRIMARY KEY (artifact_id, device_uuid, replica, chunk_idx, node_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_directory_artifact_chunk ON chunk_directory(artifact_id, chunk_idx);
CREATE INDEX IF NOT EXISTS idx_chunk_directory_node ON chunk_directory(node_id);
CREATE INDEX IF NOT EXISTS idx_chunk_directory_state ON chunk_directory(chunk_state);
CREATE INDEX IF NOT EXISTS idx_chunk_directory_update_time ON chunk_directory(last_update_time);
CREATE INDEX IF NOT EXISTS idx_chunk_directory_source_selection ON chunk_directory(artifact_id, chunk_idx, chunk_state, node_load_ratio);

-- Persistence placement plans (docs/architecture/api/policy-persistence.md)
CREATE TABLE IF NOT EXISTS artifact_placements (
    plan_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    policy TEXT CHECK (policy IN ('local_only','replicated','sharded')) NOT NULL,
    shard_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(artifact_id)
);

CREATE TABLE IF NOT EXISTS artifact_placement_shards (
    plan_id TEXT NOT NULL,
    shard_idx INTEGER NOT NULL,
    shard_id TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    content_digest TEXT NOT NULL,
    byte_range_start BIGINT NOT NULL,
    byte_range_length BIGINT NOT NULL,
    chunk_ids JSON NOT NULL,
    PRIMARY KEY (plan_id, shard_idx)
);
CREATE INDEX IF NOT EXISTS idx_artifact_placement_shards_digest ON artifact_placement_shards(content_digest);

CREATE TABLE IF NOT EXISTS artifact_placement_targets (
    plan_id TEXT NOT NULL,
    shard_idx INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    lease_id TEXT NULL,
    target_state TEXT CHECK (target_state IN ('pending','copying','complete','failed','skipped')) NOT NULL,
    degraded_reason TEXT NULL,
    PRIMARY KEY (plan_id, shard_idx, node_id)
);
CREATE INDEX IF NOT EXISTS idx_artifact_placement_targets_node ON artifact_placement_targets(node_id);
CREATE INDEX IF NOT EXISTS idx_artifact_placement_targets_plan_state ON artifact_placement_targets(plan_id, target_state);

CREATE TABLE IF NOT EXISTS artifact_placement_summary (
    plan_id TEXT PRIMARY KEY,
    plan_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_persistence_status (
    task_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    state TEXT CHECK (state IN ('pending','running','success','failed','degraded')) NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    last_error TEXT NULL,
    degraded_reason TEXT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_artifact_persistence_status_artifact_state ON artifact_persistence_status(artifact_id, state);

-- Key mapping: Human key -> artifact_id with optional routing hints
CREATE TABLE IF NOT EXISTS key_mappings (
    key TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    replica_uuid TEXT NULL,
    daemon_address TEXT NULL,
    ttl_seconds BIGINT NULL,
    generation BIGINT NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'IMMUTABLE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_key_mappings_artifact ON key_mappings(artifact_id);

-- Disk locations (durable shared-disk persistence locations)
CREATE TABLE IF NOT EXISTS artifact_disk_locations (
    artifact_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    kind TEXT CHECK (kind IN ('MANAGED','IMPORTED')) NOT NULL DEFAULT 'MANAGED',
    -- Soft delete marker for managed disk GC. Deleted entries are ignored for disk fallback.
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (artifact_id, cluster_id, relative_path)
);
CREATE INDEX IF NOT EXISTS idx_artifact_disk_locations_artifact ON artifact_disk_locations(artifact_id);
CREATE INDEX IF NOT EXISTS idx_artifact_disk_locations_cluster ON artifact_disk_locations(cluster_id);

-- View metadata anchored to artifacts (canonical or assemblies)
CREATE TABLE IF NOT EXISTS views (
    artifact_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    view_spec_json TEXT NOT NULL,
    view_size BIGINT NOT NULL,
    view_data_hash TEXT,
    verified_at TIMESTAMP WITH TIME ZONE,
    canonical_size_bytes BIGINT NULL,
    canonical_bytes_covered BIGINT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (artifact_id, view_id)
);

CREATE INDEX IF NOT EXISTS idx_views_artifact ON views(artifact_id);
CREATE INDEX IF NOT EXISTS idx_views_verified_at ON views(artifact_id, verified_at);
CREATE INDEX IF NOT EXISTS idx_views_view_id ON views(view_id);

-- Canonical coverage ranges per view (piece coverage manifest)
CREATE TABLE IF NOT EXISTS view_coverage_ranges (
    artifact_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    range_offset BIGINT NOT NULL,
    range_length BIGINT NOT NULL,
    PRIMARY KEY (artifact_id, view_id, range_offset, range_length)
);

CREATE INDEX IF NOT EXISTS idx_view_coverage_artifact ON view_coverage_ranges(artifact_id);
CREATE INDEX IF NOT EXISTS idx_view_coverage_view ON view_coverage_ranges(artifact_id, view_id);

-- Immutable, content-addressed layout specs (v2)
CREATE TABLE IF NOT EXISTS layout_specs (
    layout_id TEXT PRIMARY KEY,            -- "mh:..." over deterministic proto
    index_multihash TEXT NOT NULL,
    layout_proto BLOB NOT NULL,
    layout_json TEXT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_layout_specs_index_mh ON layout_specs(index_multihash);

-- Unsealed assembly -> layout binding (mutable pointer, versioned)
CREATE TABLE IF NOT EXISTS assembly_layout_bindings (
    assembly_id TEXT PRIMARY KEY,
    layout_id TEXT NOT NULL,
    binding_version BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_assembly_layout_bindings_layout ON assembly_layout_bindings(layout_id);

-- Sealed artifact -> layout attachments (immutable, idempotent)
CREATE TABLE IF NOT EXISTS artifact_layout_attachments (
    mi2_id TEXT NOT NULL,
    layout_id TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (mi2_id, layout_id)
);
CREATE INDEX IF NOT EXISTS idx_artifact_layout_attachments_mi2 ON artifact_layout_attachments(mi2_id);
CREATE INDEX IF NOT EXISTS idx_artifact_layout_attachments_layout ON artifact_layout_attachments(layout_id);

-- Per-assembly operational runtime policy (mutable; not content-addressed)
CREATE TABLE IF NOT EXISTS assembly_runtime_policies (
    assembly_id TEXT PRIMARY KEY,
    policy_version BIGINT NOT NULL,
    policy_json TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Unified operations for long-tail workflows (v2)
CREATE TABLE IF NOT EXISTS operations (
    operation_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    target_artifact_id TEXT NOT NULL,
    state TEXT CHECK (state IN ('pending','running','success','failed','cancelled','degraded')) NOT NULL,
    status_proto BLOB NOT NULL,
    snapshot_proto BLOB NULL,
    lease_owner TEXT NULL,
    lease_token TEXT NULL,
    lease_generation BIGINT NOT NULL DEFAULT 0,
    lease_expires_at TIMESTAMP WITH TIME ZONE NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_operations_target ON operations(kind, target_artifact_id);
CREATE INDEX IF NOT EXISTS idx_operations_state ON operations(state);

-- Unsealed proof commitments (assembly-scoped; replicated overlaps)
CREATE TABLE IF NOT EXISTS assembly_proof_commitments (
    assembly_id TEXT NOT NULL,
    tensor_name TEXT NOT NULL,
    proof_schema_version TEXT NOT NULL,
    proof_chunk_idx BIGINT NOT NULL,
    digest BLOB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (assembly_id, tensor_name, proof_schema_version, proof_chunk_idx)
);
CREATE INDEX IF NOT EXISTS idx_assembly_proof_commitments_tensor ON assembly_proof_commitments(assembly_id, tensor_name);

-- Sealed proof commitments (MI2-scoped; long-lived truth)
CREATE TABLE IF NOT EXISTS tensor_proof_commitments (
    mi2_id TEXT NOT NULL,
    tensor_name TEXT NOT NULL,
    proof_schema_version TEXT NOT NULL,
    proof_chunk_idx BIGINT NOT NULL,
    digest BLOB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (mi2_id, tensor_name, proof_schema_version, proof_chunk_idx)
);
CREATE INDEX IF NOT EXISTS idx_tensor_proof_commitments_tensor ON tensor_proof_commitments(mi2_id, tensor_name);

-- Per-piece proof digests (audit/debug + conflict attribution)
CREATE TABLE IF NOT EXISTS piece_proof_digests (
    assembly_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    tensor_name TEXT NOT NULL,
    proof_schema_version TEXT NOT NULL,
    proof_chunk_idx BIGINT NOT NULL,
    digest BLOB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (assembly_id, view_id, tensor_name, proof_schema_version, proof_chunk_idx)
);
CREATE INDEX IF NOT EXISTS idx_piece_proof_digests_tensor ON piece_proof_digests(assembly_id, view_id, tensor_name);

-- Assembly → sealed bindings (cgid -> mi2)
CREATE TABLE IF NOT EXISTS artifact_bindings (
    from_artifact_id TEXT PRIMARY KEY,
    to_artifact_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'seal',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_artifact_bindings_to ON artifact_bindings(to_artifact_id);

-- Leaf digests anchored to canonical or view ByteSpaces
CREATE TABLE IF NOT EXISTS leaves (
    artifact_id TEXT NOT NULL,
    space_kind CHAR(1) NOT NULL,
    space_id TEXT NOT NULL,
    leaf_idx BIGINT NOT NULL,
    digest BLOB NOT NULL,
    PRIMARY KEY (artifact_id, space_kind, space_id, leaf_idx)
);

CREATE INDEX IF NOT EXISTS idx_leaves_space ON leaves(artifact_id, space_kind, space_id);

-- ===================== End Global Store =====================
