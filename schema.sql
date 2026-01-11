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

-- Workers table
CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    node_address TEXT NOT NULL,
    grpc_port INTEGER NOT NULL,
    p2p_port INTEGER NOT NULL,
    mem_pool_total_size BIGINT NOT NULL,
    mem_pool_available_size BIGINT NOT NULL,
    accepting_new_requests BOOLEAN NOT NULL DEFAULT TRUE,
    registered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state_version BIGINT NOT NULL DEFAULT 1,
    state_checksum TEXT NOT NULL DEFAULT '',

    -- Prevent duplicate registration for the same address:port
    UNIQUE(node_address, grpc_port)
);

ALTER TABLE workers ADD COLUMN IF NOT EXISTS state_version BIGINT;
ALTER TABLE workers ADD COLUMN IF NOT EXISTS state_checksum TEXT;
UPDATE workers SET state_version = 1 WHERE state_version IS NULL;
UPDATE workers SET state_checksum = '' WHERE state_checksum IS NULL;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_workers_last_heartbeat ON workers (last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_workers_accepting_requests ON workers (accepting_new_requests, last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_workers_node_id ON workers (node_id);
CREATE INDEX IF NOT EXISTS idx_workers_registered_at ON workers (registered_at);
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
    disk_path TEXT NULL,
    ttl_seconds BIGINT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_key_mappings_artifact ON key_mappings(artifact_id);

-- Variant views anchored to canonical artifacts
CREATE TABLE IF NOT EXISTS variants (
    artifact_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    view_spec_json TEXT NOT NULL,
    view_size BIGINT NOT NULL,
    view_data_hash TEXT,
    verified_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (artifact_id, view_id)
);

CREATE INDEX IF NOT EXISTS idx_variants_artifact ON variants(artifact_id);
CREATE INDEX IF NOT EXISTS idx_variants_verified_at ON variants(artifact_id, verified_at);
CREATE INDEX IF NOT EXISTS idx_variants_view_id ON variants(view_id);

-- Leaf digests anchored to canonical or variant ByteSpaces
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
