-- TensorCast canonical schema (schema.sql)
--
-- This file is the single source of truth for persistent relational data
-- structures across the repository (see docs/designs/0001-docs-system-reorg-design.md).
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

    -- Prevent duplicate registration for the same address:port
    UNIQUE(node_address, grpc_port)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_workers_last_heartbeat ON workers (last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_workers_accepting_requests ON workers (accepting_new_requests, last_heartbeat);
CREATE INDEX IF NOT EXISTS idx_workers_node_id ON workers (node_id);
CREATE INDEX IF NOT EXISTS idx_workers_registered_at ON workers (registered_at);

-- Artifacts: content-addressed artifact IDs (design-0007)
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,          -- "mi2:<index_multihash>:<data_multihash>"
    index_multihash TEXT NOT NULL,         -- Multibase over multihash (sha2-256), base32
    data_multihash TEXT NOT NULL,          -- Multibase over multihash (sha2-256 root), base32
    schema_version TEXT NOT NULL,          -- e.g., "v2"
    encoding TEXT NOT NULL,                -- e.g., "json" or future "cbor"
    hash_params_json TEXT NULL,            -- JSON string for hashing params (e.g., chunk_size, fanout)
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
    schema_version TEXT NOT NULL,          -- e.g., "v2"
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

-- Distributed Virtual Memory Pool (DVMP) chunk directory
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

-- RFC-0014: Human key → artifact_id mapping with optional routing hints
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

-- ===================== End Global Store =====================

