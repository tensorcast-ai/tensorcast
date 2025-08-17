--  Copyright (c) 2025, StepCast Team.

-- Workers表
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

    -- 组合唯一索引，防止同一地址端口重复注册
    UNIQUE(node_address, grpc_port)
);

-- 性能索引
CREATE INDEX idx_workers_last_heartbeat ON workers (last_heartbeat);
CREATE INDEX idx_workers_accepting_requests ON workers (accepting_new_requests, last_heartbeat);
CREATE INDEX idx_workers_node_id ON workers (node_id);
CREATE INDEX idx_workers_registered_at ON workers (registered_at);

CREATE TABLE model_replicas (
    replica_id UUID PRIMARY KEY,
    model_name VARCHAR NOT NULL,
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
    -- 关联到 workers 表
    worker_id TEXT,
    -- Memory replica fields
    is_memory_replica BOOLEAN DEFAULT FALSE,
    tensor_index_key TEXT NULL,
    source_process_id TEXT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 新增计数器表，用于高频更新字段隔离
CREATE TABLE IF NOT EXISTS replica_counters (
    replica_id UUID PRIMARY KEY,
    current_requests INTEGER NOT NULL DEFAULT 0,
    last_assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- Note: DuckDB foreign key constraints can cause issues with updates
    -- Manual cleanup is required when model_replicas entries are deleted
    -- Foreign key relationship: replica_id -> model_replicas(replica_id)
);

-- 针对负载均衡查询的索引
CREATE INDEX idx_replica_counters_current_requests ON replica_counters(current_requests);
CREATE INDEX idx_replica_counters_last_assigned ON replica_counters(last_assigned_at);

CREATE INDEX idx_model_replicas_model_name ON model_replicas(model_name);
CREATE INDEX idx_model_replicas_model_name_available ON model_replicas(model_name, is_available);
CREATE INDEX idx_model_replicas_updated_at ON model_replicas(updated_at);
CREATE INDEX idx_model_replicas_node_id ON model_replicas(node_id);
CREATE INDEX idx_model_replicas_node_address ON model_replicas(node_address);
CREATE INDEX idx_replicas_worker ON model_replicas(worker_id);
CREATE INDEX idx_model_replicas_tensor_index_key ON model_replicas(tensor_index_key);
CREATE INDEX idx_model_replicas_memory_replica ON model_replicas(is_memory_replica, model_name);

-- Table for storing deduplicated tensor indices
CREATE TABLE IF NOT EXISTS model_indices (
    index_key TEXT PRIMARY KEY,            -- SHA-256 hash of canonical JSON index
    schema_version TEXT NOT NULL,          -- Schema version (e.g., "v2")
    encoding TEXT NOT NULL,                -- Encoding format (e.g., "json")
    size_bytes BIGINT NOT NULL,            -- Size of the index data
    index_data BLOB NOT NULL,              -- Canonical JSON bytes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for model_indices table
CREATE INDEX idx_model_indices_created_at ON model_indices(created_at);
CREATE INDEX idx_model_indices_size ON model_indices(size_bytes);

-- Table for tracking model transports
CREATE TABLE model_transports (
    transport_id UUID PRIMARY KEY,
    replica_id UUID NOT NULL,
    model_name VARCHAR NOT NULL,
    source_node_id VARCHAR NOT NULL,
    source_address VARCHAR NOT NULL,
    source_port INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    status VARCHAR NOT NULL DEFAULT 'in_progress',
    -- FOREIGN KEY (replica_id) REFERENCES model_replicas(replica_id)
);

CREATE INDEX idx_model_transports_replica_id ON model_transports(replica_id);
CREATE INDEX idx_model_transports_source_node_id ON model_transports(source_node_id);
CREATE INDEX idx_model_transports_status ON model_transports(status);
CREATE INDEX idx_model_transports_created_at ON model_transports(created_at);
CREATE INDEX idx_model_transports_completed_at ON model_transports(completed_at);

-- 添加触发器，当worker被删除时，相关replica标记为不可用
-- 注意：这个触发器在SQLite中可能需要根据实际使用的数据库进行调整
-- CREATE TRIGGER mark_replicas_unavailable_on_worker_delete
-- AFTER DELETE ON workers
-- FOR EACH ROW
-- BEGIN
--     UPDATE model_replicas
--     SET is_available = FALSE
--     WHERE worker_id = OLD.worker_id;
-- END;

-- ========== Chunk Directory for Distributed Memory Pool (DVMP) ==========
-- Tracks global chunk distribution for distributed virtual memory pool

CREATE TABLE IF NOT EXISTS chunk_directory (
    -- Primary key components
    model_id TEXT NOT NULL,
    chunk_idx INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    device_uuid TEXT NOT NULL,
    replica INTEGER NOT NULL DEFAULT 0,

    -- Chunk state (matches ChunkState enum)
    -- HOT=0, LOCKED_TX=1, COPIED_GPU=2, COLD=3, EVICTED=4
    chunk_state INTEGER NOT NULL DEFAULT 0,

    -- Metadata
    last_update_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- For intelligent source selection
    node_load_ratio FLOAT DEFAULT 0.0,

    -- InstanceKey uniquely identifies a model instance
    PRIMARY KEY (model_id, device_uuid, replica, chunk_idx, node_id)
);

-- Performance indexes for chunk queries
CREATE INDEX idx_chunk_directory_model_chunk ON chunk_directory(model_id, chunk_idx);
CREATE INDEX idx_chunk_directory_node ON chunk_directory(node_id);
CREATE INDEX idx_chunk_directory_state ON chunk_directory(chunk_state);
CREATE INDEX idx_chunk_directory_update_time ON chunk_directory(last_update_time);

-- Composite index for finding best source for a chunk
CREATE INDEX idx_chunk_directory_source_selection ON chunk_directory(
    model_id, chunk_idx, chunk_state, node_load_ratio
);
