---
slug: 0016-artifact-view-v1-global-store
title: Plan — Global Store: Variants & Leaves (v1)
links:
  design: ../designs/0016-artifact-view-v1.md
areas: ["global_store","schema","proto"]
related_code:
  - tensorcast/global_store/**
  - proto/tensorcast/global_store/v1/global_store.proto
  - schema.sql
---

# Objective

Add durable anchors for variant identity and fast verification leaves in Global Store, extend RPCs to read/write view metadata and leaves, and wire repositories/servicer to support daemon flows.

# Phases & Milestones

- [x] Phase 1: Schema
  - [x] Append `variants` and `leaves` tables to `schema.sql` with indexes
  - [x] Verify `db_utils.init_db()` applies the updated schema
  - [x] Update existing tables (`artifacts`, `artifact_indices`) to default `schema_version='v3'` and drop legacy v2 comments

- [x] Phase 2: Proto
  - [x] Extend `GetArtifactInfoById` request/response with view-aware options
  - [x] Add `UpdateArtifactViewState` RPC for variant upsert and batched leaf writes

- [x] Phase 3: Servicer & repositories
  - [x] Implement repositories for `variants` and `leaves`
  - [x] Implement servicer handlers with transactions and basic validation

- [x] Phase 4: Tests
  - [x] Unit tests for repos and servicer endpoints
  - [x] Partial coverage error detail structure validation

- [x] Phase 5: Daemon integration glue
  - [x] Support `space=view_id` queries that fall back to canonical replicas when variants unavailable
  - [x] Emit `PARTIAL_COVERAGE` detail when requested leaves/ranges missing
  - [x] Provide helper for daemons to upsert variant metadata while keeping canonical `chunk_directory` untouched

# Tasks (Detailed TODO)

- [x] Update `schema.sql`
  - [x] Add table `variants(artifact_id TEXT, view_id TEXT, view_spec_json TEXT, view_size BIGINT, view_data_hash TEXT NULL, verified_at TIMESTAMPTZ NULL, created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(artifact_id, view_id))`
  - [x] Add indexes: `(artifact_id)`, `(artifact_id, verified_at)`, `(view_id)`
  - [x] Add table `leaves(artifact_id TEXT, space_kind CHAR(1), space_id TEXT, leaf_idx BIGINT, digest BLOB, PRIMARY KEY(artifact_id, space_kind, space_id, leaf_idx))`
  - [x] Add index on `(artifact_id, space_kind, space_id)`

- [x] Extend `proto/tensorcast/global_store/v1/global_store.proto`
  - [x] `GetArtifactInfoByIdRequest`: options `include_replicas`, `include_leaves`, `oneof space { bool canonical; string view_id; }`, `leaf_idxs`, `include_view_meta`
  - [x] `GetArtifactInfoByIdResponse`: repeated `Leaf{leaf_idx,digest}`, `ViewMeta{view_spec_json,view_size,view_data_hash,verified_at}`
  - [x] `UpdateArtifactViewState(artifact_id, VariantUpsert, repeated LeafWrite)`
  - [x] Regenerate: `bash tools/build_proto_python.sh`

- [x] Implement repositories under `tensorcast/global_store/repositories/`
  - [x] `VariantRepository` with upsert/get
  - [x] `LeafRepository` with batch write and select by `(space_kind, space_id, leaf_idxs)`

- [x] Update servicer `tensorcast/global_store/grpc_service.py`
  - [x] Implement `UpdateArtifactViewState`
  - [x] Extend `GetArtifactInfoById` to return leaves and view meta when requested
  - [x] When `space=view_id` but data absent, respond with `Status.NOT_FOUND` or `PARTIAL_COVERAGE` detail per design

- [x] Tests (Python)
  - [x] Repo: upsert variant, write/read leaves
  - [x] Servicer: end-to-end read/write, error codes
  - [x] Functional: `GetArtifactInfoById(space=view_id)` happy path + canonical fallback + missing leaves error
  - [x] Validate that any attempt to write descriptors with `schema_version!=v3` fails early

# Code Anchors

```24:31:proto/tensorcast/global_store/v1/global_store.proto
// Content-addressed query by artifact_id (mi2:...)
rpc GetArtifactInfoById(GetArtifactInfoByIdRequest) returns (GetArtifactInfoByIdResponse) {}
```

```16:33:tensorcast/global_store/db_utils.py
def _resolve_schema_path() -> str:
    # uses repo-root schema.sql when available
```

```16:55:schema.sql
-- TensorCast canonical schema (schema.sql)
-- ... existing tables: artifacts, artifact_replicas, artifact_indices, chunk_directory, key_mappings
```

# Commands

```bash
bash tools/build_proto_python.sh
uv run ruff check . && uv run ruff format .
uv run mypy ./tensorcast
uv run pytest tests/python/global_store
```

# Risks & Notes

- Leaf digests stored raw (32 bytes); decode on demand at clients.
- Keep GS routing canonical for v1; no ByteSpace-aware chunk directory.
