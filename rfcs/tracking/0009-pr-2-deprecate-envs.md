# Tracking Issue — PR-2: Remove legacy environment variables

- Owner: Communicator (C++)
- Milestone: P2 (staged RDMA + ACK)
- Status: Completed
- RFC: rfcs/0009-unified-memory-stager-and-staged-p2p.md (Section 13.9)

## Goals
- Remove legacy env-based configuration entirely from engine codepaths.
- Require typed `CommunicatorConfig` everywhere.

## Env mapping (doc-only)
- Previously used envs (now unsupported): `DEFAULT_DEV`, `GPU_TCP_STAGER_CHUNK_SIZE_MB`, `GPU_TCP_STAGER_NUM_BUFFERS`,
  `GPU_TCP_RECV_NUM_BUFFERS`, `RDMA_ACK_TTL_MS`, `STAGER_NUMA_ENABLE`, `STAGER_NUMA_GPU_MAP`, `STAGER_NUMA_NIC_MAP`.
  Use typed config fields instead (see CommunicatorConfig schema in RFC-0009).

## Tasks
- [x] Remove all direct env reads from `engine.cc`.
- [x] Delete any deprecated env loaders and gates.
- [x] Update tests and callsites to construct via `CommunicatorConfig`.
- [x] Update docs to show typed config only.

## Acceptance
- [x] Engine compiles with no env reads; typed config required.
- [x] No occurrence of legacy constructor in repository.
 - [x] PR updated to include env removal: https://github.com/tensorcast-ai/tensorcast/pull/49

Artifacts
- Code: removed `core/communicator/misc/envs.*`; migrated RDMA/TCP knobs to `CommunicatorConfig` and engine wiring.
- Docs: updated communicator README, migration guide, and root README with typed examples.
