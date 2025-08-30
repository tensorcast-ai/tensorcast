# Tracking Issue — PR-3: Docs update to typed CommunicatorConfig only

- Owner: Docs
- Milestone: P2
- Status: Completed
- RFC: rfcs/0009-unified-memory-stager-and-staged-p2p.md (Section 13.9)

## Goals
- Replace env-based configuration guidance with typed `CommunicatorConfig` examples.
- Add a migration guide with field mapping and example YAML.

## Tasks
- [x] Add developer guide: `developer-guides/core/communicator/communicator-config-migration.md`.
- [x] Update `web-docs/sidebars.ts` to include the new page.
- [x] Remove or rewrite env-based config mentions in docs and README.
- [x] Provide typed YAML examples (include rdma/transport knobs).

## Acceptance
- [x] Migration guide present and linked from sidebar.
- [x] README includes a typed CommunicatorConfig YAML example; env-based examples removed.
- [x] PR updated with docs changes: https://github.com/tensorcast-ai/tensorcast/pull/49
