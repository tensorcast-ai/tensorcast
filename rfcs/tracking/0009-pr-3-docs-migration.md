# Tracking Issue — PR-3: Docs update to typed CommunicatorConfig only

- Owner: Docs
- Milestone: P2
- Status: Planned
- RFC: rfcs/0009-unified-memory-stager-and-staged-p2p.md (Section 13.9)

## Goals
- Replace env-based configuration guidance with typed `CommunicatorConfig` examples.
- Add a migration guide with field mapping and example YAML.

## Tasks
- [ ] Add developer guide: `developer-guides/core/communicator/communicator-config-migration.md`.
- [ ] Update `web-docs/sidebars.ts` to include the new page.
- [ ] Remove or rewrite env-based config mentions in docs and README.
- [ ] Provide before/after snippets and failure modes (gate behavior).

## Acceptance
- [ ] Docs build passes; sidebar shows the migration guide under Communicator.
- [ ] README points to the new guide; no env-based examples remain.

