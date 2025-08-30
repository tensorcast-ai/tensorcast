# Tracking Issue — PR-1: Require CommunicatorConfig (flip default)

- Owner: Communicator (C++)
- Milestone: P2 (staged RDMA + ACK)
- Status: Completed
- RFC: rfcs/0009-unified-memory-stager-and-staged-p2p.md (Section 13.9)

## Goals
- Make typed `CommunicatorConfig` mandatory by default for `CommunicateEngine` construction.
- Keep a temporary deprecated shim for legacy constructor that immediately builds a `CommunicatorConfig` (no env reads).

## Tasks
- [x] Remove legacy bool constructor; require typed `CommunicatorConfig`.
- [x] Update internal callsites in core to pass `CommunicatorConfig`.
- [ ] Update daemon and Python wiring to provide `CommunicatorConfig` (file/CLI injection).
- [ ] Update CHANGELOG and README deprecations section to mention the new requirement.

## Acceptance
- [x] All engine paths in core construct via typed `CommunicatorConfig`.
- [ ] CI passes without relying on legacy constructors.
