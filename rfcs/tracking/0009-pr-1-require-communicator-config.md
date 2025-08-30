# Tracking Issue — PR-1: Require CommunicatorConfig (flip default)

- Owner: Communicator (C++)
- Milestone: P2 (staged RDMA + ACK)
- Status: Planned
- RFC: rfcs/0009-unified-memory-stager-and-staged-p2p.md (Section 13.9)

## Goals
- Make typed `CommunicatorConfig` mandatory by default for `CommunicateEngine` construction.
- Keep a temporary deprecated shim for legacy constructor that immediately builds a `CommunicatorConfig` (no env reads).

## Tasks
- [ ] Add deprecated shim constructor that builds `CommunicatorConfig` from explicit params only (no env).
- [ ] Emit one-time WARN deprecation log when shim is used.
- [ ] Update daemon and Python wiring to provide `CommunicatorConfig` (file/CLI injection).
- [ ] Update internal callsites in core to pass `CommunicatorConfig`.
- [ ] Add unit tests for failure when no typed config is provided and legacy env gate is off.
- [ ] Update CHANGELOG and README deprecations section to mention the new requirement.

## Acceptance
- [ ] All engine and daemon paths construct via typed `CommunicatorConfig`.
- [ ] CI passes without relying on legacy constructors.

