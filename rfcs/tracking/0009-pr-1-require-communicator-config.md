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
- [x] Update daemon wiring to accept typed config (`StoreDaemonServicer` uses `CommunicationManager.from_config` when `config.communicator` is present).
- [x] Update README to include typed CommunicatorConfig YAML example.

## Acceptance
- [x] All engine paths in core construct via typed `CommunicatorConfig`.
- [x] PR opened and reviewed: https://github.com/tensorcast-ai/tensorcast/pull/49

Notes
- Python CLI injection for passing CommunicatorConfig is provided via `CommunicationManager.from_config` and `StoreDaemonConfig.communicator` (see `tensorcast/store_daemon/config.py`).
