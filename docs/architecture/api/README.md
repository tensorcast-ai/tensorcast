---
title: API Architecture
description: Public API design and internal API flows for TensorCast
---

# API Architecture

This section documents the public SDK surface and the internal API flows that
implement registration, materialization, policy, persistence, and region-backed
operations. The content is aligned to current code, not legacy design docs.
These docs are the canonical reference for API behavior going forward.

## Reading Start Points

- Application user: `./api-design.md`
- SDK maintainer: `./api-design.md` then `./registration-flow.md` and
  `./materialization-flow.md`
- Daemon engineer: `./registration-flow.md`, `./policy-persistence.md`,
  `./region-backed.md`
- Global Store engineer: `./policy-persistence.md`
- Core Store engineer: `./registration-flow.md` and `./policy-persistence.md`
- Observability or on-call: `./error-retry-observability.md`

## Document Map

- `./api-design.md`
- `./registration-flow.md`
- `./materialization-flow.md`
- `./policy-persistence.md`
- `./region-backed.md`
- `./error-retry-observability.md`

## Related Module Docs

- `../../../tensorcast/api/README.md`
- `../../../tensorcast/api/store/README.md`
- `../../../daemon/README.md`
- `../../../tensorcast/global_store/README.md`
- `../../../core/store/README.md`
