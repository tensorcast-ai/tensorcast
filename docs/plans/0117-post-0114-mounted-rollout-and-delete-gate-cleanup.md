---
slug: post-0114-mounted-rollout-and-delete-gate-cleanup
title: Post-0114 Mounted Rollout and Delete-Gate Cleanup Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "serving"]
related_code:
  - docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md
  - docs/designs/0115-explicit-source-bound-execution-planning-and-fail-fast-lane-selection.md
  - docs/designs/0112-binding-native-serving-realization-and-publication.md
  - docs/designs/0117-post-0114-mounted-rollout-and-delete-gate-cleanup.md
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/state/lip_manager.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - /data/workspace/internal-vllm/vllm/model_executor/model_loader/tensorcast_loader.py
links:
  design: ../designs/0117-post-0114-mounted-rollout-and-delete-gate-cleanup.md
---

# Objective

Track the remaining post-`0114` rollout and cleanup work without keeping the
audited Step3p5 closure open:

- second-family mounted validation,
- broader mounted operator-visible metrics,
- and delete-gate cleanup after that evidence exists.

# Current State & Grounding

- `0114` is now implemented for the audited Step3p5 path.
- the completed companion plans for `0114` and `0115` have been folded back
  into their implemented designs and deleted, so this `0117` plan is now the
  active rollout and cleanup checklist.
- The real Step3p5 operator runbook is documented and proved:
  daemon `execution_commit` shows `collective_handled=1`,
  `actual_collective_committed_bytes=25550556928`,
  `dominant_executor=OwnerFileCollectiveExecutor`;
  mounted success logs now surface
  `publish_hash_rounds=0`,
  `publish_hash_location=seal`,
  `publish_hash_backend=gpu`,
  `publish_hash_bytes`,
  `publish_hash_wall_time_ms`,
  and `publish_hash_identity_forming=True`;
  `/health` returns `200`;
  `/v1/models` lists the Step3p5 root path.
- There is still no documented local mounted runbook for a second
  representative `BINDING_FINALIZE` family in this environment.
- This was rechecked on 2026-04-10:
  `tensorcast_family_registry.py` still admits only `mixtral` and `step3p5`
  as serving-only `REPRESENTATION_CHANGING` families, but local searches under
  `/mnt/shared-storage`, `/mnt/cj`, `/mnt/step3-alignment`, and
  `/data/models` found no Mixtral checkpoint directory and no `config.json`
  advertising `model_type=mixtral` / `MixtralForCausalLM`.
- `0114` local closure cleanup already landed:
  `internal-vllm` now reports `source_bound_contract_path=collective_first_v4`
  rather than the stale pre-v4 alias, the SDK/internal-vLLM
  publication surface now uses canonical `*_publication` names instead of
  bridge-named helpers, and the repo docs README now treats `0113` as
  historical-only context.
- Transitional bridge logic and delete gates should not be removed until the
  broader rollout evidence exists.

# Phases & Milestones

- [ ] Phase 1: Mounted Rollout Evidence
  - [ ] Milestone 1: document and run a second representative
    `BINDING_FINALIZE` mounted family locally.
  - [ ] Milestone 2: capture mounted collective and hash evidence directly from
    operator-visible surfaces for that family.
- [ ] Phase 2: Evidence Surface Hardening
  - [ ] Milestone 1: surface any remaining mounted executor metrics required
    for operator closure beyond the Step3p5 minimum packet.
  - [ ] Milestone 2: make the mounted evidence packet stable enough that
    delete-gate cleanup can depend on it.
- [ ] Phase 3: Delete-Gate Cleanup
  - [ ] Milestone 1: remove transitional bridge logic and residual-map-based
    gating that only existed to carry 0114 through rollout.
  - [ ] Milestone 2: retire stale active references and historical aliases that
    should no longer survive once the mounted rollout evidence is complete.

# Tasks

- [ ] Select one representative second-family mounted sample with a documented
  local runbook or keep the gate explicitly blocked if no such sample exists.
  Current blocker:
  only Step3p5 is locally runnable in the known mounted roots on 2026-04-10.
- [ ] Capture a mounted closure packet for that family comparable to the
  Step3p5 packet.
- [ ] Audit mounted operator-visible surfaces for any remaining missing
  metrics:
  collective unique-source bytes,
  peer-transfer bytes,
  peak temporary bytes,
  batch count,
  and any residual hash facts not already surfaced.
- [ ] Delete transitional bridge / residual-map cleanup only after the mounted
  evidence packet is stable and proven on both families.
- [ ] Update `docs/README.md` and related historical references when the active
  rollout owner changes again.

# Test / Rollout / Backout

- Planned checks:
  - rerun the real CLI-daemon plus `vllm serve ... tensorcast_init_mode=connect`
    operator path for the chosen second family
  - continue running the targeted repo checks already used in `0114` whenever
    delete-gate cleanup touches daemon/runtime behavior
- Rollout rule:
  - treat second-family absence as an explicit blocker, not as permission to
    substitute an undocumented family ad hoc.
- Backout rule:
  - if cleanup removes evidence or reintroduces ambiguity, revert the cleanup
    commits rather than restoring hidden transitional branches.

# Risks & Tracking

- [ ] Risk: second-family validation stays environment-blocked.
  Mitigation: keep the blocker explicit and documented in the active plan.
- [ ] Risk: mounted evidence remains split across daemon and integration logs in
  a way operators cannot reliably consume.
  Mitigation: harden typed operator-visible surfaces before deleting the
  transitional gates that still depend on them.
- [ ] Risk: delete-gate cleanup accidentally absorbs execution-semantics or
  same-binding closeout-contract work that belongs under `0115` or `0112`.
  Mitigation: keep this plan scoped to rollout evidence and cleanup only.
