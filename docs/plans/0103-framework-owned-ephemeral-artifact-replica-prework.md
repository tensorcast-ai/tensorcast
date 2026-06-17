---
slug: framework-owned-ephemeral-artifact-replica-prework
title: Framework-Owned Ephemeral Artifact Replica Prerequisite Plan
status: implemented
areas: ["daemon", "sdk", "proto", "core", "global_store", "serving", "tests"]
created: 2026-05-16
last_updated: 2026-05-16
related_code:
  - docs/designs/0103-volatile-publication-subjects-and-multi-replica-semantics.md
  - docs/designs/0111-source-to-serving-builder-and-representation-publication.md
  - docs/designs/0112-binding-native-serving-realization-and-publication.md
  - docs/designs/0116-prefetch-serving-binding-target.md
  - proto/tensorcast/common/v1/capability_token.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_controller.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/target_publish_service.cc
  - daemon/state/binding_registry.*
  - daemon/state/target_publication_registry.*
  - daemon/state/lip_manager.*
  - daemon/state/lifecycle_kernel.*
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/serving/preload.py
  - tensorcast/global_store/repositories/replica_repository.py
  - core/store/components/global_store_client.*
links:
  design: ../designs/0103-volatile-publication-subjects-and-multi-replica-semantics.md
  dependencies:
    - ../designs/0084-binding-unified-model-and-contract.md
    - ../designs/0094-unified-lifecycle-kernel-and-capability-families.md
    - ../designs/0111-source-to-serving-builder-and-representation-publication.md
    - ../designs/0112-binding-native-serving-realization-and-publication.md
    - ../designs/0116-prefetch-serving-binding-target.md
---

# Objective

Close the TensorCast prerequisites required before implementing the
vllm framework-owned ephemeral artifact replica plan at:

`/opt/vllm/docs/tensorcast/plans/tensorcast_framework_owned_ephemeral_artifact_replica_plan.md`

This is a TensorCast prework plan only. It does not implement the vLLM
producer/consumer integration. The handoff criterion is that TensorCast can
safely publish an already artifact-backed daemon-owned binding current value as
a routable ephemeral artifact replica, and can fail closed on stale tokens,
retire, owner exit, and direct RPC bypass attempts.

# Implementation Status

Implemented in TensorCast on 2026-05-16. The cutover introduces
`BindingCurrentValuePublicationScope` and
`binding_current_value_publication_token`, removes standalone publish authority
from target materialization responses, binds publish validation to the current
`BindingRegistry` generation, terminalizes publication state on retire and
owner-exit lifecycle cleanup, and adds daemon-side mutation guards for active
published current values. Optional Global Store expiry diagnostics in Phase 6
remain intentionally deferred because expiry is not the safety boundary.

The required semantic boundary is:

- `local_serving_ref` remains local-only recovery metadata.
- local-only values must first become artifact-backed current values through
  promotion or serving closeout.
- `publish_replica()` publishes only an artifact-backed current value.
- The only token that can authorize `PublishTargetReplica` is a
  `BindingCurrentValuePublicationToken`. It proves one concrete daemon-owned
  binding current value generation is still current, artifact-backed, owner
  alive, unretired, and layout/device/selection compatible.
- `retire()` and owner exit terminalize both discovery and publication
  capability.

# Current State And Grounding

Reviewed baseline: current `tensorcast-280` code as of 2026-05-16.

## Existing Foundations

- Daemon-owned bindings already track `binding_id`, `binding_layout_id`,
  `current_binding_value_id`, `seal_generation`, owner pid, target layout hash,
  device UUID, current selection, and
  `binding_current_value_publication_token` in
  `BindingRegistry::Record`.
- `AcquireBindingValue` exists for same-daemon local acquire and validates
  local peer, expected device/member/layout/schema/build metadata, retained
  status, and caller pid in
  `daemon/service/controllers/materialization_controller.cc`.
- `MaterializeIntoTarget` and `MaterializeIntoMappedTarget` no longer mint
  publication authority or insert standalone publication records; binding
  current-value commit/promote paths own that authority.
- `PublishTargetReplica` validates capability-token signature/audience/issuer,
  token scope against request, token scope against
  `TargetPublicationRegistry::Record`, commits LIP routable exports, and
  registers a memory replica in Global Store.
- `RetirePublishedReplica` already marks the GS replica unavailable, waits for
  drain when requested, and revokes the LIP export.
- Global Store source selection already filters worker presence, worker
  heartbeat freshness, availability, capacity, and exportable transport
  metadata.

## Blocking Gaps

These gaps must be closed before the vllm plan starts using the
feature.

1. Local acquire has a snapshot race.
   The local-ref branch validates under the binding record lock, then mints a
   lease, then refreshes the binding record and fills the response from the
   refreshed record. A current-value swap between validation and response fill
   can return a different value than the one validated.

2. Promotion and serving closeout do not return publishable tokens.
   `PromoteBindingCurrentValueResponse` and `BindingPromotionStatus` do not
   carry a `BindingCurrentValuePublicationToken`.
   `promote_binding_current_value_impl()` clears the record token on success.
   Representation closeout also clears the token after committing the
   artifact-backed current value.

3. Publication authority is split between target materialization and binding
   currentness.
   Current materialize-into-target token minting lets a filled target buffer
   become publishable without proving it is still the current value of a
   daemon-owned binding. The vllm plan needs the opposite boundary:
   publication starts from binding lifecycle currentness, not from the
   materialization pipeline.

4. Retired target publication tokens were not binding-generation fenced.
   `TargetPublicationScope` included publication, selection, byte-space, device
   UUID, owner pid, layout hash, and operation id, but not `binding_id`,
   `binding_layout_id`, `binding_value_id`, or `seal_generation`.
   `BindingCurrentValuePublicationScope` is now the only generated publication
   credential scope, so `PublishTargetReplica` proves that the token is still
   for the current binding value.

5. Published current values are not daemon-guarded against mutation.
   Python SDK guards reject mutation while `_published_lease_id` is set, but
   daemon RPCs such as `BeginBindingUpdate` do not check whether the current
   value has an active published replica. Direct RPC callers can bypass SDK
   state.

6. Retire and owner-exit cleanup are not capability-terminal.
   `RetirePublishedReplica` drains and revokes LIP/GS state, but it does not
   erase the target publication record or release the lifecycle publish
   capability. Owner-pid cleanup removes bindings, external handle leases, and
   LIP exports, but it does not terminalize `TargetPublicationRegistry` or the
   publication capability.

7. Global Store is not a correctness boundary for ephemeral lifetime.
   `MemoryInfo.source_process_id` is populated by the C++ client but is not a
   durable source-selection fence. `artifact_replicas.expires_at` exists, but
   ordinary source selection does not filter candidates by expiry. MVP safety
   must rely on explicit daemon unregister, LIP revoke, lifecycle capability
   terminalization, owner-exit cleanup, and stale-token rejection.

# Invariants For Implementers

- Do not make `local_serving_ref` a P2P source authority.
- Do not publish local-only values directly.
- Do not make `publish_replica()` create serving lineage or replace
  `PublishedModelVersion` / representation closeout.
- Do not add SDK direct Global Store access for key mapping, artifact metadata,
  or control-path operations.
- Do not use broad fallback behavior to hide missing token, stale generation,
  owner mismatch, or transport-readiness mismatch. Fail closed with a concrete
  error.
- Do not preserve legacy materialize-into-target publication authority as an
  independent publish semantic. `MaterializeIntoTarget` may materialize bytes
  and may feed a binding transition, but it must not produce a publishable token
  by itself.
- Do not introduce a scope-kind dispatcher that accepts both
  `MATERIALIZE_INTO_TARGET` and `BINDING_CURRENT_VALUE` token families.
  `PublishTargetReplica` has one authority model: binding current value
  publication.
- Rename publish-facing proto/API fields from `target_publication_token` to
  `binding_current_value_publication_token` in the cutover change. A temporary
  internal alias may exist inside one implementation PR, but it is not accepted
  as the final state. Materialize responses must not fill any publish-token
  field.

# Non-Negotiable Risk Controls

These controls are part of the plan, not optional review advice.

- [x] Treat Phase 0 through Phase 4 as one consistency tranche for any
      framework-facing enablement. Individual phases can land separately, but
      no downstream vLLM path may depend on publication until all four pass.
- [x] Keep `PublishTargetReplica` fail-closed by default. Missing binding
      fields, absent current binding value, local-only state, owner mismatch,
      stale generation, retired token, or active mutation must return explicit
      errors, not fallback to materialize-target publish.
- [x] Remove standalone publish authority from `MaterializeIntoTarget` and
      `MaterializeIntoMappedTarget` in the same change set that introduces
      binding-current-value publication validation. Avoid a transition window
      where both token models are accepted.
- [x] Mint publishable tokens only from a daemon-owned binding record snapshot
      captured under lock after the value is artifact-backed current. Never
      mint from a target materialization result alone.
- [x] Clear or invalidate token state on every generation-changing transition
      before exposing the new state to callers. Mint the replacement token only
      after the new current value is committed and revalidated.
- [x] Make lifecycle terminalization a shared helper used by explicit retire,
      owner-exit cleanup, and publish-failure cleanup after partial LIP/GS
      success. One-off cleanup code is not acceptable.
- [x] Mutation guards must be exact to
      `(binding_id, binding_value_id, seal_generation)`. Do not block unrelated
      binding generations or unrelated bindings.
- [x] Do not add ad-hoc feature flags or environment variables to manage this
      rollout. If a configuration knob is needed, place it in the unified
      runtime config system.
- [x] GS expiry or source filtering may improve diagnostics, but it must never
      be the primary safety mechanism for ephemeral publication lifetime.

# Phases And Milestones

- [x] Phase 0: Local Acquire Snapshot Hardening
  - [x] Add a daemon-side local acquire snapshot helper, for example
        `validate_and_snapshot_local_serving_ref(...)`, owned near
        `MaterializationController` or `BindingRegistry`.
  - [x] Validate and copy all response-critical fields under the same binding
        record lock: binding id, value id, seal generation, handle bytes,
        payload descriptors, target index, target layout size, device UUID,
        member, target layout hash, daemon id/session id, tensor schema hash,
        serving build digest, state, and retained/local-owner policy.
  - [x] Fill `AcquireBindingValueResponse` only from that immutable snapshot.
        Do not refetch the record to populate mem handle, payload descriptors,
        target index bytes, current value, or acquired value.
  - [x] Preserve the current retained-binding behavior: retained bindings
        acquire attachment refs; ordinary caller-owned local acquire does not
        pretend to be retained.
  - [x] Keep `AcquireBindingValue` local-only via loopback/UDS checks.
  - [x] Update or add daemon tests for stale `local_serving_ref`, wrong owner
        pid, wrong device UUID, wrong member, wrong layout hash, wrong schema
        hash, wrong serving build digest, and response snapshot stability.

- [x] Phase 1: Canonical Token Name And Propagation After Artifact-Backed Transition
  - [x] Add the canonical token/scope concept:
        `BindingCurrentValuePublicationToken`, encoded as a capability token
        whose scope is a new `BindingCurrentValuePublicationScope`.
        `PublishTargetReplica` must accept only this semantic token.
  - [x] Add or rename the capability audience so binding current value
        publication cannot be confused with legacy target materialization
        publication. Preferred shape:
        `CAPABILITY_AUDIENCE_BINDING_CURRENT_VALUE_PUBLICATION`.
  - [x] Extend `PromoteBindingCurrentValueResponse` with
        `bytes binding_current_value_publication_token`.
  - [x] Extend `BindingPromotionStatus` with
        `bytes binding_current_value_publication_token`.
  - [x] Rename `PublishTargetReplicaRequest.target_publication_token` to
        `binding_current_value_publication_token` and update SDK/C++ callsites.
        Reusing the existing field number is acceptable; accepting old token
        semantics is not.
  - [x] Rename binding-centric response fields that currently use
        `target_publication_token` (`CreateOwnedBindingResponse`,
        `RefillOwnedBindingResponse`, and any binding artifact commit response
        added later) to `binding_current_value_publication_token` in the same
        proto change.
  - [x] Regenerate protobuf code with `bash tools/build_proto_python.sh`.
  - [x] Update `promote_binding_current_value_impl()` so that after the current
        value becomes `BINDING_STATE_READY_ARTIFACT`, the daemon mints a
        `BindingCurrentValuePublicationToken` for that same binding value
        generation.
  - [x] Preserve existing fast-path behavior for already mi2-backed current
        values, but ensure it can return or recreate a valid token when the
        value is publishable.
  - [x] Update async promotion job status propagation so completed jobs carry
        the token through `StartPromoteBindingCurrentValue` and
        `GetBindingPromotionStatus`.
  - [x] Update representation closeout paths in
        `assembly_operation_service.cc` so successful
        `complete_pure_transform_publication_from_binding(...)` and
        `complete_binding_finalize_publication_from_binding(...)` leave the
        binding current value artifact-backed and publish-token capable.
  - [x] Update `OwnedBindingSlot.promote_current_value()`,
        `start_promote_current_value()`, and `get_promotion_status()` to save
        returned `BindingCurrentValuePublicationToken` bytes instead of
        clearing publication-token state on successful promotion.
  - [x] Update `tensorcast/serving/preload.py` helpers to preserve promotion
        token state when polling async promotion.
  - [x] Keep local-only `publish_replica()` failure behavior unchanged when no
        artifact-backed selection or no binding-current-value publication token
        exists.

- [x] Phase 2: Remove Materialize Publish Authority And Bind Tokens To
      `BindingValueRef`
  - [x] Replace publication validation use of retired
        `TargetPublicationScope` with `BindingCurrentValuePublicationScope`.
        The scope must contain:
        `publication_id`, `binding_id`, `binding_layout_id`,
        `binding_value_id`, `seal_generation`, owner pid/session identity,
        device UUID, target layout hash, byte-space, resolved selection, and
        operation id.
  - [x] Extend `TargetPublicationRegistry::Record`, or rename it to a
        binding-current publication registry if that is cleaner, with the same
        binding value identity fields and terminal lifecycle fields.
  - [x] Add a binding-value snapshot type for token minting. The snapshot must
        be captured under the binding record lock and include owner pid, device
        UUID, target layout hash, selection, byte-space, storage layout, and
        binding value identity.
  - [x] Stop minting publishable tokens from `MaterializeIntoTarget` and
        `MaterializeIntoMappedTarget`. Remove/reserve
        `MaterializeIntoTargetResponse.target_publication_token` and delete all
        SDK reads of that response field. The response must not carry bytes
        accepted by `PublishTargetReplica`.
  - [x] Remove standalone target-materialization publication records. A
        publication registry record may be created only from a daemon-owned
        binding current value snapshot.
  - [x] Remove or ignore client-provided target publication authority in
        `CreateBindingRequest.target_publication_token` and
        `CommitBindingArtifactRequest.target_publication_token`. If these RPCs
        commit an artifact-backed binding current value, the daemon mints a new
        binding-current-value token after the commit from the locked current
        value snapshot.
  - [x] Add `BindingRegistry` access to `TargetPublishService::Dep`, or move the
        binding-backed publish validation into a service that can see both
        `TargetPublicationRegistry` and `BindingRegistry`.
  - [x] In `PublishTargetReplica`, validate in this order:
        token signature/audience/issuer/expiry; request byte-space and
        operation id; token scope vs publication record; token scope vs current
        `BindingRegistry` record; current value artifact-backed status; owner
        alive/session match; selection/byte-space/layout/device/owner equality.
  - [x] Reject all tokens that use the old target-materialization publication
        audience/scope, have missing binding fields, or cannot be matched to the
        daemon-owned binding current value.
  - [x] Make begin-update, refill, swap, mark-dirty, mark-ready-local, and
        artifact commit paths clear stale token state for any new generation.
  - [x] Add tests for binding value id mismatch, seal generation mismatch,
        owner pid mismatch, device UUID mismatch, layout mismatch, selection
        mismatch, byte-space mismatch, missing binding fields, old
        materialize-target token rejection, and `MaterializeIntoTarget` no
        longer returning a publishable token.

- [x] Phase 3: Publish, Retire, And Owner-Exit Lifecycle Closure
  - [x] Add terminal publication state to `TargetPublicationRegistry::Record`,
        including consumed/published marker, `published_lease_id`,
        `published_replica_id`, and publish operation id.
  - [x] Make successful `PublishTargetReplica` atomically mark the publication
        record consumed/published after GS registration and LIP commit succeed.
  - [x] Support idempotent replay for the same token and same operation id by
        returning the same lease id and replica id without creating a second
        active replica.
  - [x] Reject the same token with a different operation id after first publish.
  - [x] Add a terminalize/retire method owned by `TargetPublishService` or a
        shared publication lifecycle helper. `RetirePublishedReplica` must call
        it after successful LIP/GS revoke.
  - [x] On retire, erase or terminalize the target publication registry record,
        release the lifecycle publish capability, release the retention lease,
        and make future redemption of the same token fail.
  - [x] Hook owner-exit cleanup into the same terminalization path. The pid
        monitor callback currently reaches lifecycle, region registry, handle
        leases, and binding registry; extend it or a daemon-level coordinator so
        publication registry and lifecycle capability cleanup happen too.
  - [x] Ensure owner-exit cleanup also unregisters any published GS replica and
        revokes local exports. Reuse `LipManager` revoke/sweep behavior, but do
        not rely on it as the only cleanup path.
  - [x] Add race tests for publish vs retire, publish vs owner exit, and
        duplicate publish during async `StartPublishTargetReplica`.

- [x] Phase 4: Daemon-Side Published-Value Mutation Guard
  - [x] Add active published-current state to `BindingRegistry::Record`, or an
        equivalent authoritative guard reachable from all mutation paths.
  - [x] On successful binding-backed publish, mark the exact
        `(binding_id, binding_value_id, seal_generation)` as published.
  - [x] Make `BeginBindingUpdate`, refill, direct artifact commit, freeze,
        seal, dirty transition, close, and other mutation paths reject while an
        active published current value exists. Error message should instruct
        callers to call `retire()` first.
  - [x] Clear the guard only after successful retire or owner-exit cleanup.
  - [x] Add direct RPC tests that bypass Python SDK and prove mutation is
        rejected while published.

- [x] Phase 5: SDK And Serving Readiness Handoff
  - [x] Update Python error messages for `OwnedBindingSlot.publish_replica()`:
        non artifact-backed current value, missing token, stale token, already
        published, and retire-required mutation should each be actionable.
  - [x] Update `InplaceSlot` and artifact materialization convenience APIs so
        `MaterializeIntoTarget` no longer implies publishability. They must
        either route publish through a daemon-owned binding current value or
        fail with an actionable error.
  - [x] Ensure `owner_pid=None` paths use the runtime/effective pid helper
        consistently instead of high-level framework `os.getpid()` shortcuts.
  - [x] Add SDK tests for promotion token save, async promotion status token
        save, local-only publish failure, materialize-only publish rejection,
        publish after promotion success, and mutation rejection while
        published.
  - [x] Add a TensorCast-side readiness contract note for vLLM handoff:
        default health must not wait for async publish; explicit P2P-ready mode
        must fail closed if publication fails.
  - [x] Do not start vllm integration until Phase 0 through Phase 4 are
        accepted and tested.

: Optional GS Diagnostics Only
  - [ ] Decide whether to persist `MemoryInfo.expires_at` into
        `artifact_replicas.expires_at` for memory replica registration.
  - [ ] If implemented, filter expired memory replicas from source selection and
        add diagnostics explaining expiry rejection.
  - [ ] Do not treat GS expiry as an MVP safety boundary. Retire and owner-exit
        terminal cleanup remain required even if expiry filtering exists.
  - [ ] Do not add binding generation fields to the main GS schema for MVP. Use
        logs/spans/diagnostics if debugging needs them.

# Test Plan

Run tests with the repository-required commands and environment.

## Proto

- [x] After any `.proto` change:

```bash
bash tools/build_proto_python.sh
```

## C++ / Daemon

- [x] Local acquire hardening:

```bash
bazel test //daemon:grpc_service_impl_operation_rpc_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
```

- [x] Publish token scope, old materialize-token rejection, stale token,
      duplicate publish, and lifecycle:

```bash
bazel test //daemon:grpc_service_impl_publish_target_replica_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
```

  Required coverage in this target:
  binding value id mismatch, seal generation mismatch, owner/session mismatch,
  device/layout/selection/byte-space mismatch, removed
  `TargetPublicationScope`/target-materialization token authority,
  `MaterializeIntoTarget` response has no publishable token, duplicate same-op
  idempotence, duplicate different-op rejection, and published-value terminal
  cleanup.

- [x] Retire cleanup:

```bash
bazel test //daemon:grpc_service_impl_retire_published_replica_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
```

- [x] Owned binding promotion and mutation guards:

```bash
bazel test //daemon:owned_binding_service_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
```

- [x] Lifecycle / owner-exit integration:

```bash
bazel test //daemon:session_lifecycle_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
```

- [x] Final daemon build gate:

```bash
bazel build //daemon:tensorcast_daemon \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
```

## Python SDK

- [x] Activate the venv before running Python commands:

```bash
source .venv/bin/activate
pytest tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/test_serving_preload_acquire.py
```

- [x] Lint and format touched Python files:

```bash
source .venv/bin/activate
ruff check tensorcast/api/store tensorcast/serving tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/test_serving_preload_acquire.py
ruff format tensorcast/api/store tensorcast/serving tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/test_serving_preload_acquire.py
```

# Acceptance Gates

- [x] A stale `local_serving_ref` acquire fails or returns the exact old
      validated snapshot; it never validates one value and returns another.
- [x] Local-only current values still cannot be published.
- [x] Promotion success returns and saves a
      `BindingCurrentValuePublicationToken` for the same binding value
      generation.
- [x] Serving representation closeout success leaves the binding current value
      publish-token capable without redefining `publish_replica()` as serving
      lineage publication.
- [x] `PublishTargetReplica` accepts only
      `BindingCurrentValuePublicationToken` semantics and rejects
      materialize-target tokens, missing binding fields, stale
      `binding_value_id`, stale `seal_generation`, wrong owner pid, wrong
      device UUID, wrong layout hash, wrong selection, and wrong byte-space.
- [x] `MaterializeIntoTarget` and `MaterializeIntoMappedTarget` still
      materialize bytes correctly, but no longer return a publishable token or
      create standalone publication authority.
- [x] Duplicate same-token same-operation publish returns the same result;
      duplicate same-token different-operation publish fails.
- [x] Retire removes GS discovery, revokes local export, terminalizes the
      target publication record, releases lifecycle capability, and rejects
      same-token republish.
- [x] Owner exit removes GS discovery, revokes local export, terminalizes the
      publication record/capability, and rejects same-token republish.
- [x] Direct daemon RPC mutation paths reject while the current binding value is
      actively published.
- [x] Existing materialize-into-target publish tests are rewritten to establish
      a daemon-owned binding current artifact-backed value before publishing;
      no test relies on target materialization as publication authority.
- [x] SDK publish and retire behavior is covered by Python tests.

 For vllm work

Only start the vllm producer/consumer work after all of the following
are true:

- Phase 0 through Phase 4 are implemented, reviewed, and tested.
- A TensorCast daemon-owned binding can be frozen local-ready, promoted to
  artifact-backed current value, published, discovered through Global Store,
  retired, and then rejected on same-token republish.
- A direct RPC attempt to mutate the published current value fails before any
  bytes are overwritten.
- An owner-exit test proves new consumers cannot silently read the old source.
- Python SDK exposes a stable, documented sequence:
  `promote_current_value()` or serving closeout, then `publish_replica()`, then
  `retire()`.
- Diagnostics distinguish at least these rejection families: missing token,
  local-only value, stale binding generation, owner mismatch, retired token,
  active published mutation, and no eligible source.

# Rollout And Backout

- Land Phase 0 independently. It should not change public publish behavior.
- Land proto/token propagation in Phase 1 with regenerated stubs and SDK tests.
- Land Phase 2 as the semantic cutover: remove materialize-target publish
  authority and require `BindingCurrentValuePublicationToken` validation in
  `PublishTargetReplica` in the same change set.
- Land lifecycle terminal cleanup and mutation guards before enabling any
  framework integration.
- Keep the vllm optional publish path disabled by default until
  TensorCast gates pass.
- Backout strategy for Phase 2 is to disable binding-current-value publication
  entirely and keep ordinary materialization working. Do not back out by
  re-enabling standalone materialize-target publish tokens.

# Risks And Tracking

- [ ] Token mint timing: minting before the binding current value is stable can
      issue a token for the wrong generation. Always mint from a locked snapshot
      or revalidate immediately after minting before storing the token.
- [ ] Retire race: LIP/GS revoke without target publication terminalization can
      permit republish within token TTL. Retire must call one shared terminal
      cleanup path.
- [ ] Owner-exit race: pid cleanup currently spans several subsystems. Add tests
      that kill or simulate the owner while publish is pending and while publish
      is active.
- [ ] Materialize cutover risk: `MaterializeIntoTarget`, `InplaceSlot`, and
      artifact convenience APIs may currently assume token-only publication.
      Update those SDK paths to either create/use a daemon-owned binding current
      value before publish or fail with an actionable "publish requires binding
      current value" error. Do not keep a legacy token branch.
- [ ] Dual-scope drift risk: any lingering validation branch that accepts old
      target materialization scope will become a future consistency hole. Add a
      negative test that mints an old-style token and proves
      `PublishTargetReplica` rejects it.
- [ ] GS expiry temptation: expiry filtering is useful diagnostics, but it must
      not replace explicit unregister/revoke/capability cleanup.
- [ ] vLLM readiness coupling: do not let optional producer publish extend the
      default health critical path. P2P-required mode can wait and fail closed;
      default serving health cannot.

# References

- Target design owner: `docs/designs/0103-volatile-publication-subjects-and-multi-replica-semantics.md`
- Binding current value semantics: `docs/designs/0084-binding-unified-model-and-contract.md`
- Serving publication trunk: `docs/designs/0111-source-to-serving-builder-and-representation-publication.md`
- Binding-native serving boundary: `docs/designs/0112-binding-native-serving-realization-and-publication.md`
- Local-ready acquire contract: `docs/designs/0116-prefetch-serving-binding-target.md`
- vllm downstream plan:
  `/opt/vllm/docs/tensorcast/plans/tensorcast_framework_owned_ephemeral_artifact_replica_plan.md`
