use clap_cache_core::{
    CacheManager, Capabilities, Commit, Config, Error, Labels, Namespace, Operation, Plan,
    PlanRequest, Priority, RetentionConfig, Scope, SlotCapabilities, SlotState,
};

fn namespace(byte: u8) -> Namespace {
    Namespace([byte; 32])
}

fn labels(session: u64) -> Labels {
    Labels {
        tenant: 1,
        project: 2,
        harness: 3,
        agent: 4,
        session,
        scope: Scope::Session,
        priority: Priority::Interactive,
        side_request: false,
    }
}

fn manager(slots: u32) -> CacheManager {
    CacheManager::new(
        Config {
            slot_count: slots,
            min_reuse_tokens: 2,
            logical_token_capacity: usize::MAX,
            max_anchors: slots,
            automatic_checkpoints: Default::default(),
        },
        fixed_retention(slots),
    )
    .unwrap()
}

fn fixed_retention(slots: u32) -> RetentionConfig {
    RetentionConfig {
        hard_max_retained_entries: slots,
        physical_byte_budget: None,
        high_watermark_bytes: 0,
        low_watermark_bytes: 0,
    }
}

fn request<'a>(
    tokens: &'a [i32],
    namespace: Namespace,
    session: u64,
    capabilities: u64,
) -> PlanRequest<'a> {
    PlanRequest {
        namespace,
        tokens,
        stable_boundaries: &[],
        labels: labels(session),
        capabilities: Capabilities(capabilities),
        slot_capabilities: None,
        output_reserve: 0,
        estimated_bytes_per_token: 16,
        result_state: SlotState::Session,
    }
}

fn commit_idle(manager: &mut CacheManager, plan: &Plan, len: usize, state: SlotState) {
    manager
        .commit(
            plan.id,
            Commit {
                resident_tokens: len,
                actual_state: state,
                physical_bytes: len as u64 * 16,
                prefill_us_saved: plan.reuse_tokens as u64 * 10,
            },
        )
        .unwrap();
    let snapshot = manager.slot(plan.target.slot).unwrap();
    if snapshot.busy {
        manager
            .set_busy(snapshot.id, snapshot.generation, false)
            .unwrap();
    }
}

fn materialize_anchor(
    manager: &mut CacheManager,
    tokens: &[i32],
    name_space: Namespace,
    scope: Scope,
) -> u32 {
    let mut anchor_request = request(tokens, name_space, 0, 0);
    anchor_request.result_state = SlotState::Anchor;
    anchor_request.labels.scope = scope;
    let anchor = manager.plan(anchor_request).unwrap();
    let slot = anchor.target.slot;
    commit_idle(manager, &anchor, tokens.len(), SlotState::Anchor);
    slot
}

#[test]
fn fresh_then_exact_partial_branch_uses_longest_prefix() {
    let mut cache = manager(3);
    let original = [1, 2, 3, 4, 5];
    let seed = cache.plan(request(&original, namespace(1), 10, 0)).unwrap();
    assert_eq!(seed.operation, Operation::Fresh);
    commit_idle(&mut cache, &seed, original.len(), SlotState::Session);

    let branch_tokens = [1, 2, 3, 9, 10];
    let branch = cache
        .plan(request(
            &branch_tokens,
            namespace(1),
            11,
            Capabilities::PARTIAL_PREFIX_BRANCH,
        ))
        .unwrap();
    assert_eq!(branch.operation, Operation::Branch);
    assert_eq!(branch.reuse_tokens, 3);
    assert_ne!(branch.target.slot, branch.donor.as_ref().unwrap().slot);

    let decision = cache
        .commit(
            branch.id,
            Commit {
                resident_tokens: branch_tokens.len(),
                actual_state: SlotState::Session,
                physical_bytes: 80,
                prefill_us_saved: 30,
            },
        )
        .unwrap();
    assert!(decision.hit);
    assert_eq!(decision.realized_reuse_tokens, 3);
    assert_eq!(cache.telemetry().read_leases, 0);
    assert_eq!(cache.telemetry().write_leases, 0);
}

#[test]
fn exact_match_plans_only_the_backend_materializable_prefix() {
    let mut cache = manager(2);
    let tokens = [1, 2, 3, 4, 5];
    let seed = cache.plan(request(&tokens, namespace(1), 10, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);

    let plan = cache
        .plan(request(
            &tokens,
            namespace(1),
            11,
            Capabilities::WHOLE_STATE_COPY
                | Capabilities::PARTIAL_PREFIX_BRANCH
                | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS,
        ))
        .unwrap();
    assert_eq!(plan.operation, Operation::Branch);
    assert_eq!(plan.reuse_tokens, tokens.len() - 1);

    let decision = cache
        .commit(
            plan.id,
            Commit {
                resident_tokens: tokens.len() - 1,
                actual_state: SlotState::Session,
                physical_bytes: 64,
                prefill_us_saved: 40,
            },
        )
        .unwrap();
    assert_eq!(decision.planned_reuse_tokens, tokens.len() - 1);
    assert_eq!(decision.realized_reuse_tokens, tokens.len() - 1);
}

#[test]
fn namespace_isolation_prevents_reuse_of_identical_tokens() {
    let mut cache = manager(2);
    let tokens = [7, 8, 9];
    let seed = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);

    let isolated = cache
        .plan(request(
            &tokens,
            namespace(2),
            2,
            Capabilities::WHOLE_STATE_COPY,
        ))
        .unwrap();
    assert_eq!(isolated.operation, Operation::Fresh);
    assert_eq!(isolated.reuse_tokens, 0);
    assert!(isolated.donor.is_none());
    cache.abort(isolated.id).unwrap();
}

#[test]
fn same_session_continues_in_place_but_anchor_never_does() {
    let mut cache = manager(3);
    let tokens = [1, 2, 3];
    let seed = cache.plan(request(&tokens, namespace(1), 42, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);

    let continued = [1, 2, 3, 4];
    let plan = cache
        .plan(request(&continued, namespace(1), 42, 0))
        .unwrap();
    assert_eq!(plan.operation, Operation::Continue);
    assert_eq!(plan.target.slot, seed.target.slot);
    commit_idle(&mut cache, &plan, continued.len(), SlotState::Session);

    let anchor_tokens = [20, 21, 22];
    let mut anchor_request = request(&anchor_tokens, namespace(1), 99, 0);
    anchor_request.result_state = SlotState::Anchor;
    let anchor = cache.plan(anchor_request).unwrap();
    commit_idle(&mut cache, &anchor, anchor_tokens.len(), SlotState::Anchor);

    let anchor_extension = [20, 21, 22, 23];
    let restore = cache
        .plan(request(
            &anchor_extension,
            namespace(1),
            99,
            Capabilities::WHOLE_STATE_COPY,
        ))
        .unwrap();
    assert_eq!(restore.operation, Operation::Restore);
    assert_ne!(restore.target.slot, anchor.target.slot);
}

#[test]
fn unspecified_session_continues_by_token_affinity_but_explicit_sessions_branch() {
    let mut cache = manager(3);
    let tokens = [1, 2, 3];
    let seed = cache.plan(request(&tokens, namespace(1), 0, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);
    let continued = [1, 2, 3, 4];
    let affine = cache.plan(request(&continued, namespace(1), 0, 0)).unwrap();
    assert_eq!(affine.operation, Operation::Continue);
    commit_idle(&mut cache, &affine, continued.len(), SlotState::Session);

    let explicit = cache
        .plan(request(
            &[1, 2, 3, 4, 5],
            namespace(1),
            9,
            Capabilities::PARTIAL_PREFIX_BRANCH,
        ))
        .unwrap();
    assert_eq!(explicit.operation, Operation::Branch);
    assert_ne!(explicit.target.slot, affine.target.slot);
}

#[test]
fn short_side_request_evicts_an_older_side_slot_not_recent_primary() {
    let mut cache = manager(2);
    let primary_tokens: Vec<i32> = (0..16_000).collect();
    let primary = cache
        .plan(request(&primary_tokens, namespace(1), 0, 0))
        .unwrap();
    commit_idle(
        &mut cache,
        &primary,
        primary_tokens.len(),
        SlotState::Session,
    );
    let mut old_side_request = request(&[90, 91], namespace(1), 7, 0);
    old_side_request.labels.side_request = true;
    old_side_request.labels.priority = Priority::Background;
    let old_side = cache.plan(old_side_request).unwrap();
    commit_idle(&mut cache, &old_side, 2, SlotState::Session);

    let mut title_request = request(&[92, 93], namespace(1), 8, 0);
    title_request.labels.side_request = true;
    title_request.labels.priority = Priority::Background;
    let title = cache.plan(title_request).unwrap();
    assert_eq!(title.target.slot, old_side.target.slot);
    assert_ne!(title.target.slot, primary.target.slot);
}

#[test]
fn saturated_anchor_pool_evicts_an_idle_anchor_instead_of_returning_no_capacity() {
    let mut cache = manager(3);
    for suffix in 0..3 {
        let tokens = [30, suffix];
        let mut anchor_request = request(&tokens, namespace(1), suffix as u64 + 1, 0);
        anchor_request.result_state = SlotState::Anchor;
        let anchor = cache.plan(anchor_request).unwrap();
        commit_idle(&mut cache, &anchor, tokens.len(), SlotState::Anchor);
    }

    let fresh_tokens = [90, 91, 92];
    let fresh = cache
        .plan(request(&fresh_tokens, namespace(2), 44, 0))
        .unwrap();
    assert_eq!(fresh.operation, Operation::Fresh);
    assert_eq!(fresh.reuse_tokens, 0);
    assert_eq!(fresh.evictions.len(), 1);
}

#[test]
fn exact_anchor_creation_is_a_namespace_scoped_noop_and_capacity_is_replaceable() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 2,
            min_reuse_tokens: 2,
            logical_token_capacity: usize::MAX,
            max_anchors: 1,
            automatic_checkpoints: Default::default(),
        },
        fixed_retention(2),
    )
    .unwrap();
    let tokens = [30, 31, 32];
    let mut first_request = request(&tokens, namespace(1), 1, 0);
    first_request.result_state = SlotState::Anchor;
    let first = cache.plan(first_request).unwrap();
    commit_idle(&mut cache, &first, tokens.len(), SlotState::Anchor);
    let first_snapshot = cache.slot(first.target.slot).unwrap();

    let mut duplicate_request = request(&tokens, namespace(1), 2, 0);
    duplicate_request.result_state = SlotState::Anchor;
    let duplicate = cache.plan(duplicate_request).unwrap();
    assert_eq!(duplicate.operation, Operation::Noop);
    assert_eq!(duplicate.target.slot, first.target.slot);
    assert!(duplicate.evictions.is_empty());
    let decision = cache
        .commit(
            duplicate.id,
            Commit {
                resident_tokens: tokens.len(),
                actual_state: SlotState::Anchor,
                physical_bytes: 0,
                prefill_us_saved: 0,
            },
        )
        .unwrap();
    assert_eq!(decision.operation, Operation::Noop);
    assert_eq!(cache.slot(first.target.slot).unwrap(), first_snapshot);
    assert_eq!(cache.telemetry().anchors, 1);

    let mut isolated_request = request(&tokens, namespace(2), 3, 0);
    isolated_request.result_state = SlotState::Anchor;
    let replacement = cache.plan(isolated_request).unwrap();
    assert_eq!(replacement.operation, Operation::Fresh);
    assert_eq!(replacement.evictions.len(), 1);
    assert_eq!(replacement.evictions[0].slot, first.target.slot);
    cache.abort(replacement.id).unwrap();
}

#[test]
fn long_harness_output_reserve_does_not_discard_a_legal_donor() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 2,
            min_reuse_tokens: 2,
            logical_token_capacity: 128,
            max_anchors: 2,
            automatic_checkpoints: Default::default(),
        },
        fixed_retention(2),
    )
    .unwrap();
    let harness: Vec<i32> = (0..32).collect();
    let first = cache
        .plan(request(
            &harness,
            namespace(1),
            10,
            Capabilities::PARTIAL_PREFIX_BRANCH,
        ))
        .unwrap();
    commit_idle(&mut cache, &first, harness.len(), SlotState::Session);

    let unrelated = [90, 91, 92, 93];
    let other = cache
        .plan(request(&unrelated, namespace(2), 20, 0))
        .unwrap();
    commit_idle(&mut cache, &other, unrelated.len(), SlotState::Session);

    let mut continued = harness.clone();
    continued.extend([40, 41, 42, 43]);
    let mut request = request(
        &continued,
        namespace(1),
        11,
        Capabilities::PARTIAL_PREFIX_BRANCH,
    );
    request.output_reserve = 128 - continued.len();
    let hit = cache.plan(request).unwrap();
    assert_eq!(hit.operation, Operation::Branch);
    assert!(hit.donor.is_some());
    assert_eq!(hit.reuse_tokens, harness.len());
    commit_idle(&mut cache, &hit, continued.len(), SlotState::Session);

    let telemetry = cache.telemetry();
    assert_eq!(telemetry.read_leases, 0);
    assert_eq!(telemetry.write_leases, 0);
}

#[test]
fn mixed_trimmability_keeps_a_trimmable_donor_eligible() {
    let mut cache = manager(3);
    let nontrimmable_tokens = [1, 2, 3, 4];
    let nontrimmable = cache
        .plan(request(&nontrimmable_tokens, namespace(1), 1, 0))
        .unwrap();
    commit_idle(
        &mut cache,
        &nontrimmable,
        nontrimmable_tokens.len(),
        SlotState::Session,
    );
    let trimmable_tokens = [1, 2, 9, 9];
    let trimmable = cache
        .plan(request(&trimmable_tokens, namespace(1), 2, 0))
        .unwrap();
    commit_idle(
        &mut cache,
        &trimmable,
        trimmable_tokens.len(),
        SlotState::Session,
    );

    let mut slots = [SlotCapabilities(0); 3];
    slots[nontrimmable.target.slot as usize] = SlotCapabilities(
        SlotCapabilities::MATERIALIZED | SlotCapabilities::WRITABLE | SlotCapabilities::COPY,
    );
    slots[trimmable.target.slot as usize] = SlotCapabilities::ALL;
    let empty = (0..3)
        .find(|&slot| slot != nontrimmable.target.slot && slot != trimmable.target.slot)
        .unwrap();
    slots[empty as usize] = SlotCapabilities(SlotCapabilities::WRITABLE);

    let requested = [1, 2, 8, 8];
    let mut incoming = request(
        &requested,
        namespace(1),
        3,
        Capabilities::PARTIAL_PREFIX_BRANCH,
    );
    incoming.slot_capabilities = Some(&slots);
    let plan = cache.plan(incoming).unwrap();
    assert_eq!(plan.operation, Operation::Branch);
    assert_eq!(plan.donor.unwrap().slot, trimmable.target.slot);
    assert_eq!(plan.reuse_tokens, 2);
}

#[test]
fn excluded_candidate_replans_fresh_without_a_ghost_donor() {
    let mut cache = manager(2);
    let seed_tokens = [1, 2, 3, 4];
    let seed = cache
        .plan(request(&seed_tokens, namespace(1), 1, 0))
        .unwrap();
    commit_idle(&mut cache, &seed, seed_tokens.len(), SlotState::Session);

    let requested = [1, 2, 3, 9];
    let mut slots = [SlotCapabilities(SlotCapabilities::WRITABLE); 2];
    slots[seed.target.slot as usize] =
        SlotCapabilities(SlotCapabilities::MATERIALIZED | SlotCapabilities::WRITABLE);
    let mut incoming = request(
        &requested,
        namespace(1),
        2,
        Capabilities::PARTIAL_PREFIX_BRANCH,
    );
    incoming.slot_capabilities = Some(&slots);
    let plan = cache.plan(incoming).unwrap();
    assert_eq!(plan.operation, Operation::Fresh);
    assert_eq!(plan.reuse_tokens, 0);
    assert!(plan.donor.is_none());
}

#[test]
fn no_writable_target_returns_backpressure_instead_of_reusing_a_donor() {
    let mut cache = manager(2);
    let seed_tokens = [1, 2, 3, 4];
    let seed = cache
        .plan(request(&seed_tokens, namespace(1), 1, 0))
        .unwrap();
    commit_idle(&mut cache, &seed, seed_tokens.len(), SlotState::Session);

    let mut slots = [SlotCapabilities(0); 2];
    slots[seed.target.slot as usize] = SlotCapabilities(
        SlotCapabilities::MATERIALIZED
            | SlotCapabilities::PARTIAL_SUFFIX_TRIM
            | SlotCapabilities::COPY,
    );
    let requested = [1, 2, 3, 9];
    let mut incoming = request(
        &requested,
        namespace(1),
        2,
        Capabilities::PARTIAL_PREFIX_BRANCH,
    );
    incoming.slot_capabilities = Some(&slots);
    assert_eq!(cache.plan(incoming), Err(Error::NoCapacity));
}

#[test]
fn all_writable_targets_busy_returns_retryable_slot_busy() {
    let mut cache = manager(2);
    for session in 1..=2 {
        let tokens = [session as i32, 10 + session as i32];
        let plan = cache
            .plan(request(&tokens, namespace(1), session, 0))
            .unwrap();
        cache
            .commit(
                plan.id,
                Commit {
                    resident_tokens: tokens.len(),
                    actual_state: SlotState::Session,
                    physical_bytes: 0,
                    prefill_us_saved: 0,
                },
            )
            .unwrap();
    }
    assert_eq!(
        cache.plan(request(&[99, 100], namespace(1), 3, 0)),
        Err(Error::SlotBusy)
    );
    let slot = cache.slot(0).unwrap();
    cache.set_busy(slot.id, slot.generation, false).unwrap();
    let retry = cache.plan(request(&[99, 100], namespace(1), 3, 0)).unwrap();
    assert_eq!(retry.operation, Operation::Fresh);
    cache.abort(retry.id).unwrap();
}

#[test]
fn exact_anchor_restore_survives_a_busy_primary_and_namespace_stays_isolated() {
    let mut cache = manager(4);
    let boundary = [1, 2, 3, 4];
    let mut anchor_request = request(&boundary, namespace(1), 10, 0);
    anchor_request.result_state = SlotState::Anchor;
    let anchor = cache.plan(anchor_request).unwrap();
    commit_idle(&mut cache, &anchor, boundary.len(), SlotState::Anchor);

    let primary_tokens = [1, 2, 3, 4, 5, 6];
    let primary = cache
        .plan(request(
            &primary_tokens,
            namespace(1),
            11,
            Capabilities::WHOLE_STATE_COPY,
        ))
        .unwrap();
    let primary_decision = cache
        .commit(
            primary.id,
            Commit {
                resident_tokens: boundary.len(),
                actual_state: SlotState::Session,
                physical_bytes: 0,
                prefill_us_saved: 0,
            },
        )
        .unwrap();
    let primary_generation = cache
        .advance(
            primary_decision.target_slot,
            cache.slot(primary_decision.target_slot).unwrap().generation,
            &primary_tokens[boundary.len()..],
            SlotState::Session,
            true,
            0,
        )
        .unwrap();
    assert_eq!(
        cache.slot(primary_decision.target_slot).unwrap().generation,
        primary_generation
    );

    let side_tokens = [1, 2, 3, 4, 9];
    let side = cache
        .plan(request(
            &side_tokens,
            namespace(1),
            12,
            Capabilities::WHOLE_STATE_COPY | Capabilities::SAFE_BUSY_DONOR,
        ))
        .unwrap();
    assert_eq!(side.operation, Operation::Restore);
    assert_eq!(side.reuse_tokens, boundary.len());
    cache.abort(side.id).unwrap();

    let isolated = cache
        .plan(request(
            &side_tokens,
            namespace(2),
            13,
            Capabilities::WHOLE_STATE_COPY | Capabilities::SAFE_BUSY_DONOR,
        ))
        .unwrap();
    assert_eq!(isolated.operation, Operation::Fresh);
    assert_eq!(isolated.reuse_tokens, 0);
}

#[test]
fn unsupported_partial_copy_falls_back_to_fresh_without_false_hit() {
    let mut cache = manager(2);
    let seed_tokens = [1, 2, 3, 4];
    let seed = cache
        .plan(request(&seed_tokens, namespace(1), 1, 0))
        .unwrap();
    commit_idle(&mut cache, &seed, seed_tokens.len(), SlotState::Session);

    let requested = [1, 2, 8, 9];
    let plan = cache
        .plan(request(
            &requested,
            namespace(1),
            2,
            Capabilities::WHOLE_STATE_COPY,
        ))
        .unwrap();
    assert_eq!(plan.operation, Operation::Fresh);
    assert_eq!(plan.reuse_tokens, 0);
    assert!(plan.donor.is_none());
}

#[test]
fn busy_donor_requires_explicit_capability_and_is_never_mutated() {
    let mut cache = manager(3);
    let tokens = [1, 2, 3];
    let seed = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    cache
        .commit(
            seed.id,
            Commit {
                resident_tokens: tokens.len(),
                actual_state: SlotState::Session,
                physical_bytes: 48,
                prefill_us_saved: 0,
            },
        )
        .unwrap();
    let busy_generation = cache.slot(seed.target.slot).unwrap().generation;

    let extension = [1, 2, 3, 4];
    let no_busy_read = cache
        .plan(request(
            &extension,
            namespace(1),
            2,
            Capabilities::WHOLE_STATE_COPY,
        ))
        .unwrap();
    assert_eq!(no_busy_read.operation, Operation::Fresh);
    cache.abort(no_busy_read.id).unwrap();

    let busy_read = cache
        .plan(request(
            &extension,
            namespace(1),
            2,
            Capabilities::WHOLE_STATE_COPY | Capabilities::SAFE_BUSY_DONOR,
        ))
        .unwrap();
    assert_eq!(busy_read.operation, Operation::Branch);
    assert_ne!(busy_read.target.slot, seed.target.slot);
    assert_eq!(
        cache.slot(seed.target.slot).unwrap().generation,
        busy_generation
    );
    assert_eq!(cache.slot(seed.target.slot).unwrap().read_leases, 1);
    cache.abort(busy_read.id).unwrap();
    assert_eq!(cache.slot(seed.target.slot).unwrap().read_leases, 0);
    assert_eq!(
        cache.slot(seed.target.slot).unwrap().generation,
        busy_generation
    );
}

#[test]
fn abort_invalidates_every_mutation_target_and_preserves_read_only_donors() {
    let mut cache = manager(2);
    let tokens = [1, 2, 3];
    let seed = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);
    let donor_before = cache.slot(seed.target.slot).unwrap();

    let branch_tokens = [1, 2, 4];
    let branch = cache
        .plan(request(
            &branch_tokens,
            namespace(1),
            2,
            Capabilities::PARTIAL_PREFIX_BRANCH,
        ))
        .unwrap();
    let target_before = cache.slot(branch.target.slot).unwrap().generation;
    cache.abort(branch.id).unwrap();

    let donor_after = cache.slot(seed.target.slot).unwrap();
    assert_eq!(donor_after.generation, donor_before.generation);
    assert_eq!(donor_after.resident_len, donor_before.resident_len);
    assert_eq!(donor_after.read_leases, 0);
    assert!(cache.slot(branch.target.slot).unwrap().generation > target_before);
    assert_eq!(cache.telemetry().write_leases, 0);
}

#[test]
fn reset_invalidates_entries_generations_and_outstanding_plans() {
    let mut cache = manager(2);
    let tokens = [1, 2, 3];
    let plan = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    let target = plan.target.clone();
    let old_epoch = cache.epoch();
    let new_epoch = cache.reset();
    assert!(new_epoch > old_epoch);
    assert_eq!(
        cache.commit(
            plan.id,
            Commit {
                resident_tokens: 3,
                actual_state: SlotState::Session,
                physical_bytes: 0,
                prefill_us_saved: 0,
            }
        ),
        Err(Error::PlanConsumed)
    );
    assert!(cache.slot(target.slot).unwrap().generation > target.generation);
    assert_eq!(cache.telemetry().active_slots, 0);
    assert_eq!(cache.telemetry().prefix_nodes, 0);
}

#[test]
fn explicit_invalidation_is_generation_guarded_and_removes_prefix() {
    let mut cache = manager(2);
    let tokens = [1, 2, 3];
    let seed = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);
    let slot = cache.slot(seed.target.slot).unwrap();
    assert_eq!(
        cache.invalidate(slot.id, slot.generation - 1),
        Err(Error::StalePlan)
    );
    let generation = cache.invalidate(slot.id, slot.generation).unwrap();
    let invalidated = cache.slot(slot.id).unwrap();
    assert_eq!(generation, invalidated.generation);
    assert_eq!(invalidated.state, SlotState::Empty);
    assert_eq!(invalidated.resident_len, 0);

    let next = cache
        .plan(request(
            &tokens,
            namespace(1),
            2,
            Capabilities::WHOLE_STATE_COPY,
        ))
        .unwrap();
    assert_eq!(next.operation, Operation::Fresh);
    assert_eq!(next.reuse_tokens, 0);
}

#[test]
fn confirmed_snapshot_replaces_decode_suffix_exactly() {
    let mut cache = manager(2);
    let prompt = [1, 2, 3];
    let seed = cache.plan(request(&prompt, namespace(1), 1, 0)).unwrap();
    commit_idle(&mut cache, &seed, prompt.len(), SlotState::Session);
    let initial = cache.slot(seed.target.slot).unwrap();
    let decoded = cache
        .advance(
            initial.id,
            initial.generation,
            &[4, 5],
            SlotState::Session,
            true,
            0,
        )
        .unwrap();
    let restored = cache
        .confirm(
            initial.id,
            decoded,
            &prompt,
            SlotState::PromptBoundary,
            false,
            0,
        )
        .unwrap();
    let slot = cache.slot(initial.id).unwrap();
    assert_eq!(slot.generation, restored);
    assert_eq!(slot.resident_len, prompt.len());
    assert_eq!(slot.state, SlotState::PromptBoundary);
    assert!(!slot.busy);
}

#[test]
fn commit_rejects_unconfirmed_length_and_publishes_nothing() {
    let mut cache = manager(1);
    let tokens = [1, 2, 3];
    let plan = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    assert_eq!(
        cache.commit(
            plan.id,
            Commit {
                resident_tokens: 4,
                actual_state: SlotState::Session,
                physical_bytes: 0,
                prefill_us_saved: 0,
            }
        ),
        Err(Error::InvalidArgument)
    );
    let slot = cache.slot(plan.target.slot).unwrap();
    assert_eq!(slot.state, SlotState::Empty);
    assert_eq!(slot.resident_len, 0);
}

#[test]
fn generation_guards_advance_and_anchors_cannot_be_extended() {
    let mut cache = manager(2);
    let tokens = [1, 2];
    let seed = cache.plan(request(&tokens, namespace(1), 1, 0)).unwrap();
    commit_idle(&mut cache, &seed, tokens.len(), SlotState::Session);
    let current = cache.slot(seed.target.slot).unwrap();
    assert_eq!(
        cache.advance(
            current.id,
            current.generation - 1,
            &[3],
            SlotState::Session,
            false,
            0
        ),
        Err(Error::StalePlan)
    );
    let next_generation = cache
        .advance(
            current.id,
            current.generation,
            &[3],
            SlotState::PromptBoundary,
            false,
            64,
        )
        .unwrap();
    assert!(next_generation > current.generation);

    let anchor_tokens = [8, 9];
    let mut anchor_request = request(&anchor_tokens, namespace(1), 2, 0);
    anchor_request.result_state = SlotState::Anchor;
    let anchor = cache.plan(anchor_request).unwrap();
    commit_idle(&mut cache, &anchor, 2, SlotState::Anchor);
    let anchor_slot = cache.slot(anchor.target.slot).unwrap();
    assert_eq!(
        cache.advance(
            anchor_slot.id,
            anchor_slot.generation,
            &[10],
            SlotState::Anchor,
            false,
            0
        ),
        Err(Error::SlotBusy)
    );
}

#[test]
fn deterministic_value_eviction_protects_anchor_and_interactive_session() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 2,
            min_reuse_tokens: 2,
            logical_token_capacity: 7,
            max_anchors: 1,
            automatic_checkpoints: Default::default(),
        },
        fixed_retention(2),
    )
    .unwrap();

    let anchor_tokens = [10, 11];
    let mut anchor_request = request(&anchor_tokens, namespace(1), 1, 0);
    anchor_request.result_state = SlotState::Anchor;
    anchor_request.labels.scope = Scope::Harness;
    let anchor = cache.plan(anchor_request).unwrap();
    commit_idle(&mut cache, &anchor, 2, SlotState::Anchor);

    let background_tokens = [20, 21];
    let mut background_request = request(&background_tokens, namespace(1), 2, 0);
    background_request.labels.priority = Priority::Background;
    let background = cache.plan(background_request).unwrap();
    commit_idle(&mut cache, &background, 2, SlotState::Session);

    let incoming = [30, 31, 32, 33, 34];
    let plan = cache.plan(request(&incoming, namespace(1), 3, 0)).unwrap();
    assert!(plan
        .evictions
        .iter()
        .any(|slot| slot.slot == background.target.slot));
    assert!(!plan
        .evictions
        .iter()
        .any(|slot| slot.slot == anchor.target.slot));
}

#[test]
fn exact_prefix_property_holds_over_many_deterministic_sequences() {
    use std::collections::BTreeMap;

    let mut cache = manager(8);
    let mut resident: BTreeMap<u32, (Namespace, Vec<i32>)> = BTreeMap::new();
    for seed in 0..64_i32 {
        let base = [seed % 4, (seed / 4) % 4, seed, 99];
        let request_namespace = namespace((seed % 2) as u8);
        let plan = cache
            .plan(request(
                &base,
                request_namespace,
                seed as u64 + 1,
                Capabilities::PARTIAL_PREFIX_BRANCH | Capabilities::WHOLE_STATE_COPY,
            ))
            .unwrap();
        if let Some(donor) = &plan.donor {
            let (donor_namespace, donor_tokens) = resident.get(&donor.slot).unwrap();
            let exact = donor_tokens
                .iter()
                .zip(base.iter())
                .take_while(|(left, right)| left == right)
                .count();
            assert_eq!(*donor_namespace, request_namespace);
            assert_eq!(plan.reuse_tokens, exact);
        } else {
            assert_eq!(plan.reuse_tokens, 0);
        }
        for victim in &plan.evictions {
            resident.remove(&victim.slot);
        }
        commit_idle(&mut cache, &plan, base.len(), SlotState::Session);
        resident.insert(plan.target.slot, (request_namespace, base.to_vec()));
    }
    assert!(cache.telemetry().prefix_nodes > 0);
}

#[test]
fn deeper_foreign_namespace_does_not_hide_a_shorter_local_prefix() {
    let mut cache = manager(3);
    let local = [1, 2, 8];
    let local_plan = cache.plan(request(&local, namespace(1), 1, 0)).unwrap();
    commit_idle(&mut cache, &local_plan, local.len(), SlotState::Session);

    let foreign = [1, 2, 3, 4];
    let foreign_plan = cache.plan(request(&foreign, namespace(2), 2, 0)).unwrap();
    commit_idle(&mut cache, &foreign_plan, foreign.len(), SlotState::Session);

    let requested = [1, 2, 3, 9];
    let plan = cache
        .plan(request(
            &requested,
            namespace(1),
            3,
            Capabilities::PARTIAL_PREFIX_BRANCH,
        ))
        .unwrap();
    assert_eq!(plan.donor.as_ref().unwrap().slot, local_plan.target.slot);
    assert_eq!(plan.reuse_tokens, 2);
}

#[test]
fn recurrent_fresh_plan_does_not_invent_an_unauthorized_boundary() {
    let mut cache = manager(4);
    let stable: Vec<i32> = (0..128).collect();
    let mut first_tokens = stable.clone();
    first_tokens.extend([1000, 1001]);
    let first = cache
        .plan(request(&first_tokens, namespace(1), 1, 0))
        .unwrap();
    commit_idle(&mut cache, &first, first_tokens.len(), SlotState::Session);

    let mut second_tokens = stable;
    second_tokens.extend([2000, 2001]);
    let slot_capabilities = [
        SlotCapabilities(SlotCapabilities::MATERIALIZED | SlotCapabilities::COPY),
        SlotCapabilities(SlotCapabilities::WRITABLE),
        SlotCapabilities(SlotCapabilities::WRITABLE),
        SlotCapabilities(SlotCapabilities::WRITABLE),
    ];
    let mut second_request = request(
        &second_tokens,
        namespace(1),
        2,
        Capabilities::WHOLE_STATE_COPY | Capabilities::RECURRENT_OR_HYBRID,
    );
    second_request.slot_capabilities = Some(&slot_capabilities);
    let second = cache.plan(second_request).unwrap();
    assert_eq!(second.operation, Operation::Fresh);
    assert_eq!(second.reuse_tokens, 0);
    assert_eq!(second.anchor_tokens, 0);
    assert!(second.anchor_boundaries.is_empty());
}

#[test]
fn cold_seed_boundary_materializes_once_then_restores_without_losing_session() {
    let mut cache = manager(4);
    let stable: Vec<i32> = (0..128).collect();
    let mut seed_tokens = stable.clone();
    seed_tokens.extend([1000, 1001]);
    let boundaries = [stable.len()];
    let slots = [
        SlotCapabilities(SlotCapabilities::WRITABLE),
        SlotCapabilities(SlotCapabilities::WRITABLE),
        SlotCapabilities(SlotCapabilities::WRITABLE),
        SlotCapabilities(SlotCapabilities::WRITABLE),
    ];
    let mut seed_request = request(
        &seed_tokens,
        namespace(1),
        1,
        Capabilities::WHOLE_STATE_COPY
            | Capabilities::RECURRENT_OR_HYBRID
            | Capabilities::PROMPT_BOUNDARY_SNAPSHOT,
    );
    seed_request.stable_boundaries = &boundaries;
    seed_request.slot_capabilities = Some(&slots);
    let seed = cache.plan(seed_request).unwrap();
    assert_eq!(seed.operation, Operation::Fresh);
    assert_eq!(seed.anchor_tokens, stable.len());
    assert_eq!(seed.anchor_boundaries, [stable.len()]);
    commit_idle(&mut cache, &seed, seed_tokens.len(), SlotState::Session);

    let mut anchor_request = request(&stable, namespace(1), 0, 0);
    anchor_request.result_state = SlotState::Anchor;
    let anchor = cache.plan(anchor_request).unwrap();
    commit_idle(&mut cache, &anchor, stable.len(), SlotState::Anchor);
    let anchor_snapshot = cache.slot(anchor.target.slot).unwrap();
    assert_eq!(anchor_snapshot.state, SlotState::Anchor);
    assert_eq!(anchor_snapshot.namespace, namespace(1));
    assert_eq!(anchor_snapshot.resident_len, stable.len());

    let mut flags = [SlotCapabilities(SlotCapabilities::WRITABLE); 4];
    flags[seed.target.slot as usize] = SlotCapabilities(
        SlotCapabilities::MATERIALIZED | SlotCapabilities::WRITABLE | SlotCapabilities::COPY,
    );
    flags[anchor.target.slot as usize] =
        SlotCapabilities(SlotCapabilities::MATERIALIZED | SlotCapabilities::COPY);
    for session in [2, 3] {
        let mut tokens = stable.clone();
        tokens.push(2000 + session as i32);
        let mut incoming = request(
            &tokens,
            namespace(1),
            session,
            Capabilities::WHOLE_STATE_COPY | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS,
        );
        incoming.slot_capabilities = Some(&flags);
        let restore = cache.plan(incoming).unwrap();
        assert_eq!(restore.operation, Operation::Restore);
        assert_eq!(restore.donor.as_ref().unwrap().slot, anchor.target.slot);
        assert_eq!(restore.reuse_tokens, stable.len());
        cache.abort(restore.id).unwrap();
    }

    let mut continuation = seed_tokens.clone();
    continuation.push(3000);
    let mut continuation_request = request(
        &continuation,
        namespace(1),
        1,
        Capabilities::WHOLE_STATE_COPY,
    );
    continuation_request.slot_capabilities = Some(&flags);
    let continued = cache.plan(continuation_request).unwrap();
    assert_eq!(continued.operation, Operation::Continue);
    assert_eq!(continued.target.slot, seed.target.slot);
    cache.abort(continued.id).unwrap();

    let mut foreign = request(
        &continuation,
        namespace(2),
        4,
        Capabilities::WHOLE_STATE_COPY,
    );
    foreign.slot_capabilities = Some(&flags);
    let isolated = cache.plan(foreign).unwrap();
    assert_eq!(isolated.operation, Operation::Fresh);
    assert_eq!(isolated.reuse_tokens, 0);
}

#[test]
fn mlx_and_gguf_hybrid_nested_anchor_matrix_restores_each_project_longest() {
    for backend_capabilities in [
        Capabilities::WHOLE_STATE_COPY
            | Capabilities::RECURRENT_OR_HYBRID
            | Capabilities::PROMPT_BOUNDARY_SNAPSHOT
            | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS,
        Capabilities::WHOLE_STATE_COPY
            | Capabilities::RECURRENT_OR_HYBRID
            | Capabilities::PROMPT_BOUNDARY_SNAPSHOT
            | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS,
    ] {
        let mut cache = manager(8);
        let project_a = [1, 2, 3, 4, 10, 11, 12, 13, 14, 90, 91];
        let project_b = [1, 2, 3, 4, 20, 21, 22, 23, 24, 92, 93];
        let boundaries = [4, 9, 9, 4];

        let mut seed_a = request(&project_a, namespace(1), 101, backend_capabilities);
        seed_a.stable_boundaries = &boundaries;
        let seed_a_plan = cache.plan(seed_a).unwrap();
        assert_eq!(seed_a_plan.anchor_boundaries, [4, 9]);
        cache.abort(seed_a_plan.id).unwrap();
        let global = materialize_anchor(&mut cache, &project_a[..4], namespace(1), Scope::Harness);
        let anchor_a =
            materialize_anchor(&mut cache, &project_a[..9], namespace(1), Scope::Project);

        let mut seed_b = request(&project_b, namespace(1), 202, backend_capabilities);
        seed_b.stable_boundaries = &boundaries;
        let seed_b_plan = cache.plan(seed_b).unwrap();
        assert_eq!(seed_b_plan.operation, Operation::Restore);
        assert_eq!(seed_b_plan.donor.as_ref().unwrap().slot, global);
        assert_eq!(seed_b_plan.reuse_tokens, 4);
        assert_eq!(seed_b_plan.anchor_boundaries, [9]);
        cache.abort(seed_b_plan.id).unwrap();
        let anchor_b =
            materialize_anchor(&mut cache, &project_b[..9], namespace(1), Scope::Project);

        for (tokens, expected) in [(project_a, anchor_a), (project_b, anchor_b)] {
            let reopened = cache
                .plan(request(&tokens, namespace(1), 303, backend_capabilities))
                .unwrap();
            assert_eq!(reopened.operation, Operation::Restore);
            assert_eq!(reopened.donor.as_ref().unwrap().slot, expected);
            assert_eq!(reopened.reuse_tokens, 9);
            cache.abort(reopened.id).unwrap();
        }

        let mut duplicate = request(&project_a, namespace(1), 404, backend_capabilities);
        duplicate.stable_boundaries = &boundaries;
        let duplicate_plan = cache.plan(duplicate).unwrap();
        assert!(duplicate_plan.anchor_boundaries.is_empty());
        cache.abort(duplicate_plan.id).unwrap();

        let foreign = cache
            .plan(request(&project_a, namespace(2), 505, backend_capabilities))
            .unwrap();
        assert_eq!(foreign.operation, Operation::Fresh);
        assert_eq!(foreign.reuse_tokens, 0);
        cache.abort(foreign.id).unwrap();

        let ghost_tokens = [1, 2, 3, 4, 30, 31];
        let mut ghost_request = request(&ghost_tokens, namespace(1), 0, 0);
        ghost_request.result_state = SlotState::Anchor;
        let ghost = cache.plan(ghost_request).unwrap();
        cache.abort(ghost.id).unwrap();
        assert!((0..8).all(|slot| cache.slot(slot).is_none_or(|entry| {
            entry.state != SlotState::Anchor || entry.resident_len != ghost_tokens.len()
        })));
    }
}

#[test]
fn gguf_unified_storage_authorizes_logical_shared_prefix_anchors() {
    let mut cache = manager(4);
    let tokens = [1, 2, 3, 4, 5, 6];
    let boundaries = [2, 5];
    let mut incoming = request(
        &tokens,
        namespace(1),
        1,
        Capabilities::PROMPT_BOUNDARY_SNAPSHOT | Capabilities::UNIFIED_STORAGE,
    );
    incoming.stable_boundaries = &boundaries;
    let plan = cache.plan(incoming).unwrap();
    assert_eq!(plan.anchor_boundaries, boundaries);
    cache.abort(plan.id).unwrap();
}

#[test]
fn automatic_token_checkpoints_are_exact_adaptive_and_bounded() {
    let mut cache = manager(10);
    let tokens: Vec<i32> = (0..17_000).collect();
    let plan = cache
        .plan(request(
            &tokens,
            namespace(1),
            1,
            Capabilities::PROMPT_BOUNDARY_SNAPSHOT,
        ))
        .unwrap();
    assert_eq!(
        plan.anchor_boundaries,
        [2_048, 4_096, 6_144, 8_192, 10_240, 12_288, 14_336, 16_384]
    );
    assert!(plan
        .anchor_boundaries
        .iter()
        .all(|&offset| offset < tokens.len()));
    cache.abort(plan.id).unwrap();

    let short: Vec<i32> = (0..2_047).collect();
    let short_plan = cache
        .plan(request(
            &short,
            namespace(1),
            2,
            Capabilities::PROMPT_BOUNDARY_SNAPSHOT,
        ))
        .unwrap();
    assert!(short_plan.anchor_boundaries.is_empty());
    cache.abort(short_plan.id).unwrap();
}

#[test]
fn automatic_checkpoint_budget_admits_a_fresh_namespace_baseline() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 4,
            min_reuse_tokens: 16,
            logical_token_capacity: usize::MAX,
            max_anchors: 8,
            automatic_checkpoints: Default::default(),
        },
        RetentionConfig {
            hard_max_retained_entries: 8,
            physical_byte_budget: Some(1_000_000),
            high_watermark_bytes: 900_000,
            low_watermark_bytes: 800_000,
        },
    )
    .unwrap();
    let first_tokens: Vec<i32> = (0..2_048).collect();
    let mut anchor_request = request(&first_tokens, namespace(1), 0, 0);
    anchor_request.result_state = SlotState::Anchor;
    anchor_request.estimated_bytes_per_token = 0;
    let anchor = cache.plan(anchor_request).unwrap();
    cache
        .commit(
            anchor.id,
            Commit {
                resident_tokens: first_tokens.len(),
                actual_state: SlotState::Anchor,
                physical_bytes: 200_000,
                prefill_us_saved: 0,
            },
        )
        .unwrap();

    let fresh: Vec<i32> = (10_000..15_000).collect();
    let plan = cache
        .plan(request(
            &fresh,
            namespace(2),
            2,
            Capabilities::PROMPT_BOUNDARY_SNAPSHOT,
        ))
        .unwrap();
    assert_eq!(plan.anchor_boundaries, [2_048, 4_096]);
    cache.abort(plan.id).unwrap();
}

#[test]
fn nontrimmable_full_donors_leave_deep_checkpoint_executable_across_projects() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 16,
            min_reuse_tokens: 16,
            logical_token_capacity: usize::MAX,
            max_anchors: 12,
            automatic_checkpoints: Default::default(),
        },
        RetentionConfig {
            hard_max_retained_entries: 16,
            physical_byte_budget: Some(1_100_000),
            high_watermark_bytes: 1_000_000,
            low_watermark_bytes: 900_000,
        },
    )
    .unwrap();
    let project_a: Vec<i32> = (0..14_472).collect();
    let mut seed = request(
        &project_a,
        namespace(1),
        0,
        Capabilities::PROMPT_BOUNDARY_SNAPSHOT,
    );
    seed.labels.project = 10;
    seed.labels.scope = Scope::Project;
    let seed_plan = cache.plan(seed).unwrap();
    let checkpoints = seed_plan.anchor_boundaries.clone();
    cache.abort(seed_plan.id).unwrap();
    assert!(checkpoints.contains(&14_336));

    for boundary in checkpoints {
        let mut anchor_request = request(&project_a[..boundary], namespace(1), 0, 0);
        anchor_request.labels.project = 10;
        anchor_request.labels.scope = Scope::Project;
        anchor_request.result_state = SlotState::Anchor;
        anchor_request.estimated_bytes_per_token = 16;
        let anchor = cache.plan(anchor_request).unwrap();
        cache
            .commit(
                anchor.id,
                Commit {
                    resident_tokens: boundary,
                    actual_state: SlotState::Anchor,
                    physical_bytes: boundary as u64 * 16,
                    prefill_us_saved: 0,
                },
            )
            .unwrap();
    }
    assert!((0..16).any(|slot| {
        cache.slot(slot).is_some_and(|snapshot| {
            snapshot.state == SlotState::Anchor && snapshot.resident_len == 14_336
        })
    }));

    for project in 10..13 {
        let mut donor_request = request(&project_a, namespace(1), 0, 0);
        donor_request.labels.project = project;
        donor_request.labels.scope = Scope::Project;
        donor_request.result_state = SlotState::PromptBoundary;
        donor_request.estimated_bytes_per_token = 0;
        let donor = cache.plan(donor_request).unwrap();
        cache
            .commit(
                donor.id,
                Commit {
                    resident_tokens: project_a.len(),
                    actual_state: SlotState::PromptBoundary,
                    physical_bytes: 0,
                    prefill_us_saved: 0,
                },
            )
            .unwrap();
        let snapshot = cache.slot(donor.target.slot).unwrap();
        cache
            .set_busy(snapshot.id, snapshot.generation, false)
            .unwrap();
    }

    let mut project_b = project_a[..14_426].to_vec();
    project_b.extend(20_000..22_547);
    let slot_capabilities = (0..16)
        .map(|slot| match cache.slot(slot) {
            Some(snapshot) if snapshot.state == SlotState::Anchor => SlotCapabilities(
                SlotCapabilities::MATERIALIZED
                    | SlotCapabilities::WRITABLE
                    | SlotCapabilities::COPY,
            ),
            Some(snapshot) if snapshot.state == SlotState::PromptBoundary => {
                SlotCapabilities(SlotCapabilities::MATERIALIZED | SlotCapabilities::WRITABLE)
            }
            _ => SlotCapabilities::ALL,
        })
        .collect::<Vec<_>>();
    let mut incoming = request(
        &project_b,
        namespace(1),
        0,
        Capabilities::WHOLE_STATE_COPY
            | Capabilities::PARTIAL_PREFIX_BRANCH
            | Capabilities::PARTIAL_SUFFIX_TRIM
            | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS,
    );
    incoming.labels.project = 20;
    incoming.labels.scope = Scope::Project;
    incoming.slot_capabilities = Some(&slot_capabilities);
    let plan = cache.plan(incoming).unwrap();
    assert_eq!(plan.operation, Operation::Restore);
    assert_eq!(plan.reuse_tokens, 14_336);
    assert!(plan.candidates.iter().any(|candidate| {
        candidate.state == SlotState::PromptBoundary
            && candidate.shared_prefix_tokens == 14_426
            && candidate.rejection == clap_cache_core::CandidateRejection::Nontrim
    }));
    cache.abort(plan.id).unwrap();
}

#[test]
fn impossible_automatic_checkpoint_budget_skips_without_failing_generation() {
    let checkpoints = clap_cache_core::AutomaticCheckpointConfig {
        memory_budget_cap_bytes: 1_000,
        ..Default::default()
    };
    let mut cache = CacheManager::new(
        Config {
            slot_count: 2,
            min_reuse_tokens: 16,
            logical_token_capacity: usize::MAX,
            max_anchors: 2,
            automatic_checkpoints: checkpoints,
        },
        RetentionConfig {
            hard_max_retained_entries: 2,
            physical_byte_budget: Some(100_000),
            high_watermark_bytes: 90_000,
            low_watermark_bytes: 80_000,
        },
    )
    .unwrap();
    let tokens: Vec<i32> = (0..5_000).collect();
    let mut incoming = request(
        &tokens,
        namespace(1),
        1,
        Capabilities::PROMPT_BOUNDARY_SNAPSHOT,
    );
    incoming.estimated_bytes_per_token = 16;
    let plan = cache.plan(incoming).unwrap();
    assert!(plan.anchor_boundaries.is_empty());
    cache.abort(plan.id).unwrap();
}

#[test]
fn anchor_pressure_keeps_structural_and_reused_project_boundaries() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 4,
            min_reuse_tokens: 2,
            logical_token_capacity: usize::MAX,
            max_anchors: 3,
            automatic_checkpoints: Default::default(),
        },
        fixed_retention(4),
    )
    .unwrap();
    let global = materialize_anchor(&mut cache, &[1, 2], namespace(1), Scope::Harness);
    let project_a = materialize_anchor(&mut cache, &[1, 2, 10, 11], namespace(1), Scope::Project);
    let project_b = materialize_anchor(&mut cache, &[1, 2, 20, 21], namespace(1), Scope::Project);

    let reused = [1, 2, 20, 21, 99];
    let restore = cache
        .plan(request(
            &reused,
            namespace(1),
            9,
            Capabilities::WHOLE_STATE_COPY | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS,
        ))
        .unwrap();
    assert_eq!(restore.donor.as_ref().unwrap().slot, project_b);
    commit_idle(&mut cache, &restore, 4, SlotState::Session);

    let replacement_tokens = [1, 2, 30, 31];
    let mut replacement_request = request(&replacement_tokens, namespace(1), 0, 0);
    replacement_request.result_state = SlotState::Anchor;
    replacement_request.labels.scope = Scope::Project;
    let replacement = cache.plan(replacement_request).unwrap();
    assert_eq!(replacement.target.slot, project_a);
    assert!(replacement
        .evictions
        .iter()
        .any(|slot| slot.slot == project_a));
    assert!(!replacement.evictions.iter().any(|slot| slot.slot == global));
    assert!(!replacement
        .evictions
        .iter()
        .any(|slot| slot.slot == project_b));
    cache.abort(replacement.id).unwrap();
}

#[test]
fn generic_recent_conversation_retention_uses_empty_space_then_evicts_low_value_work() {
    let mut cache = manager(4);
    let harness: Vec<i32> = (0..16_384).collect();
    let mut primary_request = request(&harness, namespace(1), 1, 0);
    primary_request.labels.scope = Scope::Harness;
    let primary = cache.plan(primary_request).unwrap();
    commit_idle(&mut cache, &primary, harness.len(), SlotState::Session);

    let unrelated: Vec<i32> = (50_000..50_096).collect();
    let mut short_request = request(&unrelated, namespace(1), 2, 0);
    short_request.labels.side_request = true;
    short_request.labels.priority = Priority::Background;
    let short = cache.plan(short_request).unwrap();
    assert!(short.evictions.is_empty());
    commit_idle(&mut cache, &short, unrelated.len(), SlotState::Session);

    for session in 3..=4 {
        let tokens = [session as i32, 70_000 + session as i32];
        let filler = cache
            .plan(request(&tokens, namespace(1), session, 0))
            .unwrap();
        assert!(filler.evictions.is_empty());
        commit_idle(&mut cache, &filler, tokens.len(), SlotState::Session);
    }

    let incoming = [80_000, 80_001, 80_002];
    let pressure = cache.plan(request(&incoming, namespace(1), 5, 0)).unwrap();
    assert_eq!(pressure.evictions.len(), 1);
    assert_eq!(pressure.evictions[0].slot, short.target.slot);
    assert_ne!(pressure.evictions[0].slot, primary.target.slot);
}

#[test]
fn candidate_diagnostics_are_deterministically_truncated_at_sixteen() {
    let mut cache = manager(17);
    for session in 0..17 {
        let tokens = [1_000 + session as i32, 2_000 + session as i32];
        let plan = cache
            .plan(request(&tokens, namespace(1), session + 1, 0))
            .unwrap();
        commit_idle(&mut cache, &plan, tokens.len(), SlotState::Session);
    }

    let incoming = [9_000, 9_001];
    let plan = cache
        .plan(request(&incoming, namespace(1), 100, 0))
        .unwrap();
    assert_eq!(plan.candidates.len(), 16);
    assert_eq!(
        plan.candidates
            .iter()
            .map(|candidate| candidate.slot)
            .collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    cache.abort(plan.id).unwrap();
}
