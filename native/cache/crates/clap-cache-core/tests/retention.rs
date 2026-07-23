use std::collections::BTreeMap;

use clap_cache_core::{
    CacheManager, Capabilities, Commit, Config, Error, Labels, Namespace, PlanRequest, Priority,
    RetentionConfig, Scope, SlotState,
};

fn namespace(byte: u8) -> Namespace {
    Namespace([byte; 32])
}

fn labels(session: u64) -> Labels {
    Labels {
        tenant: session,
        project: 0,
        harness: 0,
        agent: 0,
        session,
        scope: Scope::Session,
        priority: Priority::Interactive,
        side_request: false,
    }
}

fn manager(initial: u32, hard_max: u32) -> CacheManager {
    CacheManager::new(
        Config {
            slot_count: initial,
            min_reuse_tokens: 2,
            logical_token_capacity: 1,
            max_anchors: hard_max,
            automatic_checkpoints: Default::default(),
        },
        RetentionConfig {
            hard_max_retained_entries: hard_max,
            physical_byte_budget: Some(100),
            high_watermark_bytes: 80,
            low_watermark_bytes: 50,
        },
    )
    .unwrap()
}

fn request(tokens: &[i32], ns: Namespace, session: u64, state: SlotState) -> PlanRequest<'_> {
    PlanRequest {
        namespace: ns,
        tokens,
        stable_boundaries: &[],
        labels: labels(session),
        capabilities: Capabilities(0),
        slot_capabilities: None,
        output_reserve: usize::MAX,
        estimated_bytes_per_token: 1,
        result_state: state,
    }
}

fn seed(
    cache: &mut CacheManager,
    tokens: &[i32],
    ns: Namespace,
    session: u64,
    state: SlotState,
    bytes: u64,
) -> (u32, u64) {
    let plan = cache.plan(request(tokens, ns, session, state)).unwrap();
    let slot = plan.target.slot;
    cache
        .commit(
            plan.id,
            Commit {
                resident_tokens: tokens.len(),
                actual_state: state,
                physical_bytes: bytes,
                prefill_us_saved: 0,
            },
        )
        .unwrap();
    let snapshot = cache.slot(slot).unwrap();
    if snapshot.busy {
        cache
            .set_busy(snapshot.id, snapshot.generation, false)
            .unwrap();
    }
    let snapshot = cache.slot(slot).unwrap();
    (slot, snapshot.generation)
}

fn seed_priority(
    cache: &mut CacheManager,
    tokens: &[i32],
    ns: Namespace,
    session: u64,
    state: SlotState,
    bytes: u64,
    priority: Priority,
) -> (u32, u64) {
    let mut input = request(tokens, ns, session, state);
    input.labels.priority = priority;
    let plan = cache.plan(input).unwrap();
    let slot = plan.target.slot;
    cache.commit(plan.id, Commit { resident_tokens: tokens.len(), actual_state: state,
        physical_bytes: bytes, prefill_us_saved: 0 }).unwrap();
    let snapshot = cache.slot(slot).unwrap();
    if snapshot.busy { cache.set_busy(snapshot.id, snapshot.generation, false).unwrap(); }
    (slot, cache.slot(slot).unwrap().generation)
}

#[test]
fn priority_ordinals_order_session_anchor_and_byte_pressure_eviction() {
    assert_eq!(Priority::Background as u32, 0);
    assert_eq!(Priority::Normal as u32, 1);
    assert_eq!(Priority::Interactive as u32, 2);

    let mut sessions = manager(3, 3);
    let background = seed_priority(&mut sessions, &[1, 1], namespace(1), 1,
        SlotState::Session, 1, Priority::Background).0;
    seed_priority(&mut sessions, &[2, 2], namespace(2), 2,
        SlotState::Session, 1, Priority::Normal);
    seed_priority(&mut sessions, &[3, 3], namespace(3), 3,
        SlotState::Session, 1, Priority::Interactive);
    assert_eq!(sessions.plan(request(&[4, 4], namespace(4), 4, SlotState::Session))
        .unwrap().target.slot, background);

    let mut anchors = manager(2, 2);
    let background_anchor = seed_priority(&mut anchors, &[1, 1], namespace(1), 1,
        SlotState::Anchor, 1, Priority::Background).0;
    seed_priority(&mut anchors, &[2, 2], namespace(2), 2,
        SlotState::Anchor, 1, Priority::Interactive);
    let mut normal_anchor = request(&[3, 3], namespace(3), 3, SlotState::Anchor);
    normal_anchor.labels.priority = Priority::Normal;
    assert_eq!(anchors.plan(normal_anchor).unwrap().target.slot, background_anchor);

    let mut pressure = manager(4, 4);
    let pressure_background = seed_priority(&mut pressure, &[1, 1], namespace(1), 1,
        SlotState::Session, 30, Priority::Background).0;
    let pressure_normal = seed_priority(&mut pressure, &[2, 2], namespace(2), 2,
        SlotState::Session, 30, Priority::Normal).0;
    seed_priority(&mut pressure, &[3, 3], namespace(3), 3,
        SlotState::Session, 30, Priority::Interactive);
    let victims: Vec<_> = pressure.plan(request(&[4, 4], namespace(4), 4, SlotState::Session))
        .unwrap().evictions.into_iter().map(|victim| victim.slot).collect();
    assert_eq!(victims, vec![pressure_background, pressure_normal]);
}

#[test]
fn grows_past_one_hundred_with_stable_ids_and_hard_ceiling() {
    let mut cache = manager(2, 128);
    let initial_generation = cache.slot(0).unwrap().generation;
    for expected in 2..128 {
        let registered = cache.register_slot().unwrap();
        assert_eq!(registered.slot, expected);
        assert_eq!(registered.generation, 1);
    }
    assert_eq!(cache.register_slot(), Err(Error::NoCapacity));
    assert_eq!(cache.telemetry().total_slots, 128);

    cache.reset();
    assert_eq!(cache.slot(0).unwrap().generation, initial_generation + 1);
    assert_eq!(cache.slot(127).unwrap().generation, 2);
}

#[test]
fn byte_budget_ignores_logical_capacity_and_output_reserve_below_high() {
    let mut cache = manager(4, 4);
    let tokens = [1, 2, 3, 4];
    let first = seed(&mut cache, &tokens, namespace(1), 1, SlotState::Session, 70);
    let plan = cache
        .plan(request(&[9, 8, 7, 6], namespace(2), 2, SlotState::Session))
        .unwrap();
    assert!(plan.evictions.is_empty());
    assert_eq!(cache.slot(first.0).unwrap().physical_bytes, 70);
    cache.abort(plan.id).unwrap();
}

#[test]
fn pressure_evicts_deterministically_to_low_and_abort_cleans_leases() {
    let mut cache = manager(6, 6);
    for index in 0..3 {
        seed(
            &mut cache,
            &[index, index + 10],
            namespace(index as u8 + 1),
            index as u64 + 1,
            SlotState::Session,
            30,
        );
    }
    assert!(cache.telemetry().under_pressure);
    let mut pressured = request(&[99; 10], namespace(9), 9, SlotState::Session);
    pressured.estimated_bytes_per_token = 1;
    let plan = cache.plan(pressured).unwrap();
    let victims: Vec<_> = plan.evictions.iter().map(|slot| slot.slot).collect();
    assert_eq!(victims, vec![0, 1]);
    assert_eq!(cache.telemetry().write_leases, 3);

    cache.abort(plan.id).unwrap();
    assert_eq!(cache.telemetry().write_leases, 0);
    assert_eq!(cache.telemetry().physical_bytes, 30);
    assert_eq!(cache.telemetry().active_slots, 1);
}

#[test]
fn failed_pressure_commit_invalidates_all_authorized_mutations() {
    let mut cache = manager(6, 6);
    for index in 0..3 {
        seed(
            &mut cache,
            &[index, index + 10],
            namespace(index as u8 + 1),
            index as u64 + 1,
            SlotState::Session,
            30,
        );
    }
    let plan = cache
        .plan(request(&[99; 10], namespace(9), 9, SlotState::Session))
        .unwrap();
    assert_eq!(plan.evictions.len(), 2);
    assert_eq!(
        cache.commit(
            plan.id,
            Commit {
                resident_tokens: 11,
                actual_state: SlotState::Session,
                physical_bytes: 10,
                prefill_us_saved: 0,
            },
        ),
        Err(Error::InvalidArgument)
    );
    assert_eq!(cache.telemetry().write_leases, 0);
    assert_eq!(cache.telemetry().physical_bytes, 30);
    assert_eq!(cache.telemetry().active_slots, 1);
}

#[test]
fn pressure_excludes_existing_read_and_write_leases() {
    let mut cache = manager(6, 6);
    let donor_tokens = [1, 2, 3];
    let (donor, _) = seed(
        &mut cache,
        &donor_tokens,
        namespace(1),
        1,
        SlotState::Session,
        30,
    );
    let mut branch_request = request(&donor_tokens, namespace(1), 2, SlotState::Session);
    branch_request.capabilities = Capabilities(Capabilities::WHOLE_STATE_COPY);
    let branch = cache.plan(branch_request).unwrap();
    assert_eq!(branch.donor.as_ref().unwrap().slot, donor);
    let leased_target = branch.target.slot;

    seed(&mut cache, &[4, 4], namespace(2), 3, SlotState::Session, 30);
    seed(&mut cache, &[5, 5], namespace(3), 4, SlotState::Session, 30);
    let pressured = cache
        .plan(request(&[8; 10], namespace(8), 8, SlotState::Session))
        .unwrap();
    assert!(!pressured.evictions.iter().any(|slot| slot.slot == donor));
    assert!(!pressured
        .evictions
        .iter()
        .any(|slot| slot.slot == leased_target));

    cache.abort(pressured.id).unwrap();
    cache.abort(branch.id).unwrap();
    assert_eq!(cache.telemetry().read_leases, 0);
    assert_eq!(cache.telemetry().write_leases, 0);
}

#[test]
fn busy_and_protected_anchors_are_ineligible_pressure_victims() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 6,
            min_reuse_tokens: 2,
            logical_token_capacity: usize::MAX,
            max_anchors: 6,
            automatic_checkpoints: Default::default(),
        },
        RetentionConfig {
            hard_max_retained_entries: 6,
            physical_byte_budget: Some(150),
            high_watermark_bytes: 100,
            low_watermark_bytes: 70,
        },
    )
    .unwrap();
    let (busy_slot, busy_generation) =
        seed(&mut cache, &[1, 1], namespace(1), 1, SlotState::Session, 30);
    cache.set_busy(busy_slot, busy_generation, true).unwrap();
    let (anchor_slot, anchor_generation) =
        seed(&mut cache, &[2, 2], namespace(2), 2, SlotState::Anchor, 30);
    cache
        .set_anchor_protected(anchor_slot, anchor_generation, true)
        .unwrap();
    let (eligible, _) = seed(&mut cache, &[3, 3], namespace(3), 3, SlotState::Session, 30);
    let (eligible_two, _) = seed(&mut cache, &[4, 4], namespace(4), 4, SlotState::Session, 30);

    let plan = cache
        .plan(request(&[8; 10], namespace(8), 8, SlotState::Session))
        .unwrap();
    assert!(plan.evictions.iter().any(|slot| slot.slot == eligible));
    assert!(plan.evictions.iter().any(|slot| slot.slot == eligible_two));
    assert!(!plan.evictions.iter().any(|slot| slot.slot == busy_slot));
    assert!(!plan.evictions.iter().any(|slot| slot.slot == anchor_slot));
    cache.abort(plan.id).unwrap();
}

#[test]
fn pressure_prefers_over_fair_share_namespace() {
    let mut cache = manager(6, 6);
    let (a0, _) = seed(&mut cache, &[1, 0], namespace(1), 1, SlotState::Session, 30);
    let (a1, _) = seed(&mut cache, &[1, 1], namespace(1), 2, SlotState::Session, 30);
    let (b0, _) = seed(&mut cache, &[2, 0], namespace(2), 3, SlotState::Session, 30);
    let plan = cache
        .plan(request(&[7; 10], namespace(2), 7, SlotState::Session))
        .unwrap();
    let victims: Vec<_> = plan.evictions.iter().map(|slot| slot.slot).collect();
    assert_eq!(victims, vec![a0, a1]);
    assert!(!victims.contains(&b0));
    cache.abort(plan.id).unwrap();
}

#[test]
fn physical_bytes_follow_commit_advance_invalidate_and_reset() {
    let mut cache = manager(2, 2);
    let (slot, generation) = seed(&mut cache, &[1, 2], namespace(1), 1, SlotState::Session, 20);
    let telemetry = cache.telemetry();
    assert_eq!(telemetry.session_slots, 1);
    assert_eq!(telemetry.session_bytes, 20);
    assert_eq!(telemetry.active_bytes, 20);

    let generation = cache
        .advance(slot, generation, &[3], SlotState::Session, false, 35)
        .unwrap();
    assert_eq!(cache.telemetry().physical_bytes, 35);
    let generation = cache.invalidate(slot, generation).unwrap();
    assert_eq!(cache.telemetry().physical_bytes, 0);
    assert_eq!(cache.slot(slot).unwrap().generation, generation);

    seed(&mut cache, &[4, 5], namespace(2), 2, SlotState::Anchor, 40);
    assert_eq!(cache.telemetry().anchor_bytes, 40);
    cache.reset();
    assert_eq!(cache.telemetry().physical_bytes, 0);
    assert_eq!(cache.telemetry().active_slots, 0);
}

#[test]
fn branch_pressure_preserves_the_only_lower_depth_band_donor() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 10,
            min_reuse_tokens: 16,
            logical_token_capacity: usize::MAX,
            max_anchors: 10,
            automatic_checkpoints: Default::default(),
        },
        RetentionConfig {
            hard_max_retained_entries: 12,
            physical_byte_budget: Some(200_000),
            high_watermark_bytes: 160_000,
            low_watermark_bytes: 110_000,
        },
    )
    .unwrap();
    let seed_tokens: Vec<i32> = (0..17_021).collect();
    let mut anchors = BTreeMap::new();
    for offset in (2_048..=16_384).step_by(2_048) {
        let mut anchor_request =
            request(&seed_tokens[..offset], namespace(1), 0, SlotState::Anchor);
        anchor_request.estimated_bytes_per_token = 0;
        let anchor = cache.plan(anchor_request).unwrap();
        let slot = anchor.target.slot;
        cache
            .commit(
                anchor.id,
                Commit {
                    resident_tokens: offset,
                    actual_state: SlotState::Anchor,
                    physical_bytes: 20_000,
                    prefill_us_saved: 0,
                },
            )
            .unwrap();
        anchors.insert(offset, slot);
    }

    let mut branch = seed_tokens[..14_408].to_vec();
    branch.push(-1);
    let mut branch_request = request(&branch, namespace(1), 2, SlotState::Session);
    branch_request.capabilities =
        Capabilities(Capabilities::WHOLE_STATE_COPY | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS);
    let plan = cache.plan(branch_request).unwrap();
    assert_eq!(plan.reuse_tokens, 14_336);
    assert!(!plan
        .evictions
        .iter()
        .any(|victim| victim.slot == anchors[&8_192]));
    cache
        .commit(
            plan.id,
            Commit {
                resident_tokens: 14_336,
                actual_state: SlotState::Session,
                physical_bytes: 8_508,
                prefill_us_saved: 0,
            },
        )
        .unwrap();

    let mut lower_branch = seed_tokens[..8_508].to_vec();
    lower_branch.push(-2);
    let mut lower_request = request(&lower_branch, namespace(1), 3, SlotState::Session);
    lower_request.capabilities =
        Capabilities(Capabilities::WHOLE_STATE_COPY | Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS);
    let lower = cache.plan(lower_request).unwrap();
    assert_eq!(lower.reuse_tokens, 8_192);
    assert_eq!(lower.donor.unwrap().slot, anchors[&8_192]);
    cache.abort(lower.id).unwrap();
}

#[test]
fn retained_pressure_cannot_reject_an_executable_session_target() {
    let mut cache = CacheManager::new(
        Config {
            slot_count: 3,
            min_reuse_tokens: 2,
            logical_token_capacity: usize::MAX,
            max_anchors: 2,
            automatic_checkpoints: Default::default(),
        },
        RetentionConfig {
            hard_max_retained_entries: 3,
            physical_byte_budget: Some(100),
            high_watermark_bytes: 80,
            low_watermark_bytes: 50,
        },
    )
    .unwrap();
    for suffix in [10, 20] {
        let tokens = [1, suffix];
        let mut anchor_request = request(&tokens, namespace(1), 0, SlotState::Anchor);
        anchor_request.labels.scope = Scope::Harness;
        anchor_request.estimated_bytes_per_token = 0;
        let anchor = cache.plan(anchor_request).unwrap();
        cache
            .commit(
                anchor.id,
                Commit {
                    resident_tokens: tokens.len(),
                    actual_state: SlotState::Anchor,
                    physical_bytes: 45,
                    prefill_us_saved: 0,
                },
            )
            .unwrap();
    }
    let mut incoming = request(&[99; 17_000], namespace(1), 7, SlotState::Session);
    incoming.estimated_bytes_per_token = 1_000;
    let plan = cache.plan(incoming).unwrap();
    assert_eq!(plan.operation, clap_cache_core::Operation::Fresh);
    cache.abort(plan.id).unwrap();
}
