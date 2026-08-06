use clap_cache_core::{
    AutomaticCheckpointConfig, CacheManager, Capabilities, Commit, Config, Labels, Namespace,
    PlanRequest, Priority, RetentionConfig, Scope, SlotState,
};

const SLOTS: u32 = 32;
const BYTES_PER_TOKEN: u64 = 16;

fn namespace(byte: u8) -> Namespace {
    Namespace([byte; 32])
}

fn labels(session: u64) -> Labels {
    Labels {
        tenant: 1,
        project: 1,
        harness: 1,
        agent: 1,
        session,
        scope: Scope::Session,
        priority: Priority::Interactive,
        side_request: false,
    }
}

fn request<'a>(tokens: &'a [i32], session: u64, now_ms: u64, state: SlotState) -> PlanRequest<'a> {
    PlanRequest {
        namespace: namespace(1),
        tokens,
        stable_boundaries: &[],
        labels: labels(session),
        capabilities: Capabilities(Capabilities::WHOLE_STATE_COPY),
        slot_capabilities: None,
        output_reserve: 0,
        estimated_bytes_per_token: BYTES_PER_TOKEN,
        now_ms,
        result_state: state,
    }
}

fn manager() -> CacheManager {
    CacheManager::new(
        Config {
            slot_count: SLOTS,
            min_reuse_tokens: 2,
            logical_token_capacity: usize::MAX,
            max_anchors: SLOTS,
            max_anchors_per_session: 4,
            max_anchor_bytes_per_session: 16_384,
            session_idle_ttl_ms: 60_000,
            automatic_checkpoints: AutomaticCheckpointConfig {
                enabled: false,
                ..Default::default()
            },
        },
        RetentionConfig {
            hard_max_retained_entries: SLOTS,
            physical_byte_budget: None,
            high_watermark_bytes: 0,
            low_watermark_bytes: 0,
        },
    )
    .unwrap()
}

fn commit_anchor(cache: &mut CacheManager, tokens: &[i32], session: u64, now_ms: u64) {
    let plan = cache
        .plan(request(tokens, session, now_ms, SlotState::Anchor))
        .unwrap();
    cache
        .commit(
            plan.id,
            Commit {
                resident_tokens: tokens.len(),
                actual_state: SlotState::Anchor,
                // The sequence backend may not expose authoritative physical
                // bytes. Policy accounting must still bound this session.
                physical_bytes: 0,
                prefill_us_saved: plan.reuse_tokens as u64 * 10,
            },
        )
        .unwrap();
}

fn session_anchor_count(cache: &CacheManager, session: u64) -> usize {
    (0..SLOTS)
        .filter(|&slot| {
            cache.slot(slot).is_some_and(|snapshot| {
                snapshot.state == SlotState::Anchor && snapshot.labels.session == session
            })
        })
        .count()
}

#[test]
fn noisy_agent_and_many_users_remain_bounded_through_hard_ceiling_churn() {
    let mut cache = manager();

    // One tool-heavy agent publishes far more logical boundaries than the
    // backend can retain. It must remain bounded before other users arrive.
    let mut noisy = Vec::new();
    for turn in 0..100 {
        noisy.push(turn);
        noisy.push(10_000 + turn);
        commit_anchor(&mut cache, &noisy, 1, turn as u64 + 1);
        assert!(session_anchor_count(&cache, 1) <= 4);
        assert!(cache.telemetry().active_slots <= SLOTS);
    }
    assert_eq!(session_anchor_count(&cache, 1), 4);

    // More users than retained entries can still execute correctly. The pool
    // churns at its hard ceiling instead of treating entries as user seats.
    for session in 2..=80 {
        let tokens = [session as i32, 50_000 + session as i32, 60_000];
        commit_anchor(&mut cache, &tokens, session, 1_000 + session);
        assert!(cache.telemetry().active_slots <= SLOTS);
        assert!(session_anchor_count(&cache, session) <= 1);
    }
    assert_eq!(cache.telemetry().active_slots, SLOTS);
    assert_eq!(cache.telemetry().anchors, SLOTS);

    // Saturation never makes inference itself dependent on retaining another
    // anchor. A fresh user gets a writable session by recycling one cache
    // entry, then completes normally.
    let prompt = [90_000, 90_001, 90_002, 90_003];
    let plan = cache
        .plan(request(&prompt, 10_000, 2_000, SlotState::Session))
        .unwrap();
    assert!(!plan.evictions.is_empty());
    cache
        .commit(
            plan.id,
            Commit {
                resident_tokens: prompt.len(),
                actual_state: SlotState::Session,
                physical_bytes: 64,
                prefill_us_saved: 0,
            },
        )
        .unwrap();
    let session = cache.slot(plan.target.slot).unwrap();
    cache
        .set_busy(session.id, session.generation, false)
        .unwrap();
    cache.touch(session.id, session.generation, 2_001).unwrap();

    // The noisy agent can return after saturation. It may miss, but it cannot
    // exceed its owner budget or fail because all 32 retained entries are in use.
    noisy.extend([99_001, 99_002]);
    commit_anchor(&mut cache, &noisy, 1, 2_100);
    assert!(session_anchor_count(&cache, 1) <= 4);
    assert!(cache.telemetry().active_slots <= SLOTS);

    let telemetry = cache.telemetry();
    assert_eq!(telemetry.anchor_publications, 180);
    assert!(telemetry.evictions > 0);
    assert!(telemetry.session_policy_evictions >= 96);
    assert_eq!(telemetry.session_budget_rejections, 0);
    assert_eq!(telemetry.physical_bytes, 64);
    assert!(telemetry.anchor_accounted_bytes > 0);
}

#[test]
fn expiry_reclaims_cold_users_without_touching_recent_or_active_work() {
    let mut cache = manager();
    for session in 1..=12 {
        let tokens = [session as i32, 100 + session as i32, 200];
        commit_anchor(&mut cache, &tokens, session, session * 1_000);
    }

    let active_tokens = [700, 701, 702];
    let active_plan = cache
        .plan(request(&active_tokens, 99, 59_000, SlotState::Session))
        .unwrap();
    cache
        .commit(
            active_plan.id,
            Commit {
                resident_tokens: active_tokens.len(),
                actual_state: SlotState::Session,
                physical_bytes: 48,
                prefill_us_saved: 0,
            },
        )
        .unwrap();

    let expired = cache.expire_idle(70_000);
    assert_eq!(expired.len(), 10);
    assert_eq!(cache.telemetry().expired_slots, 10);
    assert_eq!(cache.telemetry().anchors, 2);
    assert_eq!(cache.telemetry().session_slots, 1);
    assert!(cache.slot(active_plan.target.slot).unwrap().busy);

    let released = cache.release_session(namespace(1), 12).unwrap();
    assert_eq!(released.len(), 1);
    assert_eq!(cache.telemetry().released_session_slots, 1);
}
