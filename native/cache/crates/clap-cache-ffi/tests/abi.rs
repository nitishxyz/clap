use std::mem::{offset_of, size_of, MaybeUninit};
use std::ptr;

use clap_cache_ffi::*;

#[test]
fn config_layout_matches_the_public_c_header() {
    assert_eq!(CLAP_CACHE_ABI_VERSION, 3);
    assert_eq!(size_of::<ClapCacheConfig>(), 72);
    assert_eq!(offset_of!(ClapCacheConfig, version), 0);
    assert_eq!(offset_of!(ClapCacheConfig, struct_size), 4);
    assert_eq!(offset_of!(ClapCacheConfig, slot_count), 8);
    assert_eq!(offset_of!(ClapCacheConfig, max_anchors), 12);
    assert_eq!(offset_of!(ClapCacheConfig, min_reuse_tokens), 16);
    assert_eq!(offset_of!(ClapCacheConfig, logical_token_capacity), 24);
    assert_eq!(offset_of!(ClapCacheConfig, automatic_checkpoint_mode), 32);
    assert_eq!(offset_of!(ClapCacheConfig, automatic_checkpoint_max), 36);
    assert_eq!(
        offset_of!(ClapCacheConfig, automatic_checkpoint_min_tokens),
        40
    );
    assert_eq!(
        offset_of!(ClapCacheConfig, automatic_checkpoint_interval_tokens),
        48
    );
    assert_eq!(
        offset_of!(ClapCacheConfig, automatic_checkpoint_memory_basis_points),
        56
    );
    assert_eq!(offset_of!(ClapCacheConfig, reserved), 60);
    assert_eq!(
        offset_of!(ClapCacheConfig, automatic_checkpoint_memory_cap_bytes),
        64
    );
    assert_eq!(size_of::<ClapCacheRetentionTelemetry>(), 112);
    assert_eq!(
        offset_of!(ClapCacheRetentionTelemetry, automatic_checkpoint_slots),
        48
    );
    assert_eq!(
        offset_of!(ClapCacheRetentionTelemetry, automatic_checkpoint_bytes),
        56
    );
    assert_eq!(
        offset_of!(
            ClapCacheRetentionTelemetry,
            automatic_checkpoint_byte_budget
        ),
        64
    );
    assert_eq!(offset_of!(ClapCacheRetentionTelemetry, under_pressure), 104);
}

fn labels(session: u64) -> ClapCacheLabels {
    ClapCacheLabels {
        version: CLAP_CACHE_ABI_VERSION,
        struct_size: size_of::<ClapCacheLabels>() as u32,
        tenant: 1,
        project: 2,
        harness: 3,
        agent: 4,
        session,
        scope: 1,
        priority: 2,
        side_request: 0,
        reserved: [0; 7],
    }
}

#[test]
fn ffi_lifecycle_exports_owned_plan_and_metrics() {
    unsafe {
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 2,
            max_anchors: 1,
            min_reuse_tokens: 2,
            logical_token_capacity: u64::MAX,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(clap_cache_create(&config, &mut cache), ClapCacheStatus::Ok);
        assert!(!cache.is_null());

        let tokens = [1, 2, 3];
        let request = ClapCacheRequest {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRequest>() as u32,
            namespace_fingerprint: [9; 32],
            tokens: tokens.as_ptr(),
            tokens_len: tokens.len(),
            labels: labels(7),
            capabilities: 0,
            slot_capabilities: ptr::null(),
            slot_capabilities_len: 0,
            stable_boundaries: ptr::null(),
            stable_boundaries_len: 0,
            output_reserve: 0,
            estimated_bytes_per_token: 16,
            result_state: 1,
            reserved: 0,
        };
        let mut plan = ptr::null_mut();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );

        let mut view = MaybeUninit::<ClapCachePlanView>::uninit();
        assert_eq!(
            clap_cache_plan_view(plan, view.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let view = view.assume_init();
        assert_eq!(view.operation, 0);
        assert_eq!(view.reuse_tokens, 0);

        let commit = ClapCacheCommit {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheCommit>() as u32,
            resident_tokens: tokens.len() as u64,
            actual_state: 1,
            reserved: 0,
            physical_bytes: 48,
            prefill_us_saved: 0,
        };
        let mut decision = MaybeUninit::<ClapCacheDecision>::uninit();
        assert_eq!(
            clap_cache_commit(cache, plan, &commit, decision.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let mut consumed_view = MaybeUninit::<ClapCachePlanView>::uninit();
        assert_eq!(
            clap_cache_plan_view(plan, consumed_view.as_mut_ptr()),
            ClapCacheStatus::PlanConsumed
        );

        let mut telemetry = MaybeUninit::<ClapCacheTelemetry>::uninit();
        assert_eq!(
            clap_cache_get_telemetry(cache, telemetry.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let telemetry = telemetry.assume_init();
        assert_eq!(telemetry.plans, 1);
        assert_eq!(telemetry.commits, 1);
        assert_eq!(telemetry.active_slots, 1);
        assert_eq!(telemetry.physical_bytes, 48);

        clap_cache_plan_destroy(plan);
        clap_cache_destroy(cache);
    }
}

#[test]
fn ffi_rejects_nulls_and_version_mismatches_without_unwinding() {
    unsafe {
        assert_eq!(
            clap_cache_create(ptr::null(), ptr::null_mut()),
            ClapCacheStatus::InvalidArgument
        );
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION + 1,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 1,
            max_anchors: 1,
            min_reuse_tokens: 1,
            logical_token_capacity: 10,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(
            clap_cache_create(&config, &mut cache),
            ClapCacheStatus::InvalidArgument
        );
        assert!(cache.is_null());
    }
}

#[test]
fn ffi_abort_releases_plan_and_reset_changes_epoch() {
    unsafe {
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 1,
            max_anchors: 1,
            min_reuse_tokens: 1,
            logical_token_capacity: 10,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(clap_cache_create(&config, &mut cache), ClapCacheStatus::Ok);
        let tokens = [1];
        let request = ClapCacheRequest {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRequest>() as u32,
            namespace_fingerprint: [1; 32],
            tokens: tokens.as_ptr(),
            tokens_len: 1,
            labels: labels(1),
            capabilities: 0,
            slot_capabilities: ptr::null(),
            slot_capabilities_len: 0,
            stable_boundaries: ptr::null(),
            stable_boundaries_len: 0,
            output_reserve: 0,
            estimated_bytes_per_token: 1,
            result_state: 1,
            reserved: 0,
        };
        let mut plan = ptr::null_mut();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        assert_eq!(clap_cache_abort(cache, plan), ClapCacheStatus::Ok);
        assert_eq!(clap_cache_abort(cache, plan), ClapCacheStatus::PlanConsumed);
        clap_cache_plan_destroy(plan);

        let mut epoch = 0;
        assert_eq!(clap_cache_reset(cache, &mut epoch), ClapCacheStatus::Ok);
        assert!(epoch > 1);
        clap_cache_destroy(cache);
    }
}

#[test]
fn ffi_no_capacity_is_request_local_and_next_plan_succeeds() {
    unsafe {
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 1,
            max_anchors: 1,
            min_reuse_tokens: 1,
            logical_token_capacity: 1,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(clap_cache_create(&config, &mut cache), ClapCacheStatus::Ok);
        let too_large = [1, 2];
        let mut request = ClapCacheRequest {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRequest>() as u32,
            namespace_fingerprint: [4; 32],
            tokens: too_large.as_ptr(),
            tokens_len: too_large.len(),
            labels: labels(1),
            capabilities: 0,
            slot_capabilities: ptr::null(),
            slot_capabilities_len: 0,
            stable_boundaries: ptr::null(),
            stable_boundaries_len: 0,
            output_reserve: 0,
            estimated_bytes_per_token: 1,
            result_state: 1,
            reserved: 0,
        };
        let mut plan = ptr::null_mut();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        assert_eq!(clap_cache_abort(cache, plan), ClapCacheStatus::Ok);
        clap_cache_plan_destroy(plan);
        plan = ptr::null_mut();

        let fits = [1];
        request.tokens = fits.as_ptr();
        request.tokens_len = fits.len();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        assert_eq!(clap_cache_abort(cache, plan), ClapCacheStatus::Ok);
        clap_cache_plan_destroy(plan);
        clap_cache_destroy(cache);
    }
}

#[test]
fn ffi_output_reserve_does_not_hide_a_legal_donor() {
    unsafe {
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 2,
            max_anchors: 2,
            min_reuse_tokens: 2,
            logical_token_capacity: 128,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(clap_cache_create(&config, &mut cache), ClapCacheStatus::Ok);

        let harness: Vec<i32> = (0..32).collect();
        let mut request = ClapCacheRequest {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRequest>() as u32,
            namespace_fingerprint: [7; 32],
            tokens: harness.as_ptr(),
            tokens_len: harness.len(),
            labels: labels(1),
            capabilities: 1 << 1,
            slot_capabilities: ptr::null(),
            slot_capabilities_len: 0,
            stable_boundaries: ptr::null(),
            stable_boundaries_len: 0,
            output_reserve: 1,
            estimated_bytes_per_token: 1,
            result_state: 1,
            reserved: 0,
        };
        let mut plan = ptr::null_mut();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        let commit = ClapCacheCommit {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheCommit>() as u32,
            resident_tokens: harness.len() as u64,
            actual_state: 1,
            reserved: 0,
            physical_bytes: 0,
            prefill_us_saved: 0,
        };
        let mut decision = MaybeUninit::<ClapCacheDecision>::uninit();
        assert_eq!(
            clap_cache_commit(cache, plan, &commit, decision.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let first_decision = decision.assume_init();
        let mut slot = MaybeUninit::<ClapCacheSlotInfo>::uninit();
        assert_eq!(
            clap_cache_get_slot(cache, first_decision.target_slot, slot.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let first_slot = slot.assume_init();
        assert_eq!(
            clap_cache_set_busy(
                cache,
                ClapCacheSlotRef {
                    slot: first_decision.target_slot,
                    reserved: 0,
                    generation: first_slot.generation,
                },
                0,
            ),
            ClapCacheStatus::Ok
        );
        clap_cache_plan_destroy(plan);

        let unrelated = [90, 91, 92, 93];
        request.namespace_fingerprint = [8; 32];
        request.tokens = unrelated.as_ptr();
        request.tokens_len = unrelated.len();
        request.labels = labels(2);
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        let other_commit = ClapCacheCommit {
            resident_tokens: unrelated.len() as u64,
            ..commit
        };
        assert_eq!(
            clap_cache_commit(cache, plan, &other_commit, decision.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let other_decision = decision.assume_init();
        assert_eq!(
            clap_cache_get_slot(cache, other_decision.target_slot, slot.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let other_slot = slot.assume_init();
        assert_eq!(
            clap_cache_set_busy(
                cache,
                ClapCacheSlotRef {
                    slot: other_decision.target_slot,
                    reserved: 0,
                    generation: other_slot.generation,
                },
                0,
            ),
            ClapCacheStatus::Ok
        );
        clap_cache_plan_destroy(plan);

        let mut continued = harness.clone();
        continued.extend([40, 41, 42, 43]);
        request.namespace_fingerprint = [7; 32];
        request.tokens = continued.as_ptr();
        request.tokens_len = continued.len();
        request.labels = labels(3);
        request.output_reserve = (128 - continued.len()) as u64;
        let mut materialization = [0_u8; 2];
        materialization[first_decision.target_slot as usize] = CLAP_CACHE_SLOT_MATERIALIZED
            | CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM
            | CLAP_CACHE_SLOT_COPY;
        materialization[other_decision.target_slot as usize] = CLAP_CACHE_SLOT_MATERIALIZED
            | CLAP_CACHE_SLOT_WRITABLE
            | CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM
            | CLAP_CACHE_SLOT_COPY;
        request.slot_capabilities = materialization.as_ptr();
        request.slot_capabilities_len = materialization.len();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        let mut view = MaybeUninit::<ClapCachePlanView>::uninit();
        assert_eq!(
            clap_cache_plan_view(plan, view.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let view = view.assume_init();
        assert_eq!(view.operation, 2);
        assert_eq!(view.has_donor, 1);
        assert_eq!(view.reuse_tokens, harness.len() as u64);
        assert_eq!(clap_cache_abort(cache, plan), ClapCacheStatus::Ok);
        clap_cache_plan_destroy(plan);

        materialization[first_decision.target_slot as usize] = 0;
        materialization[other_decision.target_slot as usize] = CLAP_CACHE_SLOT_WRITABLE;
        request.slot_capabilities = materialization.as_ptr();
        plan = ptr::null_mut();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        let mut excluded_view = MaybeUninit::<ClapCachePlanView>::uninit();
        assert_eq!(
            clap_cache_plan_view(plan, excluded_view.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let excluded = excluded_view.assume_init();
        assert_eq!(excluded.operation, 0);
        assert_eq!(excluded.has_donor, 0);
        assert_eq!(excluded.reuse_tokens, 0);
        assert_eq!(clap_cache_abort(cache, plan), ClapCacheStatus::Ok);
        clap_cache_plan_destroy(plan);

        let mut telemetry = MaybeUninit::<ClapCacheTelemetry>::uninit();
        assert_eq!(
            clap_cache_get_telemetry(cache, telemetry.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let telemetry = telemetry.assume_init();
        assert_eq!(telemetry.read_leases, 0);
        assert_eq!(telemetry.write_leases, 0);
        clap_cache_destroy(cache);
    }
}

#[test]
fn ffi_dynamic_registration_is_bounded_and_reports_retention_policy() {
    unsafe {
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 2,
            max_anchors: 4,
            min_reuse_tokens: 1,
            logical_token_capacity: 1,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let retention = ClapCacheRetentionConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRetentionConfig>() as u32,
            hard_max_retained_entries: 4,
            reserved: 0,
            physical_byte_budget: 1024,
            high_watermark_bytes: 900,
            low_watermark_bytes: 700,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(
            clap_cache_create_with_retention(&config, &retention, &mut cache),
            ClapCacheStatus::Ok
        );
        for expected in 2..4 {
            let mut slot = MaybeUninit::<ClapCacheSlotRef>::uninit();
            assert_eq!(
                clap_cache_register_slot(cache, slot.as_mut_ptr()),
                ClapCacheStatus::Ok
            );
            let slot = slot.assume_init();
            assert_eq!(slot.slot, expected);
            assert_eq!(slot.generation, 1);
        }
        let mut excess = MaybeUninit::<ClapCacheSlotRef>::uninit();
        assert_eq!(
            clap_cache_register_slot(cache, excess.as_mut_ptr()),
            ClapCacheStatus::NoCapacity
        );

        let mut telemetry = MaybeUninit::<ClapCacheRetentionTelemetry>::uninit();
        assert_eq!(
            clap_cache_get_retention_telemetry(cache, telemetry.as_mut_ptr()),
            ClapCacheStatus::Ok
        );
        let telemetry = telemetry.assume_init();
        assert_eq!(telemetry.total_slots, 4);
        assert_eq!(telemetry.physical_byte_budget, 1024);
        assert_eq!(telemetry.high_watermark_bytes, 900);
        assert_eq!(telemetry.low_watermark_bytes, 700);
        clap_cache_destroy(cache);
    }
}

#[test]
fn candidate_copy_reports_required_capacity_without_partial_writes() {
    unsafe {
        let config = ClapCacheConfig {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheConfig>() as u32,
            slot_count: 2,
            max_anchors: 1,
            min_reuse_tokens: 1,
            logical_token_capacity: 32,
            automatic_checkpoint_mode: 0,
            automatic_checkpoint_max: 0,
            automatic_checkpoint_min_tokens: 0,
            automatic_checkpoint_interval_tokens: 0,
            automatic_checkpoint_memory_basis_points: 0,
            reserved: 0,
            automatic_checkpoint_memory_cap_bytes: 0,
        };
        let mut cache = ptr::null_mut();
        assert_eq!(clap_cache_create(&config, &mut cache), ClapCacheStatus::Ok);
        let tokens = [1, 2, 3];
        let mut generation = 0;
        assert_eq!(
            clap_cache_confirm(
                cache,
                ClapCacheSlotRef {
                    slot: 0,
                    reserved: 0,
                    generation: 1,
                },
                tokens.as_ptr(),
                tokens.len(),
                1,
                0,
                0,
                &mut generation,
            ),
            ClapCacheStatus::Ok
        );
        let request = ClapCacheRequest {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRequest>() as u32,
            namespace_fingerprint: [0; 32],
            tokens: tokens.as_ptr(),
            tokens_len: tokens.len(),
            labels: labels(0),
            capabilities: 0,
            slot_capabilities: ptr::null(),
            slot_capabilities_len: 0,
            stable_boundaries: ptr::null(),
            stable_boundaries_len: 0,
            output_reserve: 0,
            estimated_bytes_per_token: 0,
            result_state: 1,
            reserved: 0,
        };
        let mut plan = ptr::null_mut();
        assert_eq!(
            clap_cache_plan(cache, &request, &mut plan),
            ClapCacheStatus::Ok
        );
        let mut count = 0;
        assert_eq!(
            clap_cache_plan_candidates(plan, ptr::null_mut(), 0, &mut count),
            ClapCacheStatus::NoCapacity
        );
        assert_eq!(count, 1);
        let mut candidate = MaybeUninit::<ClapCacheCandidateEvaluation>::uninit();
        assert_eq!(
            clap_cache_plan_candidates(plan, candidate.as_mut_ptr(), 1, &mut count),
            ClapCacheStatus::Ok
        );
        let candidate = candidate.assume_init();
        assert_eq!(candidate.slot, 0);
        assert_eq!(candidate.shared_prefix_tokens, 3);
        assert_eq!(candidate.selected, 1);
        clap_cache_plan_destroy(plan);
        clap_cache_destroy(cache);
    }
}
