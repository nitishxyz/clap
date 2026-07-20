//! Panic-contained stable C ABI for `clap-cache-core`.

use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Mutex;

use clap_cache_core::{
    AutomaticCheckpointConfig, CacheManager, Capabilities, Commit, Config, Error, Labels,
    Namespace, PlanRequest, Priority, RetentionConfig, Scope, SlotCapabilities, SlotState,
};

pub const CLAP_CACHE_ABI_VERSION: u32 = 3;
pub const CLAP_CACHE_SLOT_MATERIALIZED: u8 = SlotCapabilities::MATERIALIZED;
pub const CLAP_CACHE_SLOT_WRITABLE: u8 = SlotCapabilities::WRITABLE;
pub const CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM: u8 = SlotCapabilities::PARTIAL_SUFFIX_TRIM;
pub const CLAP_CACHE_SLOT_COPY: u8 = SlotCapabilities::COPY;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClapCacheStatus {
    Ok = 0,
    InvalidArgument = 1,
    NoCapacity = 2,
    StalePlan = 3,
    PlanConsumed = 4,
    SlotBusy = 5,
    Unsupported = 6,
    Panic = 255,
}

impl From<Error> for ClapCacheStatus {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidArgument => Self::InvalidArgument,
            Error::NoCapacity => Self::NoCapacity,
            Error::StalePlan => Self::StalePlan,
            Error::PlanConsumed => Self::PlanConsumed,
            Error::SlotBusy => Self::SlotBusy,
            Error::Unsupported => Self::Unsupported,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheConfig {
    pub version: u32,
    pub struct_size: u32,
    pub slot_count: u32,
    pub max_anchors: u32,
    pub min_reuse_tokens: u64,
    pub logical_token_capacity: u64,
    /// Zero uses the safe default, one enables, and two disables.
    pub automatic_checkpoint_mode: u32,
    pub automatic_checkpoint_max: u32,
    pub automatic_checkpoint_min_tokens: u64,
    pub automatic_checkpoint_interval_tokens: u64,
    pub automatic_checkpoint_memory_basis_points: u32,
    pub reserved: u32,
    pub automatic_checkpoint_memory_cap_bytes: u64,
}

/// Additive dynamic-retention configuration.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheRetentionConfig {
    pub version: u32,
    pub struct_size: u32,
    pub hard_max_retained_entries: u32,
    pub reserved: u32,
    /// Zero disables byte policy. Otherwise low <= high <= budget is required.
    pub physical_byte_budget: u64,
    pub high_watermark_bytes: u64,
    pub low_watermark_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheLabels {
    pub version: u32,
    pub struct_size: u32,
    pub tenant: u64,
    pub project: u64,
    pub harness: u64,
    pub agent: u64,
    pub session: u64,
    pub scope: u32,
    pub priority: u32,
    pub side_request: u8,
    pub reserved: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheRequest {
    pub version: u32,
    pub struct_size: u32,
    pub namespace_fingerprint: [u8; 32],
    pub tokens: *const i32,
    pub tokens_len: usize,
    pub labels: ClapCacheLabels,
    pub capabilities: u64,
    pub slot_capabilities: *const u8,
    pub slot_capabilities_len: usize,
    pub stable_boundaries: *const u64,
    pub stable_boundaries_len: usize,
    pub output_reserve: u64,
    pub estimated_bytes_per_token: u64,
    pub result_state: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ClapCacheSlotRef {
    pub slot: u32,
    pub reserved: u32,
    pub generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCachePlanView {
    pub version: u32,
    pub struct_size: u32,
    pub epoch: u64,
    pub operation: u32,
    pub has_donor: u8,
    pub reserved0: [u8; 3],
    pub target: ClapCacheSlotRef,
    pub donor: ClapCacheSlotRef,
    pub reuse_tokens: u64,
    pub anchor_tokens: u64,
    pub eviction_count: u32,
    pub result_state: u32,
    pub decision_us: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheCandidateEvaluation {
    pub version: u32,
    pub struct_size: u32,
    pub slot: u32,
    pub state: u32,
    pub generation: u64,
    pub shared_prefix_tokens: u64,
    pub namespace_compatible: u8,
    pub model_compatible: u8,
    pub session_compatible: u8,
    pub generation_compatible: u8,
    pub busy_eligible: u8,
    pub lease_eligible: u8,
    pub materialized: u8,
    pub trim_eligible: u8,
    pub copy_eligible: u8,
    pub eligible: u8,
    pub selected: u8,
    pub reserved: u8,
    pub rejection: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheCommit {
    pub version: u32,
    pub struct_size: u32,
    pub resident_tokens: u64,
    pub actual_state: u32,
    pub reserved: u32,
    pub physical_bytes: u64,
    pub prefill_us_saved: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheDecision {
    pub version: u32,
    pub struct_size: u32,
    pub hit: u8,
    pub has_donor: u8,
    pub reserved0: [u8; 2],
    pub operation: u32,
    pub scope: u32,
    pub target_slot: u32,
    pub donor_slot: u32,
    pub planned_reuse_tokens: u64,
    pub realized_reuse_tokens: u64,
    pub decision_us: u64,
    pub eviction_count: u32,
    pub reserved1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheTelemetry {
    pub version: u32,
    pub struct_size: u32,
    pub plans: u64,
    pub hits: u64,
    pub misses: u64,
    pub commits: u64,
    pub aborts: u64,
    pub stale_commits: u64,
    pub evictions: u64,
    pub resets: u64,
    pub planned_reuse_tokens: u64,
    pub realized_reuse_tokens: u64,
    pub prefill_us_saved: u64,
    pub active_slots: u32,
    pub anchors: u32,
    pub read_leases: u32,
    pub write_leases: u32,
    pub prefix_nodes: u64,
    pub physical_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheRetentionTelemetry {
    pub version: u32,
    pub struct_size: u32,
    pub total_slots: u32,
    pub session_slots: u32,
    pub anchor_slots: u32,
    pub active_slots: u32,
    pub total_bytes: u64,
    pub session_bytes: u64,
    pub anchor_bytes: u64,
    pub automatic_checkpoint_slots: u32,
    pub reserved0: u32,
    pub automatic_checkpoint_bytes: u64,
    pub automatic_checkpoint_byte_budget: u64,
    pub active_bytes: u64,
    pub physical_byte_budget: u64,
    pub high_watermark_bytes: u64,
    pub low_watermark_bytes: u64,
    pub under_pressure: u8,
    pub reserved: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClapCacheSlotInfo {
    pub version: u32,
    pub struct_size: u32,
    pub generation: u64,
    pub resident_len: u64,
    pub state: u32,
    pub busy: u8,
    pub write_leased: u8,
    pub reserved0: [u8; 2],
    pub read_leases: u32,
    pub scope: u32,
    pub session: u64,
    pub last_used: u64,
    pub reuse_count: u64,
    pub physical_bytes: u64,
}

pub struct ClapCache {
    manager: Mutex<CacheManager>,
}

pub struct ClapCachePlan {
    owner: *const ClapCache,
    plan: Option<clap_cache_core::Plan>,
}

fn ffi_status(f: impl FnOnce() -> Result<(), ClapCacheStatus>) -> ClapCacheStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => ClapCacheStatus::Ok,
        Ok(Err(status)) => status,
        Err(_) => ClapCacheStatus::Panic,
    }
}

fn valid_header(version: u32, struct_size: u32, expected: usize) -> bool {
    version == CLAP_CACHE_ABI_VERSION && struct_size as usize >= expected
}

fn slot_state(value: u32) -> Result<SlotState, ClapCacheStatus> {
    match value {
        0 => Ok(SlotState::Empty),
        1 => Ok(SlotState::Session),
        2 => Ok(SlotState::PromptBoundary),
        3 => Ok(SlotState::Anchor),
        _ => Err(ClapCacheStatus::InvalidArgument),
    }
}

fn labels(value: ClapCacheLabels) -> Result<Labels, ClapCacheStatus> {
    if !valid_header(
        value.version,
        value.struct_size,
        size_of::<ClapCacheLabels>(),
    ) {
        return Err(ClapCacheStatus::InvalidArgument);
    }
    let scope = match value.scope {
        0 => Scope::None,
        1 => Scope::Session,
        2 => Scope::Agent,
        3 => Scope::Project,
        4 => Scope::Harness,
        5 => Scope::Tenant,
        _ => return Err(ClapCacheStatus::InvalidArgument),
    };
    let priority = match value.priority {
        0 => Priority::Background,
        1 => Priority::Interactive,
        _ => return Err(ClapCacheStatus::InvalidArgument),
    };
    Ok(Labels {
        tenant: value.tenant,
        project: value.project,
        harness: value.harness,
        agent: value.agent,
        session: value.session,
        scope,
        priority,
        side_request: value.side_request != 0,
    })
}

fn lock(cache: &ClapCache) -> Result<std::sync::MutexGuard<'_, CacheManager>, ClapCacheStatus> {
    cache.manager.lock().map_err(|_| ClapCacheStatus::Panic)
}

fn core_config(config: ClapCacheConfig) -> Result<Config, ClapCacheStatus> {
    if !valid_header(
        config.version,
        config.struct_size,
        size_of::<ClapCacheConfig>(),
    ) {
        return Err(ClapCacheStatus::InvalidArgument);
    }
    if config.automatic_checkpoint_mode > 2 {
        return Err(ClapCacheStatus::InvalidArgument);
    }
    let defaults = AutomaticCheckpointConfig::default();
    Ok(Config {
        slot_count: config.slot_count,
        max_anchors: config.max_anchors,
        min_reuse_tokens: usize::try_from(config.min_reuse_tokens)
            .map_err(|_| ClapCacheStatus::InvalidArgument)?,
        logical_token_capacity: usize::try_from(config.logical_token_capacity)
            .map_err(|_| ClapCacheStatus::InvalidArgument)?,
        automatic_checkpoints: AutomaticCheckpointConfig {
            enabled: config.automatic_checkpoint_mode != 2,
            minimum_prompt_tokens: if config.automatic_checkpoint_min_tokens == 0 {
                defaults.minimum_prompt_tokens
            } else {
                usize::try_from(config.automatic_checkpoint_min_tokens)
                    .map_err(|_| ClapCacheStatus::InvalidArgument)?
            },
            target_interval_tokens: if config.automatic_checkpoint_interval_tokens == 0 {
                defaults.target_interval_tokens
            } else {
                usize::try_from(config.automatic_checkpoint_interval_tokens)
                    .map_err(|_| ClapCacheStatus::InvalidArgument)?
            },
            max_checkpoints: if config.automatic_checkpoint_max == 0 {
                defaults.max_checkpoints
            } else {
                config.automatic_checkpoint_max
            },
            memory_budget_basis_points: if config.automatic_checkpoint_memory_basis_points == 0 {
                defaults.memory_budget_basis_points
            } else {
                config.automatic_checkpoint_memory_basis_points
            },
            memory_budget_cap_bytes: config.automatic_checkpoint_memory_cap_bytes,
        },
    })
}

/// Creates one manager for one resident model/physical worker context.
///
/// # Safety
/// `config` must be readable and `out_cache` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_create(
    config: *const ClapCacheConfig,
    out_cache: *mut *mut ClapCache,
) -> ClapCacheStatus {
    ffi_status(|| {
        if config.is_null() || out_cache.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: pointers were checked and are required by the ABI to be valid.
        let config = unsafe { *config };
        let core = core_config(config)?;
        let manager = CacheManager::new(
            core,
            RetentionConfig {
                hard_max_retained_entries: config.slot_count,
                physical_byte_budget: None,
                high_watermark_bytes: 0,
                low_watermark_bytes: 0,
            },
        )
        .map_err(ClapCacheStatus::from)?;
        let cache = Box::new(ClapCache {
            manager: Mutex::new(manager),
        });
        // SAFETY: out_cache is valid and ownership transfers to the caller.
        unsafe { *out_cache = Box::into_raw(cache) };
        Ok(())
    })
}

/// Creates a dynamically growable manager with an optional physical-byte policy.
///
/// # Safety
/// Inputs must be readable and `out_cache` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_create_with_retention(
    config: *const ClapCacheConfig,
    retention: *const ClapCacheRetentionConfig,
    out_cache: *mut *mut ClapCache,
) -> ClapCacheStatus {
    ffi_status(|| {
        if config.is_null() || retention.is_null() || out_cache.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked non-null; callers provide readable ABI values.
        let config = unsafe { *config };
        let retention = unsafe { *retention };
        if !valid_header(
            retention.version,
            retention.struct_size,
            size_of::<ClapCacheRetentionConfig>(),
        ) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        let manager = CacheManager::new(
            core_config(config)?,
            RetentionConfig {
                hard_max_retained_entries: retention.hard_max_retained_entries,
                physical_byte_budget: (retention.physical_byte_budget != 0)
                    .then_some(retention.physical_byte_budget),
                high_watermark_bytes: retention.high_watermark_bytes,
                low_watermark_bytes: retention.low_watermark_bytes,
            },
        )
        .map_err(ClapCacheStatus::from)?;
        let cache = Box::new(ClapCache {
            manager: Mutex::new(manager),
        });
        // SAFETY: out pointer is writable and receives ownership.
        unsafe { *out_cache = Box::into_raw(cache) };
        Ok(())
    })
}

/// # Safety
/// `cache` must be null or a live handle returned by `clap_cache_create`, and
/// must be relinquished exactly once after all owned plans are destroyed.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_destroy(cache: *mut ClapCache) {
    if cache.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller relinquishes a pointer returned by clap_cache_create once.
        drop(unsafe { Box::from_raw(cache) });
    }));
}

/// # Safety
/// All pointers must reference valid ABI objects; request token storage must
/// remain readable for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_plan(
    cache: *mut ClapCache,
    request: *const ClapCacheRequest,
    out_plan: *mut *mut ClapCachePlan,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || request.is_null() || out_plan.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: non-null ABI pointers are caller-owned for this call.
        let request = unsafe { *request };
        if !valid_header(
            request.version,
            request.struct_size,
            size_of::<ClapCacheRequest>(),
        ) || (request.tokens.is_null() && request.tokens_len != 0)
            || (request.slot_capabilities.is_null() && request.slot_capabilities_len != 0)
            || (request.stable_boundaries.is_null() && request.stable_boundaries_len != 0)
        {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: pointer/length describe a readable slice for this call.
        let tokens = unsafe { slice::from_raw_parts(request.tokens, request.tokens_len) };
        // SAFETY: pointer/length were validated above and are borrowed only for this call.
        let slot_capabilities = if request.slot_capabilities_len == 0 {
            None
        } else {
            Some(unsafe {
                slice::from_raw_parts(request.slot_capabilities, request.slot_capabilities_len)
            })
        };
        let slot_capabilities = slot_capabilities.map(|slots| {
            slots
                .iter()
                .copied()
                .map(SlotCapabilities)
                .collect::<Vec<_>>()
        });
        let stable_boundaries = if request.stable_boundaries_len == 0 {
            Vec::new()
        } else {
            // SAFETY: pointer/length were validated above and are borrowed only for this call.
            unsafe {
                slice::from_raw_parts(request.stable_boundaries, request.stable_boundaries_len)
            }
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ClapCacheStatus::InvalidArgument)?
        };
        let core_request = PlanRequest {
            namespace: Namespace(request.namespace_fingerprint),
            tokens,
            stable_boundaries: &stable_boundaries,
            labels: labels(request.labels)?,
            capabilities: Capabilities(request.capabilities),
            slot_capabilities: slot_capabilities.as_deref(),
            output_reserve: usize::try_from(request.output_reserve)
                .map_err(|_| ClapCacheStatus::InvalidArgument)?,
            estimated_bytes_per_token: request.estimated_bytes_per_token,
            result_state: slot_state(request.result_state)?,
        };
        // SAFETY: checked above; no reference escapes the call.
        let cache_ref = unsafe { &*cache };
        let plan = lock(cache_ref)?
            .plan(core_request)
            .map_err(ClapCacheStatus::from)?;
        let handle = Box::new(ClapCachePlan {
            owner: cache.cast_const(),
            plan: Some(plan),
        });
        // SAFETY: out pointer is valid and ownership transfers to caller.
        unsafe { *out_plan = Box::into_raw(handle) };
        Ok(())
    })
}

/// # Safety
/// `plan` must be live and `out_view` must be writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_plan_view(
    plan: *const ClapCachePlan,
    out_view: *mut ClapCachePlanView,
) -> ClapCacheStatus {
    ffi_status(|| {
        if plan.is_null() || out_view.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked non-null and caller guarantees live plan handle.
        let plan = unsafe { &*plan }
            .plan
            .as_ref()
            .ok_or(ClapCacheStatus::PlanConsumed)?;
        let donor = plan
            .donor
            .as_ref()
            .map_or(ClapCacheSlotRef::default(), |slot| ClapCacheSlotRef {
                slot: slot.slot,
                reserved: 0,
                generation: slot.generation,
            });
        let view = ClapCachePlanView {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCachePlanView>() as u32,
            epoch: plan.epoch,
            operation: plan.operation as u32,
            has_donor: u8::from(plan.donor.is_some()),
            reserved0: [0; 3],
            target: ClapCacheSlotRef {
                slot: plan.target.slot,
                reserved: 0,
                generation: plan.target.generation,
            },
            donor,
            reuse_tokens: plan.reuse_tokens as u64,
            anchor_tokens: plan.anchor_tokens as u64,
            eviction_count: plan.evictions.len() as u32,
            result_state: plan.result_state as u32,
            decision_us: plan.decision_us,
        };
        // SAFETY: caller provided writable result storage.
        unsafe { *out_view = view };
        Ok(())
    })
}

/// # Safety
/// `plan` and `out_count` must be live, and `out_slots` must hold `capacity`
/// writable elements when capacity is nonzero.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_plan_evictions(
    plan: *const ClapCachePlan,
    out_slots: *mut ClapCacheSlotRef,
    capacity: usize,
    out_count: *mut usize,
) -> ClapCacheStatus {
    ffi_status(|| {
        if plan.is_null() || out_count.is_null() || (out_slots.is_null() && capacity != 0) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked non-null and caller guarantees live plan handle.
        let plan = unsafe { &*plan }
            .plan
            .as_ref()
            .ok_or(ClapCacheStatus::PlanConsumed)?;
        // SAFETY: out_count is writable.
        unsafe { *out_count = plan.evictions.len() };
        if capacity < plan.evictions.len() {
            return Err(ClapCacheStatus::NoCapacity);
        }
        for (index, slot) in plan.evictions.iter().enumerate() {
            // SAFETY: capacity was checked for every written element.
            unsafe {
                ptr::write(
                    out_slots.add(index),
                    ClapCacheSlotRef {
                        slot: slot.slot,
                        reserved: 0,
                        generation: slot.generation,
                    },
                )
            };
        }
        Ok(())
    })
}

/// Copies the ordered exact prefix lengths that Rust authorized for physical
/// snapshotting during this request.
///
/// # Safety
/// `plan` and `out_count` must be live, and `out_boundaries` must hold
/// `capacity` writable elements when capacity is nonzero.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_plan_anchor_boundaries(
    plan: *const ClapCachePlan,
    out_boundaries: *mut u64,
    capacity: usize,
    out_count: *mut usize,
) -> ClapCacheStatus {
    ffi_status(|| {
        if plan.is_null() || out_count.is_null() || (out_boundaries.is_null() && capacity != 0) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked non-null and caller guarantees a live plan handle.
        let plan = unsafe { &*plan }
            .plan
            .as_ref()
            .ok_or(ClapCacheStatus::PlanConsumed)?;
        // SAFETY: out_count is writable.
        unsafe { *out_count = plan.anchor_boundaries.len() };
        if capacity < plan.anchor_boundaries.len() {
            return Err(ClapCacheStatus::NoCapacity);
        }
        for (index, boundary) in plan.anchor_boundaries.iter().copied().enumerate() {
            // SAFETY: capacity was checked for every written element.
            unsafe { ptr::write(out_boundaries.add(index), boundary as u64) };
        }
        Ok(())
    })
}

/// Copies the plan's immutable, ordered candidate diagnostics into caller-owned
/// storage. A null output with zero capacity queries the required element count.
///
/// # Safety
/// `plan` and `out_count` must be live, and `out_candidates` must hold
/// `capacity` writable elements when capacity is nonzero.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_plan_candidates(
    plan: *const ClapCachePlan,
    out_candidates: *mut ClapCacheCandidateEvaluation,
    capacity: usize,
    out_count: *mut usize,
) -> ClapCacheStatus {
    ffi_status(|| {
        if plan.is_null() || out_count.is_null() || (out_candidates.is_null() && capacity != 0) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked non-null and caller guarantees a live plan handle.
        let plan = unsafe { &*plan }
            .plan
            .as_ref()
            .ok_or(ClapCacheStatus::PlanConsumed)?;
        // SAFETY: out_count is writable.
        unsafe { *out_count = plan.candidates.len() };
        if capacity < plan.candidates.len() {
            return Err(ClapCacheStatus::NoCapacity);
        }
        for (index, candidate) in plan.candidates.iter().enumerate() {
            let value = ClapCacheCandidateEvaluation {
                version: CLAP_CACHE_ABI_VERSION,
                struct_size: size_of::<ClapCacheCandidateEvaluation>() as u32,
                slot: candidate.slot,
                state: candidate.state as u32,
                generation: candidate.generation,
                shared_prefix_tokens: candidate.shared_prefix_tokens as u64,
                namespace_compatible: u8::from(candidate.namespace_compatible),
                model_compatible: u8::from(candidate.model_compatible),
                session_compatible: u8::from(candidate.session_compatible),
                generation_compatible: u8::from(candidate.generation_compatible),
                busy_eligible: u8::from(candidate.busy_eligible),
                lease_eligible: u8::from(candidate.lease_eligible),
                materialized: u8::from(candidate.materialized),
                trim_eligible: u8::from(candidate.trim_eligible),
                copy_eligible: u8::from(candidate.copy_eligible),
                eligible: u8::from(candidate.eligible),
                selected: u8::from(candidate.selected),
                reserved: 0,
                rejection: candidate.rejection as u32,
            };
            // SAFETY: capacity was checked for every written element.
            unsafe { ptr::write(out_candidates.add(index), value) };
        }
        Ok(())
    })
}

/// # Safety
/// All pointers must be live, writable where applicable, and `plan` must have
/// been created by `cache` and not previously consumed.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_commit(
    cache: *mut ClapCache,
    plan: *mut ClapCachePlan,
    commit: *const ClapCacheCommit,
    out_decision: *mut ClapCacheDecision,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || plan.is_null() || commit.is_null() || out_decision.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked non-null and caller guarantees live handles.
        let handle = unsafe { &mut *plan };
        if handle.owner != cache.cast_const() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: commit points to readable ABI storage.
        let commit = unsafe { *commit };
        if !valid_header(
            commit.version,
            commit.struct_size,
            size_of::<ClapCacheCommit>(),
        ) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        let plan_id = handle
            .plan
            .as_ref()
            .ok_or(ClapCacheStatus::PlanConsumed)?
            .id;
        // SAFETY: checked non-null above.
        let cache_ref = unsafe { &*cache };
        let decision = lock(cache_ref)?
            .commit(
                plan_id,
                Commit {
                    resident_tokens: usize::try_from(commit.resident_tokens)
                        .map_err(|_| ClapCacheStatus::InvalidArgument)?,
                    actual_state: slot_state(commit.actual_state)?,
                    physical_bytes: commit.physical_bytes,
                    prefill_us_saved: commit.prefill_us_saved,
                },
            )
            .map_err(ClapCacheStatus::from)?;
        handle.plan = None;
        let exported = ClapCacheDecision {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheDecision>() as u32,
            hit: u8::from(decision.hit),
            has_donor: u8::from(decision.donor_slot.is_some()),
            reserved0: [0; 2],
            operation: decision.operation as u32,
            scope: decision.scope as u32,
            target_slot: decision.target_slot,
            donor_slot: decision.donor_slot.unwrap_or(0),
            planned_reuse_tokens: decision.planned_reuse_tokens as u64,
            realized_reuse_tokens: decision.realized_reuse_tokens as u64,
            decision_us: decision.decision_us,
            eviction_count: decision.evicted_slots.len() as u32,
            reserved1: 0,
        };
        // SAFETY: writable output pointer checked above.
        unsafe { *out_decision = exported };
        Ok(())
    })
}

/// # Safety
/// Both handles must be live and `plan` must have been created by `cache`.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_abort(
    cache: *mut ClapCache,
    plan: *mut ClapCachePlan,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || plan.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: checked pointers are live ABI handles.
        let handle = unsafe { &mut *plan };
        if handle.owner != cache.cast_const() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        let plan_id = handle
            .plan
            .as_ref()
            .ok_or(ClapCacheStatus::PlanConsumed)?
            .id;
        // SAFETY: checked non-null above.
        lock(unsafe { &*cache })?
            .abort(plan_id)
            .map_err(ClapCacheStatus::from)?;
        handle.plan = None;
        Ok(())
    })
}

/// # Safety
/// `plan` must be null or a live plan handle relinquished exactly once. Its
/// owning cache must remain live until this call returns.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_plan_destroy(plan: *mut ClapCachePlan) {
    if plan.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller relinquishes a pointer returned by clap_cache_plan once.
        let mut handle = unsafe { Box::from_raw(plan) };
        if let Some(plan) = handle.plan.take() {
            // Unconsumed destruction fails closed and releases every lease. The ABI
            // requires all plan handles to be destroyed before their owner manager.
            // SAFETY: that ownership ordering keeps owner live for this operation.
            let owner = unsafe { &*handle.owner };
            if let Ok(mut manager) = owner.manager.lock() {
                let _ = manager.abort(plan.id);
            }
        }
    }));
}

/// # Safety
/// `cache` and `out_generation` must be valid, and `tokens` must describe a
/// readable array of `tokens_len` elements for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_advance(
    cache: *mut ClapCache,
    slot: ClapCacheSlotRef,
    tokens: *const i32,
    tokens_len: usize,
    state: u32,
    busy: u8,
    physical_bytes: u64,
    out_generation: *mut u64,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out_generation.is_null() || (tokens.is_null() && tokens_len != 0) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: pointer and length are valid for this call.
        let tokens = unsafe { slice::from_raw_parts(tokens, tokens_len) };
        // SAFETY: cache is checked non-null.
        let generation = lock(unsafe { &*cache })?
            .advance(
                slot.slot,
                slot.generation,
                tokens,
                slot_state(state)?,
                busy != 0,
                physical_bytes,
            )
            .map_err(ClapCacheStatus::from)?;
        // SAFETY: out pointer is checked non-null.
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// # Safety
/// `cache` and `out_generation` must be valid, and `tokens` must describe a
/// readable array of `tokens_len` elements for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_confirm(
    cache: *mut ClapCache,
    slot: ClapCacheSlotRef,
    tokens: *const i32,
    tokens_len: usize,
    state: u32,
    busy: u8,
    physical_bytes: u64,
    out_generation: *mut u64,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out_generation.is_null() || (tokens.is_null() && tokens_len != 0) {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: pointers were checked and remain borrowed for this call.
        let tokens = unsafe { slice::from_raw_parts(tokens, tokens_len) };
        let generation = lock(unsafe { &*cache })?
            .confirm(
                slot.slot,
                slot.generation,
                tokens,
                slot_state(state)?,
                busy != 0,
                physical_bytes,
            )
            .map_err(ClapCacheStatus::from)?;
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// # Safety
/// `cache` must be a live manager handle.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_set_busy(
    cache: *mut ClapCache,
    slot: ClapCacheSlotRef,
    busy: u8,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache is checked non-null.
        lock(unsafe { &*cache })?
            .set_busy(slot.slot, slot.generation, busy != 0)
            .map_err(ClapCacheStatus::from)
    })
}

/// Registers one physical slot and returns its stable ID/generation.
///
/// # Safety
/// `cache` must be live and `out_slot` writable.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_register_slot(
    cache: *mut ClapCache,
    out_slot: *mut ClapCacheSlotRef,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out_slot.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache and output pointers are checked non-null.
        let slot = lock(unsafe { &*cache })?
            .register_slot()
            .map_err(ClapCacheStatus::from)?;
        unsafe {
            *out_slot = ClapCacheSlotRef {
                slot: slot.slot,
                reserved: 0,
                generation: slot.generation,
            }
        };
        Ok(())
    })
}

/// Changes pressure protection for an exact structural anchor generation.
///
/// # Safety
/// `cache` must be a live manager handle.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_set_anchor_protected(
    cache: *mut ClapCache,
    slot: ClapCacheSlotRef,
    protected: u8,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache is checked non-null.
        lock(unsafe { &*cache })?
            .set_anchor_protected(slot.slot, slot.generation, protected != 0)
            .map_err(ClapCacheStatus::from)
    })
}

/// # Safety
/// `cache` must be live and `out_generation` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_invalidate(
    cache: *mut ClapCache,
    slot: ClapCacheSlotRef,
    out_generation: *mut u64,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out_generation.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache and output pointers are checked non-null.
        let generation = lock(unsafe { &*cache })?
            .invalidate(slot.slot, slot.generation)
            .map_err(ClapCacheStatus::from)?;
        unsafe { *out_generation = generation };
        Ok(())
    })
}

/// # Safety
/// `cache` must be live and `out_epoch` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_reset(
    cache: *mut ClapCache,
    out_epoch: *mut u64,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out_epoch.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: pointers checked non-null.
        let epoch = lock(unsafe { &*cache })?.reset();
        unsafe { *out_epoch = epoch };
        Ok(())
    })
}

/// # Safety
/// `cache` must be live and `out` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_get_telemetry(
    cache: *const ClapCache,
    out: *mut ClapCacheTelemetry,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache is checked non-null.
        let telemetry = lock(unsafe { &*cache })?.telemetry();
        let exported = ClapCacheTelemetry {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheTelemetry>() as u32,
            plans: telemetry.plans,
            hits: telemetry.hits,
            misses: telemetry.misses,
            commits: telemetry.commits,
            aborts: telemetry.aborts,
            stale_commits: telemetry.stale_commits,
            evictions: telemetry.evictions,
            resets: telemetry.resets,
            planned_reuse_tokens: telemetry.planned_reuse_tokens,
            realized_reuse_tokens: telemetry.realized_reuse_tokens,
            prefill_us_saved: telemetry.prefill_us_saved,
            active_slots: telemetry.active_slots,
            anchors: telemetry.anchors,
            read_leases: telemetry.read_leases,
            write_leases: telemetry.write_leases,
            prefix_nodes: telemetry.prefix_nodes,
            physical_bytes: telemetry.physical_bytes,
        };
        // SAFETY: output pointer checked non-null.
        unsafe { *out = exported };
        Ok(())
    })
}

/// # Safety
/// `cache` must be live and `out` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_get_retention_telemetry(
    cache: *const ClapCache,
    out: *mut ClapCacheRetentionTelemetry,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache is checked non-null.
        let telemetry = lock(unsafe { &*cache })?.telemetry();
        let exported = ClapCacheRetentionTelemetry {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheRetentionTelemetry>() as u32,
            total_slots: telemetry.total_slots,
            session_slots: telemetry.session_slots,
            anchor_slots: telemetry.anchors,
            active_slots: telemetry.active_slots,
            total_bytes: telemetry.physical_bytes,
            session_bytes: telemetry.session_bytes,
            anchor_bytes: telemetry.anchor_bytes,
            automatic_checkpoint_slots: telemetry.automatic_checkpoint_slots,
            reserved0: 0,
            automatic_checkpoint_bytes: telemetry.automatic_checkpoint_bytes,
            automatic_checkpoint_byte_budget: telemetry.automatic_checkpoint_byte_budget,
            active_bytes: telemetry.active_bytes,
            physical_byte_budget: telemetry.physical_byte_budget,
            high_watermark_bytes: telemetry.high_watermark_bytes,
            low_watermark_bytes: telemetry.low_watermark_bytes,
            under_pressure: u8::from(telemetry.under_pressure),
            reserved: [0; 7],
        };
        // SAFETY: output pointer checked non-null.
        unsafe { *out = exported };
        Ok(())
    })
}

/// # Safety
/// `cache` must be live and `out` writable for this call.
#[no_mangle]
pub unsafe extern "C" fn clap_cache_get_slot(
    cache: *const ClapCache,
    slot: u32,
    out: *mut ClapCacheSlotInfo,
) -> ClapCacheStatus {
    ffi_status(|| {
        if cache.is_null() || out.is_null() {
            return Err(ClapCacheStatus::InvalidArgument);
        }
        // SAFETY: cache is checked non-null.
        let slot = lock(unsafe { &*cache })?
            .slot(slot)
            .ok_or(ClapCacheStatus::InvalidArgument)?;
        let exported = ClapCacheSlotInfo {
            version: CLAP_CACHE_ABI_VERSION,
            struct_size: size_of::<ClapCacheSlotInfo>() as u32,
            generation: slot.generation,
            resident_len: slot.resident_len as u64,
            state: slot.state as u32,
            busy: u8::from(slot.busy),
            write_leased: u8::from(slot.write_leased),
            reserved0: [0; 2],
            read_leases: slot.read_leases,
            scope: slot.labels.scope as u32,
            session: slot.labels.session,
            last_used: slot.last_used,
            reuse_count: slot.reuse_count,
            physical_bytes: slot.physical_bytes,
        };
        // SAFETY: output pointer checked non-null.
        unsafe { *out = exported };
        Ok(())
    })
}
