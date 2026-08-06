//! Stable C ABI type layouts for `clap-cache-ffi`. Layout changes are ABI
//! changes and must bump `CLAP_CACHE_ABI_VERSION`.

use clap_cache_core::Error;

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
    /// Zero disables the per-session retained-anchor cap.
    pub max_anchors_per_session: u32,
    /// Zero disables the per-session automatic-checkpoint cap.
    pub automatic_checkpoint_max_per_session: u32,
    /// Zero disables the per-session policy-accounted anchor byte cap.
    pub max_anchor_bytes_per_session: u64,
    /// Zero disables wall-clock idle expiry for non-zero sessions.
    pub session_idle_ttl_ms: u64,
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
    pub session_policy_evictions: u64,
    pub session_budget_rejections: u64,
    pub anchor_publications: u64,
    pub anchor_publication_skips: u64,
    pub expired_slots: u64,
    pub expired_accounted_bytes: u64,
    pub released_session_slots: u64,
    pub released_session_accounted_bytes: u64,
    pub anchor_accounted_bytes: u64,
    pub max_anchor_bytes_per_session: u64,
    pub session_idle_ttl_ms: u64,
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
    pub last_used_ms: u64,
    pub reuse_count: u64,
    pub physical_bytes: u64,
    pub accounted_bytes: u64,
}
