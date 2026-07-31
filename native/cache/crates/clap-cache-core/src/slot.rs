//! Logical slot state and pure ranking helpers shared by the coordinator.

use crate::{Capabilities, Generation, Labels, Namespace, PlanId, PlanRequest, Scope, SlotId, SlotState};

#[derive(Clone, Debug)]
pub(crate) struct Slot {
    pub(crate) id: SlotId,
    pub(crate) generation: Generation,
    pub(crate) namespace: Namespace,
    pub(crate) tokens: Vec<i32>,
    pub(crate) state: SlotState,
    pub(crate) busy: bool,
    pub(crate) read_leases: u32,
    pub(crate) writer: Option<PlanId>,
    pub(crate) labels: Labels,
    pub(crate) last_used: u64,
    pub(crate) reuse_count: u64,
    pub(crate) physical_bytes: u64,
    pub(crate) saved_us: u64,
    pub(crate) protected: bool,
}

impl Slot {
    pub(crate) fn new(id: SlotId) -> Self {
        Self {
            id,
            generation: 1,
            namespace: Namespace::default(),
            tokens: Vec::new(),
            state: SlotState::Empty,
            busy: false,
            read_leases: 0,
            writer: None,
            labels: Labels::default(),
            last_used: 0,
            reuse_count: 0,
            physical_bytes: 0,
            saved_us: 0,
            protected: false,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.state == SlotState::Empty || self.tokens.is_empty()
    }
}

pub(crate) fn common_prefix(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

pub(crate) fn donor_rank(slot: &Slot, request: &PlanRequest<'_>) -> (u8, u8, u64) {
    let same_session = ((slot.labels.session == 0 && request.labels.session == 0)
        || (slot.labels.session != 0 && slot.labels.session == request.labels.session))
        as u8;
    let cheap = request
        .capabilities
        .contains(Capabilities::ZERO_COPY_BRANCH) as u8;
    (same_session, cheap, slot.last_used)
}

/// Lower values are evicted first. Stable slot ID is used by callers as the
/// final deterministic tie-breaker.
pub(crate) fn eviction_value(slot: &Slot) -> (u32, u8, u8, u64, u64, u64, usize) {
    (
        slot.labels.priority as u32,
        (!slot.labels.side_request) as u8,
        (slot.state == SlotState::Anchor) as u8,
        slot.reuse_count,
        slot.saved_us,
        slot.last_used,
        slot.tokens.len(),
    )
}

pub(crate) fn anchor_eviction_value(slot: &Slot) -> (u32, u8, u64, u64, u64, usize) {
    let structural = matches!(slot.labels.scope, Scope::Harness | Scope::Tenant) as u8;
    (
        slot.labels.priority as u32,
        structural,
        slot.reuse_count,
        slot.saved_us,
        slot.last_used,
        slot.tokens.len(),
    )
}
