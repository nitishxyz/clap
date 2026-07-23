//! Backend-independent logical KV-cache policy.
//! Rust owns exact token metadata and reservations; native adapters own all
//! physical cache objects and execute plans outside this crate.

mod radix;

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Instant;

use radix::RadixIndex;

pub type SlotId = u32;
pub type Generation = u64;
pub type PlanId = u64;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Namespace(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum SlotState {
    #[default]
    Empty = 0,
    Session = 1,
    PromptBoundary = 2,
    Anchor = 3,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum Scope {
    #[default]
    None = 0,
    Session = 1,
    Agent = 2,
    Project = 3,
    Harness = 4,
    Tenant = 5,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum Priority {
    Background = 0,
    #[default]
    Normal = 1,
    Interactive = 2,
}
/// Backend abilities are explicit and may be refreshed for every request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Capabilities(pub u64);

impl Capabilities {
    pub const PARTIAL_SUFFIX_TRIM: u64 = 1 << 0;
    pub const PARTIAL_PREFIX_BRANCH: u64 = 1 << 1;
    pub const WHOLE_STATE_COPY: u64 = 1 << 2;
    pub const SAFE_BUSY_DONOR: u64 = 1 << 3;
    pub const ZERO_COPY_BRANCH: u64 = 1 << 4;
    pub const PROMPT_BOUNDARY_SNAPSHOT: u64 = 1 << 5;
    pub const SLIDING_WINDOW: u64 = 1 << 6;
    pub const RECURRENT_OR_HYBRID: u64 = 1 << 7;
    pub const UNIFIED_STORAGE: u64 = 1 << 8;
    pub const RELIABLE_RESIDENT_LENGTH: u64 = 1 << 9;
    pub const KV_QUANTIZED: u64 = 1 << 10;
    /// The backend must execute the final prompt token to materialize logits,
    /// so an exact full-prompt match can only reuse through the prior token.
    pub const RETAIN_LAST_TOKEN_FOR_LOGITS: u64 = 1 << 11;

    pub const fn contains(self, flag: u64) -> bool {
        self.0 & flag == flag
    }
}

/// Request-local facts about the backend object associated with one logical
/// slot. Adapters refresh these before every plan so Rust only selects an
/// operation that the specific donor and target can materialize.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SlotCapabilities(pub u8);

impl SlotCapabilities {
    pub const MATERIALIZED: u8 = 1 << 0;
    pub const WRITABLE: u8 = 1 << 1;
    pub const PARTIAL_SUFFIX_TRIM: u8 = 1 << 2;
    pub const COPY: u8 = 1 << 3;
    pub const ALL: Self = Self(Self::MATERIALIZED | Self::WRITABLE | Self::PARTIAL_SUFFIX_TRIM | Self::COPY);

    pub const fn contains(self, flag: u8) -> bool {
        self.0 & flag == flag
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Labels {
    pub tenant: u64,
    pub project: u64,
    pub harness: u64,
    pub agent: u64,
    pub session: u64,
    pub scope: Scope,
    pub priority: Priority,
    pub side_request: bool,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub slot_count: u32,
    pub min_reuse_tokens: usize,
    pub logical_token_capacity: usize,
    pub max_anchors: u32,
    pub automatic_checkpoints: AutomaticCheckpointConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutomaticCheckpointConfig {
    pub enabled: bool,
    pub minimum_prompt_tokens: usize,
    pub target_interval_tokens: usize,
    pub max_checkpoints: u32,
    /// Maximum share of the retained physical byte budget, in basis points.
    pub memory_budget_basis_points: u32,
    /// Optional absolute cap. Zero means no additional absolute cap.
    pub memory_budget_cap_bytes: u64,
}

impl Default for AutomaticCheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            minimum_prompt_tokens: 2_048,
            target_interval_tokens: 2_048,
            max_checkpoints: 8,
            memory_budget_basis_points: 2_500,
            memory_budget_cap_bytes: 0,
        }
    }
}

/// Retained-memory policy for every cache manager.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionConfig {
    pub hard_max_retained_entries: u32,
    pub physical_byte_budget: Option<u64>,
    pub high_watermark_bytes: u64,
    pub low_watermark_bytes: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            slot_count: 16,
            min_reuse_tokens: 16,
            logical_token_capacity: usize::MAX,
            max_anchors: 8,
            automatic_checkpoints: AutomaticCheckpointConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlanRequest<'a> {
    pub namespace: Namespace,
    pub tokens: &'a [i32],
    /// Exact template-aware prefix boundaries supplied by the backend. Rust
    /// remains responsible for deciding whether one is valuable and feasible.
    pub stable_boundaries: &'a [usize],
    pub labels: Labels,
    pub capabilities: Capabilities,
    /// One entry per configured slot. `None` is retained for direct core
    /// callers; native adapters always provide an explicit snapshot.
    pub slot_capabilities: Option<&'a [SlotCapabilities]>,
    pub output_reserve: usize,
    pub estimated_bytes_per_token: u64,
    pub result_state: SlotState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Operation {
    Fresh = 0,
    Continue = 1,
    Branch = 2,
    Restore = 3,
    Noop = 4,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotRef {
    pub slot: SlotId,
    pub generation: Generation,
}

pub const MAX_PLAN_CANDIDATES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum CandidateRejection {
    None = 0,
    Namespace = 1,
    ModelDomain = 2,
    Generation = 3,
    BusyLease = 4,
    Materialization = 5,
    Session = 6,
    Nontrim = 7,
    Capability = 8,
    MinPrefix = 9,
    Capacity = 10,
    AbsentAnchor = 11,
    LowerRank = 12,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEvaluation {
    pub slot: SlotId,
    pub generation: Generation,
    pub state: SlotState,
    pub shared_prefix_tokens: usize,
    pub namespace_compatible: bool,
    pub model_compatible: bool,
    pub session_compatible: bool,
    pub generation_compatible: bool,
    pub busy_eligible: bool,
    pub lease_eligible: bool,
    pub materialized: bool,
    pub trim_eligible: bool,
    pub copy_eligible: bool,
    pub eligible: bool,
    pub selected: bool,
    pub rejection: CandidateRejection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub id: PlanId,
    pub epoch: u64,
    pub operation: Operation,
    pub target: SlotRef,
    pub donor: Option<SlotRef>,
    pub reuse_tokens: usize,
    pub anchor_tokens: usize,
    /// Ordered exact boundaries authorized for physical snapshotting while
    /// this request prefills. Adapters must not invent additional boundaries.
    pub anchor_boundaries: Vec<usize>,
    pub evictions: Vec<SlotRef>,
    pub namespace: Namespace,
    pub requested_tokens: Vec<i32>,
    pub labels: Labels,
    pub result_state: SlotState,
    pub decision_us: u64,
    pub candidates: Vec<CandidateEvaluation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Commit {
    /// Backend-confirmed prefix of the planned request now resident in target.
    pub resident_tokens: usize,
    pub actual_state: SlotState,
    pub physical_bytes: u64,
    pub prefill_us_saved: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Decision {
    pub hit: bool,
    pub planned_reuse_tokens: usize,
    pub realized_reuse_tokens: usize,
    pub operation: Operation,
    pub scope: Scope,
    pub donor_slot: Option<SlotId>,
    pub target_slot: SlotId,
    pub evicted_slots: Vec<SlotId>,
    pub decision_us: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Telemetry {
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
    pub total_slots: u32,
    pub session_slots: u32,
    pub session_bytes: u64,
    pub anchor_bytes: u64,
    pub automatic_checkpoint_slots: u32,
    pub automatic_checkpoint_bytes: u64,
    pub automatic_checkpoint_byte_budget: u64,
    pub active_bytes: u64,
    pub physical_byte_budget: u64,
    pub high_watermark_bytes: u64,
    pub low_watermark_bytes: u64,
    pub under_pressure: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotSnapshot {
    pub id: SlotId,
    pub generation: Generation,
    pub namespace: Namespace,
    pub resident_len: usize,
    pub state: SlotState,
    pub busy: bool,
    pub read_leases: u32,
    pub write_leased: bool,
    pub labels: Labels,
    pub last_used: u64,
    pub reuse_count: u64,
    pub physical_bytes: u64,
    pub protected: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidArgument,
    NoCapacity,
    StalePlan,
    PlanConsumed,
    SlotBusy,
    Unsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug)]
struct Slot {
    id: SlotId,
    generation: Generation,
    namespace: Namespace,
    tokens: Vec<i32>,
    state: SlotState,
    busy: bool,
    read_leases: u32,
    writer: Option<PlanId>,
    labels: Labels,
    last_used: u64,
    reuse_count: u64,
    physical_bytes: u64,
    saved_us: u64,
    protected: bool,
}

impl Slot {
    fn new(id: SlotId) -> Self {
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

    fn is_empty(&self) -> bool {
        self.state == SlotState::Empty || self.tokens.is_empty()
    }
}

#[derive(Clone, Debug)]
struct Pending {
    plan: Plan,
}

pub struct CacheManager {
    config: Config,
    retention: RetentionConfig,
    epoch: u64,
    clock: u64,
    next_plan: PlanId,
    slots: Vec<Slot>,
    index: RadixIndex,
    pending: BTreeMap<PlanId, Pending>,
    telemetry: Telemetry,
    last_decision: Option<Decision>,
}

impl CacheManager {
    pub fn new(config: Config, retention: RetentionConfig) -> Result<Self, Error> {
        if config.slot_count == 0
            || retention.hard_max_retained_entries < config.slot_count
            || config.max_anchors > retention.hard_max_retained_entries
            || config.automatic_checkpoints.minimum_prompt_tokens == 0
            || config.automatic_checkpoints.target_interval_tokens == 0
            || config.automatic_checkpoints.memory_budget_basis_points > 10_000
            || retention.physical_byte_budget.is_some_and(|budget| {
                budget == 0
                    || retention.low_watermark_bytes > retention.high_watermark_bytes
                    || retention.high_watermark_bytes > budget
            })
            || (retention.physical_byte_budget.is_none()
                && (retention.high_watermark_bytes != 0 || retention.low_watermark_bytes != 0))
        {
            return Err(Error::InvalidArgument);
        }
        let mut manager = Self {
            slots: (0..config.slot_count).map(Slot::new).collect(),
            config,
            retention,
            epoch: 1,
            clock: 0,
            next_plan: 1,
            index: RadixIndex::default(),
            pending: BTreeMap::new(),
            telemetry: Telemetry::default(),
            last_decision: None,
        };
        manager.refresh_gauges();
        Ok(manager)
    }

    /// Registers one backend-owned physical slot. IDs are append-only and are
    /// therefore stable for the manager lifetime, including across reset.
    pub fn register_slot(&mut self) -> Result<SlotRef, Error> {
        if self.slots.len() >= self.retention.hard_max_retained_entries as usize {
            return Err(Error::NoCapacity);
        }
        let id = SlotId::try_from(self.slots.len()).map_err(|_| Error::NoCapacity)?;
        self.slots.push(Slot::new(id));
        self.refresh_gauges();
        Ok(self.slot_ref(id))
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn plan(&mut self, request: PlanRequest<'_>) -> Result<Plan, Error> {
        let started = Instant::now();
        if request.tokens.is_empty()
            || request.result_state == SlotState::Empty
            || request
                .slot_capabilities
                .is_some_and(|slots| slots.len() != self.slots.len())
        {
            return Err(Error::InvalidArgument);
        }
        if request.result_state == SlotState::Anchor {
            if let Some(slot) = self.slots.iter().find(|slot| {
                slot.state == SlotState::Anchor
                    && slot.namespace == request.namespace
                    && slot.tokens == request.tokens
                    && slot.writer.is_none()
            }) {
                let plan_id = self.next_plan;
                self.next_plan = self.next_plan.checked_add(1).ok_or(Error::NoCapacity)?;
                let decision_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
                let candidates = self.candidate_evaluations(&request, Some(slot.id));
                let plan = Plan {
                    id: plan_id,
                    epoch: self.epoch,
                    operation: Operation::Noop,
                    target: self.slot_ref(slot.id),
                    donor: Some(self.slot_ref(slot.id)),
                    reuse_tokens: request.tokens.len(),
                    anchor_tokens: 0,
                    anchor_boundaries: Vec::new(),
                    evictions: Vec::new(),
                    namespace: request.namespace,
                    requested_tokens: request.tokens.to_vec(),
                    labels: request.labels,
                    result_state: request.result_state,
                    decision_us,
                    candidates,
                };
                self.telemetry.plans += 1;
                self.telemetry.hits += 1;
                self.telemetry.planned_reuse_tokens += request.tokens.len() as u64;
                self.pending.insert(plan_id, Pending { plan: plan.clone() });
                self.refresh_gauges();
                return Ok(plan);
            }
        }
        let plan_id = self.next_plan;
        self.next_plan = self.next_plan.checked_add(1).ok_or(Error::NoCapacity)?;

        let donor = self.choose_donor(&request);
        let continuation = donor.filter(|&(slot, prefix)| {
            let candidate = &self.slots[slot as usize];
            request.result_state != SlotState::Anchor
                && !candidate.busy
                && candidate.state != SlotState::Anchor
                && self
                    .slot_capabilities(&request, slot)
                    .contains(SlotCapabilities::WRITABLE)
                && ((candidate.labels.session == 0 && request.labels.session == 0)
                    || (candidate.labels.session != 0
                        && candidate.labels.session == request.labels.session))
                && (prefix == candidate.tokens.len()
                    || (request
                        .capabilities
                        .contains(Capabilities::PARTIAL_SUFFIX_TRIM)
                        && self
                            .slot_capabilities(&request, slot)
                            .contains(SlotCapabilities::PARTIAL_SUFFIX_TRIM)))
        });

        let (operation, target_id, selected_donor, reuse_tokens) =
            if let Some((slot, prefix)) = continuation {
                (Operation::Continue, slot, Some(slot), prefix)
            } else if let Some((slot, prefix)) = donor {
                let candidate = &self.slots[slot as usize];
                let can_copy = if prefix == candidate.tokens.len() {
                    request
                        .capabilities
                        .contains(Capabilities::WHOLE_STATE_COPY)
                        || request
                            .capabilities
                            .contains(Capabilities::PARTIAL_PREFIX_BRANCH)
                } else {
                    request
                        .capabilities
                        .contains(Capabilities::PARTIAL_PREFIX_BRANCH)
                };
                if can_copy {
                    let target = self.choose_target(Some(slot), &request)?;
                    let kind = if candidate.state == SlotState::Anchor {
                        Operation::Restore
                    } else {
                        Operation::Branch
                    };
                    (kind, target, Some(slot), prefix)
                } else {
                    let target = self.choose_target(Some(slot), &request)?;
                    (Operation::Fresh, target, None, 0)
                }
            } else {
                let target = self.choose_target(None, &request)?;
                (Operation::Fresh, target, None, 0)
            };
        let anchor_boundaries = if request.result_state != SlotState::Anchor
            && request
                .capabilities
                .contains(Capabilities::PROMPT_BOUNDARY_SNAPSHOT)
        {
            self.choose_anchor_boundaries(&request, reuse_tokens)
        } else {
            Vec::new()
        };
        let anchor_tokens = anchor_boundaries.last().copied().unwrap_or(0);

        let mut eviction_ids = Vec::new();
        if !self.slots[target_id as usize].is_empty() && operation != Operation::Continue {
            eviction_ids.push(target_id);
        }
        eviction_ids.extend(self.pressure_victims(
            target_id,
            selected_donor,
            request.namespace,
            if request.result_state == SlotState::Anchor {
                request
                    .estimated_bytes_per_token
                    .saturating_mul(request.tokens.len() as u64)
            } else {
                // Active execution storage is not retained checkpoint
                // storage. Charging a full sibling prompt here can reject
                // generation even though the target becomes the active
                // session and optional anchors are independently bounded.
                0
            },
            request.result_state == SlotState::Anchor,
        )?);
        if request.result_state == SlotState::Anchor
            && self.is_automatic_checkpoint_len(request.tokens.len())
        {
            eviction_ids.extend(
                self.automatic_checkpoint_budget_victims(
                    target_id,
                    selected_donor,
                    request.namespace,
                    request
                        .estimated_bytes_per_token
                        .saturating_mul(request.tokens.len() as u64),
                )?,
            );
        }
        eviction_ids.sort_unstable();
        eviction_ids.dedup();

        let mut write_ids = eviction_ids.clone();
        write_ids.push(target_id);
        write_ids.sort_unstable();
        write_ids.dedup();
        for &slot in &write_ids {
            let candidate = &self.slots[slot as usize];
            if candidate.busy || candidate.writer.is_some() || candidate.read_leases > 0 {
                return Err(Error::SlotBusy);
            }
        }
        if let Some(slot) = selected_donor {
            let candidate = &self.slots[slot as usize];
            if candidate.writer.is_some()
                || (candidate.busy && !request.capabilities.contains(Capabilities::SAFE_BUSY_DONOR))
            {
                return Err(Error::SlotBusy);
            }
        }

        for &slot in &write_ids {
            self.slots[slot as usize].writer = Some(plan_id);
        }
        if let Some(slot) = selected_donor {
            if slot != target_id {
                self.slots[slot as usize].read_leases += 1;
            }
        }

        let decision_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        let candidates = self.candidate_evaluations(&request, selected_donor);
        let plan = Plan {
            id: plan_id,
            epoch: self.epoch,
            operation,
            target: self.slot_ref(target_id),
            donor: selected_donor.map(|slot| self.slot_ref(slot)),
            reuse_tokens,
            anchor_tokens,
            anchor_boundaries,
            evictions: eviction_ids
                .iter()
                .map(|&slot| self.slot_ref(slot))
                .collect(),
            namespace: request.namespace,
            requested_tokens: request.tokens.to_vec(),
            labels: request.labels,
            result_state: request.result_state,
            decision_us,
            candidates,
        };
        self.telemetry.plans += 1;
        self.telemetry.planned_reuse_tokens += reuse_tokens as u64;
        if reuse_tokens > 0 {
            self.telemetry.hits += 1;
        } else {
            self.telemetry.misses += 1;
        }
        self.pending.insert(plan_id, Pending { plan: plan.clone() });
        self.refresh_gauges();
        Ok(plan)
    }

    fn candidate_evaluations(
        &self,
        request: &PlanRequest<'_>,
        selected: Option<SlotId>,
    ) -> Vec<CandidateEvaluation> {
        let reusable_prefix_len = if request
            .capabilities
            .contains(Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS)
        {
            request.tokens.len().saturating_sub(1)
        } else {
            request.tokens.len()
        };
        let mut candidates = self
            .slots
            .iter()
            .filter(|slot| !slot.is_empty())
            .map(|slot| {
                let capabilities = self.slot_capabilities(request, slot.id);
                let shared_prefix_tokens =
                    common_prefix(&slot.tokens, request.tokens).min(reusable_prefix_len);
                let namespace_compatible = slot.namespace == request.namespace;
                // The namespace is the adapter's installation-keyed model domain.
                let model_compatible = namespace_compatible;
                let session_compatible = (slot.labels.session == 0 && request.labels.session == 0)
                    || (slot.labels.session != 0 && slot.labels.session == request.labels.session);
                let generation_compatible = true;
                let busy_eligible =
                    !slot.busy || request.capabilities.contains(Capabilities::SAFE_BUSY_DONOR);
                let lease_eligible = slot.writer.is_none();
                let materialized = capabilities.contains(SlotCapabilities::MATERIALIZED);
                let whole_state = shared_prefix_tokens == slot.tokens.len();
                let trim_eligible = whole_state
                    || (request
                        .capabilities
                        .contains(Capabilities::PARTIAL_SUFFIX_TRIM)
                        && capabilities.contains(SlotCapabilities::PARTIAL_SUFFIX_TRIM));
                let copy_eligible = capabilities.contains(SlotCapabilities::COPY)
                    && ((whole_state
                        && request
                            .capabilities
                            .contains(Capabilities::WHOLE_STATE_COPY))
                        || (request
                            .capabilities
                            .contains(Capabilities::PARTIAL_PREFIX_BRANCH)
                            && (whole_state
                                || capabilities.contains(SlotCapabilities::PARTIAL_SUFFIX_TRIM))));
                let continuation_eligible = request.result_state != SlotState::Anchor
                    && !slot.busy
                    && slot.state != SlotState::Anchor
                    && session_compatible
                    && capabilities.contains(SlotCapabilities::WRITABLE)
                    && trim_eligible;
                let eligible = namespace_compatible
                    && generation_compatible
                    && busy_eligible
                    && lease_eligible
                    && materialized
                    && shared_prefix_tokens >= self.config.min_reuse_tokens
                    && (continuation_eligible || copy_eligible);
                let is_selected = selected == Some(slot.id);
                let rejection = if is_selected {
                    CandidateRejection::None
                } else if !namespace_compatible {
                    CandidateRejection::Namespace
                } else if !generation_compatible {
                    CandidateRejection::Generation
                } else if !busy_eligible || !lease_eligible {
                    CandidateRejection::BusyLease
                } else if !materialized {
                    CandidateRejection::Materialization
                } else if shared_prefix_tokens < self.config.min_reuse_tokens {
                    CandidateRejection::MinPrefix
                } else if !session_compatible && !copy_eligible {
                    CandidateRejection::Session
                } else if !trim_eligible && !copy_eligible {
                    CandidateRejection::Nontrim
                } else if !eligible {
                    CandidateRejection::Capability
                } else {
                    CandidateRejection::LowerRank
                };
                CandidateEvaluation {
                    slot: slot.id,
                    generation: slot.generation,
                    state: slot.state,
                    shared_prefix_tokens,
                    namespace_compatible,
                    model_compatible,
                    session_compatible,
                    generation_compatible,
                    busy_eligible,
                    lease_eligible,
                    materialized,
                    trim_eligible,
                    copy_eligible,
                    eligible,
                    selected: is_selected,
                    rejection,
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .shared_prefix_tokens
                .cmp(&left.shared_prefix_tokens)
                .then_with(|| right.selected.cmp(&left.selected))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        candidates.truncate(MAX_PLAN_CANDIDATES);
        candidates
    }

    fn choose_donor(&self, request: &PlanRequest<'_>) -> Option<(SlotId, usize)> {
        let reusable_prefix_len = if request
            .capabilities
            .contains(Capabilities::RETAIN_LAST_TOKEN_FOR_LOGITS)
        {
            request.tokens.len().saturating_sub(1)
        } else {
            request.tokens.len()
        };
        self.slots
            .iter()
            .filter_map(|slot| {
                let id = slot.id;
                if slot.namespace != request.namespace {
                    return None;
                }
                if slot.is_empty() || slot.writer.is_some() {
                    return None;
                }
                let slot_capabilities = self.slot_capabilities(request, id);
                if !slot_capabilities.contains(SlotCapabilities::MATERIALIZED) {
                    return None;
                }
                if slot.busy && !request.capabilities.contains(Capabilities::SAFE_BUSY_DONOR) {
                    return None;
                }
                let prefix = common_prefix(&slot.tokens, request.tokens).min(reusable_prefix_len);
                let whole_state = prefix == slot.tokens.len();
                let session_affine = (slot.labels.session == 0 && request.labels.session == 0)
                    || (slot.labels.session != 0 && slot.labels.session == request.labels.session);
                let same_session = request.result_state != SlotState::Anchor
                    && !slot.busy
                    && slot.state != SlotState::Anchor
                    && session_affine;
                let can_continue = same_session
                    && slot_capabilities.contains(SlotCapabilities::WRITABLE)
                    && (whole_state
                        || (request
                            .capabilities
                            .contains(Capabilities::PARTIAL_SUFFIX_TRIM)
                            && slot_capabilities.contains(SlotCapabilities::PARTIAL_SUFFIX_TRIM)));
                let can_copy = (whole_state
                    && request
                        .capabilities
                        .contains(Capabilities::WHOLE_STATE_COPY)
                    && slot_capabilities.contains(SlotCapabilities::COPY))
                    || (request
                        .capabilities
                        .contains(Capabilities::PARTIAL_PREFIX_BRANCH)
                        && slot_capabilities.contains(SlotCapabilities::COPY)
                        && slot_capabilities.contains(SlotCapabilities::PARTIAL_SUFFIX_TRIM));
                (prefix >= self.config.min_reuse_tokens && (can_continue || can_copy))
                    .then_some((id, prefix))
            })
            .max_by(|(left_id, left_prefix), (right_id, right_prefix)| {
                let left = &self.slots[*left_id as usize];
                let right = &self.slots[*right_id as usize];
                let left_anchor = (request.labels.side_request && left.state == SlotState::Anchor) as u8;
                let right_anchor = (request.labels.side_request && right.state == SlotState::Anchor) as u8;
                left_anchor
                    .cmp(&right_anchor)
                    .then_with(|| left_prefix
                    .cmp(right_prefix)
                    .then_with(|| donor_rank(left, request).cmp(&donor_rank(right, request))))
                    .then_with(|| right_id.cmp(left_id))
            })
    }

    fn choose_anchor_boundaries(
        &self,
        request: &PlanRequest<'_>,
        reuse_tokens: usize,
    ) -> Vec<usize> {
        let automatic = self.propose_automatic_checkpoints(request.tokens.len());
        let eligible = |boundary: usize| {
            boundary >= self.config.min_reuse_tokens
                && boundary > reuse_tokens
                && boundary < request.tokens.len()
                && (self.retention.physical_byte_budget.is_some()
                    || boundary <= self.config.logical_token_capacity)
                && !self.slots.iter().any(|slot| {
                    slot.state == SlotState::Anchor
                        && slot.namespace == request.namespace
                        && slot.tokens == request.tokens[..boundary]
                })
        };
        let mut semantic = request
            .stable_boundaries
            .iter()
            .copied()
            .filter(|&boundary| eligible(boundary))
            .collect::<Vec<_>>();
        semantic.sort_unstable();
        semantic.dedup();
        let mut automatic = automatic
            .into_iter()
            .filter(|&boundary| eligible(boundary) && !semantic.contains(&boundary))
            .collect::<Vec<_>>();
        let existing_anchors = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Anchor)
            .count();
        let available = (self.config.max_anchors as usize).saturating_sub(existing_anchors);
        let replaceable_automatic = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && slot.namespace == request.namespace
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
            })
            .count();
        semantic.truncate(available);
        let automatic_available = available
            .saturating_sub(semantic.len())
            .saturating_add(replaceable_automatic);
        automatic.truncate(self.config.automatic_checkpoints.max_checkpoints as usize);
        automatic = self.checkpoint_coverage_order(automatic);
        let mut authorized = semantic;
        let mut physical_cost = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && slot.namespace == request.namespace
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
            })
            .map(|slot| slot.physical_bytes)
            .sum::<u64>();
        let byte_limit = self.automatic_checkpoint_byte_limit();
        let baseline_cost = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && slot.namespace == request.namespace
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
            })
            .min_by_key(|slot| slot.tokens.len())
            .map_or(0, |slot| slot.physical_bytes);
        let mut automatic_added = 0;
        for boundary in automatic {
            if automatic_added >= automatic_available {
                break;
            }
            let incremental = request
                .estimated_bytes_per_token
                .saturating_mul(boundary as u64);
            if byte_limit != u64::MAX
                && incremental > 0
                && physical_cost.saturating_add(incremental) > byte_limit
            {
                if baseline_cost.saturating_add(incremental) > byte_limit {
                    continue;
                }
                physical_cost = baseline_cost;
            }
            physical_cost = physical_cost.saturating_add(incremental);
            authorized.push(boundary);
            automatic_added += 1;
        }
        authorized.sort_unstable();
        authorized
    }

    fn checkpoint_coverage_order(&self, mut boundaries: Vec<usize>) -> Vec<usize> {
        boundaries.sort_unstable();
        let shallowest = boundaries.first().copied();
        let deepest = boundaries.last().copied();
        let mut baseline = Vec::new();
        let mut remainder = Vec::new();
        let mut seen_bands = BTreeMap::<u32, ()>::new();
        for boundary in boundaries {
            if Some(boundary) == shallowest || Some(boundary) == deepest {
                continue;
            }
            let band = self.checkpoint_depth_band(boundary);
            if seen_bands.insert(band, ()).is_none() {
                baseline.push(boundary);
            } else {
                remainder.push(boundary);
            }
        }
        remainder.reverse();
        let mut ordered = Vec::new();
        if let Some(boundary) = shallowest {
            ordered.push(boundary);
        }
        if let Some(boundary) = deepest.filter(|deep| Some(*deep) != shallowest) {
            ordered.push(boundary);
        }
        ordered.extend(baseline);
        ordered.extend(remainder);
        ordered
    }

    fn checkpoint_depth_band(&self, tokens: usize) -> u32 {
        let interval = self
            .config
            .automatic_checkpoints
            .target_interval_tokens
            .max(1);
        (tokens / interval).max(1).ilog2()
    }

    fn is_automatic_checkpoint_len(&self, tokens: usize) -> bool {
        let policy = &self.config.automatic_checkpoints;
        policy.enabled
            && tokens >= policy.minimum_prompt_tokens
            && tokens % policy.target_interval_tokens == 0
    }

    fn propose_automatic_checkpoints(&self, prompt_tokens: usize) -> Vec<usize> {
        let policy = &self.config.automatic_checkpoints;
        if !policy.enabled || prompt_tokens < policy.minimum_prompt_tokens || prompt_tokens <= 1 {
            return Vec::new();
        }
        let reusable = prompt_tokens - 1;
        let base = policy.target_interval_tokens;
        let proposed_at_base = reusable / base;
        let multiplier = proposed_at_base
            .div_ceil(policy.max_checkpoints.max(1) as usize)
            .max(1);
        let interval = base.saturating_mul(multiplier);
        (interval..=reusable)
            .step_by(interval)
            .take(policy.max_checkpoints as usize)
            .collect()
    }

    fn automatic_checkpoint_byte_limit(&self) -> u64 {
        let policy = &self.config.automatic_checkpoints;
        let fraction = self
            .retention
            .physical_byte_budget
            .map_or(u64::MAX, |budget| {
                budget.saturating_mul(policy.memory_budget_basis_points as u64) / 10_000
            });
        if policy.memory_budget_cap_bytes == 0 {
            fraction
        } else {
            fraction.min(policy.memory_budget_cap_bytes)
        }
    }

    fn choose_target(
        &self,
        donor: Option<SlotId>,
        request: &PlanRequest<'_>,
    ) -> Result<SlotId, Error> {
        let anchor_count = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Anchor)
            .count();
        if request.result_state == SlotState::Anchor
            && anchor_count >= self.config.max_anchors as usize
        {
            return self
                .slots
                .iter()
                .filter(|slot| {
                    slot.state == SlotState::Anchor
                        && Some(slot.id) != donor
                        && !slot.busy
                        && slot.writer.is_none()
                        && slot.read_leases == 0
                        && !slot.protected
                        && self
                            .slot_capabilities(request, slot.id)
                            .contains(SlotCapabilities::WRITABLE)
                })
                .min_by_key(|slot| (anchor_eviction_value(slot), slot.id))
                .map(|slot| slot.id)
                .ok_or(Error::NoCapacity);
        }
        if let Some(slot) = self.slots.iter().find(|slot| {
            slot.is_empty()
                && slot.writer.is_none()
                && !slot.busy
                && self
                    .slot_capabilities(request, slot.id)
                    .contains(SlotCapabilities::WRITABLE)
        }) {
            return Ok(slot.id);
        }
        let preferred = self
            .slots
            .iter()
            .filter(|slot| {
                Some(slot.id) != donor
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
                    && self
                        .slot_capabilities(request, slot.id)
                        .contains(SlotCapabilities::WRITABLE)
                    && !(slot.state == SlotState::Anchor
                        && request.result_state != SlotState::Anchor)
            })
            .min_by_key(|slot| (eviction_value(slot), slot.id))
            .map(|slot| slot.id);
        if let Some(slot_id) = preferred {
            return Ok(slot_id);
        }
        // Anchors are valuable but not allowed to deadlock the pool. If all
        // otherwise writable slots are anchors, evict the lowest-value one
        // (never the selected donor) so a session can still start fresh or
        // restore another exact anchor.
        let fallback = self
            .slots
            .iter()
            .filter(|slot| {
                Some(slot.id) != donor
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
                    && self
                        .slot_capabilities(request, slot.id)
                        .contains(SlotCapabilities::WRITABLE)
            })
            .min_by_key(|slot| (eviction_value(slot), slot.id))
            .map(|slot| slot.id);
        if let Some(slot) = fallback {
            return Ok(slot);
        }
        let temporarily_blocked = self.slots.iter().any(|slot| {
            Some(slot.id) != donor
                && self
                    .slot_capabilities(request, slot.id)
                    .contains(SlotCapabilities::WRITABLE)
                && (slot.busy || slot.writer.is_some() || slot.read_leases > 0)
        });
        Err(if temporarily_blocked {
            Error::SlotBusy
        } else {
            Error::NoCapacity
        })
    }

    fn pressure_victims(
        &self,
        target: SlotId,
        donor: Option<SlotId>,
        request_namespace: Namespace,
        projected_target_bytes: u64,
        enforce_low_watermark: bool,
    ) -> Result<Vec<SlotId>, Error> {
        let Some(_) = self.retention.physical_byte_budget else {
            return Ok(Vec::new());
        };
        let current: u64 = self.slots.iter().map(|slot| slot.physical_bytes).sum();
        let target_current = self.slots[target as usize].physical_bytes;
        let mut projected = current
            .saturating_sub(target_current)
            .saturating_add(projected_target_bytes);
        if current <= self.retention.high_watermark_bytes
            && projected <= self.retention.high_watermark_bytes
        {
            return Ok(Vec::new());
        }

        let mut namespace_bytes = BTreeMap::<Namespace, u64>::new();
        for slot in self.slots.iter().filter(|slot| !slot.is_empty()) {
            *namespace_bytes.entry(slot.namespace).or_default() = namespace_bytes
                .get(&slot.namespace)
                .copied()
                .unwrap_or_default()
                .saturating_add(slot.physical_bytes);
        }
        namespace_bytes.entry(request_namespace).or_default();
        let fair_share = self.retention.physical_byte_budget.unwrap_or_default()
            / namespace_bytes.len().max(1) as u64;
        let mut candidates: Vec<&Slot> = self
            .slots
            .iter()
            .filter(|slot| {
                slot.id != target
                    && Some(slot.id) != donor
                    && !slot.is_empty()
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
            })
            .collect();
        candidates.sort_by_key(|slot| {
            let coverage_representative = self.is_checkpoint_coverage_representative(slot);
            (
                slot.labels.priority as u32,
                (!slot.labels.side_request) as u8,
                (slot.state == SlotState::Anchor) as u8,
                coverage_representative as u8,
                slot.reuse_count,
                slot.saved_us,
                namespace_bytes
                    .get(&slot.namespace)
                    .copied()
                    .unwrap_or_default()
                    <= fair_share,
                slot.last_used,
                Reverse(slot.physical_bytes),
                slot.tokens.len(),
                slot.id,
            )
        });
        let mut victims = Vec::new();
        for slot in candidates {
            if projected <= self.retention.low_watermark_bytes {
                break;
            }
            projected = projected.saturating_sub(slot.physical_bytes);
            victims.push(slot.id);
        }
        if enforce_low_watermark && projected > self.retention.low_watermark_bytes {
            return Err(Error::NoCapacity);
        }
        Ok(victims)
    }

    fn automatic_checkpoint_budget_victims(
        &self,
        target: SlotId,
        donor: Option<SlotId>,
        request_namespace: Namespace,
        projected_target_bytes: u64,
    ) -> Result<Vec<SlotId>, Error> {
        let limit = self.automatic_checkpoint_byte_limit();
        if limit == u64::MAX || projected_target_bytes == 0 {
            return Ok(Vec::new());
        }
        let target_current = self.slots[target as usize].physical_bytes;
        let current = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
            })
            .map(|slot| slot.physical_bytes)
            .sum::<u64>();
        let mut projected = current
            .saturating_sub(target_current)
            .saturating_add(projected_target_bytes);
        if projected <= limit {
            return Ok(Vec::new());
        }

        let mut namespace_bytes = BTreeMap::<Namespace, u64>::new();
        for slot in self.slots.iter().filter(|slot| {
            slot.state == SlotState::Anchor && self.is_automatic_checkpoint_len(slot.tokens.len())
        }) {
            *namespace_bytes.entry(slot.namespace).or_default() = namespace_bytes
                .get(&slot.namespace)
                .copied()
                .unwrap_or_default()
                .saturating_add(slot.physical_bytes);
        }
        namespace_bytes.entry(request_namespace).or_default();
        let fair_share = limit / namespace_bytes.len().max(1) as u64;
        let mut candidates = self
            .slots
            .iter()
            .filter(|slot| {
                slot.id != target
                    && Some(slot.id) != donor
                    && slot.state == SlotState::Anchor
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|slot| {
            let shallowest = !self.slots.iter().any(|other| {
                other.id != slot.id
                    && other.state == SlotState::Anchor
                    && other.namespace == slot.namespace
                    && self.is_automatic_checkpoint_len(other.tokens.len())
                    && other.tokens.len() < slot.tokens.len()
            });
            (
                slot.namespace != request_namespace,
                namespace_bytes
                    .get(&slot.namespace)
                    .copied()
                    .unwrap_or_default()
                    <= fair_share,
                shallowest,
                slot.reuse_count,
                slot.saved_us,
                slot.last_used,
                Reverse(slot.physical_bytes),
                slot.id,
            )
        });
        let mut victims = Vec::new();
        for slot in candidates {
            if projected <= limit {
                break;
            }
            projected = projected.saturating_sub(slot.physical_bytes);
            victims.push(slot.id);
        }
        if projected > limit {
            return Err(Error::NoCapacity);
        }
        Ok(victims)
    }

    fn is_checkpoint_coverage_representative(&self, candidate: &Slot) -> bool {
        if candidate.state != SlotState::Anchor
            || !self.is_automatic_checkpoint_len(candidate.tokens.len())
        {
            return false;
        }
        let band = self.checkpoint_depth_band(candidate.tokens.len());
        !self.slots.iter().any(|slot| {
            slot.id != candidate.id
                && slot.state == SlotState::Anchor
                && slot.namespace == candidate.namespace
                && self.is_automatic_checkpoint_len(slot.tokens.len())
                && self.checkpoint_depth_band(slot.tokens.len()) == band
                && slot.tokens.len() < candidate.tokens.len()
                && candidate.tokens[..slot.tokens.len()] == slot.tokens
        })
    }

    fn slot_capabilities(&self, request: &PlanRequest<'_>, slot: SlotId) -> SlotCapabilities {
        request
            .slot_capabilities
            .map_or(SlotCapabilities::ALL, |slots| slots[slot as usize])
    }

    pub fn commit(&mut self, plan_id: PlanId, commit: Commit) -> Result<Decision, Error> {
        let Some(pending) = self.pending.remove(&plan_id) else {
            return Err(Error::PlanConsumed);
        };
        let plan = pending.plan;
        if plan.epoch != self.epoch || !self.generations_match(&plan) {
            self.release(&plan);
            self.telemetry.stale_commits += 1;
            self.refresh_gauges();
            return Err(Error::StalePlan);
        }
        if commit.resident_tokens > plan.requested_tokens.len()
            || commit.actual_state == SlotState::Empty
        {
            self.release(&plan);
            if plan.operation != Operation::Noop {
                for victim in &plan.evictions {
                    if victim.slot != plan.target.slot {
                        self.invalidate_slot(victim.slot);
                    }
                }
                self.invalidate_slot(plan.target.slot);
            }
            self.refresh_gauges();
            return Err(Error::InvalidArgument);
        }

        if plan.operation == Operation::Noop {
            if commit.resident_tokens != plan.requested_tokens.len()
                || commit.actual_state != SlotState::Anchor
            {
                self.refresh_gauges();
                return Err(Error::InvalidArgument);
            }
            let decision = Decision {
                hit: true,
                planned_reuse_tokens: plan.reuse_tokens,
                realized_reuse_tokens: plan.reuse_tokens,
                operation: plan.operation,
                scope: plan.labels.scope,
                donor_slot: plan.donor.map(|slot| slot.slot),
                target_slot: plan.target.slot,
                evicted_slots: Vec::new(),
                decision_us: plan.decision_us,
            };
            self.telemetry.commits += 1;
            self.telemetry.realized_reuse_tokens += plan.reuse_tokens as u64;
            self.last_decision = Some(decision.clone());
            self.refresh_gauges();
            return Ok(decision);
        }

        for victim in &plan.evictions {
            if victim.slot != plan.target.slot {
                self.invalidate_slot(victim.slot);
                self.telemetry.evictions += 1;
            }
        }
        if plan
            .evictions
            .iter()
            .any(|slot| slot.slot == plan.target.slot)
        {
            self.telemetry.evictions += 1;
        }
        self.remove_index(plan.target.slot);
        self.clock += 1;
        {
            let target = &mut self.slots[plan.target.slot as usize];
            target.generation += 1;
            target.namespace = plan.namespace;
            target.tokens = plan.requested_tokens[..commit.resident_tokens].to_vec();
            target.state = commit.actual_state;
            target.labels = plan.labels.clone();
            target.last_used = self.clock;
            target.physical_bytes = commit.physical_bytes;
            target.saved_us = target.saved_us.saturating_add(commit.prefill_us_saved);
            target.busy = commit.actual_state == SlotState::Session;
            target.protected = commit.actual_state == SlotState::Anchor
                && matches!(target.labels.scope, Scope::Harness | Scope::Tenant);
            if plan.reuse_tokens > 0 {
                target.reuse_count += 1;
            }
            target.writer = None;
        }
        if let Some(donor) = &plan.donor {
            if donor.slot != plan.target.slot {
                let slot = &mut self.slots[donor.slot as usize];
                slot.read_leases = slot.read_leases.saturating_sub(1);
                slot.reuse_count += 1;
                slot.last_used = self.clock;
            }
        }
        let target = &self.slots[plan.target.slot as usize];
        self.index
            .insert(target.namespace, &target.tokens, target.id);

        let realized = plan.reuse_tokens.min(commit.resident_tokens);
        let decision = Decision {
            hit: realized > 0,
            planned_reuse_tokens: plan.reuse_tokens,
            realized_reuse_tokens: realized,
            operation: plan.operation,
            scope: plan.labels.scope,
            donor_slot: plan.donor.map(|slot| slot.slot),
            target_slot: plan.target.slot,
            evicted_slots: plan.evictions.iter().map(|slot| slot.slot).collect(),
            decision_us: plan.decision_us,
        };
        self.telemetry.commits += 1;
        self.telemetry.realized_reuse_tokens += realized as u64;
        self.telemetry.prefill_us_saved += commit.prefill_us_saved;
        self.last_decision = Some(decision.clone());
        self.refresh_gauges();
        Ok(decision)
    }

    /// Aborting fails closed: every slot authorized for physical mutation is
    /// invalidated because a backend operation may have partially succeeded.
    /// Read-only donors remain intact.
    pub fn abort(&mut self, plan_id: PlanId) -> Result<(), Error> {
        let Some(pending) = self.pending.remove(&plan_id) else {
            return Err(Error::PlanConsumed);
        };
        let plan = pending.plan;
        self.release(&plan);
        if plan.epoch == self.epoch && plan.operation != Operation::Noop {
            for victim in &plan.evictions {
                if victim.slot != plan.target.slot {
                    self.invalidate_slot(victim.slot);
                }
            }
            self.invalidate_slot(plan.target.slot);
        }
        self.telemetry.aborts += 1;
        self.refresh_gauges();
        Ok(())
    }

    /// Publishes backend-confirmed decode/prefill progress after a committed
    /// plan. Generation guards prevent late request completion from corrupting a
    /// recycled slot.
    pub fn advance(
        &mut self,
        slot_id: SlotId,
        generation: Generation,
        appended_tokens: &[i32],
        state: SlotState,
        busy: bool,
        physical_bytes: u64,
    ) -> Result<Generation, Error> {
        let Some(slot) = self.slots.get(slot_id as usize) else {
            return Err(Error::InvalidArgument);
        };
        if slot.generation != generation {
            return Err(Error::StalePlan);
        }
        if slot.writer.is_some() || slot.state == SlotState::Anchor {
            return Err(Error::SlotBusy);
        }
        self.remove_index(slot_id);
        self.clock += 1;
        let slot = &mut self.slots[slot_id as usize];
        slot.tokens.extend_from_slice(appended_tokens);
        slot.generation += 1;
        slot.state = state;
        slot.busy = busy;
        slot.last_used = self.clock;
        slot.physical_bytes = physical_bytes;
        self.index.insert(slot.namespace, &slot.tokens, slot.id);
        let generation = slot.generation;
        self.refresh_gauges();
        Ok(generation)
    }

    /// Replaces a slot with an exact backend-confirmed boundary. This is used
    /// after decode when a backend restores an earlier prompt snapshot.
    pub fn confirm(
        &mut self,
        slot_id: SlotId,
        generation: Generation,
        tokens: &[i32],
        state: SlotState,
        busy: bool,
        physical_bytes: u64,
    ) -> Result<Generation, Error> {
        let slot = self
            .slots
            .get(slot_id as usize)
            .ok_or(Error::InvalidArgument)?;
        if slot.generation != generation {
            return Err(Error::StalePlan);
        }
        if slot.writer.is_some() || slot.read_leases > 0 || state == SlotState::Empty {
            return Err(Error::SlotBusy);
        }
        self.remove_index(slot_id);
        self.clock += 1;
        let slot = &mut self.slots[slot_id as usize];
        slot.tokens = tokens.to_vec();
        slot.generation += 1;
        slot.state = state;
        slot.busy = busy;
        slot.last_used = self.clock;
        slot.physical_bytes = physical_bytes;
        self.index.insert(slot.namespace, &slot.tokens, slot.id);
        let generation = slot.generation;
        self.refresh_gauges();
        Ok(generation)
    }

    pub fn set_busy(
        &mut self,
        slot_id: SlotId,
        generation: Generation,
        busy: bool,
    ) -> Result<(), Error> {
        let slot = self
            .slots
            .get_mut(slot_id as usize)
            .ok_or(Error::InvalidArgument)?;
        if slot.generation != generation {
            return Err(Error::StalePlan);
        }
        if slot.writer.is_some() {
            return Err(Error::SlotBusy);
        }
        slot.busy = busy;
        self.refresh_gauges();
        Ok(())
    }

    /// Marks a structural anchor as ineligible for target selection and
    /// pressure eviction. Protection is generation-guarded and cleared by
    /// invalidation/reset.
    pub fn set_anchor_protected(
        &mut self,
        slot_id: SlotId,
        generation: Generation,
        protected: bool,
    ) -> Result<(), Error> {
        let slot = self
            .slots
            .get_mut(slot_id as usize)
            .ok_or(Error::InvalidArgument)?;
        if slot.generation != generation {
            return Err(Error::StalePlan);
        }
        if slot.state != SlotState::Anchor {
            return Err(Error::InvalidArgument);
        }
        if slot.writer.is_some() || slot.read_leases > 0 || slot.busy {
            return Err(Error::SlotBusy);
        }
        slot.protected = protected;
        Ok(())
    }

    /// Invalidates backend state that can no longer be proven exact. The
    /// generation guard prevents a late failure from clearing a recycled slot.
    pub fn invalidate(
        &mut self,
        slot_id: SlotId,
        generation: Generation,
    ) -> Result<Generation, Error> {
        let slot = self
            .slots
            .get(slot_id as usize)
            .ok_or(Error::InvalidArgument)?;
        if slot.generation != generation {
            return Err(Error::StalePlan);
        }
        if slot.writer.is_some() || slot.read_leases > 0 {
            return Err(Error::SlotBusy);
        }
        self.invalidate_slot(slot_id);
        let generation = self.slots[slot_id as usize].generation;
        self.refresh_gauges();
        Ok(generation)
    }

    pub fn reset(&mut self) -> u64 {
        self.epoch = self.epoch.saturating_add(1);
        self.clock = 0;
        self.pending.clear();
        self.index.clear();
        for slot in &mut self.slots {
            let id = slot.id;
            let generation = slot.generation.saturating_add(1);
            *slot = Slot::new(id);
            slot.generation = generation;
        }
        self.telemetry.resets += 1;
        self.last_decision = None;
        self.refresh_gauges();
        self.epoch
    }

    pub fn slot(&self, slot_id: SlotId) -> Option<SlotSnapshot> {
        self.slots.get(slot_id as usize).map(|slot| SlotSnapshot {
            id: slot.id,
            generation: slot.generation,
            namespace: slot.namespace,
            resident_len: slot.tokens.len(),
            state: slot.state,
            busy: slot.busy,
            read_leases: slot.read_leases,
            write_leased: slot.writer.is_some(),
            labels: slot.labels.clone(),
            last_used: slot.last_used,
            reuse_count: slot.reuse_count,
            physical_bytes: slot.physical_bytes,
            protected: slot.protected,
        })
    }

    pub fn telemetry(&self) -> Telemetry {
        self.telemetry.clone()
    }

    pub fn last_decision(&self) -> Option<&Decision> {
        self.last_decision.as_ref()
    }

    fn slot_ref(&self, slot: SlotId) -> SlotRef {
        SlotRef {
            slot,
            generation: self.slots[slot as usize].generation,
        }
    }

    fn generations_match(&self, plan: &Plan) -> bool {
        self.slots[plan.target.slot as usize].generation == plan.target.generation
            && plan.donor.as_ref().map_or(true, |donor| {
                self.slots[donor.slot as usize].generation == donor.generation
            })
            && plan
                .evictions
                .iter()
                .all(|victim| self.slots[victim.slot as usize].generation == victim.generation)
    }

    fn release(&mut self, plan: &Plan) {
        let target = &mut self.slots[plan.target.slot as usize];
        if target.writer == Some(plan.id) {
            target.writer = None;
        }
        for victim in &plan.evictions {
            let slot = &mut self.slots[victim.slot as usize];
            if slot.writer == Some(plan.id) {
                slot.writer = None;
            }
        }
        if let Some(donor) = &plan.donor {
            if donor.slot != plan.target.slot {
                let slot = &mut self.slots[donor.slot as usize];
                slot.read_leases = slot.read_leases.saturating_sub(1);
            }
        }
    }

    fn remove_index(&mut self, slot_id: SlotId) {
        let slot = &self.slots[slot_id as usize];
        if !slot.is_empty() {
            self.index.remove(slot.namespace, &slot.tokens, slot.id);
        }
    }

    fn invalidate_slot(&mut self, slot_id: SlotId) {
        self.remove_index(slot_id);
        let slot = &mut self.slots[slot_id as usize];
        let id = slot.id;
        let generation = slot.generation.saturating_add(1);
        *slot = Slot::new(id);
        slot.generation = generation;
    }

    fn refresh_gauges(&mut self) {
        self.telemetry.total_slots = self.slots.len() as u32;
        self.telemetry.active_slots =
            self.slots.iter().filter(|slot| !slot.is_empty()).count() as u32;
        self.telemetry.session_slots = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Session)
            .count() as u32;
        self.telemetry.anchors = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Anchor)
            .count() as u32;
        self.telemetry.read_leases = self.slots.iter().map(|slot| slot.read_leases).sum();
        self.telemetry.write_leases = self
            .slots
            .iter()
            .filter(|slot| slot.writer.is_some())
            .count() as u32;
        self.telemetry.prefix_nodes = self.index.node_count() as u64;
        self.telemetry.physical_bytes = self.slots.iter().map(|slot| slot.physical_bytes).sum();
        self.telemetry.active_bytes = self
            .slots
            .iter()
            .filter(|slot| !slot.is_empty())
            .map(|slot| slot.physical_bytes)
            .sum();
        self.telemetry.session_bytes = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Session)
            .map(|slot| slot.physical_bytes)
            .sum();
        self.telemetry.automatic_checkpoint_slots = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
            })
            .count() as u32;
        self.telemetry.automatic_checkpoint_bytes = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
            })
            .map(|slot| slot.physical_bytes)
            .sum();
        self.telemetry.automatic_checkpoint_byte_budget = self.automatic_checkpoint_byte_limit();
        self.telemetry.anchor_bytes = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Anchor)
            .map(|slot| slot.physical_bytes)
            .sum();
        self.telemetry.physical_byte_budget =
            self.retention.physical_byte_budget.unwrap_or_default();
        self.telemetry.high_watermark_bytes = self.retention.high_watermark_bytes;
        self.telemetry.low_watermark_bytes = self.retention.low_watermark_bytes;
        self.telemetry.under_pressure = self.retention.physical_byte_budget.is_some()
            && self.telemetry.physical_bytes > self.retention.high_watermark_bytes;
    }
}

fn common_prefix(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn donor_rank(slot: &Slot, request: &PlanRequest<'_>) -> (u8, u8, u64) {
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
fn eviction_value(slot: &Slot) -> (u32, u8, u8, u64, u64, u64, usize) {
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

fn anchor_eviction_value(slot: &Slot) -> (u32, u8, u64, u64, u64, usize) {
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
