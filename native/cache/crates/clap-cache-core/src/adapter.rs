//! Versioned contract between backend-neutral cache policy and physical cache
//! implementations. The checked adapter validates negotiation and capabilities
//! before delegating any operation that can mutate backend state.

use std::fmt;

mod descriptors;

pub use descriptors::{
    llama_sequence_descriptor, mlx_sequence_descriptor, paged_engine_skeleton_descriptor,
};

pub const PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterKind {
    Sequence,
    Paged,
    PagedSkeleton,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AdapterOperation {
    Inspect = 0,
    Continue = 1,
    Restore = 2,
    Fork = 3,
    Trim = 4,
    Snapshot = 5,
    Release = 6,
    Export = 7,
    Import = 8,
    Promote = 9,
    Demote = 10,
}

impl AdapterOperation {
    const fn bit(self) -> u16 {
        1 << self as u8
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdapterOperations(u16);

impl AdapterOperations {
    pub const NONE: Self = Self(0);

    pub const fn of(operations: &[AdapterOperation]) -> Self {
        let mut bits = 0;
        let mut index = 0;
        while index < operations.len() {
            bits |= operations[index].bit();
            index += 1;
        }
        Self(bits)
    }

    pub const fn contains(self, operation: AdapterOperation) -> bool {
        self.0 & operation.bit() != 0
    }

    pub const fn bits(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestoreGranularity {
    WholeState,
    Block,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForkSemantics {
    WholeStateCopy,
    CopyOnWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteAccounting {
    Unknown,
    Estimated,
    Exact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheTier {
    Device,
    Host,
    LocalStorage,
    Remote,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferFormat {
    pub identity: String,
    pub version: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalFormatIdentity {
    pub backend: String,
    pub engine: String,
    pub cache_format: String,
    pub cache_format_version: u16,
    pub model_domain: [u8; 32],
    pub kv_data_type: Option<String>,
    pub block_tokens: Option<u32>,
}

impl PhysicalFormatIdentity {
    pub fn import_compatible_with(&self, source: &Self) -> bool {
        self == source
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterConstraints {
    pub restore_granularity: RestoreGranularity,
    pub fork_semantics: ForkSemantics,
    pub minimum_trim_tokens: Option<u32>,
    pub safe_busy_donor: bool,
    pub prompt_boundary_snapshots: bool,
    pub recurrent_or_hybrid: bool,
    pub byte_accounting: ByteAccounting,
    pub tiers: Vec<CacheTier>,
    pub transfer_format: Option<TransferFormat>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterDescriptor {
    pub contract_version: u16,
    pub kind: AdapterKind,
    pub operations: AdapterOperations,
    pub format: PhysicalFormatIdentity,
    pub constraints: AdapterConstraints,
}

impl AdapterDescriptor {
    pub fn validate(&self) -> Result<(), AdapterError> {
        if self.contract_version != PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION {
            return Err(AdapterError::ContractVersion {
                expected: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
                actual: self.contract_version,
            });
        }
        if self.format.backend.is_empty()
            || self.format.engine.is_empty()
            || self.format.cache_format.is_empty()
            || self.format.cache_format_version == 0
            || self.constraints.tiers.is_empty()
            || !self.operations.contains(AdapterOperation::Inspect)
        {
            return Err(AdapterError::InvalidDescriptor);
        }
        if self.operations.contains(AdapterOperation::Trim)
            != self.constraints.minimum_trim_tokens.is_some()
        {
            return Err(AdapterError::InvalidDescriptor);
        }
        if self.operations.contains(AdapterOperation::Export)
            != self.operations.contains(AdapterOperation::Import)
            || self.operations.contains(AdapterOperation::Import)
                != self.constraints.transfer_format.is_some()
        {
            return Err(AdapterError::InvalidDescriptor);
        }
        if self.kind == AdapterKind::Paged && self.format.block_tokens.is_none() {
            return Err(AdapterError::InvalidDescriptor);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PhysicalObjectRef {
    pub id: u64,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterRequest {
    pub contract_version: u16,
    pub operation: AdapterOperation,
    pub target: PhysicalObjectRef,
    pub donor: Option<PhysicalObjectRef>,
    pub token_count: Option<u64>,
    pub source_format: Option<PhysicalFormatIdentity>,
    pub transfer_format: Option<TransferFormat>,
    pub source_tier: Option<CacheTier>,
    pub target_tier: Option<CacheTier>,
}

impl AdapterRequest {
    pub fn new(operation: AdapterOperation, target: PhysicalObjectRef) -> Self {
        Self {
            contract_version: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
            operation,
            target,
            donor: None,
            token_count: None,
            source_format: None,
            transfer_format: None,
            source_tier: None,
            target_tier: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnknownBytesReason {
    BackendUnavailable,
    BackendDoesNotReport,
    ObjectNotMaterialized,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservedBytes {
    Known(u64),
    Unknown(UnknownBytesReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalObjectSnapshot {
    pub object: PhysicalObjectRef,
    pub resident_tokens: u64,
    pub tier: CacheTier,
    pub bytes: ObservedBytes,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterOutcome {
    pub target: PhysicalObjectRef,
    pub resident_tokens: u64,
    pub tier: CacheTier,
    pub bytes: ObservedBytes,
}

pub trait PhysicalCacheBackend {
    fn descriptor(&self) -> &AdapterDescriptor;
    fn inspect(&self) -> Result<Vec<PhysicalObjectSnapshot>, String>;
    fn execute(&mut self, request: &AdapterRequest) -> Result<AdapterOutcome, String>;
    fn invalidate(&mut self, target: PhysicalObjectRef) -> Result<(), String>;
}

pub struct CheckedPhysicalCacheAdapter<B> {
    backend: B,
}

impl<B: PhysicalCacheBackend> CheckedPhysicalCacheAdapter<B> {
    pub fn new(backend: B) -> Result<Self, AdapterError> {
        backend.descriptor().validate()?;
        Ok(Self { backend })
    }

    pub fn descriptor(&self) -> &AdapterDescriptor {
        self.backend.descriptor()
    }

    pub fn inspect(
        &self,
        contract_version: u16,
    ) -> Result<Vec<PhysicalObjectSnapshot>, AdapterError> {
        self.negotiate(contract_version)?;
        self.backend
            .inspect()
            .map_err(AdapterError::PhysicalFailure)
    }

    pub fn execute(&mut self, request: &AdapterRequest) -> Result<AdapterOutcome, AdapterError> {
        self.validate_request(request)?;
        match self.backend.execute(request) {
            Ok(outcome) if self.valid_outcome(request, &outcome) => Ok(outcome),
            Ok(_) => self.invalidate_after_failure(
                request.target,
                "invalid backend outcome".to_owned(),
                AdapterError::InvalidOutcome,
            ),
            Err(error) => self.invalidate_after_failure(
                request.target,
                error.clone(),
                AdapterError::PhysicalFailure(error),
            ),
        }
    }

    pub fn abort(
        &mut self,
        contract_version: u16,
        target: PhysicalObjectRef,
    ) -> Result<(), AdapterError> {
        self.negotiate(contract_version)?;
        self.backend
            .invalidate(target)
            .map_err(AdapterError::PhysicalFailure)
    }

    pub fn into_inner(self) -> B {
        self.backend
    }

    fn negotiate(&self, contract_version: u16) -> Result<(), AdapterError> {
        if contract_version != self.backend.descriptor().contract_version {
            return Err(AdapterError::ContractVersion {
                expected: self.backend.descriptor().contract_version,
                actual: contract_version,
            });
        }
        Ok(())
    }

    fn validate_request(&self, request: &AdapterRequest) -> Result<(), AdapterError> {
        self.negotiate(request.contract_version)?;
        let descriptor = self.backend.descriptor();
        if !descriptor.operations.contains(request.operation) {
            return Err(AdapterError::UnsupportedOperation(request.operation));
        }
        match request.operation {
            AdapterOperation::Restore | AdapterOperation::Fork => {
                if request.donor.is_none() || request.donor == Some(request.target) {
                    return Err(AdapterError::InvalidRequest);
                }
            }
            AdapterOperation::Trim | AdapterOperation::Snapshot => {
                if request.token_count == Some(0) || request.token_count.is_none() {
                    return Err(AdapterError::InvalidRequest);
                }
            }
            AdapterOperation::Import => {
                let source = request
                    .source_format
                    .as_ref()
                    .ok_or(AdapterError::InvalidRequest)?;
                if !descriptor.format.import_compatible_with(source)
                    || request.transfer_format.as_ref()
                        != descriptor.constraints.transfer_format.as_ref()
                {
                    return Err(AdapterError::IncompatibleFormat);
                }
            }
            AdapterOperation::Export => {
                if request.transfer_format.as_ref()
                    != descriptor.constraints.transfer_format.as_ref()
                {
                    return Err(AdapterError::IncompatibleFormat);
                }
            }
            AdapterOperation::Promote | AdapterOperation::Demote => {
                let source = request.source_tier.ok_or(AdapterError::InvalidRequest)?;
                let target = request.target_tier.ok_or(AdapterError::InvalidRequest)?;
                if source == target
                    || !descriptor.constraints.tiers.contains(&source)
                    || !descriptor.constraints.tiers.contains(&target)
                {
                    return Err(AdapterError::InvalidRequest);
                }
            }
            AdapterOperation::Inspect | AdapterOperation::Continue | AdapterOperation::Release => {}
        }
        Ok(())
    }

    fn valid_outcome(&self, request: &AdapterRequest, outcome: &AdapterOutcome) -> bool {
        if outcome.target != request.target
            || !self
                .backend
                .descriptor()
                .constraints
                .tiers
                .contains(&outcome.tier)
        {
            return false;
        }
        match (
            self.backend.descriptor().constraints.byte_accounting,
            &outcome.bytes,
        ) {
            (ByteAccounting::Unknown, ObservedBytes::Known(_))
            | (ByteAccounting::Exact, ObservedBytes::Unknown(_)) => false,
            (ByteAccounting::Unknown, ObservedBytes::Unknown(_))
            | (ByteAccounting::Estimated, _)
            | (ByteAccounting::Exact, ObservedBytes::Known(_)) => true,
        }
    }

    fn invalidate_after_failure<T>(
        &mut self,
        target: PhysicalObjectRef,
        operation: String,
        error: AdapterError,
    ) -> Result<T, AdapterError> {
        match self.backend.invalidate(target) {
            Ok(()) => Err(error),
            Err(invalidation) => Err(AdapterError::InvalidationFailure {
                operation,
                invalidation,
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdapterError {
    ContractVersion {
        expected: u16,
        actual: u16,
    },
    InvalidDescriptor,
    UnsupportedOperation(AdapterOperation),
    InvalidRequest,
    InvalidOutcome,
    IncompatibleFormat,
    PhysicalFailure(String),
    InvalidationFailure {
        operation: String,
        invalidation: String,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AdapterError {}
