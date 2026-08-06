//! Reference-counted immutable logical blocks and versioned block tables.
//!
//! This module is deliberately independent of sequence-slot policy. It proves
//! ownership, lease, accounting, copy-on-write, and transactional invariants
//! against a narrow physical backend before native paged engines are involved.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{adapter::PhysicalObjectRef, Namespace};

mod audit;
mod memory;
mod store;
mod transaction;

pub use memory::InMemoryBlockBackend;

pub type LogicalBlockId = u64;
pub type BlockTableGeneration = u64;
pub type BlockTransactionId = u64;
pub type BlockLeaseId = u64;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LogicalBlockRef {
    pub id: LogicalBlockId,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BlockTableOwner {
    Prefix { namespace: Namespace, prefix: u64 },
    Session { namespace: Namespace, session: u64 },
}

impl BlockTableOwner {
    pub const fn namespace(&self) -> Namespace {
        match self {
            Self::Prefix { namespace, .. } | Self::Session { namespace, .. } => *namespace,
        }
    }

    fn validate(&self) -> Result<(), LogicalBlockError> {
        match self {
            Self::Prefix { prefix, .. } if *prefix == 0 => Err(LogicalBlockError::InvalidArgument),
            Self::Session { session, .. } if *session == 0 => {
                Err(LogicalBlockError::InvalidArgument)
            }
            Self::Prefix { .. } | Self::Session { .. } => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockTableRef {
    pub owner: BlockTableOwner,
    pub generation: BlockTableGeneration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalBlockSpec {
    pub model_domain: [u8; 32],
    pub tokens: Vec<i32>,
    pub physical_bytes: u64,
}

impl LogicalBlockSpec {
    fn validate(&self) -> Result<(), LogicalBlockError> {
        if self.tokens.is_empty() || self.physical_bytes == 0 {
            return Err(LogicalBlockError::InvalidArgument);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct BlockKey {
    namespace: Namespace,
    model_domain: [u8; 32],
    tokens: Vec<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalBlockSnapshot {
    pub block: LogicalBlockRef,
    pub namespace: Namespace,
    pub model_domain: [u8; 32],
    pub tokens: Vec<i32>,
    pub physical: PhysicalObjectRef,
    pub physical_bytes: u64,
    pub references: u64,
    pub read_leases: u32,
    pub write_leased: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockTableSnapshot {
    pub table: BlockTableRef,
    pub blocks: Vec<LogicalBlockRef>,
    pub logical_tokens: u64,
    pub logical_bytes: u64,
    pub unique_bytes: u64,
    pub last_used: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockLeaseKind {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockLease {
    pub id: BlockLeaseId,
    pub block: LogicalBlockRef,
    pub kind: BlockLeaseKind,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LogicalBlockTelemetry {
    pub tables: u64,
    pub blocks: u64,
    pub unique_physical_bytes: u64,
    pub logical_referenced_bytes: u64,
    pub logical_referenced_tokens: u64,
    pub read_leases: u64,
    pub write_leases: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockEvictionPlan {
    pub requested_bytes: u64,
    pub reclaimable_bytes: u64,
    pub tables: Vec<BlockTableRef>,
}

impl BlockEvictionPlan {
    pub const fn satisfied(&self) -> bool {
        self.reclaimable_bytes >= self.requested_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockTransactionKind {
    Publish,
    Import,
    Release,
    Evict,
    Reclaim,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalBlockCreate {
    pub block: LogicalBlockRef,
    pub token_count: u64,
    pub physical_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPhysicalBlock {
    pub block: LogicalBlockRef,
    pub physical: PhysicalObjectRef,
    pub physical_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalBlockSnapshot {
    pub physical: PhysicalObjectRef,
    pub physical_bytes: u64,
}

pub trait LogicalBlockPhysicalBackend {
    fn prepare(
        &mut self,
        transaction: BlockTransactionId,
        kind: BlockTransactionKind,
        creates: &[PhysicalBlockCreate],
        releases: &[PhysicalObjectRef],
    ) -> Result<Vec<PreparedPhysicalBlock>, String>;

    fn commit(&mut self, transaction: BlockTransactionId) -> Result<(), String>;

    fn abort(&mut self, transaction: BlockTransactionId) -> Result<(), String>;

    fn inspect(&self) -> Result<Vec<PhysicalBlockSnapshot>, String>;
}

#[derive(Clone, Debug)]
struct StoredBlock {
    block: LogicalBlockRef,
    key: BlockKey,
    physical: PhysicalObjectRef,
    physical_bytes: u64,
    references: u64,
    read_leases: BTreeSet<BlockLeaseId>,
    write_lease: Option<BlockLeaseId>,
}

impl StoredBlock {
    fn leased(&self) -> bool {
        !self.read_leases.is_empty() || self.write_lease.is_some()
    }

    fn snapshot(&self) -> LogicalBlockSnapshot {
        LogicalBlockSnapshot {
            block: self.block,
            namespace: self.key.namespace,
            model_domain: self.key.model_domain,
            tokens: self.key.tokens.clone(),
            physical: self.physical,
            physical_bytes: self.physical_bytes,
            references: self.references,
            read_leases: self.read_leases.len() as u32,
            write_leased: self.write_lease.is_some(),
        }
    }
}

#[derive(Clone, Debug)]
struct StoredTable {
    generation: BlockTableGeneration,
    blocks: Vec<LogicalBlockRef>,
    last_used: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UpdateMode {
    Replace,
    Append,
}

pub struct LogicalBlockStore<B> {
    backend: B,
    blocks: BTreeMap<LogicalBlockId, StoredBlock>,
    block_index: BTreeMap<BlockKey, LogicalBlockRef>,
    tables: BTreeMap<BlockTableOwner, StoredTable>,
    next_block_id: LogicalBlockId,
    next_transaction_id: BlockTransactionId,
    next_lease_id: BlockLeaseId,
    clock: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogicalBlockError {
    InvalidArgument,
    ArithmeticOverflow,
    TableAlreadyExists,
    TableNotFound,
    BlockNotFound,
    StaleGeneration,
    IncompatibleNamespace,
    ConflictingBlockMetadata,
    LeaseConflict,
    InvalidLease,
    BackendNotEmpty,
    InvalidBackendOutcome,
    BackendPrepare(String),
    BackendCommit(String),
    BackendAbort(String),
    BackendInspect(String),
    InvariantViolation,
}

impl fmt::Display for LogicalBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LogicalBlockError {}
