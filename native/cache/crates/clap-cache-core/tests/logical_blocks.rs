use clap_cache_core::adapter::PhysicalObjectRef;
use clap_cache_core::logical_block::{
    BlockLeaseKind, BlockTableOwner, BlockTransactionId, BlockTransactionKind,
    InMemoryBlockBackend, LogicalBlockError, LogicalBlockPhysicalBackend, LogicalBlockSpec,
    LogicalBlockStore, PhysicalBlockCreate, PhysicalBlockSnapshot, PreparedPhysicalBlock,
};
use clap_cache_core::Namespace;

const DOMAIN: [u8; 32] = [4; 32];
const OTHER_DOMAIN: [u8; 32] = [5; 32];
const NAMESPACE: Namespace = Namespace([7; 32]);
const OTHER_NAMESPACE: Namespace = Namespace([8; 32]);

fn prefix(id: u64) -> BlockTableOwner {
    BlockTableOwner::Prefix {
        namespace: NAMESPACE,
        prefix: id,
    }
}

fn session(id: u64) -> BlockTableOwner {
    BlockTableOwner::Session {
        namespace: NAMESPACE,
        session: id,
    }
}

fn spec(tokens: &[i32], bytes: u64) -> LogicalBlockSpec {
    LogicalBlockSpec {
        model_domain: DOMAIN,
        tokens: tokens.to_vec(),
        physical_bytes: bytes,
    }
}

fn store() -> LogicalBlockStore<InMemoryBlockBackend> {
    LogicalBlockStore::new(InMemoryBlockBackend::new()).unwrap()
}

struct InvalidOutcomeBackend(InMemoryBlockBackend);

impl LogicalBlockPhysicalBackend for InvalidOutcomeBackend {
    fn prepare(
        &mut self,
        transaction: BlockTransactionId,
        kind: BlockTransactionKind,
        creates: &[PhysicalBlockCreate],
        releases: &[PhysicalObjectRef],
    ) -> Result<Vec<PreparedPhysicalBlock>, String> {
        let mut prepared = self.0.prepare(transaction, kind, creates, releases)?;
        if let Some(first) = prepared.first_mut() {
            first.physical_bytes += 1;
        }
        Ok(prepared)
    }

    fn commit(&mut self, transaction: BlockTransactionId) -> Result<(), String> {
        self.0.commit(transaction)
    }

    fn abort(&mut self, transaction: BlockTransactionId) -> Result<(), String> {
        self.0.abort(transaction)
    }

    fn inspect(&self) -> Result<Vec<PhysicalBlockSnapshot>, String> {
        self.0.inspect()
    }
}

#[test]
fn copy_on_write_fork_shares_immutable_blocks_and_isolates_appends() {
    let mut store = store();
    let source = store
        .publish_new(prefix(1), vec![spec(&[1, 2], 100), spec(&[3, 4], 100)])
        .unwrap();
    let fork = store.fork(&source.table, session(1)).unwrap();

    assert_eq!(source.blocks, fork.blocks);
    assert_eq!(store.telemetry().unique_physical_bytes, 200);
    assert_eq!(store.telemetry().logical_referenced_bytes, 400);
    for block in &source.blocks {
        assert_eq!(store.block(*block).unwrap().references, 2);
    }

    let appended = store.append(&fork.table, vec![spec(&[5, 6], 100)]).unwrap();
    assert_eq!(appended.blocks.len(), 3);
    assert_eq!(store.table(&prefix(1)).unwrap().blocks, source.blocks);
    assert_eq!(store.telemetry().unique_physical_bytes, 300);
    assert_eq!(store.telemetry().logical_referenced_bytes, 500);
    assert_eq!(store.block(appended.blocks[0]).unwrap().references, 2);
    assert_eq!(store.block(appended.blocks[2]).unwrap().references, 1);
    store.validate_invariants().unwrap();
}

#[test]
fn referenced_or_leased_blocks_are_never_reclaimed() {
    let mut store = store();
    let first = store
        .publish_new(prefix(1), vec![spec(&[1, 2], 128)])
        .unwrap();
    let second = store.fork(&first.table, session(1)).unwrap();
    let block = first.blocks[0];
    let read = store.acquire_read(block).unwrap();

    assert_eq!(store.release_table(&first.table).unwrap(), 0);
    assert_eq!(store.block(block).unwrap().references, 1);
    assert_eq!(store.backend().object_count(), 1);
    assert_eq!(store.release_table(&second.table).unwrap(), 0);
    assert_eq!(store.block(block).unwrap().references, 0);
    assert_eq!(store.backend().object_count(), 1);
    assert_eq!(
        store.acquire_write(block),
        Err(LogicalBlockError::LeaseConflict)
    );

    store.release_lease(read).unwrap();
    assert_eq!(store.backend().object_count(), 0);
    assert_eq!(store.telemetry().blocks, 0);
    store.validate_invariants().unwrap();
}

#[test]
fn write_leases_are_exclusive_and_generation_guarded() {
    let mut store = store();
    let table = store
        .publish_new(prefix(1), vec![spec(&[1, 2], 128)])
        .unwrap();
    let block = table.blocks[0];
    let write = store.acquire_write(block).unwrap();
    assert_eq!(write.kind, BlockLeaseKind::Write);
    assert_eq!(
        store.acquire_read(block),
        Err(LogicalBlockError::LeaseConflict)
    );
    assert_eq!(
        store.acquire_write(block),
        Err(LogicalBlockError::LeaseConflict)
    );

    let mut stale = write;
    stale.block.generation += 1;
    assert_eq!(
        store.release_lease(stale),
        Err(LogicalBlockError::StaleGeneration)
    );
    assert_eq!(store.telemetry().write_leases, 1);
    store.release_lease(write).unwrap();
    assert_eq!(store.telemetry().write_leases, 0);
    store.validate_invariants().unwrap();
}

#[test]
fn publish_import_and_release_are_atomic_on_backend_failure() {
    let mut store = store();
    store.backend_mut().fail_next_prepare("allocation failed");
    assert_eq!(
        store.publish_new(prefix(1), vec![spec(&[1, 2], 100)]),
        Err(LogicalBlockError::BackendPrepare(
            "allocation failed".to_owned()
        ))
    );
    assert!(store.tables().is_empty());
    assert!(store.blocks().is_empty());

    store.backend_mut().fail_next_commit("import commit failed");
    assert_eq!(
        store.import_new(prefix(2), vec![spec(&[3, 4], 100)]),
        Err(LogicalBlockError::BackendCommit(
            "import commit failed".to_owned()
        ))
    );
    assert_eq!(store.backend().staged_count(), 0);
    assert_eq!(store.backend().object_count(), 0);
    assert!(store.tables().is_empty());

    let table = store
        .publish_new(prefix(3), vec![spec(&[5, 6], 100)])
        .unwrap();
    let before = store.table(&prefix(3)).unwrap();
    store
        .backend_mut()
        .fail_next_commit("release commit failed");
    assert_eq!(
        store.release_table(&table.table),
        Err(LogicalBlockError::BackendCommit(
            "release commit failed".to_owned()
        ))
    );
    assert_eq!(store.table(&prefix(3)), Some(before));
    assert_eq!(store.backend().object_count(), 1);
    assert_eq!(store.backend().staged_count(), 0);
    store.validate_invariants().unwrap();
}

#[test]
fn invalid_physical_outcomes_abort_before_logical_publication() {
    let backend = InvalidOutcomeBackend(InMemoryBlockBackend::new());
    let mut store = LogicalBlockStore::new(backend).unwrap();

    assert_eq!(
        store.publish_new(prefix(1), vec![spec(&[1, 2], 100)]),
        Err(LogicalBlockError::InvalidBackendOutcome)
    );
    assert!(store.tables().is_empty());
    assert!(store.blocks().is_empty());
    assert_eq!(store.backend().0.object_count(), 0);
    assert_eq!(store.backend().0.staged_count(), 0);
    store.validate_invariants().unwrap();
}

#[test]
fn stale_table_generations_change_no_references_or_physical_state() {
    let mut store = store();
    let first = store
        .publish_new(prefix(1), vec![spec(&[1, 2], 100)])
        .unwrap();
    let second = store
        .append(&first.table, vec![spec(&[3, 4], 100)])
        .unwrap();
    let before_tables = store.tables();
    let before_blocks = store.blocks();
    let before_objects = store.backend().object_count();

    assert_eq!(
        store.append(&first.table, vec![spec(&[5, 6], 100)]),
        Err(LogicalBlockError::StaleGeneration)
    );
    assert_eq!(
        store.release_table(&first.table),
        Err(LogicalBlockError::StaleGeneration)
    );
    assert_eq!(store.tables(), before_tables);
    assert_eq!(store.blocks(), before_blocks);
    assert_eq!(store.backend().object_count(), before_objects);
    assert_eq!(store.table(&prefix(1)).unwrap().table, second.table);
    store.validate_invariants().unwrap();
}

#[test]
fn eviction_counts_shared_bytes_once_and_reclaims_atomically() {
    let mut store = store();
    let first = store
        .publish_new(prefix(1), vec![spec(&[1], 100), spec(&[2], 100)])
        .unwrap();
    let second = store
        .publish_new(prefix(2), vec![spec(&[1], 100), spec(&[3], 100)])
        .unwrap();
    assert_eq!(first.blocks[0], second.blocks[0]);
    assert_eq!(store.telemetry().unique_physical_bytes, 300);
    assert_eq!(store.telemetry().logical_referenced_bytes, 400);

    let plan = store.plan_eviction(150);
    assert!(plan.satisfied());
    assert_eq!(plan.tables.len(), 2);
    assert_eq!(plan.reclaimable_bytes, 300);
    assert_eq!(store.execute_eviction(&plan).unwrap(), 300);
    assert!(store.tables().is_empty());
    assert!(store.blocks().is_empty());
    assert_eq!(store.backend().object_count(), 0);
    store.validate_invariants().unwrap();
}

#[test]
fn sequence_trace_migration_is_deterministic_and_namespace_isolated() {
    let mut store = store();
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
    let first = store
        .migrate_sequence(prefix(1), None, DOMAIN, &tokens, 4, 16)
        .unwrap();
    let second = store
        .migrate_sequence(prefix(2), None, DOMAIN, &tokens, 4, 16)
        .unwrap();
    assert_eq!(first.blocks, second.blocks);
    assert_eq!(store.telemetry().unique_physical_bytes, 128);
    assert_eq!(store.telemetry().logical_referenced_bytes, 256);

    let replay = store
        .migrate_sequence(
            prefix(1),
            Some(first.table.generation),
            DOMAIN,
            &tokens,
            4,
            16,
        )
        .unwrap();
    assert_eq!(replay.blocks, first.blocks);
    assert_eq!(replay.table.generation, first.table.generation + 1);

    let isolated_owner = BlockTableOwner::Prefix {
        namespace: OTHER_NAMESPACE,
        prefix: 1,
    };
    let isolated = store
        .migrate_sequence(isolated_owner, None, DOMAIN, &tokens, 4, 16)
        .unwrap();
    assert_ne!(isolated.blocks, first.blocks);
    assert_eq!(store.telemetry().unique_physical_bytes, 256);
    store.validate_invariants().unwrap();
}

#[test]
fn model_domain_and_exact_tokens_define_block_compatibility() {
    let mut store = store();
    let first = store
        .publish_new(prefix(1), vec![spec(&[1, 2], 100)])
        .unwrap();
    let other_domain = LogicalBlockSpec {
        model_domain: OTHER_DOMAIN,
        tokens: vec![1, 2],
        physical_bytes: 100,
    };
    let second = store.publish_new(prefix(2), vec![other_domain]).unwrap();
    assert_ne!(first.blocks, second.blocks);

    assert_eq!(
        store.publish_new(prefix(3), vec![spec(&[1, 2], 101)]),
        Err(LogicalBlockError::ConflictingBlockMetadata)
    );
    store.validate_invariants().unwrap();
}

#[test]
fn generated_operation_sequences_preserve_reference_and_backend_invariants() {
    for seed in 1..=32_u64 {
        let mut store = store();
        let mut random = 0x4d595df4d0f33173_u64 ^ seed;
        for step in 0..250_u64 {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let id = (random % 12) + 1;
            let owner = if random & 1 == 0 {
                prefix(id)
            } else {
                session(id)
            };
            let existing = store.table(&owner);
            match (random >> 8) % 7 {
                0 if existing.is_none() => {
                    let token = ((step + id + seed) % 17) as i32;
                    let _ = store.publish_new(owner, vec![spec(&[token], 64)]);
                }
                1 if existing.is_some() => {
                    let table = existing.unwrap();
                    let token = ((step * 3 + id + seed) % 19) as i32;
                    let _ = store.append(&table.table, vec![spec(&[token], 64)]);
                }
                2 if existing.is_some() => {
                    let table = existing.unwrap();
                    let token = ((step * 5 + id + seed) % 23) as i32;
                    let _ = store.replace(&table.table, vec![spec(&[token], 64)]);
                }
                3 if existing.is_some() => {
                    let table = existing.unwrap();
                    let _ = store.release_table(&table.table);
                }
                4 => {
                    if let Some(source) = store.tables().first().cloned() {
                        let target = session(id + 100);
                        if store.table(&target).is_none() {
                            let _ = store.fork(&source.table, target);
                        }
                    }
                }
                5 => {
                    if let Some(block) = store.blocks().first().cloned() {
                        if let Ok(lease) = store.acquire_read(block.block) {
                            store.release_lease(lease).unwrap();
                        }
                    }
                }
                _ => {
                    let plan = store.plan_eviction(64);
                    let _ = store.execute_eviction(&plan);
                }
            }
            store.validate_invariants().unwrap();
        }
    }
}
