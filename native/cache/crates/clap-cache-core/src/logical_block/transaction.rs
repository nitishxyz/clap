use super::*;

impl<B: LogicalBlockPhysicalBackend> LogicalBlockStore<B> {
    pub(super) fn update_table(
        &mut self,
        kind: BlockTransactionKind,
        owner: BlockTableOwner,
        expected_generation: Option<BlockTableGeneration>,
        mode: UpdateMode,
        specs: Vec<LogicalBlockSpec>,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        owner.validate()?;
        if specs.is_empty() {
            return Err(LogicalBlockError::InvalidArgument);
        }
        for spec in &specs {
            spec.validate()?;
        }
        let old_table = self.tables.get(&owner).cloned();
        match (expected_generation, &old_table) {
            (None, Some(_)) => return Err(LogicalBlockError::TableAlreadyExists),
            (Some(_), None) => return Err(LogicalBlockError::TableNotFound),
            (Some(expected), Some(table)) if expected != table.generation => {
                return Err(LogicalBlockError::StaleGeneration)
            }
            (None, None) | (Some(_), Some(_)) => {}
        }
        if mode == UpdateMode::Append && old_table.is_none() {
            return Err(LogicalBlockError::TableNotFound);
        }
        let mut desired = if mode == UpdateMode::Append {
            old_table
                .as_ref()
                .expect("append requires existing table")
                .blocks
                .clone()
        } else {
            Vec::new()
        };
        let mut local_new = BTreeMap::<BlockKey, LogicalBlockRef>::new();
        let mut creates = Vec::<PhysicalBlockCreate>::new();
        let mut new_keys = BTreeMap::<LogicalBlockId, BlockKey>::new();
        for spec in specs {
            let key = BlockKey {
                namespace: owner.namespace(),
                model_domain: spec.model_domain,
                tokens: spec.tokens,
            };
            if let Some(existing) = self.block_index.get(&key).copied() {
                if self.blocks[&existing.id].physical_bytes != spec.physical_bytes {
                    return Err(LogicalBlockError::ConflictingBlockMetadata);
                }
                desired.push(existing);
                continue;
            }
            if let Some(existing) = local_new.get(&key).copied() {
                let create = creates
                    .iter()
                    .find(|create| create.block == existing)
                    .expect("local new block must have create record");
                if create.physical_bytes != spec.physical_bytes {
                    return Err(LogicalBlockError::ConflictingBlockMetadata);
                }
                desired.push(existing);
                continue;
            }
            let block = self.allocate_block_ref();
            creates.push(PhysicalBlockCreate {
                block,
                token_count: key.tokens.len() as u64,
                physical_bytes: spec.physical_bytes,
            });
            local_new.insert(key.clone(), block);
            new_keys.insert(block.id, key);
            desired.push(block);
        }
        let old_blocks = old_table
            .as_ref()
            .map(|table| table.blocks.as_slice())
            .unwrap_or_default();
        let old_counts = counts(old_blocks);
        let new_counts = counts(&desired);
        let releases = old_counts
            .iter()
            .filter_map(|(id, removed)| {
                let block = &self.blocks[id];
                let retained = new_counts.get(id).copied().unwrap_or(0);
                (block.references == removed.saturating_sub(retained) && !block.leased())
                    .then_some(block.physical)
            })
            .collect::<Vec<_>>();
        let prepared = self.run_physical_transaction(kind, &creates, &releases)?;
        for block in old_blocks {
            self.blocks
                .get_mut(&block.id)
                .expect("old table block must exist")
                .references -= 1;
        }
        for prepared_block in prepared {
            let key = new_keys
                .remove(&prepared_block.block.id)
                .expect("validated prepared block must have a key");
            self.block_index.insert(key.clone(), prepared_block.block);
            self.blocks.insert(
                prepared_block.block.id,
                StoredBlock {
                    block: prepared_block.block,
                    key,
                    physical: prepared_block.physical,
                    physical_bytes: prepared_block.physical_bytes,
                    references: 0,
                    read_leases: BTreeSet::new(),
                    write_lease: None,
                },
            );
        }
        for block in &desired {
            self.blocks
                .get_mut(&block.id)
                .expect("desired block must exist")
                .references += 1;
        }
        let release_ids = releases
            .iter()
            .filter_map(|physical| {
                self.blocks
                    .iter()
                    .find_map(|(id, block)| (block.physical == *physical).then_some(*id))
            })
            .collect::<Vec<_>>();
        for id in release_ids {
            self.remove_unreferenced_block(id);
        }
        self.clock = self.clock.saturating_add(1);
        let generation = old_table
            .map(|table| table.generation.saturating_add(1))
            .unwrap_or(1);
        self.tables.insert(
            owner.clone(),
            StoredTable {
                generation,
                blocks: desired,
                last_used: self.clock,
            },
        );
        Ok(self.table(&owner).expect("updated table must exist"))
    }

    pub(super) fn remove_tables(
        &mut self,
        kind: BlockTransactionKind,
        tables: &[BlockTableRef],
    ) -> Result<u64, LogicalBlockError> {
        if tables.is_empty() {
            return Ok(0);
        }
        let mut owners = BTreeSet::new();
        let mut removed_counts = BTreeMap::<LogicalBlockId, u64>::new();
        for table_ref in tables {
            let table = self.valid_table(table_ref)?;
            if !owners.insert(table_ref.owner.clone()) {
                return Err(LogicalBlockError::InvalidArgument);
            }
            for block in &table.blocks {
                *removed_counts.entry(block.id).or_default() += 1;
            }
        }
        let release_blocks = removed_counts
            .iter()
            .filter_map(|(id, removed)| {
                let block = &self.blocks[id];
                (block.references == *removed && !block.leased()).then_some((*id, block.physical))
            })
            .collect::<Vec<_>>();
        let releases = release_blocks
            .iter()
            .map(|(_, physical)| *physical)
            .collect::<Vec<_>>();
        self.run_physical_transaction(kind, &[], &releases)?;
        for (id, removed) in removed_counts {
            self.blocks
                .get_mut(&id)
                .expect("validated table block must exist")
                .references -= removed;
        }
        for owner in owners {
            self.tables.remove(&owner);
        }
        let reclaimed = release_blocks
            .iter()
            .map(|(id, _)| self.blocks[id].physical_bytes)
            .sum();
        for (id, _) in release_blocks {
            self.remove_unreferenced_block(id);
        }
        Ok(reclaimed)
    }

    pub(super) fn run_physical_transaction(
        &mut self,
        kind: BlockTransactionKind,
        creates: &[PhysicalBlockCreate],
        releases: &[PhysicalObjectRef],
    ) -> Result<Vec<PreparedPhysicalBlock>, LogicalBlockError> {
        if creates.is_empty() && releases.is_empty() {
            return Ok(Vec::new());
        }
        let transaction = self.allocate_transaction_id();
        let prepared = self
            .backend
            .prepare(transaction, kind, creates, releases)
            .map_err(LogicalBlockError::BackendPrepare)?;
        let collides_with_live_block = prepared.iter().any(|outcome| {
            self.blocks
                .values()
                .any(|block| block.physical == outcome.physical)
        });
        if !valid_prepared(creates, &prepared) || collides_with_live_block {
            let abort = self.backend.abort(transaction);
            return match abort {
                Ok(()) => Err(LogicalBlockError::InvalidBackendOutcome),
                Err(error) => Err(LogicalBlockError::BackendAbort(error)),
            };
        }
        if let Err(error) = self.backend.commit(transaction) {
            return match self.backend.abort(transaction) {
                Ok(()) => Err(LogicalBlockError::BackendCommit(error)),
                Err(abort) => Err(LogicalBlockError::BackendAbort(format!(
                    "commit: {error}; abort: {abort}"
                ))),
            };
        }
        Ok(prepared)
    }

    pub(super) fn valid_table(
        &self,
        table: &BlockTableRef,
    ) -> Result<&StoredTable, LogicalBlockError> {
        let stored = self
            .tables
            .get(&table.owner)
            .ok_or(LogicalBlockError::TableNotFound)?;
        if stored.generation != table.generation {
            return Err(LogicalBlockError::StaleGeneration);
        }
        Ok(stored)
    }

    pub(super) fn stored_block(
        &self,
        block: LogicalBlockRef,
    ) -> Result<&StoredBlock, LogicalBlockError> {
        let stored = self
            .blocks
            .get(&block.id)
            .ok_or(LogicalBlockError::BlockNotFound)?;
        if stored.block.generation != block.generation {
            return Err(LogicalBlockError::StaleGeneration);
        }
        Ok(stored)
    }

    pub(super) fn stored_block_mut(
        &mut self,
        block: LogicalBlockRef,
    ) -> Result<&mut StoredBlock, LogicalBlockError> {
        let stored = self
            .blocks
            .get_mut(&block.id)
            .ok_or(LogicalBlockError::BlockNotFound)?;
        if stored.block.generation != block.generation {
            return Err(LogicalBlockError::StaleGeneration);
        }
        Ok(stored)
    }

    pub(super) fn table_snapshot(
        &self,
        owner: &BlockTableOwner,
        table: &StoredTable,
    ) -> BlockTableSnapshot {
        let logical_tokens = table
            .blocks
            .iter()
            .map(|block| self.blocks[&block.id].key.tokens.len() as u64)
            .sum();
        let logical_bytes = table
            .blocks
            .iter()
            .map(|block| self.blocks[&block.id].physical_bytes)
            .sum();
        let unique_bytes = table
            .blocks
            .iter()
            .map(|block| block.id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|id| self.blocks[&id].physical_bytes)
            .sum();
        BlockTableSnapshot {
            table: BlockTableRef {
                owner: owner.clone(),
                generation: table.generation,
            },
            blocks: table.blocks.clone(),
            logical_tokens,
            logical_bytes,
            unique_bytes,
            last_used: table.last_used,
        }
    }

    pub(super) fn remove_unreferenced_block(&mut self, id: LogicalBlockId) {
        let block = self.blocks.remove(&id).expect("reclaimed block must exist");
        debug_assert_eq!(block.references, 0);
        debug_assert!(!block.leased());
        self.block_index.remove(&block.key);
    }

    fn allocate_block_ref(&mut self) -> LogicalBlockRef {
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.saturating_add(1);
        LogicalBlockRef { id, generation: 1 }
    }

    fn allocate_transaction_id(&mut self) -> BlockTransactionId {
        let id = self.next_transaction_id;
        self.next_transaction_id = self.next_transaction_id.saturating_add(1);
        id
    }

    pub(super) fn allocate_lease_id(&mut self) -> BlockLeaseId {
        let id = self.next_lease_id;
        self.next_lease_id = self.next_lease_id.saturating_add(1);
        id
    }
}

fn counts(blocks: &[LogicalBlockRef]) -> BTreeMap<LogicalBlockId, u64> {
    let mut counts = BTreeMap::new();
    for block in blocks {
        *counts.entry(block.id).or_default() += 1;
    }
    counts
}

fn valid_prepared(creates: &[PhysicalBlockCreate], prepared: &[PreparedPhysicalBlock]) -> bool {
    if creates.len() != prepared.len() {
        return false;
    }
    let mut physical = BTreeSet::new();
    creates.iter().zip(prepared).all(|(create, outcome)| {
        create.block == outcome.block
            && create.physical_bytes == outcome.physical_bytes
            && outcome.physical.id != 0
            && outcome.physical.generation != 0
            && physical.insert((outcome.physical.id, outcome.physical.generation))
    })
}
