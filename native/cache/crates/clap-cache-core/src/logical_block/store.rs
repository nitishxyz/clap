use super::*;

impl<B: LogicalBlockPhysicalBackend> LogicalBlockStore<B> {
    pub fn new(backend: B) -> Result<Self, LogicalBlockError> {
        if !backend
            .inspect()
            .map_err(LogicalBlockError::BackendInspect)?
            .is_empty()
        {
            return Err(LogicalBlockError::BackendNotEmpty);
        }
        Ok(Self {
            backend,
            blocks: BTreeMap::new(),
            block_index: BTreeMap::new(),
            tables: BTreeMap::new(),
            next_block_id: 1,
            next_transaction_id: 1,
            next_lease_id: 1,
            clock: 0,
        })
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn into_backend(self) -> B {
        self.backend
    }

    pub fn table(&self, owner: &BlockTableOwner) -> Option<BlockTableSnapshot> {
        self.tables
            .get(owner)
            .map(|table| self.table_snapshot(owner, table))
    }

    pub fn block(&self, block: LogicalBlockRef) -> Result<LogicalBlockSnapshot, LogicalBlockError> {
        Ok(self.stored_block(block)?.snapshot())
    }

    pub fn blocks(&self) -> Vec<LogicalBlockSnapshot> {
        self.blocks.values().map(StoredBlock::snapshot).collect()
    }

    pub fn tables(&self) -> Vec<BlockTableSnapshot> {
        self.tables
            .iter()
            .map(|(owner, table)| self.table_snapshot(owner, table))
            .collect()
    }

    pub fn publish_new(
        &mut self,
        owner: BlockTableOwner,
        specs: Vec<LogicalBlockSpec>,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        self.update_table(
            BlockTransactionKind::Publish,
            owner,
            None,
            UpdateMode::Replace,
            specs,
        )
    }

    pub fn replace(
        &mut self,
        table: &BlockTableRef,
        specs: Vec<LogicalBlockSpec>,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        self.update_table(
            BlockTransactionKind::Publish,
            table.owner.clone(),
            Some(table.generation),
            UpdateMode::Replace,
            specs,
        )
    }

    pub fn append(
        &mut self,
        table: &BlockTableRef,
        specs: Vec<LogicalBlockSpec>,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        self.update_table(
            BlockTransactionKind::Publish,
            table.owner.clone(),
            Some(table.generation),
            UpdateMode::Append,
            specs,
        )
    }

    pub fn import_new(
        &mut self,
        owner: BlockTableOwner,
        specs: Vec<LogicalBlockSpec>,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        self.update_table(
            BlockTransactionKind::Import,
            owner,
            None,
            UpdateMode::Replace,
            specs,
        )
    }

    pub fn replace_import(
        &mut self,
        table: &BlockTableRef,
        specs: Vec<LogicalBlockSpec>,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        self.update_table(
            BlockTransactionKind::Import,
            table.owner.clone(),
            Some(table.generation),
            UpdateMode::Replace,
            specs,
        )
    }

    pub fn migrate_sequence(
        &mut self,
        owner: BlockTableOwner,
        expected_generation: Option<BlockTableGeneration>,
        model_domain: [u8; 32],
        tokens: &[i32],
        block_tokens: usize,
        bytes_per_token: u64,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        if tokens.is_empty() || block_tokens == 0 || bytes_per_token == 0 {
            return Err(LogicalBlockError::InvalidArgument);
        }
        let specs = tokens
            .chunks(block_tokens)
            .map(|chunk| {
                let physical_bytes = u64::try_from(chunk.len())
                    .ok()
                    .and_then(|len| len.checked_mul(bytes_per_token))
                    .ok_or(LogicalBlockError::ArithmeticOverflow)?;
                Ok(LogicalBlockSpec {
                    model_domain,
                    tokens: chunk.to_vec(),
                    physical_bytes,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.update_table(
            BlockTransactionKind::Import,
            owner,
            expected_generation,
            UpdateMode::Replace,
            specs,
        )
    }

    pub fn fork(
        &mut self,
        source: &BlockTableRef,
        target: BlockTableOwner,
    ) -> Result<BlockTableSnapshot, LogicalBlockError> {
        target.validate()?;
        let source_table = self.valid_table(source)?.clone();
        if self.tables.contains_key(&target) {
            return Err(LogicalBlockError::TableAlreadyExists);
        }
        if target.namespace() != source.owner.namespace() {
            return Err(LogicalBlockError::IncompatibleNamespace);
        }
        for block in &source_table.blocks {
            let stored = self.stored_block(*block)?;
            if stored.key.namespace != target.namespace() {
                return Err(LogicalBlockError::IncompatibleNamespace);
            }
        }
        for block in &source_table.blocks {
            self.blocks
                .get_mut(&block.id)
                .expect("validated source block must exist")
                .references += 1;
        }
        self.clock = self.clock.saturating_add(1);
        self.tables.insert(
            target.clone(),
            StoredTable {
                generation: 1,
                blocks: source_table.blocks,
                last_used: self.clock,
            },
        );
        Ok(self.table(&target).expect("forked table must exist"))
    }

    pub fn release_table(&mut self, table: &BlockTableRef) -> Result<u64, LogicalBlockError> {
        self.remove_tables(BlockTransactionKind::Release, std::slice::from_ref(table))
    }

    pub fn touch(&mut self, table: &BlockTableRef) -> Result<(), LogicalBlockError> {
        self.valid_table(table)?;
        self.clock = self.clock.saturating_add(1);
        self.tables
            .get_mut(&table.owner)
            .expect("validated table must exist")
            .last_used = self.clock;
        Ok(())
    }

    pub fn acquire_read(
        &mut self,
        block: LogicalBlockRef,
    ) -> Result<BlockLease, LogicalBlockError> {
        let lease_id = self.allocate_lease_id();
        let stored = self.stored_block_mut(block)?;
        if stored.write_lease.is_some() {
            return Err(LogicalBlockError::LeaseConflict);
        }
        stored.read_leases.insert(lease_id);
        Ok(BlockLease {
            id: lease_id,
            block,
            kind: BlockLeaseKind::Read,
        })
    }

    pub fn acquire_write(
        &mut self,
        block: LogicalBlockRef,
    ) -> Result<BlockLease, LogicalBlockError> {
        let lease_id = self.allocate_lease_id();
        let stored = self.stored_block_mut(block)?;
        if stored.write_lease.is_some() || !stored.read_leases.is_empty() {
            return Err(LogicalBlockError::LeaseConflict);
        }
        stored.write_lease = Some(lease_id);
        Ok(BlockLease {
            id: lease_id,
            block,
            kind: BlockLeaseKind::Write,
        })
    }

    pub fn release_lease(&mut self, lease: BlockLease) -> Result<(), LogicalBlockError> {
        let stored = self.stored_block(lease.block)?;
        let valid = match lease.kind {
            BlockLeaseKind::Read => stored.read_leases.contains(&lease.id),
            BlockLeaseKind::Write => stored.write_lease == Some(lease.id),
        };
        if !valid {
            return Err(LogicalBlockError::InvalidLease);
        }
        let last_owner = stored.references == 0
            && stored.read_leases.len() == usize::from(lease.kind == BlockLeaseKind::Read)
            && (stored.write_lease.is_none() || lease.kind == BlockLeaseKind::Write);
        if last_owner {
            let physical = stored.physical;
            self.run_physical_transaction(BlockTransactionKind::Reclaim, &[], &[physical])?;
        }
        {
            let stored = self.stored_block_mut(lease.block)?;
            match lease.kind {
                BlockLeaseKind::Read => {
                    stored.read_leases.remove(&lease.id);
                }
                BlockLeaseKind::Write => stored.write_lease = None,
            }
        }
        if last_owner {
            self.remove_unreferenced_block(lease.block.id);
        }
        Ok(())
    }

    pub fn plan_eviction(&self, requested_bytes: u64) -> BlockEvictionPlan {
        if requested_bytes == 0 {
            return BlockEvictionPlan {
                requested_bytes,
                reclaimable_bytes: 0,
                tables: Vec::new(),
            };
        }
        let mut candidates = self
            .tables
            .iter()
            .map(|(owner, table)| (table.last_used, owner.clone(), table.generation))
            .collect::<Vec<_>>();
        candidates.sort();
        let mut selected_counts = BTreeMap::<LogicalBlockId, u64>::new();
        let mut tables = Vec::new();
        let mut reclaimable_bytes = 0;
        for (_, owner, generation) in candidates {
            let table = &self.tables[&owner];
            for block in &table.blocks {
                *selected_counts.entry(block.id).or_default() += 1;
            }
            tables.push(BlockTableRef { owner, generation });
            reclaimable_bytes = selected_counts
                .iter()
                .filter_map(|(id, selected)| {
                    let block = &self.blocks[id];
                    (*selected == block.references && !block.leased())
                        .then_some(block.physical_bytes)
                })
                .sum();
            if reclaimable_bytes >= requested_bytes {
                break;
            }
        }
        BlockEvictionPlan {
            requested_bytes,
            reclaimable_bytes,
            tables,
        }
    }

    pub fn execute_eviction(&mut self, plan: &BlockEvictionPlan) -> Result<u64, LogicalBlockError> {
        self.remove_tables(BlockTransactionKind::Evict, &plan.tables)
    }
}
