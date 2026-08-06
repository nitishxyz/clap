use super::*;

impl<B: LogicalBlockPhysicalBackend> LogicalBlockStore<B> {
    pub fn telemetry(&self) -> LogicalBlockTelemetry {
        let mut telemetry = LogicalBlockTelemetry {
            tables: self.tables.len() as u64,
            blocks: self.blocks.len() as u64,
            ..LogicalBlockTelemetry::default()
        };
        for block in self.blocks.values() {
            telemetry.unique_physical_bytes = telemetry
                .unique_physical_bytes
                .saturating_add(block.physical_bytes);
            telemetry.logical_referenced_bytes = telemetry
                .logical_referenced_bytes
                .saturating_add(block.physical_bytes.saturating_mul(block.references));
            telemetry.logical_referenced_tokens = telemetry
                .logical_referenced_tokens
                .saturating_add((block.key.tokens.len() as u64).saturating_mul(block.references));
            telemetry.read_leases = telemetry
                .read_leases
                .saturating_add(block.read_leases.len() as u64);
            telemetry.write_leases = telemetry
                .write_leases
                .saturating_add(u64::from(block.write_lease.is_some()));
        }
        telemetry
    }

    pub fn validate_invariants(&self) -> Result<(), LogicalBlockError> {
        let mut expected_references = BTreeMap::<LogicalBlockId, u64>::new();
        for (owner, table) in &self.tables {
            owner.validate()?;
            if table.generation == 0 || table.blocks.is_empty() {
                return Err(LogicalBlockError::InvariantViolation);
            }
            for block_ref in &table.blocks {
                let block = self
                    .blocks
                    .get(&block_ref.id)
                    .ok_or(LogicalBlockError::InvariantViolation)?;
                if block.block != *block_ref || block.key.namespace != owner.namespace() {
                    return Err(LogicalBlockError::InvariantViolation);
                }
                *expected_references.entry(block_ref.id).or_default() += 1;
            }
        }
        if self.block_index.len() != self.blocks.len() {
            return Err(LogicalBlockError::InvariantViolation);
        }
        for (id, block) in &self.blocks {
            if block.block.id != *id
                || block.block.generation == 0
                || block.physical_bytes == 0
                || block.key.tokens.is_empty()
                || block.references != expected_references.get(id).copied().unwrap_or(0)
                || (block.references == 0 && !block.leased())
                || (!block.read_leases.is_empty() && block.write_lease.is_some())
                || self.block_index.get(&block.key) != Some(&block.block)
            {
                return Err(LogicalBlockError::InvariantViolation);
            }
        }
        let mut physical = self
            .backend
            .inspect()
            .map_err(LogicalBlockError::BackendInspect)?;
        physical.sort_by_key(|entry| (entry.physical.id, entry.physical.generation));
        let mut expected_physical = self
            .blocks
            .values()
            .map(|block| PhysicalBlockSnapshot {
                physical: block.physical,
                physical_bytes: block.physical_bytes,
            })
            .collect::<Vec<_>>();
        expected_physical.sort_by_key(|entry| (entry.physical.id, entry.physical.generation));
        if physical != expected_physical {
            return Err(LogicalBlockError::InvariantViolation);
        }
        Ok(())
    }
}
