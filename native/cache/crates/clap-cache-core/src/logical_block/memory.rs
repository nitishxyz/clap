use super::*;

#[derive(Clone, Debug)]
struct StagedPhysicalTransaction {
    creates: Vec<PreparedPhysicalBlock>,
    releases: Vec<PhysicalObjectRef>,
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryBlockBackend {
    objects: BTreeMap<(u64, u64), u64>,
    staged: BTreeMap<BlockTransactionId, StagedPhysicalTransaction>,
    next_object_id: u64,
    fail_prepare: Option<String>,
    fail_commit: Option<String>,
    fail_abort: Option<String>,
}

impl InMemoryBlockBackend {
    pub fn new() -> Self {
        Self {
            next_object_id: 1,
            ..Self::default()
        }
    }

    pub fn fail_next_prepare(&mut self, error: impl Into<String>) {
        self.fail_prepare = Some(error.into());
    }

    pub fn fail_next_commit(&mut self, error: impl Into<String>) {
        self.fail_commit = Some(error.into());
    }

    pub fn fail_next_abort(&mut self, error: impl Into<String>) {
        self.fail_abort = Some(error.into());
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    pub fn staged_count(&self) -> usize {
        self.staged.len()
    }
}

impl LogicalBlockPhysicalBackend for InMemoryBlockBackend {
    fn prepare(
        &mut self,
        transaction: BlockTransactionId,
        _kind: BlockTransactionKind,
        creates: &[PhysicalBlockCreate],
        releases: &[PhysicalObjectRef],
    ) -> Result<Vec<PreparedPhysicalBlock>, String> {
        if let Some(error) = self.fail_prepare.take() {
            return Err(error);
        }
        if transaction == 0 || self.staged.contains_key(&transaction) {
            return Err("invalid or duplicate transaction".to_owned());
        }
        let mut release_keys = BTreeSet::new();
        for release in releases {
            let key = (release.id, release.generation);
            if !self.objects.contains_key(&key) || !release_keys.insert(key) {
                return Err("release references an absent physical block".to_owned());
            }
        }
        let prepared = creates
            .iter()
            .map(|create| {
                let physical = PhysicalObjectRef {
                    id: self.next_object_id,
                    generation: 1,
                };
                self.next_object_id = self.next_object_id.saturating_add(1);
                PreparedPhysicalBlock {
                    block: create.block,
                    physical,
                    physical_bytes: create.physical_bytes,
                }
            })
            .collect::<Vec<_>>();
        self.staged.insert(
            transaction,
            StagedPhysicalTransaction {
                creates: prepared.clone(),
                releases: releases.to_vec(),
            },
        );
        Ok(prepared)
    }

    fn commit(&mut self, transaction: BlockTransactionId) -> Result<(), String> {
        if let Some(error) = self.fail_commit.take() {
            return Err(error);
        }
        let staged = self
            .staged
            .remove(&transaction)
            .ok_or_else(|| "transaction was not prepared".to_owned())?;
        for release in staged.releases {
            self.objects.remove(&(release.id, release.generation));
        }
        for create in staged.creates {
            self.objects.insert(
                (create.physical.id, create.physical.generation),
                create.physical_bytes,
            );
        }
        Ok(())
    }

    fn abort(&mut self, transaction: BlockTransactionId) -> Result<(), String> {
        if let Some(error) = self.fail_abort.take() {
            return Err(error);
        }
        self.staged.remove(&transaction);
        Ok(())
    }

    fn inspect(&self) -> Result<Vec<PhysicalBlockSnapshot>, String> {
        Ok(self
            .objects
            .iter()
            .map(|((id, generation), physical_bytes)| PhysicalBlockSnapshot {
                physical: PhysicalObjectRef {
                    id: *id,
                    generation: *generation,
                },
                physical_bytes: *physical_bytes,
            })
            .collect())
    }
}
