use super::{CacheManager, Error, Generation, Namespace, SlotId};

impl CacheManager {
    pub fn touch(
        &mut self,
        slot_id: SlotId,
        generation: Generation,
        now_ms: u64,
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
        if slot.is_empty() {
            return Ok(());
        }
        self.clock = self.clock.saturating_add(1);
        slot.last_used = self.clock;
        slot.last_used_ms = now_ms;
        Ok(())
    }

    pub fn expire_idle(&mut self, now_ms: u64) -> Vec<SlotId> {
        let ttl = self.config.session_idle_ttl_ms;
        if ttl == 0 {
            return Vec::new();
        }
        let victims = self
            .slots
            .iter()
            .filter(|slot| {
                !slot.is_empty()
                    && slot.labels.session != 0
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
                    && now_ms.saturating_sub(slot.last_used_ms) >= ttl
            })
            .map(|slot| slot.id)
            .collect::<Vec<_>>();
        let bytes = victims
            .iter()
            .map(|&slot| self.slots[slot as usize].accounted_bytes)
            .sum::<u64>();
        for &slot in &victims {
            self.invalidate_slot(slot);
        }
        self.telemetry.expired_slots = self
            .telemetry
            .expired_slots
            .saturating_add(victims.len() as u64);
        self.telemetry.expired_accounted_bytes =
            self.telemetry.expired_accounted_bytes.saturating_add(bytes);
        self.refresh_gauges();
        victims
    }

    pub fn release_session(
        &mut self,
        namespace: Namespace,
        session: u64,
    ) -> Result<Vec<SlotId>, Error> {
        if session == 0 {
            return Err(Error::InvalidArgument);
        }
        let victims = self
            .slots
            .iter()
            .filter(|slot| {
                !slot.is_empty()
                    && slot.namespace == namespace
                    && slot.labels.session == session
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
            })
            .map(|slot| slot.id)
            .collect::<Vec<_>>();
        let bytes = victims
            .iter()
            .map(|&slot| self.slots[slot as usize].accounted_bytes)
            .sum::<u64>();
        for &slot in &victims {
            self.invalidate_slot(slot);
        }
        self.telemetry.released_session_slots = self
            .telemetry
            .released_session_slots
            .saturating_add(victims.len() as u64);
        self.telemetry.released_session_accounted_bytes = self
            .telemetry
            .released_session_accounted_bytes
            .saturating_add(bytes);
        self.refresh_gauges();
        Ok(victims)
    }
}
