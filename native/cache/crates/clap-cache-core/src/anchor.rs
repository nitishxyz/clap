use std::collections::BTreeMap;

use super::slot::{anchor_eviction_value, Slot};
use super::{CacheManager, PlanRequest, SlotCapabilities, SlotState};

impl CacheManager {
    pub(super) fn session_budget_victims(
        &self,
        target: u32,
        donor: Option<u32>,
        request: &PlanRequest<'_>,
        projected_target_bytes: u64,
    ) -> Result<Vec<u32>, super::Error> {
        let limit = self.config.max_anchor_bytes_per_session;
        if request.labels.session == 0 || limit == 0 || projected_target_bytes == 0 {
            return Ok(Vec::new());
        }
        let target_current = self.slots[target as usize].accounted_bytes;
        let mut projected = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor && self.same_limited_session(slot, request)
            })
            .map(|slot| slot.accounted_bytes)
            .sum::<u64>()
            .saturating_sub(target_current)
            .saturating_add(projected_target_bytes);
        if projected <= limit {
            return Ok(Vec::new());
        }
        let mut candidates = self
            .slots
            .iter()
            .filter(|slot| {
                slot.id != target
                    && Some(slot.id) != donor
                    && slot.state == SlotState::Anchor
                    && self.same_limited_session(slot, request)
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|slot| (anchor_eviction_value(slot), slot.id));
        let mut victims = Vec::new();
        for slot in candidates {
            if projected <= limit {
                break;
            }
            projected = projected.saturating_sub(slot.accounted_bytes);
            victims.push(slot.id);
        }
        if projected > limit {
            return Err(super::Error::NoCapacity);
        }
        Ok(victims)
    }

    pub(super) fn choose_anchor_boundaries(
        &mut self,
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
        let publication_candidates = semantic.len().saturating_add(automatic.len());
        let existing_anchors = self
            .slots
            .iter()
            .filter(|slot| slot.state == SlotState::Anchor)
            .count();
        let existing_session_anchors = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor && self.same_limited_session(slot, request)
            })
            .count();
        let global_available = (self.config.max_anchors as usize).saturating_sub(existing_anchors);
        let session_available = self
            .session_anchor_limit(request)
            .saturating_sub(existing_session_anchors);
        let available = global_available.min(session_available);
        let replaceable_automatic = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && slot.namespace == request.namespace
                    && slot.labels.session == request.labels.session
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
                    && !slot.busy
                    && slot.writer.is_none()
                    && slot.read_leases == 0
                    && !slot.protected
            })
            .count();
        let overflow = semantic.split_off(semantic.len().min(available));
        let mut replacement_slots = Vec::new();
        for boundary in overflow.into_iter().rev() {
            let replacement = self
                .slots
                .iter()
                .filter(|slot| {
                    slot.state == SlotState::Anchor
                        && slot.namespace == request.namespace
                        && slot.labels.session == request.labels.session
                        && slot.tokens.len() < boundary
                        && slot.tokens == request.tokens[..slot.tokens.len()]
                        && !slot.busy
                        && slot.writer.is_none()
                        && slot.read_leases == 0
                        && !slot.protected
                        && !replacement_slots.contains(&slot.id)
                        && self
                            .slot_capabilities(request, slot.id)
                            .contains(SlotCapabilities::WRITABLE)
                })
                .min_by_key(|slot| (anchor_eviction_value(slot), slot.id));
            if let Some(slot) = replacement {
                replacement_slots.push(slot.id);
                semantic.push(boundary);
            }
        }
        let replaced_automatic = replacement_slots
            .iter()
            .filter(|&&slot| {
                self.is_automatic_checkpoint_len(self.slots[slot as usize].tokens.len())
            })
            .count();
        let semantic_empty_slots = semantic.len().saturating_sub(replacement_slots.len());
        let replacement_automatic_remaining =
            replaceable_automatic.saturating_sub(replaced_automatic);
        let automatic_global_available = global_available
            .saturating_sub(semantic_empty_slots)
            .saturating_add(replacement_automatic_remaining);
        let automatic_session_available = session_available
            .saturating_sub(semantic_empty_slots)
            .saturating_add(replacement_automatic_remaining);
        let existing_session_automatic = self
            .slots
            .iter()
            .filter(|slot| {
                slot.state == SlotState::Anchor
                    && self.same_limited_session(slot, request)
                    && self.is_automatic_checkpoint_len(slot.tokens.len())
                    && !replacement_slots.contains(&slot.id)
            })
            .count();
        let automatic_per_session_available = self
            .session_checkpoint_limit(request)
            .saturating_sub(existing_session_automatic);
        let automatic_available = automatic_global_available
            .min(automatic_session_available)
            .min(automatic_per_session_available);
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
        self.telemetry.anchor_publication_skips = self
            .telemetry
            .anchor_publication_skips
            .saturating_add(publication_candidates.saturating_sub(authorized.len()) as u64);
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

    pub(super) fn checkpoint_depth_band(&self, tokens: usize) -> u32 {
        let interval = self
            .config
            .automatic_checkpoints
            .target_interval_tokens
            .max(1);
        (tokens / interval).max(1).ilog2()
    }

    pub(super) fn is_automatic_checkpoint_len(&self, tokens: usize) -> bool {
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

    pub(super) fn same_limited_session(&self, slot: &Slot, request: &PlanRequest<'_>) -> bool {
        request.labels.session != 0
            && slot.namespace == request.namespace
            && slot.labels.session == request.labels.session
    }

    pub(super) fn session_anchor_limit(&self, request: &PlanRequest<'_>) -> usize {
        if request.labels.session == 0 || self.config.max_anchors_per_session == 0 {
            usize::MAX
        } else {
            self.config.max_anchors_per_session as usize
        }
    }

    fn session_checkpoint_limit(&self, request: &PlanRequest<'_>) -> usize {
        let limit = self
            .config
            .automatic_checkpoints
            .max_checkpoints_per_session;
        if request.labels.session == 0 || limit == 0 {
            usize::MAX
        } else {
            limit as usize
        }
    }

    pub(super) fn automatic_checkpoint_byte_limit(&self) -> u64 {
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
}
