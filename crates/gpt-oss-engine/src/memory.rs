//! Checked logical memory reservations for the bounded CPU service.
//!
//! These values describe ownership promises. They are deliberately not RSS,
//! allocator, mapping, or disk-cache measurements.

use gpt_oss_core::prelude::RequestId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClass {
    Request,
    KvCache,
    StagedKv,
    TokenVectors,
    GenerationState,
    AcceleratorStaging,
    Delivery,
    ResponseStore,
    Diagnostics,
}

impl MemoryClass {
    pub const ALL: [Self; 9] = [
        Self::Request,
        Self::KvCache,
        Self::StagedKv,
        Self::TokenVectors,
        Self::GenerationState,
        Self::AcceleratorStaging,
        Self::Delivery,
        Self::ResponseStore,
        Self::Diagnostics,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::KvCache => "kv_cache",
            Self::StagedKv => "staged_kv",
            Self::TokenVectors => "token_vectors",
            Self::GenerationState => "generation_state",
            Self::AcceleratorStaging => "accelerator_staging",
            Self::Delivery => "delivery",
            Self::ResponseStore => "response_store",
            Self::Diagnostics => "diagnostics",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryEstimate {
    #[serde(default)]
    pub by_class: BTreeMap<MemoryClass, u128>,
}

impl MemoryEstimate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, class: MemoryClass, bytes: u128) -> Result<Self, GrantFailure> {
        self.checked_add(class, bytes)?;
        Ok(self)
    }

    pub fn get(&self, class: MemoryClass) -> u128 {
        self.by_class.get(&class).copied().unwrap_or(0)
    }

    pub fn checked_add(&mut self, class: MemoryClass, bytes: u128) -> Result<(), GrantFailure> {
        let current = self.get(class);
        let next = current
            .checked_add(bytes)
            .ok_or(GrantFailure::EstimateOverflow)?;
        if next == 0 {
            self.by_class.remove(&class);
        } else {
            self.by_class.insert(class, next);
        }
        Ok(())
    }

    pub fn total(&self) -> Result<u128, GrantFailure> {
        self.by_class.values().try_fold(0u128, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or(GrantFailure::EstimateOverflow)
        })
    }

    pub fn checked_merge(&self, other: &Self) -> Result<Self, GrantFailure> {
        let mut merged = self.clone();
        for (&class, &bytes) in &other.by_class {
            merged.checked_add(class, bytes)?;
        }
        Ok(merged)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GrantId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantPhase {
    Granted,
    Active,
    Persistent,
    Released,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryGrant {
    pub id: GrantId,
    pub request_id: RequestId,
    pub granted: MemoryEstimate,
    pub used_estimate: MemoryEstimate,
    pub phase: GrantPhase,
}

impl MemoryGrant {
    pub fn total_granted(&self) -> Result<u128, GrantFailure> {
        self.granted.total()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantFailure {
    #[error("memory estimate overflow")]
    EstimateOverflow,
    #[error("request already has a grant")]
    DuplicateRequest,
    #[error("grant ID space exhausted")]
    DuplicateGrantId,
    #[error("unknown grant")]
    UnknownGrant,
    #[error("per-request logical memory limit exceeded")]
    PerRequestLimit,
    #[error("maximum number of active request grants reached")]
    GlobalRequestLimit,
    #[error("global logical memory limit exceeded")]
    GlobalMemoryLimit,
    #[error("delivery memory limit exceeded")]
    DeliveryLimit,
    #[error("response store memory limit exceeded")]
    StoreLimit,
    #[error("memory expansion denied")]
    ExpansionDenied,
    #[error("refund exceeds owned bytes")]
    RefundUnderflow,
    #[error("grant is already released")]
    ReleasedGrant,
    #[error("reservation manager is unavailable")]
    ManagerUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservationLimits {
    pub max_active_requests: usize,
    pub per_request_bytes: u128,
    pub global_bytes: u128,
    #[serde(default)]
    pub by_class: BTreeMap<MemoryClass, u128>,
}

impl ReservationLimits {
    pub fn bounded(
        max_active_requests: usize,
        per_request_bytes: u128,
        global_bytes: u128,
    ) -> Self {
        Self {
            max_active_requests,
            per_request_bytes,
            global_bytes,
            by_class: BTreeMap::new(),
        }
    }

    pub fn with_class_limit(mut self, class: MemoryClass, bytes: u128) -> Self {
        self.by_class.insert(class, bytes);
        self
    }
}

impl Default for ReservationLimits {
    fn default() -> Self {
        Self::bounded(1, u128::MAX, u128::MAX)
    }
}

#[derive(Debug)]
pub struct ReservationLedger {
    limits: ReservationLimits,
    grants: BTreeMap<GrantId, MemoryGrant>,
    request_grants: HashMap<RequestId, GrantId>,
    reserved_by_class: BTreeMap<MemoryClass, u128>,
    total_reserved: u128,
    next_id: u64,
    available: bool,
}

impl ReservationLedger {
    pub fn new(limits: ReservationLimits) -> Result<Self, GrantFailure> {
        if limits.max_active_requests == 0 {
            return Err(GrantFailure::GlobalRequestLimit);
        }
        Ok(Self {
            limits,
            grants: BTreeMap::new(),
            request_grants: HashMap::new(),
            reserved_by_class: BTreeMap::new(),
            total_reserved: 0,
            next_id: 1,
            available: true,
        })
    }

    pub const fn limits(&self) -> &ReservationLimits {
        &self.limits
    }

    pub fn set_available(&mut self, available: bool) {
        self.available = available;
    }

    pub fn active_grants(&self) -> usize {
        self.grants
            .values()
            .filter(|grant| grant.phase != GrantPhase::Released)
            .count()
    }

    pub const fn total_reserved(&self) -> u128 {
        self.total_reserved
    }

    pub fn reserved(&self, class: MemoryClass) -> u128 {
        self.reserved_by_class.get(&class).copied().unwrap_or(0)
    }

    pub fn grant(
        &mut self,
        request_id: RequestId,
        estimate: MemoryEstimate,
    ) -> Result<MemoryGrant, GrantFailure> {
        if !self.available {
            return Err(GrantFailure::ManagerUnavailable);
        }
        if self.request_grants.contains_key(&request_id) {
            return Err(GrantFailure::DuplicateRequest);
        }
        if self.active_grants() >= self.limits.max_active_requests {
            return Err(GrantFailure::GlobalRequestLimit);
        }
        self.validate_delta(None, &estimate, false)?;
        let id = GrantId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(GrantFailure::DuplicateGrantId)?;
        if self.grants.contains_key(&id) {
            return Err(GrantFailure::DuplicateGrantId);
        }

        self.apply_add(&estimate)?;
        let grant = MemoryGrant {
            id,
            request_id,
            granted: estimate,
            used_estimate: MemoryEstimate::default(),
            phase: GrantPhase::Granted,
        };
        self.grants.insert(id, grant.clone());
        self.request_grants.insert(request_id, id);
        debug_assert!(self.invariants_hold());
        Ok(grant)
    }

    pub fn activate(&mut self, id: GrantId) -> Result<(), GrantFailure> {
        let grant = self.grants.get_mut(&id).ok_or(GrantFailure::UnknownGrant)?;
        match grant.phase {
            GrantPhase::Granted => grant.phase = GrantPhase::Active,
            GrantPhase::Active | GrantPhase::Persistent => {}
            GrantPhase::Released => return Err(GrantFailure::ReleasedGrant),
        }
        Ok(())
    }

    pub fn expand(&mut self, id: GrantId, delta: MemoryEstimate) -> Result<(), GrantFailure> {
        let current = self.grants.get(&id).ok_or(GrantFailure::UnknownGrant)?;
        if current.phase == GrantPhase::Released {
            return Err(GrantFailure::ReleasedGrant);
        }
        self.validate_delta(Some(current), &delta, true)?;
        let expanded = current.granted.checked_merge(&delta)?;
        self.apply_add(&delta)?;
        self.grants
            .get_mut(&id)
            .expect("grant remained present")
            .granted = expanded;
        debug_assert!(self.invariants_hold());
        Ok(())
    }

    /// Mark named granted bytes as currently owned. This cannot exceed a grant.
    pub fn consume(
        &mut self,
        id: GrantId,
        class: MemoryClass,
        bytes: u128,
    ) -> Result<(), GrantFailure> {
        let grant = self.grants.get_mut(&id).ok_or(GrantFailure::UnknownGrant)?;
        if grant.phase == GrantPhase::Released {
            return Err(GrantFailure::ReleasedGrant);
        }
        let next = grant
            .used_estimate
            .get(class)
            .checked_add(bytes)
            .ok_or(GrantFailure::EstimateOverflow)?;
        if next > grant.granted.get(class) {
            return Err(GrantFailure::ExpansionDenied);
        }
        grant.used_estimate.by_class.insert(class, next);
        Ok(())
    }

    /// Reduce both the logical promise and current ownership for one class.
    pub fn refund(
        &mut self,
        id: GrantId,
        class: MemoryClass,
        bytes: u128,
    ) -> Result<(), GrantFailure> {
        let grant = self.grants.get(&id).ok_or(GrantFailure::UnknownGrant)?;
        if grant.phase == GrantPhase::Released {
            return Err(GrantFailure::ReleasedGrant);
        }
        let granted = grant.granted.get(class);
        if bytes > granted {
            return Err(GrantFailure::RefundUnderflow);
        }
        let used = grant.used_estimate.get(class);
        let new_used = used.saturating_sub(bytes);
        self.apply_sub(class, bytes)?;
        let grant = self.grants.get_mut(&id).expect("grant remained present");
        set_class(&mut grant.granted, class, granted - bytes);
        set_class(&mut grant.used_estimate, class, new_used);
        debug_assert!(self.invariants_hold());
        Ok(())
    }

    /// Atomically move a promise between classes without changing grant total.
    pub fn transfer(
        &mut self,
        id: GrantId,
        from: MemoryClass,
        to: MemoryClass,
        bytes: u128,
    ) -> Result<(), GrantFailure> {
        if from == to || bytes == 0 {
            return Ok(());
        }
        let grant = self.grants.get(&id).ok_or(GrantFailure::UnknownGrant)?;
        if grant.phase == GrantPhase::Released {
            return Err(GrantFailure::ReleasedGrant);
        }
        if grant.granted.get(from) < bytes {
            return Err(GrantFailure::RefundUnderflow);
        }
        let mut delta = MemoryEstimate::new();
        delta.checked_add(to, bytes)?;
        self.validate_class_delta(to, bytes, true)?;
        let from_granted = grant.granted.get(from);
        let to_granted = grant
            .granted
            .get(to)
            .checked_add(bytes)
            .ok_or(GrantFailure::EstimateOverflow)?;
        let moved_used = grant.used_estimate.get(from).min(bytes);
        let from_used = grant.used_estimate.get(from) - moved_used;
        let to_used = grant
            .used_estimate
            .get(to)
            .checked_add(moved_used)
            .ok_or(GrantFailure::EstimateOverflow)?;

        self.apply_sub(from, bytes)?;
        self.apply_class_add(to, bytes)?;
        let grant = self.grants.get_mut(&id).expect("grant remained present");
        set_class(&mut grant.granted, from, from_granted - bytes);
        set_class(&mut grant.granted, to, to_granted);
        set_class(&mut grant.used_estimate, from, from_used);
        set_class(&mut grant.used_estimate, to, to_used);
        debug_assert!(self.invariants_hold());
        Ok(())
    }

    pub fn transfer_to_persistent_store(&mut self, id: GrantId) -> Result<(), GrantFailure> {
        let grant = self.grants.get_mut(&id).ok_or(GrantFailure::UnknownGrant)?;
        if grant.phase == GrantPhase::Released {
            return Err(GrantFailure::ReleasedGrant);
        }
        grant.phase = GrantPhase::Persistent;
        Ok(())
    }

    /// Release is idempotent. The return value says whether ownership changed.
    pub fn release(&mut self, id: GrantId) -> Result<bool, GrantFailure> {
        let Some(grant) = self.grants.get(&id) else {
            return Ok(false);
        };
        if grant.phase == GrantPhase::Released {
            return Ok(false);
        }
        let estimate = grant.granted.clone();
        let request_id = grant.request_id;
        for (&class, &bytes) in &estimate.by_class {
            self.apply_sub(class, bytes)?;
        }
        let grant = self.grants.get_mut(&id).expect("grant remained present");
        grant.phase = GrantPhase::Released;
        grant.granted = MemoryEstimate::default();
        grant.used_estimate = MemoryEstimate::default();
        self.request_grants.remove(&request_id);
        debug_assert!(self.invariants_hold());
        Ok(true)
    }

    pub fn release_all(&mut self) -> Result<usize, GrantFailure> {
        let ids = self
            .grants
            .iter()
            .filter_map(|(&id, grant)| (grant.phase != GrantPhase::Released).then_some(id))
            .collect::<Vec<_>>();
        for id in &ids {
            self.release(*id)?;
        }
        Ok(ids.len())
    }

    pub fn grant_snapshot(&self, id: GrantId) -> Option<MemoryGrant> {
        self.grants.get(&id).cloned()
    }

    pub fn request_grant(&self, request_id: RequestId) -> Option<GrantId> {
        self.request_grants.get(&request_id).copied()
    }

    pub fn invariants_hold(&self) -> bool {
        let mut classes = BTreeMap::<MemoryClass, u128>::new();
        let mut total = 0u128;
        for grant in self
            .grants
            .values()
            .filter(|grant| grant.phase != GrantPhase::Released)
        {
            for class in MemoryClass::ALL {
                if grant.used_estimate.get(class) > grant.granted.get(class) {
                    return false;
                }
            }
            for (&class, &bytes) in &grant.granted.by_class {
                let Some(next) = classes.get(&class).copied().unwrap_or(0).checked_add(bytes)
                else {
                    return false;
                };
                classes.insert(class, next);
                let Some(next_total) = total.checked_add(bytes) else {
                    return false;
                };
                total = next_total;
            }
        }
        classes == self.reserved_by_class && total == self.total_reserved
    }

    fn validate_delta(
        &self,
        existing: Option<&MemoryGrant>,
        delta: &MemoryEstimate,
        expansion: bool,
    ) -> Result<(), GrantFailure> {
        let delta_total = delta.total()?;
        let existing_total = existing.map_or(Ok(0), |grant| grant.granted.total())?;
        if existing_total
            .checked_add(delta_total)
            .ok_or(GrantFailure::EstimateOverflow)?
            > self.limits.per_request_bytes
        {
            return Err(if expansion {
                GrantFailure::ExpansionDenied
            } else {
                GrantFailure::PerRequestLimit
            });
        }
        if self
            .total_reserved
            .checked_add(delta_total)
            .ok_or(GrantFailure::EstimateOverflow)?
            > self.limits.global_bytes
        {
            return Err(if expansion {
                GrantFailure::ExpansionDenied
            } else {
                GrantFailure::GlobalMemoryLimit
            });
        }
        for (&class, &bytes) in &delta.by_class {
            self.validate_class_delta(class, bytes, expansion)?;
        }
        Ok(())
    }

    fn validate_class_delta(
        &self,
        class: MemoryClass,
        bytes: u128,
        expansion: bool,
    ) -> Result<(), GrantFailure> {
        if let Some(&limit) = self.limits.by_class.get(&class) {
            let next = self
                .reserved(class)
                .checked_add(bytes)
                .ok_or(GrantFailure::EstimateOverflow)?;
            if next > limit {
                return Err(if expansion {
                    GrantFailure::ExpansionDenied
                } else {
                    match class {
                        MemoryClass::Delivery => GrantFailure::DeliveryLimit,
                        MemoryClass::ResponseStore => GrantFailure::StoreLimit,
                        _ => GrantFailure::GlobalMemoryLimit,
                    }
                });
            }
        }
        Ok(())
    }

    fn apply_add(&mut self, estimate: &MemoryEstimate) -> Result<(), GrantFailure> {
        for (&class, &bytes) in &estimate.by_class {
            self.apply_class_add(class, bytes)?;
        }
        Ok(())
    }

    fn apply_class_add(&mut self, class: MemoryClass, bytes: u128) -> Result<(), GrantFailure> {
        let current = self.reserved(class);
        self.reserved_by_class.insert(
            class,
            current
                .checked_add(bytes)
                .ok_or(GrantFailure::EstimateOverflow)?,
        );
        self.total_reserved = self
            .total_reserved
            .checked_add(bytes)
            .ok_or(GrantFailure::EstimateOverflow)?;
        Ok(())
    }

    fn apply_sub(&mut self, class: MemoryClass, bytes: u128) -> Result<(), GrantFailure> {
        let current = self.reserved(class);
        let next = current
            .checked_sub(bytes)
            .ok_or(GrantFailure::RefundUnderflow)?;
        if next == 0 {
            self.reserved_by_class.remove(&class);
        } else {
            self.reserved_by_class.insert(class, next);
        }
        self.total_reserved = self
            .total_reserved
            .checked_sub(bytes)
            .ok_or(GrantFailure::RefundUnderflow)?;
        Ok(())
    }
}

fn set_class(estimate: &mut MemoryEstimate, class: MemoryClass, value: u128) {
    if value == 0 {
        estimate.by_class.remove(&class);
    } else {
        estimate.by_class.insert(class, value);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuKvGeometry {
    pub scalar_bytes: u128,
    pub kv_heads: u128,
    pub head_dim: u128,
    pub full_layers: u128,
    pub sliding_layers: u128,
    pub sliding_window: u128,
}

impl CpuKvGeometry {
    pub fn logical_bytes(&self, context: u128) -> Result<u128, GrantFailure> {
        let full = checked_product(&[self.full_layers, context])?;
        let sliding = checked_product(&[self.sliding_layers, context.min(self.sliding_window)])?;
        checked_product(&[2, self.scalar_bytes, self.kv_heads, self.head_dim])?
            .checked_mul(
                full.checked_add(sliding)
                    .ok_or(GrantFailure::EstimateOverflow)?,
            )
            .ok_or(GrantFailure::EstimateOverflow)
    }

    pub fn staged_bytes(&self, rows: u128) -> Result<u128, GrantFailure> {
        let layers = self
            .full_layers
            .checked_add(self.sliding_layers)
            .ok_or(GrantFailure::EstimateOverflow)?;
        checked_product(&[
            rows,
            2,
            self.scalar_bytes,
            self.kv_heads,
            self.head_dim,
            layers,
        ])
    }
}

fn checked_product(values: &[u128]) -> Result<u128, GrantFailure> {
    values.iter().try_fold(1u128, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(GrantFailure::EstimateOverflow)
    })
}

/// Named fields from Linux `smaps_rollup`; all values are sampled facts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SmapsRollup {
    pub source: PathBuf,
    pub fields_bytes: BTreeMap<String, u128>,
}

impl SmapsRollup {
    #[cfg(target_os = "linux")]
    pub fn sample_self() -> std::io::Result<Self> {
        Self::from_path("/proc/self/smaps_rollup")
    }

    #[cfg(not(target_os = "linux"))]
    pub fn sample_self() -> std::io::Result<Self> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "smaps_rollup is Linux-specific",
        ))
    }

    pub fn from_path(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let source = path.into();
        let text = std::fs::read_to_string(&source)?;
        let mut fields_bytes = BTreeMap::new();
        for line in text.lines().skip(1) {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let mut parts = value.split_whitespace();
            let Some(number) = parts.next().and_then(|value| value.parse::<u128>().ok()) else {
                continue;
            };
            let multiplier = match parts.next() {
                Some("kB") => 1024,
                None => 1,
                Some(_) => continue,
            };
            if let Some(bytes) = number.checked_mul(multiplier) {
                fields_bytes.insert(name.to_string(), bytes);
            }
        }
        Ok(Self {
            source,
            fields_bytes,
        })
    }

    pub fn bytes(&self, field: &str) -> Option<u128> {
        self.fields_bytes.get(field).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(class: MemoryClass, bytes: u128) -> MemoryEstimate {
        MemoryEstimate::new().with(class, bytes).unwrap()
    }

    #[test]
    fn denial_is_atomic_and_class_specific() {
        let limits =
            ReservationLimits::bounded(2, 100, 150).with_class_limit(MemoryClass::Delivery, 10);
        let mut ledger = ReservationLedger::new(limits).unwrap();
        assert_eq!(
            ledger.grant(RequestId(1), estimate(MemoryClass::Delivery, 11)),
            Err(GrantFailure::DeliveryLimit)
        );
        assert_eq!(ledger.total_reserved(), 0);
        assert_eq!(ledger.active_grants(), 0);
    }

    #[test]
    fn duplicate_request_expand_refund_transfer_and_double_release() {
        let limits = ReservationLimits::bounded(2, 200, 300)
            .with_class_limit(MemoryClass::Delivery, 100)
            .with_class_limit(MemoryClass::ResponseStore, 100);
        let mut ledger = ReservationLedger::new(limits).unwrap();
        let grant = ledger
            .grant(RequestId(1), estimate(MemoryClass::Delivery, 50))
            .unwrap();
        assert_eq!(
            ledger.grant(RequestId(1), estimate(MemoryClass::Delivery, 1)),
            Err(GrantFailure::DuplicateRequest)
        );
        ledger
            .expand(grant.id, estimate(MemoryClass::Delivery, 10))
            .unwrap();
        ledger.consume(grant.id, MemoryClass::Delivery, 40).unwrap();
        ledger
            .transfer(
                grant.id,
                MemoryClass::Delivery,
                MemoryClass::ResponseStore,
                20,
            )
            .unwrap();
        assert_eq!(ledger.reserved(MemoryClass::Delivery), 40);
        assert_eq!(ledger.reserved(MemoryClass::ResponseStore), 20);
        ledger.refund(grant.id, MemoryClass::Delivery, 10).unwrap();
        assert_eq!(ledger.total_reserved(), 50);
        assert!(ledger.release(grant.id).unwrap());
        assert!(!ledger.release(grant.id).unwrap());
        assert_eq!(ledger.total_reserved(), 0);
        assert!(ledger.invariants_hold());
    }

    #[test]
    fn every_formula_detects_overflow_and_sliding_kv_plateaus() {
        let overflow = CpuKvGeometry {
            scalar_bytes: u128::MAX,
            kv_heads: 2,
            head_dim: 1,
            full_layers: 1,
            sliding_layers: 1,
            sliding_window: 1,
        };
        assert_eq!(
            overflow.logical_bytes(1),
            Err(GrantFailure::EstimateOverflow)
        );
        assert_eq!(
            overflow.staged_bytes(1),
            Err(GrantFailure::EstimateOverflow)
        );

        let geometry = CpuKvGeometry {
            scalar_bytes: 2,
            kv_heads: 8,
            head_dim: 64,
            full_layers: 0,
            sliding_layers: 12,
            sliding_window: 128,
        };
        assert_eq!(
            geometry.logical_bytes(128).unwrap(),
            geometry.logical_bytes(8192).unwrap()
        );
        assert_eq!(geometry.staged_bytes(1).unwrap(), 24_576);
    }

    #[test]
    fn full_layers_continue_growing() {
        let geometry = CpuKvGeometry {
            scalar_bytes: 2,
            kv_heads: 8,
            head_dim: 64,
            full_layers: 12,
            sliding_layers: 12,
            sliding_window: 128,
        };
        assert!(geometry.logical_bytes(8192).unwrap() > geometry.logical_bytes(128).unwrap());
        assert_eq!(geometry.logical_bytes(8192).unwrap(), 204_472_320);
        assert_eq!(geometry.staged_bytes(2048).unwrap(), 100_663_296);
    }

    #[test]
    fn parses_named_smaps_fields_without_collapsing_dimensions() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            temp.path(),
            "0000-ffff ---p 00000000 00:00 0 [rollup]\nRss: 10 kB\nPss: 4 kB\nAnonymous: 3 kB\n",
        )
        .unwrap();
        let sample = SmapsRollup::from_path(temp.path().to_path_buf()).unwrap();
        assert_eq!(sample.bytes("Rss"), Some(10 * 1024));
        assert_eq!(sample.bytes("Pss"), Some(4 * 1024));
        assert_eq!(sample.bytes("Anonymous"), Some(3 * 1024));
    }
}
