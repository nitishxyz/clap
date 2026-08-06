use clap_cache_core::adapter::{
    llama_sequence_descriptor, mlx_sequence_descriptor, paged_engine_skeleton_descriptor,
    AdapterDescriptor, AdapterError, AdapterKind, AdapterOperation, AdapterOperations,
    AdapterOutcome, AdapterRequest, ByteAccounting, CacheTier, CheckedPhysicalCacheAdapter,
    ObservedBytes, PhysicalCacheBackend, PhysicalFormatIdentity, PhysicalObjectRef,
    PhysicalObjectSnapshot, PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
};

const MODEL_DOMAIN: [u8; 32] = [7; 32];
const TARGET: PhysicalObjectRef = PhysicalObjectRef {
    id: 11,
    generation: 3,
};
const DONOR: PhysicalObjectRef = PhysicalObjectRef {
    id: 8,
    generation: 2,
};

#[derive(Clone)]
struct FakeBackend {
    descriptor: AdapterDescriptor,
    snapshots: Vec<PhysicalObjectSnapshot>,
    execute_calls: usize,
    execute_error: Option<String>,
    outcome_target: Option<PhysicalObjectRef>,
    invalidation_error: Option<String>,
    invalidated: Vec<PhysicalObjectRef>,
}

impl FakeBackend {
    fn new(descriptor: AdapterDescriptor) -> Self {
        Self {
            descriptor,
            snapshots: Vec::new(),
            execute_calls: 0,
            execute_error: None,
            outcome_target: None,
            invalidation_error: None,
            invalidated: Vec::new(),
        }
    }
}

impl PhysicalCacheBackend for FakeBackend {
    fn descriptor(&self) -> &AdapterDescriptor {
        &self.descriptor
    }

    fn inspect(&self) -> Result<Vec<PhysicalObjectSnapshot>, String> {
        Ok(self.snapshots.clone())
    }

    fn execute(&mut self, request: &AdapterRequest) -> Result<AdapterOutcome, String> {
        self.execute_calls += 1;
        if let Some(error) = &self.execute_error {
            return Err(error.clone());
        }
        Ok(AdapterOutcome {
            target: self.outcome_target.unwrap_or(request.target),
            resident_tokens: request.token_count.unwrap_or(64),
            tier: CacheTier::Device,
            bytes: match self.descriptor.constraints.byte_accounting {
                ByteAccounting::Unknown => ObservedBytes::Unknown(
                    clap_cache_core::adapter::UnknownBytesReason::BackendDoesNotReport,
                ),
                ByteAccounting::Estimated | ByteAccounting::Exact => ObservedBytes::Known(4096),
            },
        })
    }

    fn invalidate(&mut self, target: PhysicalObjectRef) -> Result<(), String> {
        self.invalidated.push(target);
        if let Some(error) = &self.invalidation_error {
            return Err(error.clone());
        }
        Ok(())
    }
}

fn request(operation: AdapterOperation) -> AdapterRequest {
    AdapterRequest::new(operation, TARGET)
}

#[test]
fn sequence_descriptors_are_honest_about_current_backend_constraints() {
    let llama = llama_sequence_descriptor(MODEL_DOMAIN, false, true, Some("f16"));
    llama.validate().unwrap();
    assert_eq!(llama.kind, AdapterKind::Sequence);
    assert!(llama.operations.contains(AdapterOperation::Continue));
    assert!(llama.operations.contains(AdapterOperation::Restore));
    assert!(llama.operations.contains(AdapterOperation::Fork));
    assert!(llama.operations.contains(AdapterOperation::Trim));
    assert!(llama.operations.contains(AdapterOperation::Snapshot));
    assert!(!llama.operations.contains(AdapterOperation::Export));
    assert_eq!(
        llama.constraints.fork_semantics,
        clap_cache_core::adapter::ForkSemantics::CopyOnWrite
    );
    assert_eq!(llama.constraints.byte_accounting, ByteAccounting::Unknown);
    assert_eq!(llama.format.cache_format, "llama-sequence");

    let hybrid = llama_sequence_descriptor(MODEL_DOMAIN, true, false, None);
    hybrid.validate().unwrap();
    assert!(!hybrid.operations.contains(AdapterOperation::Trim));
    assert!(!hybrid.operations.contains(AdapterOperation::Snapshot));
    assert!(hybrid.constraints.recurrent_or_hybrid);
    assert_eq!(
        hybrid.constraints.fork_semantics,
        clap_cache_core::adapter::ForkSemantics::WholeStateCopy
    );

    let mlx = mlx_sequence_descriptor(MODEL_DOMAIN, Some("f16"));
    mlx.validate().unwrap();
    assert_eq!(mlx.kind, AdapterKind::Sequence);
    assert!(mlx.operations.contains(AdapterOperation::Restore));
    assert!(mlx.operations.contains(AdapterOperation::Fork));
    assert!(!mlx.operations.contains(AdapterOperation::Trim));
    assert_eq!(mlx.constraints.byte_accounting, ByteAccounting::Estimated);
    assert_eq!(mlx.format.cache_format, "mlx-cache-array");
}

#[test]
fn paged_engine_skeleton_fails_closed_until_an_engine_adapter_exists() {
    let descriptor = paged_engine_skeleton_descriptor("vllm", MODEL_DOMAIN);
    descriptor.validate().unwrap();
    assert_eq!(descriptor.kind, AdapterKind::PagedSkeleton);
    assert!(descriptor.operations.contains(AdapterOperation::Inspect));
    assert!(!descriptor.operations.contains(AdapterOperation::Continue));
    assert!(!descriptor.operations.contains(AdapterOperation::Fork));
    assert!(!descriptor.operations.contains(AdapterOperation::Import));

    let mut adapter = CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor)).unwrap();
    assert_eq!(
        adapter.execute(&request(AdapterOperation::Continue)),
        Err(AdapterError::UnsupportedOperation(
            AdapterOperation::Continue
        ))
    );
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 0);
    assert!(backend.invalidated.is_empty());
}

#[test]
fn unsupported_operations_fail_before_physical_mutation() {
    let descriptor = mlx_sequence_descriptor(MODEL_DOMAIN, None);
    let mut adapter = CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor)).unwrap();

    assert_eq!(
        adapter.execute(&request(AdapterOperation::Export)),
        Err(AdapterError::UnsupportedOperation(AdapterOperation::Export))
    );
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 0);
    assert!(backend.invalidated.is_empty());
}

#[test]
fn malformed_requests_fail_before_physical_mutation() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut adapter = CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor)).unwrap();
    let restore_without_donor = request(AdapterOperation::Restore);

    assert_eq!(
        adapter.execute(&restore_without_donor),
        Err(AdapterError::InvalidRequest)
    );
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 0);
    assert!(backend.invalidated.is_empty());
}

#[test]
fn operation_failure_invalidates_the_uncertain_target() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut backend = FakeBackend::new(descriptor);
    backend.execute_error = Some("backend crashed after copy".to_owned());
    let mut adapter = CheckedPhysicalCacheAdapter::new(backend).unwrap();
    let mut restore = request(AdapterOperation::Restore);
    restore.donor = Some(DONOR);

    assert_eq!(
        adapter.execute(&restore),
        Err(AdapterError::PhysicalFailure(
            "backend crashed after copy".to_owned()
        ))
    );
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 1);
    assert_eq!(backend.invalidated, vec![TARGET]);
}

#[test]
fn invalidation_failure_is_never_hidden() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut backend = FakeBackend::new(descriptor);
    backend.execute_error = Some("copy failed".to_owned());
    backend.invalidation_error = Some("target could not be cleared".to_owned());
    let mut adapter = CheckedPhysicalCacheAdapter::new(backend).unwrap();
    let mut restore = request(AdapterOperation::Restore);
    restore.donor = Some(DONOR);

    assert_eq!(
        adapter.execute(&restore),
        Err(AdapterError::InvalidationFailure {
            operation: "copy failed".to_owned(),
            invalidation: "target could not be cleared".to_owned(),
        })
    );
}

#[test]
fn invalid_backend_outcomes_are_rejected_and_invalidated() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut backend = FakeBackend::new(descriptor);
    backend.outcome_target = Some(DONOR);
    let mut adapter = CheckedPhysicalCacheAdapter::new(backend).unwrap();

    assert_eq!(
        adapter.execute(&request(AdapterOperation::Continue)),
        Err(AdapterError::InvalidOutcome)
    );
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 1);
    assert_eq!(backend.invalidated, vec![TARGET]);
}

#[test]
fn abort_invalidates_the_target_without_executing_an_operation() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut adapter = CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor)).unwrap();

    adapter
        .abort(PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION, TARGET)
        .unwrap();
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 0);
    assert_eq!(backend.invalidated, vec![TARGET]);
}

#[test]
fn rolling_version_skew_fails_before_inspection_or_mutation() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut adapter = CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor)).unwrap();
    let incompatible = PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION + 1;
    let mut continue_request = request(AdapterOperation::Continue);
    continue_request.contract_version = incompatible;

    assert_eq!(
        adapter.inspect(incompatible),
        Err(AdapterError::ContractVersion {
            expected: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
            actual: incompatible,
        })
    );
    assert_eq!(
        adapter.execute(&continue_request),
        Err(AdapterError::ContractVersion {
            expected: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
            actual: incompatible,
        })
    );
    let backend = adapter.into_inner();
    assert_eq!(backend.execute_calls, 0);
    assert!(backend.invalidated.is_empty());
}

#[test]
fn import_requires_exact_physical_and_transfer_format_identity() {
    let mut descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    descriptor.operations = AdapterOperations::of(&[
        AdapterOperation::Inspect,
        AdapterOperation::Export,
        AdapterOperation::Import,
    ]);
    descriptor.constraints.minimum_trim_tokens = None;
    descriptor.constraints.transfer_format = Some(clap_cache_core::adapter::TransferFormat {
        identity: "clap-test-blocks".to_owned(),
        version: 1,
    });
    descriptor.validate().unwrap();
    let mut adapter =
        CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor.clone())).unwrap();
    let mut import = request(AdapterOperation::Import);
    import.source_format = Some(PhysicalFormatIdentity {
        model_domain: [9; 32],
        ..descriptor.format.clone()
    });
    import.transfer_format = descriptor.constraints.transfer_format.clone();

    assert_eq!(
        adapter.execute(&import),
        Err(AdapterError::IncompatibleFormat)
    );
    assert_eq!(adapter.into_inner().execute_calls, 0);
}

#[test]
fn byte_telemetry_preserves_explicit_unknown_values() {
    let descriptor = llama_sequence_descriptor(MODEL_DOMAIN, false, true, None);
    let mut adapter = CheckedPhysicalCacheAdapter::new(FakeBackend::new(descriptor)).unwrap();
    let outcome = adapter
        .execute(&request(AdapterOperation::Continue))
        .unwrap();

    assert_eq!(
        outcome.bytes,
        ObservedBytes::Unknown(clap_cache_core::adapter::UnknownBytesReason::BackendDoesNotReport)
    );
    assert_ne!(outcome.bytes, ObservedBytes::Known(0));
}
