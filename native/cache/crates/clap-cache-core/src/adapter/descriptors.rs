use super::{
    AdapterConstraints, AdapterDescriptor, AdapterKind, AdapterOperation, AdapterOperations,
    ByteAccounting, CacheTier, ForkSemantics, PhysicalFormatIdentity, RestoreGranularity,
    PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
};

fn local_format(
    backend: &str,
    engine: &str,
    cache_format: &str,
    model_domain: [u8; 32],
    kv_data_type: Option<&str>,
) -> PhysicalFormatIdentity {
    PhysicalFormatIdentity {
        backend: backend.to_owned(),
        engine: engine.to_owned(),
        cache_format: cache_format.to_owned(),
        cache_format_version: 1,
        model_domain,
        kv_data_type: kv_data_type.map(str::to_owned),
        block_tokens: None,
    }
}

pub fn llama_sequence_descriptor(
    model_domain: [u8; 32],
    hybrid: bool,
    prompt_boundary_snapshots: bool,
    kv_data_type: Option<&str>,
) -> AdapterDescriptor {
    let mut operations = vec![
        AdapterOperation::Inspect,
        AdapterOperation::Continue,
        AdapterOperation::Restore,
        AdapterOperation::Fork,
        AdapterOperation::Release,
    ];
    if !hybrid {
        operations.push(AdapterOperation::Trim);
    }
    if prompt_boundary_snapshots {
        operations.push(AdapterOperation::Snapshot);
    }
    AdapterDescriptor {
        contract_version: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
        kind: AdapterKind::Sequence,
        operations: AdapterOperations::of(&operations),
        format: local_format(
            "llama",
            "llama.cpp",
            "llama-sequence",
            model_domain,
            kv_data_type,
        ),
        constraints: AdapterConstraints {
            restore_granularity: RestoreGranularity::WholeState,
            fork_semantics: if hybrid {
                ForkSemantics::WholeStateCopy
            } else {
                ForkSemantics::CopyOnWrite
            },
            minimum_trim_tokens: (!hybrid).then_some(1),
            safe_busy_donor: true,
            prompt_boundary_snapshots,
            recurrent_or_hybrid: hybrid,
            byte_accounting: ByteAccounting::Unknown,
            tiers: vec![CacheTier::Device],
            transfer_format: None,
        },
    }
}

pub fn mlx_sequence_descriptor(
    model_domain: [u8; 32],
    kv_data_type: Option<&str>,
) -> AdapterDescriptor {
    AdapterDescriptor {
        contract_version: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
        kind: AdapterKind::Sequence,
        operations: AdapterOperations::of(&[
            AdapterOperation::Inspect,
            AdapterOperation::Continue,
            AdapterOperation::Restore,
            AdapterOperation::Fork,
            AdapterOperation::Snapshot,
            AdapterOperation::Release,
        ]),
        format: local_format(
            "mlx",
            "mlx-lm",
            "mlx-cache-array",
            model_domain,
            kv_data_type,
        ),
        constraints: AdapterConstraints {
            restore_granularity: RestoreGranularity::WholeState,
            fork_semantics: ForkSemantics::WholeStateCopy,
            minimum_trim_tokens: None,
            safe_busy_donor: false,
            prompt_boundary_snapshots: true,
            recurrent_or_hybrid: false,
            byte_accounting: ByteAccounting::Estimated,
            tiers: vec![CacheTier::Device],
            transfer_format: None,
        },
    }
}

pub fn paged_engine_skeleton_descriptor(engine: &str, model_domain: [u8; 32]) -> AdapterDescriptor {
    AdapterDescriptor {
        contract_version: PHYSICAL_CACHE_ADAPTER_CONTRACT_VERSION,
        kind: AdapterKind::PagedSkeleton,
        operations: AdapterOperations::of(&[AdapterOperation::Inspect]),
        format: local_format(
            "paged",
            engine,
            "unimplemented-paged-adapter",
            model_domain,
            None,
        ),
        constraints: AdapterConstraints {
            restore_granularity: RestoreGranularity::Block,
            fork_semantics: ForkSemantics::CopyOnWrite,
            minimum_trim_tokens: None,
            safe_busy_donor: false,
            prompt_boundary_snapshots: false,
            recurrent_or_hybrid: false,
            byte_accounting: ByteAccounting::Unknown,
            tiers: vec![CacheTier::Device],
            transfer_format: None,
        },
    }
}
