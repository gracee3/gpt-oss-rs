#![cfg(feature = "cuda")]

use gpt_oss_model_runner::cpu_repack::{OWNER_EXPERT_BYTES, OWNER_REPACK_TEMP_BYTES_MAX};
use gpt_oss_model_runner::heterogeneous::contract::GptOssPhase;
use gpt_oss_model_runner::heterogeneous::control::heterogeneous_control_shell_device_bytes;
use gpt_oss_model_runner::heterogeneous::packing::{
    relay_pinned_capacity_bytes, H4_DECODE_PINNED_CAP_BYTES, H4_PREFILL_PINNED_CAP_BYTES,
};
use gpt_oss_model_runner::heterogeneous::reduction::GPT_OSS_REDUCER_OWNED_DEVICE_BYTES;
use gpt_oss_model_runner::heterogeneous::relay::result_relay_owned_device_bytes;
use gpt_oss_model_runner::heterogeneous::router::exact_router_owned_device_bytes;
use gpt_oss_model_runner::heterogeneous::{
    CONSERVATIVE_OWNER_EXPERT_BYTES, GPT_OSS_SELECTED_EXPERT_EXECUTOR_BYTES,
    GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES, GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES,
};
use gpt_oss_model_runner::model_loader::gpt_oss_native::GptOssNativeConfig;
use gpt_oss_model_runner::model_loader::owner_selective::{
    ExecutionReserveDisposition, ExecutionReservePlan, OWNER_SELECTIVE_GPU_RESERVE_BYTES,
    OWNER_SELECTIVE_PROOF_CONTEXT_CAP, OWNER_SELECTIVE_TEMPORARY_CAP_BYTES,
};

#[test]
fn conservative_owner_bytes_cover_both_physical_representations() {
    assert_eq!(GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, 13_236_480);
    assert_eq!(OWNER_EXPERT_BYTES, 13_253_760);
    assert_eq!(CONSERVATIVE_OWNER_EXPERT_BYTES, 13_253_760);
    assert!(OWNER_REPACK_TEMP_BYTES_MAX < 2 * 1024 * 1024);
    assert!(OWNER_REPACK_TEMP_BYTES_MAX < OWNER_SELECTIVE_TEMPORARY_CAP_BYTES);
}

fn config(layers: usize, experts: usize) -> GptOssNativeConfig {
    GptOssNativeConfig {
        num_hidden_layers: layers,
        num_experts: experts,
        experts_per_token: 4,
        vocab_size: 201_088,
        hidden_size: 2_880,
        intermediate_size: 2_880,
        head_dim: 64,
        num_attention_heads: 64,
        num_key_value_heads: 8,
    }
}

#[test]
fn proof_execution_reserve_ledgers_are_byte_exact_and_materialization_is_explicit() {
    let cases = [
        (config(24, 32), 210_252_960, 599_040, 201_326_592),
        (config(36, 128), 334_848_048, 875_520, 301_989_888),
    ];
    for (config, layer_owner_bytes, remote_bytes, kv_bytes) in cases {
        let plan = ExecutionReservePlan::from_config(&config, config.num_hidden_layers).unwrap();
        assert_eq!(
            plan.disposition,
            ExecutionReserveDisposition::PostExecutorAdmissionRuntimePlanReviewed
        );
        assert_eq!(plan.context_cap as usize, OWNER_SELECTIVE_PROOF_CONTEXT_CAP);
        assert_eq!(plan.layer_owner.kv_cache_bytes, kv_bytes);
        assert_eq!(plan.layer_owner.planned_owned_bytes, layer_owner_bytes);
        assert_eq!(plan.remote_gpu.planned_owned_bytes, remote_bytes);
        for device in [&plan.layer_owner, &plan.remote_gpu] {
            assert_eq!(device.reserve_cap_bytes, OWNER_SELECTIVE_GPU_RESERVE_BYTES);
            assert_eq!(
                device.reviewed_deferred_after_admission_bytes
                    + device.runtime_and_safety_remainder_bytes,
                OWNER_SELECTIVE_GPU_RESERVE_BYTES
            );
            assert_eq!(
                device.materialized_before_admission_bytes,
                GPT_OSS_SELECTED_EXPERT_EXECUTOR_BYTES as u64
            );
            assert_eq!(
                device.materialized_before_admission_bytes
                    + device.reviewed_deferred_after_admission_bytes,
                device.planned_owned_bytes
            );
        }
        assert_eq!(plan.layer_owner.selected_expert_executor_bytes, 46_080);
        assert_eq!(plan.remote_gpu.selected_expert_executor_bytes, 46_080);
        assert_eq!(GPT_OSS_SELECTED_EXPERT_EXECUTOR_BYTES, 46_080);
        assert_eq!(
            plan.layer_owner.result_slot_bytes,
            (config.num_hidden_layers
                * config.experts_per_token
                * GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES) as u64
        );
        assert_eq!(
            plan.remote_gpu.result_slot_bytes,
            (config.num_hidden_layers
                * config.experts_per_token
                * GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES) as u64
        );
        assert_eq!(
            plan.layer_owner.kv_cache_bytes + plan.layer_owner.layer_owner_shell_fixed_bytes,
            heterogeneous_control_shell_device_bytes(
                config.num_hidden_layers,
                config.vocab_size,
                OWNER_SELECTIVE_PROOF_CONTEXT_CAP,
            )
            .unwrap() as u64
        );
        assert_eq!(
            plan.layer_owner.router_bytes,
            (config.num_hidden_layers
                * exact_router_owned_device_bytes(config.num_experts, 1).unwrap())
                as u64
        );
        assert_eq!(
            plan.layer_owner.relay_result_arena_bytes,
            (config.num_hidden_layers * result_relay_owned_device_bytes(1).unwrap()) as u64
        );
        assert_eq!(
            plan.layer_owner.reduction_bytes,
            (config.num_hidden_layers * GPT_OSS_REDUCER_OWNED_DEVICE_BYTES) as u64
        );
        let (decode_raw, decode_cap) = relay_pinned_capacity_bytes(GptOssPhase::Decode, 1).unwrap();
        assert_eq!(decode_raw, 74_944);
        assert_eq!(decode_cap, H4_DECODE_PINNED_CAP_BYTES);
        assert_eq!(
            plan.decode_pinned_relay_raw_capacity_bytes,
            decode_raw as u64
        );
        assert_eq!(plan.decode_pinned_relay_cap_bytes, decode_cap as u64);
        assert!(plan.decode_pinned_relay_raw_capacity_bytes <= plan.decode_pinned_relay_cap_bytes);
        assert!(!plan.decode_pinned_relay_materialized_at_construction);
        assert_eq!(
            plan.prefill_pinned_relay_cap_bytes,
            H4_PREFILL_PINNED_CAP_BYTES as u64
        );
        assert!(!plan.prefill_pinned_relay_materialized_at_construction);
        let (prefill_raw, prefill_cap) =
            relay_pinned_capacity_bytes(GptOssPhase::Prefill, 64).unwrap();
        assert_eq!(prefill_raw, 4_796_416);
        assert_eq!(prefill_cap, H4_PREFILL_PINNED_CAP_BYTES);
        assert!(prefill_raw <= prefill_cap);
        plan.validate().unwrap();
    }
}

#[test]
fn remote_result_layer_property_is_bounded_monotonic_and_checked() {
    for config in [config(24, 32), config(36, 128)] {
        let mut prior = None;
        for remote_layers in 0..=config.num_hidden_layers {
            let plan = ExecutionReservePlan::from_config(&config, remote_layers).unwrap();
            assert_eq!(plan.remote_result_layers as usize, remote_layers);
            assert_eq!(plan.remote_gpu.kv_cache_bytes, 0);
            assert_eq!(plan.remote_gpu.router_bytes, 0);
            assert_eq!(plan.remote_gpu.relay_result_arena_bytes, 0);
            assert_eq!(plan.remote_gpu.reduction_bytes, 0);
            if let Some(prior) = prior {
                assert_eq!(
                    plan.remote_gpu.result_slot_bytes - prior,
                    4 * 2_880 * size_of::<u16>() as u64
                );
            }
            prior = Some(plan.remote_gpu.result_slot_bytes);
        }
        assert!(ExecutionReservePlan::from_config(&config, config.num_hidden_layers + 1).is_err());
    }
}

#[test]
fn execution_reserve_validation_rejects_ledger_and_materialization_tampering() {
    let baseline = ExecutionReservePlan::from_config(&config(36, 128), 36).unwrap();

    let mut plan = baseline.clone();
    plan.layer_owner.materialized_before_admission_bytes += 1;
    assert!(plan.validate().is_err());

    let mut plan = baseline.clone();
    plan.remote_gpu.reviewed_deferred_after_admission_bytes += 1;
    assert!(plan.validate().is_err());

    let mut plan = baseline.clone();
    plan.layer_owner.runtime_and_safety_remainder_bytes += 1;
    assert!(plan.validate().is_err());

    let mut plan = baseline.clone();
    plan.decode_pinned_relay_raw_capacity_bytes = plan.decode_pinned_relay_cap_bytes + 1;
    assert!(plan.validate().is_err());

    let mut plan = baseline.clone();
    plan.decode_pinned_relay_materialized_at_construction = true;
    assert!(plan.validate().is_err());

    let mut plan = baseline;
    plan.prefill_pinned_relay_materialized_at_construction = true;
    assert!(plan.validate().is_err());
}

#[test]
fn every_fixed_runtime_shape_mismatch_is_rejected() {
    let baseline = config(36, 128);
    let mut invalid = Vec::new();
    let mut case = baseline.clone();
    case.num_hidden_layers = 0;
    invalid.push(("zero layers", case));
    let mut case = baseline.clone();
    case.num_hidden_layers = 24;
    invalid.push(("layer/expert pairing", case));
    let mut case = baseline.clone();
    case.num_experts = 32;
    invalid.push(("expert count", case));
    let mut case = baseline.clone();
    case.experts_per_token = 3;
    invalid.push(("top-k", case));
    let mut case = baseline.clone();
    case.vocab_size = 0;
    invalid.push(("zero vocabulary", case));
    let mut case = baseline.clone();
    case.vocab_size = 201_087;
    invalid.push(("vocabulary width", case));
    let mut case = baseline.clone();
    case.hidden_size = 42;
    invalid.push(("hidden width", case));
    let mut case = baseline.clone();
    case.intermediate_size = 42;
    invalid.push(("intermediate width", case));
    let mut case = baseline.clone();
    case.head_dim = 32;
    invalid.push(("head dimension", case));
    let mut case = baseline.clone();
    case.num_attention_heads = 32;
    invalid.push(("attention heads", case));
    let mut case = baseline;
    case.num_key_value_heads = 4;
    invalid.push(("K/V heads", case));

    for (label, config) in invalid {
        assert!(
            ExecutionReservePlan::from_config(&config, 0).is_err(),
            "unsupported {label} unexpectedly produced an execution plan"
        );
    }
}

#[test]
fn execution_reserve_reporter_arithmetic_overflow_fails_closed() {
    let error = heterogeneous_control_shell_device_bytes(36, 201_088, usize::MAX).unwrap_err();
    assert!(error.to_string().contains("overflow"));
}

#[test]
fn local_20b_120b_config_metadata_matches_the_reviewed_plans() {
    if std::env::var_os("GPT_OSS_RUN_EXECUTION_RESERVE_METADATA").is_none() {
        eprintln!("GPT_OSS_RUN_EXECUTION_RESERVE_METADATA unset; skipping local metadata gate");
        return;
    }
    for (path, layers, experts, expected_kv) in [
        (
            "/data/models/openai/gpt-oss-20b/original/config.json",
            24,
            32,
            201_326_592,
        ),
        (
            "/data/models/openai/gpt-oss-120b/original/config.json",
            36,
            128,
            301_989_888,
        ),
    ] {
        let bytes = std::fs::read(path).unwrap();
        let config: GptOssNativeConfig = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            (config.num_hidden_layers, config.num_experts),
            (layers, experts)
        );
        let plan = ExecutionReservePlan::from_config(&config, layers).unwrap();
        assert_eq!(plan.layer_owner.kv_cache_bytes, expected_kv);
        assert_eq!(plan.context_cap, 4_096);
    }
}
