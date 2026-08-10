use std::path::PathBuf;

use gpt_oss_cpu_kernels::KernelPath;
use gpt_oss_model_runner::model_loader::dtype::DType;
use gpt_oss_model_runner::{CpuGptOssConfig, CpuModelRunner, CpuTensorStore};
use half::bf16;

fn official_snapshot() -> Option<PathBuf> {
    match std::env::var_os("GPT_OSS_TEST_MODEL") {
        Some(path) => Some(path.into()),
        None => {
            eprintln!("GPT_OSS_TEST_MODEL is unset; skipping official checkpoint gate");
            None
        }
    }
}

#[test]
fn official_full_sliding_and_moe_layers_match_scalar_dispatch() {
    if std::env::var_os("GPT_OSS_RUN_LAYER_GATE").is_none() {
        eprintln!("GPT_OSS_RUN_LAYER_GATE is unset; skipping expensive official layer gate");
        return;
    }
    let snapshot = official_snapshot().expect("GPT_OSS_TEST_MODEL is required for layer gate");
    let cache = PathBuf::from(
        std::env::var_os("GPT_OSS_RS_CACHE")
            .expect("GPT_OSS_RS_CACHE is required for official repack artifacts"),
    );
    let mut scalar = CpuModelRunner::load(&snapshot, &cache, KernelPath::Scalar, 4, 8192).unwrap();
    let mut optimized = CpuModelRunner::load(&snapshot, &cache, KernelPath::Auto, 4, 8192).unwrap();
    let hidden = (0..2880)
        .map(|index| bf16::from_f32(((index % 97) as f32 - 48.0) / 97.0))
        .collect::<Vec<_>>();

    for layer in [0, 1] {
        let expected = scalar.conformance_layer(layer, &hidden, 17).unwrap();
        let actual = optimized.conformance_layer(layer, &hidden, 17).unwrap();
        assert_eq!(expected.len(), actual.len());
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            let expected = expected.to_f32();
            let actual = actual.to_f32();
            let tolerance = 0.015625_f32.max(expected.abs() * 0.003);
            assert!(
                (expected - actual).abs() <= tolerance,
                "layer {layer} value {index}: scalar={expected}, optimized={actual}"
            );
        }
    }
}

#[test]
fn official_checkpoint_config_and_tensor_views_decode() {
    let Some(snapshot) = official_snapshot() else {
        return;
    };
    let config = CpuGptOssConfig::from_snapshot(&snapshot).unwrap();
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.hidden_size, 2880);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.num_local_experts, 32);
    assert_eq!(config.num_experts_per_tok, 4);
    assert_eq!(config.sliding_window, 128);
    assert_eq!(
        config
            .layer_types
            .iter()
            .filter(|kind| kind.as_str() == "sliding_attention")
            .count(),
        12
    );

    let store = CpuTensorStore::open(&snapshot).unwrap();
    assert!(store.len() > 300);
    let q = store
        .tensor("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q.dtype(), DType::BF16);
    assert_eq!(q.shape(), &[4096, 2880]);
    assert!(q
        .bf16()
        .unwrap()
        .iter()
        .take(128)
        .all(|value| value.to_f32().is_finite()));

    let blocks = store
        .tensor("model.layers.0.mlp.experts.gate_up_proj_blocks")
        .unwrap();
    let scales = store
        .tensor("model.layers.0.mlp.experts.gate_up_proj_scales")
        .unwrap();
    assert_eq!(blocks.dtype(), DType::U8);
    assert_eq!(blocks.shape(), &[32, 5760, 90, 16]);
    assert_eq!(scales.shape(), &[32, 5760, 90]);
    assert_eq!(blocks.bytes().len(), scales.bytes().len() * 16);
}
