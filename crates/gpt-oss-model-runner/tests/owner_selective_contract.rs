#![cfg(feature = "cuda")]

use gpt_oss_model_runner::cpu_repack::{OWNER_EXPERT_BYTES, OWNER_REPACK_TEMP_BYTES_MAX};
use gpt_oss_model_runner::heterogeneous::{
    CONSERVATIVE_OWNER_EXPERT_BYTES, GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES,
};
use gpt_oss_model_runner::model_loader::owner_selective::OWNER_SELECTIVE_TEMPORARY_CAP_BYTES;

#[test]
fn conservative_owner_bytes_cover_both_physical_representations() {
    assert_eq!(GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES, 13_236_480);
    assert_eq!(OWNER_EXPERT_BYTES, 13_253_760);
    assert_eq!(CONSERVATIVE_OWNER_EXPERT_BYTES, 13_253_760);
    assert!(OWNER_REPACK_TEMP_BYTES_MAX < 2 * 1024 * 1024);
    assert!(OWNER_REPACK_TEMP_BYTES_MAX < OWNER_SELECTIVE_TEMPORARY_CAP_BYTES);
}
