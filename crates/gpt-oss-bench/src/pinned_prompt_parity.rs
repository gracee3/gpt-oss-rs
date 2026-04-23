use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use gpt_oss_core::prelude::TokenId;
use gpt_oss_tokenizer::{ProtocolMessage, ToolDefinition};
use serde::{Deserialize, Serialize};

pub const PINNED_PROMPT_MANIFEST_SCHEMA_VERSION: &str = "pinned-prompt-manifest/v1";
pub const PINNED_PROMPT_ARTIFACT_SCHEMA_VERSION: &str = "pinned-prompt-artifact/v2";
pub const PINNED_PROMPT_COMPARE_SCHEMA_VERSION: &str = "pinned-prompt-compare/v3";
pub const PINNED_PROMPT_INTERMEDIATE_ARTIFACT_SCHEMA_VERSION: &str =
    "pinned-prompt-intermediate-artifact/v2";
pub const PINNED_PROMPT_INTERMEDIATE_COMPARE_SCHEMA_VERSION: &str =
    "pinned-prompt-intermediate-compare/v2";
pub const PINNED_PROMPT_OFFICIAL_CAPTURE_INPUT_SCHEMA_VERSION: &str =
    "pinned-prompt-official-capture-input/v1";
pub const PINNED_PROMPT_OFFICIAL_CAPTURE_OUTPUT_SCHEMA_VERSION: &str =
    "pinned-prompt-official-capture-output/v1";
pub const PINNED_PROMPT_OFFICIAL_INTERMEDIATE_CAPTURE_INPUT_SCHEMA_VERSION: &str =
    "pinned-prompt-official-intermediate-capture-input/v2";
pub const PINNED_PROMPT_OFFICIAL_INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA_VERSION: &str =
    "pinned-prompt-official-intermediate-capture-output/v2";
pub const DEFAULT_PINNED_PROMPT_TOP_K: usize = 8;
pub const PINNED_PROMPT_RENDERER_HARMONY_GPT_OSS_RS: &str = "harmony_gpt_oss_rs";
pub const PINNED_PROMPT_INTERMEDIATE_COMPARE_TOLERANCE: f32 = 1e-2;

pub fn default_manifest_schema_version() -> String {
    PINNED_PROMPT_MANIFEST_SCHEMA_VERSION.to_string()
}

pub fn default_artifact_schema_version() -> String {
    PINNED_PROMPT_ARTIFACT_SCHEMA_VERSION.to_string()
}

pub fn default_compare_schema_version() -> String {
    PINNED_PROMPT_COMPARE_SCHEMA_VERSION.to_string()
}

pub fn default_intermediate_artifact_schema_version() -> String {
    PINNED_PROMPT_INTERMEDIATE_ARTIFACT_SCHEMA_VERSION.to_string()
}

pub fn default_intermediate_compare_schema_version() -> String {
    PINNED_PROMPT_INTERMEDIATE_COMPARE_SCHEMA_VERSION.to_string()
}

pub fn default_official_capture_input_schema_version() -> String {
    PINNED_PROMPT_OFFICIAL_CAPTURE_INPUT_SCHEMA_VERSION.to_string()
}

pub fn default_official_capture_output_schema_version() -> String {
    PINNED_PROMPT_OFFICIAL_CAPTURE_OUTPUT_SCHEMA_VERSION.to_string()
}

pub fn default_official_intermediate_capture_input_schema_version() -> String {
    PINNED_PROMPT_OFFICIAL_INTERMEDIATE_CAPTURE_INPUT_SCHEMA_VERSION.to_string()
}

pub fn default_official_intermediate_capture_output_schema_version() -> String {
    PINNED_PROMPT_OFFICIAL_INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA_VERSION.to_string()
}

fn default_reference_kind() -> ReferenceKind {
    ReferenceKind::LocalCandidate
}

fn default_authority_level() -> AuthorityLevel {
    AuthorityLevel::Scaffold
}

fn default_top_k() -> usize {
    DEFAULT_PINNED_PROMPT_TOP_K
}

pub fn default_prompt_renderer() -> String {
    PINNED_PROMPT_RENDERER_HARMONY_GPT_OSS_RS.to_string()
}

pub fn default_intermediate_compare_tolerance() -> f32 {
    PINNED_PROMPT_INTERMEDIATE_COMPARE_TOLERANCE
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    LocalCandidate,
    OfficialReference,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityLevel {
    Scaffold,
    Authoritative,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptManifest {
    #[serde(default = "default_manifest_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    #[serde(default = "default_top_k")]
    pub default_top_k: usize,
    #[serde(default)]
    pub default_capture_source: CaptureSource,
    #[serde(default = "default_reference_kind")]
    pub reference_kind: ReferenceKind,
    #[serde(default = "default_authority_level")]
    pub authority_level: AuthorityLevel,
    pub cases: Vec<PinnedPromptCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptCase {
    pub id: String,
    #[serde(default)]
    pub instructions: Option<String>,
    pub messages: Vec<ProtocolMessage>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub top_k: Option<usize>,
}

impl PinnedPromptCase {
    pub fn effective_top_k(&self, default_top_k: usize) -> usize {
        self.top_k.unwrap_or(default_top_k).max(1)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CaptureSource {
    Runner,
    Worker,
    OfficialTorch,
}

impl Default for CaptureSource {
    fn default() -> Self {
        Self::Runner
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptCaptureProvenance {
    pub model: String,
    pub capture_source: CaptureSource,
    pub reference_kind: ReferenceKind,
    pub authority_level: AuthorityLevel,
    #[serde(default = "default_prompt_renderer")]
    pub prompt_renderer: String,
    pub visible_devices: String,
    pub max_model_len: usize,
    pub gpu_memory_utilization: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptArtifact {
    #[serde(default = "default_artifact_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    pub provenance: PinnedPromptCaptureProvenance,
    pub cases: Vec<PinnedPromptCaseArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptCaseArtifact {
    pub id: String,
    pub input_token_ids: Vec<TokenId>,
    pub argmax_token_id: TokenId,
    pub argmax_token_text: String,
    pub final_position_top_k: Vec<PinnedTopLogit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedTopLogit {
    pub token_id: TokenId,
    pub token_text: String,
    pub logit: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum PinnedPromptIntermediateBoundary {
    FinalTokenPostFinalNormPreUnembedding,
    TransformerLayerOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptOfficialCaptureInput {
    #[serde(default = "default_official_capture_input_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    pub case_id: String,
    #[serde(default = "default_prompt_renderer")]
    pub prompt_renderer: String,
    pub model: String,
    pub official_model: String,
    pub input_token_ids: Vec<TokenId>,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptOfficialCaptureOutput {
    #[serde(default = "default_official_capture_output_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    pub case_id: String,
    pub backend: String,
    pub official_model: String,
    #[serde(default = "default_prompt_renderer")]
    pub prompt_renderer: String,
    pub input_token_ids: Vec<TokenId>,
    pub argmax_token_id: TokenId,
    pub final_position_top_k: Vec<PinnedPromptOfficialTopLogit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptOfficialTopLogit {
    pub token_id: TokenId,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptOfficialIntermediateCaptureInput {
    #[serde(default = "default_official_intermediate_capture_input_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    pub case_id: String,
    #[serde(default = "default_prompt_renderer")]
    pub prompt_renderer: String,
    pub model: String,
    pub official_model: String,
    pub input_token_ids: Vec<TokenId>,
    pub boundary: PinnedPromptIntermediateBoundary,
    #[serde(default)]
    pub layer_idx: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptOfficialIntermediateCaptureOutput {
    #[serde(default = "default_official_intermediate_capture_output_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    pub case_id: String,
    pub backend: String,
    pub official_model: String,
    #[serde(default = "default_prompt_renderer")]
    pub prompt_renderer: String,
    pub input_token_ids: Vec<TokenId>,
    pub boundary: PinnedPromptIntermediateBoundary,
    #[serde(default)]
    pub layer_idx: Option<usize>,
    pub hidden_size: usize,
    pub final_token_hidden_f32: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptIntermediateArtifact {
    #[serde(default = "default_intermediate_artifact_schema_version")]
    pub schema_version: String,
    pub suite_id: String,
    pub boundary: PinnedPromptIntermediateBoundary,
    #[serde(default)]
    pub layer_idx: Option<usize>,
    pub provenance: PinnedPromptCaptureProvenance,
    pub cases: Vec<PinnedPromptIntermediateCaseArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptIntermediateCaseArtifact {
    pub id: String,
    pub input_token_ids: Vec<TokenId>,
    pub hidden_size: usize,
    pub final_token_hidden_f32: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptCompareReport {
    #[serde(default = "default_compare_schema_version")]
    pub schema_version: String,
    pub left_suite_id: String,
    pub right_suite_id: String,
    pub suite_id_match: bool,
    pub left_provenance: PinnedPromptCaptureProvenance,
    pub right_provenance: PinnedPromptCaptureProvenance,
    pub compare_kind: PinnedPromptCompareKind,
    pub provenance_match: bool,
    pub local_gate_eligible: bool,
    pub local_gate_pass: bool,
    pub correctness_implied: bool,
    pub missing_from_left: Vec<String>,
    pub missing_from_right: Vec<String>,
    pub cases: Vec<PinnedPromptCaseComparison>,
    pub content_match: bool,
    pub overall_match: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PinnedPromptCompareKind {
    SameSourceSelfCompare,
    RunnerVsWorkerLocalCompare,
    LocalVsOfficialReferenceCompare,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptCaseComparison {
    pub id: String,
    pub input_token_ids_match: bool,
    pub argmax_token_id_match: bool,
    pub top_k_token_ids_match: bool,
    pub top_k_logits_match: bool,
    pub max_shared_top_k_abs_diff: f32,
    pub shared_top_k_logit_diffs: Vec<PinnedTopLogitDiff>,
    pub matches: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedTopLogitDiff {
    pub token_id: TokenId,
    pub left_logit: f32,
    pub right_logit: f32,
    pub abs_diff: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptIntermediateCompareReport {
    #[serde(default = "default_intermediate_compare_schema_version")]
    pub schema_version: String,
    pub left_suite_id: String,
    pub right_suite_id: String,
    pub suite_id_match: bool,
    pub left_boundary: PinnedPromptIntermediateBoundary,
    pub right_boundary: PinnedPromptIntermediateBoundary,
    pub boundary_match: bool,
    pub left_layer_idx: Option<usize>,
    pub right_layer_idx: Option<usize>,
    pub layer_idx_match: bool,
    #[serde(default = "default_intermediate_compare_tolerance")]
    pub tolerance: f32,
    pub left_provenance: PinnedPromptCaptureProvenance,
    pub right_provenance: PinnedPromptCaptureProvenance,
    pub compare_kind: PinnedPromptIntermediateCompareKind,
    pub provenance_match: bool,
    pub boundary_authoritative: bool,
    pub end_to_end_correctness_implied: bool,
    pub missing_from_left: Vec<String>,
    pub missing_from_right: Vec<String>,
    pub cases: Vec<PinnedPromptIntermediateCaseComparison>,
    pub content_match: bool,
    pub overall_match: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PinnedPromptIntermediateCompareKind {
    SameSourceSelfCompare,
    LocalVsOfficialReferenceIntermediateCompare,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PinnedPromptIntermediateCaseComparison {
    pub id: String,
    pub input_token_ids_match: bool,
    pub hidden_size_match: bool,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub matched_within_tolerance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeForwardIntermediateCapture {
    pub artifact_kind: String,
    pub capture_source: CaptureSource,
    pub prompt: String,
    pub prompt_token_ids: Vec<TokenId>,
    pub input_token_ids: Vec<TokenId>,
    pub prompt_token_count: usize,
    pub final_position: usize,
    pub restricted_model_path: String,
    pub visible_devices: String,
    pub seam_identifier: String,
    pub layer_idx: usize,
    pub hidden_state_dim: usize,
    pub last_token_hidden_state: Vec<f32>,
}

pub fn duplicate_case_ids<'a, I>(cases: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for id in cases {
        if !seen.insert(id.to_string()) {
            duplicates.insert(id.to_string());
        }
    }
    duplicates.into_iter().collect()
}

pub fn validate_manifest(manifest: &PinnedPromptManifest) -> Result<()> {
    let duplicates = duplicate_case_ids(manifest.cases.iter().map(|case| case.id.as_str()));
    if !duplicates.is_empty() {
        bail!(
            "manifest contains duplicate case ids: {}",
            duplicates.join(", ")
        );
    }
    if manifest.default_top_k == 0 {
        bail!("manifest default_top_k must be greater than 0");
    }
    validate_authority_reference_pair(
        manifest.authority_level,
        manifest.reference_kind,
        "manifest",
    )?;
    for case in &manifest.cases {
        if case.top_k == Some(0) {
            bail!("manifest case '{}' has invalid top_k=0", case.id);
        }
    }
    Ok(())
}

pub fn validate_artifact(artifact: &PinnedPromptArtifact) -> Result<()> {
    let duplicates = duplicate_case_ids(artifact.cases.iter().map(|case| case.id.as_str()));
    if !duplicates.is_empty() {
        bail!(
            "artifact contains duplicate case ids: {}",
            duplicates.join(", ")
        );
    }
    validate_authority_reference_pair(
        artifact.provenance.authority_level,
        artifact.provenance.reference_kind,
        "artifact",
    )?;
    Ok(())
}

pub fn validate_intermediate_artifact(artifact: &PinnedPromptIntermediateArtifact) -> Result<()> {
    let duplicates = duplicate_case_ids(artifact.cases.iter().map(|case| case.id.as_str()));
    if !duplicates.is_empty() {
        bail!(
            "intermediate artifact contains duplicate case ids: {}",
            duplicates.join(", ")
        );
    }
    validate_authority_reference_pair(
        artifact.provenance.authority_level,
        artifact.provenance.reference_kind,
        "intermediate artifact",
    )?;
    validate_intermediate_boundary_layer_idx(
        artifact.boundary,
        artifact.layer_idx,
        "intermediate artifact",
    )?;
    for case in &artifact.cases {
        if case.hidden_size == 0 {
            bail!(
                "intermediate artifact case '{}' has invalid hidden_size=0",
                case.id
            );
        }
        if case.final_token_hidden_f32.len() != case.hidden_size {
            bail!(
                "intermediate artifact case '{}' hidden size mismatch: hidden_size={}, values={}",
                case.id,
                case.hidden_size,
                case.final_token_hidden_f32.len()
            );
        }
    }
    Ok(())
}

pub fn validate_intermediate_boundary_layer_idx(
    boundary: PinnedPromptIntermediateBoundary,
    layer_idx: Option<usize>,
    subject: &str,
) -> Result<()> {
    match (boundary, layer_idx) {
        (PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding, None) => Ok(()),
        (
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            Some(layer_idx),
        ) => bail!(
            "{subject} boundary 'final_token_post_final_norm_pre_unembedding' requires layer_idx=None, got {layer_idx}"
        ),
        (PinnedPromptIntermediateBoundary::TransformerLayerOutput, Some(_)) => Ok(()),
        (PinnedPromptIntermediateBoundary::TransformerLayerOutput, None) => bail!(
            "{subject} boundary 'transformer_layer_output' requires layer_idx"
        ),
    }
}

fn validate_authority_reference_pair(
    authority_level: AuthorityLevel,
    reference_kind: ReferenceKind,
    subject: &str,
) -> Result<()> {
    if authority_level == AuthorityLevel::Authoritative
        && reference_kind != ReferenceKind::OfficialReference
    {
        bail!("{subject} authority_level=authoritative requires reference_kind=official_reference");
    }
    Ok(())
}

pub fn compare_artifacts(
    left: &PinnedPromptArtifact,
    right: &PinnedPromptArtifact,
) -> PinnedPromptCompareReport {
    let left_cases = left
        .cases
        .iter()
        .map(|case| (case.id.clone(), case))
        .collect::<BTreeMap<_, _>>();
    let right_cases = right
        .cases
        .iter()
        .map(|case| (case.id.clone(), case))
        .collect::<BTreeMap<_, _>>();

    let left_ids = left_cases.keys().cloned().collect::<BTreeSet<_>>();
    let right_ids = right_cases.keys().cloned().collect::<BTreeSet<_>>();
    let missing_from_left = right_ids.difference(&left_ids).cloned().collect::<Vec<_>>();
    let missing_from_right = left_ids.difference(&right_ids).cloned().collect::<Vec<_>>();

    let mut cases = Vec::new();
    for case_id in left_ids.intersection(&right_ids) {
        let left_case = left_cases.get(case_id).expect("shared case must exist");
        let right_case = right_cases.get(case_id).expect("shared case must exist");
        cases.push(compare_case(left_case, right_case));
    }

    let suite_id_match = left.suite_id == right.suite_id;
    let content_match = suite_id_match
        && missing_from_left.is_empty()
        && missing_from_right.is_empty()
        && cases.iter().all(|case| case.matches);
    let provenance_match = left.provenance == right.provenance;
    let compare_kind = classify_compare_kind(&left.provenance, &right.provenance);
    let local_gate_eligible = matches!(
        compare_kind,
        PinnedPromptCompareKind::SameSourceSelfCompare
            | PinnedPromptCompareKind::RunnerVsWorkerLocalCompare
    );
    let local_gate_pass = local_gate_eligible && content_match;
    let correctness_implied =
        compare_kind == PinnedPromptCompareKind::LocalVsOfficialReferenceCompare;

    PinnedPromptCompareReport {
        schema_version: default_compare_schema_version(),
        left_suite_id: left.suite_id.clone(),
        right_suite_id: right.suite_id.clone(),
        suite_id_match,
        left_provenance: left.provenance.clone(),
        right_provenance: right.provenance.clone(),
        compare_kind,
        provenance_match,
        local_gate_eligible,
        local_gate_pass,
        correctness_implied,
        missing_from_left,
        missing_from_right,
        cases,
        content_match,
        overall_match: content_match,
    }
}

pub fn compare_intermediate_artifacts(
    left: &PinnedPromptIntermediateArtifact,
    right: &PinnedPromptIntermediateArtifact,
) -> PinnedPromptIntermediateCompareReport {
    let left_cases = left
        .cases
        .iter()
        .map(|case| (case.id.clone(), case))
        .collect::<BTreeMap<_, _>>();
    let right_cases = right
        .cases
        .iter()
        .map(|case| (case.id.clone(), case))
        .collect::<BTreeMap<_, _>>();

    let left_ids = left_cases.keys().cloned().collect::<BTreeSet<_>>();
    let right_ids = right_cases.keys().cloned().collect::<BTreeSet<_>>();
    let missing_from_left = right_ids.difference(&left_ids).cloned().collect::<Vec<_>>();
    let missing_from_right = left_ids.difference(&right_ids).cloned().collect::<Vec<_>>();

    let mut cases = Vec::new();
    for case_id in left_ids.intersection(&right_ids) {
        let left_case = left_cases.get(case_id).expect("shared case must exist");
        let right_case = right_cases.get(case_id).expect("shared case must exist");
        cases.push(compare_intermediate_case(left_case, right_case));
    }

    let suite_id_match = left.suite_id == right.suite_id;
    let boundary_match = left.boundary == right.boundary;
    let layer_idx_match = left.layer_idx == right.layer_idx;
    let content_match = suite_id_match
        && boundary_match
        && layer_idx_match
        && missing_from_left.is_empty()
        && missing_from_right.is_empty()
        && cases.iter().all(|case| case.matched_within_tolerance);
    let provenance_match = left.provenance == right.provenance;
    let compare_kind = classify_intermediate_compare_kind(
        &left.provenance,
        &right.provenance,
        boundary_match && layer_idx_match,
    );
    let boundary_authoritative = compare_kind
        == PinnedPromptIntermediateCompareKind::LocalVsOfficialReferenceIntermediateCompare;

    PinnedPromptIntermediateCompareReport {
        schema_version: default_intermediate_compare_schema_version(),
        left_suite_id: left.suite_id.clone(),
        right_suite_id: right.suite_id.clone(),
        suite_id_match,
        left_boundary: left.boundary,
        right_boundary: right.boundary,
        boundary_match,
        left_layer_idx: left.layer_idx,
        right_layer_idx: right.layer_idx,
        layer_idx_match,
        tolerance: default_intermediate_compare_tolerance(),
        left_provenance: left.provenance.clone(),
        right_provenance: right.provenance.clone(),
        compare_kind,
        provenance_match,
        boundary_authoritative,
        end_to_end_correctness_implied: false,
        missing_from_left,
        missing_from_right,
        cases,
        content_match,
        overall_match: content_match,
    }
}

fn classify_compare_kind(
    left: &PinnedPromptCaptureProvenance,
    right: &PinnedPromptCaptureProvenance,
) -> PinnedPromptCompareKind {
    if left == right {
        return PinnedPromptCompareKind::SameSourceSelfCompare;
    }

    let left_is_local_pre_authoritative = is_local_pre_authoritative(left);
    let right_is_local_pre_authoritative = is_local_pre_authoritative(right);
    let left_is_official_reference = is_authoritative_official_reference(left);
    let right_is_official_reference = is_authoritative_official_reference(right);

    if (left_is_official_reference && right_is_local_pre_authoritative)
        || (right_is_official_reference && left_is_local_pre_authoritative)
    {
        return PinnedPromptCompareKind::LocalVsOfficialReferenceCompare;
    }

    let runner_worker_pair = matches!(
        (left.capture_source, right.capture_source),
        (CaptureSource::Runner, CaptureSource::Worker)
            | (CaptureSource::Worker, CaptureSource::Runner)
    );
    let matching_except_capture_source = left.model == right.model
        && left.reference_kind == right.reference_kind
        && left.authority_level == right.authority_level
        && left.visible_devices == right.visible_devices
        && left.max_model_len == right.max_model_len
        && left.gpu_memory_utilization == right.gpu_memory_utilization;

    if left_is_local_pre_authoritative
        && right_is_local_pre_authoritative
        && runner_worker_pair
        && matching_except_capture_source
    {
        return PinnedPromptCompareKind::RunnerVsWorkerLocalCompare;
    }

    PinnedPromptCompareKind::Other
}

fn is_local_pre_authoritative(provenance: &PinnedPromptCaptureProvenance) -> bool {
    matches!(
        provenance.capture_source,
        CaptureSource::Runner | CaptureSource::Worker
    ) && provenance.reference_kind == ReferenceKind::LocalCandidate
        && provenance.authority_level != AuthorityLevel::Authoritative
}

fn is_authoritative_official_reference(provenance: &PinnedPromptCaptureProvenance) -> bool {
    provenance.capture_source == CaptureSource::OfficialTorch
        && provenance.reference_kind == ReferenceKind::OfficialReference
        && provenance.authority_level == AuthorityLevel::Authoritative
}

fn compare_case(
    left: &PinnedPromptCaseArtifact,
    right: &PinnedPromptCaseArtifact,
) -> PinnedPromptCaseComparison {
    let input_token_ids_match = left.input_token_ids == right.input_token_ids;
    let argmax_token_id_match = left.argmax_token_id == right.argmax_token_id;

    let left_top_k_ids = left
        .final_position_top_k
        .iter()
        .map(|entry| entry.token_id)
        .collect::<Vec<_>>();
    let right_top_k_ids = right
        .final_position_top_k
        .iter()
        .map(|entry| entry.token_id)
        .collect::<Vec<_>>();
    let top_k_token_ids_match = left_top_k_ids == right_top_k_ids;

    let right_by_token = right
        .final_position_top_k
        .iter()
        .map(|entry| (entry.token_id, entry))
        .collect::<BTreeMap<_, _>>();
    let mut shared_top_k_logit_diffs = left
        .final_position_top_k
        .iter()
        .filter_map(|left_entry| {
            right_by_token.get(&left_entry.token_id).map(|right_entry| {
                let abs_diff = (left_entry.logit - right_entry.logit).abs();
                PinnedTopLogitDiff {
                    token_id: left_entry.token_id,
                    left_logit: left_entry.logit,
                    right_logit: right_entry.logit,
                    abs_diff,
                }
            })
        })
        .collect::<Vec<_>>();
    shared_top_k_logit_diffs.sort_by(|a, b| {
        a.token_id
            .cmp(&b.token_id)
            .then_with(|| a.abs_diff.total_cmp(&b.abs_diff))
    });

    let max_shared_top_k_abs_diff = shared_top_k_logit_diffs
        .iter()
        .map(|entry| entry.abs_diff)
        .fold(0.0f32, f32::max);
    let top_k_logits_match = shared_top_k_logit_diffs
        .iter()
        .all(|entry| entry.abs_diff == 0.0)
        && shared_top_k_logit_diffs.len() == left.final_position_top_k.len()
        && left.final_position_top_k.len() == right.final_position_top_k.len();

    let matches = input_token_ids_match
        && argmax_token_id_match
        && top_k_token_ids_match
        && top_k_logits_match;

    PinnedPromptCaseComparison {
        id: left.id.clone(),
        input_token_ids_match,
        argmax_token_id_match,
        top_k_token_ids_match,
        top_k_logits_match,
        max_shared_top_k_abs_diff,
        shared_top_k_logit_diffs,
        matches,
    }
}

fn classify_intermediate_compare_kind(
    left: &PinnedPromptCaptureProvenance,
    right: &PinnedPromptCaptureProvenance,
    boundary_and_layer_match: bool,
) -> PinnedPromptIntermediateCompareKind {
    if boundary_and_layer_match && left == right {
        return PinnedPromptIntermediateCompareKind::SameSourceSelfCompare;
    }

    let left_is_local_pre_authoritative = is_local_pre_authoritative(left);
    let right_is_local_pre_authoritative = is_local_pre_authoritative(right);
    let left_is_official_reference = is_authoritative_official_reference(left);
    let right_is_official_reference = is_authoritative_official_reference(right);

    if boundary_and_layer_match
        && ((left_is_official_reference && right_is_local_pre_authoritative)
            || (right_is_official_reference && left_is_local_pre_authoritative))
    {
        return PinnedPromptIntermediateCompareKind::LocalVsOfficialReferenceIntermediateCompare;
    }

    PinnedPromptIntermediateCompareKind::Other
}

pub fn rendered_case_input_token_ids(
    manifest: &PinnedPromptManifest,
    case_id: &str,
) -> Result<Vec<TokenId>> {
    let case = manifest
        .cases
        .iter()
        .find(|case| case.id == case_id)
        .with_context(|| format!("manifest missing case '{}'", case_id))?;
    let protocol = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()?;
    let rendered =
        protocol.render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)?;
    Ok(rendered.token_ids)
}

pub fn import_runtime_forward_intermediate_capture(
    manifest: &PinnedPromptManifest,
    case_id: &str,
    capture: &RuntimeForwardIntermediateCapture,
    max_model_len: usize,
    gpu_memory_utilization: f32,
) -> Result<PinnedPromptIntermediateArtifact> {
    if capture.artifact_kind != "runner_prefill_last_token_hidden_state" {
        bail!(
            "runtime-forward intermediate import requires artifact_kind='runner_prefill_last_token_hidden_state', got '{}'",
            capture.artifact_kind
        );
    }
    if capture.capture_source != CaptureSource::Runner {
        bail!(
            "runtime-forward intermediate import requires capture_source='runner', got '{:?}'",
            capture.capture_source
        );
    }
    if capture.seam_identifier != "transformer_layer_output" {
        bail!(
            "runtime-forward intermediate import requires seam_identifier='transformer_layer_output', got '{}'",
            capture.seam_identifier
        );
    }

    let rendered_input_token_ids = rendered_case_input_token_ids(manifest, case_id)?;
    if capture.input_token_ids != rendered_input_token_ids {
        bail!(
            "runtime-forward intermediate import input_token_ids mismatch for case '{}'",
            case_id
        );
    }

    let artifact = PinnedPromptIntermediateArtifact {
        schema_version: default_intermediate_artifact_schema_version(),
        suite_id: manifest.suite_id.clone(),
        boundary: PinnedPromptIntermediateBoundary::TransformerLayerOutput,
        layer_idx: Some(capture.layer_idx),
        provenance: PinnedPromptCaptureProvenance {
            model: capture.restricted_model_path.clone(),
            capture_source: CaptureSource::Runner,
            reference_kind: manifest.reference_kind,
            authority_level: manifest.authority_level,
            prompt_renderer: default_prompt_renderer(),
            visible_devices: capture.visible_devices.clone(),
            max_model_len,
            gpu_memory_utilization,
        },
        cases: vec![PinnedPromptIntermediateCaseArtifact {
            id: case_id.to_string(),
            input_token_ids: capture.input_token_ids.clone(),
            hidden_size: capture.hidden_state_dim,
            final_token_hidden_f32: capture.last_token_hidden_state.clone(),
        }],
    };
    validate_intermediate_artifact(&artifact)?;
    Ok(artifact)
}

fn compare_intermediate_case(
    left: &PinnedPromptIntermediateCaseArtifact,
    right: &PinnedPromptIntermediateCaseArtifact,
) -> PinnedPromptIntermediateCaseComparison {
    let input_token_ids_match = left.input_token_ids == right.input_token_ids;
    let hidden_size_match = left.hidden_size == right.hidden_size
        && left.final_token_hidden_f32.len() == right.final_token_hidden_f32.len()
        && left.final_token_hidden_f32.len() == left.hidden_size;
    let shared_len = left
        .final_token_hidden_f32
        .len()
        .min(right.final_token_hidden_f32.len());

    let (max_abs_diff, mean_abs_diff) = if shared_len == 0 {
        (0.0, 0.0)
    } else {
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f32;
        for idx in 0..shared_len {
            let abs_diff =
                (left.final_token_hidden_f32[idx] - right.final_token_hidden_f32[idx]).abs();
            max_abs_diff = max_abs_diff.max(abs_diff);
            sum_abs_diff += abs_diff;
        }
        (max_abs_diff, sum_abs_diff / shared_len as f32)
    };

    let matched_within_tolerance = input_token_ids_match
        && hidden_size_match
        && max_abs_diff <= default_intermediate_compare_tolerance();

    PinnedPromptIntermediateCaseComparison {
        id: left.id.clone(),
        input_token_ids_match,
        hidden_size_match,
        max_abs_diff,
        mean_abs_diff,
        matched_within_tolerance,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use gpt_oss_core::prelude::TokenId;
    use gpt_oss_tokenizer::HarmonyProtocol;

    use super::{
        compare_artifacts, compare_intermediate_artifacts, default_artifact_schema_version,
        default_intermediate_artifact_schema_version, import_runtime_forward_intermediate_capture,
        validate_artifact, validate_intermediate_artifact,
        validate_intermediate_boundary_layer_idx, validate_manifest, AuthorityLevel, CaptureSource,
        PinnedPromptArtifact, PinnedPromptCaptureProvenance, PinnedPromptCaseArtifact,
        PinnedPromptCompareKind, PinnedPromptIntermediateArtifact,
        PinnedPromptIntermediateBoundary, PinnedPromptIntermediateCaseArtifact,
        PinnedPromptIntermediateCompareKind, PinnedPromptManifest, PinnedTopLogit, ReferenceKind,
        RuntimeForwardIntermediateCapture,
    };

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("pinned-prompts")
    }

    fn smoke_fixture_path() -> PathBuf {
        fixture_dir().join("smoke.json")
    }

    fn fixture_paths() -> Vec<PathBuf> {
        let mut fixtures = fs::read_dir(fixture_dir())
            .expect("read pinned prompt fixture dir")
            .map(|entry| entry.expect("read fixture entry").path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
            .collect::<Vec<_>>();
        fixtures.sort();
        fixtures
    }

    #[test]
    fn checked_in_fixtures_validate_and_render() {
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let fixtures = fixture_paths();
        assert!(
            fixtures.len() >= 2,
            "expected at least two checked-in PPP fixtures"
        );

        for path in fixtures {
            let fixture = fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("read fixture {}", path.display()));
            let manifest: PinnedPromptManifest = serde_json::from_str(&fixture)
                .unwrap_or_else(|_| panic!("parse fixture {}", path.display()));
            validate_manifest(&manifest)
                .unwrap_or_else(|_| panic!("validate fixture {}", path.display()));

            for case in &manifest.cases {
                let rendered = protocol
                    .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
                    .unwrap_or_else(|_| {
                        panic!("render case '{}' from {}", case.id, path.display())
                    });
                assert!(
                    !rendered.token_ids.is_empty(),
                    "rendered fixture case '{}' from {} to an empty prompt",
                    case.id,
                    path.display()
                );
            }
        }
    }

    #[test]
    fn smoke_fixture_roundtrips_artifact_content() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");

        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let artifact = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::Runner,
        );
        validate_artifact(&artifact).expect("validate synthetic artifact");
        let roundtrip: PinnedPromptArtifact =
            serde_json::from_str(&serde_json::to_string_pretty(&artifact).expect("serialize"))
                .expect("deserialize");
        validate_artifact(&roundtrip).expect("validate roundtrip artifact");

        let report = compare_artifacts(&artifact, &roundtrip);
        assert_eq!(
            report.compare_kind,
            PinnedPromptCompareKind::SameSourceSelfCompare
        );
        assert!(report.content_match);
        assert!(report.overall_match);
        assert!(report.provenance_match);
        assert!(report.local_gate_eligible);
        assert!(report.local_gate_pass);
        assert!(!report.correctness_implied);
        assert!(report.missing_from_left.is_empty());
        assert!(report.missing_from_right.is_empty());
        assert_eq!(report.cases.len(), 1);
        assert!(report.cases[0].matches);
    }

    #[test]
    fn compare_report_separates_provenance_from_content_match() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let runner = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
        );
        let worker = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::Worker,
        );

        let report = compare_artifacts(&runner, &worker);
        assert_eq!(
            report.compare_kind,
            PinnedPromptCompareKind::RunnerVsWorkerLocalCompare
        );
        assert!(report.content_match);
        assert!(report.overall_match);
        assert!(!report.provenance_match);
        assert!(report.local_gate_eligible);
        assert!(report.local_gate_pass);
        assert!(!report.correctness_implied);
        assert_eq!(report.left_provenance.capture_source, CaptureSource::Runner);
        assert_eq!(
            report.right_provenance.capture_source,
            CaptureSource::Worker
        );
    }

    #[test]
    fn local_gate_reports_content_mismatch_without_implying_correctness() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let runner = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
        );
        let mut worker = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::Worker,
        );
        worker.cases[0].argmax_token_id = 11;
        worker.cases[0].argmax_token_text = "bye".to_string();

        let report = compare_artifacts(&runner, &worker);
        assert_eq!(
            report.compare_kind,
            PinnedPromptCompareKind::RunnerVsWorkerLocalCompare
        );
        assert!(report.local_gate_eligible);
        assert!(!report.content_match);
        assert!(!report.overall_match);
        assert!(!report.local_gate_pass);
        assert!(!report.correctness_implied);
        assert!(!report.cases[0].matches);
    }

    #[test]
    fn compare_report_classifies_other_pairs_as_not_local_gate() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let runner = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
        );
        let mut worker = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::Worker,
        );
        worker.provenance.visible_devices = "0,1".to_string();

        let report = compare_artifacts(&runner, &worker);
        assert_eq!(report.compare_kind, PinnedPromptCompareKind::Other);
        assert!(!report.provenance_match);
        assert!(!report.local_gate_eligible);
        assert!(!report.local_gate_pass);
        assert!(report.content_match);
        assert!(report.overall_match);
        assert!(!report.correctness_implied);
    }

    #[test]
    fn compare_report_classifies_local_vs_official_reference() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let runner = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
        );
        let official = sample_official_artifact(&manifest, case.id.clone(), rendered.token_ids);

        let report = compare_artifacts(&runner, &official);
        assert_eq!(
            report.compare_kind,
            PinnedPromptCompareKind::LocalVsOfficialReferenceCompare
        );
        assert!(!report.provenance_match);
        assert!(!report.local_gate_eligible);
        assert!(!report.local_gate_pass);
        assert!(report.content_match);
        assert!(report.overall_match);
        assert!(report.correctness_implied);
    }

    #[test]
    fn official_reference_compare_can_fail_without_losing_authority_semantics() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let runner = sample_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
        );
        let mut official = sample_official_artifact(&manifest, case.id.clone(), rendered.token_ids);
        official.cases[0].argmax_token_id = 42;
        official.cases[0].argmax_token_text = "mismatch".to_string();

        let report = compare_artifacts(&runner, &official);
        assert_eq!(
            report.compare_kind,
            PinnedPromptCompareKind::LocalVsOfficialReferenceCompare
        );
        assert!(!report.local_gate_eligible);
        assert!(!report.local_gate_pass);
        assert!(!report.content_match);
        assert!(!report.overall_match);
        assert!(report.correctness_implied);
    }

    #[test]
    fn authoritative_requires_official_reference() {
        let manifest = PinnedPromptManifest {
            schema_version: super::default_manifest_schema_version(),
            suite_id: "bad-manifest".to_string(),
            default_top_k: 4,
            default_capture_source: CaptureSource::Runner,
            reference_kind: ReferenceKind::LocalCandidate,
            authority_level: AuthorityLevel::Authoritative,
            cases: vec![],
        };
        let manifest_error = validate_manifest(&manifest).expect_err("reject invalid manifest");
        assert_eq!(
            manifest_error.to_string(),
            "manifest authority_level=authoritative requires reference_kind=official_reference"
        );

        let artifact = PinnedPromptArtifact {
            schema_version: default_artifact_schema_version(),
            suite_id: "bad-artifact".to_string(),
            provenance: PinnedPromptCaptureProvenance {
                model: "synthetic-smoke-model".to_string(),
                capture_source: CaptureSource::Runner,
                reference_kind: ReferenceKind::LocalCandidate,
                authority_level: AuthorityLevel::Authoritative,
                prompt_renderer: super::default_prompt_renderer(),
                visible_devices: "<test>".to_string(),
                max_model_len: 32,
                gpu_memory_utilization: 0.0,
            },
            cases: vec![],
        };
        let artifact_error = validate_artifact(&artifact).expect_err("reject invalid artifact");
        assert_eq!(
            artifact_error.to_string(),
            "artifact authority_level=authoritative requires reference_kind=official_reference"
        );
    }

    #[test]
    fn intermediate_artifact_roundtrips_and_self_compares() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let artifact = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::OfficialTorch,
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            None,
            0.0,
        );
        validate_intermediate_artifact(&artifact).expect("validate synthetic intermediate");
        let roundtrip: PinnedPromptIntermediateArtifact =
            serde_json::from_str(&serde_json::to_string_pretty(&artifact).expect("serialize"))
                .expect("deserialize");
        validate_intermediate_artifact(&roundtrip).expect("validate roundtrip intermediate");

        let report = compare_intermediate_artifacts(&artifact, &roundtrip);
        assert_eq!(
            report.compare_kind,
            PinnedPromptIntermediateCompareKind::SameSourceSelfCompare
        );
        assert!(report.boundary_match);
        assert!(report.layer_idx_match);
        assert!(report.content_match);
        assert!(report.overall_match);
        assert!(report.provenance_match);
        assert!(!report.boundary_authoritative);
        assert!(!report.end_to_end_correctness_implied);
        assert_eq!(report.cases.len(), 1);
        assert!(report.cases[0].matched_within_tolerance);
    }

    #[test]
    fn intermediate_compare_classifies_local_vs_official_within_tolerance() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let local = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            None,
            0.0,
        );
        let official = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::OfficialTorch,
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            None,
            super::default_intermediate_compare_tolerance() / 2.0,
        );

        let report = compare_intermediate_artifacts(&local, &official);
        assert_eq!(
            report.compare_kind,
            PinnedPromptIntermediateCompareKind::LocalVsOfficialReferenceIntermediateCompare
        );
        assert!(report.boundary_match);
        assert!(report.layer_idx_match);
        assert!(!report.provenance_match);
        assert!(report.boundary_authoritative);
        assert!(!report.end_to_end_correctness_implied);
        assert!(report.content_match);
        assert!(report.overall_match);
        assert!(report.cases[0].matched_within_tolerance);
        assert!(report.cases[0].max_abs_diff <= report.tolerance);
    }

    #[test]
    fn intermediate_compare_reports_out_of_tolerance_mismatch() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let local = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            None,
            0.0,
        );
        let official = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::OfficialTorch,
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            None,
            report_tolerance_breach_delta(),
        );

        let report = compare_intermediate_artifacts(&local, &official);
        assert_eq!(
            report.compare_kind,
            PinnedPromptIntermediateCompareKind::LocalVsOfficialReferenceIntermediateCompare
        );
        assert!(report.boundary_authoritative);
        assert!(!report.content_match);
        assert!(!report.overall_match);
        assert!(!report.cases[0].matched_within_tolerance);
        assert!(report.cases[0].max_abs_diff > report.tolerance);
    }

    #[test]
    fn intermediate_artifact_rejects_hidden_size_mismatch() {
        let mut artifact = sample_intermediate_artifact(
            &PinnedPromptManifest {
                schema_version: super::default_manifest_schema_version(),
                suite_id: "bad-intermediate".to_string(),
                default_top_k: 4,
                default_capture_source: CaptureSource::Runner,
                reference_kind: ReferenceKind::OfficialReference,
                authority_level: AuthorityLevel::Authoritative,
                cases: vec![],
            },
            "bad-case".to_string(),
            vec![1, 2, 3],
            CaptureSource::OfficialTorch,
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            None,
            0.0,
        );
        artifact.cases[0].hidden_size += 1;

        let error =
            validate_intermediate_artifact(&artifact).expect_err("reject invalid intermediate");
        assert_eq!(
            error.to_string(),
            "intermediate artifact case 'bad-case' hidden size mismatch: hidden_size=5, values=4"
        );
    }

    #[test]
    fn transformer_layer_output_requires_layer_idx() {
        let error = validate_intermediate_boundary_layer_idx(
            PinnedPromptIntermediateBoundary::TransformerLayerOutput,
            None,
            "intermediate artifact",
        )
        .expect_err("layer output boundary should require layer_idx");
        assert_eq!(
            error.to_string(),
            "intermediate artifact boundary 'transformer_layer_output' requires layer_idx"
        );
    }

    #[test]
    fn final_norm_boundary_rejects_unexpected_layer_idx() {
        let error = validate_intermediate_boundary_layer_idx(
            PinnedPromptIntermediateBoundary::FinalTokenPostFinalNormPreUnembedding,
            Some(23),
            "intermediate artifact",
        )
        .expect_err("final norm boundary should reject layer_idx");
        assert_eq!(
            error.to_string(),
            "intermediate artifact boundary 'final_token_post_final_norm_pre_unembedding' requires layer_idx=None, got 23"
        );
    }

    #[test]
    fn intermediate_compare_distinguishes_boundary_from_layer_idx_match() {
        let fixture = fs::read_to_string(smoke_fixture_path()).expect("read smoke fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse smoke fixture");
        validate_manifest(&manifest).expect("validate smoke fixture");
        let protocol = HarmonyProtocol::gpt_oss().expect("construct harmony protocol");
        let case = manifest.cases.first().expect("smoke fixture case");
        let rendered = protocol
            .render_prompt(&case.messages, case.instructions.as_deref(), &case.tools)
            .expect("render smoke fixture");

        let left = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids.clone(),
            CaptureSource::Runner,
            PinnedPromptIntermediateBoundary::TransformerLayerOutput,
            Some(23),
            0.0,
        );
        let right = sample_intermediate_artifact(
            &manifest,
            case.id.clone(),
            rendered.token_ids,
            CaptureSource::OfficialTorch,
            PinnedPromptIntermediateBoundary::TransformerLayerOutput,
            Some(22),
            0.0,
        );

        let report = compare_intermediate_artifacts(&left, &right);
        assert!(report.boundary_match);
        assert!(!report.layer_idx_match);
        assert_eq!(report.left_layer_idx, Some(23));
        assert_eq!(report.right_layer_idx, Some(22));
        assert_eq!(
            report.compare_kind,
            PinnedPromptIntermediateCompareKind::Other
        );
        assert!(!report.content_match);
        assert!(!report.overall_match);
        assert!(report.cases[0].matched_within_tolerance);
    }

    #[test]
    fn v1_final_norm_intermediate_artifact_defaults_layer_idx_to_none() {
        let artifact_json = serde_json::json!({
            "schema_version": "pinned-prompt-intermediate-artifact/v1",
            "suite_id": "smoke",
            "boundary": "final_token_post_final_norm_pre_unembedding",
            "provenance": {
                "model": "synthetic-smoke-model",
                "capture_source": "runner",
                "reference_kind": "local_candidate",
                "authority_level": "scaffold",
                "prompt_renderer": "harmony_gpt_oss_rs",
                "visible_devices": "<test>",
                "max_model_len": 32,
                "gpu_memory_utilization": 0.0
            },
            "cases": [{
                "id": "smoke-case",
                "input_token_ids": [1, 2, 3],
                "hidden_size": 4,
                "final_token_hidden_f32": [1.0, 2.0, 3.0, 4.0]
            }]
        });

        let artifact: PinnedPromptIntermediateArtifact =
            serde_json::from_value(artifact_json).expect("parse v1 intermediate artifact");
        assert_eq!(artifact.layer_idx, None);
        validate_intermediate_artifact(&artifact).expect("validate v1 intermediate artifact");
    }

    #[test]
    fn runtime_forward_import_rejects_invalid_inputs() {
        let fixture = fs::read_to_string(fixture_dir().join("developer-message.json"))
            .expect("read developer-message fixture");
        let manifest: PinnedPromptManifest =
            serde_json::from_str(&fixture).expect("parse developer-message fixture");
        validate_manifest(&manifest).expect("validate developer-message fixture");

        let rendered_case =
            super::rendered_case_input_token_ids(&manifest, "developer-message-user-smoke")
                .expect("render developer-message case");

        let base_capture = RuntimeForwardIntermediateCapture {
            artifact_kind: "runner_prefill_last_token_hidden_state".to_string(),
            capture_source: CaptureSource::Runner,
            prompt: "<ppp-rendered-token-ids>".to_string(),
            prompt_token_ids: rendered_case.clone(),
            input_token_ids: rendered_case.clone(),
            prompt_token_count: rendered_case.len(),
            final_position: rendered_case.len() - 1,
            restricted_model_path: "/tmp/model".to_string(),
            visible_devices: "1".to_string(),
            seam_identifier: "transformer_layer_output".to_string(),
            layer_idx: 23,
            hidden_state_dim: 4,
            last_token_hidden_state: vec![0.25, -0.5, 1.0, 2.0],
        };

        let wrong_kind = RuntimeForwardIntermediateCapture {
            artifact_kind: "wrong".to_string(),
            ..base_capture.clone()
        };
        let error = import_runtime_forward_intermediate_capture(
            &manifest,
            "developer-message-user-smoke",
            &wrong_kind,
            128,
            0.75,
        )
        .expect_err("reject wrong artifact kind");
        assert_eq!(
            error.to_string(),
            "runtime-forward intermediate import requires artifact_kind='runner_prefill_last_token_hidden_state', got 'wrong'"
        );

        let wrong_seam = RuntimeForwardIntermediateCapture {
            seam_identifier: "wrong".to_string(),
            ..base_capture.clone()
        };
        let error = import_runtime_forward_intermediate_capture(
            &manifest,
            "developer-message-user-smoke",
            &wrong_seam,
            128,
            0.75,
        )
        .expect_err("reject wrong seam identifier");
        assert_eq!(
            error.to_string(),
            "runtime-forward intermediate import requires seam_identifier='transformer_layer_output', got 'wrong'"
        );

        let wrong_tokens = RuntimeForwardIntermediateCapture {
            input_token_ids: vec![999],
            ..base_capture.clone()
        };
        let error = import_runtime_forward_intermediate_capture(
            &manifest,
            "developer-message-user-smoke",
            &wrong_tokens,
            128,
            0.75,
        )
        .expect_err("reject token mismatch");
        assert_eq!(
            error.to_string(),
            "runtime-forward intermediate import input_token_ids mismatch for case 'developer-message-user-smoke'"
        );

        let wrong_hidden_size = RuntimeForwardIntermediateCapture {
            hidden_state_dim: 5,
            ..base_capture
        };
        let error = import_runtime_forward_intermediate_capture(
            &manifest,
            "developer-message-user-smoke",
            &wrong_hidden_size,
            128,
            0.75,
        )
        .expect_err("reject hidden size mismatch");
        assert_eq!(
            error.to_string(),
            "intermediate artifact case 'developer-message-user-smoke' hidden size mismatch: hidden_size=5, values=4"
        );
    }

    fn sample_artifact(
        manifest: &PinnedPromptManifest,
        case_id: String,
        input_token_ids: Vec<TokenId>,
        capture_source: CaptureSource,
    ) -> PinnedPromptArtifact {
        PinnedPromptArtifact {
            schema_version: default_artifact_schema_version(),
            suite_id: manifest.suite_id.clone(),
            provenance: PinnedPromptCaptureProvenance {
                model: "synthetic-smoke-model".to_string(),
                capture_source,
                reference_kind: manifest.reference_kind,
                authority_level: manifest.authority_level,
                prompt_renderer: super::default_prompt_renderer(),
                visible_devices: "<test>".to_string(),
                max_model_len: 32,
                gpu_memory_utilization: 0.0,
            },
            cases: vec![PinnedPromptCaseArtifact {
                id: case_id,
                input_token_ids,
                argmax_token_id: 7,
                argmax_token_text: "hi".to_string(),
                final_position_top_k: vec![
                    PinnedTopLogit {
                        token_id: 7,
                        token_text: "hi".to_string(),
                        logit: 1.5,
                    },
                    PinnedTopLogit {
                        token_id: 9,
                        token_text: " there".to_string(),
                        logit: 0.25,
                    },
                ],
            }],
        }
    }

    fn sample_official_artifact(
        manifest: &PinnedPromptManifest,
        case_id: String,
        input_token_ids: Vec<TokenId>,
    ) -> PinnedPromptArtifact {
        PinnedPromptArtifact {
            schema_version: default_artifact_schema_version(),
            suite_id: manifest.suite_id.clone(),
            provenance: PinnedPromptCaptureProvenance {
                model: "/data/models/openai/gpt-oss-20b/original".to_string(),
                capture_source: CaptureSource::OfficialTorch,
                reference_kind: ReferenceKind::OfficialReference,
                authority_level: AuthorityLevel::Authoritative,
                prompt_renderer: super::default_prompt_renderer(),
                visible_devices: "<official-reference>".to_string(),
                max_model_len: 0,
                gpu_memory_utilization: 0.0,
            },
            cases: vec![PinnedPromptCaseArtifact {
                id: case_id,
                input_token_ids,
                argmax_token_id: 7,
                argmax_token_text: "hi".to_string(),
                final_position_top_k: vec![
                    PinnedTopLogit {
                        token_id: 7,
                        token_text: "hi".to_string(),
                        logit: 1.5,
                    },
                    PinnedTopLogit {
                        token_id: 9,
                        token_text: " there".to_string(),
                        logit: 0.25,
                    },
                ],
            }],
        }
    }

    fn sample_intermediate_artifact(
        manifest: &PinnedPromptManifest,
        case_id: String,
        input_token_ids: Vec<TokenId>,
        capture_source: CaptureSource,
        boundary: PinnedPromptIntermediateBoundary,
        layer_idx: Option<usize>,
        delta: f32,
    ) -> PinnedPromptIntermediateArtifact {
        let (reference_kind, authority_level, model, visible_devices) = match capture_source {
            CaptureSource::OfficialTorch => (
                ReferenceKind::OfficialReference,
                AuthorityLevel::Authoritative,
                "/data/models/openai/gpt-oss-20b/original".to_string(),
                "<official-reference>".to_string(),
            ),
            CaptureSource::Runner | CaptureSource::Worker => (
                manifest.reference_kind,
                manifest.authority_level,
                "synthetic-smoke-model".to_string(),
                "<test>".to_string(),
            ),
        };

        PinnedPromptIntermediateArtifact {
            schema_version: default_intermediate_artifact_schema_version(),
            suite_id: manifest.suite_id.clone(),
            boundary,
            layer_idx,
            provenance: PinnedPromptCaptureProvenance {
                model,
                capture_source,
                reference_kind,
                authority_level,
                prompt_renderer: super::default_prompt_renderer(),
                visible_devices,
                max_model_len: 32,
                gpu_memory_utilization: 0.0,
            },
            cases: vec![PinnedPromptIntermediateCaseArtifact {
                id: case_id,
                input_token_ids,
                hidden_size: 4,
                final_token_hidden_f32: vec![1.0 + delta, 2.0, 3.0, 4.0],
            }],
        }
    }

    fn report_tolerance_breach_delta() -> f32 {
        super::default_intermediate_compare_tolerance() * 2.0
    }
}
