//! Native, batch-one GPT-OSS CPU model runner.
//!
//! Dense weights remain borrowed from memory-mapped SafeTensors shards. Only
//! MXFP4 expert tensors are transformed into the versioned CPU repack cache.
//! Transformer operation boundaries are BF16 with FP32 accumulation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

use half::bf16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_core::types::{SequenceId, TokenId};
use gpt_oss_cpu_kernels::{
    accumulate_mxfp4_bf16_block, DispatchPlan, KernelPath, Kernels, Mxfp4WeightLayout,
    Q8ActivationView, Q8Block, ResidualQ8ActivationView, ResidualQ8Block, QUANT_BLOCK_SIZE,
};

use crate::cpu_repack::{CpuRepackCache, RepackedMxfp4, SourceIdentity};
use crate::cpu_tensor_store::{CpuTensor, CpuTensorStore};
use crate::model_loader::dtype::DType;

const DEFAULT_SWIGLU_ALPHA: f32 = 1.702;
const DEFAULT_SWIGLU_LIMIT: f32 = 7.0;

/// Internal activation-projection strategy for CPU MXFP4 experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CpuExpertProjection {
    Q8,
    ResidualQ8,
    ExactBf16,
}

impl CpuExpertProjection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::ResidualQ8 => "residual-q8",
            Self::ExactBf16 => "exact-bf16",
        }
    }
}

impl Default for CpuExpertProjection {
    fn default() -> Self {
        Self::ResidualQ8
    }
}

impl fmt::Display for CpuExpertProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for CpuExpertProjection {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "q8" => Ok(Self::Q8),
            "residual-q8" | "residual_q8" => Ok(Self::ResidualQ8),
            "exact-bf16" | "exact_bf16" => Ok(Self::ExactBf16),
            _ => Err(format!("unknown CPU expert projection mode '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuModelRunnerOptions {
    pub kernel_path: KernelPath,
    pub threads: usize,
    pub context_cap: usize,
    pub expert_projection: CpuExpertProjection,
}

/// Monotonic version of committed CPU sequence-model state.
///
/// Prepared work records this value and commit rejects the work if another
/// operation reset or advanced the sequence first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CpuStateRevision(u64);

impl CpuStateRevision {
    pub const fn value(self) -> u64 {
        self.0
    }

    fn next(self) -> Result<Self> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or_else(|| LLMError::ModelError("CPU sequence revision overflow".into()))
    }
}

/// One ordered input row in a CPU model step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStepRow {
    pub sequence_id: SequenceId,
    pub token_id: TokenId,
    pub absolute_position: usize,
    pub logits_required: bool,
}

impl CpuStepRow {
    pub const fn new(
        sequence_id: SequenceId,
        token_id: TokenId,
        absolute_position: usize,
        logits_required: bool,
    ) -> Self {
        Self {
            sequence_id,
            token_id,
            absolute_position,
            logits_required,
        }
    }
}

/// Validated ordered rows for one transactional CPU execution.
///
/// Rows for the same sequence may recur, but their absolute positions must be
/// consecutive in batch order. M3 executes these rows sequentially; M2 keeps
/// this contract while changing the implementation to layer-major execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStepBatch {
    rows: Vec<CpuStepRow>,
}

impl CpuStepBatch {
    pub fn new(rows: Vec<CpuStepRow>) -> Result<Self> {
        if rows.is_empty() {
            return Err(LLMError::ConfigError(
                "CPU step batch must contain at least one row".into(),
            ));
        }
        let mut next_positions = HashMap::new();
        for row in &rows {
            match next_positions.entry(row.sequence_id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(row.absolute_position.checked_add(1).ok_or_else(|| {
                        LLMError::ConfigError("CPU step row position overflow".into())
                    })?);
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if row.absolute_position != *entry.get() {
                        return Err(LLMError::ConfigError(format!(
                            "CPU step positions for sequence {} are not consecutive",
                            row.sequence_id
                        )));
                    }
                    *entry.get_mut() = row.absolute_position.checked_add(1).ok_or_else(|| {
                        LLMError::ConfigError("CPU step row position overflow".into())
                    })?;
                }
            }
        }
        Ok(Self { rows })
    }

    pub fn single(row: CpuStepRow) -> Self {
        Self { rows: vec![row] }
    }

    pub fn rows(&self) -> &[CpuStepRow] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

fn default_alpha() -> f32 {
    DEFAULT_SWIGLU_ALPHA
}

fn default_swiglu_limit() -> f32 {
    DEFAULT_SWIGLU_LIMIT
}

#[derive(Debug, Clone, Deserialize)]
pub struct CpuGptOssConfig {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub head_dim: usize,
    pub num_local_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub experts_per_token: Option<usize>,
    pub layer_types: Vec<String>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default)]
    pub attention_bias: bool,
    pub rope_scaling: Option<CpuRopeScaling>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CpuRopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_beta_slow")]
    pub beta_slow: f64,
    #[serde(default)]
    pub truncate: bool,
}

fn default_beta_fast() -> f64 {
    32.0
}

fn default_beta_slow() -> f64 {
    1.0
}

impl CpuGptOssConfig {
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let path = snapshot.join("config.json");
        let mut config: Self = serde_json::from_slice(&std::fs::read(&path)?).map_err(|error| {
            LLMError::ModelError(format!(
                "invalid GPT-OSS config {}: {error}",
                path.display()
            ))
        })?;
        if config.num_experts_per_tok == 0 {
            config.num_experts_per_tok = config.experts_per_token.unwrap_or(0);
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.architectures.first().map(String::as_str) != Some("GptOssForCausalLM") {
            return Err(LLMError::ModelError(
                "CPU backend only supports GptOssForCausalLM".into(),
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers
            || self.hidden_size == 0
            || self.vocab_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_local_experts
            || self.hidden_size % QUANT_BLOCK_SIZE != 0
            || self.intermediate_size % QUANT_BLOCK_SIZE != 0
            || self.head_dim % 2 != 0
            || self.sliding_window == 0
            || self.max_position_embeddings == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.alpha.is_finite()
            || !self.swiglu_limit.is_finite()
            || self.alpha <= 0.0
            || self.swiglu_limit <= 0.0
        {
            return Err(LLMError::ModelError(
                "invalid GPT-OSS CPU model dimensions".into(),
            ));
        }
        if self
            .experts_per_token
            .is_some_and(|alias| alias != self.num_experts_per_tok)
        {
            return Err(LLMError::ModelError(
                "GPT-OSS expert-count aliases disagree".into(),
            ));
        }
        if self
            .rope_scaling
            .as_ref()
            .is_some_and(|rope| rope.rope_type != "yarn")
        {
            return Err(LLMError::ModelError(
                "CPU GPT-OSS only supports YaRN rope scaling".into(),
            ));
        }
        if self.rope_scaling.as_ref().is_some_and(|rope| {
            !rope.factor.is_finite()
                || rope.factor < 1.0
                || rope.original_max_position_embeddings == 0
                || !rope.beta_fast.is_finite()
                || !rope.beta_slow.is_finite()
                || rope.beta_fast <= 0.0
                || rope.beta_slow <= 0.0
        }) {
            return Err(LLMError::ModelError(
                "invalid GPT-OSS YaRN configuration".into(),
            ));
        }
        if self.layer_types.iter().any(|layer| {
            !matches!(
                layer.as_str(),
                "full_attention" | "sliding_attention" | "local_attention"
            )
        }) {
            return Err(LLMError::ModelError(
                "unsupported GPT-OSS attention layer type".into(),
            ));
        }
        Ok(())
    }
}

/// Immutable, model-scale CPU resources shared by every sequence.
pub struct CpuModel {
    config: CpuGptOssConfig,
    store: CpuTensorStore,
    layers: Vec<CpuLayer>,
    final_norm: Vec<f32>,
    kernels: Kernels,
    pool: rayon::ThreadPool,
    rope: YarnRope,
    expert_projection: CpuExpertProjection,
}

/// Batch-one compatibility facade over one shared [`CpuModel`].
pub struct CpuModelRunner {
    model: Arc<CpuModel>,
    state: CpuSequenceModelState,
    execution: CpuExecutionContext,
}

/// Mutable model state owned by exactly one CPU sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuSequenceModelState {
    caches: Vec<CpuKvCache>,
    context_cap: usize,
    position: usize,
    token_history: Vec<TokenId>,
    revision: CpuStateRevision,
    aborted: bool,
}

/// Worker-local reusable storage for transactional CPU execution.
#[derive(Debug, Default)]
pub struct CpuExecutionContext {
    active: bool,
    prepared_rows: usize,
    failure: Option<CpuExecutionFailurePoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuExecutionFailurePoint {
    BeforeStaging,
    AfterLayer(usize),
    BeforeLogits,
    AfterLogits,
}

/// Selected intermediate values from the final token of a CPU prefill.
///
/// Captures are opt-in and intended for offline oracle comparison. Normal
/// serving does not allocate these vectors.
#[derive(Debug, Clone, Serialize)]
pub struct CpuLayerTrace {
    pub layer_index: usize,
    pub input_norm: Vec<f32>,
    pub query_after_rope: Vec<f32>,
    pub key_after_rope: Vec<f32>,
    pub value_projection: Vec<f32>,
    pub attention_context: Vec<f32>,
    pub attention_projection: Vec<f32>,
    pub post_attention_residual: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub routing_weights: Vec<f32>,
    pub experts: Vec<CpuExpertTrace>,
    pub moe_output: Vec<f32>,
    pub layer_output: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuExpertTrace {
    pub rank: usize,
    pub expert_index: usize,
    pub gate_up_projection: Vec<f32>,
    pub swiglu: Vec<f32>,
    pub down_projection: Vec<f32>,
    pub weighted_output: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuTopLogit {
    pub token_id: usize,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuPrefillTrace {
    pub prompt_token_ids: Vec<u32>,
    pub context_token_ids: Vec<u32>,
    pub trace_step: usize,
    pub expert_projection: CpuExpertProjection,
    pub compatibility_kernel_path: String,
    pub dispatch_plan: String,
    pub mxfp4_gemv_kernel: String,
    pub mxfp4_weight_layout: String,
    pub layers: Vec<CpuLayerTrace>,
    pub final_norm: Vec<f32>,
    pub top_logits: Vec<CpuTopLogit>,
}

struct CpuMoeStep {
    router_logits: Vec<f32>,
    selected_experts: Vec<usize>,
    routing_weights: Vec<f32>,
    experts: Vec<CpuExpertTrace>,
    output: Vec<f32>,
}

enum PreparedExpertInput {
    Q8(Vec<Q8Block>),
    ResidualQ8(Vec<ResidualQ8Block>),
    ExactBf16(Vec<bf16>),
}

struct CpuLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    q_weight: String,
    q_bias: Vec<f32>,
    k_weight: String,
    k_bias: Vec<f32>,
    v_weight: String,
    v_bias: Vec<f32>,
    o_weight: String,
    o_bias: Vec<f32>,
    sinks: Vec<f32>,
    router_weight: String,
    router_bias: Vec<f32>,
    gate_up: RepackedMxfp4,
    gate_up_bias: Vec<f32>,
    down: RepackedMxfp4,
    down_bias: Vec<f32>,
    sliding: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuKvCache {
    keys: Vec<bf16>,
    values: Vec<bf16>,
    len: usize,
    token_width: usize,
    capacity: usize,
    start_position: usize,
}

impl CpuKvCache {
    pub fn new(token_width: usize, capacity: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            len: 0,
            token_width,
            capacity,
            start_position: 0,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn start_position(&self) -> usize {
        self.start_position
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn token_width(&self) -> usize {
        self.token_width
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.len = 0;
        self.start_position = 0;
    }

    #[cfg(test)]
    fn append(&mut self, position: usize, key: &[bf16], value: &[bf16]) -> Result<()> {
        if key.len() != self.token_width || value.len() != self.token_width {
            return Err(LLMError::ModelError("invalid CPU KV token width".into()));
        }
        if self.len > 0 && position != self.start_position + self.len {
            return Err(LLMError::ModelError(
                "non-consecutive CPU KV position".into(),
            ));
        }
        self.append_validated(position, key, value);
        Ok(())
    }

    fn append_validated(&mut self, position: usize, key: &[bf16], value: &[bf16]) {
        debug_assert_eq!(key.len(), self.token_width);
        debug_assert_eq!(value.len(), self.token_width);
        debug_assert!(self.len == 0 || position == self.start_position + self.len);
        if self.len == self.capacity {
            self.keys.copy_within(self.token_width.., 0);
            self.values.copy_within(self.token_width.., 0);
            self.keys.truncate((self.len - 1) * self.token_width);
            self.values.truncate((self.len - 1) * self.token_width);
            self.len -= 1;
            self.start_position += 1;
        }
        if self.len == 0 {
            self.start_position = position;
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.len += 1;
    }

    fn key(&self, token: usize, kv_head: usize, head_dim: usize) -> &[bf16] {
        let start = token * self.token_width + kv_head * head_dim;
        &self.keys[start..start + head_dim]
    }

    fn value(&self, token: usize, kv_head: usize, head_dim: usize) -> &[bf16] {
        let start = token * self.token_width + kv_head * head_dim;
        &self.values[start..start + head_dim]
    }
}

impl CpuSequenceModelState {
    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn context_cap(&self) -> usize {
        self.context_cap
    }

    pub const fn revision(&self) -> CpuStateRevision {
        self.revision
    }

    pub const fn is_aborted(&self) -> bool {
        self.aborted
    }

    pub fn token_history(&self) -> &[TokenId] {
        &self.token_history
    }

    pub fn caches(&self) -> &[CpuKvCache] {
        &self.caches
    }

    pub fn reset(&mut self) -> Result<()> {
        for cache in &mut self.caches {
            cache.clear();
        }
        self.position = 0;
        self.token_history.clear();
        self.aborted = false;
        self.revision = self.revision.next()?;
        Ok(())
    }

    pub fn abort(&mut self) -> Result<()> {
        self.aborted = true;
        self.revision = self.revision.next()?;
        Ok(())
    }
}

impl CpuExecutionContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn prepared_rows(&self) -> usize {
        self.prepared_rows
    }

    fn begin(&mut self) -> Result<()> {
        if self.active {
            return Err(LLMError::ModelError(
                "CPU execution context is already active".into(),
            ));
        }
        self.active = true;
        self.prepared_rows = 0;
        Ok(())
    }

    fn finish(&mut self, prepared_rows: usize) {
        self.prepared_rows = prepared_rows;
        self.active = false;
    }

    #[cfg(test)]
    fn inject_failure(&mut self, failure: CpuExecutionFailurePoint) {
        self.failure = Some(failure);
    }
}

#[derive(Debug, Clone)]
struct StagedKvRow {
    position: usize,
    key: Vec<bf16>,
    value: Vec<bf16>,
}

#[derive(Debug)]
struct PreparedSequenceDelta {
    sequence_id: SequenceId,
    expected_revision: CpuStateRevision,
    expected_position: usize,
    tokens: Vec<TokenId>,
    staged_layers: Vec<Vec<StagedKvRow>>,
}

/// Output metadata for one prepared CPU input row.
#[derive(Debug, Clone)]
pub struct PreparedCpuRow {
    pub sequence_id: SequenceId,
    pub token_id: TokenId,
    pub absolute_position: usize,
    logits: Option<Vec<f32>>,
    layer_traces: Vec<CpuLayerTrace>,
    final_norm: Vec<f32>,
}

impl PreparedCpuRow {
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }
}

/// Fully computed CPU step whose model-state effects are not yet visible.
///
/// Dropping or explicitly discarding this value is a no-op. Commit first
/// validates every state revision and shape, then applies all sequence deltas.
#[derive(Debug)]
pub struct PreparedCpuStep {
    rows: Vec<PreparedCpuRow>,
    sequences: Vec<PreparedSequenceDelta>,
}

impl PreparedCpuStep {
    pub fn rows(&self) -> &[PreparedCpuRow] {
        &self.rows
    }

    pub fn discard(self) {}

    pub fn commit(
        self,
        sequences: &mut [(SequenceId, &mut CpuSequenceModelState)],
    ) -> Result<Vec<PreparedCpuRow>> {
        let mut supplied = HashMap::with_capacity(sequences.len());
        for (index, (sequence_id, state)) in sequences.iter().enumerate() {
            if supplied.insert(*sequence_id, index).is_some() {
                return Err(LLMError::ModelError(format!(
                    "duplicate mutable CPU sequence state {sequence_id}"
                )));
            }
            if state.aborted {
                return Err(LLMError::ModelError(format!(
                    "CPU sequence {sequence_id} is aborted"
                )));
            }
        }

        let mut next_revisions = HashMap::with_capacity(self.sequences.len());
        for delta in &self.sequences {
            let Some(&index) = supplied.get(&delta.sequence_id) else {
                return Err(LLMError::ModelError(format!(
                    "missing mutable CPU sequence state {}",
                    delta.sequence_id
                )));
            };
            let state = &sequences[index].1;
            if state.revision != delta.expected_revision
                || state.position != delta.expected_position
            {
                return Err(LLMError::ModelError(format!(
                    "stale prepared CPU step for sequence {}",
                    delta.sequence_id
                )));
            }
            if state.caches.len() != delta.staged_layers.len()
                || delta.tokens.is_empty()
                || delta.expected_position + delta.tokens.len() > state.context_cap
            {
                return Err(LLMError::ModelError(
                    "invalid prepared CPU sequence delta".into(),
                ));
            }
            for (cache, rows) in state.caches.iter().zip(&delta.staged_layers) {
                if rows.len() != delta.tokens.len() {
                    return Err(LLMError::ModelError(
                        "incomplete prepared CPU KV layers".into(),
                    ));
                }
                for (offset, row) in rows.iter().enumerate() {
                    if row.position != delta.expected_position + offset
                        || row.key.len() != cache.token_width
                        || row.value.len() != cache.token_width
                    {
                        return Err(LLMError::ModelError("invalid prepared CPU KV row".into()));
                    }
                }
            }
            next_revisions.insert(delta.sequence_id, state.revision.next()?);
        }

        for delta in self.sequences {
            let index = supplied[&delta.sequence_id];
            let state = &mut sequences[index].1;
            for (cache, rows) in state.caches.iter_mut().zip(delta.staged_layers) {
                for row in rows {
                    cache.append_validated(row.position, &row.key, &row.value);
                }
            }
            state.position += delta.tokens.len();
            state.token_history.extend(delta.tokens);
            state.revision = next_revisions[&delta.sequence_id];
        }
        Ok(self.rows)
    }
}

impl CpuModel {
    /// Load model-scale CPU resources once and return shared ownership.
    pub fn load(
        snapshot: impl AsRef<Path>,
        repack_root: impl AsRef<Path>,
        kernel_path: KernelPath,
        threads: usize,
        expert_projection: CpuExpertProjection,
    ) -> Result<Arc<Self>> {
        if threads == 0 {
            return Err(LLMError::ConfigError(
                "CPU thread count must be non-zero".into(),
            ));
        }
        let snapshot = snapshot.as_ref();
        let config = CpuGptOssConfig::from_snapshot(snapshot)?;
        let store = CpuTensorStore::open(snapshot)?;
        let identity = SourceIdentity::from_store(&store)?;
        let repack = CpuRepackCache::new(repack_root.as_ref(), identity);
        let mut kernels =
            Kernels::new(kernel_path).map_err(|error| LLMError::ConfigError(error.to_string()))?;
        if expert_projection == CpuExpertProjection::ExactBf16 {
            kernels = kernels.with_exact_bf16_mxfp4();
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("gpt-oss-cpu-{index}"))
            .build()
            .map_err(|error| LLMError::ModelError(format!("CPU thread pool: {error}")))?;
        let rope = YarnRope::new(&config)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(load_layer(
                &store,
                &repack,
                &config,
                index,
                kernels.dispatch_plan().mxfp4_weight_layout(),
            )?);
        }
        let final_norm = load_vector_len(&store.tensor("model.norm.weight")?, config.hidden_size)?;
        validate_shape(
            &store.tensor("model.embed_tokens.weight")?,
            &[config.vocab_size, config.hidden_size],
        )?;
        validate_shape(
            &store.tensor("lm_head.weight")?,
            &[config.vocab_size, config.hidden_size],
        )?;

        Ok(Arc::new(Self {
            config,
            store,
            layers,
            final_norm,
            kernels,
            pool,
            rope,
            expert_projection,
        }))
    }

    pub fn config(&self) -> &CpuGptOssConfig {
        &self.config
    }

    pub const fn kernel_path(&self) -> KernelPath {
        self.kernels.path()
    }

    pub const fn kernel_dispatch_plan(&self) -> DispatchPlan {
        self.kernels.dispatch_plan()
    }

    pub const fn expert_projection(&self) -> CpuExpertProjection {
        self.expert_projection
    }

    pub fn new_sequence_state(&self, context_cap: usize) -> Result<CpuSequenceModelState> {
        if context_cap == 0 || context_cap > self.config.max_position_embeddings {
            return Err(LLMError::ConfigError(format!(
                "CPU context cap {context_cap} exceeds checkpoint maximum {}",
                self.config.max_position_embeddings
            )));
        }
        let token_width = self.config.num_key_value_heads * self.config.head_dim;
        let caches = self
            .layers
            .iter()
            .map(|layer| {
                CpuKvCache::new(
                    token_width,
                    if layer.sliding {
                        self.config.sliding_window
                    } else {
                        context_cap
                    },
                )
            })
            .collect();
        Ok(CpuSequenceModelState {
            caches,
            context_cap,
            position: 0,
            token_history: Vec::with_capacity(context_cap),
            revision: CpuStateRevision::default(),
            aborted: false,
        })
    }
}

impl CpuModelRunner {
    pub fn load(
        snapshot: impl AsRef<Path>,
        repack_root: impl AsRef<Path>,
        kernel_path: KernelPath,
        threads: usize,
        context_cap: usize,
    ) -> Result<Self> {
        Self::load_with_options(
            snapshot,
            repack_root,
            CpuModelRunnerOptions {
                kernel_path,
                threads,
                context_cap,
                expert_projection: CpuExpertProjection::default(),
            },
        )
    }

    pub fn load_with_options(
        snapshot: impl AsRef<Path>,
        repack_root: impl AsRef<Path>,
        options: CpuModelRunnerOptions,
    ) -> Result<Self> {
        if options.context_cap == 0 {
            return Err(LLMError::ConfigError(
                "CPU context cap must be non-zero".into(),
            ));
        }
        let model = CpuModel::load(
            snapshot,
            repack_root,
            options.kernel_path,
            options.threads,
            options.expert_projection,
        )?;
        Self::from_model(model, options.context_cap)
    }

    pub fn from_model(model: Arc<CpuModel>, context_cap: usize) -> Result<Self> {
        let state = model.new_sequence_state(context_cap)?;
        Ok(Self {
            model,
            state,
            execution: CpuExecutionContext::new(),
        })
    }

    pub fn model(&self) -> &Arc<CpuModel> {
        &self.model
    }

    pub fn into_parts(self) -> (Arc<CpuModel>, CpuSequenceModelState, CpuExecutionContext) {
        (self.model, self.state, self.execution)
    }

    pub fn config(&self) -> &CpuGptOssConfig {
        self.model.config()
    }

    pub fn kernel_path(&self) -> KernelPath {
        self.model.kernel_path()
    }

    pub fn kernel_dispatch_plan(&self) -> DispatchPlan {
        self.model.kernel_dispatch_plan()
    }

    pub fn expert_projection(&self) -> CpuExpertProjection {
        self.model.expert_projection()
    }

    pub const fn position(&self) -> usize {
        self.state.position()
    }

    pub fn caches(&self) -> &[CpuKvCache] {
        self.state.caches()
    }

    /// Execute one isolated transformer layer for local conformance fixtures.
    /// This does not advance the model-wide decode position.
    pub fn conformance_layer(
        &mut self,
        layer_index: usize,
        hidden: &[bf16],
        position: usize,
    ) -> Result<Vec<bf16>> {
        if layer_index >= self.model.layers.len() || hidden.len() != self.model.config.hidden_size {
            return Err(LLMError::ModelError(
                "invalid isolated CPU layer conformance input".into(),
            ));
        }
        self.state.caches[layer_index].clear();
        let mut staged = Vec::with_capacity(1);
        let (output, _) = self.model.forward_layer_with_trace(
            layer_index,
            hidden,
            position,
            &self.state.caches[layer_index],
            &mut staged,
            false,
        )?;
        for row in staged {
            self.state.caches[layer_index].append_validated(row.position, &row.key, &row.value);
        }
        Ok(output)
    }

    pub fn reset(&mut self) {
        if self.state.reset().is_err() {
            self.state.aborted = true;
        }
    }

    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(LLMError::ModelError("CPU prefill is empty".into()));
        }
        self.reset();
        let mut logits = Vec::new();
        for &token in token_ids {
            logits = self.forward_token(token)?;
        }
        Ok(logits)
    }

    /// Run a normal prefill while capturing selected layer intermediates for
    /// the final prompt token. Generated trace data is never retained by the
    /// serving path.
    pub fn prefill_trace(
        &mut self,
        token_ids: &[u32],
        selected_layers: &[usize],
        top_k: usize,
    ) -> Result<(Vec<f32>, CpuPrefillTrace)> {
        if token_ids.is_empty() {
            return Err(LLMError::ModelError("CPU prefill is empty".into()));
        }
        self.validate_trace_layers(selected_layers)?;
        self.reset();
        for &token in &token_ids[..token_ids.len() - 1] {
            self.forward_token(token)?;
        }
        let (logits, layers, final_norm) =
            self.forward_token_with_trace(token_ids[token_ids.len() - 1], selected_layers)?;
        let trace = self.finish_trace(&logits, layers, final_norm, top_k, 0);
        Ok((logits, trace))
    }

    pub fn decode(&mut self, token_id: u32) -> Result<Vec<f32>> {
        if self.state.position == 0 {
            return Err(LLMError::ModelError(
                "CPU decode requires an existing prefill cache".into(),
            ));
        }
        self.forward_token(token_id)
    }

    /// Decode one context token and capture the resulting logits used to
    /// select generated token `trace_step`.
    pub fn decode_trace(
        &mut self,
        token_id: u32,
        selected_layers: &[usize],
        top_k: usize,
        trace_step: usize,
    ) -> Result<(Vec<f32>, CpuPrefillTrace)> {
        if self.state.position == 0 {
            return Err(LLMError::ModelError(
                "CPU decode trace requires an existing prefill cache".into(),
            ));
        }
        if trace_step == 0 {
            return Err(LLMError::ConfigError(
                "CPU decode trace step must be greater than zero".into(),
            ));
        }
        self.validate_trace_layers(selected_layers)?;
        let (logits, layers, final_norm) =
            self.forward_token_with_trace(token_id, selected_layers)?;
        let trace = self.finish_trace(&logits, layers, final_norm, top_k, trace_step);
        Ok((logits, trace))
    }

    fn validate_trace_layers(&self, selected_layers: &[usize]) -> Result<()> {
        if selected_layers
            .iter()
            .any(|&index| index >= self.model.layers.len())
        {
            return Err(LLMError::ConfigError(
                "CPU trace layer index is out of range".into(),
            ));
        }
        Ok(())
    }

    fn finish_trace(
        &self,
        logits: &[f32],
        layers: Vec<CpuLayerTrace>,
        final_norm: Vec<f32>,
        top_k: usize,
        trace_step: usize,
    ) -> CpuPrefillTrace {
        let dispatch_plan = self.kernel_dispatch_plan();
        CpuPrefillTrace {
            prompt_token_ids: self.state.token_history.clone(),
            context_token_ids: self.state.token_history.clone(),
            trace_step,
            expert_projection: self.model.expert_projection,
            compatibility_kernel_path: self.kernel_path().to_string(),
            dispatch_plan: dispatch_plan.to_string(),
            mxfp4_gemv_kernel: dispatch_plan.mxfp4_gemv().to_string(),
            mxfp4_weight_layout: dispatch_plan.mxfp4_weight_layout().to_string(),
            layers,
            final_norm,
            top_logits: top_logits(logits, top_k),
        }
    }

    fn forward_token(&mut self, token_id: u32) -> Result<Vec<f32>> {
        self.forward_token_with_trace(token_id, &[])
            .map(|(logits, _, _)| logits)
    }

    fn forward_token_with_trace(
        &mut self,
        token_id: u32,
        selected_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<CpuLayerTrace>, Vec<f32>)> {
        let sequence_id = SequenceId(0);
        let batch = CpuStepBatch::single(CpuStepRow::new(
            sequence_id,
            token_id,
            self.state.position,
            true,
        ));
        let prepared = self.model.prepare_step_with_trace(
            &mut self.execution,
            &batch,
            &[(sequence_id, &self.state)],
            Some((0, selected_layers)),
        )?;
        let rows = prepared.commit(&mut [(sequence_id, &mut self.state)])?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| LLMError::ModelError("CPU prepared step returned no row".into()))?;
        let logits = row.logits.ok_or_else(|| {
            LLMError::ModelError("CPU compatibility step returned no logits".into())
        })?;
        Ok((logits, row.layer_traces, row.final_norm))
    }
}

impl CpuModel {
    /// Execute an ordered CPU batch without changing committed sequence state.
    pub fn prepare_step(
        &self,
        execution: &mut CpuExecutionContext,
        batch: &CpuStepBatch,
        sequences: &[(SequenceId, &CpuSequenceModelState)],
    ) -> Result<PreparedCpuStep> {
        self.prepare_step_with_trace(execution, batch, sequences, None)
    }

    fn prepare_step_with_trace(
        &self,
        execution: &mut CpuExecutionContext,
        batch: &CpuStepBatch,
        sequences: &[(SequenceId, &CpuSequenceModelState)],
        trace_request: Option<(usize, &[usize])>,
    ) -> Result<PreparedCpuStep> {
        execution.begin()?;
        let failure = execution.failure.take();
        let result = self.prepare_step_inner(batch, sequences, trace_request, failure);
        execution.finish(result.as_ref().map_or(0, |prepared| prepared.rows.len()));
        result
    }

    fn prepare_step_inner(
        &self,
        batch: &CpuStepBatch,
        sequences: &[(SequenceId, &CpuSequenceModelState)],
        trace_request: Option<(usize, &[usize])>,
        failure: Option<CpuExecutionFailurePoint>,
    ) -> Result<PreparedCpuStep> {
        if failure == Some(CpuExecutionFailurePoint::BeforeStaging) {
            return Err(LLMError::ModelError(
                "injected CPU failure before staging".into(),
            ));
        }
        let mut states = HashMap::with_capacity(sequences.len());
        let mut state_addresses = HashMap::with_capacity(sequences.len());
        for (sequence_id, state) in sequences {
            if states.insert(*sequence_id, *state).is_some() {
                return Err(LLMError::ModelError(format!(
                    "duplicate CPU sequence state {sequence_id}"
                )));
            }
            let address = std::ptr::from_ref(*state) as usize;
            if let Some(other_id) = state_addresses.insert(address, *sequence_id) {
                return Err(LLMError::ModelError(format!(
                    "CPU sequence state is aliased by IDs {other_id} and {sequence_id}"
                )));
            }
            if state.aborted {
                return Err(LLMError::ModelError(format!(
                    "CPU sequence {sequence_id} is aborted"
                )));
            }
            if state.caches.len() != self.layers.len() {
                return Err(LLMError::ModelError(
                    "CPU sequence layer count does not match model".into(),
                ));
            }
        }

        let mut deltas = Vec::<PreparedSequenceDelta>::new();
        let mut delta_indices = HashMap::<SequenceId, usize>::new();
        let mut outputs = Vec::with_capacity(batch.len());
        for (row_index, row) in batch.rows().iter().enumerate() {
            let state = states.get(&row.sequence_id).copied().ok_or_else(|| {
                LLMError::ModelError(format!("missing CPU sequence state {}", row.sequence_id))
            })?;
            let delta_index = *delta_indices.entry(row.sequence_id).or_insert_with(|| {
                let index = deltas.len();
                deltas.push(PreparedSequenceDelta {
                    sequence_id: row.sequence_id,
                    expected_revision: state.revision,
                    expected_position: state.position,
                    tokens: Vec::new(),
                    staged_layers: vec![Vec::new(); self.layers.len()],
                });
                index
            });
            let delta = &mut deltas[delta_index];
            let expected_position = state
                .position
                .checked_add(delta.tokens.len())
                .ok_or_else(|| LLMError::ModelError("CPU sequence position overflow".into()))?;
            if row.absolute_position != expected_position {
                return Err(LLMError::ModelError(format!(
                    "CPU sequence {} expected position {expected_position}, got {}",
                    row.sequence_id, row.absolute_position
                )));
            }
            if expected_position >= state.context_cap {
                return Err(LLMError::ModelError(format!(
                    "CPU context cap {} exceeded",
                    state.context_cap
                )));
            }
            let token = row.token_id as usize;
            if token >= self.config.vocab_size {
                return Err(LLMError::ModelError(format!(
                    "token {token} exceeds vocabulary {}",
                    self.config.vocab_size
                )));
            }

            let embedding = self.store.tensor("model.embed_tokens.weight")?;
            let embedding = embedding.bf16()?;
            let start = token * self.config.hidden_size;
            let mut hidden = embedding[start..start + self.config.hidden_size].to_vec();
            let trace_layers = trace_request
                .filter(|(requested_row, _)| *requested_row == row_index)
                .map_or(&[][..], |(_, layers)| layers);
            let mut layer_traces = Vec::with_capacity(trace_layers.len());
            for layer_index in 0..self.layers.len() {
                let capture = trace_layers.contains(&layer_index);
                let (next_hidden, trace) = self.forward_layer_with_trace(
                    layer_index,
                    &hidden,
                    row.absolute_position,
                    &state.caches[layer_index],
                    &mut delta.staged_layers[layer_index],
                    capture,
                )?;
                hidden = next_hidden;
                if let Some(trace) = trace {
                    layer_traces.push(trace);
                }
                if failure == Some(CpuExecutionFailurePoint::AfterLayer(layer_index)) {
                    return Err(LLMError::ModelError(format!(
                        "injected CPU failure after layer {layer_index}"
                    )));
                }
            }

            let mut final_norm = Vec::new();
            let logits = if row.logits_required || !trace_layers.is_empty() {
                if failure == Some(CpuExecutionFailurePoint::BeforeLogits) {
                    return Err(LLMError::ModelError(
                        "injected CPU failure before logits".into(),
                    ));
                }
                let normalized = self.norm_boundary(&hidden, &self.final_norm)?;
                if !trace_layers.is_empty() {
                    final_norm = bf16_slice_to_f32(&normalized);
                }
                if row.logits_required {
                    let mut logits = self.project_bf16("lm_head.weight", &normalized, None)?;
                    fp32_to_bf16_roundtrip(&mut logits);
                    if logits.iter().any(|value| !value.is_finite()) {
                        return Err(LLMError::ModelError(
                            "CPU model produced non-finite logits".into(),
                        ));
                    }
                    if failure == Some(CpuExecutionFailurePoint::AfterLogits) {
                        return Err(LLMError::ModelError(
                            "injected CPU failure after logits".into(),
                        ));
                    }
                    Some(logits)
                } else {
                    None
                }
            } else {
                None
            };
            delta.tokens.push(row.token_id);
            outputs.push(PreparedCpuRow {
                sequence_id: row.sequence_id,
                token_id: row.token_id,
                absolute_position: row.absolute_position,
                logits,
                layer_traces,
                final_norm,
            });
        }
        Ok(PreparedCpuStep {
            rows: outputs,
            sequences: deltas,
        })
    }

    fn forward_layer_with_trace(
        &self,
        index: usize,
        hidden: &[bf16],
        position: usize,
        cache: &CpuKvCache,
        staged: &mut Vec<StagedKvRow>,
        capture: bool,
    ) -> Result<(Vec<bf16>, Option<CpuLayerTrace>)> {
        let layer = &self.layers[index];
        let normalized = self.norm_boundary(hidden, &layer.input_norm)?;

        let mut q = self.project_bf16(&layer.q_weight, &normalized, Some(&layer.q_bias))?;
        let mut k = self.project_bf16(&layer.k_weight, &normalized, Some(&layer.k_bias))?;
        let v = self.project_bf16(&layer.v_weight, &normalized, Some(&layer.v_bias))?;
        fp32_to_bf16_roundtrip(&mut q);
        fp32_to_bf16_roundtrip(&mut k);
        self.rope
            .apply(&mut q, self.config.num_attention_heads, position)?;
        self.rope
            .apply(&mut k, self.config.num_key_value_heads, position)?;
        fp32_to_bf16_roundtrip(&mut q);
        fp32_to_bf16_roundtrip(&mut k);
        let key = k
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let value = v
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let value_projection = capture.then(|| bf16_slice_to_f32(&value));
        staged.push(StagedKvRow {
            position,
            key,
            value,
        });

        let attention_context = attention_one_staged(
            &q,
            cache,
            staged,
            &layer.sinks,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )?;
        let attention = attention_context
            .iter()
            .copied()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let mut projected = self.project_bf16(&layer.o_weight, &attention, Some(&layer.o_bias))?;
        // Official F.linear returns BF16 here before the residual addition.
        fp32_to_bf16_roundtrip(&mut projected);
        let after_attention = add_residual(hidden, &projected);

        let post_attention_normalized =
            self.norm_boundary(&after_attention, &layer.post_attention_norm)?;
        let moe = self.moe_one(layer, &post_attention_normalized, capture)?;
        let layer_output = add_residual(&after_attention, &moe.output);
        let trace = capture.then(|| CpuLayerTrace {
            layer_index: index,
            input_norm: bf16_slice_to_f32(&normalized),
            query_after_rope: q,
            key_after_rope: k,
            value_projection: value_projection.unwrap_or_default(),
            attention_context,
            attention_projection: projected,
            post_attention_residual: bf16_slice_to_f32(&after_attention),
            router_logits: moe.router_logits,
            selected_experts: moe.selected_experts,
            routing_weights: moe.routing_weights,
            experts: moe.experts,
            moe_output: moe.output,
            layer_output: bf16_slice_to_f32(&layer_output),
        });
        Ok((layer_output, trace))
    }

    fn norm_boundary(&self, input: &[bf16], weight: &[f32]) -> Result<Vec<bf16>> {
        let input = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        self.kernels
            .rms_norm(&input, weight, self.config.rms_norm_eps, &mut output)
            .map_err(kernel_error)?;
        Ok(output.into_iter().map(bf16::from_f32).collect())
    }

    fn project_bf16(
        &self,
        weight_name: &str,
        input: &[bf16],
        bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let tensor = self.store.tensor(weight_name)?;
        let shape = tensor.shape();
        if shape.len() != 2 || shape[1] != input.len() {
            return Err(LLMError::ModelError(format!(
                "CPU projection {weight_name} has shape {shape:?}, input {}",
                input.len()
            )));
        }
        let rows = shape[0];
        if bias.is_some_and(|bias| bias.len() != rows) {
            return Err(LLMError::ModelError(format!(
                "CPU projection {weight_name} bias shape mismatch"
            )));
        }
        let weights = tensor.bf16()?;
        let kernels = self.kernels;
        let mut output = vec![0.0_f32; rows];
        self.pool.install(|| {
            output
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(row, destination)| {
                    let row_start = row * input.len();
                    kernels
                        .bf16_matvec(
                            &weights[row_start..row_start + input.len()],
                            1,
                            input.len(),
                            input,
                            std::slice::from_mut(destination),
                        )
                        .map_err(kernel_error)?;
                    if let Some(bias) = bias {
                        *destination += bias[row];
                    }
                    Ok::<(), LLMError>(())
                })
        })?;
        Ok(output)
    }

    fn moe_one(&self, layer: &CpuLayer, input: &[bf16], capture: bool) -> Result<CpuMoeStep> {
        let mut router =
            self.project_bf16(&layer.router_weight, input, Some(&layer.router_bias))?;
        fp32_to_bf16_roundtrip(&mut router);
        if router.iter().any(|value| !value.is_finite()) {
            return Err(LLMError::ModelError(
                "CPU router produced non-finite logits".into(),
            ));
        }
        let selected = stable_top_k(&router, self.config.num_experts_per_tok);
        let route_logits = selected
            .iter()
            .map(|index| router[*index])
            .collect::<Vec<_>>();
        let route_weights = softmax(&route_logits)
            .into_iter()
            .map(|weight| bf16::from_f32(weight).to_f32())
            .collect::<Vec<_>>();
        let prepared_input = self.prepare_expert_input(input)?;
        let mut output = vec![0.0_f32; self.config.hidden_size];
        let mut expert_traces = Vec::with_capacity(if capture { selected.len() } else { 0 });

        for (rank, &expert) in selected.iter().enumerate() {
            let mut gate_up =
                self.project_mxfp4(&layer.gate_up, expert, &prepared_input, &layer.gate_up_bias)?;
            fp32_to_bf16_roundtrip(&mut gate_up);
            let activated = gpt_oss_swiglu(
                &gate_up,
                self.config.intermediate_size,
                self.config.alpha,
                self.config.swiglu_limit,
            )?;
            let activated = activated
                .into_iter()
                .map(bf16::from_f32)
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            let activated_bf16 = activated
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>();
            let prepared_activated = self.prepare_expert_input(&activated_bf16)?;
            let mut expert_output =
                self.project_mxfp4(&layer.down, expert, &prepared_activated, &layer.down_bias)?;
            fp32_to_bf16_roundtrip(&mut expert_output);
            let route_weight = bf16::from_f32(route_weights[rank]).to_f32();
            let weighted_output = expert_output
                .iter()
                .map(|value| *value * route_weight)
                .collect::<Vec<_>>();
            for (destination, value) in output.iter_mut().zip(&weighted_output) {
                *destination += *value;
            }
            if capture {
                expert_traces.push(CpuExpertTrace {
                    rank,
                    expert_index: expert,
                    gate_up_projection: gate_up,
                    swiglu: activated,
                    down_projection: expert_output,
                    weighted_output,
                });
            }
        }
        // The official expert-weight einsum returns BF16 before the residual
        // addition.
        fp32_to_bf16_roundtrip(&mut output);
        Ok(CpuMoeStep {
            router_logits: router,
            selected_experts: selected,
            routing_weights: route_weights,
            experts: expert_traces,
            output,
        })
    }

    fn prepare_expert_input(&self, input: &[bf16]) -> Result<PreparedExpertInput> {
        match self.expert_projection {
            CpuExpertProjection::Q8 => {
                let input = bf16_slice_to_f32(input);
                Ok(PreparedExpertInput::Q8(
                    self.kernels.quantize_q8(&input).map_err(kernel_error)?,
                ))
            }
            CpuExpertProjection::ResidualQ8 => {
                let input = bf16_slice_to_f32(input);
                Ok(PreparedExpertInput::ResidualQ8(
                    self.kernels
                        .quantize_residual_q8(&input)
                        .map_err(kernel_error)?,
                ))
            }
            CpuExpertProjection::ExactBf16 => Ok(PreparedExpertInput::ExactBf16(input.to_vec())),
        }
    }

    fn project_mxfp4(
        &self,
        weights: &RepackedMxfp4,
        expert: usize,
        input: &PreparedExpertInput,
        bias: &[f32],
    ) -> Result<Vec<f32>> {
        let [experts, rows, blocks] = weights.shape();
        let input_blocks = match input {
            PreparedExpertInput::Q8(input) => input.len(),
            PreparedExpertInput::ResidualQ8(input) => input.len(),
            PreparedExpertInput::ExactBf16(input) => input.len() / QUANT_BLOCK_SIZE,
        };
        if expert >= experts || input_blocks != blocks || bias.len() != experts * rows {
            return Err(LLMError::ModelError(
                "invalid CPU MXFP4 projection dimensions".into(),
            ));
        }
        let kernels = self.kernels;
        let view = weights.expert_view(expert)?;
        let expert_bias = &bias[expert * rows..(expert + 1) * rows];
        let mut output = vec![0.0_f32; rows];
        match input {
            PreparedExpertInput::Q8(input) => self.pool.install(|| {
                output
                    .par_chunks_mut(8)
                    .enumerate()
                    .try_for_each(|(tile, destination)| {
                        kernels
                            .mxfp4_q8_gemv_tile(
                                view,
                                tile * 8,
                                Q8ActivationView::new(input),
                                expert_bias,
                                destination,
                            )
                            .map_err(kernel_error)
                    })
            })?,
            PreparedExpertInput::ResidualQ8(input) => self.pool.install(|| {
                output
                    .par_chunks_mut(8)
                    .enumerate()
                    .try_for_each(|(tile, destination)| {
                        kernels
                            .mxfp4_residual_q8_gemv_tile(
                                view,
                                tile * 8,
                                ResidualQ8ActivationView::new(input),
                                expert_bias,
                                destination,
                            )
                            .map_err(kernel_error)
                    })
            })?,
            PreparedExpertInput::ExactBf16(input) => self.pool.install(|| {
                output
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(row, destination)| {
                        let mut total = expert_bias[row];
                        let mut lanes = [0.0_f32; 16];
                        for (block_index, activation) in
                            input.chunks_exact(QUANT_BLOCK_SIZE).enumerate()
                        {
                            let weight = view.block(row, block_index).map_err(kernel_error)?;
                            let activation: &[bf16; QUANT_BLOCK_SIZE] =
                                activation.try_into().map_err(|_| {
                                    LLMError::ModelError(
                                        "invalid exact BF16 activation block".into(),
                                    )
                                })?;
                            accumulate_mxfp4_bf16_block(&weight, activation, &mut lanes);
                        }
                        total += lanes.into_iter().sum::<f32>();
                        *destination = total;
                        Ok::<(), LLMError>(())
                    })
            })?,
        }
        Ok(output)
    }
}

fn load_layer(
    store: &CpuTensorStore,
    repack: &CpuRepackCache,
    config: &CpuGptOssConfig,
    index: usize,
    mxfp4_layout: Mxfp4WeightLayout,
) -> Result<CpuLayer> {
    let prefix = format!("model.layers.{index}");
    let attention = format!("{prefix}.self_attn");
    let experts = format!("{prefix}.mlp.experts");
    let q_weight = format!("{attention}.q_proj.weight");
    let k_weight = format!("{attention}.k_proj.weight");
    let v_weight = format!("{attention}.v_proj.weight");
    let o_weight = format!("{attention}.o_proj.weight");
    validate_shape(
        &store.tensor(&q_weight)?,
        &[
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        ],
    )?;
    validate_shape(
        &store.tensor(&k_weight)?,
        &[
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        ],
    )?;
    validate_shape(
        &store.tensor(&v_weight)?,
        &[
            config.num_key_value_heads * config.head_dim,
            config.hidden_size,
        ],
    )?;
    validate_shape(
        &store.tensor(&o_weight)?,
        &[
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        ],
    )?;

    let gate_blocks_name = format!("{experts}.gate_up_proj_blocks");
    let gate_scales_name = format!("{experts}.gate_up_proj_scales");
    let down_blocks_name = format!("{experts}.down_proj_blocks");
    let down_scales_name = format!("{experts}.down_proj_scales");
    let gate_up = repack.open_or_create(
        &gate_blocks_name,
        &store.tensor(&gate_blocks_name)?,
        &store.tensor(&gate_scales_name)?,
        mxfp4_layout,
    )?;
    let down = repack.open_or_create(
        &down_blocks_name,
        &store.tensor(&down_blocks_name)?,
        &store.tensor(&down_scales_name)?,
        mxfp4_layout,
    )?;
    if gate_up.shape()
        != [
            config.num_local_experts,
            config.intermediate_size * 2,
            config.hidden_size.div_ceil(QUANT_BLOCK_SIZE),
        ]
        || down.shape()
            != [
                config.num_local_experts,
                config.hidden_size,
                config.intermediate_size.div_ceil(QUANT_BLOCK_SIZE),
            ]
    {
        return Err(LLMError::ModelError(format!(
            "layer {index} has unexpected MXFP4 expert shapes"
        )));
    }

    Ok(CpuLayer {
        input_norm: load_vector_len(
            &store.tensor(&format!("{prefix}.input_layernorm.weight"))?,
            config.hidden_size,
        )?,
        post_attention_norm: load_vector_len(
            &store.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            config.hidden_size,
        )?,
        q_bias: load_optional_bias(
            store,
            &format!("{attention}.q_proj.bias"),
            config.num_attention_heads * config.head_dim,
            config.attention_bias,
        )?,
        q_weight,
        k_bias: load_optional_bias(
            store,
            &format!("{attention}.k_proj.bias"),
            config.num_key_value_heads * config.head_dim,
            config.attention_bias,
        )?,
        k_weight,
        v_bias: load_optional_bias(
            store,
            &format!("{attention}.v_proj.bias"),
            config.num_key_value_heads * config.head_dim,
            config.attention_bias,
        )?,
        v_weight,
        o_bias: load_optional_bias(
            store,
            &format!("{attention}.o_proj.bias"),
            config.hidden_size,
            config.attention_bias,
        )?,
        o_weight,
        sinks: load_vector_len(
            &store.tensor(&format!("{attention}.sinks"))?,
            config.num_attention_heads,
        )?,
        router_weight: {
            let name = format!("{prefix}.mlp.router.weight");
            validate_shape(
                &store.tensor(&name)?,
                &[config.num_local_experts, config.hidden_size],
            )?;
            name
        },
        router_bias: load_vector_len(
            &store.tensor(&format!("{prefix}.mlp.router.bias"))?,
            config.num_local_experts,
        )?,
        gate_up,
        gate_up_bias: load_vector_len(
            &store.tensor(&format!("{experts}.gate_up_proj_bias"))?,
            config.num_local_experts * config.intermediate_size * 2,
        )?,
        down,
        down_bias: load_vector_len(
            &store.tensor(&format!("{experts}.down_proj_bias"))?,
            config.num_local_experts * config.hidden_size,
        )?,
        sliding: matches!(
            config.layer_types[index].as_str(),
            "sliding_attention" | "local_attention"
        ),
    })
}

fn load_optional_bias(
    store: &CpuTensorStore,
    name: &str,
    len: usize,
    required: bool,
) -> Result<Vec<f32>> {
    if store.contains(name) {
        let bias = load_vector(&store.tensor(name)?)?;
        if bias.len() != len {
            return Err(LLMError::ModelError(format!(
                "bias {name} has length {}, expected {len}",
                bias.len()
            )));
        }
        Ok(bias)
    } else if required {
        Err(LLMError::ModelError(format!(
            "missing required bias {name}"
        )))
    } else {
        Ok(vec![0.0; len])
    }
}

fn load_vector(tensor: &CpuTensor<'_>) -> Result<Vec<f32>> {
    match tensor.dtype() {
        DType::F32 => Ok(tensor.f32()?.to_vec()),
        DType::BF16 => Ok(tensor.bf16()?.iter().map(|value| value.to_f32()).collect()),
        DType::F16 => Ok(tensor.f16()?.iter().map(|value| value.to_f32()).collect()),
        dtype => Err(LLMError::ModelError(format!(
            "tensor {} has unsupported vector dtype {dtype}",
            tensor.name()
        ))),
    }
}

fn load_vector_len(tensor: &CpuTensor<'_>, expected: usize) -> Result<Vec<f32>> {
    let values = load_vector(tensor)?;
    if values.len() != expected {
        return Err(LLMError::ModelError(format!(
            "tensor {} has {} elements, expected {expected}",
            tensor.name(),
            values.len()
        )));
    }
    Ok(values)
}

fn validate_shape(tensor: &CpuTensor<'_>, expected: &[usize]) -> Result<()> {
    if tensor.shape() != expected {
        return Err(LLMError::ModelError(format!(
            "tensor {} has shape {:?}, expected {expected:?}",
            tensor.name(),
            tensor.shape()
        )));
    }
    Ok(())
}

fn add_residual(residual: &[bf16], update: &[f32]) -> Vec<bf16> {
    residual
        .iter()
        .zip(update)
        .map(|(residual, update)| bf16::from_f32(residual.to_f32() + update))
        .collect()
}

fn fp32_to_bf16_roundtrip(values: &mut [f32]) {
    for value in values {
        *value = bf16::from_f32(*value).to_f32();
    }
}

fn bf16_slice_to_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn top_logits(logits: &[f32], k: usize) -> Vec<CpuTopLogit> {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices.truncate(k.min(indices.len()));
    indices
        .into_iter()
        .map(|token_id| CpuTopLogit {
            token_id,
            logit: logits[token_id],
        })
        .collect()
}

fn stable_top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices.truncate(k.min(indices.len()));
    indices
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = exponentials.iter().sum::<f32>();
    exponentials
        .into_iter()
        .map(|value| value / denominator)
        .collect()
}

fn gpt_oss_swiglu(
    gate_up: &[f32],
    intermediate: usize,
    alpha: f32,
    limit: f32,
) -> Result<Vec<f32>> {
    // Semantics cross-checked against mistral.rs `gptoss_swiglu` in
    // `mistralrs-core/src/models/gpt_oss.rs` at
    // 8010b6a0578e416120b590ed72fd46ed5f24ee85 (MIT).
    if gate_up.len() != intermediate * 2 {
        return Err(LLMError::ModelError(
            "invalid interleaved GPT-OSS SwiGLU shape".into(),
        ));
    }
    Ok((0..intermediate)
        .map(|index| {
            let gate = gate_up[index * 2].min(limit);
            let up = gate_up[index * 2 + 1].clamp(-limit, limit);
            // The official operator retains the BF16 input dtype at every
            // tensor operation. PyTorch applies the scalar alpha in FP32 and
            // rounds the product, sigmoid, both multiplies, and the add back
            // to BF16.
            let scaled_gate = bf16::from_f32(gate * alpha).to_f32();
            let sigmoid = bf16::from_f32(1.0 / (1.0 + (-scaled_gate).exp())).to_f32();
            let glu = bf16::from_f32(gate * sigmoid).to_f32();
            let linear = bf16::from_f32(up + 1.0).to_f32();
            bf16::from_f32(glu * linear).to_f32()
        })
        .collect())
}

#[cfg(test)]
fn attention_one(
    query: &[f32],
    cache: &CpuKvCache,
    sinks: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    attention_one_staged(query, cache, &[], sinks, num_heads, num_kv_heads, head_dim)
}

fn attention_one_staged(
    query: &[f32],
    cache: &CpuKvCache,
    staged: &[StagedKvRow],
    sinks: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    // Learned sinks participate in the softmax denominator with an implicit
    // zero value. This follows mistral.rs `sinks_attn_cpu` and
    // `softmax_with_sinks` at the pinned MIT revision in the provenance doc.
    if query.len() != num_heads * head_dim
        || sinks.len() != num_heads
        || cache.token_width != num_kv_heads * head_dim
        || staged
            .iter()
            .any(|row| row.key.len() != cache.token_width || row.value.len() != cache.token_width)
    {
        return Err(LLMError::ModelError(
            "invalid CPU attention dimensions".into(),
        ));
    }
    let groups = num_heads / num_kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let combined_len = cache.len + staged.len();
    let visible_len = combined_len.min(cache.capacity);
    let skipped = combined_len - visible_len;
    let mut output = vec![0.0_f32; query.len()];
    for head in 0..num_heads {
        let q = &query[head * head_dim..(head + 1) * head_dim];
        let kv_head = head / groups;
        let mut scores = Vec::with_capacity(visible_len);
        for token in 0..visible_len {
            let combined = skipped + token;
            let key = if combined < cache.len {
                cache.key(combined, kv_head, head_dim)
            } else {
                let row = &staged[combined - cache.len];
                let start = kv_head * head_dim;
                &row.key[start..start + head_dim]
            };
            let dot = q
                .iter()
                .zip(key)
                .map(|(query, key)| *query * key.to_f32())
                .sum::<f32>();
            // The official PyTorch reference executes both the BF16 QK
            // contraction and its in-place scale in BF16. Preserve both
            // rounding boundaries before softmax.
            let dot = bf16::from_f32(dot).to_f32();
            let score = bf16::from_f32(dot * scale).to_f32();
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(sinks[head], f32::max);
        let denominator = (sinks[head] - maximum).exp()
            + scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f32>();
        let destination = &mut output[head * head_dim..(head + 1) * head_dim];
        for (token, score) in scores.into_iter().enumerate() {
            // torch.softmax retains the BF16 input dtype even though its
            // reduction is accumulated in FP32.
            let probability = bf16::from_f32((score - maximum).exp() / denominator).to_f32();
            let combined = skipped + token;
            let value = if combined < cache.len {
                cache.value(combined, kv_head, head_dim)
            } else {
                let row = &staged[combined - cache.len];
                let start = kv_head * head_dim;
                &row.value[start..start + head_dim]
            };
            for (destination, value) in destination.iter_mut().zip(value) {
                *destination += probability * value.to_f32();
            }
        }
        // The final BF16 einsum rounds the attention/value contraction before
        // the output projection.
        for value in destination {
            *value = bf16::from_f32(*value).to_f32();
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct YarnRope {
    inv_freq: Vec<f32>,
    attention_scale: f32,
    head_dim: usize,
}

impl YarnRope {
    fn new(config: &CpuGptOssConfig) -> Result<Self> {
        // Focused port of mistral.rs `GptOssRotaryEmbedding::new` in
        // `mistralrs-core/src/layers.rs` at
        // 8010b6a0578e416120b590ed72fd46ed5f24ee85 (MIT).
        let Some(scaling) = &config.rope_scaling else {
            let inv_freq = (0..config.head_dim)
                .step_by(2)
                .map(|index| {
                    1.0_f32 / (config.rope_theta as f32).powf(index as f32 / config.head_dim as f32)
                })
                .collect();
            return Ok(Self {
                inv_freq,
                attention_scale: 1.0,
                head_dim: config.head_dim,
            });
        };
        let dim = config.head_dim;
        let correction = |rotations: f64| {
            (dim as f64
                * (scaling.original_max_position_embeddings as f64
                    / (rotations * 2.0 * std::f64::consts::PI))
                    .ln())
                / (2.0 * config.rope_theta.ln())
        };
        let mut low = correction(scaling.beta_fast);
        let mut high = correction(scaling.beta_slow);
        if scaling.truncate {
            low = low.floor();
            high = high.ceil();
        }
        low = low.max(0.0);
        high = high.min((dim - 1) as f64);
        let range = (high - low).abs().max(0.001);
        let inv_freq = (0..dim / 2)
            .map(|index| {
                // Match the official PyTorch reference's float32 tensor
                // construction. The correction bounds are Python/f64 scalar
                // values, but every tensor operation rounds to float32.
                let frequency = (config.rope_theta as f32).powf((index * 2) as f32 / dim as f32);
                let extrapolation = 1.0_f32 / frequency;
                let interpolation = 1.0_f32 / (scaling.factor as f32 * frequency);
                let ramp = ((index as f32 - low as f32) / range as f32).clamp(0.0, 1.0);
                interpolation * ramp + extrapolation * (1.0 - ramp)
            })
            .collect();
        Ok(Self {
            inv_freq,
            attention_scale: (0.1 * scaling.factor.ln() + 1.0) as f32,
            head_dim: dim,
        })
    }

    fn apply(&self, values: &mut [f32], heads: usize, position: usize) -> Result<()> {
        if values.len() != heads * self.head_dim {
            return Err(LLMError::ModelError("invalid YaRN tensor shape".into()));
        }
        let half = self.head_dim / 2;
        for head in 0..heads {
            let start = head * self.head_dim;
            for index in 0..half {
                let angle = position as f32 * self.inv_freq[index];
                // The official reference converts cos/sin to the query's
                // BF16 dtype before applying rotary embedding. Preserve each
                // BF16 multiply and add/subtract boundary here.
                let cosine = bf16::from_f32(angle.cos() * self.attention_scale).to_f32();
                let sine = bf16::from_f32(angle.sin() * self.attention_scale).to_f32();
                let left = bf16::from_f32(values[start + index]).to_f32();
                let right = bf16::from_f32(values[start + half + index]).to_f32();
                let left_cosine = bf16::from_f32(left * cosine).to_f32();
                let right_sine = bf16::from_f32(right * sine).to_f32();
                let right_cosine = bf16::from_f32(right * cosine).to_f32();
                let left_sine = bf16::from_f32(left * sine).to_f32();
                values[start + index] = bf16::from_f32(left_cosine - right_sine).to_f32();
                values[start + half + index] = bf16::from_f32(right_cosine + left_sine).to_f32();
            }
        }
        Ok(())
    }
}

fn kernel_error(error: gpt_oss_cpu_kernels::KernelError) -> LLMError {
    LLMError::ModelError(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn cpu_step_batch_preserves_identity_positions_and_logits_flags() {
        let batch = CpuStepBatch::new(vec![
            CpuStepRow::new(SequenceId(7), 11, 3, false),
            CpuStepRow::new(SequenceId(9), 22, 0, true),
            CpuStepRow::new(SequenceId(7), 12, 4, true),
        ])
        .unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.rows()[0].sequence_id, SequenceId(7));
        assert!(!batch.rows()[0].logits_required);
        assert!(batch.rows()[1].logits_required);
        assert_eq!(batch.rows()[2].absolute_position, 4);
    }

    #[test]
    fn cpu_step_batch_rejects_empty_nonconsecutive_and_overflow_rows() {
        assert!(CpuStepBatch::new(Vec::new()).is_err());
        assert!(CpuStepBatch::new(vec![
            CpuStepRow::new(SequenceId(1), 1, 3, false),
            CpuStepRow::new(SequenceId(1), 2, 5, true),
        ])
        .is_err());
        assert!(
            CpuStepBatch::new(vec![CpuStepRow::new(SequenceId(1), 1, usize::MAX, true,)]).is_err()
        );
    }

    #[test]
    fn cpu_state_revisions_advance_monotonically() {
        let revision = CpuStateRevision::default();
        assert_eq!(revision.value(), 0);
        assert_eq!(revision.next().unwrap().value(), 1);
    }

    struct TensorFixture {
        name: String,
        dtype: &'static str,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn bf16_fixture(name: impl Into<String>, shape: &[usize], value: f32) -> TensorFixture {
        let elements = shape.iter().product::<usize>();
        let values = vec![bf16::from_f32(value); elements];
        TensorFixture {
            name: name.into(),
            dtype: "BF16",
            shape: shape.to_vec(),
            bytes: bytemuck::cast_slice(&values).to_vec(),
        }
    }

    fn f32_fixture(name: impl Into<String>, shape: &[usize], value: f32) -> TensorFixture {
        TensorFixture {
            name: name.into(),
            dtype: "F32",
            shape: shape.to_vec(),
            bytes: bytemuck::cast_slice(&vec![value; shape.iter().product()]).to_vec(),
        }
    }

    fn u8_fixture(name: impl Into<String>, shape: &[usize], value: u8) -> TensorFixture {
        TensorFixture {
            name: name.into(),
            dtype: "U8",
            shape: shape.to_vec(),
            bytes: vec![value; shape.iter().product()],
        }
    }

    fn write_fixture_shard(path: &Path, tensors: Vec<TensorFixture>) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for tensor in tensors {
            let start = data.len();
            data.extend_from_slice(&tensor.bytes);
            let end = data.len();
            header.insert(
                tensor.name,
                serde_json::json!({
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                    "data_offsets": [start, end]
                }),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&data).unwrap();
    }

    fn synthetic_snapshot() -> tempfile::TempDir {
        let temp = tempdir().unwrap();
        let config = serde_json::json!({
            "architectures": ["GptOssForCausalLM"],
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "rms_norm_eps": 0.00001,
            "rope_theta": 150000.0,
            "sliding_window": 2,
            "head_dim": 8,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "layer_types": ["sliding_attention", "full_attention"],
            "attention_bias": false,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": false
            }
        });
        std::fs::write(
            temp.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();

        let mut tensors = vec![
            bf16_fixture("model.embed_tokens.weight", &[64, 32], 0.125),
            bf16_fixture("model.norm.weight", &[32], 1.0),
            bf16_fixture("lm_head.weight", &[64, 32], 0.03125),
        ];
        for layer in 0..2 {
            let prefix = format!("model.layers.{layer}");
            tensors.extend([
                bf16_fixture(format!("{prefix}.input_layernorm.weight"), &[32], 1.0),
                bf16_fixture(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    &[32],
                    1.0,
                ),
                bf16_fixture(format!("{prefix}.self_attn.q_proj.weight"), &[16, 32], 0.0),
                bf16_fixture(format!("{prefix}.self_attn.k_proj.weight"), &[8, 32], 0.0),
                bf16_fixture(format!("{prefix}.self_attn.v_proj.weight"), &[8, 32], 0.0),
                bf16_fixture(format!("{prefix}.self_attn.o_proj.weight"), &[32, 16], 0.0),
                f32_fixture(format!("{prefix}.self_attn.sinks"), &[2], 0.0),
                bf16_fixture(format!("{prefix}.mlp.router.weight"), &[2, 32], 0.0),
                f32_fixture(format!("{prefix}.mlp.router.bias"), &[2], 0.0),
                u8_fixture(
                    format!("{prefix}.mlp.experts.gate_up_proj_blocks"),
                    &[2, 64, 1, 16],
                    0,
                ),
                u8_fixture(
                    format!("{prefix}.mlp.experts.gate_up_proj_scales"),
                    &[2, 64, 1],
                    127,
                ),
                f32_fixture(
                    format!("{prefix}.mlp.experts.gate_up_proj_bias"),
                    &[2, 64],
                    0.0,
                ),
                u8_fixture(
                    format!("{prefix}.mlp.experts.down_proj_blocks"),
                    &[2, 32, 1, 16],
                    0,
                ),
                u8_fixture(
                    format!("{prefix}.mlp.experts.down_proj_scales"),
                    &[2, 32, 1],
                    127,
                ),
                f32_fixture(
                    format!("{prefix}.mlp.experts.down_proj_bias"),
                    &[2, 32],
                    0.0,
                ),
            ]);
        }
        write_fixture_shard(&temp.path().join("model.safetensors"), tensors);
        temp
    }

    #[test]
    fn sliding_cache_retains_exact_window() {
        let mut cache = CpuKvCache::new(2, 3);
        for position in 0..5 {
            let values = [bf16::from_f32(position as f32), bf16::ONE];
            cache.append(position, &values, &values).unwrap();
        }
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.start_position(), 2);
        assert_eq!(cache.key(0, 0, 2)[0].to_f32(), 2.0);
    }

    #[test]
    fn gpt_oss_sliding_cache_retains_128_tokens() {
        let mut cache = CpuKvCache::new(1, 128);
        for position in 0..130 {
            let value = [bf16::from_f32(position as f32)];
            cache.append(position, &value, &value).unwrap();
        }
        assert_eq!(cache.len(), 128);
        assert_eq!(cache.start_position(), 2);
        assert_eq!(cache.key(0, 0, 1)[0].to_f32(), 2.0);
        assert_eq!(cache.key(127, 0, 1)[0].to_f32(), 129.0);
    }

    #[test]
    fn learned_sink_only_changes_softmax_denominator() {
        let mut cache = CpuKvCache::new(2, 4);
        cache
            .append(
                0,
                &[bf16::ONE, bf16::ZERO],
                &[bf16::from_f32(4.0), bf16::ZERO],
            )
            .unwrap();
        let without = attention_one(&[1.0, 0.0], &cache, &[-100.0], 1, 1, 2).unwrap();
        let with = attention_one(&[1.0, 0.0], &cache, &[10.0], 1, 1, 2).unwrap();
        assert!((without[0] - 4.0).abs() < 1e-4);
        assert!(with[0] < 0.001);
    }

    #[test]
    fn attention_preserves_official_bf16_operator_boundaries() {
        let mut cache = CpuKvCache::new(2, 3);
        let keys = [[0.5, -0.25], [-0.75, 0.125], [0.375, 0.625]];
        let values = [[1.25, -0.5], [-0.75, 2.0], [0.375, -1.5]];
        for (position, (key, value)) in keys.into_iter().zip(values).enumerate() {
            cache
                .append(
                    position,
                    &key.map(bf16::from_f32),
                    &value.map(bf16::from_f32),
                )
                .unwrap();
        }

        let output = attention_one(&[0.3125, -0.6875], &cache, &[0.2], 1, 1, 2).unwrap();
        assert_eq!(
            output
                .into_iter()
                .map(|value| bf16::from_f32(value).to_bits())
                .collect::<Vec<_>>(),
            vec![16032, 48494]
        );
    }

    #[test]
    fn full_and_sliding_attention_share_current_token_semantics() {
        let mut full = CpuKvCache::new(2, 8);
        let mut sliding = CpuKvCache::new(2, 2);
        for position in 0..2 {
            let key = [bf16::from_f32(position as f32), bf16::ONE];
            let value = [bf16::from_f32(position as f32 + 1.0), bf16::ZERO];
            full.append(position, &key, &value).unwrap();
            sliding.append(position, &key, &value).unwrap();
        }
        let q = [0.25, 0.5];
        assert_eq!(
            attention_one(&q, &full, &[0.0], 1, 1, 2).unwrap(),
            attention_one(&q, &sliding, &[0.0], 1, 1, 2).unwrap()
        );
    }

    #[test]
    fn yarn_position_zero_applies_attention_scale() {
        let mut config = CpuGptOssConfig {
            architectures: vec!["GptOssForCausalLM".into()],
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 150000.0,
            sliding_window: 2,
            head_dim: 4,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            experts_per_token: Some(1),
            layer_types: vec!["full_attention".into()],
            alpha: 1.702,
            swiglu_limit: 7.0,
            attention_bias: false,
            rope_scaling: Some(CpuRopeScaling {
                rope_type: "yarn".into(),
                factor: 32.0,
                original_max_position_embeddings: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                truncate: false,
            }),
        };
        let rope = YarnRope::new(&config).unwrap();
        let mut values = [1.0, 2.0, 3.0, 4.0];
        rope.apply(&mut values, 1, 0).unwrap();
        let scale = bf16::from_f32(0.1 * 32_f32.ln() + 1.0).to_f32();
        assert_eq!(values[0], bf16::from_f32(scale).to_f32());
        assert_eq!(values[3], bf16::from_f32(4.0 * scale).to_f32());

        config.head_dim = 64;
        let rope = YarnRope::new(&config).unwrap();
        let mut values = (0..64)
            .map(|index| (index as f32 - 32.0) * 0.03125)
            .collect::<Vec<_>>();
        rope.apply(&mut values, 1, 121).unwrap();
        let expected_bits = [
            15750, 15856, 48986, 16004, 16128, 49040, 49038, 16200, 49012, 16252, 16047, 48923,
            49010, 49014, 48994, 48970, 48950, 48932, 48919, 48909, 48897, 48877, 48855, 48834,
            48812, 48790, 48769, 48727, 48684, 48641, 48556, 48428, 49068, 49063, 49008, 49050,
            49035, 16012, 48692, 48964, 16112, 16042, 49013, 48971, 48780, 15876, 16066, 16137,
            16162, 16180, 16193, 16204, 16215, 16226, 16236, 16247, 16257, 16262, 16268, 16273,
            16278, 16284, 16289, 16295,
        ];
        assert_eq!(
            values
                .iter()
                .map(|value| bf16::from_f32(*value).to_bits())
                .collect::<Vec<_>>(),
            expected_bits
        );
    }

    #[test]
    fn interleaved_swiglu_matches_scalar_formula() {
        let output = gpt_oss_swiglu(&[2.0, 3.0, -1.0, 8.0], 2, 1.702, 7.0).unwrap();
        assert_eq!(output, vec![7.75, -1.234375]);
    }

    #[test]
    fn stable_router_ties_prefer_lower_expert_index() {
        assert_eq!(stable_top_k(&[1.0, 2.0, 2.0, 0.0], 2), vec![1, 2]);
    }

    #[test]
    fn synthetic_full_forward_preserves_prefill_decode_continuity() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let mut incremental =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        incremental.prefill(&[1, 2]).unwrap();
        let decoded = incremental.decode(3).unwrap();
        assert_eq!(incremental.position(), 3);
        assert_eq!(incremental.caches()[0].len(), 2);
        assert_eq!(incremental.caches()[0].start_position(), 1);
        assert_eq!(incremental.caches()[1].len(), 3);

        let mut full =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        let expected = full.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(decoded, expected);
        assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn opt_in_prefill_trace_preserves_logits_and_selected_layers() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let mut traced =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        let (traced_logits, trace) = traced.prefill_trace(&[1, 2, 3], &[0, 1], 3).unwrap();

        let mut plain =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        let plain_logits = plain.prefill(&[1, 2, 3]).unwrap();
        assert_eq!(traced_logits, plain_logits);
        assert_eq!(trace.layers.len(), 2);
        assert_eq!(trace.layers[0].layer_index, 0);
        assert_eq!(trace.layers[1].layer_index, 1);
        assert_eq!(trace.trace_step, 0);
        assert_eq!(trace.context_token_ids, vec![1, 2, 3]);
        assert_eq!(trace.expert_projection, CpuExpertProjection::ResidualQ8);
        assert_eq!(trace.final_norm.len(), 32);
        assert_eq!(trace.top_logits.len(), 3);
        assert!(trace.dispatch_plan.contains("mxfp4_q8_dot=scalar"));
    }

    #[test]
    fn decode_trace_step_captures_selecting_context_without_changing_logits() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let mut traced =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        traced.prefill(&[1, 2]).unwrap();
        let (traced_logits, trace) = traced.decode_trace(3, &[0], 4, 1).unwrap();

        let mut plain =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        plain.prefill(&[1, 2]).unwrap();
        let plain_logits = plain.decode(3).unwrap();

        assert_eq!(traced_logits, plain_logits);
        assert_eq!(trace.trace_step, 1);
        assert_eq!(trace.prompt_token_ids, vec![1, 2, 3]);
        assert_eq!(trace.context_token_ids, vec![1, 2, 3]);
        assert_eq!(trace.layers.len(), 1);
        assert_eq!(trace.layers[0].experts.len(), 1);
    }

    #[test]
    fn decode_trace_step_six_captures_six_generated_context_tokens() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let mut traced =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        traced.prefill(&[1]).unwrap();
        for token in 2..=6 {
            traced.decode(token).unwrap();
        }
        let (traced_logits, trace) = traced.decode_trace(7, &[1], 2, 6).unwrap();

        let mut plain =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        plain.prefill(&[1]).unwrap();
        let mut plain_logits = Vec::new();
        for token in 2..=7 {
            plain_logits = plain.decode(token).unwrap();
        }

        assert_eq!(traced_logits, plain_logits);
        assert_eq!(trace.trace_step, 6);
        assert_eq!(trace.context_token_ids, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn trace_rejects_invalid_layer_and_decode_step_zero() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let mut runner =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        assert!(runner.prefill_trace(&[1], &[2], 1).is_err());
        runner.prefill(&[1]).unwrap();
        assert!(runner.decode_trace(2, &[0], 1, 0).is_err());
    }

    #[test]
    fn every_expert_projection_mode_is_traceable_and_deterministic() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        for expert_projection in [
            CpuExpertProjection::Q8,
            CpuExpertProjection::ResidualQ8,
            CpuExpertProjection::ExactBf16,
        ] {
            let mut runner = CpuModelRunner::load_with_options(
                snapshot.path(),
                &cache,
                CpuModelRunnerOptions {
                    kernel_path: KernelPath::Scalar,
                    threads: 2,
                    context_cap: 16,
                    expert_projection,
                },
            )
            .unwrap();
            let (traced, trace) = runner.prefill_trace(&[1, 2], &[0], 2).unwrap();

            let mut plain = CpuModelRunner::load_with_options(
                snapshot.path(),
                &cache,
                CpuModelRunnerOptions {
                    kernel_path: KernelPath::Scalar,
                    threads: 2,
                    context_cap: 16,
                    expert_projection,
                },
            )
            .unwrap();
            assert_eq!(traced, plain.prefill(&[1, 2]).unwrap());
            assert_eq!(trace.expert_projection, expert_projection);
            if expert_projection == CpuExpertProjection::ExactBf16 {
                assert_eq!(trace.mxfp4_gemv_kernel, "exact-bf16-row");
                assert_eq!(trace.mxfp4_weight_layout, "CanonicalAdjacentV1");
                assert!(trace.dispatch_plan.contains("mxfp4_gemv=exact-bf16-row"));
                assert!(trace
                    .dispatch_plan
                    .contains("mxfp4_layout=CanonicalAdjacentV1"));
            }
        }
    }

    #[test]
    fn x8_runner_matches_scalar_with_and_without_trace() {
        if Kernels::new(KernelPath::Avx2).is_err() {
            return;
        }
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let mut scalar =
            CpuModelRunner::load(snapshot.path(), &cache, KernelPath::Scalar, 2, 16).unwrap();
        let expected = scalar.prefill(&[1, 2, 3]).unwrap();

        for (path, expected_kernel) in [
            (KernelPath::Avx2, "avx2-x8"),
            (KernelPath::Auto, "avx2-x8"),
            (KernelPath::Avx512Vnni, "avx512-vnni-x8"),
        ] {
            if Kernels::new(path).is_err() {
                continue;
            }
            let mut traced = CpuModelRunner::load(snapshot.path(), &cache, path, 2, 16).unwrap();
            let (actual, trace) = traced.prefill_trace(&[1, 2, 3], &[0, 1], 3).unwrap();
            assert_eq!(actual, expected, "{path} trace changed logits");
            assert_eq!(trace.mxfp4_gemv_kernel, expected_kernel);
            assert_eq!(trace.mxfp4_weight_layout, "InterleavedSplitX8V2");
            assert!(trace
                .dispatch_plan
                .contains(&format!("mxfp4_gemv={expected_kernel}")));
            assert!(trace
                .dispatch_plan
                .contains("mxfp4_layout=InterleavedSplitX8V2"));

            let mut plain = CpuModelRunner::load(snapshot.path(), &cache, path, 2, 16).unwrap();
            assert_eq!(plain.prefill(&[1, 2, 3]).unwrap(), actual);
        }
    }

    #[test]
    fn compatibility_runners_share_one_immutable_model_mapping() {
        let snapshot = synthetic_snapshot();
        let cache = snapshot.path().join("repack");
        let model = CpuModel::load(
            snapshot.path(),
            &cache,
            KernelPath::Scalar,
            2,
            CpuExpertProjection::default(),
        )
        .unwrap();
        let first = CpuModelRunner::from_model(model.clone(), 16).unwrap();
        let second = CpuModelRunner::from_model(model.clone(), 16).unwrap();

        assert!(Arc::ptr_eq(first.model(), second.model()));
        assert_eq!(Arc::strong_count(&model), 3);
        assert_eq!(first.position(), 0);
        assert_eq!(second.position(), 0);
        assert_eq!(first.caches().len(), second.caches().len());
    }

    #[test]
    fn prepared_steps_discard_on_drop_and_reject_stale_commit() {
        let snapshot = synthetic_snapshot();
        let model = CpuModel::load(
            snapshot.path(),
            snapshot.path().join("repack"),
            KernelPath::Scalar,
            2,
            CpuExpertProjection::default(),
        )
        .unwrap();
        let mut state = model.new_sequence_state(16).unwrap();
        let baseline = state.clone();
        let batch = CpuStepBatch::single(CpuStepRow::new(SequenceId(1), 3, 0, true));
        let mut execution = CpuExecutionContext::new();

        let dropped = model
            .prepare_step(&mut execution, &batch, &[(SequenceId(1), &state)])
            .unwrap();
        assert!(dropped.rows()[0].logits().is_some());
        drop(dropped);
        assert_eq!(state, baseline);

        let first = model
            .prepare_step(&mut execution, &batch, &[(SequenceId(1), &state)])
            .unwrap();
        let stale = model
            .prepare_step(&mut execution, &batch, &[(SequenceId(1), &state)])
            .unwrap();
        first.commit(&mut [(SequenceId(1), &mut state)]).unwrap();
        let committed = state.clone();
        assert!(stale.commit(&mut [(SequenceId(1), &mut state)]).is_err());
        assert_eq!(state, committed);
        assert_eq!(state.position(), 1);
        assert_eq!(state.revision().value(), 1);
    }

    #[test]
    fn shared_model_executes_independent_interleaved_sequences() {
        let snapshot = synthetic_snapshot();
        let model = CpuModel::load(
            snapshot.path(),
            snapshot.path().join("repack"),
            KernelPath::Scalar,
            2,
            CpuExpertProjection::default(),
        )
        .unwrap();
        let mut first = model.new_sequence_state(16).unwrap();
        let mut second = model.new_sequence_state(16).unwrap();
        let batch = CpuStepBatch::new(vec![
            CpuStepRow::new(SequenceId(1), 1, 0, false),
            CpuStepRow::new(SequenceId(2), 2, 0, false),
            CpuStepRow::new(SequenceId(1), 3, 1, true),
            CpuStepRow::new(SequenceId(2), 4, 1, true),
        ])
        .unwrap();
        let mut execution = CpuExecutionContext::new();
        let prepared = model
            .prepare_step(
                &mut execution,
                &batch,
                &[(SequenceId(1), &first), (SequenceId(2), &second)],
            )
            .unwrap();
        assert!(prepared.rows()[0].logits().is_none());
        assert!(prepared.rows()[1].logits().is_none());
        let first_logits = prepared.rows()[2].logits().unwrap().to_vec();
        let second_logits = prepared.rows()[3].logits().unwrap().to_vec();
        prepared
            .commit(&mut [(SequenceId(1), &mut first), (SequenceId(2), &mut second)])
            .unwrap();

        let mut isolated_first = CpuModelRunner::from_model(model.clone(), 16).unwrap();
        let mut isolated_second = CpuModelRunner::from_model(model.clone(), 16).unwrap();
        assert_eq!(first_logits, isolated_first.prefill(&[1, 3]).unwrap());
        assert_eq!(second_logits, isolated_second.prefill(&[2, 4]).unwrap());
        assert_eq!(first.token_history(), &[1, 3]);
        assert_eq!(second.token_history(), &[2, 4]);
        assert_eq!(first.position(), 2);
        assert_eq!(second.position(), 2);
    }

    #[test]
    fn injected_failures_leave_full_and_sliding_state_unchanged() {
        let snapshot = synthetic_snapshot();
        let model = CpuModel::load(
            snapshot.path(),
            snapshot.path().join("repack"),
            KernelPath::Scalar,
            2,
            CpuExpertProjection::default(),
        )
        .unwrap();
        let mut state = model.new_sequence_state(16).unwrap();
        let mut execution = CpuExecutionContext::new();
        let initial = CpuStepBatch::new(vec![
            CpuStepRow::new(SequenceId(1), 1, 0, false),
            CpuStepRow::new(SequenceId(1), 2, 1, true),
        ])
        .unwrap();
        model
            .prepare_step(&mut execution, &initial, &[(SequenceId(1), &state)])
            .unwrap()
            .commit(&mut [(SequenceId(1), &mut state)])
            .unwrap();
        assert_eq!(state.caches()[0].len(), 2);
        assert_eq!(state.caches()[1].len(), 2);

        for failure in [
            CpuExecutionFailurePoint::BeforeStaging,
            CpuExecutionFailurePoint::AfterLayer(0),
            CpuExecutionFailurePoint::AfterLayer(1),
            CpuExecutionFailurePoint::BeforeLogits,
            CpuExecutionFailurePoint::AfterLogits,
        ] {
            let baseline = state.clone();
            execution.inject_failure(failure);
            let batch = CpuStepBatch::single(CpuStepRow::new(SequenceId(1), 3, 2, true));
            assert!(model
                .prepare_step(&mut execution, &batch, &[(SequenceId(1), &state)])
                .is_err());
            assert_eq!(state, baseline, "failure={failure:?}");
        }

        let extension = CpuStepBatch::new(vec![
            CpuStepRow::new(SequenceId(1), 3, 2, false),
            CpuStepRow::new(SequenceId(1), 4, 3, true),
        ])
        .unwrap();
        model
            .prepare_step(&mut execution, &extension, &[(SequenceId(1), &state)])
            .unwrap()
            .commit(&mut [(SequenceId(1), &mut state)])
            .unwrap();
        assert_eq!(state.caches()[0].len(), 2);
        assert_eq!(state.caches()[0].start_position(), 2);
        assert_eq!(state.caches()[1].len(), 4);
        assert_eq!(state.caches()[1].start_position(), 0);
    }

    #[test]
    fn sequence_reset_abort_and_alias_validation_are_explicit() {
        let snapshot = synthetic_snapshot();
        let model = CpuModel::load(
            snapshot.path(),
            snapshot.path().join("repack"),
            KernelPath::Scalar,
            2,
            CpuExpertProjection::default(),
        )
        .unwrap();
        let mut state = model.new_sequence_state(16).unwrap();
        let mut execution = CpuExecutionContext::new();
        let batch = CpuStepBatch::single(CpuStepRow::new(SequenceId(1), 1, 0, true));
        assert!(model
            .prepare_step(
                &mut execution,
                &batch,
                &[(SequenceId(1), &state), (SequenceId(2), &state)],
            )
            .is_err());
        state.abort().unwrap();
        assert!(model
            .prepare_step(&mut execution, &batch, &[(SequenceId(1), &state)])
            .is_err());
        state.reset().unwrap();
        assert!(!state.is_aborted());
        assert_eq!(state.position(), 0);
        assert!(state.caches().iter().all(|cache| cache.len() == 0));
        model
            .prepare_step(&mut execution, &batch, &[(SequenceId(1), &state)])
            .unwrap()
            .discard();
        assert_eq!(state.position(), 0);
    }

    #[test]
    fn context_cap_cannot_exceed_checkpoint_limit() {
        let snapshot = synthetic_snapshot();
        let error = match CpuModelRunner::load(
            snapshot.path(),
            snapshot.path().join("repack"),
            KernelPath::Scalar,
            1,
            33,
        ) {
            Ok(_) => panic!("oversized context unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("checkpoint maximum 32"));
    }
}
