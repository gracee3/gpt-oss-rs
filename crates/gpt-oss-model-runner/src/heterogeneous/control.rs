//! Bounded all-layer GPU0 owner shell for the H7 retained-continuation control.
//!
//! This is deliberately a serial decode-shaped correctness adapter. It reuses
//! H3 resident dense weights, H2 selected experts, H4 routing/relay, and H5
//! reduction. It does not expose a general model runner or a grouped-prefill
//! fallback.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaSlice, CudaStream};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::device::{list_devices, resolve_stable_device, StableCudaDeviceId};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::kernel_loader::{compiled_ptx_dir, KernelLoader};
use gpt_oss_gpu::pinned_memory::BoundedPinnedLease;
use sha2::{Digest, Sha256};

use crate::cpu_runner::CpuGptOssConfig;
use crate::model_loader::owner_selective::OwnerSelectiveModel;

use super::contract::{GptOssPhase, GPT_OSS_HIDDEN_SIZE};
use super::cpu_expert::CpuX8SelectedExpertWorker;
#[cfg(feature = "heterogeneous-test-faults")]
use super::cuda_expert::SelectedExpertInjectedFault;
use super::cuda_expert::{
    CudaSelectedExpertResultSlot, OwnedSelectedExpertFailure, PendingOwnedSelectedExpert,
    SelectedExpertFirstDivergenceTrace, SelectedExpertTraceStorage,
};
use super::layer::{
    dense, launch_append_kv, launch_attention, launch_embedding, launch_projection,
    launch_residual, launch_rms_norm, launch_rope, rope_tables, APPEND_KV, ATTENTION, EMBEDDING,
    HEAD_DIM, KV_WIDTH, MAX_VISIBLE_TOKENS, MODULE, NUM_HEADS, NUM_KV_HEADS, PROJECTION, Q_WIDTH,
    RESIDUAL, RMS_NORM, ROPE, ROPE_PAIRS,
};
use super::packing::{pack_routes_bounded, PackedDispatchPlan, PackedDispatchRoute};
use super::placement::{ExpertOwner, GptOssExpertKey};
use super::reduction::{
    CudaRankOrderedReducer, PreparedRankOrderedReduction, RankOrderedReductionExecution,
};
use super::relay::{
    pack_remote_inputs, CudaResultRelay, RelayPinnedPoolStats, RelayPinnedPools,
    RelayPinnedReservation,
};
use super::router::{CudaExactRouter, ExactRouterExecution};
use super::{CanonicalRouteContract, ExactRouterWeightsView, ExpertResultDescriptor};

const MAX_CONTROL_TOKENS: usize = 96;

pub fn heterogeneous_control_shell_device_bytes(
    num_layers: usize,
    vocab_size: usize,
    context_cap: usize,
) -> Result<usize> {
    let query_values = Q_WIDTH
        .checked_mul(2)
        .ok_or_else(|| LLMError::ModelError("control query bytes overflow".into()))?;
    let kv_temporary_values = KV_WIDTH
        .checked_mul(2)
        .ok_or_else(|| LLMError::ModelError("control K/V temporary bytes overflow".into()))?;
    let rope_values = ROPE_PAIRS
        .checked_mul(2)
        .ok_or_else(|| LLMError::ModelError("control RoPE bytes overflow".into()))?;
    let vocabulary_values = vocab_size
        .checked_mul(2)
        .ok_or_else(|| LLMError::ModelError("control vocabulary bytes overflow".into()))?;
    let fixed_values = GPT_OSS_HIDDEN_SIZE
        .checked_mul(6)
        .and_then(|values| values.checked_add(query_values))
        .and_then(|values| values.checked_add(kv_temporary_values))
        .and_then(|values| values.checked_add(rope_values))
        .and_then(|values| values.checked_add(vocabulary_values))
        .ok_or_else(|| LLMError::ModelError("control shell fixed bytes overflow".into()))?;
    let kv_values = num_layers
        .checked_mul(context_cap)
        .and_then(|values| values.checked_mul(KV_WIDTH))
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| LLMError::ModelError("control shell K/V bytes overflow".into()))?;
    fixed_values
        .checked_add(kv_values)
        .and_then(|values| values.checked_mul(size_of::<u16>()))
        .ok_or_else(|| LLMError::ModelError("control shell device bytes overflow".into()))
}

/// A routed layer owns all five pinned leases until it reaches an explicit
/// proven-drained release. Any early return conservatively retains the whole
/// reservation for process lifetime; this prevents Rust unwinding from
/// returning a lease to the warmed pool while an unclassified CUDA path could
/// still name it.
struct FailClosedRelayReservation(Option<RelayPinnedReservation>);

impl FailClosedRelayReservation {
    const fn new(reservation: RelayPinnedReservation) -> Self {
        Self(Some(reservation))
    }

    fn take(&mut self) -> RelayPinnedReservation {
        self.0.take().expect("fail-closed relay reservation")
    }

    fn replace(&mut self, reservation: RelayPinnedReservation) {
        debug_assert!(self.0.is_none());
        self.0 = Some(reservation);
    }

    fn router_leases(&mut self) -> (&mut BoundedPinnedLease<u16>, &mut BoundedPinnedLease<u8>) {
        let reservation = self.0.as_mut().expect("fail-closed relay reservation");
        (
            &mut reservation.source_activation,
            &mut reservation.route_descriptors,
        )
    }

    fn packing_leases(&mut self) -> (&BoundedPinnedLease<u16>, &mut BoundedPinnedLease<u16>) {
        let reservation = self.0.as_mut().expect("fail-closed relay reservation");
        (
            &reservation.source_activation,
            &mut reservation.remote_gpu_input,
        )
    }

    fn worker_leases(
        &mut self,
    ) -> (
        &BoundedPinnedLease<u16>,
        &BoundedPinnedLease<u16>,
        &mut BoundedPinnedLease<u16>,
        &mut BoundedPinnedLease<u16>,
    ) {
        let reservation = self.0.as_mut().expect("fail-closed relay reservation");
        (
            &reservation.source_activation,
            &reservation.remote_gpu_input,
            &mut reservation.remote_gpu_result,
            &mut reservation.cpu_result,
        )
    }

    fn release_drained(mut self) -> Result<()> {
        self.take().release_drained()
    }
}

impl Deref for FailClosedRelayReservation {
    type Target = RelayPinnedReservation;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("fail-closed relay reservation")
    }
}

impl DerefMut for FailClosedRelayReservation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("fail-closed relay reservation")
    }
}

impl Drop for FailClosedRelayReservation {
    fn drop(&mut self) {
        if let Some(reservation) = self.0.take() {
            std::mem::forget(reservation);
        }
    }
}

/// Owns every reusable per-layer allocation from route admission until either
/// a proven terminal returns it to the runtime pools or an uncertain drain
/// deliberately retains it for process lifetime.
struct ControlLayerResources {
    reservation: FailClosedRelayReservation,
    local_slots: [Option<CudaSelectedExpertResultSlot>; 4],
    remote_slots: [Option<CudaSelectedExpertResultSlot>; 4],
}

impl ControlLayerResources {
    fn new(reservation: RelayPinnedReservation) -> Self {
        Self {
            reservation: FailClosedRelayReservation::new(reservation),
            local_slots: std::array::from_fn(|_| None),
            remote_slots: std::array::from_fn(|_| None),
        }
    }

    fn restore_slots(
        &mut self,
        local_pool: &mut [Option<CudaSelectedExpertResultSlot>; 4],
        remote_pool: &mut [Option<CudaSelectedExpertResultSlot>; 4],
    ) -> Result<()> {
        for (index, slot) in self.local_slots.iter_mut().enumerate() {
            if let Some(slot) = slot.take() {
                if local_pool[index].replace(slot).is_some() {
                    return Err(LLMError::GpuError(format!(
                        "control local result pool slot {index} was already occupied during recovery"
                    )));
                }
            }
        }
        for (index, slot) in self.remote_slots.iter_mut().enumerate() {
            if let Some(slot) = slot.take() {
                if remote_pool[index].replace(slot).is_some() {
                    return Err(LLMError::GpuError(format!(
                        "control remote result pool slot {index} was already occupied during recovery"
                    )));
                }
            }
        }
        Ok(())
    }

    fn release_reservation(mut self) -> Result<()> {
        let reservation =
            std::mem::replace(&mut self.reservation, FailClosedRelayReservation(None));
        reservation.release_drained()
    }
}

impl Drop for ControlLayerResources {
    fn drop(&mut self) {
        for slot in &mut self.local_slots {
            if let Some(slot) = slot.take() {
                std::mem::forget(slot);
            }
        }
        for slot in &mut self.remote_slots {
            if let Some(slot) = slot.take() {
                std::mem::forget(slot);
            }
        }
        // `FailClosedRelayReservation::drop` retains its five pinned leases.
    }
}

/// One terminal token result. Logits never leave the fixed shell staging
/// allocation except through this compact identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeterogeneousControlTokenOutput {
    pub token_id: u32,
    pub logits_bf16_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeterogeneousControlExpertExecution {
    pub descriptor: ExpertResultDescriptor,
    pub kernel_elapsed_ms: f32,
    pub input_d2d_bytes: usize,
    pub input_h2d_bytes: usize,
    pub output_d2h_bytes: usize,
    pub cpu_elapsed_ns: Option<u64>,
    pub trace: Option<SelectedExpertFirstDivergenceTrace>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeterogeneousControlLayerExecution {
    pub layer: u16,
    pub plan: PackedDispatchPlan,
    pub router: ExactRouterExecution,
    pub experts: Vec<HeterogeneousControlExpertExecution>,
    pub reduction: RankOrderedReductionExecution,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeterogeneousControlStepExecution {
    pub output: HeterogeneousControlTokenOutput,
    pub captured_layer: Option<HeterogeneousControlLayerExecution>,
}

#[must_use = "the transaction may be discarded only when every component drain is proven"]
pub struct HeterogeneousControlStepFailure {
    pub error: LLMError,
    pub drain_proven: bool,
}

struct ControlLayerWorkFailure {
    error: LLMError,
    drain_proven: bool,
}

pub(crate) struct ControlResidentExpertInput {
    pub slice: Arc<CudaSlice<u16>>,
    pub stable_device: StableCudaDeviceId,
}

struct ControlHostStaging {
    cosine_bf16_bits: [u16; ROPE_PAIRS],
    sine_bf16_bits: [u16; ROPE_PAIRS],
    hidden_bf16_bits: Vec<u16>,
    logits_bf16_bits: Vec<u16>,
}

impl ControlHostStaging {
    fn new(vocab_size: usize) -> Self {
        Self {
            cosine_bf16_bits: [0; ROPE_PAIRS],
            sine_bf16_bits: [0; ROPE_PAIRS],
            hidden_bf16_bits: vec![0; GPT_OSS_HIDDEN_SIZE],
            logits_bf16_bits: vec![0; vocab_size],
        }
    }
}

struct ControlCudaState {
    stream: Arc<CudaStream>,
    loader: KernelLoader,
    hidden: CudaSlice<u16>,
    input_norm: CudaSlice<u16>,
    query: CudaSlice<u16>,
    key: CudaSlice<u16>,
    value: CudaSlice<u16>,
    keys: Vec<CudaSlice<u16>>,
    values: Vec<CudaSlice<u16>>,
    cosine: CudaSlice<u16>,
    sine: CudaSlice<u16>,
    attention: CudaSlice<u16>,
    attention_projection: CudaSlice<u16>,
    post_attention_residual: CudaSlice<u16>,
    router_input: Arc<CudaSlice<u16>>,
    final_norm: CudaSlice<u16>,
    logits: CudaSlice<u16>,
    zero_lm_bias: CudaSlice<u8>,
}

/// Fixed-allocation, one-token-at-a-time owner shell. K/V rows at
/// `committed_tokens` are private until `commit_prepared_token` advances the
/// single visible length. Discard leaves an unreachable row that the next
/// attempt deterministically overwrites.
pub struct CudaHeterogeneousControlShell {
    stable_device: StableCudaDeviceId,
    state: Option<ControlCudaState>,
    host_staging: Option<ControlHostStaging>,
    quarantined_host_staging: Option<ControlHostStaging>,
    committed_tokens: usize,
    prepared_position: Option<usize>,
    next_layer: usize,
    num_layers: usize,
    vocab_size: usize,
    poisoned: bool,
}

impl CudaHeterogeneousControlShell {
    pub fn new(model: &OwnerSelectiveModel, config: &CpuGptOssConfig) -> Result<Self> {
        validate_control_config(model, config)?;
        let stable_device = model.placement().layer_owner().stable_id.clone();
        let resolved = resolve_stable_device(&stable_device, &list_devices())
            .map_err(|error| LLMError::GpuError(format!("stable control device: {error}")))?;
        let context =
            CudaContext::new(resolved.transient_ordinal).map_err(cuda_error("control context"))?;
        let stream = context.new_stream().map_err(cuda_error("control stream"))?;
        let loader = KernelLoader::new(
            Arc::clone(&context),
            Arc::clone(&stream),
            compiled_ptx_dir(),
        )?;
        for function in [
            EMBEDDING, RMS_NORM, PROJECTION, ROPE, APPEND_KV, ATTENTION, RESIDUAL,
        ] {
            if !loader.has_func(MODULE, function) {
                return Err(LLMError::GpuError(format!(
                    "control PTX function {function} is unavailable"
                )));
            }
        }
        let alloc = |values, label| stream.alloc_zeros::<u16>(values).map_err(cuda_error(label));
        let mut keys = Vec::with_capacity(config.num_hidden_layers);
        let mut values = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            keys.push(alloc(
                MAX_VISIBLE_TOKENS * KV_WIDTH,
                "control K/V key allocation",
            )?);
            values.push(alloc(
                MAX_VISIBLE_TOKENS * KV_WIDTH,
                "control K/V value allocation",
            )?);
        }
        let state = ControlCudaState {
            hidden: alloc(GPT_OSS_HIDDEN_SIZE, "control hidden allocation")?,
            input_norm: alloc(GPT_OSS_HIDDEN_SIZE, "control input-norm allocation")?,
            query: alloc(Q_WIDTH, "control query allocation")?,
            key: alloc(KV_WIDTH, "control key allocation")?,
            value: alloc(KV_WIDTH, "control value allocation")?,
            keys,
            values,
            cosine: alloc(ROPE_PAIRS, "control cosine allocation")?,
            sine: alloc(ROPE_PAIRS, "control sine allocation")?,
            attention: alloc(Q_WIDTH, "control attention allocation")?,
            attention_projection: alloc(
                GPT_OSS_HIDDEN_SIZE,
                "control attention-projection allocation",
            )?,
            post_attention_residual: alloc(
                GPT_OSS_HIDDEN_SIZE,
                "control post-attention allocation",
            )?,
            router_input: Arc::new(alloc(
                GPT_OSS_HIDDEN_SIZE,
                "control router-input allocation",
            )?),
            final_norm: alloc(GPT_OSS_HIDDEN_SIZE, "control final-norm allocation")?,
            logits: alloc(config.vocab_size, "control logits allocation")?,
            zero_lm_bias: stream
                .alloc_zeros::<u8>(config.vocab_size * size_of::<u16>())
                .map_err(cuda_error("control zero LM bias allocation"))?,
            stream,
            loader,
        };
        state
            .stream
            .synchronize()
            .map_err(cuda_error("control construction drain"))?;
        Ok(Self {
            stable_device,
            state: Some(state),
            host_staging: Some(ControlHostStaging::new(config.vocab_size)),
            quarantined_host_staging: None,
            committed_tokens: 0,
            prepared_position: None,
            next_layer: 0,
            num_layers: config.num_hidden_layers,
            vocab_size: config.vocab_size,
            poisoned: false,
        })
    }

    pub fn stable_device(&self) -> &StableCudaDeviceId {
        &self.stable_device
    }

    pub const fn committed_tokens(&self) -> usize {
        self.committed_tokens
    }

    pub fn owned_device_bytes(&self) -> usize {
        heterogeneous_control_shell_device_bytes(
            self.num_layers,
            self.vocab_size,
            MAX_VISIBLE_TOKENS,
        )
        .expect("validated GPT-OSS control dimensions fit usize")
    }

    pub fn owned_host_staging_bytes(&self) -> usize {
        (ROPE_PAIRS * 2 + GPT_OSS_HIDDEN_SIZE + self.vocab_size) * size_of::<u16>()
    }

    pub fn begin_token(
        &mut self,
        model: &OwnerSelectiveModel,
        config: &CpuGptOssConfig,
        token_id: u32,
        position: usize,
    ) -> Result<()> {
        self.require_reusable()?;
        validate_control_config(model, config)?;
        if self.prepared_position.is_some()
            || position != self.committed_tokens
            || position >= MAX_CONTROL_TOKENS
            || position >= MAX_VISIBLE_TOKENS
            || usize::try_from(token_id).map_or(true, |token| token >= self.vocab_size)
        {
            return Err(LLMError::ModelError(
                "control token identity/position is not admissible".into(),
            ));
        }
        let embeddings = dense(
            model,
            "model.embed_tokens.weight",
            self.vocab_size * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let state = self.state.as_mut().expect("active control state");
        let submitted = launch_embedding(
            &state.stream,
            &state.loader,
            embeddings.allocation(),
            &mut state.hidden,
            token_id,
        )
        .and_then(|()| {
            state
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("control embedding terminal event"))
        });
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => return Err(self.fail_after_enqueue(primary, false)),
        };
        if let Err(error) = terminal.synchronize() {
            return Err(self
                .fail_after_enqueue(cuda_error("control embedding terminal drain")(error), false));
        }
        self.prepared_position = Some(position);
        self.next_layer = 0;
        Ok(())
    }

    /// Execute one layer's dense/attention prefix. The caller must finish the
    /// same layer through the exact router, selected experts, reducer, and
    /// `finish_layer_residual_resident` before advancing.
    pub fn execute_layer_prefix(
        &mut self,
        model: &OwnerSelectiveModel,
        config: &CpuGptOssConfig,
        layer: usize,
    ) -> Result<()> {
        self.require_reusable()?;
        if self.prepared_position != Some(self.committed_tokens)
            || self.next_layer != layer
            || layer >= self.num_layers
        {
            return Err(LLMError::ModelError(
                "control layer prefix is out of transaction order".into(),
            ));
        }
        let prefix = format!("model.layers.{layer}");
        let input_norm_weight = dense(
            model,
            &format!("{prefix}.input_layernorm.weight"),
            GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let q_weight = dense(
            model,
            &format!("{prefix}.self_attn.q_proj.weight"),
            Q_WIDTH * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let q_bias = dense(
            model,
            &format!("{prefix}.self_attn.q_proj.bias"),
            Q_WIDTH * size_of::<u16>(),
        )?;
        let k_weight = dense(
            model,
            &format!("{prefix}.self_attn.k_proj.weight"),
            KV_WIDTH * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let k_bias = dense(
            model,
            &format!("{prefix}.self_attn.k_proj.bias"),
            KV_WIDTH * size_of::<u16>(),
        )?;
        let v_weight = dense(
            model,
            &format!("{prefix}.self_attn.v_proj.weight"),
            KV_WIDTH * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let v_bias = dense(
            model,
            &format!("{prefix}.self_attn.v_proj.bias"),
            KV_WIDTH * size_of::<u16>(),
        )?;
        let sinks = dense(
            model,
            &format!("{prefix}.self_attn.sinks"),
            NUM_HEADS * size_of::<u16>(),
        )?;
        let o_weight = dense(
            model,
            &format!("{prefix}.self_attn.o_proj.weight"),
            GPT_OSS_HIDDEN_SIZE * Q_WIDTH * size_of::<u16>(),
        )?;
        let o_bias = dense(
            model,
            &format!("{prefix}.self_attn.o_proj.bias"),
            GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let post_norm_weight = dense(
            model,
            &format!("{prefix}.post_attention_layernorm.weight"),
            GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let position = self.committed_tokens;
        let (cosine, sine) = rope_tables(config, position)?;
        let mut staging = self.host_staging.take().expect("reusable control staging");
        staging.cosine_bf16_bits.copy_from_slice(&cosine);
        staging.sine_bf16_bits.copy_from_slice(&sine);
        let state = self.state.as_mut().expect("active control state");
        let submitted = (|| -> Result<_> {
            state
                .stream
                .memcpy_htod(&staging.cosine_bf16_bits, &mut state.cosine)
                .map_err(cuda_error("control cosine H2D"))?;
            state
                .stream
                .memcpy_htod(&staging.sine_bf16_bits, &mut state.sine)
                .map_err(cuda_error("control sine H2D"))?;
            launch_rms_norm(
                &state.stream,
                &state.loader,
                &state.hidden,
                input_norm_weight.allocation(),
                &mut state.input_norm,
                config.rms_norm_eps,
            )?;
            launch_projection(
                &state.stream,
                &state.loader,
                &state.input_norm,
                q_weight.allocation(),
                q_bias.allocation(),
                &mut state.query,
                Q_WIDTH,
            )?;
            launch_projection(
                &state.stream,
                &state.loader,
                &state.input_norm,
                k_weight.allocation(),
                k_bias.allocation(),
                &mut state.key,
                KV_WIDTH,
            )?;
            launch_projection(
                &state.stream,
                &state.loader,
                &state.input_norm,
                v_weight.allocation(),
                v_bias.allocation(),
                &mut state.value,
                KV_WIDTH,
            )?;
            launch_rope(
                &state.stream,
                &state.loader,
                &mut state.query,
                &state.cosine,
                &state.sine,
                NUM_HEADS,
            )?;
            launch_rope(
                &state.stream,
                &state.loader,
                &mut state.key,
                &state.cosine,
                &state.sine,
                NUM_KV_HEADS,
            )?;
            launch_append_kv(
                &state.stream,
                &state.loader,
                &state.key,
                &state.value,
                &mut state.keys[layer],
                &mut state.values[layer],
                position,
            )?;
            launch_attention(
                &state.stream,
                &state.loader,
                &state.query,
                &state.keys[layer],
                &state.values[layer],
                sinks.allocation(),
                &mut state.attention,
                position + 1,
            )?;
            launch_projection(
                &state.stream,
                &state.loader,
                &state.attention,
                o_weight.allocation(),
                o_bias.allocation(),
                &mut state.attention_projection,
                GPT_OSS_HIDDEN_SIZE,
            )?;
            launch_residual(
                &state.stream,
                &state.loader,
                &state.hidden,
                &state.attention_projection,
                &mut state.post_attention_residual,
            )?;
            launch_rms_norm(
                &state.stream,
                &state.loader,
                &state.post_attention_residual,
                post_norm_weight.allocation(),
                Arc::get_mut(&mut state.router_input).ok_or_else(|| {
                    LLMError::GpuError(
                        "control router input still has an outstanding device borrower".into(),
                    )
                })?,
                config.rms_norm_eps,
            )?;
            state
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("control layer-prefix terminal event"))
        })();
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => {
                self.host_staging = Some(staging);
                return Err(self.fail_after_enqueue(primary, true));
            }
        };
        if let Err(error) = terminal.synchronize() {
            self.host_staging = Some(staging);
            return Err(self.fail_after_enqueue(
                cuda_error("control layer-prefix terminal drain")(error),
                true,
            ));
        }
        self.host_staging = Some(staging);
        Ok(())
    }

    pub(crate) fn resident_expert_input(&self) -> Result<ControlResidentExpertInput> {
        self.require_reusable()?;
        let state = self.state.as_ref().expect("active control state");
        Ok(ControlResidentExpertInput {
            slice: Arc::clone(&state.router_input),
            stable_device: self.stable_device.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn route_resident_decode(
        &self,
        router: &mut CudaExactRouter,
        layer: u16,
        placement_epoch: u64,
        source_activation: &mut BoundedPinnedLease<u16>,
        route_records: &mut BoundedPinnedLease<u8>,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<ExactRouterExecution> {
        self.require_reusable()?;
        router.execute_device_and_download(
            layer,
            GptOssPhase::Decode,
            placement_epoch,
            1,
            self.state
                .as_ref()
                .expect("active control state")
                .router_input
                .as_ref(),
            &self.stable_device,
            source_activation,
            route_records,
            timeline,
        )
    }

    /// Consume the reducer's resident output, complete the layer residual, and
    /// carry the result device-to-device into the next layer's hidden state.
    pub fn finish_layer_residual_resident(
        &mut self,
        layer: usize,
        reducer: &CudaRankOrderedReducer,
    ) -> Result<()> {
        self.require_reusable()?;
        if self.next_layer != layer
            || reducer.stable_device() != &self.stable_device
            || self.prepared_position != Some(self.committed_tokens)
        {
            return Err(LLMError::GpuError(
                "control resident residual identity/order mismatch".into(),
            ));
        }
        let state = self.state.as_mut().expect("active control state");
        let submitted = (|| -> Result<_> {
            state
                .stream
                .memcpy_dtod(reducer.output_device(), &mut state.attention_projection)
                .map_err(cuda_error("control reduced update D2D"))?;
            launch_residual(
                &state.stream,
                &state.loader,
                &state.post_attention_residual,
                &state.attention_projection,
                Arc::get_mut(&mut state.router_input).ok_or_else(|| {
                    LLMError::GpuError(
                        "control final residual cannot reuse borrowed router input".into(),
                    )
                })?,
            )?;
            state
                .stream
                .memcpy_dtod(state.router_input.as_ref(), &mut state.hidden)
                .map_err(cuda_error("control next-layer hidden D2D"))?;
            state
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("control resident residual terminal event"))
        })();
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => return Err(self.fail_after_enqueue(primary, false)),
        };
        if let Err(error) = terminal.synchronize() {
            return Err(self.fail_after_enqueue(
                cuda_error("control resident residual terminal drain")(error),
                false,
            ));
        }
        self.next_layer = self
            .next_layer
            .checked_add(1)
            .ok_or_else(|| LLMError::ModelError("control layer index overflows".into()))?;
        Ok(())
    }

    pub fn download_hidden(&mut self) -> Result<Vec<u16>> {
        self.require_reusable()?;
        let mut staging = self.host_staging.take().expect("reusable control staging");
        let state = self.state.as_mut().expect("active control state");
        let submitted = state
            .stream
            .memcpy_dtoh(&state.hidden, &mut staging.hidden_bf16_bits)
            .map_err(cuda_error("control hidden D2H"))
            .and_then(|()| {
                state
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(cuda_error("control hidden D2H terminal event"))
            });
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => {
                self.host_staging = Some(staging);
                return Err(self.fail_after_enqueue(primary, true));
            }
        };
        if let Err(error) = terminal.synchronize() {
            self.host_staging = Some(staging);
            return Err(self
                .fail_after_enqueue(cuda_error("control hidden D2H terminal drain")(error), true));
        }
        let output = staging.hidden_bf16_bits.clone();
        self.host_staging = Some(staging);
        Ok(output)
    }

    /// Finish exact final normalization and BF16 LM-head projection, then
    /// choose the lower token ID on an equal BF16 logit.
    pub fn finish_token(
        &mut self,
        model: &OwnerSelectiveModel,
        config: &CpuGptOssConfig,
    ) -> Result<HeterogeneousControlTokenOutput> {
        self.require_reusable()?;
        if self.prepared_position != Some(self.committed_tokens)
            || self.next_layer != self.num_layers
        {
            return Err(LLMError::ModelError(
                "control token cannot produce logits before all layers finish".into(),
            ));
        }
        let final_norm = dense(
            model,
            "model.norm.weight",
            GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let lm_head = dense(
            model,
            "lm_head.weight",
            self.vocab_size * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        )?;
        let mut staging = self.host_staging.take().expect("reusable control staging");
        let state = self.state.as_mut().expect("active control state");
        let submitted = (|| -> Result<_> {
            launch_rms_norm(
                &state.stream,
                &state.loader,
                &state.hidden,
                final_norm.allocation(),
                &mut state.final_norm,
                config.rms_norm_eps,
            )?;
            launch_projection(
                &state.stream,
                &state.loader,
                &state.final_norm,
                lm_head.allocation(),
                &state.zero_lm_bias,
                &mut state.logits,
                self.vocab_size,
            )?;
            state
                .stream
                .memcpy_dtoh(&state.logits, &mut staging.logits_bf16_bits)
                .map_err(cuda_error("control logits D2H"))?;
            state
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("control logits terminal event"))
        })();
        let terminal = match submitted {
            Ok(terminal) => terminal,
            Err(primary) => {
                self.host_staging = Some(staging);
                return Err(self.fail_after_enqueue(primary, true));
            }
        };
        if let Err(error) = terminal.synchronize() {
            self.host_staging = Some(staging);
            return Err(
                self.fail_after_enqueue(cuda_error("control logits terminal drain")(error), true)
            );
        }
        let mut best = None::<(u16, u32)>;
        for (token, bits) in staging.logits_bf16_bits.iter().copied().enumerate() {
            let value = half::bf16::from_bits(bits).to_f32();
            if !value.is_finite() {
                self.host_staging = Some(staging);
                return Err(LLMError::ModelError(
                    "control LM head produced a non-finite BF16 logit".into(),
                ));
            }
            let token = u32::try_from(token)
                .map_err(|_| LLMError::ModelError("control token ID overflows u32".into()))?;
            if best.is_none_or(|(best_bits, best_token)| {
                let best_value = half::bf16::from_bits(best_bits).to_f32();
                value > best_value || (value == best_value && token < best_token)
            }) {
                best = Some((bits, token));
            }
        }
        let token_id = best
            .map(|(_, token)| token)
            .ok_or_else(|| LLMError::ModelError("control vocabulary is empty".into()))?;
        let digest = Sha256::digest(bytemuck::cast_slice(&staging.logits_bf16_bits));
        let logits_bf16_sha256 = format!("{digest:x}");
        self.host_staging = Some(staging);
        Ok(HeterogeneousControlTokenOutput {
            token_id,
            logits_bf16_sha256,
        })
    }

    pub fn commit_prepared_token(&mut self) -> Result<()> {
        self.require_reusable()?;
        let position = self.prepared_position.ok_or_else(|| {
            LLMError::ModelError("control has no prepared token to commit".into())
        })?;
        if position != self.committed_tokens || self.next_layer != self.num_layers {
            return Err(LLMError::ModelError(
                "control prepared token is incomplete at commit".into(),
            ));
        }
        self.committed_tokens = self
            .committed_tokens
            .checked_add(1)
            .ok_or_else(|| LLMError::ModelError("control visible length overflows".into()))?;
        self.prepared_position = None;
        self.next_layer = 0;
        Ok(())
    }

    pub fn discard_prepared_token(&mut self) -> Result<()> {
        self.require_reusable()?;
        if self.prepared_position.is_none() {
            return Err(LLMError::ModelError(
                "control has no prepared token to discard".into(),
            ));
        }
        self.drain()?;
        self.prepared_position = None;
        self.next_layer = 0;
        Ok(())
    }

    pub fn drain(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(LLMError::GpuError(
                "poisoned control shell cannot prove a later drain".into(),
            ));
        }
        match self
            .state
            .as_ref()
            .expect("active control state")
            .stream
            .synchronize()
        {
            Ok(()) => Ok(()),
            Err(error) => {
                self.poisoned = true;
                self.quarantined_host_staging = self.host_staging.take();
                Err(cuda_error("control shell drain")(error))
            }
        }
    }

    pub(crate) const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    const fn has_prepared_token(&self) -> bool {
        self.prepared_position.is_some()
    }

    pub(crate) fn quarantine_external_device_use(&mut self) {
        self.poisoned = true;
        self.quarantined_host_staging = self.host_staging.take();
    }

    fn require_reusable(&self) -> Result<()> {
        if self.poisoned || self.state.is_none() || self.host_staging.is_none() {
            return Err(LLMError::GpuError(
                "control shell is poisoned or lacks owned staging".into(),
            ));
        }
        Ok(())
    }

    fn fail_after_enqueue(
        &mut self,
        primary: LLMError,
        host_staging_may_be_referenced: bool,
    ) -> LLMError {
        match self
            .state
            .as_ref()
            .expect("active control state")
            .stream
            .synchronize()
        {
            Ok(()) => primary,
            Err(drain) => {
                self.poisoned = true;
                if host_staging_may_be_referenced {
                    self.quarantined_host_staging = self.host_staging.take();
                }
                LLMError::GpuError(format!(
                    "control shell work failed ({primary}); mandatory fallback drain failed ({drain}); state quarantined"
                ))
            }
        }
    }
}

impl Drop for CudaHeterogeneousControlShell {
    fn drop(&mut self) {
        if self.poisoned {
            if let Some(state) = self.state.take() {
                std::mem::forget(state);
            }
            if let Some(staging) = self.quarantined_host_staging.take() {
                std::mem::forget(staging);
            }
        }
    }
}

/// H7 execution owner. All routers, relay arenas, reducer workspaces, result
/// slots, CPU scratch, and five pinned pools are constructed before the first
/// token is dispatched. One relay/reducer pair per layer permits every layer
/// in a token to bind the same transaction generation exactly once.
pub struct HeterogeneousControlRuntime {
    shell: CudaHeterogeneousControlShell,
    routers: Vec<CudaExactRouter>,
    relays: Vec<CudaResultRelay>,
    reducers: Vec<CudaRankOrderedReducer>,
    pools: RelayPinnedPools,
    local_slots: Vec<[Option<CudaSelectedExpertResultSlot>; 4]>,
    remote_slots: Vec<[Option<CudaSelectedExpertResultSlot>; 4]>,
    cpu_worker: CpuX8SelectedExpertWorker,
    poisoned: bool,
}

impl HeterogeneousControlRuntime {
    pub fn new(model: &mut OwnerSelectiveModel, config: &CpuGptOssConfig) -> Result<Self> {
        let shell = CudaHeterogeneousControlShell::new(model, config)?;
        let mut routers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let weight_name = format!("model.layers.{layer}.mlp.router.weight");
            let bias_name = format!("model.layers.{layer}.mlp.router.bias");
            let weights = checkpoint_bf16_bits(
                model,
                &weight_name,
                config.num_local_experts * GPT_OSS_HIDDEN_SIZE,
            )?;
            let bias = checkpoint_bf16_bits(model, &bias_name, config.num_local_experts)?;
            routers.push(CudaExactRouter::new(
                model.placement().layer_owner().stable_id.clone(),
                1,
                ExactRouterWeightsView {
                    experts: config.num_local_experts,
                    weight_bf16_bits: weights,
                    bias_bf16_bits: bias,
                },
            )?);
        }
        let pools = RelayPinnedPools::warm_exact(
            routers
                .first()
                .ok_or_else(|| LLMError::ModelError("control has no router".into()))?,
            1,
        )?;
        let mut relays = Vec::with_capacity(config.num_hidden_layers);
        let mut reducers = Vec::with_capacity(config.num_hidden_layers);
        for router in &routers {
            let relay = CudaResultRelay::new(router, 1)?;
            let reducer = CudaRankOrderedReducer::new(&relay)?;
            relays.push(relay);
            reducers.push(reducer);
        }
        let remote_by_layer = (0..config.num_hidden_layers)
            .map(|layer| {
                model.placement().assignments().any(|(key, owner)| {
                    usize::from(key.layer) == layer
                        && matches!(owner, ExpertOwner::RemoteGpu { .. })
                })
            })
            .collect::<Vec<_>>();
        let mut local_slots = Vec::with_capacity(config.num_hidden_layers);
        let mut remote_slots = Vec::with_capacity(config.num_hidden_layers);
        {
            let parts = model.execution_parts();
            for has_remote in remote_by_layer {
                let mut layer_slots: [Option<CudaSelectedExpertResultSlot>; 4] =
                    std::array::from_fn(|_| None);
                for slot in &mut layer_slots {
                    *slot = Some(parts.layer_owner_executor.allocate_result_slot()?);
                }
                local_slots.push(layer_slots);
                let mut layer_remote_slots: [Option<CudaSelectedExpertResultSlot>; 4] =
                    std::array::from_fn(|_| None);
                if has_remote {
                    for slot in &mut layer_remote_slots {
                        *slot = Some(parts.remote_executor.allocate_result_slot()?);
                    }
                }
                remote_slots.push(layer_remote_slots);
            }
        }
        Ok(Self {
            shell,
            routers,
            relays,
            reducers,
            pools,
            local_slots,
            remote_slots,
            cpu_worker: CpuX8SelectedExpertWorker::new(),
            poisoned: false,
        })
    }

    pub fn shell(&self) -> &CudaHeterogeneousControlShell {
        &self.shell
    }

    pub fn pinned_pool_stats(&self) -> RelayPinnedPoolStats {
        self.pools.stats()
    }

    pub const fn cpu_scratch_bytes(&self) -> usize {
        self.cpu_worker.scratch_bytes()
    }

    pub const fn cpu_high_water_jobs(&self) -> usize {
        self.cpu_worker.high_water_jobs()
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_remote_expert_failure_for_test(
        &mut self,
        model: &mut OwnerSelectiveModel,
        fault: SelectedExpertInjectedFault,
    ) -> Result<()> {
        if self.poisoned {
            return Err(LLMError::GpuError(
                "poisoned H7 runtime cannot arm a selected-expert fault".into(),
            ));
        }
        model
            .execution_parts()
            .remote_executor
            .inject_next_failure(fault)
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_remote_expert_failure_after_for_test(
        &mut self,
        model: &mut OwnerSelectiveModel,
        fault: SelectedExpertInjectedFault,
        successful_submissions_before_fault: usize,
    ) -> Result<()> {
        if self.poisoned {
            return Err(LLMError::GpuError(
                "poisoned H7 runtime cannot arm a selected-expert fault".into(),
            ));
        }
        model
            .execution_parts()
            .remote_executor
            .inject_failure_after_successful_submissions(fault, successful_submissions_before_fault)
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn is_poisoned_for_test(&self) -> bool {
        self.poisoned
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn shell_is_poisoned_for_test(&self) -> bool {
        self.shell.is_poisoned()
    }

    /// Execute a complete token without publishing it. `transaction_generation`
    /// is shared by all per-layer relay/reducer pairs and must be the H5 step
    /// ID. Only `commit_prepared_token` makes the private K/V rows visible.
    pub fn execute_step(
        &mut self,
        model: &mut OwnerSelectiveModel,
        config: &CpuGptOssConfig,
        token_id: u32,
        transaction_generation: u64,
        capture_layer: Option<usize>,
        timeline: &CorrelatedTimeline,
    ) -> std::result::Result<HeterogeneousControlStepExecution, HeterogeneousControlStepFailure>
    {
        if self.poisoned || transaction_generation == 0 {
            return Err(HeterogeneousControlStepFailure {
                error: LLMError::GpuError(
                    "control runtime is poisoned or has an invalid generation".into(),
                ),
                drain_proven: !self.poisoned,
            });
        }
        let position = self.shell.committed_tokens();
        if let Err(error) = self.shell.begin_token(model, config, token_id, position) {
            let drain_proven = !self.shell.is_poisoned();
            return Err(self.finish_failed_step(model, error, drain_proven));
        }
        let mut captured_layer = None;
        for layer in 0..config.num_hidden_layers {
            let capture = capture_layer == Some(layer);
            let execution = match self.execute_layer(
                model,
                config,
                layer,
                transaction_generation,
                capture,
                timeline,
            ) {
                Ok(execution) => execution,
                Err(failure) => {
                    return Err(self.finish_failed_step(
                        model,
                        failure.error,
                        failure.drain_proven,
                    ));
                }
            };
            if capture {
                captured_layer = Some(execution);
            }
        }
        let output = match self.shell.finish_token(model, config) {
            Ok(output) => output,
            Err(error) => {
                let drain_proven = !self.shell.is_poisoned();
                return Err(self.finish_failed_step(model, error, drain_proven));
            }
        };
        Ok(HeterogeneousControlStepExecution {
            output,
            captured_layer,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn execute_layer(
        &mut self,
        model: &mut OwnerSelectiveModel,
        config: &CpuGptOssConfig,
        layer: usize,
        transaction_generation: u64,
        capture: bool,
        timeline: &CorrelatedTimeline,
    ) -> std::result::Result<HeterogeneousControlLayerExecution, ControlLayerWorkFailure> {
        // Evidence storage is reserved before the router or any selected
        // expert work enters a stream.
        let mut trace_storage: [Option<SelectedExpertTraceStorage>; 4] =
            std::array::from_fn(|_| capture.then(SelectedExpertTraceStorage::new));
        let mut expert_evidence: [Option<HeterogeneousControlExpertExecution>; 4] =
            std::array::from_fn(|_| None);
        let mut completed_expert_evidence = Vec::with_capacity(4);
        let reservation = self
            .pools
            .try_reserve_all(transaction_generation)
            .map_err(|error| ControlLayerWorkFailure {
                error,
                drain_proven: true,
            })?;
        let mut resources = ControlLayerResources::new(reservation);
        if let Err(error) = self.shell.execute_layer_prefix(model, config, layer) {
            return Err(self.cleanup_control_layer(model, layer, resources, error, true));
        }
        let layer_u16 = match u16::try_from(layer) {
            Ok(layer) => layer,
            Err(_) => {
                return Err(self.cleanup_control_layer(
                    model,
                    layer,
                    resources,
                    LLMError::ModelError("control layer overflows u16".into()),
                    true,
                ));
            }
        };
        let routed = {
            let router = match self.routers.get_mut(layer) {
                Some(router) => router,
                None => {
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        LLMError::ModelError("control router layer is missing".into()),
                        true,
                    ));
                }
            };
            let (source_activation, route_descriptors) = resources.reservation.router_leases();
            self.shell.route_resident_decode(
                router,
                layer_u16,
                model.placement().placement_epoch(),
                source_activation,
                route_descriptors,
                Some(timeline),
            )
        };
        let routed = match routed {
            Ok(routed) => routed,
            Err(error) => {
                return Err(self.cleanup_control_layer(model, layer, resources, error, true));
            }
        };
        let plan = match pack_routes_bounded(&routed.batch, model.placement()) {
            Ok(plan) => plan,
            Err(error) => {
                return Err(self.cleanup_control_layer(model, layer, resources, error, true));
            }
        };
        let packed = {
            let (source_activation, remote_gpu_input) = resources.reservation.packing_leases();
            pack_remote_inputs(&plan, source_activation, remote_gpu_input)
        };
        if let Err(error) = packed {
            return Err(self.cleanup_control_layer(model, layer, resources, error, true));
        }
        let prepared_reduction = match PreparedRankOrderedReduction::prepare(
            &routed.batch,
            model.placement(),
            transaction_generation,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                return Err(self.cleanup_control_layer(model, layer, resources, error, true));
            }
        };
        let descriptors = prepared_reduction.expected_results().to_vec();
        let relay_bound = match self.relays.get_mut(layer) {
            Some(relay) => relay.bind_decode_generation(transaction_generation, &plan),
            None => Err(LLMError::ModelError(
                "control relay layer is missing".into(),
            )),
        };
        if let Err(error) = relay_bound {
            return Err(self.cleanup_control_layer(model, layer, resources, error, true));
        }

        let local_routes = plan
            .local_gpu
            .iter()
            .flat_map(|owner| owner.routes.iter().cloned())
            .collect::<Vec<_>>();
        let cpu_routes = plan
            .cpu
            .iter()
            .flat_map(|owner| owner.routes.iter().cloned())
            .collect::<Vec<_>>();
        let remote_routes = plan
            .remote_gpu
            .iter()
            .flat_map(|owner| owner.routes.iter().cloned())
            .collect::<Vec<_>>();
        if local_routes.len() + cpu_routes.len() + remote_routes.len() != 4
            || local_routes.len() > 4
            || cpu_routes.len() > 4
            || remote_routes.len() > 4
        {
            return Err(self.cleanup_control_layer(
                model,
                layer,
                resources,
                LLMError::ModelError(
                    "control decode dispatch must contain exactly four bounded owner routes".into(),
                ),
                true,
            ));
        }
        let resident = if local_routes.is_empty() {
            None
        } else {
            match self.shell.resident_expert_input() {
                Ok(resident) => Some(resident),
                Err(error) => {
                    return Err(self.cleanup_control_layer(model, layer, resources, error, true));
                }
            }
        };
        for (pool_slot, route) in local_routes.iter().enumerate() {
            let slot = match self.local_slots[layer][pool_slot].take() {
                Some(slot) => slot,
                None => {
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        LLMError::GpuError("control local result slot is missing".into()),
                        true,
                    ));
                }
            };
            match slot.bind_drained_for_route_owned(transaction_generation, &route.descriptor) {
                Ok(slot) => resources.local_slots[pool_slot] = Some(slot),
                Err(failure) => {
                    resources.local_slots[pool_slot] = Some(failure.result_slot);
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        failure.error,
                        true,
                    ));
                }
            }
        }
        for (pool_slot, route) in remote_routes.iter().enumerate() {
            let slot = match self.remote_slots[layer][pool_slot].take() {
                Some(slot) => slot,
                None => {
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        LLMError::GpuError(format!(
                            "control remote result slot {pool_slot} is missing"
                        )),
                        true,
                    ));
                }
            };
            match slot.bind_drained_for_route_owned(transaction_generation, &route.descriptor) {
                Ok(slot) => resources.remote_slots[pool_slot] = Some(slot),
                Err(failure) => {
                    resources.remote_slots[pool_slot] = Some(failure.result_slot);
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        failure.error,
                        true,
                    ));
                }
            }
        }

        let mut nonlocal_completions = Vec::with_capacity(4);
        let worker_result: std::result::Result<(), ControlLayerWorkFailure> = (|| {
            let (source_activation, remote_gpu_input, remote_gpu_result, cpu_result) =
                resources.reservation.worker_leases();
            let parts = model.execution_parts();

            // Every route identity, weight handle, trace buffer, result slot,
            // and GPU0 input handle is resolved before the first enqueue.
            let mut local_jobs = Vec::with_capacity(4);
            for (pool_slot, route) in local_routes.iter().cloned().enumerate() {
                let weights = parts
                    .layer_owner_experts
                    .get(&GptOssExpertKey {
                        layer: layer_u16,
                        expert: route.descriptor.route.expert_id,
                    })
                    .map(Arc::clone)
                    .ok_or_else(|| ControlLayerWorkFailure {
                        error: LLMError::ModelError("control local expert weight missing".into()),
                        drain_proven: true,
                    })?;
                let trace = if capture {
                    Some(
                        take_trace_storage(&mut trace_storage, &route).map_err(|error| {
                            ControlLayerWorkFailure {
                                error,
                                drain_proven: true,
                            }
                        })?,
                    )
                } else {
                    None
                };
                local_jobs.push((
                    pool_slot,
                    route.descriptor.clone(),
                    route,
                    weights,
                    Arc::clone(
                        &resident
                            .as_ref()
                            .expect("local route has resident input")
                            .slice,
                    ),
                    trace,
                ));
            }
            let mut remote_jobs = Vec::with_capacity(4);
            for (pool_slot, route) in remote_routes.iter().cloned().enumerate() {
                let weights = parts
                    .remote_gpu_experts
                    .get(&GptOssExpertKey {
                        layer: layer_u16,
                        expert: route.descriptor.route.expert_id,
                    })
                    .map(Arc::clone)
                    .ok_or_else(|| ControlLayerWorkFailure {
                        error: LLMError::ModelError("control remote expert weight missing".into()),
                        drain_proven: true,
                    })?;
                let trace = if capture {
                    Some(
                        take_trace_storage(&mut trace_storage, &route).map_err(|error| {
                            ControlLayerWorkFailure {
                                error,
                                drain_proven: true,
                            }
                        })?,
                    )
                } else {
                    None
                };
                remote_jobs.push((pool_slot, route.descriptor.clone(), route, weights, trace));
            }
            let mut cpu_jobs = Vec::with_capacity(4);
            for route in cpu_routes.iter().cloned() {
                let record =
                    parts
                        .cpu_layers
                        .get(&layer_u16)
                        .ok_or_else(|| ControlLayerWorkFailure {
                            error: LLMError::ModelError(
                                "control CPU owner layer record missing".into(),
                            ),
                            drain_proven: true,
                        })?;
                let view = record
                    .expert_view(route.descriptor.route.expert_id)
                    .map_err(|error| ControlLayerWorkFailure {
                        error,
                        drain_proven: true,
                    })?;
                let trace = if capture {
                    Some(
                        take_trace_storage(&mut trace_storage, &route).map_err(|error| {
                            ControlLayerWorkFailure {
                                error,
                                drain_proven: true,
                            }
                        })?,
                    )
                } else {
                    None
                };
                cpu_jobs.push((route, view, trace));
            }

            let stable_device = resident.as_ref().map(|resident| &resident.stable_device);
            let mut local_jobs = local_jobs.into_iter();
            let mut remote_jobs = remote_jobs.into_iter();
            {
                let mut primary_failure = None;

                let local_pending = match local_jobs.next() {
                    Some((pool_slot, descriptor, route, weights, input, trace)) => {
                        let slot = resources.local_slots[pool_slot]
                            .take()
                            .expect("prepared first local result slot");
                        let prepared = match trace {
                            Some(trace) => parts.layer_owner_executor.prepare_owned_device(
                                GptOssPhase::Decode,
                                descriptor,
                                weights,
                                input,
                                stable_device.expect("local route has stable device"),
                                slot,
                                trace,
                            ),
                            None => parts.layer_owner_executor.prepare_owned_device_output_only(
                                GptOssPhase::Decode,
                                descriptor,
                                weights,
                                input,
                                stable_device.expect("local route has stable device"),
                                slot,
                            ),
                        };
                        match prepared.and_then(|prepared| {
                            prepared.submit_with_timeline(timeline, "h7_gpu0_local_first")
                        }) {
                            Ok(pending) => Some((pool_slot, route, pending)),
                            Err(failure) => {
                                primary_failure = Some(classify_control_owned_failure(
                                    failure,
                                    &mut resources.local_slots[pool_slot],
                                ));
                                None
                            }
                        }
                    }
                    None => None,
                };

                let remote_pending = if primary_failure.is_none() {
                    match remote_jobs.next() {
                        Some((pool_slot, descriptor, route, weights, trace)) => {
                            let slot = resources.remote_slots[pool_slot]
                                .take()
                                .expect("prepared first remote result slot");
                            let input_start = route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
                            let input = &remote_gpu_input.as_slice()
                                [input_start..input_start + GPT_OSS_HIDDEN_SIZE];
                            let prepared = match trace {
                                Some(trace) => parts.remote_executor.prepare_owned_pinned(
                                    GptOssPhase::Decode,
                                    descriptor,
                                    weights,
                                    input,
                                    slot,
                                    trace,
                                ),
                                None => parts.remote_executor.prepare_owned_pinned_output_only(
                                    GptOssPhase::Decode,
                                    descriptor,
                                    weights,
                                    input,
                                    slot,
                                ),
                            };
                            match prepared.and_then(|prepared| {
                                prepared.submit_with_timeline(timeline, "h7_gpu1_remote")
                            }) {
                                Ok(pending) => Some((pool_slot, route, pending)),
                                Err(failure) => {
                                    merge_control_failure(
                                        &mut primary_failure,
                                        classify_control_owned_failure(
                                            failure,
                                            &mut resources.remote_slots[pool_slot],
                                        ),
                                    );
                                    None
                                }
                            }
                        }
                        None => None,
                    }
                } else {
                    None
                };

                // The capacity-one CPU worker consumes every CPU-owned route while
                // the first submitted job on each GPU is in flight. All output
                // rows use their category-global packed slot.
                if primary_failure.is_none() {
                    for (route, view, mut trace) in cpu_jobs {
                        let execution = match trace.as_mut() {
                            Some(trace) => self.cpu_worker.execute_into_pinned_with_trace(
                                layer_u16,
                                &route.descriptor,
                                route.owner_route_slot,
                                view,
                                &source_activation.as_slice()[..GPT_OSS_HIDDEN_SIZE],
                                cpu_result,
                                trace,
                                Some(timeline),
                            ),
                            None => self.cpu_worker.execute_into_pinned_device_only(
                                layer_u16,
                                &route.descriptor,
                                route.owner_route_slot,
                                view,
                                &source_activation.as_slice()[..GPT_OSS_HIDDEN_SIZE],
                                cpu_result,
                                Some(timeline),
                            ),
                        };
                        match execution {
                            Ok(execution) => {
                                let slot = route.descriptor.canonical_result_slot as usize;
                                if expert_evidence[slot].is_some() {
                                    primary_failure = Some(ControlLayerWorkFailure {
                                        error: LLMError::ModelError(
                                            "duplicate CPU canonical result slot".into(),
                                        ),
                                        drain_proven: true,
                                    });
                                    break;
                                }
                                expert_evidence[slot] = Some(HeterogeneousControlExpertExecution {
                                    descriptor: ExpertResultDescriptor::from_packed_route(
                                        &route.descriptor,
                                    ),
                                    kernel_elapsed_ms: 0.0,
                                    input_d2d_bytes: 0,
                                    input_h2d_bytes: 0,
                                    output_d2h_bytes: execution.output_bytes,
                                    cpu_elapsed_ns: Some(execution.elapsed_ns),
                                    trace: trace.map(SelectedExpertTraceStorage::into_trace),
                                });
                                nonlocal_completions.push(execution.route_contract);
                            }
                            Err(error) => {
                                primary_failure = Some(ControlLayerWorkFailure {
                                    error,
                                    drain_proven: true,
                                });
                                break;
                            }
                        }
                    }
                }

                // Drain both initially submitted owners regardless of which owner
                // failed. This is the all-sibling terminal barrier.
                if let Some((pool_slot, route, pending)) = local_pending {
                    match drain_control_pending(
                        pending,
                        capture,
                        None,
                        0,
                        timeline,
                        "h7_gpu0_local_first",
                        route,
                    ) {
                        Ok(completion) => {
                            match store_gpu_evidence(&mut expert_evidence, completion) {
                                Ok(stored) => {
                                    resources.local_slots[pool_slot] = Some(stored.result_slot);
                                }
                                Err((error, completion)) => {
                                    resources.local_slots[pool_slot] = Some(completion.result_slot);
                                    merge_control_failure(
                                        &mut primary_failure,
                                        ControlLayerWorkFailure {
                                            error,
                                            drain_proven: true,
                                        },
                                    );
                                }
                            }
                        }
                        Err(failure) => merge_control_failure(
                            &mut primary_failure,
                            classify_control_owned_failure(
                                failure,
                                &mut resources.local_slots[pool_slot],
                            ),
                        ),
                    }
                }
                if let Some((pool_slot, route, pending)) = remote_pending {
                    let output_slot = route.owner_route_slot;
                    match drain_control_pending(
                        pending,
                        capture,
                        Some(remote_gpu_result),
                        output_slot,
                        timeline,
                        "h7_gpu1_remote",
                        route,
                    ) {
                        Ok(completion) => {
                            match store_gpu_evidence(&mut expert_evidence, completion) {
                                Ok(stored) => {
                                    nonlocal_completions.push(stored.route_contract);
                                    resources.remote_slots[pool_slot] = Some(stored.result_slot);
                                }
                                Err((error, completion)) => {
                                    resources.remote_slots[pool_slot] =
                                        Some(completion.result_slot);
                                    merge_control_failure(
                                        &mut primary_failure,
                                        ControlLayerWorkFailure {
                                            error,
                                            drain_proven: true,
                                        },
                                    );
                                }
                            }
                        }
                        Err(failure) => merge_control_failure(
                            &mut primary_failure,
                            classify_control_owned_failure(
                                failure,
                                &mut resources.remote_slots[pool_slot],
                            ),
                        ),
                    }
                }
                if let Some(failure) = primary_failure {
                    return Err(failure);
                }
            }

            for (pool_slot, descriptor, route, weights, input, trace) in local_jobs {
                let slot = resources.local_slots[pool_slot]
                    .take()
                    .expect("prepared serial local result slot");
                let prepared = match trace {
                    Some(trace) => parts.layer_owner_executor.prepare_owned_device(
                        GptOssPhase::Decode,
                        descriptor,
                        weights,
                        input,
                        stable_device.expect("local route has stable device"),
                        slot,
                        trace,
                    ),
                    None => parts.layer_owner_executor.prepare_owned_device_output_only(
                        GptOssPhase::Decode,
                        descriptor,
                        weights,
                        input,
                        stable_device.expect("local route has stable device"),
                        slot,
                    ),
                };
                let prepared = prepared.map_err(|failure| {
                    classify_control_owned_failure(failure, &mut resources.local_slots[pool_slot])
                })?;
                let pending = prepared
                    .submit_with_timeline(timeline, "h7_gpu0_local_serial")
                    .map_err(|failure| {
                        classify_control_owned_failure(
                            failure,
                            &mut resources.local_slots[pool_slot],
                        )
                    })?;
                let completion = drain_control_pending(
                    pending,
                    capture,
                    None,
                    0,
                    timeline,
                    "h7_gpu0_local_serial",
                    route,
                )
                .map_err(|failure| {
                    classify_control_owned_failure(failure, &mut resources.local_slots[pool_slot])
                })?;
                match store_gpu_evidence(&mut expert_evidence, completion) {
                    Ok(stored) => {
                        resources.local_slots[pool_slot] = Some(stored.result_slot);
                    }
                    Err((error, completion)) => {
                        resources.local_slots[pool_slot] = Some(completion.result_slot);
                        return Err(ControlLayerWorkFailure {
                            error,
                            drain_proven: true,
                        });
                    }
                }
            }

            for (pool_slot, descriptor, route, weights, trace) in remote_jobs {
                let slot = resources.remote_slots[pool_slot]
                    .take()
                    .expect("prepared serial remote result slot");
                let input_start = route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
                let input =
                    &remote_gpu_input.as_slice()[input_start..input_start + GPT_OSS_HIDDEN_SIZE];
                let prepared = match trace {
                    Some(trace) => parts.remote_executor.prepare_owned_pinned(
                        GptOssPhase::Decode,
                        descriptor,
                        weights,
                        input,
                        slot,
                        trace,
                    ),
                    None => parts.remote_executor.prepare_owned_pinned_output_only(
                        GptOssPhase::Decode,
                        descriptor,
                        weights,
                        input,
                        slot,
                    ),
                };
                let prepared = prepared.map_err(|failure| {
                    classify_control_owned_failure(failure, &mut resources.remote_slots[pool_slot])
                })?;
                let pending = prepared
                    .submit_with_timeline(timeline, "h7_gpu1_remote_serial")
                    .map_err(|failure| {
                        classify_control_owned_failure(
                            failure,
                            &mut resources.remote_slots[pool_slot],
                        )
                    })?;
                let output_slot = route.owner_route_slot;
                let completion = drain_control_pending(
                    pending,
                    capture,
                    Some(remote_gpu_result),
                    output_slot,
                    timeline,
                    "h7_gpu1_remote_serial",
                    route,
                )
                .map_err(|failure| {
                    classify_control_owned_failure(failure, &mut resources.remote_slots[pool_slot])
                })?;
                match store_gpu_evidence(&mut expert_evidence, completion) {
                    Ok(stored) => {
                        nonlocal_completions.push(stored.route_contract);
                        resources.remote_slots[pool_slot] = Some(stored.result_slot);
                    }
                    Err((error, completion)) => {
                        resources.remote_slots[pool_slot] = Some(completion.result_slot);
                        return Err(ControlLayerWorkFailure {
                            error,
                            drain_proven: true,
                        });
                    }
                }
            }
            Ok(())
        })();
        drop(resident);
        if let Err(failure) = worker_result {
            return Err(self.cleanup_control_layer(
                model,
                layer,
                resources,
                failure.error,
                failure.drain_proven,
            ));
        }

        let completed = {
            let relay = match self.relays.get_mut(layer) {
                Some(relay) => relay,
                None => {
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        LLMError::ModelError("control relay layer is missing".into()),
                        true,
                    ));
                }
            };
            relay.upload_results_bound(
                &plan,
                resources.reservation.take(),
                &nonlocal_completions,
                Some(timeline),
            )
        };
        match completed {
            Ok(completed) => resources.reservation.replace(completed.reservation),
            Err(failure) => {
                let drain_proven = failure.reservation.is_some();
                if let Some(returned) = failure.reservation {
                    resources.reservation.replace(returned);
                }
                return Err(self.cleanup_control_layer(
                    model,
                    layer,
                    resources,
                    failure.error,
                    drain_proven,
                ));
            }
        }
        for (pool_slot, route) in local_routes.iter().enumerate() {
            let slot = resources.local_slots[pool_slot]
                .take()
                .expect("drained local result slot before canonical D2D");
            let descriptor = &descriptors[route.descriptor.canonical_result_slot as usize];
            let uploaded = self.relays[layer].upload_local_device_result_with_timeline(
                transaction_generation,
                descriptor,
                slot,
                timeline,
            );
            match uploaded {
                Ok(completed) => resources.local_slots[pool_slot] = Some(completed.result_slot),
                Err(failure) => {
                    let drain_proven = failure.result_slot.is_some();
                    if let Some(slot) = failure.result_slot {
                        resources.local_slots[pool_slot] = Some(slot);
                    }
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        failure.error,
                        drain_proven,
                    ));
                }
            }
        }
        let reduction = {
            let (relay, reducer) = match (self.relays.get_mut(layer), self.reducers.get_mut(layer))
            {
                (Some(relay), Some(reducer)) => (relay, reducer),
                _ => {
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        LLMError::ModelError("control relay/reducer layer is missing".into()),
                        true,
                    ));
                }
            };
            reducer.reduce_relay_classified(relay, prepared_reduction)
        };
        let reduction = match reduction {
            Ok(reduction) => reduction,
            Err(failure) => {
                return Err(self.cleanup_control_layer(
                    model,
                    layer,
                    resources,
                    failure.error,
                    failure.drain_proven,
                ));
            }
        };
        let residual = {
            let reducer = match self.reducers.get(layer) {
                Some(reducer) => reducer,
                None => {
                    return Err(self.cleanup_control_layer(
                        model,
                        layer,
                        resources,
                        LLMError::ModelError("control reducer layer is missing".into()),
                        true,
                    ));
                }
            };
            self.shell.finish_layer_residual_resident(layer, reducer)
        };
        if let Err(error) = residual {
            let drain_proven = !self.shell.is_poisoned();
            return Err(self.cleanup_control_layer(model, layer, resources, error, drain_proven));
        }
        for evidence in expert_evidence {
            let Some(evidence) = evidence else {
                return Err(self.cleanup_control_layer(
                    model,
                    layer,
                    resources,
                    LLMError::ModelError(
                        "control expert evidence is missing a canonical slot".into(),
                    ),
                    true,
                ));
            };
            completed_expert_evidence.push(evidence);
        }
        if let Err(error) =
            resources.restore_slots(&mut self.local_slots[layer], &mut self.remote_slots[layer])
        {
            self.poisoned = true;
            return Err(ControlLayerWorkFailure {
                error,
                drain_proven: true,
            });
        }
        if let Err(error) = resources.release_reservation() {
            self.poisoned = true;
            return Err(ControlLayerWorkFailure {
                error,
                drain_proven: true,
            });
        }
        Ok(HeterogeneousControlLayerExecution {
            layer: layer_u16,
            plan,
            router: routed,
            experts: completed_expert_evidence,
            reduction,
        })
    }

    fn cleanup_control_layer(
        &mut self,
        model: &mut OwnerSelectiveModel,
        layer: usize,
        mut resources: ControlLayerResources,
        primary: LLMError,
        drain_proven: bool,
    ) -> ControlLayerWorkFailure {
        if !drain_proven {
            self.quarantine_all_components(model);
            return ControlLayerWorkFailure {
                error: primary,
                drain_proven: false,
            };
        }
        if let Err(drain) = self.drain_components(model, true) {
            return ControlLayerWorkFailure {
                error: LLMError::GpuError(format!(
                    "H7 layer {layer} failed ({primary}); all-component cleanup drain was not proven ({drain})"
                )),
                drain_proven: false,
            };
        }
        if let Err(recover) =
            resources.restore_slots(&mut self.local_slots[layer], &mut self.remote_slots[layer])
        {
            self.poisoned = true;
            return ControlLayerWorkFailure {
                error: LLMError::GpuError(format!(
                    "H7 layer {layer} failed ({primary}); drained result-slot recovery failed ({recover})"
                )),
                drain_proven: true,
            };
        }
        if let Err(release) = resources.release_reservation() {
            self.poisoned = true;
            return ControlLayerWorkFailure {
                error: LLMError::GpuError(format!(
                    "H7 layer {layer} failed ({primary}); drained pinned reservation release failed ({release})"
                )),
                drain_proven: true,
            };
        }
        ControlLayerWorkFailure {
            error: primary,
            drain_proven: true,
        }
    }

    fn finish_failed_step(
        &mut self,
        model: &mut OwnerSelectiveModel,
        primary: LLMError,
        drain_proven: bool,
    ) -> HeterogeneousControlStepFailure {
        if !drain_proven {
            self.quarantine_all_components(model);
            return HeterogeneousControlStepFailure {
                error: primary,
                drain_proven: false,
            };
        }
        if let Err(drain) = self.drain_components(model, true) {
            return HeterogeneousControlStepFailure {
                error: LLMError::GpuError(format!(
                    "H7 step failed ({primary}); final all-component drain was not proven ({drain})"
                )),
                drain_proven: false,
            };
        }
        if self.shell.has_prepared_token() {
            if let Err(discard) = self.shell.discard_prepared_token() {
                self.quarantine_all_components(model);
                return HeterogeneousControlStepFailure {
                    error: LLMError::GpuError(format!(
                        "H7 step failed ({primary}); private token discard was not proven ({discard})"
                    )),
                    drain_proven: false,
                };
            }
        }
        HeterogeneousControlStepFailure {
            error: primary,
            drain_proven: true,
        }
    }

    fn quarantine_all_components(&mut self, model: &mut OwnerSelectiveModel) {
        self.poisoned = true;
        model.quarantine_execution();
        self.shell.quarantine_external_device_use();
        for router in &mut self.routers {
            router.quarantine_unproven_device_work();
        }
        for reducer in &mut self.reducers {
            reducer.quarantine_unproven_device_work();
        }
        for relay in &mut self.relays {
            relay.quarantine_unproven_device_work();
        }
        for layer in &mut self.local_slots {
            for slot in layer {
                if let Some(slot) = slot.take() {
                    std::mem::forget(slot);
                }
            }
        }
        for layer in &mut self.remote_slots {
            for slot in layer {
                if let Some(slot) = slot.take() {
                    std::mem::forget(slot);
                }
            }
        }
    }

    pub fn commit_prepared_token(&mut self) -> Result<()> {
        self.shell.commit_prepared_token()
    }

    pub fn prepared_hidden_bf16_bits(&mut self) -> Result<Vec<u16>> {
        self.shell.download_hidden()
    }

    pub fn discard_prepared_token(&mut self, model: &mut OwnerSelectiveModel) -> Result<()> {
        self.drain_components(model, true)?;
        self.shell.discard_prepared_token()
    }

    pub fn drain(&mut self, model: &mut OwnerSelectiveModel) -> Result<()> {
        self.drain_components(model, false)
    }

    fn drain_components(
        &mut self,
        model: &mut OwnerSelectiveModel,
        abandon_active: bool,
    ) -> Result<()> {
        let mut failures = Vec::new();
        for router in &mut self.routers {
            if let Err(error) = router.drain() {
                failures.push(format!("router: {error}"));
            }
        }
        if let Err(error) = model.drain() {
            failures.push(format!("expert owners: {error}"));
        }
        if let Err(error) = self.shell.drain() {
            failures.push(format!("layer owner shell: {error}"));
        }
        for reducer in &mut self.reducers {
            if let Err(error) = reducer.prove_transaction_drain() {
                failures.push(format!("reducer: {error}"));
            }
        }
        if failures.is_empty() {
            for relay in &mut self.relays {
                let result = match (abandon_active, relay.active_decode_generation()) {
                    (true, Some(generation)) => relay.abandon_decode_generation(generation, true),
                    (false, Some(_)) => Err(LLMError::GpuError(
                        "control final drain found an active relay generation".into(),
                    )),
                    _ => relay.prove_transaction_drain(),
                };
                if let Err(error) = result {
                    failures.push(format!("relay: {error}"));
                }
            }
        } else {
            for relay in &mut self.relays {
                relay.quarantine_unproven_device_work();
            }
        }
        if failures.is_empty() {
            return Ok(());
        }
        self.quarantine_all_components(model);
        Err(LLMError::GpuError(format!(
            "heterogeneous control drain was not proven: {}",
            failures.join("; ")
        )))
    }
}

struct GpuCompletion {
    descriptor: ExpertResultDescriptor,
    route_contract: CanonicalRouteContract,
    result_slot: CudaSelectedExpertResultSlot,
    kernel_elapsed_ms: f32,
    input_d2d_bytes: usize,
    input_h2d_bytes: usize,
    output_d2h_bytes: usize,
    trace: Option<SelectedExpertFirstDivergenceTrace>,
}

impl GpuCompletion {
    fn traced(
        route: PackedDispatchRoute,
        execution: super::cuda_expert::OwnedSelectedExpertExecution,
    ) -> Self {
        Self {
            descriptor: ExpertResultDescriptor::from_packed_route(&route.descriptor),
            route_contract: execution.route_contract,
            result_slot: execution.result_slot,
            kernel_elapsed_ms: execution.kernel_elapsed_ms,
            input_d2d_bytes: execution.input_d2d_bytes,
            input_h2d_bytes: execution.input_h2d_bytes,
            output_d2h_bytes: execution.output_d2h_bytes,
            trace: Some(execution.trace),
        }
    }

    fn output(
        route: PackedDispatchRoute,
        execution: super::cuda_expert::OwnedSelectedExpertOutput,
    ) -> Self {
        Self {
            descriptor: ExpertResultDescriptor::from_packed_route(&route.descriptor),
            route_contract: execution.route_contract,
            result_slot: execution.result_slot,
            kernel_elapsed_ms: execution.kernel_elapsed_ms,
            input_d2d_bytes: execution.input_d2d_bytes,
            input_h2d_bytes: execution.input_h2d_bytes,
            output_d2h_bytes: execution.output_d2h_bytes,
            trace: None,
        }
    }
}

fn drain_control_pending(
    pending: PendingOwnedSelectedExpert<'_>,
    capture: bool,
    pinned_output: Option<&mut BoundedPinnedLease<u16>>,
    pinned_output_slot: u32,
    timeline: &CorrelatedTimeline,
    actor: &str,
    route: PackedDispatchRoute,
) -> std::result::Result<GpuCompletion, OwnedSelectedExpertFailure> {
    if capture {
        pending
            .drain_with_trace_at(pinned_output, pinned_output_slot, timeline, actor)
            .map(|execution| GpuCompletion::traced(route, execution))
    } else {
        pending
            .drain_output_only_at(pinned_output, pinned_output_slot, timeline, actor)
            .map(|execution| GpuCompletion::output(route, execution))
    }
}

fn classify_control_owned_failure(
    failure: OwnedSelectedExpertFailure,
    slot: &mut Option<CudaSelectedExpertResultSlot>,
) -> ControlLayerWorkFailure {
    let (error, recovered, drain_proven, _pinned_referenced, _retained) = failure.into_parts();
    if let Some(recovered) = recovered {
        *slot = Some(recovered);
    }
    ControlLayerWorkFailure {
        error,
        drain_proven,
    }
}

fn merge_control_failure(
    primary: &mut Option<ControlLayerWorkFailure>,
    secondary: ControlLayerWorkFailure,
) {
    if let Some(primary) = primary {
        primary.drain_proven &= secondary.drain_proven;
    } else {
        *primary = Some(secondary);
    }
}

struct StoredGpuCompletion {
    route_contract: CanonicalRouteContract,
    result_slot: CudaSelectedExpertResultSlot,
}

fn store_gpu_evidence(
    evidence: &mut [Option<HeterogeneousControlExpertExecution>; 4],
    completion: GpuCompletion,
) -> std::result::Result<StoredGpuCompletion, (LLMError, GpuCompletion)> {
    let slot = completion.route_contract.result_slot as usize;
    if slot >= evidence.len() || evidence[slot].is_some() {
        return Err((
            LLMError::ModelError(
                "control GPU completion duplicated a canonical result slot".into(),
            ),
            completion,
        ));
    }
    let GpuCompletion {
        descriptor,
        route_contract,
        result_slot,
        kernel_elapsed_ms,
        input_d2d_bytes,
        input_h2d_bytes,
        output_d2h_bytes,
        trace,
    } = completion;
    evidence[slot] = Some(HeterogeneousControlExpertExecution {
        descriptor,
        kernel_elapsed_ms,
        input_d2d_bytes,
        input_h2d_bytes,
        output_d2h_bytes,
        cpu_elapsed_ns: None,
        trace,
    });
    Ok(StoredGpuCompletion {
        route_contract,
        result_slot,
    })
}

fn take_trace_storage(
    storage: &mut [Option<SelectedExpertTraceStorage>; 4],
    route: &PackedDispatchRoute,
) -> Result<SelectedExpertTraceStorage> {
    storage
        .get_mut(route.descriptor.canonical_result_slot as usize)
        .and_then(Option::take)
        .ok_or_else(|| LLMError::ModelError("control selected-expert trace slot is missing".into()))
}

fn checkpoint_bf16_bits<'a>(
    model: &'a OwnerSelectiveModel,
    name: &str,
    expected_values: usize,
) -> Result<&'a [u16]> {
    let tensor = model.checkpoint().tensor(name)?;
    let values = bytemuck::try_cast_slice::<u8, u16>(tensor.bytes())
        .map_err(|error| LLMError::ModelError(format!("control tensor {name}: {error}")))?;
    if values.len() != expected_values {
        return Err(LLMError::ModelError(format!(
            "control tensor {name} has {} BF16 values, expected {expected_values}",
            values.len()
        )));
    }
    Ok(values)
}

fn validate_control_config(model: &OwnerSelectiveModel, config: &CpuGptOssConfig) -> Result<()> {
    let native = model.checkpoint().config();
    if native.num_hidden_layers != config.num_hidden_layers
        || native.num_experts != config.num_local_experts
        || native.vocab_size != config.vocab_size
        || config.hidden_size != GPT_OSS_HIDDEN_SIZE
        || config.num_attention_heads != NUM_HEADS
        || config.num_key_value_heads != NUM_KV_HEADS
        || config.head_dim != HEAD_DIM
        || config.sliding_window != MAX_VISIBLE_TOKENS
        || config.num_hidden_layers == 0
        || config.num_hidden_layers > u16::MAX as usize
        || config.vocab_size > u32::MAX as usize
        || config.layer_types.iter().any(|kind| {
            !matches!(
                kind.as_str(),
                "sliding_attention" | "local_attention" | "full_attention"
            )
        })
    {
        return Err(LLMError::ModelError(
            "H7 control supports only the validated GPT-OSS 20B decode shape".into(),
        ));
    }
    Ok(())
}

fn cuda_error(context: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{context}: {error}"))
}
