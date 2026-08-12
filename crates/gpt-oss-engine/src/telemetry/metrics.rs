//! Bounded `gpt_oss_*` metric vocabulary and description registration.

use std::time::Duration;

use metrics::{describe_counter, describe_gauge, describe_histogram, Unit};

pub const REQUEST_ADMISSION_TOTAL: &str = "gpt_oss_request_admission_total";
pub const REQUEST_TERMINAL_TOTAL: &str = "gpt_oss_request_terminal_total";
pub const PHASE_DURATION_SECONDS: &str = "gpt_oss_phase_duration_seconds";
pub const TOKENS_TOTAL: &str = "gpt_oss_tokens_total";
pub const DELIVERY_BYTES_TOTAL: &str = "gpt_oss_delivery_bytes_total";
pub const DELIVERY_COALESCES_TOTAL: &str = "gpt_oss_delivery_coalesces_total";
pub const SERVICE_STATE: &str = "gpt_oss_service_state";
pub const CURRENT_REQUESTS: &str = "gpt_oss_current_requests";
pub const OWNER_FAILURES_TOTAL: &str = "gpt_oss_owner_failures_total";
pub const RESERVATION_EVENTS_TOTAL: &str = "gpt_oss_reservation_events_total";
pub const RESERVATION_BYTES: &str = "gpt_oss_reservation_bytes";
pub const SCHEDULED_ROWS: &str = "gpt_oss_scheduled_rows";
pub const DISPATCH_RESULTS_TOTAL: &str = "gpt_oss_dispatch_results_total";

// Compatibility names for internal GPU instrumentation. Values stay inside
// the stable namespace while callers migrate to bounded labels.
pub const REQUEST_LATENCY: &str = "gpt_oss_phase_duration_seconds";
pub const TTFT: &str = "gpt_oss_time_to_first_token_seconds";
pub const ITL: &str = "gpt_oss_inter_token_latency_seconds";
pub const FORWARD_TIME: &str = "gpt_oss_forward_time_seconds";
pub const SAMPLE_TIME: &str = "gpt_oss_sample_time_seconds";
pub const TOKENS_PER_SECOND: &str = "gpt_oss_tokens_per_second";
pub const RUNNING_REQUESTS: &str = "gpt_oss_current_requests";
pub const WAITING_REQUESTS: &str = "gpt_oss_waiting_requests";
pub const GPU_CACHE_USAGE: &str = "gpt_oss_gpu_cache_usage_percent";
pub const WORKER_TOKENS_PER_SECOND: &str = "gpt_oss_worker_tokens_per_second";
pub const PREEMPTIONS_TOTAL: &str = "gpt_oss_preemptions_total";
pub const REQUESTS_TOTAL: &str = "gpt_oss_requests_total";
pub const FINISHED_REQUESTS_TOTAL: &str = "gpt_oss_finished_requests_total";
pub const PROMPT_TOKENS_TOTAL: &str = "gpt_oss_prompt_tokens_total";
pub const GENERATION_TOKENS_TOTAL: &str = "gpt_oss_generation_tokens_total";
pub const FORWARD_PASSES_TOTAL: &str = "gpt_oss_forward_passes_total";
pub const TOKENS_SAMPLED_TOTAL: &str = "gpt_oss_tokens_sampled_total";
pub const STEPS_TOTAL: &str = "gpt_oss_engine_steps_total";

macro_rules! bounded_label_enum {
    ($name:ident { $($variant:ident => $value:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $value),+ }
            }
        }
    };
}

bounded_label_enum!(RouteClass {
    Completions => "completions",
    ChatCompletions => "chat_completions",
    Responses => "responses",
    Models => "models",
    Health => "health",
    Ready => "ready",
});
bounded_label_enum!(DeliveryMode { Streaming => "streaming", NonStreaming => "non_streaming" });
bounded_label_enum!(ResultClass {
    Accepted => "accepted",
    Rejected => "rejected",
    Completed => "completed",
    Failed => "failed",
    Cancelled => "cancelled",
    Abandoned => "abandoned",
});
bounded_label_enum!(Phase {
    Validation => "validation",
    Tokenization => "tokenization",
    Admission => "admission",
    Queue => "queue",
    Execute => "execute",
    Commit => "commit",
    Delivery => "delivery",
    Terminal => "terminal",
});
bounded_label_enum!(BackendClass { Cpu => "cpu", Cuda => "cuda", Mock => "mock" });
bounded_label_enum!(TokenClass { Prompt => "prompt", Committed => "committed", Delivered => "delivered", Abandoned => "abandoned" });
bounded_label_enum!(ReservationEvent { Grant => "grant", Expand => "expand", Refund => "refund", Transfer => "transfer", Release => "release", Reject => "reject" });
bounded_label_enum!(DispatchResult { Selected => "selected", Fallback => "fallback", Rejected => "rejected" });

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasonCode {
    None,
    Failure(crate::service::StableFailureCode),
}

impl ReasonCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Failure(code) => code.as_str(),
        }
    }
}

pub fn record_admission(backend: BackendClass, result: ResultClass, reason: ReasonCode) {
    metrics::counter!(
        REQUEST_ADMISSION_TOTAL,
        "backend" => backend.as_str(),
        "result" => result.as_str(),
        "reason" => reason.as_str()
    )
    .increment(1);
}

pub fn record_terminal(backend: BackendClass, result: ResultClass, reason: ReasonCode) {
    metrics::counter!(
        REQUEST_TERMINAL_TOTAL,
        "backend" => backend.as_str(),
        "result" => result.as_str(),
        "reason" => reason.as_str()
    )
    .increment(1);
}

pub fn record_phase_duration(
    backend: BackendClass,
    phase: Phase,
    result: ResultClass,
    duration: Duration,
) {
    metrics::histogram!(
        PHASE_DURATION_SECONDS,
        "backend" => backend.as_str(),
        "phase" => phase.as_str(),
        "result" => result.as_str()
    )
    .record(duration.as_secs_f64());
}

pub fn record_tokens(backend: BackendClass, kind: TokenClass, count: usize) {
    if count == 0 {
        return;
    }
    metrics::counter!(
        TOKENS_TOTAL,
        "backend" => backend.as_str(),
        "kind" => kind.as_str()
    )
    .increment(count as u64);
}

pub fn record_dispatch(backend: BackendClass, result: DispatchResult, reason: ReasonCode) {
    metrics::counter!(
        DISPATCH_RESULTS_TOTAL,
        "backend" => backend.as_str(),
        "result" => result.as_str(),
        "reason" => reason.as_str()
    )
    .increment(1);
}

pub fn adjust_current_requests(backend: BackendClass, delta: f64) {
    metrics::gauge!(
        CURRENT_REQUESTS,
        "backend" => backend.as_str(),
        "phase" => Phase::Admission.as_str()
    )
    .increment(delta);
}

pub fn record_reservation(
    event: ReservationEvent,
    class: crate::memory::MemoryClass,
    result: ResultClass,
    reserved_bytes: u128,
) {
    metrics::counter!(
        RESERVATION_EVENTS_TOTAL,
        "event" => event.as_str(),
        "class" => class.as_str(),
        "result" => result.as_str()
    )
    .increment(1);
    metrics::gauge!(RESERVATION_BYTES, "class" => class.as_str())
        .set(reserved_bytes.min(f64::MAX as u128) as f64);
}

pub fn register_descriptions() {
    describe_counter!(
        REQUEST_ADMISSION_TOTAL,
        "Requests admitted or rejected by bounded policy"
    );
    describe_counter!(REQUEST_TERMINAL_TOTAL, "Terminal request outcomes");
    describe_histogram!(
        PHASE_DURATION_SECONDS,
        Unit::Seconds,
        "Monotonic request phase duration"
    );
    describe_counter!(
        TOKENS_TOTAL,
        "Prompt, committed, delivered, or abandoned tokens"
    );
    describe_counter!(
        DELIVERY_BYTES_TOTAL,
        Unit::Bytes,
        "Serialized delivery bytes by bounded result"
    );
    describe_counter!(DELIVERY_COALESCES_TOTAL, "Adjacent text delivery coalesces");
    describe_gauge!(SERVICE_STATE, "One-hot service lifecycle state");
    describe_gauge!(CURRENT_REQUESTS, "Current requests in a bounded phase");
    describe_counter!(OWNER_FAILURES_TOTAL, "Canonical owner failures");
    describe_counter!(
        RESERVATION_EVENTS_TOTAL,
        "Logical reservation lifecycle events"
    );
    describe_gauge!(
        RESERVATION_BYTES,
        Unit::Bytes,
        "Logical bytes reserved by memory class"
    );
    describe_histogram!(SCHEDULED_ROWS, "Rows scheduled per native CPU iteration");
    describe_counter!(DISPATCH_RESULTS_TOTAL, "Bounded backend dispatch outcomes");

    describe_histogram!(
        TTFT,
        Unit::Seconds,
        "Time to first delivered byte-bearing token event"
    );
    describe_histogram!(ITL, Unit::Seconds, "Inter-token delivery duration");
    describe_histogram!(FORWARD_TIME, Unit::Seconds, "Backend forward pass duration");
    describe_histogram!(SAMPLE_TIME, Unit::Seconds, "Token sampling duration");
    describe_gauge!(TOKENS_PER_SECOND, "Tokens generated per second");
    describe_gauge!(WAITING_REQUESTS, "Waiting request count");
    describe_gauge!(GPU_CACHE_USAGE, "GPU KV-cache usage percentage");
    describe_gauge!(WORKER_TOKENS_PER_SECOND, "Worker token throughput");
    describe_counter!(PREEMPTIONS_TOTAL, "Preemption count");
    describe_counter!(REQUESTS_TOTAL, "Request count");
    describe_counter!(FINISHED_REQUESTS_TOTAL, "Finished request count");
    describe_counter!(PROMPT_TOKENS_TOTAL, "Committed prompt tokens");
    describe_counter!(GENERATION_TOKENS_TOTAL, "Committed generation tokens");
    describe_counter!(FORWARD_PASSES_TOTAL, "Forward pass count");
    describe_counter!(TOKENS_SAMPLED_TOTAL, "Sampled token count");
    describe_counter!(STEPS_TOTAL, "Engine step count");
}

pub fn record_service_state(state: crate::service::ServiceState) {
    for candidate in [
        crate::service::ServiceState::Starting,
        crate::service::ServiceState::Ready,
        crate::service::ServiceState::Draining,
        crate::service::ServiceState::Failed,
        crate::service::ServiceState::Stopped,
    ] {
        metrics::gauge!(SERVICE_STATE, "state" => candidate.as_str()).set(if candidate == state {
            1.0
        } else {
            0.0
        });
    }
}
