use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_cpu_kernels::KernelPath;
use gpt_oss_model_runner::{CpuModelRunner, CpuPrefillTrace};
use gpt_oss_tokenizer::{
    FunctionDefinition, HarmonyProtocol, ProtocolMessage, ToolDefinition, ToolParameterProperty,
    ToolParameters, HARMONY_CALL_TOKEN_ID, HARMONY_RETURN_TOKEN_ID,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_FIXTURES: &str = "crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json";

#[derive(Debug, Parser)]
#[command(about = "Pinned GPT-OSS CPU Harmony parity and prefill trace runner")]
struct Cli {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    repack_cache: PathBuf,

    #[arg(long, default_value = DEFAULT_FIXTURES)]
    fixtures: PathBuf,

    #[arg(long)]
    scenario: String,

    #[arg(long, default_value = "auto")]
    kernel: KernelPath,

    #[arg(long, default_value_t = 4)]
    threads: usize,

    #[arg(long, default_value_t = 8)]
    max_new_tokens: usize,

    #[arg(long, value_delimiter = ',')]
    trace_layers: Vec<usize>,

    #[arg(long, default_value_t = 8)]
    top_k: usize,

    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
struct FixtureManifest {
    schema_version: u32,
    model: serde_json::Value,
    official_oracle: serde_json::Value,
    llama_cpp: serde_json::Value,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Scenario {
    id: String,
    #[serde(flatten)]
    input: ScenarioInput,
    expected_prompt_tokens: usize,
    prompt_text_sha256: String,
    prompt_token_ids_sha256: String,
    #[serde(default)]
    official_greedy_tokens: Option<Vec<u32>>,
    #[serde(default)]
    llama_ubatch_1_greedy_tokens: Option<Vec<u32>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ScenarioInput {
    UserText { content: String },
    RepeatedSegments { segments: usize },
    ToolHistory,
}

#[derive(Debug, Serialize)]
struct ParityCapture<'a> {
    schema_version: u32,
    scenario: &'a str,
    fixture_manifest: &'a Path,
    model_path: &'a Path,
    repack_cache: &'a Path,
    kernel: String,
    prompt_text: String,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    expected_official_greedy_tokens: Option<&'a [u32]>,
    pinned_llama_ubatch_1_greedy_tokens: Option<&'a [u32]>,
    prompt_seconds: f64,
    generation_seconds: f64,
    trace: Option<CpuPrefillTrace>,
    pinned_model: &'a serde_json::Value,
    pinned_official_oracle: &'a serde_json::Value,
    pinned_llama_cpp: &'a serde_json::Value,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let manifest = load_manifest(&cli.fixtures)?;
    let scenario = manifest
        .scenarios
        .iter()
        .find(|scenario| scenario.id == cli.scenario)
        .with_context(|| format!("unknown fixture scenario '{}'", cli.scenario))?;
    let rendered = render_scenario(scenario)?;
    verify_rendered_fixture(scenario, &rendered.text, &rendered.token_ids)?;

    let context_cap = rendered
        .token_ids
        .len()
        .checked_add(cli.max_new_tokens)
        .context("context length overflow")?;
    let mut runner = CpuModelRunner::load(
        &cli.model,
        &cli.repack_cache,
        cli.kernel,
        cli.threads,
        context_cap,
    )?;

    let prompt_start = Instant::now();
    let (mut logits, trace) = if cli.trace_layers.is_empty() {
        (runner.prefill(&rendered.token_ids)?, None)
    } else {
        let (logits, trace) =
            runner.prefill_trace(&rendered.token_ids, &cli.trace_layers, cli.top_k)?;
        (logits, Some(trace))
    };
    let prompt_seconds = prompt_start.elapsed().as_secs_f64();

    let generation_start = Instant::now();
    let mut generated_token_ids = Vec::with_capacity(cli.max_new_tokens);
    for _ in 0..cli.max_new_tokens {
        let token_id = greedy_token(&logits)? as u32;
        generated_token_ids.push(token_id);
        if matches!(token_id, HARMONY_RETURN_TOKEN_ID | HARMONY_CALL_TOKEN_ID) {
            break;
        }
        logits = runner.decode(token_id)?;
    }
    let generation_seconds = generation_start.elapsed().as_secs_f64();
    if let Some(expected) = &scenario.official_greedy_tokens {
        if cli.max_new_tokens >= expected.len() && &generated_token_ids != expected {
            bail!(
                "scenario {} generated {:?}, expected official tokens {:?}",
                scenario.id,
                generated_token_ids,
                expected
            );
        }
    }

    let capture = ParityCapture {
        schema_version: manifest.schema_version,
        scenario: &scenario.id,
        fixture_manifest: &cli.fixtures,
        model_path: &cli.model,
        repack_cache: &cli.repack_cache,
        kernel: cli.kernel.to_string(),
        prompt_text: rendered.text,
        prompt_token_ids: rendered.token_ids,
        generated_token_ids,
        expected_official_greedy_tokens: scenario.official_greedy_tokens.as_deref(),
        pinned_llama_ubatch_1_greedy_tokens: scenario.llama_ubatch_1_greedy_tokens.as_deref(),
        prompt_seconds,
        generation_seconds,
        trace,
        pinned_model: &manifest.model,
        pinned_official_oracle: &manifest.official_oracle,
        pinned_llama_cpp: &manifest.llama_cpp,
    };
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_vec_pretty(&capture)?;
    std::fs::write(&cli.output, &encoded)?;
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn load_manifest(path: &Path) -> Result<FixtureManifest> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read fixture manifest {}", path.display()))?;
    let manifest: FixtureManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse fixture manifest {}", path.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported CPU parity fixture schema {}",
            manifest.schema_version
        );
    }
    Ok(manifest)
}

fn render_scenario(scenario: &Scenario) -> Result<gpt_oss_tokenizer::RenderedPrompt> {
    let (messages, tools) = match &scenario.input {
        ScenarioInput::UserText { content } => {
            (vec![ProtocolMessage::new("user", content)], Vec::new())
        }
        ScenarioInput::RepeatedSegments { segments } => (
            vec![ProtocolMessage::new("user", repeated_content(*segments))],
            Vec::new(),
        ),
        ScenarioInput::ToolHistory => tool_history(),
    };
    HarmonyProtocol::gpt_oss()?
        .render_prompt(&messages, None, &tools)
        .map_err(Into::into)
}

fn repeated_content(segments: usize) -> String {
    (0..segments)
        .map(|index| format!("Segment {index:02}: alpha beta gamma delta epsilon zeta eta theta."))
        .collect::<Vec<_>>()
        .join(" ")
        + " Summarize the repeated pattern in one sentence."
}

fn tool_history() -> (Vec<ProtocolMessage>, Vec<ToolDefinition>) {
    let messages = vec![
        ProtocolMessage::new("user", "What is the weather in Boston?"),
        ProtocolMessage::new("assistant", "{\"city\":\"Boston\"}")
            .with_channel("commentary")
            .with_recipient("functions.get_weather"),
        ProtocolMessage::new(
            "tool",
            "{\"city\":\"Boston\",\"temperature_f\":72,\"conditions\":\"sunny\"}",
        )
        .with_author_name("functions.get_weather")
        .with_channel("commentary")
        .with_recipient("assistant"),
        ProtocolMessage::new(
            "user",
            "Given the tool result, answer the user's weather question briefly.",
        ),
    ];
    let mut properties = HashMap::new();
    properties.insert(
        "city".to_string(),
        ToolParameterProperty {
            param_type: "string".to_string(),
            description: Some("City to inspect".to_string()),
            enum_values: None,
        },
    );
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Return current weather for a city".to_string()),
            parameters: Some(ToolParameters {
                schema_type: "object".to_string(),
                properties,
                required: vec!["city".to_string()],
            }),
        },
    }];
    (messages, tools)
}

fn verify_rendered_fixture(scenario: &Scenario, text: &str, token_ids: &[u32]) -> Result<()> {
    if token_ids.len() != scenario.expected_prompt_tokens {
        bail!(
            "fixture {} rendered {} tokens, expected {}",
            scenario.id,
            token_ids.len(),
            scenario.expected_prompt_tokens
        );
    }
    let token_text = token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let text_hash = sha256(text.as_bytes());
    let token_hash = sha256(token_text.as_bytes());
    if text_hash != scenario.prompt_text_sha256 || token_hash != scenario.prompt_token_ids_sha256 {
        bail!(
            "fixture {} rendering changed: text sha256 {}, token sha256 {}",
            scenario.id,
            text_hash,
            token_hash
        );
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn greedy_token(logits: &[f32]) -> Result<usize> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(left_index, left), (right_index, right)| {
            left.partial_cmp(right)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .context("model returned no finite logits")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_pinned_harmony_scenarios_render_exactly() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cpu_harmony_parity.json");
        let manifest = load_manifest(&path).unwrap();
        assert_eq!(manifest.scenarios.len(), 7);
        for scenario in &manifest.scenarios {
            let rendered = render_scenario(scenario).unwrap();
            verify_rendered_fixture(scenario, &rendered.text, &rendered.token_ids).unwrap();
        }
    }
}
