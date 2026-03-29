use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::http::StatusCode;
use axum_test::TestServer;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer as HfTokenizer;
use tokio::sync::Mutex as AsyncMutex;
use tokio_stream::wrappers::ReceiverStream;

use gpt_oss_core::prelude::{
    CompletionOutput, FinishReason, LLMError, RequestId, RequestOutput, SamplingParams,
};
use gpt_oss_engine::RuntimeMode;
use gpt_oss_server::runtime_policy::{RuntimeBackendPath, RuntimeDecision};
use gpt_oss_server::server::InferenceEngine;
use gpt_oss_server::{build_router, AppState};
use gpt_oss_tokenizer::Tokenizer;

#[derive(Debug)]
struct ScriptedEngine {
    next_request_id: AtomicU64,
    prompts: Mutex<Vec<String>>,
    prompt_token_ids: Mutex<Vec<Vec<u32>>>,
    outputs: AsyncMutex<VecDeque<Vec<RequestOutput>>>,
}

impl ScriptedEngine {
    fn new(outputs: Vec<Vec<RequestOutput>>) -> Arc<Self> {
        Arc::new(Self {
            next_request_id: AtomicU64::new(1),
            prompts: Mutex::new(Vec::new()),
            prompt_token_ids: Mutex::new(Vec::new()),
            outputs: AsyncMutex::new(VecDeque::from(outputs)),
        })
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }

    fn prompt_token_ids(&self) -> Vec<Vec<u32>> {
        self.prompt_token_ids.lock().unwrap().clone()
    }
}

#[async_trait]
impl InferenceEngine for ScriptedEngine {
    async fn generate(
        &self,
        prompt: String,
        _params: SamplingParams,
    ) -> Result<(RequestId, ReceiverStream<RequestOutput>), LLMError> {
        self.prompts.lock().unwrap().push(prompt);

        let scripted = self
            .outputs
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| LLMError::SchedulerError("missing scripted output".into()))?;

        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = tokio::sync::mpsc::channel(scripted.len().max(1));
        tokio::spawn(async move {
            for output in scripted {
                if tx.send(output).await.is_err() {
                    break;
                }
            }
        });

        Ok((request_id, ReceiverStream::new(rx)))
    }

    async fn generate_token_ids(
        &self,
        prompt: String,
        prompt_token_ids: Vec<u32>,
        _params: SamplingParams,
    ) -> Result<(RequestId, ReceiverStream<RequestOutput>), LLMError> {
        self.prompts.lock().unwrap().push(prompt);
        self.prompt_token_ids.lock().unwrap().push(prompt_token_ids);

        let scripted = self
            .outputs
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| LLMError::SchedulerError("missing scripted output".into()))?;

        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = tokio::sync::mpsc::channel(scripted.len().max(1));
        tokio::spawn(async move {
            for output in scripted {
                if tx.send(output).await.is_err() {
                    break;
                }
            }
        });

        Ok((request_id, ReceiverStream::new(rx)))
    }
}

fn make_test_tokenizer() -> Tokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("[UNK]".to_string(), 0);
    vocab.insert("[CLS]".to_string(), 1);
    vocab.insert("[SEP]".to_string(), 2);
    vocab.insert("hello".to_string(), 3);
    vocab.insert("world".to_string(), 4);
    vocab.insert("first".to_string(), 5);
    vocab.insert("second".to_string(), 6);
    vocab.insert("follow".to_string(), 7);
    vocab.insert("up".to_string(), 8);
    vocab.insert("answer".to_string(), 9);

    let wp = WordPiece::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();

    let mut hf = HfTokenizer::new(wp);
    hf.with_pre_tokenizer(Some(Whitespace {}));

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    hf.save(&path, false).unwrap();
    Tokenizer::from_file(&path).unwrap()
}

fn make_app(outputs: Vec<Vec<RequestOutput>>) -> (TestServer, Arc<ScriptedEngine>) {
    make_app_with_model("test-model", outputs)
}

fn make_app_with_model(
    model_name: &str,
    outputs: Vec<Vec<RequestOutput>>,
) -> (TestServer, Arc<ScriptedEngine>) {
    let engine = ScriptedEngine::new(outputs);
    let mut state = AppState::new(
        engine.clone(),
        model_name.to_string(),
        RuntimeDecision {
            runtime_mode: RuntimeMode::Experimental,
            backend_path: RuntimeBackendPath::Mock,
            reason: "test mock backend".into(),
        },
        make_test_tokenizer(),
    );
    state.batch_store = None;
    let state = Arc::new(state);
    (TestServer::new(build_router(state)).unwrap(), engine)
}

fn output_with_text(
    request_id: u64,
    prompt: &str,
    prompt_token_ids: &[u32],
    text: &str,
    token_ids: &[u32],
    finish_reason: Option<FinishReason>,
    finished: bool,
) -> RequestOutput {
    RequestOutput {
        request_id: RequestId(request_id),
        prompt: prompt.to_string(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        prompt_logprobs: None,
        outputs: vec![CompletionOutput {
            index: 0,
            text: text.to_string(),
            token_ids: token_ids.to_vec(),
            cumulative_logprob: -0.5,
            logprobs: None,
            finish_reason,
        }],
        finished,
    }
}

fn parse_sse_data_lines(body: &str) -> Vec<serde_json::Value> {
    body.split("\n\n")
        .filter_map(|chunk| chunk.strip_prefix("data: "))
        .filter(|chunk| *chunk != "[DONE]")
        .map(|chunk| serde_json::from_str(chunk).unwrap())
        .collect()
}

#[tokio::test]
async fn chat_completions_stream_emits_ordered_sse_events() {
    let scripted = vec![vec![
        output_with_text(1, "hello", &[3], "Hel", &[10], None, false),
        output_with_text(1, "hello", &[3], "Hello", &[10, 11], None, false),
        output_with_text(
            1,
            "hello",
            &[3],
            "Hello",
            &[10, 11],
            Some(FinishReason::Stop),
            true,
        ),
    ]];
    let (server, _engine) = make_app(scripted);

    let response = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }))
        .await;

    response.assert_status_ok();

    let body = response.text();
    let role_idx = body.find("\"role\":\"assistant\"").unwrap();
    let first_delta_idx = body.find("\"content\":\"Hel\"").unwrap();
    let second_delta_idx = body.find("\"content\":\"Hello\"").unwrap();
    let finish_idx = body.find("\"finish_reason\":\"stop\"").unwrap();
    let done_idx = body.find("data: [DONE]").unwrap();

    assert!(role_idx < first_delta_idx);
    assert!(first_delta_idx < second_delta_idx);
    assert!(second_delta_idx < finish_idx);
    assert!(finish_idx < done_idx);

    let data_lines = parse_sse_data_lines(&body);
    assert_eq!(data_lines[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(data_lines[1]["choices"][0]["delta"]["content"], "Hel");
    assert_eq!(data_lines[2]["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(data_lines[3]["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn gpt_oss_chat_completions_use_harmony_token_path() {
    let scripted = vec![vec![output_with_text(
        1,
        "prompt",
        &[1, 2, 3],
        "Hi",
        &[10],
        Some(FinishReason::Stop),
        true,
    )]];
    let (server, engine) = make_app_with_model("openai/gpt-oss-20b", scripted);

    let response = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }))
        .await;

    response.assert_status_ok();

    let prompt_token_ids = engine.prompt_token_ids();
    assert_eq!(prompt_token_ids.len(), 1);
    assert!(!prompt_token_ids[0].is_empty());

    let prompts = engine.prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("<|start|>system"));
    assert!(prompts[0].contains("<|start|>user"));
}

#[tokio::test]
async fn responses_store_round_trip_and_replay_prompt_history() {
    let scripted = vec![
        vec![output_with_text(
            1,
            "first prompt",
            &[5],
            "First answer",
            &[9],
            Some(FinishReason::Stop),
            true,
        )],
        vec![output_with_text(
            2,
            "follow up prompt",
            &[7, 8],
            "Second answer",
            &[9],
            Some(FinishReason::Stop),
            true,
        )],
    ];
    let (server, engine) = make_app(scripted);

    let first_response = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "test-model",
            "store": true,
            "input": "first question"
        }))
        .await;
    first_response.assert_status_ok();
    let first_json = first_response.json::<serde_json::Value>();
    let first_id = first_json["id"].as_str().unwrap().to_string();
    assert_eq!(
        first_json["output"][0]["content"][0]["text"],
        "First answer"
    );

    let stored = server.get(&format!("/v1/responses/{first_id}")).await;
    stored.assert_status_ok();
    let stored_json = stored.json::<serde_json::Value>();
    assert_eq!(stored_json["id"], first_id);
    assert_eq!(
        stored_json["output"][0]["content"][0]["text"],
        "First answer"
    );

    let stored_inputs = server
        .get(&format!("/v1/responses/{first_id}/input_items"))
        .await;
    stored_inputs.assert_status_ok();
    let stored_inputs_json = stored_inputs.json::<serde_json::Value>();
    assert_eq!(stored_inputs_json["data"][0]["role"], "user");
    assert_eq!(
        stored_inputs_json["data"][0]["content"][0]["text"],
        "first question"
    );

    let second_response = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "test-model",
            "store": true,
            "previous_response_id": first_id,
            "input": "follow up"
        }))
        .await;
    second_response.assert_status_ok();
    let second_json = second_response.json::<serde_json::Value>();
    assert_eq!(second_json["previous_response_id"], first_json["id"]);
    assert_eq!(
        second_json["output"][0]["content"][0]["text"],
        "Second answer"
    );

    let prompts = engine.prompts();
    assert_eq!(prompts.len(), 2);
    assert!(prompts[0].contains("first question"));
    assert!(prompts[1].contains("first question"));
    assert!(prompts[1].contains("First answer"));
    assert!(prompts[1].contains("follow up"));
}

#[tokio::test]
async fn chat_completions_returns_openai_style_model_not_found_error() {
    let (server, _engine) = make_app(vec![]);

    let response = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "wrong-model",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }))
        .await;

    response.assert_status(StatusCode::NOT_FOUND);
    let json = response.json::<serde_json::Value>();
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "model_not_found");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("model 'wrong-model' not found"));
}
