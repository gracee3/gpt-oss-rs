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
    outputs: AsyncMutex<VecDeque<Vec<RequestOutput>>>,
}

impl ScriptedEngine {
    fn new(outputs: Vec<Vec<RequestOutput>>) -> Arc<Self> {
        Arc::new(Self {
            next_request_id: AtomicU64::new(1),
            prompts: Mutex::new(Vec::new()),
            outputs: AsyncMutex::new(VecDeque::from(outputs)),
        })
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
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
    make_app_with_model(outputs, "test-model")
}

fn make_app_with_model(
    outputs: Vec<Vec<RequestOutput>>,
    model: &str,
) -> (TestServer, Arc<ScriptedEngine>) {
    let engine = ScriptedEngine::new(outputs);
    let mut state = AppState::new(
        engine.clone(),
        model.to_string(),
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

fn harmony_output(request_id: u64, completion: &str, finish_reason: FinishReason) -> RequestOutput {
    let token_ids = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
        .unwrap()
        .encode_completion_text(completion);
    output_with_text(
        request_id,
        "harmony prompt",
        &[1, 2, 3],
        completion,
        &token_ids,
        Some(finish_reason),
        true,
    )
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

fn function_tools() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }])
}

#[tokio::test]
async fn chat_function_calls_and_tool_history_work_on_primary_and_legacy_routes() {
    let call = " to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"city\":\"Boston\"}<|call|>";
    let final_answer = "<|channel|>final<|message|>It is 18C.<|return|>";
    let scripted = vec![
        vec![harmony_output(1, call, FinishReason::Stop)],
        vec![harmony_output(2, call, FinishReason::Stop)],
        vec![harmony_output(3, call, FinishReason::Stop)],
        vec![harmony_output(4, final_answer, FinishReason::Stop)],
    ];
    let (server, engine) = make_app_with_model(scripted, "openai/gpt-oss-20b");

    for route in ["/v1/chat/completions", "/tools"] {
        let response = server
            .post(route)
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": function_tools()
            }))
            .await;
        response.assert_status_ok();
        let json = response.json::<serde_json::Value>();
        assert!(json["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
    }

    let streamed = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "stream": true,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": function_tools()
        }))
        .await;
    streamed.assert_status_ok();
    let body = streamed.text();
    assert!(body.contains("\"tool_calls\""), "{body}");
    assert!(body.contains("\"finish_reason\":\"tool_calls\""), "{body}");
    assert!(body.contains("data: [DONE]"), "{body}");

    let history = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_weather", "content": "{\"temp_c\":18}"}
            ],
            "tools": function_tools()
        }))
        .await;
    history.assert_status_ok();
    let history_json = history.json::<serde_json::Value>();
    assert_eq!(
        history_json["choices"][0]["message"]["content"],
        "It is 18C."
    );
    assert!(engine.prompts()[3].contains("<|start|>functions.get_weather"));

    let unresolved = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "tool", "tool_call_id": "missing", "content": "nope"}]
        }))
        .await;
    unresolved.assert_status(StatusCode::BAD_REQUEST);
    assert!(unresolved.json::<serde_json::Value>()["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unresolved tool call"));
}

#[tokio::test]
async fn responses_function_call_follow_up_and_manual_output_validation() {
    let call = " to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"city\":\"Boston\"}<|call|>";
    let final_answer = "<|channel|>final<|message|>It is 18C.<|return|>";
    let scripted = vec![
        vec![harmony_output(1, call, FinishReason::Stop)],
        vec![harmony_output(2, final_answer, FinishReason::Stop)],
        vec![harmony_output(3, call, FinishReason::Stop)],
    ];
    let (server, engine) = make_app_with_model(scripted, "openai/gpt-oss-20b");

    let first = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "store": true,
            "input": "Weather?",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
        }))
        .await;
    first.assert_status_ok();
    let first_json = first.json::<serde_json::Value>();
    assert_eq!(first_json["output"][0]["type"], "function_call");
    let response_id = first_json["id"].as_str().unwrap();
    let call_id = first_json["output"][0]["call_id"].as_str().unwrap();

    let follow_up = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "store": true,
            "previous_response_id": response_id,
            "input": [{"type": "function_call_output", "call_id": call_id, "output": {"temp_c": 18}}],
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
        }))
        .await;
    follow_up.assert_status_ok();
    let follow_up_json = follow_up.json::<serde_json::Value>();
    assert_eq!(
        follow_up_json["output"][0]["content"][0]["text"],
        "It is 18C."
    );
    assert!(engine.prompts()[1].contains("<|start|>functions.get_weather"));

    let streamed = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "stream": true,
            "store": false,
            "input": "Weather?",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
        }))
        .await;
    streamed.assert_status_ok();
    let body = streamed.text();
    assert!(body.contains("event: response.function_call_arguments.done"));
    assert!(body.contains("event: response.completed"));

    let unresolved = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "input": [{"type": "function_call_output", "call_id": "missing", "output": "nope"}]
        }))
        .await;
    unresolved.assert_status(StatusCode::BAD_REQUEST);
    assert!(unresolved.json::<serde_json::Value>()["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unresolved call_id"));
}

#[tokio::test]
async fn length_truncated_harmony_returns_partial_chat_and_incomplete_responses() {
    let partial = "<|channel|>final<|message|>A partial answer";
    let mut scripted = (1..=4)
        .map(|id| vec![harmony_output(id, partial, FinishReason::Length)])
        .collect::<Vec<_>>();
    scripted.push(vec![harmony_output(
        5,
        "<|channel|>analysis<|message|>Hidden partial reasoning",
        FinishReason::Length,
    )]);
    let (server, _) = make_app_with_model(scripted, "openai/gpt-oss-20b");

    let chat = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Answer"}],
            "max_tokens": 3
        }))
        .await;
    chat.assert_status_ok();
    let chat_json = chat.json::<serde_json::Value>();
    assert_eq!(
        chat_json["choices"][0]["message"]["content"],
        "A partial answer"
    );
    assert_eq!(chat_json["choices"][0]["finish_reason"], "length");

    let chat_stream = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "stream": true,
            "messages": [{"role": "user", "content": "Answer"}],
            "max_tokens": 3
        }))
        .await;
    chat_stream.assert_status_ok();
    let chat_body = chat_stream.text();
    assert!(chat_body.contains("A partial answer"), "{chat_body}");
    assert!(chat_body.contains("\"finish_reason\":\"length\""));
    assert!(chat_body.contains("data: [DONE]"));

    let response = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "input": "Answer",
            "max_output_tokens": 3
        }))
        .await;
    response.assert_status_ok();
    let response_json = response.json::<serde_json::Value>();
    assert_eq!(response_json["status"], "incomplete");
    assert_eq!(
        response_json["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(
        response_json["output"][0]["content"][0]["text"],
        "A partial answer"
    );

    let response_stream = server
        .post("/v1/responses")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "stream": true,
            "input": "Answer",
            "max_output_tokens": 3
        }))
        .await;
    response_stream.assert_status_ok();
    let response_body = response_stream.text();
    assert!(response_body.contains("event: response.incomplete"));
    assert!(response_body.contains("\"status\":\"incomplete\""));
    assert!(response_body.contains("\"reason\":\"max_output_tokens\""));

    let hidden_analysis = server
        .post("/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "Answer"}],
            "max_tokens": 3
        }))
        .await;
    hidden_analysis.assert_status_ok();
    let hidden_json = hidden_analysis.json::<serde_json::Value>();
    assert_eq!(hidden_json["choices"][0]["message"]["content"], "");
    assert_eq!(hidden_json["choices"][0]["finish_reason"], "length");
}
