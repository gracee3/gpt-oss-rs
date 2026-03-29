use std::collections::HashMap;

use tokenizers::models::wordpiece::WordPiece;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer as HfTokenizer;

use gpt_oss_core::prelude::{FinishReason, RequestId, SamplingParams, SequenceId, TokenId};
use gpt_oss_engine::config::EngineConfig;
use gpt_oss_engine::sequence::SequenceGroup;
use gpt_oss_engine::{ExecutorInput, LLMEngine, SamplerOutput, Scheduler, SchedulerOutputs};
use gpt_oss_tokenizer::Tokenizer;

struct MockScheduler {
    groups: Vec<SequenceGroup>,
}

impl MockScheduler {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

impl Scheduler for MockScheduler {
    fn add_seq_group(&mut self, seq_group: SequenceGroup) {
        self.groups.push(seq_group);
    }

    fn abort_seq_group(&mut self, request_id: &RequestId) {
        self.groups.retain(|group| group.request_id != *request_id);
    }

    fn schedule(&mut self) -> SchedulerOutputs {
        let scheduled = self.groups.clone();
        SchedulerOutputs {
            num_batched_tokens: scheduled.len(),
            scheduled_seq_groups: scheduled,
            preempted: false,
        }
    }

    fn has_unfinished_seqs(&self) -> bool {
        !self.groups.is_empty()
    }

    fn get_num_unfinished_seq_groups(&self) -> usize {
        self.groups.len()
    }
}

struct ScriptedExecutor {
    steps: Vec<HashMap<SequenceId, (TokenId, f32)>>,
    cursor: usize,
}

impl ScriptedExecutor {
    fn new(steps: Vec<HashMap<SequenceId, (TokenId, f32)>>) -> Self {
        Self { steps, cursor: 0 }
    }
}

impl gpt_oss_engine::Executor for ScriptedExecutor {
    fn execute_model(
        &mut self,
        input: ExecutorInput,
    ) -> gpt_oss_core::prelude::Result<Vec<SamplerOutput>> {
        let scripted = self.steps.get(self.cursor).cloned().unwrap_or_default();
        self.cursor += 1;

        let mut outputs = Vec::new();
        for metadata in input.seq_group_metadata {
            for seq_id in metadata.seq_data.keys() {
                if let Some(&(token_id, logprob)) = scripted.get(seq_id) {
                    outputs.push(SamplerOutput {
                        seq_id: *seq_id,
                        token_id,
                        logprob,
                        top_logprobs: None,
                    });
                }
            }
        }

        Ok(outputs)
    }
}

fn make_test_tokenizer() -> Tokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("[UNK]".to_string(), 0);
    vocab.insert("[CLS]".to_string(), 1);
    vocab.insert("[SEP]".to_string(), 2);
    vocab.insert("hello".to_string(), 3);
    vocab.insert("world".to_string(), 4);
    vocab.insert("alpha".to_string(), 5);
    vocab.insert("beta".to_string(), 6);
    vocab.insert("stop".to_string(), 7);

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

fn make_engine(steps: Vec<HashMap<SequenceId, (TokenId, f32)>>) -> LLMEngine {
    LLMEngine::new(
        EngineConfig::default(),
        Box::new(ScriptedExecutor::new(steps)),
        Box::new(MockScheduler::new()),
        make_test_tokenizer(),
    )
    .unwrap()
}

#[test]
fn best_of_request_returns_highest_logprob_completion() {
    let mut engine = make_engine(vec![HashMap::from([
        (SequenceId(0), (5, -5.0)),
        (SequenceId(1), (6, -0.1)),
    ])]);

    let params = SamplingParams {
        best_of: 2,
        max_tokens: 1,
        ..Default::default()
    };

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    let outputs = engine.run().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].outputs.len(), 1);
    assert_eq!(outputs[0].outputs[0].token_ids, vec![6]);
    assert_eq!(outputs[0].outputs[0].text, "beta");
    assert_eq!(outputs[0].outputs[0].finish_reason, Some(FinishReason::Length));
    assert!((outputs[0].outputs[0].cumulative_logprob - (-0.1)).abs() < 1e-6);
}

#[test]
fn aborting_one_request_preserves_other_request_completion() {
    let mut engine = make_engine(vec![HashMap::from([(SequenceId(1), (4, -0.2))])]);

    let params = SamplingParams {
        max_tokens: 1,
        ..Default::default()
    };

    engine
        .add_request(RequestId(1), "hello".to_string(), params.clone())
        .unwrap();
    engine
        .add_request(RequestId(2), "hello".to_string(), params)
        .unwrap();

    engine.abort_request(&RequestId(1));

    let outputs = engine.run().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].request_id, RequestId(2));
    assert_eq!(outputs[0].outputs[0].text, "world");
    assert_eq!(outputs[0].outputs[0].finish_reason, Some(FinishReason::Length));
}

#[test]
fn stop_string_truncation_applies_in_end_to_end_run() {
    let mut engine = make_engine(vec![HashMap::from([(SequenceId(0), (7, -0.3))])]);

    let params = SamplingParams {
        max_tokens: 4,
        stop_strings: vec!["stop".to_string()],
        ..Default::default()
    };

    engine
        .add_request(RequestId(1), "hello".to_string(), params)
        .unwrap();

    let outputs = engine.run().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].outputs[0].token_ids, vec![7]);
    assert_eq!(outputs[0].outputs[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(outputs[0].outputs[0].text, "");
}
