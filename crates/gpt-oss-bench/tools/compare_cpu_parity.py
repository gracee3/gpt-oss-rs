#!/usr/bin/env python3
"""Compare native, official-oracle, and pinned llama.cpp CPU captures."""

import argparse
import json
import os
import tempfile
from pathlib import Path


TRACE_STAGES_BEFORE_EXPERTS = (
    "input_norm",
    "query_after_rope",
    "key_after_rope",
    "value_projection",
    "attention_context",
    "attention_projection",
    "post_attention_residual",
    "router_logits",
    "routing_weights",
)

EXPERT_TRACE_STAGES = (
    "gate_up_projection",
    "swiglu",
    "down_projection",
    "weighted_output",
)

TRACE_STAGES_AFTER_EXPERTS = (
    "moe_output",
    "layer_output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--llama", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-tolerance", type=float, default=1e-5)
    parser.add_argument("--llama-near-tie", type=float, default=1e-2)
    return parser.parse_args()


NEGATIVE_INPUT_STATUSES = {
    "fail",
    "unsupported",
    "unavailable",
    "invalid",
    "incomplete",
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"capture {path} is not a JSON object")
    return value


def write_new_atomic(path: Path, value: dict) -> None:
    """Atomically publish JSON without replacing a completed comparator result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def input_status(capture: dict) -> str:
    return capture.get("evidence_status", capture.get("status", "insufficient_evidence"))


def workload_mismatch(native: dict, official: dict) -> str | None:
    if native.get("scenario") != official.get("scenario", native.get("scenario")):
        return "scenario mismatch"
    for field in ("prompt_token_ids", "prompt_token_ids_sha256"):
        if field in native and field in official and native[field] != official[field]:
            return f"{field} mismatch"
    native_model = native.get("pinned_model", native.get("model"))
    official_model = official.get("pinned_model", official.get("model"))
    if native_model is not None and official_model is not None and native_model != official_model:
        return "model provenance mismatch"
    return None


def trace_from(capture: dict) -> dict | None:
    return capture.get("trace")


def stage_diff(native_values: list[float], official_values: list[float]) -> dict:
    if len(native_values) != len(official_values):
        raise ValueError("trace stage lengths differ")
    deltas = [abs(native - official) for native, official in zip(native_values, official_values)]
    return {
        "elements": len(deltas),
        "mean_abs_diff": sum(deltas) / max(len(deltas), 1),
        "max_abs_diff": max(deltas, default=0.0),
    }


def compare_traces(native: dict, official: dict, tolerance: float) -> dict | None:
    native_trace = trace_from(native)
    official_trace = trace_from(official)
    if native_trace is None or official_trace is None:
        return None
    official_layers = {
        layer["layer_index"]: layer for layer in official_trace.get("layers", [])
    }
    layers = []
    earliest = None
    for native_layer in native_trace.get("layers", []):
        index = native_layer["layer_index"]
        if index not in official_layers:
            continue
        official_layer = official_layers[index]
        stages = {}
        for stage in TRACE_STAGES_BEFORE_EXPERTS:
            if stage not in native_layer or stage not in official_layer:
                continue
            metric = stage_diff(native_layer[stage], official_layer[stage])
            stages[stage] = metric
            if earliest is None and metric["max_abs_diff"] > tolerance:
                earliest = {
                    "layer_index": index,
                    "stage": stage,
                    **metric,
                }
        official_experts = {
            expert["rank"]: expert for expert in official_layer.get("experts", [])
        }
        experts = []
        for native_expert in native_layer.get("experts", []):
            rank = native_expert["rank"]
            if rank not in official_experts:
                continue
            official_expert = official_experts[rank]
            expert_stages = {}
            for stage in EXPERT_TRACE_STAGES:
                if stage not in native_expert or stage not in official_expert:
                    continue
                metric = stage_diff(native_expert[stage], official_expert[stage])
                expert_stages[stage] = metric
                if earliest is None and metric["max_abs_diff"] > tolerance:
                    earliest = {
                        "layer_index": index,
                        "expert_rank": rank,
                        "expert_index": native_expert["expert_index"],
                        "stage": stage,
                        **metric,
                    }
            experts.append(
                {
                    "rank": rank,
                    "native_expert_index": native_expert["expert_index"],
                    "official_expert_index": official_expert["expert_index"],
                    "stages": expert_stages,
                }
            )
        for stage in TRACE_STAGES_AFTER_EXPERTS:
            if stage not in native_layer or stage not in official_layer:
                continue
            metric = stage_diff(native_layer[stage], official_layer[stage])
            stages[stage] = metric
            if earliest is None and metric["max_abs_diff"] > tolerance:
                earliest = {
                    "layer_index": index,
                    "stage": stage,
                    **metric,
                }
        layers.append({"layer_index": index, "stages": stages, "experts": experts})
    final_norm = stage_diff(native_trace["final_norm"], official_trace["final_norm"])
    return {
        "native_trace_step": native_trace.get("trace_step", 0),
        "official_trace_step": official_trace.get("trace_step", 0),
        "context_matches": native_trace.get(
            "context_token_ids", native_trace.get("prompt_token_ids")
        )
        == official_trace.get(
            "context_token_ids", official_trace.get("prompt_token_ids")
        ),
        "earliest_mismatch": earliest,
        "layers": layers,
        "final_norm": final_norm,
    }


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def llama_margin(llama: dict, step: int, competing_token: int) -> float | None:
    probabilities = llama.get("completion_probabilities", [])
    tokens = llama.get("tokens", [])
    if step >= len(probabilities) or step >= len(tokens):
        return None
    alternatives = probabilities[step].get("top_logprobs", [])
    chosen = tokens[step]
    chosen_logprob = next(
        (item["logprob"] for item in alternatives if item["id"] == chosen), None
    )
    competing_logprob = next(
        (item["logprob"] for item in alternatives if item["id"] == competing_token), None
    )
    if chosen_logprob is None or competing_logprob is None:
        return None
    return abs(chosen_logprob - competing_logprob)


def compare(args: argparse.Namespace) -> tuple[int, dict]:
    native = load(args.native)
    official = load(args.official)
    for role, capture in (("native", native), ("official", official)):
        status = input_status(capture)
        if status in NEGATIVE_INPUT_STATUSES:
            return 2, {
                "status": status,
                "blocking": True,
                "reason": f"{role} input retained negative status {status}",
            }
        if status not in ("insufficient_evidence", "pass"):
            return 2, {
                "status": "invalid",
                "blocking": True,
                "reason": f"{role} input has unknown status {status}",
            }
    mismatch = workload_mismatch(native, official)
    if mismatch:
        return 2, {"status": "invalid", "blocking": True, "reason": mismatch}
    native_tokens = native["generated_token_ids"]
    official_tokens = official["generated_token_ids"]
    native_official_divergence = first_divergence(native_tokens, official_tokens)
    result = {
        "status": "pass" if native_official_divergence is None else "fail",
        "scenario": native["scenario"],
        "native_tokens": native_tokens,
        "official_tokens": official_tokens,
        "native_matches_official": native_official_divergence is None,
        "native_official_first_divergence": native_official_divergence,
        "trace_comparison": compare_traces(native, official, args.trace_tolerance),
    }

    blocking = native_official_divergence is not None
    if args.llama:
        llama = load(args.llama)
        llama_status = input_status(llama)
        if llama_status in NEGATIVE_INPUT_STATUSES:
            result["llama_cpp"] = {
                "policy": "advisory",
                "status": llama_status,
                "nonblocking": True,
            }
            result["blocking"] = blocking
            return (1 if blocking else 0), result
        llama_tokens = llama["tokens"]
        divergence = first_divergence(native_tokens, llama_tokens)
        margin = None
        near_tie = False
        if divergence is not None and divergence < len(native_tokens):
            margin = llama_margin(llama, divergence, native_tokens[divergence])
            near_tie = margin is not None and margin <= args.llama_near_tie
        result["llama_cpp"] = {
            "policy": "advisory",
            "tokens": llama_tokens,
            "first_divergence": divergence,
            "competing_logit_gap": margin,
            "near_tie_threshold": args.llama_near_tie,
            "near_tie": near_tie,
            "nonblocking": True,
        }
    result["blocking"] = blocking
    return (1 if blocking else 0), result


def main() -> int:
    args = parse_args()
    try:
        code, result = compare(args)
    except FileNotFoundError as error:
        code, result = 2, {
            "status": "unavailable",
            "blocking": True,
            "reason": f"missing input: {error}",
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        code, result = 2, {
            "status": "invalid",
            "blocking": True,
            "reason": str(error),
        }

    write_new_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
