#!/usr/bin/env python3
"""Compare native, official-oracle, and pinned llama.cpp CPU captures."""

import argparse
import json
from pathlib import Path


TRACE_STAGES = (
    "input_norm",
    "query_after_rope",
    "key_after_rope",
    "value_projection",
    "attention_context",
    "attention_projection",
    "post_attention_residual",
    "router_logits",
    "routing_weights",
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


def load(path: Path) -> dict:
    return json.loads(path.read_text())


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
        for stage in TRACE_STAGES:
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
        layers.append({"layer_index": index, "stages": stages})
    final_norm = stage_diff(native_trace["final_norm"], official_trace["final_norm"])
    return {"earliest_mismatch": earliest, "layers": layers, "final_norm": final_norm}


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def llama_margin(llama: dict, step: int, competing_token: int) -> float | None:
    probabilities = llama.get("completion_probabilities", [])
    if step >= len(probabilities):
        return None
    alternatives = probabilities[step].get("top_logprobs", [])
    chosen = llama.get("tokens", [])[step]
    chosen_logprob = next(
        (item["logprob"] for item in alternatives if item["id"] == chosen), None
    )
    competing_logprob = next(
        (item["logprob"] for item in alternatives if item["id"] == competing_token), None
    )
    if chosen_logprob is None or competing_logprob is None:
        return None
    return abs(chosen_logprob - competing_logprob)


def main() -> int:
    args = parse_args()
    native = load(args.native)
    official = load(args.official)
    native_tokens = native["generated_token_ids"]
    official_tokens = official["generated_token_ids"]
    native_official_divergence = first_divergence(native_tokens, official_tokens)
    result = {
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
        llama_tokens = llama["tokens"]
        divergence = first_divergence(native_tokens, llama_tokens)
        margin = None
        near_tie = False
        if divergence is not None and divergence < len(native_tokens):
            margin = llama_margin(llama, divergence, native_tokens[divergence])
            near_tie = margin is not None and margin <= args.llama_near_tie
        llama_nonblocking = (
            native_official_divergence is None
            and (divergence is None or near_tie)
        )
        blocking = blocking or not llama_nonblocking
        result["llama_cpp"] = {
            "tokens": llama_tokens,
            "first_divergence": divergence,
            "competing_logit_gap": margin,
            "near_tie_threshold": args.llama_near_tie,
            "near_tie": near_tie,
            "nonblocking": llama_nonblocking,
        }
    result["blocking"] = blocking

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
