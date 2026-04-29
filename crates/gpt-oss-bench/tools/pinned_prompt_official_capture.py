#!/usr/bin/env python3
"""Focused PPP official ordered-MLP capture helper.

This producer-side entrypoint is a minimal port from:
  /home/emmy/openai/worktrees/pinned-prompt-parity/crates/gpt-oss-bench/tools/pinned_prompt_official_capture.py

Only the final-token ordered MLP bundle path is included here. The original
layer1 selector is preserved through the generic layer-indexed implementation,
and new layer-indexed selectors are added for controlled layer11 evidence
capture without importing the broader proof/capture helper surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any


INTERMEDIATE_CAPTURE_SCHEMA = "pinned-prompt-official-intermediate-capture-input/v2"
INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA = (
    "pinned-prompt-official-intermediate-capture-output/v2"
)
PRODUCER_FUNCTION = "capture_layer_final_token_mlp_ordered_boundary_bundle"
PORT_SOURCE = (
    "/home/emmy/openai/worktrees/pinned-prompt-parity/"
    "crates/gpt-oss-bench/tools/pinned_prompt_official_capture.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a generic final-token ordered MLP official bundle."
    )
    parser.add_argument("--input", type=Path, help="PPP intermediate capture input JSON.")
    parser.add_argument("--output", type=Path, help="PPP intermediate capture output JSON.")
    parser.add_argument(
        "--official-checkout",
        type=Path,
        help="Official gpt-oss checkout root. Required for non-dry-run capture.",
    )
    parser.add_argument(
        "--execution",
        choices=("cpu", "distributed-gpu"),
        default="cpu",
        help="Accepted for wrapper compatibility; this focused helper does not initialize distributed capture.",
    )
    parser.add_argument(
        "--boundary",
        help=(
            "Boundary selector for direct mode, for example "
            "layerN_final_token_mlp_ordered_boundary_bundle or "
            "layer11_final_token_mlp_ordered_boundary_bundle."
        ),
    )
    parser.add_argument("--layer-idx", "--layer-index", dest="layer_idx", type=int)
    parser.add_argument(
        "--model",
        type=Path,
        help="Checkpoint/model path for direct mode. Alias for --official-model.",
    )
    parser.add_argument("--official-model", type=Path, help="Official checkpoint path.")
    parser.add_argument(
        "--coarse-bundle",
        type=Path,
        help="Optional coarse ladder bundle used as the layer MLP residual seed.",
    )
    parser.add_argument(
        "--layer-input",
        type=Path,
        help="Reserved for wrapper compatibility; final-token-only layer input is not enough for attention reconstruction.",
    )
    parser.add_argument("--lane", type=int, help="Optional focus lane metadata.")
    parser.add_argument("--output-dir", type=Path, help="Optional supporting output directory.")
    parser.add_argument("--status-output", type=Path, help="Direct-mode status output JSON.")
    parser.add_argument(
        "--dry-run-schema",
        action="store_true",
        help="Emit schema/selector metadata without importing torch or loading a checkpoint.",
    )
    return parser.parse_args()


def digest_f32_values(values: list[float]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(struct.pack("<f", float(value)))
    return hasher.hexdigest()


def value_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "sha256_f32_le": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "sha256_f32_le": digest_f32_values(values),
    }


def boundary_tensor_entry(
    boundary: str,
    values: list[float],
    shape: list[int],
    dtype: str,
    layout: str,
    token_index: int | None,
    layer_index: int,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "boundary": boundary,
        "layer_index": layer_index,
        "token_index": token_index,
        "shape": shape,
        "dtype": dtype,
        "serialization_dtype": "json_f32_values",
        "layout": layout,
        "values": values,
        "summary": value_summary(values),
        **metadata,
    }


def resolve_checkpoint_dir(path: Path) -> Path:
    original_dir = path / "original"
    if original_dir.is_dir():
        return original_dir
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def parse_ordered_mlp_selector(boundary: str, layer_idx: int | None) -> int | None:
    if boundary == "layer1_final_token_mlp_ordered_boundary_bundle":
        if layer_idx is not None and layer_idx != 1:
            raise ValueError(
                "layer1_final_token_mlp_ordered_boundary_bundle requires layer_idx unset or 1"
            )
        return 1
    if boundary == "layerN_final_token_mlp_ordered_boundary_bundle":
        if layer_idx is None:
            raise ValueError("layerN_final_token_mlp_ordered_boundary_bundle requires --layer-idx")
        return layer_idx
    match = re.fullmatch(r"layer(\d+)_final_token_mlp_ordered_boundary_bundle", boundary)
    if match:
        selector_layer = int(match.group(1))
        if layer_idx is not None and layer_idx != selector_layer:
            raise ValueError(
                f"boundary selector layer {selector_layer} conflicts with layer_idx {layer_idx}"
            )
        return selector_layer
    return None


def ordered_mlp_boundary_names(layer_index: int) -> list[str]:
    return [
        f"layer{layer_index}_final_token_hidden_state_after_attention_residual_add_before_mlp",
        f"layer{layer_index}_final_token_mlp_norm_output_before_mlp_projections",
        f"layer{layer_index}_final_token_mlp_router_logits_before_routing",
        f"layer{layer_index}_final_token_mlp_topk_expert_indices_and_routing_weights",
        f"layer{layer_index}_final_token_selected_expert_outputs_before_routing_weighted_sum",
        f"layer{layer_index}_final_token_mlp_output_after_routing_weighted_sum_before_residual",
        f"layer{layer_index}_final_token_hidden_state_after_mlp_residual_add",
    ]


def find_boundary(container: Any, boundary: str) -> dict[str, Any] | None:
    if isinstance(container, dict):
        if container.get("boundary") == boundary:
            return container
        for value in container.values():
            found = find_boundary(value, boundary)
            if found is not None:
                return found
    elif isinstance(container, list):
        for value in container:
            found = find_boundary(value, boundary)
            if found is not None:
                return found
    return None


def extract_boundary_values(bundle_path: Path, boundary: str) -> tuple[list[float], dict[str, Any]]:
    bundle = load_json(bundle_path)
    entry = find_boundary(bundle, boundary)
    if entry is None:
        raise ValueError(f"boundary {boundary!r} not found in {bundle_path}")
    values = entry.get("values")
    if not isinstance(values, list):
        raise ValueError(f"boundary {boundary!r} has no JSON values array")
    return [float(value) for value in values], entry


def tensor_from_values(values: list[float], torch: Any, device: Any, dtype: Any) -> Any:
    return torch.tensor(values, dtype=torch.float32, device=device).to(dtype).view(1, -1)


def compute_layer_attention_residual(model: Any, input_token_ids: list[int], layer_index: int, torch: Any) -> Any:
    tokens = torch.as_tensor(
        input_token_ids, dtype=torch.int64, device=model.embedding.weight.device
    )
    hidden = model.embedding(tokens)
    for block_index in range(layer_index):
        hidden = model.block[block_index](hidden)
    return model.block[layer_index].attn(hidden)


def capture_layer_final_token_mlp_ordered_boundary_bundle(
    model: Any,
    input_token_ids: list[int] | None,
    torch: Any,
    layer_index: int,
    *,
    coarse_bundle: Path | None = None,
    lane: int | None = None,
) -> dict[str, Any]:
    if layer_index < 0 or layer_index >= len(model.block):
        raise ValueError(
            f"layer_idx {layer_index} is out of range for {len(model.block)} blocks"
        )

    with torch.inference_mode():
        block = model.block[layer_index]
        mlp = block.mlp
        final_token_index = None
        coarse_seed_entry = None

        if coarse_bundle is not None:
            residual_boundary = (
                f"layer{layer_index}_final_token_hidden_state_after_attention_residual_add_before_mlp"
            )
            source_values, coarse_seed_entry = extract_boundary_values(
                coarse_bundle, residual_boundary
            )
            layer_after_attention = tensor_from_values(
                source_values,
                torch,
                mlp.gate.weight.device,
                model.embedding.weight.dtype,
            )
            final_token_index = coarse_seed_entry.get("token_index")
            final_position = 0
            source_seed = "coarse_bundle_attention_residual"
        else:
            if input_token_ids is None:
                raise ValueError(
                    "input_token_ids are required when --coarse-bundle is not provided"
                )
            layer_after_attention = compute_layer_attention_residual(
                model, input_token_ids, layer_index, torch
            )
            final_token_index = len(input_token_ids) - 1
            final_position = final_token_index
            source_seed = "official_full_prefix_attention_residual"

        mlp_norm = mlp.norm
        normed = mlp_norm(layer_after_attention)
        router_logits = mlp.gate(normed)

        final_normed = normed[final_position]
        final_logits = router_logits[final_position]
        experts = torch.topk(
            final_logits, k=mlp.experts_per_token, dim=-1, sorted=True
        )
        routing_weights = torch.nn.functional.softmax(experts.values, dim=0)
        expert_indices = experts.indices

        mlp1_weight = mlp.mlp1_weight[expert_indices, ...]
        mlp1_bias = mlp.mlp1_bias[expert_indices, ...]
        mlp1_output = torch.einsum("ech,h->ec", mlp1_weight, final_normed)
        mlp1_output += mlp1_bias

        x_glu = mlp1_output[..., ::2].clamp(min=None, max=mlp.swiglu_limit)
        x_linear = mlp1_output[..., 1::2].clamp(
            min=-mlp.swiglu_limit, max=mlp.swiglu_limit
        )
        swiglu_output = x_glu * torch.sigmoid(1.702 * x_glu) * (x_linear + 1)

        mlp2_weight = mlp.mlp2_weight[expert_indices, ...]
        mlp2_bias = mlp.mlp2_bias[expert_indices, ...]
        mlp2_pre_bias = torch.einsum("ehk,ek->eh", mlp2_weight, swiglu_output)
        if mlp.world_size > 1:
            torch.distributed.all_reduce(
                mlp2_pre_bias, op=torch.distributed.ReduceOp.SUM
            )
        expert_outputs = mlp2_pre_bias + mlp2_bias
        mlp_output = torch.einsum("eh,e->h", expert_outputs, routing_weights)
        hidden_after_mlp = layer_after_attention[final_position] + mlp_output

        source_values = layer_after_attention[final_position].float().cpu().tolist()
        norm_values = final_normed.float().cpu().tolist()
        router_values = final_logits.float().cpu().tolist()
        selected_indices = [int(value) for value in expert_indices.cpu().tolist()]
        selected_logits = experts.values.float().cpu().tolist()
        routing_weight_values = routing_weights.float().cpu().tolist()
        expert_output_values_by_expert = expert_outputs.float().cpu().tolist()
        expert_output_values = [
            value
            for expert_values in expert_output_values_by_expert
            for value in expert_values
        ]
        mlp_output_values = mlp_output.float().cpu().tolist()
        hidden_after_mlp_values = hidden_after_mlp.float().cpu().tolist()
        norm_weight_values = mlp_norm.scale.float().cpu().tolist()

        gate_bias = mlp.gate.bias
        if gate_bias is None:
            gate_bias_metadata = {
                "present": False,
                "shape": None,
                "dtype": None,
                "all_zero": None,
                "nonzero_count": None,
                "sha256_f32_le": None,
            }
        else:
            gate_bias_values = gate_bias.float().cpu()
            gate_bias_metadata = {
                "present": True,
                "shape": list(gate_bias.shape),
                "dtype": str(gate_bias.dtype),
                "all_zero": bool((gate_bias_values == 0).all().item()),
                "nonzero_count": int((gate_bias_values != 0).sum().item()),
                "sha256_f32_le": digest_f32_values(gate_bias_values.tolist()),
            }

        per_expert_summary = []
        for rank, (expert_index, expert_values) in enumerate(
            zip(selected_indices, expert_output_values_by_expert, strict=True)
        ):
            per_expert_summary.append(
                {
                    "rank": rank,
                    "expert_index": expert_index,
                    **value_summary(expert_values),
                }
            )

        captured_boundaries = [
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_hidden_state_after_attention_residual_add_before_mlp",
                source_values,
                [len(source_values)],
                str(layer_after_attention.dtype),
                "flat hidden dimension vector [hidden_size]",
                final_token_index,
                layer_index,
                source_seed=source_seed,
                coarse_bundle_source=str(coarse_bundle) if coarse_bundle else None,
                coarse_seed_boundary=coarse_seed_entry.get("boundary") if coarse_seed_entry else None,
            ),
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_mlp_norm_output_before_mlp_projections",
                norm_values,
                [len(norm_values)],
                str(normed.dtype),
                "flat hidden dimension vector [hidden_size]",
                final_token_index,
                layer_index,
                source_input_boundary=(
                    f"layer{layer_index}_final_token_hidden_state_after_attention_residual_add_before_mlp"
                ),
                source_input_sha256_f32_le=digest_f32_values(source_values),
                norm_module={
                    "path": f"model.block[{layer_index}].mlp.norm",
                    "name": "norm",
                    "type": "RMSNorm",
                },
                norm_type="RMSNorm",
                norm_parameters={
                    "weight_shape": list(mlp_norm.scale.shape),
                    "weight_dtype": str(mlp_norm.scale.dtype),
                    "weight_sha256_f32_le": digest_f32_values(norm_weight_values),
                    "epsilon": float(mlp_norm.eps),
                    "bias_exists": False,
                    "centering_or_mean_subtraction_used": False,
                    "variance_is_mean_x_squared_over_hidden_dimension": True,
                },
            ),
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_mlp_router_logits_before_routing",
                router_values,
                [len(router_values)],
                str(final_logits.dtype),
                "flat expert-logit vector [num_experts]",
                final_token_index,
                layer_index,
                router_module={
                    "path": f"model.block[{layer_index}].mlp.gate",
                    "name": "gate",
                    "type": "torch.nn.Linear",
                },
                router_weight_metadata={
                    "shape": list(mlp.gate.weight.shape),
                    "dtype": str(mlp.gate.weight.dtype),
                    "layout_orientation": "[num_experts, hidden_size]; Linear computes input @ weight.T + bias",
                    "sha256_f32_le": None,
                    "sha256_omitted_reason": "router weight is not compact for this ordered bundle",
                },
                router_bias_metadata=gate_bias_metadata,
                number_of_experts=mlp.num_experts,
                output_dtype_before_serialization=str(final_logits.dtype),
            ),
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_mlp_topk_expert_indices_and_routing_weights",
                routing_weight_values,
                [len(routing_weight_values)],
                str(routing_weights.dtype),
                "top-k routing weight vector [top_k_rank] in sorted top-k order",
                final_token_index,
                layer_index,
                number_of_experts=mlp.num_experts,
                top_k=mlp.experts_per_token,
                selected_expert_indices={
                    "shape": list(expert_indices.shape),
                    "dtype": str(expert_indices.dtype),
                    "serialization_dtype": "json_int_values",
                    "values": selected_indices,
                },
                selected_expert_logits={
                    "shape": list(experts.values.shape),
                    "dtype": str(experts.values.dtype),
                    "serialization_dtype": "json_f32_values",
                    "values": selected_logits,
                    "sha256_f32_le": digest_f32_values(selected_logits),
                },
                routing_weights={
                    "shape": list(routing_weights.shape),
                    "dtype": str(routing_weights.dtype),
                    "values": routing_weight_values,
                    "sum": float(sum(routing_weight_values)),
                    "min": min(routing_weight_values),
                    "max": max(routing_weight_values),
                    "sha256_f32_le": digest_f32_values(routing_weight_values),
                },
                index_order_convention="torch.topk(..., sorted=True) order",
                normalization_function="softmax over selected logits",
            ),
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_selected_expert_outputs_before_routing_weighted_sum",
                expert_output_values,
                list(expert_outputs.shape),
                str(expert_outputs.dtype),
                "[top_k_rank, hidden_size]",
                final_token_index,
                layer_index,
                selected_expert_indices=selected_indices,
                selected_expert_order_convention="torch.topk(..., sorted=True) order",
                selected_routing_weights=routing_weight_values,
                outputs_are_unweighted=True,
                routing_weights_already_applied=False,
                per_expert_summary=per_expert_summary,
                selected_expert_computation_summary={
                    "input_boundary": f"layer{layer_index}_final_token_mlp_norm_output_before_mlp_projections",
                    "mlp1_weight_shape": list(mlp1_weight.shape),
                    "mlp1_bias_shape": list(mlp1_bias.shape),
                    "mlp2_weight_shape": list(mlp2_weight.shape),
                    "mlp2_bias_shape": list(mlp2_bias.shape),
                    "swiglu_limit": float(mlp.swiglu_limit),
                    "output_dtype_before_serialization": str(expert_outputs.dtype),
                    "computation_dtype": "torch.bfloat16",
                },
            ),
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_mlp_output_after_routing_weighted_sum_before_residual",
                mlp_output_values,
                list(mlp_output.shape),
                str(mlp_output.dtype),
                "flat hidden dimension vector [hidden_size]",
                final_token_index,
                layer_index,
                source_selected_expert_output={
                    "boundary": f"layer{layer_index}_final_token_selected_expert_outputs_before_routing_weighted_sum",
                    "shape": list(expert_outputs.shape),
                    "dtype": str(expert_outputs.dtype),
                    "layout": "[top_k_rank, hidden_size]",
                    "sha256_f32_le": digest_f32_values(expert_output_values),
                },
                source_routing_weights=routing_weight_values,
                routing_weighted_sum_semantics={
                    "selected_expert_output_dtype": str(expert_outputs.dtype),
                    "routing_weight_dtype": str(routing_weights.dtype),
                    "multiplication_dtype": str(mlp_output.dtype),
                    "accumulation_dtype": "unknown",
                    "selected_rank_accumulation_order": "torch.einsum over selected top-k rank dimension",
                },
            ),
            boundary_tensor_entry(
                f"layer{layer_index}_final_token_hidden_state_after_mlp_residual_add",
                hidden_after_mlp_values,
                list(hidden_after_mlp.shape),
                str(hidden_after_mlp.dtype),
                "flat hidden dimension vector [hidden_size]",
                final_token_index,
                layer_index,
                source_residual_input_boundary={
                    "boundary": f"layer{layer_index}_final_token_hidden_state_after_attention_residual_add_before_mlp",
                    "shape": list(layer_after_attention[final_position].shape),
                    "dtype": str(layer_after_attention.dtype),
                    "layout": "flat hidden dimension vector [hidden_size]",
                    "sha256_f32_le": digest_f32_values(source_values),
                },
                source_mlp_output_boundary={
                    "boundary": f"layer{layer_index}_final_token_mlp_output_after_routing_weighted_sum_before_residual",
                    "shape": list(mlp_output.shape),
                    "dtype": str(mlp_output.dtype),
                    "layout": "flat hidden dimension vector [hidden_size]",
                    "sha256_f32_le": digest_f32_values(mlp_output_values),
                },
                residual_add_semantics={
                    "addend_order": "post_attention_residual + mlp_weighted_expert_sum",
                    "computation_dtype": str(hidden_after_mlp.dtype),
                    "output_dtype_before_serialization": str(hidden_after_mlp.dtype),
                    "rounded_or_cast_to_bf16_after_add": str(hidden_after_mlp.dtype)
                    == "torch.bfloat16",
                },
            ),
        ]

        captured_names = [entry["boundary"] for entry in captured_boundaries]
        required_names = ordered_mlp_boundary_names(layer_index)
        missing = [name for name in required_names if name not in captured_names]
        classification = (
            f"official_layer{layer_index}_mlp_ordered_boundary_bundle_captured"
            if not missing
            else f"official_layer{layer_index}_mlp_ordered_boundary_bundle_partial"
        )
        return {
            "boundary": f"layer{layer_index}_final_token_mlp_ordered_boundary_bundle",
            "classification": classification,
            "case_scope": "developer-message-user-smoke",
            "layer_index": layer_index,
            "token_index": final_token_index,
            "focus_lane": lane,
            "bundle_scope": {
                "layer": layer_index,
                "mlp_moe_path_only": True,
                "final_token_only": True,
                "starts_from_layer_attention_residual_add": True,
                "stops_after_layer_mlp_residual_add": True,
                "captures_logits_or_later_layers": False,
                "selected_expert_internals_included": False,
            },
            "producer_metadata": {
                "producer_function": PRODUCER_FUNCTION,
                "boundary_selector": f"layer{layer_index}_final_token_mlp_ordered_boundary_bundle",
                "requested_layer_index": layer_index,
                "selected_expert_internals_included": False,
                "port_source": PORT_SOURCE,
            },
            "captured_boundaries": captured_names,
            "missing_boundaries": missing,
            "boundaries": captured_boundaries,
            "selected_expert_indices": selected_indices,
            "routing_weights": routing_weight_values,
            f"source_layer{layer_index}_mlp_input": {
                "boundary": f"layer{layer_index}_final_token_hidden_state_after_attention_residual_add_before_mlp",
                "shape": list(layer_after_attention[final_position].shape),
                "dtype": str(layer_after_attention.dtype),
                "sha256_f32_le": digest_f32_values(source_values),
                "seed": source_seed,
            },
            "next_bounded_step": (
                f"compare layer{layer_index} MLP ordered bundle and report earliest mismatching seam"
                if not missing
                else "compare captured boundaries only, or request the first missing required boundary"
            ),
        }


def build_intermediate_output(capture_input: dict[str, Any], capture_body: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA,
        "suite_id": capture_input.get("suite_id"),
        "case_id": capture_input.get("case_id"),
        "backend": "official_torch",
        "official_model": capture_input.get("official_model"),
        "prompt_renderer": capture_input.get("prompt_renderer"),
        "input_token_ids": capture_input.get("input_token_ids"),
        "boundary": capture_input.get("boundary"),
        "layer_idx": capture_input.get("layer_idx"),
        **capture_body,
    }


def build_direct_output(
    args: argparse.Namespace,
    boundary: str,
    layer_index: int,
    capture_body: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA,
        "backend": "official_torch",
        "official_model": str(args.official_model or args.model),
        "boundary": boundary,
        "layer_idx": layer_index,
        "producer_direct_mode": True,
        "layer_input": str(args.layer_input) if args.layer_input else None,
        "coarse_bundle": str(args.coarse_bundle) if args.coarse_bundle else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        **capture_body,
    }


def dry_run_schema(boundary: str, layer_index: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": INTERMEDIATE_CAPTURE_OUTPUT_SCHEMA,
        "backend": "official_torch",
        "boundary": boundary,
        "layer_idx": layer_index,
        "classification": "official_ordered_mlp_capture_schema_ready",
        "implemented": True,
        "full_capture_run": False,
        "checkpoint_loaded": False,
        "focus_lane": args.lane,
        "expected_status_output": str(args.status_output) if args.status_output else None,
        "expected_output_dir": str(args.output_dir) if args.output_dir else None,
        "expected_boundaries": ordered_mlp_boundary_names(layer_index),
        "producer_metadata": {
            "producer_function": PRODUCER_FUNCTION,
            "boundary_selector": boundary,
            "requested_layer_index": layer_index,
            "selected_expert_internals_included": False,
            "port_source": PORT_SOURCE,
        },
        "next_bounded_step": "run focused layer11 MLP evidence bundle under /tmp",
    }


def load_model(checkpoint_path: Path, official_checkout: Path):
    sys.path.insert(0, str(official_checkout))
    import torch  # noqa: WPS433
    from gpt_oss.torch.model import Transformer  # noqa: WPS433

    model = Transformer.from_checkpoint(
        str(resolve_checkpoint_dir(checkpoint_path)), device=torch.device("cpu")
    )
    model.eval()
    return model, torch


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        capture_input = load_json(args.input)
        boundary = capture_input["boundary"]
        layer_index = parse_ordered_mlp_selector(boundary, capture_input.get("layer_idx"))
        if layer_index is None:
            raise ValueError(f"unsupported intermediate boundary: {boundary}")
        if args.dry_run_schema:
            return dry_run_schema(boundary, layer_index, args)
        if args.official_checkout is None:
            raise ValueError("--official-checkout is required for non-dry-run capture")
        model_path = Path(capture_input["official_model"])
        model, torch = load_model(model_path, args.official_checkout)
        body = capture_layer_final_token_mlp_ordered_boundary_bundle(
            model,
            capture_input.get("input_token_ids"),
            torch,
            layer_index,
            coarse_bundle=args.coarse_bundle,
            lane=args.lane,
        )
        return build_intermediate_output(capture_input, body)

    boundary = args.boundary or "layerN_final_token_mlp_ordered_boundary_bundle"
    layer_index = parse_ordered_mlp_selector(boundary, args.layer_idx)
    if layer_index is None:
        raise ValueError(f"unsupported boundary selector: {boundary}")
    if args.dry_run_schema:
        return dry_run_schema(boundary, layer_index, args)
    model_path = args.official_model or args.model
    if model_path is None:
        raise ValueError("--model or --official-model is required for non-dry-run capture")
    if args.official_checkout is None:
        raise ValueError("--official-checkout is required for non-dry-run capture")
    model, torch = load_model(model_path, args.official_checkout)
    body = capture_layer_final_token_mlp_ordered_boundary_bundle(
        model,
        None,
        torch,
        layer_index,
        coarse_bundle=args.coarse_bundle,
        lane=args.lane,
    )
    return build_direct_output(args, boundary, layer_index, body)


def main() -> int:
    args = parse_args()
    if args.execution == "distributed-gpu" and not args.dry_run_schema:
        raise ValueError(
            "distributed-gpu execution is intentionally not ported in this focused helper"
        )
    output = run_capture(args)
    output_path = args.status_output or args.output
    if output_path is None:
        print(json.dumps(output, indent=2))
        return 0
    write_json(output_path, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
