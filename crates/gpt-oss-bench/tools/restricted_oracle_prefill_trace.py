#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_SCHEMA_VERSION = "restricted_oracle_prefill_trace.v2"
TOOL_COMPARE_TOLERANCE = 1e-2


def find_openai_gpt_oss_root(repo_root: Path) -> Path:
    candidates = [
        repo_root.parent / "gpt-oss",
        repo_root.parents[1] / "gpt-oss",
    ]
    for candidate in candidates:
        if (candidate / "gpt_oss").is_dir():
            return candidate
    raise FileNotFoundError(f"could not locate sibling gpt-oss checkout from {repo_root}")


OPENAI_GPT_OSS_ROOT = find_openai_gpt_oss_root(REPO_ROOT)
sys.path.insert(0, str(OPENAI_GPT_OSS_ROOT))

from gpt_oss.torch.model import ModelConfig, Transformer  # noqa: E402
from gpt_oss.torch.weights import Checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare restricted CUDA prefill activation trace against an independent PyTorch "
            "oracle."
        )
    )
    parser.add_argument(
        "--cuda-trace-json",
        type=Path,
        help="Trace file path to compare.",
    )
    parser.add_argument(
        "--batch",
        nargs="*",
        default=None,
        help=(
            "Alias for --batch-traces: compare multiple traces in one invocation using one "
            "warm session."
        ),
    )
    parser.add_argument(
        "--batch-traces",
        type=Path,
        nargs="*",
        default=None,
        help="Compare multiple traces in one invocation using one warm session.",
    )
    parser.add_argument(
        "--original-model",
        type=Path,
        required=True,
        help="Path to the original GPT-OSS checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report output path (required for one-shot full/fast modes).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for compare. Defaults to cpu.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "fast", "batch"),
        default="full",
        help="full: canonical compare; fast: debug-oriented; batch: multi-trace warm mode.",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Start persistent NDJSON request loop on stdin.",
    )
    parser.add_argument(
        "--compare-tolerance",
        type=float,
        default=TOOL_COMPARE_TOLERANCE,
        help=f"Divergence threshold for report conclusions. Default is {TOOL_COMPARE_TOLERANCE}.",
    )
    return parser.parse_args()


def tool_signature() -> str:
    script_path = Path(__file__).resolve()
    h = hashlib.sha256()
    h.update(script_path.read_bytes())
    return f"{TOOL_SCHEMA_VERSION}:{h.hexdigest()[:16]}"


def load_restricted_config(path: Path) -> ModelConfig:
    config_path = path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        json_config = json.load(handle)
    aliases = {
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "experts_per_token",
    }
    accepted = set(inspect.signature(ModelConfig).parameters)
    filtered = {}
    for key, value in json_config.items():
        mapped = aliases.get(key, key)
        if mapped in accepted:
            filtered[mapped] = value
    config = ModelConfig(**filtered)
    config.sliding_window = 0
    return config


def resolve_oracle_checkpoint_dir(path: Path) -> Path:
    original_dir = path / "original"
    if original_dir.is_dir():
        return original_dir
    return path


def load_restricted_transformer(
    restricted_config_path: Path, checkpoint_path: Path, device: torch.device
) -> Transformer:
    config = load_restricted_config(restricted_config_path)
    model = Transformer(config=config, device=device)
    model.eval()

    checkpoint = Checkpoint(str(resolve_oracle_checkpoint_dir(checkpoint_path)), device)
    per_rank_intermediate_size = config.intermediate_size

    for name, param in model.named_parameters():
        loaded_tensor = checkpoint.get(name)
        if "mlp1" in name:
            loaded_tensor = loaded_tensor[:, : 2 * per_rank_intermediate_size, ...]
        elif "mlp2_weight" in name:
            loaded_tensor = loaded_tensor[..., :per_rank_intermediate_size]
        param.data.copy_(loaded_tensor)

    with torch.no_grad():
        for block in model.block:
            block.attn.sinks.zero_()
            block.attn.sliding_window = 0

    return model


def last_token(tensor: torch.Tensor) -> list[float]:
    return tensor[-1].float().cpu().tolist()


def mean_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(lhs, rhs)) / max(len(lhs), 1)


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(lhs, rhs))


def compare_stage(name: str, cuda_values: list[float], oracle_values: list[float]) -> dict:
    return {
        "stage": name,
        "max_abs_diff": max_abs_diff(cuda_values, oracle_values),
        "mean_abs_diff": mean_abs_diff(cuda_values, oracle_values),
    }


def flatten_last_token(tensor: torch.Tensor) -> list[float]:
    return tensor[-1].reshape(-1).float().cpu().tolist()


def layer0_attention_trace(model: Transformer, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
    attn = model.block[0].attn
    norm_input = x
    normed = attn.norm(x)
    qkv = attn.qkv(normed)
    qkv_pre_bias = torch.nn.functional.linear(normed, attn.qkv.weight, bias=None)
    qkv_post_bias = qkv
    q = qkv_post_bias[:, : attn.num_attention_heads * attn.head_dim].contiguous()
    k = qkv[
        :,
        attn.num_attention_heads * attn.head_dim : (attn.num_attention_heads + attn.num_key_value_heads)
        * attn.head_dim,
    ].contiguous()
    v = qkv[
        :,
        (attn.num_attention_heads + attn.num_key_value_heads) * attn.head_dim : (attn.num_attention_heads + 2 * attn.num_key_value_heads)
        * attn.head_dim,
    ].contiguous()

    q_heads = q.view(
        -1,
        attn.num_key_value_heads,
        attn.num_attention_heads // attn.num_key_value_heads,
        attn.head_dim,
    )
    k_heads = k.view(-1, attn.num_key_value_heads, attn.head_dim)
    v_heads = v.view(-1, attn.num_key_value_heads, attn.head_dim)
    q_rope, k_rope = attn.rope(q_heads, k_heads)

    n_tokens = q_rope.shape[0]
    q_mult = attn.num_attention_heads // attn.num_key_value_heads
    k_expanded = k_rope[:, :, None, :].expand(-1, -1, q_mult, -1)
    v_expanded = v_heads[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = attn.sinks.reshape(attn.num_key_value_heads, q_mult, 1, 1).expand(
        -1, -1, n_tokens, -1
    )
    mask = torch.triu(q_rope.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    qk = torch.einsum("qhmd,khmd->hmqk", q_rope, k_expanded)
    qk *= attn.sm_scale
    qk += mask[None, None, :, :]
    qk_with_sink = torch.cat([qk, sinks], dim=-1)
    probs = torch.softmax(qk_with_sink, dim=-1)
    context = torch.einsum("hmqk,khmd->qhmd", probs[..., :-1], v_expanded).reshape(n_tokens, -1)
    o_proj = attn.out(context)
    residual_add = x + o_proj

    return {
        "attention_norm_input": last_token(norm_input),
        "attention_norm_output": last_token(normed),
        "qkv_pre_bias": flatten_last_token(qkv_pre_bias),
        "qkv_post_bias": flatten_last_token(qkv_post_bias),
        "q_proj": flatten_last_token(q),
        "k_proj": flatten_last_token(k),
        "v_proj": flatten_last_token(v),
        "q_rope": flatten_last_token(q_rope),
        "k_rope": flatten_last_token(k_rope),
        "masked_scores": qk_with_sink[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_probs": probs[:, :, -1, :].reshape(-1).float().cpu().tolist(),
        "attention_context": flatten_last_token(context),
        "o_proj": flatten_last_token(o_proj),
        "residual_add": flatten_last_token(residual_add),
        "post_attn_residual": last_token(residual_add),
    }, residual_add


def manual_qkv_from_norm(attn: torch.nn.Module, norm_last: list[float]) -> dict:
    if not norm_last:
        return {}
    norm = torch.tensor(norm_last, dtype=torch.float32, device=attn.qkv.weight.device)
    norm = norm.to(attn.qkv.weight.dtype)
    qkv_pre = torch.nn.functional.linear(norm, attn.qkv.weight, bias=None)
    qkv_post = torch.nn.functional.linear(norm, attn.qkv.weight, bias=attn.qkv.bias)
    q_dim = attn.num_attention_heads * attn.head_dim
    kv_dim = attn.num_key_value_heads * attn.head_dim
    return {
        "qkv_pre_bias": qkv_pre.float().cpu().tolist(),
        "qkv_post_bias": qkv_post.float().cpu().tolist(),
        "q_proj": qkv_post[:q_dim].float().cpu().tolist(),
        "k_proj": qkv_post[q_dim:q_dim + kv_dim].float().cpu().tolist(),
        "v_proj": qkv_post[q_dim + kv_dim:].float().cpu().tolist(),
    }


def compare_trace(
    model: Transformer,
    cuda_trace: dict,
    original_model: Path,
    mode: str,
    compare_tolerance: float = TOOL_COMPARE_TOLERANCE,
) -> dict:
    prompt_token_ids = cuda_trace["prompt_token_ids"]
    restricted_model_path = Path(cuda_trace["restricted_model_path"])
    first_parameter = next((p for p in model.parameters()), None)
    device = first_parameter.device if first_parameter is not None else torch.device("cpu")

    input_ids = torch.tensor(prompt_token_ids, dtype=torch.int64, device=device)

    with torch.inference_mode():
        x = model.embedding(input_ids)
        oracle_trace = {
            "embedding": last_token(x),
            "layers": [],
        }

        layer_cutoff = None
        if mode == "fast":
            layer_cutoff = 0

        for layer_idx, block in enumerate(model.block):
            if layer_cutoff is not None and layer_idx > layer_cutoff:
                break

            if layer_idx == 0:
                attention_trace, attn_hidden = layer0_attention_trace(model, x)
                cuda_attention = cuda_trace["trace"]["layers"][0].get("attention") or {}
                manual_from_cuda_norm = manual_qkv_from_norm(
                    model.block[0].attn, cuda_attention.get("attention_norm_output", [])
                )
                if manual_from_cuda_norm:
                    attention_trace["manual_from_cuda_norm"] = manual_from_cuda_norm
                attention_trace["manual_from_oracle_norm"] = manual_qkv_from_norm(
                    model.block[0].attn, attention_trace["attention_norm_output"]
                )
            else:
                attention_trace = None
                attn_hidden = block.attn(x)

            layer_output = block.mlp(attn_hidden)
            mlp_out = layer_output - attn_hidden
            layer_trace = {
                "layer_idx": layer_idx,
                "post_attn_residual": last_token(attn_hidden),
                "mlp_out": last_token(mlp_out),
                "layer_output": last_token(layer_output),
            }
            if attention_trace is not None:
                layer_trace["attention"] = attention_trace
            oracle_trace["layers"].append(layer_trace)
            x = layer_output

    stage_diffs: list[dict] = []
    stage_diffs.append(
        compare_stage("embedding", cuda_trace["trace"]["embedding"], oracle_trace["embedding"])
    )

    for cuda_layer, oracle_layer in zip(cuda_trace["trace"]["layers"], oracle_trace["layers"]):
        cuda_attention = cuda_layer.get("attention")
        oracle_attention = oracle_layer.get("attention")
        if cuda_attention and oracle_attention:
            for key in (
                "attention_norm_input",
                "attention_norm_output",
                "qkv_pre_bias",
                "qkv_post_bias",
                "q_proj",
                "k_proj",
                "v_proj",
                "q_rope",
                "k_rope",
                "masked_scores",
                "attention_probs",
                "attention_context",
                "o_proj",
                "residual_add",
            ):
                stage_diffs.append(
                    compare_stage(
                        f"layer{cuda_layer['layer_idx']}.{key}",
                        cuda_attention[key],
                        oracle_attention[key],
                    )
                )
            if "manual_from_cuda_norm" in oracle_attention:
                for key in ("qkv_pre_bias", "qkv_post_bias", "q_proj", "k_proj", "v_proj"):
                    stage_diffs.append(
                        compare_stage(
                            f"layer{cuda_layer['layer_idx']}.manual_from_cuda_norm.{key}",
                            cuda_attention[key],
                            oracle_attention["manual_from_cuda_norm"][key],
                        )
                    )
            for key in ("qkv_pre_bias", "qkv_post_bias", "q_proj", "k_proj", "v_proj"):
                stage_diffs.append(
                    compare_stage(
                        f"layer{cuda_layer['layer_idx']}.manual_from_oracle_norm.{key}",
                        cuda_attention[key],
                        oracle_attention["manual_from_oracle_norm"][key],
                    )
                )

        for key in ("post_attn_residual", "mlp_out", "layer_output"):
            if not cuda_layer[key] and not oracle_layer[key]:
                continue
            stage_diffs.append(
                compare_stage(
                    f"layer{cuda_layer['layer_idx']}.{key}",
                    cuda_layer[key],
                    oracle_layer[key],
                )
            )

        if mode == "fast" and len(stage_diffs) and any(
            stage["max_abs_diff"] > compare_tolerance for stage in stage_diffs
        ):
            break

    first_divergence = next(
        (stage for stage in stage_diffs if stage["max_abs_diff"] > compare_tolerance),
        None,
    )

    return {
        "prompt": cuda_trace["prompt"],
        "prompt_token_ids": prompt_token_ids,
        "restricted_model_path": str(restricted_model_path),
        "original_model_path": str(original_model),
        "oracle_device": str(device),
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_signature": tool_signature(),
        "compare_mode": mode,
        "first_divergence_stage": first_divergence["stage"] if first_divergence else None,
        "stage_diffs": stage_diffs,
        "conclusion": (
            "Shared lower-level CUDA runner/model semantic bug is favored."
            if first_divergence
            else "No prefill-stage activation divergence detected."
        ),
    }


def default_output_for_trace(trace_path: Path, mode: str) -> Path:
    stem = trace_path.with_suffix("") if trace_path.suffix else trace_path
    return Path(f"{stem}.{mode}.oracle-diff.json")


def ensure_model(session: dict, trace_path: Path, original_model: Path) -> tuple[Transformer, dict]:
    with trace_path.open("r", encoding="utf-8") as handle:
        cuda_trace = json.load(handle)

    restricted_model_path = Path(cuda_trace["restricted_model_path"])
    if session.get("model") is None:
        session["restricted_model_path"] = str(restricted_model_path)
        session["original_model"] = str(original_model)
        session["model"] = load_restricted_transformer(
            restricted_model_path,
            original_model,
            session["device"],
        )
    else:
        if session["restricted_model_path"] != str(restricted_model_path):
            raise RuntimeError("model mismatch for warm session")
        if session["original_model"] != str(original_model):
            raise RuntimeError("original checkpoint mismatch for warm session")

    return session["model"], cuda_trace


def compare_one(
    session: dict,
    trace_path: Path,
    original_model: Path,
    mode: str,
    output: Path | None,
    compare_tolerance: float,
) -> tuple[Path, dict]:
    if mode == "batch":
        mode = "full"
    model, cuda_trace = ensure_model(session, trace_path, original_model)
    report = compare_trace(model, cuda_trace, original_model, mode, compare_tolerance)

    if output is None:
        output = default_output_for_trace(trace_path, mode)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    return output, report


def summarize_response(
    response: dict,
    trace_path: Path,
    mode: str,
    output: Path,
    warm_session: bool,
    requested_mode: str,
    batch: bool,
) -> dict:
    return {
        "ok": True,
        "tool": Path(__file__).name,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_signature": tool_signature(),
        "requested_mode": requested_mode,
        "compare_mode": mode,
        "batch": batch,
        "warm_oracle": warm_session,
        "trace_json": str(trace_path),
        "output": str(output),
        "first_divergence_stage": response.get("first_divergence_stage"),
        "stage_diff_count": len(response.get("stage_diffs", [])),
        "conclusion": response.get("conclusion"),
    }


def run_batch_mode(
    trace_paths: list[Path],
    original_model: Path,
    mode: str,
    device: str,
    compare_tolerance: float,
) -> int:
    if mode not in {"full", "fast", "batch"}:
        raise SystemExit("--mode batch supports full or fast compare mode only when trace list is provided")
    compare_mode = "full" if mode == "batch" else mode
    session = {"device": torch.device(device), "model": None}
    outputs = []
    for trace_path in trace_paths:
        try:
            output, report = compare_one(
                session,
                trace_path,
                original_model,
                compare_mode,
                None,
                compare_tolerance,
            )
            outputs.append(
                summarize_response(
                    response=report,
                    trace_path=trace_path,
                    mode=compare_mode,
                    output=output,
                    warm_session=True,
                    requested_mode=mode,
                    batch=True,
                )
            )
        except Exception as exc:
            outputs.append(
                {
                    "ok": False,
                    "trace_json": str(trace_path),
                    "error": str(exc),
                    "requested_mode": mode,
                }
            )

    print(
        json.dumps(
            {
                "ok": True,
                "tool_schema_version": TOOL_SCHEMA_VERSION,
                "tool_signature": tool_signature(),
                "requested_mode": "batch",
                "compare_mode": compare_mode,
                "batch": True,
                "warm_oracle": True,
                "runs": outputs,
                "status": "ok",
            },
            indent=None,
        )
    )
    return 0


def handle_request(request: dict, session: dict, default_original_model: Path) -> dict:
    mode = request.get("mode", "full")
    if mode not in {"full", "fast", "batch"}:
        return {"ok": False, "error": f"unsupported_mode:{mode}"}

    compare_tolerance = request.get("compare_tolerance", TOOL_COMPARE_TOLERANCE)
    if not isinstance(compare_tolerance, (int, float)):
        return {"ok": False, "error": "invalid_compare_tolerance"}

    original_model = Path(request.get("original_model", str(default_original_model)))

    # Backward-compatible one-shot request fields.
    trace_json = request.get("trace_json")
    trace_jsons = request.get("trace_jsons")
    if not trace_json and not trace_jsons:
        return {"ok": False, "error": "missing_trace_json"}

    if mode == "batch" and not trace_jsons:
        return {"ok": False, "error": "batch_mode_requires_trace_jsons"}

    outputs = []

    if mode == "batch":
        if not isinstance(trace_jsons, list) or not trace_jsons:
            return {"ok": False, "error": "batch_mode_empty_trace_jsons"}
        compare_mode = request.get("batch_compare_mode", "full")
        if compare_mode not in {"full", "fast"}:
            return {"ok": False, "error": f"invalid_batch_compare_mode:{compare_mode}"}
        for item in trace_jsons:
            trace_path = Path(item)
            if not trace_path.exists():
                outputs.append({"ok": False, "trace_json": item, "error": "trace_not_found"})
                continue
            try:
                requested_output = request.get("output")
                output_path = Path(requested_output) if requested_output else None
                output, report = compare_one(
                    session,
                    trace_path,
                    original_model,
                    compare_mode,
                    output_path,
                    compare_tolerance,
                )
                outputs.append(
                    summarize_response(
                        response=report,
                        trace_path=trace_path,
                        mode=compare_mode,
                        output=output,
                        warm_session=True,
                        requested_mode=mode,
                        batch=True,
                    )
                )
            except Exception as exc:
                outputs.append(
                    {"ok": False, "trace_json": str(trace_path), "error": str(exc), "requested_mode": mode}
                )

        return {
            "ok": True,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "tool_signature": tool_signature(),
            "requested_mode": "batch",
            "compare_mode": compare_mode,
            "batch": True,
            "warm_oracle": True,
            "runs": outputs,
            "status": "ok",
        }

    if not trace_json:
        return {"ok": False, "error": "missing_trace_json"}
    trace_path = Path(trace_json)
    if not trace_path.exists():
        return {"ok": False, "error": "trace_not_found", "trace_json": str(trace_path)}

    try:
        requested_output = request.get("output")
        output_path = Path(requested_output) if requested_output else None
        output, report = compare_one(
            session,
            trace_path,
            original_model,
            mode,
            output_path,
            compare_tolerance,
        )
        return summarize_response(
            response=report,
            trace_path=trace_path,
            mode=mode,
            output=output,
            warm_session=True,
            requested_mode=mode,
            batch=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "trace_json": str(trace_path), "mode": mode}


def run_listen(original_model: Path, device: str, compare_tolerance: float) -> int:
    session = {"device": torch.device(device), "model": None}
    print(
        json.dumps(
            {
                "ok": True,
                "status": "runner_started",
                "tool": Path(__file__).name,
                "tool_schema_version": TOOL_SCHEMA_VERSION,
                "tool_signature": tool_signature(),
                "compare_tolerance": compare_tolerance,
                "warm_oracle": True,
            }
        )
    )
    sys.stdout.flush()

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue

        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"invalid_json:{exc}"}))
            continue

        op = request.get("op", "compare")
        if op == "shutdown":
            print(
                json.dumps(
                    {
                        "ok": True,
                        "op": "shutdown",
                        "tool_schema_version": TOOL_SCHEMA_VERSION,
                        "tool_signature": tool_signature(),
                    }
                )
            )
            return 0
        if op != "compare":
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"unsupported_op:{op}",
                    }
                )
            )
            continue

        response = handle_request(request, session, original_model)
        print(json.dumps(response))
        sys.stdout.flush()

    return 0


def main() -> int:
    args = parse_args()

    if args.batch and args.batch_traces:
        raise SystemExit("--batch and --batch-traces are mutually exclusive")
    if args.batch:
        if args.cuda_trace_json is not None:
            raise SystemExit("--batch cannot be used with --cuda-trace-json")
        args.batch_traces = [Path(trace_path) for trace_path in args.batch]

    if args.mode == "batch" and not args.batch_traces:
        raise SystemExit("--mode batch requires --batch-traces when not using --listen")

    if args.listen:
        return run_listen(args.original_model, args.device, args.compare_tolerance)

    if args.batch_traces:
        return run_batch_mode(
            args.batch_traces,
            args.original_model,
            args.mode,
            args.device,
            args.compare_tolerance,
        )

    if args.cuda_trace_json is None:
        raise SystemExit("--cuda-trace-json is required unless --batch-traces or --listen is used")

    if args.output is None:
        raise SystemExit("--output is required for one-shot full/fast mode")

    with args.cuda_trace_json.open("r", encoding="utf-8") as handle:
        cuda_trace = json.load(handle)

    model = load_restricted_transformer(
        Path(cuda_trace["restricted_model_path"]),
        args.original_model,
        torch.device(args.device),
    )
    report = compare_trace(
        model=model,
        cuda_trace=cuda_trace,
        original_model=args.original_model,
        mode=args.mode if args.mode != "batch" else "full",
        compare_tolerance=args.compare_tolerance,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "ok": True,
                "tool": Path(__file__).name,
                "tool_schema_version": TOOL_SCHEMA_VERSION,
                "tool_signature": tool_signature(),
                "requested_mode": args.mode,
                "compare_mode": args.mode if args.mode != "batch" else "full",
                "batch": False,
                "warm_oracle": False,
                "trace_json": str(args.cuda_trace_json),
                "output": str(args.output),
                "first_divergence_stage": report.get("first_divergence_stage"),
                "stage_diff_count": len(report.get("stage_diffs", [])),
                "status": "ok",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
