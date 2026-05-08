#!/usr/bin/env python3
"""Extract a replay-ready AVX2 GEMM-stub BF16 dot contract.

This helper is source-attribution only. It reads the already-instrumented
external PyTorch checkout and existing sampled trace artifacts, archives the
current external PyTorch diff, and emits a status JSON describing the selected
AVX2 cpublas_gemm_impl / bf16_dot_with_fp32_arith contract. It does not patch,
reset, build, or import PyTorch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("/home/emmy/openai/pytorch")
DEFAULT_BUILD_ENV = Path("/home/emmy/openai/.venvs/pytorch-src-cpu")
DEFAULT_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-avx2-contract-extraction")
DEFAULT_STATUS = Path("/tmp/fused_linear_addmm_gemm_stub_avx2_contract_extraction_status.json")
EXPECTED_COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
DISPATCH_INTERNALS_PATCH = Path(
    "/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-dispatch-internals/"
    "pre_gemm_stub_internals.patch"
)
SAMPLED_TRACE_PATCH = Path(
    "/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-sampled-trace/"
    "pre_sampled_trace.patch"
)
DISPATCH_INTERNALS_STATUS = Path("/tmp/fused_linear_addmm_gemm_stub_dispatch_internals_status.json")
SAMPLED_TRACE_STATUS = Path("/tmp/fused_linear_addmm_gemm_stub_sampled_trace_status.json")
SYNTHESIS_STATUS = Path("/tmp/fused_linear_addmm_rust_cpu_policy_synthesis_status.json")
CLOSURE_STATUS = Path("/tmp/fused_linear_addmm_rust_cpu_policy_closure_audit_status.json")
SAMPLED_TRACE_RESEARCH = Path("/home/emmy/openai/pytorch-research/fused-linear-addmm-gemm-stub-sampled-trace")
SAMPLED_LAYERS = [6, 10, 13, 16, 18, 21]
SOURCE_FILES = [
    "aten/src/ATen/native/cpu/BlasKernel.cpp",
    "aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp",
    "aten/src/ATen/native/CPUBlas.cpp",
    "aten/src/ATen/native/CPUBlas.h",
    "aten/src/ATen/native/DispatchStub.h",
    "aten/src/ATen/native/DispatchStub.cpp",
    "aten/src/ATen/cpu/vec/vec256/vec256_16bit_float.h",
    "aten/src/ATen/cpu/vec/vec256/vec256_float.h",
    "aten/src/ATen/cpu/vec/functional_base.h",
    "aten/src/ATen/cpu/vec/vec_n.h",
    "torch/headeronly/util/BFloat16.h",
]
SNIPPET_PATTERNS = {
    "cpublas_gemm_impl": "void cpublas_gemm_impl",
    "bf16_gemm_transa": "const at::BFloat16 *a",
    "bf16_dot": "float bf16_dot_with_fp32_arith",
    "dot_main_loop": "dot_with_fp32_arith_main_loop_no_bfdot",
    "dot_inner_loop": "dot_with_fp32_arith_main_inner_loop_no_bfdot",
    "dot_tail": "DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY",
    "vectorized_bfloat16_size": "class Vectorized<BFloat16>",
    "vectorized_float_size": "class Vectorized<float>",
    "avx2_float_reduce": "struct VecReduceAllSIMD<float, Op>",
    "vectorizedn_reduce": "inline T vec_reduce_all",
    "bf16_rounding": "round_to_nearest_even",
    "dispatch_fallback": "try_choose_cpu_impl",
    "dispatch_register": "REGISTER_DISPATCH(cpublas::gemm_stub",
}
FALSE_GUARDS = {
    "concrete_global_replay_policy_found": False,
    "reopen_rust_policy_synthesis": False,
    "backend_selected": False,
    "implementation_authorized": False,
    "consumer_revalidation_authorized": False,
    "runtime_behavior_changed": False,
    "production_routing_changed": False,
    "cuda_kernels_changed": False,
    "output_emitted": False,
    "ladder_continued": False,
    "final_logit_claim": False,
    "all_layer_claim": False,
    "server_claim": False,
    "context_length_claim": False,
    "tolerance_pass": False,
    "correction_metadata_applied": False,
    "rebaseline_performed": False,
    "old_artifacts_replaced": False,
    "full_model_loaded": False,
    "model_forward_run": False,
    "gpu_tensors_created": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AVX2 GEMM-stub contract.")
    parser.add_argument("--source-checkout-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--build-env-path", type=Path, default=DEFAULT_BUILD_ENV)
    parser.add_argument("--research-path", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--output", "--status-output", dest="status_output", type=Path, default=DEFAULT_STATUS)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 300) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "succeeded": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_tail": completed.stdout.splitlines()[-80:],
            "stderr_tail": completed.stderr.splitlines()[-80:],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "succeeded": False,
            "stdout": "",
            "stderr": repr(exc),
            "stdout_tail": [],
            "stderr_tail": [repr(exc)],
            "error": repr(exc),
        }


def source_state(source: Path) -> dict[str, Any]:
    head = run_cmd(["git", "rev-parse", "HEAD"], cwd=source)
    status = run_cmd(["git", "status", "--short"], cwd=source)
    diff_files = run_cmd(["git", "diff", "--name-only"], cwd=source)
    head_value = (head.get("stdout") or "").strip().splitlines()[-1] if head.get("stdout") else ""
    return {
        "head": head_value,
        "expected_commit_match": head_value == EXPECTED_COMMIT,
        "status_short": status.get("stdout_tail", []),
        "dirty_files": diff_files.get("stdout_tail", []),
    }


def archive_patch(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(["git", "diff"], cwd=str(source), text=True, capture_output=True, check=False)
    output.write_text(completed.stdout, encoding="utf-8", errors="replace")
    return {
        "path": str(output),
        "succeeded": completed.returncode == 0,
        "size_bytes": output.stat().st_size if output.is_file() else 0,
    }


def line_number_for(text: str, pattern: str) -> int | None:
    idx = text.find(pattern)
    if idx < 0:
        return None
    return text.count("\n", 0, idx) + 1


def snippet_for(path: Path, pattern: str, *, before: int = 12, after: int = 36) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if pattern in line:
            start = max(0, index - before)
            end = min(len(lines), index + after + 1)
            body = "\n".join(f"{line_no + 1}: {lines[line_no]}" for line_no in range(start, end))
            return {
                "source_file": str(path),
                "pattern": pattern,
                "found": True,
                "start_line": start + 1,
                "end_line": end,
                "body": body,
            }
    return {"source_file": str(path), "pattern": pattern, "found": False}


def collect_source_snippets(source: Path, research: Path) -> tuple[dict[str, Any], list[str]]:
    snippets_dir = research / "source-snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    files_text = {rel: (source / rel).read_text(encoding="utf-8", errors="replace") for rel in SOURCE_FILES if (source / rel).is_file()}
    pattern_to_file = {
        "cpublas_gemm_impl": "aten/src/ATen/native/cpu/BlasKernel.cpp",
        "bf16_gemm_transa": "aten/src/ATen/native/cpu/BlasKernel.cpp",
        "bf16_dot": "aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp",
        "dot_main_loop": "aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp",
        "dot_inner_loop": "aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp",
        "dot_tail": "aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp",
        "vectorized_bfloat16_size": "aten/src/ATen/cpu/vec/vec256/vec256_bfloat16.h",
        "vectorized_float_size": "aten/src/ATen/cpu/vec/vec256/vec256_float.h",
        "avx2_float_reduce": "aten/src/ATen/cpu/vec/functional_base.h",
        "vectorizedn_reduce": "aten/src/ATen/cpu/vec/vec_n.h",
        "bf16_rounding": "torch/headeronly/util/BFloat16.h",
        "dispatch_fallback": "aten/src/ATen/native/DispatchStub.cpp",
        "dispatch_register": "aten/src/ATen/native/cpu/BlasKernel.cpp",
    }
    snippets: dict[str, Any] = {}
    for name, pattern in SNIPPET_PATTERNS.items():
        rel = pattern_to_file[name]
        item = snippet_for(source / rel, pattern)
        snippets[name] = {key: value for key, value in item.items() if key != "body"}
        if item.get("found"):
            (snippets_dir / f"{name}.txt").write_text(item["body"] + "\n", encoding="utf-8")
            snippets[name]["snippet_path"] = str(snippets_dir / f"{name}.txt")
    map_lines = [
        "GEMM-stub AVX2 BF16 source contract map",
        "",
        "Files inspected:",
    ]
    for rel in SOURCE_FILES:
        path = source / rel
        map_lines.append(f"- {rel}: {'present' if path.is_file() else 'missing'}")
    map_lines.extend(["", "Key source landmarks:"])
    for name, item in snippets.items():
        map_lines.append(
            f"- {name}: {Path(item.get('source_file', '')).relative_to(source) if item.get('source_file') else '<missing>'}"
            f":{item.get('start_line')} found={item.get('found')}"
        )
    (research / "source-contract-map.txt").write_text("\n".join(map_lines) + "\n", encoding="utf-8")
    return snippets, sorted(files_text.keys())


def load_trace_inputs() -> dict[str, Any]:
    sampled_status = read_json(SAMPLED_TRACE_STATUS, {})
    internals_status = read_json(DISPATCH_INTERNALS_STATUS, {})
    residual_traces = read_json(SAMPLED_TRACE_RESEARCH / "residual-lane-traces.json", {})
    layer18 = read_json(SAMPLED_TRACE_RESEARCH / "layer18-lane1641-confirmation.json", {})
    target_summary = read_json(SAMPLED_TRACE_RESEARCH / "per-layer-target-summary.json", {})
    return {
        "sampled_status": sampled_status,
        "internals_status": internals_status,
        "residual_traces": residual_traces,
        "layer18_lane1641": layer18,
        "target_summary": target_summary,
        "synthesis_status_available": SYNTHESIS_STATUS.is_file(),
        "closure_status_available": CLOSURE_STATUS.is_file(),
    }


def collect_lanes(residual_traces: dict[str, Any]) -> tuple[list[int], dict[str, list[int]], list[dict[str, Any]]]:
    lanes_by_layer: dict[str, list[int]] = {}
    samples: list[dict[str, Any]] = []
    for layer_key, items in residual_traces.items():
        lanes: list[int] = []
        if isinstance(items, list):
            for item in items:
                lane = item.get("hidden_lane")
                if isinstance(lane, int):
                    lanes.append(lane)
                    if item.get("explanation") != "target_selection_and_bf16_rounding_boundary":
                        samples.append(
                            {
                                "layer": int(layer_key),
                                "hidden_lane": lane,
                                "baseline": item.get("baseline"),
                                "default": item.get("default"),
                                "avx2_default_dot_diff": item.get("avx2_default_dot_diff"),
                                "explanation_status": item.get("explanation"),
                            }
                        )
        lanes_by_layer[layer_key] = sorted(set(lanes))
    all_lanes = sorted({lane for lanes in lanes_by_layer.values() for lane in lanes})
    return all_lanes, lanes_by_layer, samples


def build_contract(trace_inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    avx2_contract = {
        "target_selection_rule": (
            "For this sampled host, baseline/no ATEN_CPU_CAPABILITY reports runtime AVX512, "
            "but DispatchStub finds the AVX512 gemm_stub entry null and selects the AVX2 "
            "cpublas_gemm_impl entry. Explicit avx2 and explicit avx512/avx512_bf16/"
            "avx512_vnni also select or fall back to AVX2."
        ),
        "source_function": (
            "cpublas_gemm_impl -> gemm_core_ -> BF16-specialized gemm_transa_ -> "
            "compute_dot -> CPU_CAPABILITY::bf16_dot_with_fp32_arith -> "
            "dot_with_fp32_arith_no_bfdot"
        ),
        "matrix_shape": {"M": 2880, "N": 1, "K": 4096},
        "dtype_contract": {
            "input": "BF16 weighted-V",
            "weight": "BF16 o_proj weight row through weight.T view",
            "bias": "BF16 o_proj bias",
            "opmath": "f32",
            "output": "BF16",
        },
        "beta_bias_contract": {
            "beta": 1.0,
            "bias_role": "bias/self is the prior accumulator c[j * ldc + i]",
            "prior_conversion": "BF16 prior converts to f32 in beta * c + alpha * dot expression",
        },
        "alpha_contract": 1.0,
        "vector_width": {
            "bf16_elements_per_vector": 16,
            "f32_elements_per_vector": 8,
            "source": "Vectorized<BFloat16>::size() = 16, Vectorized<float>::size() = 8 for AVX2 vec256",
        },
        "chunk_size": 64,
        "k_loop_order": (
            "j increases from 0 to len_aligned in steps of 64. For each j, registerPairIndex "
            "runs 0, 1, 2, 3; each pair loads 16 contiguous BF16 elements and updates two "
            "f32 accumulator vectors."
        ),
        "accumulator_count": {
            "vector_accumulators": 8,
            "f32_lanes_per_accumulator": 8,
            "logical_f32_accumulator_lanes": 64,
        },
        "accumulator_update_order": (
            "Within each 64-element K chunk, sum[2*p] and sum[2*p+1] are updated for p=0..3 "
            "using f32 fused multiply-add on the low and high halves of each 16-BF16 vector."
        ),
        "horizontal_reduction_order": (
            "First reduce VectorizedN<float, 8> as sum[0]+=sum[4], sum[1]+=sum[5], "
            "sum[2]+=sum[6], sum[3]+=sum[7], then sum[0]+=sum[2], sum[1]+=sum[3], "
            "then sum[0]+=sum[1]. Finally reduce the AVX2 8-lane f32 vector with "
            "128-bit, 64-bit, then 32-bit shuffles and return lane 0."
        ),
        "tail_handling": {
            "K": 4096,
            "main_loop_elements": 4096,
            "vector_tail_elements": 0,
            "scalar_tail_elements": 0,
            "note": "K=4096 is exactly divisible by the 64-element AVX2 main-loop chunk.",
        },
        "bf16_load_convert_contract": (
            "Each load is unaligned BF16 loadu of 16 elements; conversion to f32 zero-extends "
            "the BF16 payload and shifts it into the high 16 bits of an f32 bit pattern."
        ),
        "f32_product_contract": "_mm256_fmadd_ps-style f32 fused multiply-add updates each accumulator.",
        "bias_addition_point": (
            "After the f32 dot is reduced for an output lane, gemm_transa_ computes "
            "combined = beta * prior + alpha * dot in f32 before assigning to BF16."
        ),
        "final_bf16_rounding_contract": (
            "BFloat16(float) uses round-to-nearest-even by adding 0x7fff plus the truncated "
            "BF16 lsb before shifting, with NaN mapped to 0x7fc0."
        ),
        "lane_dependence": (
            "The same loop and reduction rule applies to each output lane. Each lane is an "
            "independent dot over a different weight row, so parallel scheduling should not "
            "change a lane's arithmetic order."
        ),
        "scalar_equivalent_replay_possible": False,
        "avx2_structured_replay_required": True,
    }
    default_contract = {
        "target_selection_rule": "ATEN_CPU_CAPABILITY=default selects the DEFAULT-compiled cpublas_gemm_impl.",
        "source_body": "same templated source structure compiled under DEFAULT CPU capability",
        "diagnostic_role": "contrast only; default output is not the official oracle on this host",
        "known_layer18_lane1641": (trace_inputs.get("layer18_lane1641") or {}).get("default"),
        "replay_priority": "secondary diagnostic",
    }
    layer18 = trace_inputs.get("layer18_lane1641") or {}
    confirmation = {
        "baseline": layer18.get("baseline"),
        "default": layer18.get("default"),
        "interpretation": (
            "AVX2 and DEFAULT dot values differ enough to move the fused dot+bias value across "
            "the BF16 round-to-nearest-even boundary for layer18 lane1641."
        ),
    }
    return avx2_contract, default_contract, confirmation


def classify(avx2_contract: dict[str, Any], blockers: list[str]) -> tuple[str, bool, bool]:
    required_keys = [
        "target_selection_rule",
        "source_function",
        "matrix_shape",
        "dtype_contract",
        "beta_bias_contract",
        "alpha_contract",
        "vector_width",
        "chunk_size",
        "k_loop_order",
        "accumulator_count",
        "accumulator_update_order",
        "horizontal_reduction_order",
        "tail_handling",
        "bf16_load_convert_contract",
        "f32_product_contract",
        "bias_addition_point",
        "final_bf16_rounding_contract",
        "lane_dependence",
        "scalar_equivalent_replay_possible",
        "avx2_structured_replay_required",
    ]
    complete = all(key in avx2_contract and avx2_contract[key] not in (None, "", "unknown") for key in required_keys)
    if complete and not blockers:
        return "fused_linear_addmm_gemm_stub_avx2_contract_replay_ready", True, True
    if complete:
        return "fused_linear_addmm_gemm_stub_avx2_contract_partial", False, True
    return "fused_linear_addmm_gemm_stub_avx2_contract_blocked_by_source_complexity", False, False


def main() -> int:
    args = parse_args()
    args.research_path.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "classification": "fused_linear_addmm_gemm_stub_avx2_contract_failed",
        "validation_only": True,
        "source_attribution_probe": True,
        "avx2_contract_extraction": True,
        "oracle_device": "cpu",
        "cuda_used": False,
        "source_checkout_path": str(args.source_checkout_path),
        "checked_out_commit": EXPECTED_COMMIT,
        "build_env_path": str(args.build_env_path),
        "pre_existing_patches_verified": False,
        "pre_avx2_contract_extraction_patch_path": str(args.research_path / "pre_avx2_contract_extraction.patch"),
        "pytorch_source_patched_this_branch": False,
        "pytorch_rebuilt_this_branch": False,
        "instrumented_source_files": [],
        "source_contract_files_inspected": [],
        "layers_evaluated": [],
        "lanes_evaluated": [],
        "configs_evaluated": [],
        "avx2_contract": {},
        "default_contract": {},
        "layer18_lane1641_contract_confirmation": {},
        "unexplained_residual_contract_samples": [],
        "replay_contract_complete": False,
        "replay_contract_blockers": [],
        "supports_validation_prototype": False,
        **FALSE_GUARDS,
    }
    try:
        state = source_state(args.source_checkout_path)
        status["source_state"] = state
        status["pre_existing_patches_verified"] = DISPATCH_INTERNALS_PATCH.is_file() and SAMPLED_TRACE_PATCH.is_file()
        status["pre_existing_patch_paths"] = [str(DISPATCH_INTERNALS_PATCH), str(SAMPLED_TRACE_PATCH)]
        status["pre_avx2_contract_extraction_patch"] = archive_patch(
            args.source_checkout_path,
            args.research_path / "pre_avx2_contract_extraction.patch",
        )
        status["instrumented_source_files"] = state.get("dirty_files", [])
        if not state["expected_commit_match"] or not status["pre_existing_patches_verified"]:
            status["classification"] = "fused_linear_addmm_gemm_stub_avx2_contract_failed"
            status["failure_summary"] = "PyTorch checkout commit or pre-existing patch archive verification failed."
            write_json(args.status_output, status)
            return 0

        snippets, inspected_files = collect_source_snippets(args.source_checkout_path, args.research_path)
        trace_inputs = load_trace_inputs()
        avx2_contract, default_contract, layer18_confirmation = build_contract(trace_inputs)
        all_lanes, lanes_by_layer, unexplained_samples = collect_lanes(trace_inputs["residual_traces"])
        blockers: list[str] = []
        if trace_inputs.get("sampled_status", {}).get("sampled_trace_supports_source_replay_design") is not True:
            blockers.append("sampled_trace_did_not_mark_source_replay_design_supported")
        if not all_lanes:
            blockers.append("no_residual_lane_trace_artifacts_loaded")
        classification, complete, supports_prototype = classify(avx2_contract, blockers)

        status.update(
            {
                "classification": classification,
                "source_contract_files_inspected": inspected_files,
                "source_contract_snippets": snippets,
                "layers_evaluated": trace_inputs.get("sampled_status", {}).get("sampled_layers_evaluated", SAMPLED_LAYERS),
                "lanes_evaluated": all_lanes,
                "lanes_evaluated_by_layer": lanes_by_layer,
                "configs_evaluated": trace_inputs.get("sampled_status", {}).get("configs_evaluated", []),
                "avx2_contract": avx2_contract,
                "default_contract": default_contract,
                "layer18_lane1641_contract_confirmation": layer18_confirmation,
                "unexplained_residual_contract_samples": unexplained_samples,
                "replay_contract_complete": complete,
                "replay_contract_blockers": blockers,
                "supports_validation_prototype": supports_prototype,
                "concrete_global_replay_policy_found": False,
                "reopen_rust_policy_synthesis": False,
                "trace_artifacts_used": {
                    "sampled_status": str(SAMPLED_TRACE_STATUS),
                    "dispatch_internals_status": str(DISPATCH_INTERNALS_STATUS),
                    "residual_lane_traces": str(SAMPLED_TRACE_RESEARCH / "residual-lane-traces.json"),
                    "layer18_lane1641_confirmation": str(SAMPLED_TRACE_RESEARCH / "layer18-lane1641-confirmation.json"),
                },
                "source_inspection_summary": {
                    "avx2_register_pairs_per_iteration": 4,
                    "avx2_f32_registers_per_iteration": 8,
                    "avx2_f32_elements_per_register": 8,
                    "avx2_elements_per_iteration": 64,
                    "k_4096_main_loop_iterations": 64,
                    "k_4096_tail_elements": 0,
                    "source_map_path": str(args.research_path / "source-contract-map.txt"),
                },
            }
        )

        write_json(args.research_path / "lane-contract-traces.json", trace_inputs["residual_traces"])
        write_json(args.research_path / "layer18-lane1641-contract.json", layer18_confirmation)
        write_json(args.research_path / "unexplained-residual-contract-samples.json", unexplained_samples)
        write_json(
            args.research_path / "replay-contract-summary.json",
            {
                "classification": classification,
                "replay_contract_complete": complete,
                "supports_validation_prototype": supports_prototype,
                "avx2_contract": avx2_contract,
                "default_contract": default_contract,
                "concrete_global_replay_policy_found": False,
                "reopen_rust_policy_synthesis": False,
            },
        )
        interpretation_lines = [
            f"classification: {classification}",
            f"replay_contract_complete: {complete}",
            f"supports_validation_prototype: {supports_prototype}",
            "AVX2 contract: 64-BF16 K chunks, eight f32 vector accumulators, AVX2 FMA updates,",
            "pairwise VectorizedN reduction, AVX2 horizontal shuffle reduction, f32 bias fusion, final BF16 RNE cast.",
            "No global replay policy is claimed by this extraction branch.",
        ]
        (args.research_path / "interpretation-summary.txt").write_text(
            "\n".join(interpretation_lines) + "\n",
            encoding="utf-8",
        )
        write_json(args.status_output, status)
        return 0
    except Exception as exc:  # noqa: BLE001
        status["classification"] = "fused_linear_addmm_gemm_stub_avx2_contract_failed"
        status["error"] = repr(exc)
        write_json(args.status_output, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
