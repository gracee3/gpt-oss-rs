#!/usr/bin/env python3
"""Validate runtime-forward final-readout direct-module status artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_CASE = "developer-message-user-smoke"
EXPECTED_CLASSIFICATION = "final_readout_direct_module_logits_cleared"
EXPECTED_FINAL_BLOCK_DIGEST = "3a61719e695d130f95acf108428e6307142acf66425755f54c8c1bc95f2fb257"
EXPECTED_FINAL_NORM_DIGEST = "e57a8796fd3d4d36d09713ad9780a8e0c077824458568157976b8e71b17a2139"
EXPECTED_LOGITS_DIGEST = "67f31845dd24db26cc91954607cfae8ae7ff7b9c8954cb9d3b1610ca9c635209"
EXPECTED_STALE_PPP_DIGEST = "5a7d47edfab63d59c17825b8d7b7668cc7a15ad2d107f902ca2caa05488ecd44"
EXPECTED_VOCAB_SIZE = 201088


class Validation:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def fail(self, path: str, message: str) -> None:
        self.failures.append(f"{path}: {message}")

    def expect_equal(self, path: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            self.fail(path, f"expected {expected!r}, found {actual!r}")

    def expect_true(self, path: str, actual: Any) -> None:
        if actual is not True:
            self.fail(path, f"expected true, found {actual!r}")

    def expect_zero(self, path: str, actual: Any) -> None:
        if actual != 0 and actual != 0.0:
            self.fail(path, f"expected 0.0, found {actual!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate runtime-forward final-readout direct-module status JSON."
    )
    parser.add_argument("--artifact", type=Path, required=True, help="Status JSON artifact to validate.")
    parser.add_argument(
        "--check-manifest",
        type=Path,
        help="Optional LARGE_ARTIFACTS_MANIFEST.json to sanity-check for final readout references.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_path(data: Any, keys: list[str]) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def all_key_values(data: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(data, dict):
        for current_key, value in data.items():
            if current_key == key:
                found.append(value)
            found.extend(all_key_values(value, key))
    elif isinstance(data, list):
        for value in data:
            found.extend(all_key_values(value, key))
    return found


def first_present(data: dict[str, Any], path_options: list[list[str]]) -> tuple[str, Any]:
    for keys in path_options:
        value = get_path(data, keys)
        if value is not None:
            return ".".join(keys), value
    return "", None


def validate_optional_case(data: dict[str, Any], check: Validation) -> None:
    case_values = []
    for key in ("case_id", "case", "exact_case"):
        case_values.extend(all_key_values(data, key))
    for index, value in enumerate(case_values):
        if isinstance(value, dict):
            nested = value.get("case_id") or value.get("case")
            if nested is not None and nested != EXPECTED_CASE:
                check.fail(f"case[{index}]", f"expected {EXPECTED_CASE!r}, found {nested!r}")
        elif isinstance(value, str) and value != EXPECTED_CASE:
            check.fail(f"case[{index}]", f"expected {EXPECTED_CASE!r}, found {value!r}")


def validate_metric_block(
    data: dict[str, Any],
    check: Validation,
    name: str,
    path_options: list[list[str]],
    expected_digest: str,
) -> str | None:
    path, block = first_present(data, path_options)
    if not isinstance(block, dict):
        check.fail(name, "metric block not found")
        return None

    matched_path, matched = first_present(block, [["matched"], ["digest_matched"]])
    if matched_path:
        check.expect_true(f"{path}.{matched_path}", matched)
    else:
        check.fail(path, "matched flag not found")

    max_path, max_value = first_present(block, [["max_abs_diff"], ["max"], ["maximum"]])
    if max_path:
        check.expect_zero(f"{path}.{max_path}", max_value)
    else:
        check.fail(path, "max diff not found")

    mean_path, mean_value = first_present(block, [["mean_abs_diff"], ["mean"], ["average"]])
    if mean_path:
        check.expect_zero(f"{path}.{mean_path}", mean_value)
    else:
        check.fail(path, "mean diff not found")

    digest_path, digest = first_present(
        block,
        [
            ["local_sha256_f32_le"],
            ["official_sha256_f32_le"],
            ["official_direct_module_sha256_f32_le"],
            ["digest"],
            ["sha256"],
        ],
    )
    if digest_path:
        check.expect_equal(f"{path}.{digest_path}", digest, expected_digest)
        return str(digest)

    check.fail(path, "digest not found")
    return None


def validate_top20(data: dict[str, Any], check: Validation) -> str:
    path, top20 = first_present(data, [["top20_comparison_summary"], ["top_20_comparison_summary"]])
    if top20 is None:
        return "not present"
    if not isinstance(top20, dict):
        check.fail(path, "expected object")
        return "invalid"

    for keys in (["ordered_top20_token_ids_match"], ["ordered_token_ids_match"], ["top20_set_match"], ["set_match"]):
        value = get_path(top20, keys)
        if value is not None:
            check.expect_true(f"{path}.{'.'.join(keys)}", value)

    metric_path, metric = first_present(
        top20,
        [
            ["official_top20_token_value_metric"],
            ["values_metric"],
            ["value_metric"],
        ],
    )
    if isinstance(metric, dict):
        matched = get_path(metric, ["matched"])
        if matched is not None:
            check.expect_true(f"{path}.{metric_path}.matched", matched)
        max_value = get_path(metric, ["max_abs_diff"])
        if max_value is not None:
            check.expect_zero(f"{path}.{metric_path}.max_abs_diff", max_value)
        mean_value = get_path(metric, ["mean_abs_diff"])
        if mean_value is not None:
            check.expect_zero(f"{path}.{metric_path}.mean_abs_diff", mean_value)

    return "present"


def validate_stale_note(data: dict[str, Any], check: Validation) -> None:
    note = get_path(data, ["stale_artifact_note"])
    if note is None:
        return
    if not isinstance(note, dict):
        check.fail("stale_artifact_note", "expected object")
        return
    prior = note.get("prior_ppp_logits_digest")
    if prior is not None:
        check.expect_equal("stale_artifact_note.prior_ppp_logits_digest", prior, EXPECTED_STALE_PPP_DIGEST)
    regenerated = note.get("regenerated_direct_module_logits_digest")
    if regenerated is not None:
        check.expect_equal(
            "stale_artifact_note.regenerated_direct_module_logits_digest",
            regenerated,
            EXPECTED_LOGITS_DIGEST,
        )


def validate_remaining_seam(data: dict[str, Any], check: Validation) -> None:
    value = data.get("earliest_remaining_mismatching_seam")
    if value is None:
        return
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "n/a"}:
        return
    check.fail("earliest_remaining_mismatching_seam", f"expected none/null equivalent, found {value!r}")


def validate_manifest(path: Path, check: Validation) -> None:
    if not path.exists():
        check.fail(str(path), "manifest does not exist")
        return
    try:
        manifest = load_json(path)
    except Exception as exc:  # noqa: BLE001
        check.fail(str(path), f"could not parse JSON: {exc}")
        return

    text = json.dumps(manifest, sort_keys=True)
    expected_name = "developer-message.ppp-final-token-readout-norm-and-lm-head-direct-module-bundle-status.json"
    if EXPECTED_LOGITS_DIGEST not in text and expected_name not in text:
        check.fail(
            str(path),
            "manifest does not reference the expected direct-module logits digest or artifact name",
        )


def validate_artifact(path: Path, manifest_path: Path | None) -> tuple[Validation, dict[str, Any]]:
    check = Validation()
    if not path.exists():
        check.fail(str(path), "artifact does not exist")
        return check, {}
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        check.fail(str(path), f"could not parse JSON: {exc}")
        return check, {}
    if not isinstance(data, dict):
        check.fail(str(path), "expected top-level JSON object")
        return check, {}

    validate_optional_case(data, check)
    check.expect_equal("classification", data.get("classification"), EXPECTED_CLASSIFICATION)

    final_block_digest = validate_metric_block(
        data,
        check,
        "final_block_output",
        [["final_block_output_guard_metrics"], ["final_block_output"], ["milestone_result", "final_block_output"]],
        EXPECTED_FINAL_BLOCK_DIGEST,
    )
    final_norm_digest = validate_metric_block(
        data,
        check,
        "final_norm",
        [["final_norm_guard_metrics"], ["final_norm"], ["milestone_result", "final_norm"]],
        EXPECTED_FINAL_NORM_DIGEST,
    )
    logits_digest = validate_metric_block(
        data,
        check,
        "lm_head_logits",
        [["lm_head_logits_metrics"], ["lm_head_logits"], ["milestone_result", "lm_head_logits"]],
        EXPECTED_LOGITS_DIGEST,
    )

    logits = first_present(data, [["lm_head_logits_metrics"], ["lm_head_logits"], ["milestone_result", "lm_head_logits"]])[1]
    vocab_size = logits.get("vocab_size") if isinstance(logits, dict) else None
    check.expect_equal("lm_head_logits.vocab_size", vocab_size, EXPECTED_VOCAB_SIZE)

    top20_status = validate_top20(data, check)
    validate_stale_note(data, check)
    validate_remaining_seam(data, check)
    if manifest_path is not None:
        validate_manifest(manifest_path, check)

    summary = {
        "classification": data.get("classification"),
        "final_block_digest": final_block_digest,
        "final_norm_digest": final_norm_digest,
        "logits_digest": logits_digest,
        "vocab_size": vocab_size,
        "top20_status": top20_status,
    }
    return check, summary


def main() -> int:
    args = parse_args()
    check, summary = validate_artifact(args.artifact, args.check_manifest)

    if check.failures:
        print("runtime-forward final-readout validation failed:", file=sys.stderr)
        for failure in check.failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("runtime-forward final-readout validation passed")
        print(f"artifact: {args.artifact}")
        print(f"classification: {summary['classification']}")
        print(f"final block digest: {summary['final_block_digest']}")
        print(f"final norm digest: {summary['final_norm_digest']}")
        print(f"LM-head logits digest: {summary['logits_digest']}")
        print(f"vocab size: {summary['vocab_size']}")
        print(f"top-20 comparison: {summary['top20_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
