#!/usr/bin/env python3
"""Aggregate the normal grader's independent neural-track results by frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run_root = root / "outputs" / "runs" / "latest"
    manifest = json.loads((root / "inputs" / "speech_library" / "timing_split_v1.json").read_text())
    split_for = {case["case_id"]: case["split"] for case in manifest["cases"]}

    accumulators: dict[str, dict[str, float]] = {}
    for case_id, split in split_for.items():
        grade_path = run_root / case_id / "neural_track_grade.json"
        summary_path = run_root / case_id / "neural_track_summary.json"
        if not grade_path.exists() or not summary_path.exists():
            raise RuntimeError(f"missing neural-track result for {case_id}")
        grade = json.loads(grade_path.read_text())
        summary = json.loads(summary_path.read_text())
        for key in ("aggregate", split):
            row = accumulators.setdefault(key, {
                "case_count": 0, "reference_count": 0, "matched_count": 0,
                "missing_count": 0, "extra_count": 0, "order_violations": 0,
                "weighted_center_error_ms": 0.0, "prediction_count": 0,
                "deterministic_fallback_count": 0,
            })
            row["case_count"] += 1
            for metric in ("reference_count", "matched_count", "missing_count",
                           "extra_count", "order_violations"):
                row[metric] += grade[metric]
            row["weighted_center_error_ms"] += (
                grade["mean_abs_center_error_ms"] * grade["matched_count"])
            row["prediction_count"] += summary["prediction_count"]
            row["deterministic_fallback_count"] += summary["deterministic_fallback_count"]

    report: dict[str, object] = {"schema_version": 1, "splits": {}}
    for key, raw in accumulators.items():
        matched = int(raw["matched_count"])
        reference = int(raw["reference_count"])
        predictions = int(raw["prediction_count"])
        fallback = int(raw["deterministic_fallback_count"])
        report["splits"][key] = {
            "case_count": int(raw["case_count"]),
            "reference_count": reference,
            "matched_count": matched,
            "missing_count": int(raw["missing_count"]),
            "extra_count": int(raw["extra_count"]),
            "order_violations": int(raw["order_violations"]),
            "match_rate": matched / reference if reference else 0.0,
            "mean_abs_center_error_ms": raw["weighted_center_error_ms"] / matched if matched else 0.0,
            "prediction_count": predictions,
            "deterministic_fallback_count": fallback,
            "deterministic_fallback_rate": fallback / predictions if predictions else 0.0,
        }

    output = args.output or run_root / "neural_track_grade_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    aggregate = report["splits"]["aggregate"]
    print(
        "Neural track: "
        f"match={aggregate['match_rate']:.6f}, "
        f"center_mae={aggregate['mean_abs_center_error_ms']:.3f} ms, "
        f"fallback={aggregate['deterministic_fallback_rate']:.6f}, "
        f"order={aggregate['order_violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
