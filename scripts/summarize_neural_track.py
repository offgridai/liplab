#!/usr/bin/env python3
"""Aggregate the normal grader's independent neural-track results by frozen split."""

from __future__ import annotations

import argparse
import json
import statistics
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
    alignment = {
        "speech_region_reference_count": 0, "speech_region_matched_count": 0,
        "speech_region_weighted_start_error_ms": 0.0,
        "speech_region_weighted_end_error_ms": 0.0,
        "speech_region_count_mismatch_cases": 0,
        "word_onset_reference_count": 0, "word_onset_matched_count": 0,
        "word_onset_weighted_start_error_ms": 0.0,
        "intra_word_reference_count": 0, "intra_word_matched_count": 0,
        "intra_word_weighted_center_error_ms": 0.0,
        "intra_word_weighted_start_error_ms": 0.0,
        "containment_event_count": 0, "containment_outside_count": 0,
        "containment_before_count": 0, "containment_after_count": 0,
        "syllable_target_count": 0, "syllable_observation_count": 0,
        "syllable_matched_count": 0, "syllable_weighted_error_ms": 0.0,
    }
    error_samples: dict[str, dict[str, dict[str, object]]] = {}
    for case_id, split in split_for.items():
        grade_path = run_root / case_id / "neural_track_grade.json"
        summary_path = run_root / case_id / "neural_track_summary.json"
        syllable_path = run_root / case_id / "neural_syllable_alignment_grade.json"
        samples_path = run_root / case_id / "neural_error_samples.json"
        if (not grade_path.exists() or not summary_path.exists()
                or not syllable_path.exists() or not samples_path.exists()):
            raise RuntimeError(f"missing neural-track result for {case_id}")
        grade = json.loads(grade_path.read_text())
        summary = json.loads(summary_path.read_text())
        syllable = json.loads(syllable_path.read_text())
        samples = json.loads(samples_path.read_text())
        penalty_for = {
            "viseme_center": samples["miss_penalties_ms"]["viseme"],
            "speech_region_start": samples["miss_penalties_ms"]["region"],
            "speech_region_end": samples["miss_penalties_ms"]["region"],
            "word_onset_start": samples["miss_penalties_ms"]["word_onset"],
            "intra_word_center": samples["miss_penalties_ms"]["intra_word"],
            "syllable_center": samples["miss_penalties_ms"]["syllable"],
        }
        for name, penalty in penalty_for.items():
            source = samples[name]
            matched_values = source["matched_errors_ms"]
            missing = source["reference_count"] - source["matched_count"]
            for group in ("aggregate", split):
                target = error_samples.setdefault(group, {}).setdefault(name, {
                    "reference_count": 0, "matched_count": 0,
                    "penalty_ms": penalty, "matched": [], "comprehensive": [],
                })
                target["reference_count"] += source["reference_count"]
                target["matched_count"] += source["matched_count"]
                target["matched"].extend(matched_values)
                target["comprehensive"].extend(matched_values)
                target["comprehensive"].extend([penalty] * missing)
        pause = grade["pause_alignment"]
        region_matches = pause["speech_region_matched_count"]
        alignment["speech_region_reference_count"] += pause["speech_region_reference_count"]
        alignment["speech_region_matched_count"] += region_matches
        alignment["speech_region_weighted_start_error_ms"] += (
            pause["mean_abs_speech_region_start_error_ms"] * region_matches)
        alignment["speech_region_weighted_end_error_ms"] += (
            pause["mean_abs_speech_region_end_error_ms"] * region_matches)
        alignment["speech_region_count_mismatch_cases"] += int(pause["speech_region_count_mismatch"])
        onset = grade["word_onset_alignment"]
        alignment["word_onset_reference_count"] += onset["reference_count"]
        alignment["word_onset_matched_count"] += onset["matched_count"]
        alignment["word_onset_weighted_start_error_ms"] += (
            onset["mean_abs_start_error_ms"] * onset["matched_count"])
        intra = grade["intra_word_alignment"]
        alignment["intra_word_reference_count"] += intra["reference_count"]
        alignment["intra_word_matched_count"] += intra["matched_count"]
        alignment["intra_word_weighted_center_error_ms"] += (
            intra["mean_abs_center_error_ms"] * intra["matched_count"])
        alignment["intra_word_weighted_start_error_ms"] += (
            intra["mean_abs_start_error_ms"] * intra["matched_count"])
        containment = grade["speech_region_containment"]
        alignment["containment_event_count"] += containment["event_count"]
        alignment["containment_outside_count"] += containment["events_outside_region"]
        alignment["containment_before_count"] += containment["events_starting_before_region"]
        alignment["containment_after_count"] += containment["events_ending_after_region"]
        alignment["syllable_target_count"] += syllable["target_count"]
        alignment["syllable_observation_count"] += syllable["observation_count"]
        alignment["syllable_matched_count"] += syllable["matched_count"]
        alignment["syllable_weighted_error_ms"] += (
            syllable["mean_abs_error_ms"] * syllable["matched_count"])
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

    report: dict[str, object] = {"schema_version": 3, "splits": {}}
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

    region_matches = alignment["speech_region_matched_count"]
    onset_matches = alignment["word_onset_matched_count"]
    intra_matches = alignment["intra_word_matched_count"]
    containment_events = alignment["containment_event_count"]
    syllable_matches = alignment["syllable_matched_count"]
    report["alignment"] = {
        "speech_region_match_rate": region_matches / alignment["speech_region_reference_count"],
        "speech_region_start_mae_ms": alignment["speech_region_weighted_start_error_ms"] / region_matches,
        "speech_region_end_mae_ms": alignment["speech_region_weighted_end_error_ms"] / region_matches,
        "speech_region_count_mismatch_cases": alignment["speech_region_count_mismatch_cases"],
        "word_onset_match_rate": onset_matches / alignment["word_onset_reference_count"],
        "word_onset_start_mae_ms": alignment["word_onset_weighted_start_error_ms"] / onset_matches,
        "intra_word_match_rate": intra_matches / alignment["intra_word_reference_count"],
        "intra_word_center_mae_ms": alignment["intra_word_weighted_center_error_ms"] / intra_matches,
        "intra_word_start_mae_ms": alignment["intra_word_weighted_start_error_ms"] / intra_matches,
        "speech_region_containment_outside_rate": alignment["containment_outside_count"] / containment_events,
        "speech_region_containment_before_count": alignment["containment_before_count"],
        "speech_region_containment_after_count": alignment["containment_after_count"],
        "syllable_precision": syllable_matches / alignment["syllable_observation_count"],
        "syllable_recall": syllable_matches / alignment["syllable_target_count"],
        "syllable_center_mae_ms": alignment["syllable_weighted_error_ms"] / syllable_matches,
    }
    report["comprehensive_metrics"] = {}
    for group, metrics in error_samples.items():
        report["comprehensive_metrics"][group] = {}
        for name, raw in metrics.items():
            matched_values = raw["matched"]
            comprehensive_values = raw["comprehensive"]
            reference_count = raw["reference_count"]
            matched_count = raw["matched_count"]
            report["comprehensive_metrics"][group][name] = {
                "reference_count": reference_count,
                "matched_count": matched_count,
                "coverage": matched_count / reference_count if reference_count else 0.0,
                "miss_penalty_ms": raw["penalty_ms"],
                "matched_only_mean_ms": statistics.fmean(matched_values) if matched_values else 0.0,
                "matched_only_median_ms": statistics.median(matched_values) if matched_values else 0.0,
                "all_reference_mean_ms": statistics.fmean(comprehensive_values) if comprehensive_values else 0.0,
                "all_reference_median_ms": statistics.median(comprehensive_values) if comprehensive_values else 0.0,
            }

    output = args.output or run_root / "neural_track_grade_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    aggregate = report["splits"]["aggregate"]
    comprehensive = report["comprehensive_metrics"]["aggregate"]["viseme_center"]
    print(
        "Neural track: "
        f"match={aggregate['match_rate']:.6f}, "
        f"matched_center_mean/median={comprehensive['matched_only_mean_ms']:.3f}/"
        f"{comprehensive['matched_only_median_ms']:.3f} ms, "
        f"all_reference_mean/median={comprehensive['all_reference_mean_ms']:.3f}/"
        f"{comprehensive['all_reference_median_ms']:.3f} ms, "
        f"fallback={aggregate['deterministic_fallback_rate']:.6f}, "
        f"order={aggregate['order_violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
