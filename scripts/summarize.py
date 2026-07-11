import json
import pathlib
import sys

from grade_summary import compute_summary, load_case_grades


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    rows, graded, ungraded = load_case_grades(latest, root / "inputs" / "gold")
    if not rows:
        print("No grade outputs found. Run the corpus first.", file=sys.stderr)
        return 2

    summary = compute_summary(rows, graded, ungraded)
    summary_path = latest / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"Cases: total={summary['cases']} graded={summary['graded_cases']} "
        f"qualified={summary['qualified_cases']} mismatched_regions={summary['speech_region_count_mismatch_cases']}"
    )
    print(
        f"Speech regions: match_rate={summary['speech_match_rate']:.3f} "
        f"start_ms={summary['speech_boundary_start_ms']:.1f} "
        f"end_ms={summary['speech_boundary_end_ms']:.1f} "
        f"tail_ms={summary['speech_tail_leakage_ms']:.1f}"
    )
    print(
        f"Words: match_rate={summary['word_match_rate']:.3f} "
        f"start_ms={summary['word_head_start_ms']:.1f} "
        f"duration_ms={summary['word_duration_ms']:.1f}"
    )
    print(
        f"Phonemes: coverage={summary['phoneme_coverage_rate']:.3f} "
        f"center_ms={summary['phoneme_center_ms']:.1f} "
        f"start_ms={summary['phoneme_start_ms']:.1f} "
        f"end_ms={summary['phoneme_end_ms']:.1f}"
    )
    print(
        f"Intra-word: coverage={summary['intra_word_coverage_rate']:.3f} "
        f"center_ms={summary['intra_word_center_ms']:.1f}"
    )
    if summary["text_plan_available_cases"] > 0:
        print(
            f"Text plan vs MFA: words={summary['text_word_sequence_exact_rate']:.3f} "
            f"phones={summary['text_phone_exact_rate']:.3f} "
            f"pause_acc={summary['text_pause_class_accuracy']:.3f} "
            f"break_recall={summary['text_region_break_recall']:.3f} "
            f"landmark_recall={summary['text_landmark_recall']:.3f}"
        )
        print(
            f"Text priors vs MFA: word_start_ms={summary['text_word_prior_start_ms']:.1f} "
            f"phone_center_ms={summary['text_phone_prior_center_ms']:.1f} "
            f"landmark_center_ms={summary['text_landmark_prior_center_ms']:.1f}"
        )
    if summary["runtime_detector_available_cases"] > 0:
        print(
            f"Runtime detector vs text plan: selection_rate={summary['runtime_detector_planned_landmark_selection_rate']:.3f} "
            f"timing_precision={summary['runtime_detector_planned_landmark_precision']:.3f} "
            f"timing_recall={summary['runtime_detector_planned_landmark_recall']:.3f} "
            f"center_ms={summary['runtime_detector_planned_landmark_center_ms']:.1f}"
        )
        print(
            f"Runtime pause/resume: close_p={summary['runtime_pause_close_precision']:.3f} "
            f"close_r={summary['runtime_pause_close_recall']:.3f} "
            f"close_ms={summary['runtime_pause_close_error_ms']:.1f} "
            f"resume_p={summary['runtime_resume_precision']:.3f} "
            f"resume_r={summary['runtime_resume_recall']:.3f} "
            f"resume_ms={summary['runtime_resume_error_ms']:.1f}"
        )
    if summary.get("prosodic_peak_available_cases", 0) > 0:
        print(
            f"Prosodic peaks advisory: precision={summary['prosodic_peak_precision']:.3f} "
            f"recall={summary['prosodic_peak_recall']:.3f} "
            f"center_ms={summary['prosodic_peak_error_ms']:.1f} "
            f"decision_ms={summary['prosodic_peak_decision_latency_ms']:.1f}"
        )
        print(
            f"Raw syllabic pulses vs MFA vowel intervals: precision={summary['raw_prosodic_peak_precision']:.3f} "
            f"recall={summary['raw_prosodic_peak_recall']:.3f}"
        )
    if summary.get("strict_punctuation_available_cases", 0) > 0:
        print(
            f"Strict punctuation mapping: close_p={summary['strict_punctuation_close_precision']:.3f} "
            f"close_r={summary['strict_punctuation_close_recall']:.3f} "
            f"resume_p={summary['strict_punctuation_resume_precision']:.3f} "
            f"resume_r={summary['strict_punctuation_resume_recall']:.3f}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
