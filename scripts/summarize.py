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
    print(
        f"Playback health: renderable={summary['renderable_event_rate']:.3f} "
        f"speech_coverage={summary['speech_animation_coverage_rate']:.3f} "
        f"dropout_ms={summary['speech_animation_dropout_ms']:.1f} "
        f"longest_dropout_ms={summary['longest_speech_animation_dropout_ms']:.1f} "
        f"tail_overrun_ms={summary['scheduled_tail_overrun_ms']:.1f} "
        f"after_speech={summary['scheduled_after_speech_event_count']} "
        f"late_bursts={summary['late_commit_burst_count']} "
        f"compressed={summary['compressed_cadence_event_count']} "
        f"compressed_cases={summary['compressed_cadence_case_count']} "
        f"max_compressed_run={summary['max_compressed_cadence_run']} "
        f"failed_cases={summary['playback_health_failure_cases']}"
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
    if summary.get("prosodic_time_mapper_available_cases", 0) > 0:
        print(
            f"Prosodic time mapper: progress_mean_ms={summary['prosodic_time_mapper_progress_mean_ms']:.1f} "
            f"median_ms={summary['prosodic_time_mapper_progress_median_ms']:.1f} "
            f"p90_ms={summary['prosodic_time_mapper_progress_p90_ms']:.1f} "
            f"high_conf_coverage={summary['prosodic_time_mapper_high_confidence_coverage']:.3f} "
            f"high_conf_ms={summary['prosodic_time_mapper_high_confidence_error_ms']:.1f}"
        )
        print(
            f"Prosodic mapper regions: available_cases={summary['prosodic_time_mapper_available_cases']} "
            f"ambiguous_cases={summary['prosodic_time_mapper_ambiguous_cases']} "
            f"paired_regions={summary['prosodic_time_mapper']['paired_region_count']} "
            f"skipped_outside={summary['prosodic_time_mapper']['skipped_outside_regions']}"
        )
        print(
            f"Prosodic pacing projection: syllable_exact={summary['prosodic_time_mapper_syllable_exact_rate']:.3f} "
            f"within_one={summary['prosodic_time_mapper_syllable_within_one_rate']:.3f} "
            f"prior_ms={summary['prosodic_time_mapper_baseline_projection_ms']:.1f} "
            f"corrected_ms={summary['prosodic_time_mapper_corrected_projection_ms']:.1f} "
            f"improved={summary['prosodic_time_mapper_projection_improvement_rate']:.3f}"
        )
        print(
            f"Prosodic advisory rebases: anchors={summary['prosodic_time_mapper_rebase_anchor_count']} "
            f"targets={summary['prosodic_time_mapper_rebase_target_count']} "
            f"prior_ms={summary['prosodic_time_mapper_rebase_baseline_ms']:.1f} "
            f"rebased_ms={summary['prosodic_time_mapper_rebase_ms']:.1f} "
            f"improved={summary['prosodic_time_mapper_rebase_improvement_rate']:.3f} "
            f"degraded={summary['prosodic_time_mapper_rebase_degradation_rate']:.3f}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
