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
    if summary.get("list_word_count", 0):
        print(
            f"List words: count={summary['list_word_count']} "
            f"start_ms={summary['list_word_start_ms']:.1f} "
            f"median_ms={summary['list_word_start_median_ms']:.1f}"
        )
    print(
        f"Visible timing: coverage={summary['visible_articulation_coverage_rate']:.3f} "
        f"center_ms={summary['visible_articulation_center_ms']:.1f} "
        f"start_ms={summary['visible_articulation_start_ms']:.1f} "
        f"end_ms={summary['visible_articulation_end_ms']:.1f}"
    )
    print(
        f"Intra-word: coverage={summary['intra_word_coverage_rate']:.3f} "
        f"center_ms={summary['intra_word_center_ms']:.1f}"
    )
    if summary.get("text_phone_plan_count", 0):
        roles = summary.get("visual_role_counts", {})
        print(
            f"Visual projection: phones={summary['text_phone_plan_count']} "
            f"events={summary['visible_articulation_count']} "
            f"rate={summary['visible_articulation_rate']:.3f} "
            f"primary={roles.get('primary_pose', 0)} "
            f"supporting={roles.get('supporting_pose', 0)} "
            f"coarticulated={roles.get('coarticulated', 0)} "
            f"timing_only={roles.get('timing_only', 0)} "
            f"gold_correspondence={summary.get('visual_gold_correspondence_rate', 0.0):.3f} "
            f"unmatched={summary.get('unmatched_articulation_event_count', 0)}"
        )
    if summary.get("pronunciation_word_count", 0):
        print(
            f"Pronunciations: words={summary['pronunciation_word_count']} "
            f"alternates_available={summary['pronunciation_alternate_available_word_count']} "
            f"alternates_selected={summary['pronunciation_alternate_selected_word_count']}"
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
    if summary.get("runtime_syllable_available_cases", 0) > 0:
        print(
            f"Runtime syllable assignment: precision={summary['runtime_syllable_precision']:.3f} "
            f"recall={summary['runtime_syllable_recall']:.3f} "
            f"count_ratio={summary['runtime_syllable_count_ratio']:.3f} "
            f"progress_ratio={summary.get('runtime_syllable_progress_ratio', 0.0):.3f} "
            f"center_ms={summary['runtime_syllable_error_ms']:.1f}"
        )
        syllable_diag = summary.get("runtime_syllable_diagnostics", {})
        print(
            f"Runtime syllable flow: pulses={syllable_diag.get('pulse_count', 0)} "
            f"assigned={syllable_diag.get('assignment_count', 0)} "
            f"low={syllable_diag.get('reject_low_prominence_count', 0)} "
            f"pre_section={syllable_diag.get('reject_before_section_count', 0)} "
            f"late={syllable_diag.get('late_assignment_count', 0)} "
            f"no_candidate={syllable_diag.get('reject_no_candidate_count', 0)} "
            f"ambiguous={syllable_diag.get('ambiguous_assignment_count', 0)} "
            f"duplicates={syllable_diag.get('duplicate_pulse_count', 0)}"
        )
        print(
            f"Runtime syllable progress: count={summary.get('runtime_syllable_progress_count', 0)} "
            f"mae={summary.get('runtime_syllable_progress_mean_abs_count_error', 0.0):.2f} "
            f"exact={summary.get('runtime_syllable_progress_exact_case_rate', 0.0):.3f} "
            f"within1={summary.get('runtime_syllable_progress_within_one_case_rate', 0.0):.3f} "
            f"over={summary.get('runtime_syllable_progress_over_case_count', 0)} "
            f"under={summary.get('runtime_syllable_progress_under_case_count', 0)}"
        )
        print(
            f"Runtime strong-phone flow: candidates={syllable_diag.get('strong_phone_candidate_count', 0)} "
            f"assigned={syllable_diag.get('strong_phone_assignment_count', 0)}"
        )
    if summary.get("strict_punctuation_available_cases", 0) > 0:
        print(
            f"Strict punctuation mapping: close_p={summary['strict_punctuation_close_precision']:.3f} "
            f"close_r={summary['strict_punctuation_close_recall']:.3f} "
            f"resume_p={summary['strict_punctuation_resume_precision']:.3f} "
            f"resume_r={summary['strict_punctuation_resume_recall']:.3f}"
        )
    if summary.get("soft_list_boundary_target_count", 0) > 0:
        print(
            f"List boundaries: targets={summary['soft_list_boundary_target_count']} "
            f"close_p={summary['soft_list_boundary_close_precision']:.3f} "
            f"close_r={summary['soft_list_boundary_close_recall']:.3f} "
            f"resume_p={summary['soft_list_boundary_resume_precision']:.3f} "
            f"resume_r={summary['soft_list_boundary_resume_recall']:.3f}"
        )
    if summary.get("pause_safety_available_cases", 0) > 0:
        print(
            f"Pause safety: pair_rate={summary['pause_safety_pair_rate']:.3f} "
            f"early_resume={summary['pause_safety_early_resume_count']} "
            f"leaking_boundaries={summary['pause_safety_leakage_boundary_count']} "
            f"leakage_ms={summary['pause_safety_total_leakage_ms']:.1f} "
            f"close_signed_ms={summary['pause_safety_mean_signed_close_latency_ms']:.1f} "
            f"resume_signed_ms={summary['pause_safety_mean_signed_resume_latency_ms']:.1f} "
            f"post_resume_ms={summary['pause_safety_mean_post_resume_animation_delay_ms']:.1f} "
            f"false_hold_ms={summary['pause_safety_false_hold_during_gold_speech_ms']:.1f} "
            f"timeouts={summary['pause_safety_unsafe_timeout_release_count']} "
            f"false_pause={summary['pause_safety_false_pause_resolution_count']} "
            f"unresolved={summary['pause_safety_unresolved_hold_count']}"
        )
        print(
            "Syllable continuation safety: "
            f"releases={summary['pause_safety_syllable_continuous_release_count']} "
            f"precision={summary['pause_safety_syllable_continuous_precision']:.3f} "
            f"false_pause={summary['pause_safety_syllable_continuous_false_pause_release_count']} "
            f"cases={summary['pause_safety_syllable_continuous_false_pause_case_count']} "
            f"pause_preservation={summary['pause_safety_syllable_continuous_pause_preservation_rate']:.3f} "
            f"leakage_ms={summary['pause_safety_syllable_continuous_false_pause_leakage_ms']:.1f}"
        )
        print(
            "Syllable continuation safety (all graded): "
            f"releases={summary['pause_safety_syllable_continuous_all_graded_release_count']} "
            f"false_pause={summary['pause_safety_syllable_continuous_all_graded_false_pause_release_count']} "
            f"cases={summary['pause_safety_syllable_continuous_all_graded_false_pause_case_count']} "
            f"leakage_ms={summary['pause_safety_syllable_continuous_all_graded_false_pause_leakage_ms']:.1f}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
