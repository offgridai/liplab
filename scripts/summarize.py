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
            f"phone_p={summary['text_phone_exact_precision']:.3f} "
            f"phone_r={summary['text_phone_exact_recall']:.3f} "
            f"phone_f1={summary['text_phone_exact_f1']:.3f} "
            f"base_f1={summary['text_phone_base_f1']:.3f} "
            f"pause_acc={summary['text_pause_class_accuracy']:.3f} "
            f"break_recall={summary['text_region_break_recall']:.3f} "
            f"landmark_recall={summary['text_landmark_recall']:.3f}"
        )
        worst_phone_words = list(summary.get("text_phone_mismatches_by_word", {}).items())[:8]
        if worst_phone_words:
            print(
                "Phone-plan mismatch words: "
                + ", ".join(
                    f"{word}={values['false_positive'] + values['false_negative']}"
                    for word, values in worst_phone_words
                )
            )
        print(
            f"Text syllables vs MFA: count_ratio={summary['text_syllable_count_ratio']:.3f} "
            f"count_p={summary['text_syllable_count_precision']:.3f} "
            f"count_r={summary['text_syllable_count_recall']:.3f} "
            f"nucleus_p={summary['text_nucleus_exact_precision']:.3f} "
            f"nucleus_r={summary['text_nucleus_exact_recall']:.3f} "
            f"nucleus_f1={summary['text_nucleus_exact_f1']:.3f} "
            f"base_f1={summary['text_nucleus_base_f1']:.3f} "
            f"spn_words_excluded={summary['text_syllable_ungradeable_spn_word_count']}"
        )
        print(
            f"Text duration/pause priors vs MFA: phone_duration_mae_ms={summary['text_phone_duration_mae_ms']:.1f} "
            f"pause_p={summary['text_pause_hint_precision']:.3f} "
            f"pause_r={summary['text_pause_hint_recall']:.3f} "
            f"pause_f1={summary['text_pause_hint_f1']:.3f}"
        )
        print(
            f"Text cumulative priors vs MFA: word_start_ms={summary['text_word_prior_start_ms']:.1f} "
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
            "Streaming speech-region boundaries: "
            f"pair_p={summary['streaming_region_boundary_pair_precision']:.3f} "
            f"pair_r={summary['streaming_region_boundary_pair_recall']:.3f} "
            f"pair_f1={summary['streaming_region_boundary_pair_f1']:.3f} "
            f"matched={summary['streaming_region_boundary_matched_count']}/"
            f"{summary['streaming_region_boundary_reference_count']} "
            f"initial_open={summary['streaming_region_initial_open_recall']:.3f} "
            f"final_close={summary['streaming_region_final_close_recall']:.3f}"
        )
        region_categories = summary["streaming_region_boundary_case_categories"]
        print(
            "Streaming region cases: "
            f"perfect={region_categories['perfect']} "
            f"equal_count_mismatch={region_categories['equal_count_mismatch']} "
            f"undersegmented={region_categories['undersegmented']} "
            f"oversegmented={region_categories['oversegmented']}"
        )
        print(
            "Streaming region candidate ceiling: "
            f"oracle_recall={summary['streaming_region_candidate_oracle_recall']:.3f} "
            f"matched={summary['streaming_region_candidate_matched_count']}/"
            f"{summary['streaming_region_boundary_reference_count']} "
            f"adjudication_misses={summary['streaming_region_adjudication_miss_count']} "
            f"no_candidate={summary['streaming_region_no_candidate_count']} "
            f"false_accepted={summary['streaming_region_false_accepted_count']}"
        )
        print(
            f"Runtime pause/resume: close_p={summary['runtime_pause_close_precision']:.3f} "
            f"close_r={summary['runtime_pause_close_recall']:.3f} "
            f"close_ms={summary['runtime_pause_close_error_ms']:.1f} "
            f"resume_p={summary['runtime_resume_precision']:.3f} "
            f"resume_r={summary['runtime_resume_recall']:.3f} "
            f"resume_ms={summary['runtime_resume_error_ms']:.1f}"
        )
    if summary.get("evidence_surface_available_cases", 0) > 0:
        print(
            f"Bounded evidence surface: postroll_ms={summary['evidence_surface_postroll_ms']:.0f} "
            f"preroll_ms={summary['evidence_surface_preroll_ms']:.0f}"
        )
        for landmark_type, stats in summary.get("evidence_surface_by_type", {}).items():
            print(
                f"  {landmark_type}: precision={stats['precision']:.3f} "
                f"recall={stats['recall']:.3f} f1={stats['f1']:.3f} "
                f"center_ms={stats['mean_abs_error_ms']:.1f} "
                f"latency_ms={stats['mean_decision_latency_ms']:.1f}"
            )
            if landmark_type == "pulse":
                print(
                    f"    count_ratio={stats['count_ratio']:.3f} "
                    f"exact_region_count={stats['exact_region_count_rate']:.3f} "
                    f"count_mae={stats['region_count_mae']:.3f} "
                    f"cluster_r120={stats['clustered_120_recall']:.3f} "
                    f"cluster_r150={stats['clustered_150_recall']:.3f} "
                    f"cluster_r200={stats['clustered_200_recall']:.3f} "
                    f"outside_speech={stats['outside_speech_rate']:.3f}"
                )
        print("Transcript-conditioned evidence surface:")
        for landmark_type, stats in summary.get("conditioned_evidence_surface_by_type", {}).items():
            if landmark_type not in {
                "pulse", "open", "front", "round", "fv", "sibilant", "mbp", "glide"
            }:
                continue
            print(
                f"  {landmark_type}: precision={stats['precision']:.3f} "
                f"recall={stats['recall']:.3f} f1={stats['f1']:.3f} "
                f"center_ms={stats['mean_abs_error_ms']:.1f} "
                f"latency_ms={stats['mean_decision_latency_ms']:.1f}"
            )
        print("Transcript-conditioned intra-envelope phone-family evidence:")
        for landmark_type, stats in summary.get("intra_envelope_phone_by_type", {}).items():
            print(
                f"  {landmark_type}: precision={stats['precision']:.3f} "
                f"recall={stats['recall']:.3f} f1={stats['f1']:.3f} "
                f"center_ms={stats['mean_abs_error_ms']:.1f} "
                f"latency_ms={stats['mean_decision_latency_ms']:.1f}"
            )
    if summary.get("syllable_candidate_set_available_cases", 0) > 0:
        recalls = summary.get("syllable_candidate_set_recall_at", {})
        values = " ".join(
            f"r@{rank}={recalls.get(str(rank), {}).get('recall', 0.0):.3f}"
            for rank in range(1, 7)
        )
        print(
            "Causal syllable candidate sets: "
            f"matched_pulses={summary['syllable_candidate_set_matched_pulses']} "
            f"unmatched_pulses={summary['syllable_candidate_set_unmatched_pulses']} "
            f"{values}"
        )
    if summary.get("syllable_assignment_available_cases", 0) > 0:
        print(
            "Monotonic syllable assignment: "
            f"precision={summary['syllable_assignment_exact_precision']:.3f} "
            f"recall={summary['syllable_assignment_exact_recall']:.3f} "
            f"within1={summary['syllable_assignment_within_one_accuracy']:.3f} "
            f"assigned={summary['syllable_assignment_count']}/"
            f"{summary['syllable_assignment_target_count']}"
        )
    if summary.get("resolved_pulse_track_available_cases", 0) > 0:
        print(
            "Transcript-conditioned pulse track: "
            f"count_ratio={summary['resolved_pulse_track_assignment_ratio']:.3f} "
            f"precision={summary['resolved_pulse_track_exact_precision']:.3f} "
            f"recall={summary['resolved_pulse_track_exact_recall']:.3f} "
            f"within1={summary['resolved_pulse_track_within_one_accuracy']:.3f} "
            f"assigned={summary['resolved_pulse_track_assignment_count']}/"
            f"{summary['resolved_pulse_track_target_count']}"
        )
    if summary.get("syllable_position_available_cases", 0) > 0:
        print(
            "Retrospective syllable assignment: "
            f"exact={summary['syllable_position_exact_rate']:.3f} "
            f"within1={summary['syllable_position_within_one_rate']:.3f} "
            f"word={summary['syllable_position_word_rate']:.3f} "
            f"timing_p={summary['syllable_position_timing_precision']:.3f} "
            f"timing_r={summary['syllable_position_timing_recall']:.3f}"
        )
        print(
            "Retrospective assignment confidence: "
            f"coverage={summary['syllable_position_confident_coverage']:.3f} "
            f"exact={summary['syllable_position_confident_exact_rate']:.3f}"
        )
    if summary.get("dense_position_available_cases", 0) > 0:
        print(
            "Causal current transcript position: "
            f"coverage={summary['dense_position_coverage']:.3f} "
            f"exact_recall={summary['dense_position_exact_recall']:.3f} "
            f"within1={summary['dense_position_within_one_rate']:.3f} "
            f"word={summary['dense_position_word_rate']:.3f} "
            f"ahead={summary['dense_position_ahead_rate']:.3f} "
            f"behind={summary['dense_position_behind_rate']:.3f} "
            f"signed={summary['dense_position_mean_signed_error']:+.2f}syll"
        )
    if summary.get("historical_anchor_available_cases", 0) > 0:
        print(
            "Stable historical anchors: "
            f"count={summary['historical_anchor_count']} "
            f"exact={summary['historical_anchor_exact_rate']:.3f} "
            f"within1={summary['historical_anchor_within_one_rate']:.3f} "
            f"timing_p={summary['historical_anchor_timing_precision']:.3f} "
            f"timing_r={summary['historical_anchor_timing_recall']:.3f} "
            f"latency_ms={summary['historical_anchor_median_latency_ms']:.1f} "
            f"gap_ms={summary['historical_anchor_median_gap_ms']:.1f}"
        )
        print(
            "Historical-anchor forward rebase: "
            f"coverage={summary['historical_rebase_coverage']:.3f} "
            f"exact_recall={summary['historical_rebase_exact_recall']:.3f} "
            f"within1={summary['historical_rebase_within_one_rate']:.3f} "
            f"word={summary['historical_rebase_word_rate']:.3f}"
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
        exact_rebase = summary.get("runtime_exact_pulse_rebase", {})
        speculative_rebase = summary.get("runtime_speculative_pulse_rebase", {})
        print(
            "Runtime pulse rebasing: "
            f"exact_p={exact_rebase.get('precision', 0.0):.3f} "
            f"exact_r={exact_rebase.get('recall', 0.0):.3f} "
            f"exact_n={exact_rebase.get('observation_count', 0)} "
            f"exact_shift_ms={summary.get('runtime_exact_pulse_mean_correction_ms', 0.0):.1f} "
            f"spec_p={speculative_rebase.get('precision', 0.0):.3f} "
            f"spec_r={speculative_rebase.get('recall', 0.0):.3f} "
            f"spec_n={speculative_rebase.get('observation_count', 0)} "
            f"spec_shift_ms={summary.get('runtime_speculative_pulse_mean_correction_ms', 0.0):.1f}"
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
        print(
            f"Runtime region entry: first_pulse={syllable_diag.get('first_pulse_anchor_count', 0)} "
            f"fallback={syllable_diag.get('first_pulse_fallback_count', 0)}"
        )
        print(
            f"Runtime internal-region carry: episodes={syllable_diag.get('internal_region_carry_count', 0)} "
            f"pending_events={syllable_diag.get('internal_region_carried_event_count', 0)}"
        )
        print(
            f"Runtime region cutoff: episodes={syllable_diag.get('closed_region_cutoff_count', 0)} "
            f"dropped_events={syllable_diag.get('dropped_closed_region_event_count', 0)}"
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
    if summary.get("audio_clock_available_cases", 0) > 0:
        print(
            "Audio evidence clock: "
            f"speech_updates={summary['audio_clock_speech_update_count']} "
            f"gate_open={summary['audio_clock_gate_open_rate_during_speech']:.3f} "
            f"raw_support={summary['audio_clock_raw_support_rate_during_speech']:.3f} "
            f"fail_soft={summary['audio_clock_fail_soft_rate_during_speech']:.3f} "
            f"credit_grants={summary['audio_clock_credit_grant_count']} "
            f"onset_grants={summary['audio_clock_speech_onset_grant_count']} "
            f"fail_soft_activations={summary['audio_clock_fail_soft_activation_count']} "
            f"fail_soft_advance_s={summary['audio_clock_fail_soft_advance_sec']:.2f} "
            f"lull_pause_s={summary['audio_clock_observed_lull_pause_sec']:.2f}"
        )
        print(
            "Text/audio boundary handshake: "
            f"hypotheses={summary.get('text_boundary_hypothesis_count', 0)} "
            f"confirmed_pauses={summary.get('text_boundary_confirmed_pause_count', 0)} "
            f"continuous_speech={summary.get('text_boundary_rejected_continuous_speech_count', 0)} "
            f"terminal_unresolved={summary.get('text_boundary_terminal_unresolved_count', 0)} "
            f"terminal_waiting_resume={summary.get('text_boundary_terminal_waiting_resume_count', 0)}"
        )
    if summary.get("audio_health_available_cases", 0) > 0:
        print(
            "Audio clock health vs MFA: "
            f"first_delay_ms={summary['audio_health_first_region_animation_delay_ms']:.1f} "
            f"first_p90_ms={summary['audio_health_first_region_animation_delay_p90_ms']:.1f} "
            f"starved_cases={summary['audio_health_first_region_starvation_cases']} "
            f"split_word_cases={summary['audio_health_cross_region_word_split_cases']} "
            f"split_words={summary['audio_health_cross_region_word_split_count']} "
            f"resume_exact={summary['audio_health_resume_cursor_exact_rate']:.3f} "
            f"resume_behind={summary['audio_health_resume_cursor_behind_count']} "
            f"resume_ahead={summary['audio_health_resume_cursor_ahead_count']} "
            f"uncommitted={summary['audio_health_uncommitted_visible_count']} "
            f"suffix={summary['audio_health_uncommitted_visible_suffix_count']} "
            f"false_hold={summary['audio_health_gate_closed_during_gold_speech_rate']:.3f} "
            f"pause_leak={summary['audio_health_gate_open_during_gold_pause_rate']:.3f}"
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
