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
            f"break_recall={summary['text_region_break_recall']:.3f}"
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
            f"phone_center_ms={summary['text_phone_prior_center_ms']:.1f}"
        )
    if summary.get("streaming_region_boundary_available_cases", 0) > 0:
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
    if summary.get("runtime_syllable_available_cases", 0) > 0:
        print(
            f"Runtime syllable assignment: precision={summary['runtime_syllable_precision']:.3f} "
            f"recall={summary['runtime_syllable_recall']:.3f} "
            f"count_ratio={summary['runtime_syllable_count_ratio']:.3f} "
            f"center_ms={summary['runtime_syllable_error_ms']:.1f}"
        )
    if summary.get("audio_health_available_cases", 0) > 0:
        print(
            "Runtime health vs MFA: "
            f"first_delay_ms={summary['audio_health_first_region_animation_delay_ms']:.1f} "
            f"first_p90_ms={summary['audio_health_first_region_animation_delay_p90_ms']:.1f} "
            f"starved_cases={summary['audio_health_first_region_starvation_cases']} "
            f"split_word_cases={summary['audio_health_cross_region_word_split_cases']} "
            f"split_words={summary['audio_health_cross_region_word_split_count']} "
            f"resume_exact={summary['audio_health_resume_cursor_exact_rate']:.3f} "
            f"resume_behind={summary['audio_health_resume_cursor_behind_count']} "
            f"resume_ahead={summary['audio_health_resume_cursor_ahead_count']} "
            f"uncommitted={summary['audio_health_uncommitted_visible_count']} "
            f"suffix={summary['audio_health_uncommitted_visible_suffix_count']}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
