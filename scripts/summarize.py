import pathlib

from grade_summary import compute_summary, load_case_grades, analyze_audio_word_boundary_reasons, write_audio_word_boundary_reason_analysis

root = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "runs" / "latest"
gold_root = pathlib.Path(__file__).resolve().parents[1] / "inputs" / "gold"
rows, graded, ungraded = load_case_grades(root, gold_root)

for row in rows:
    pause = row["pause_alignment"]
    word = row["word_onset_alignment"]
    intra = row["intra_word_alignment"]
    flags = []
    if row["reference_count"] > 0 and row["matched_count"] == 0:
        flags.append("DEGENERATE")
    if row["layers"]["speech"].get("count_mismatch", False):
        flags.append("speech_regions_mismatch")
    if row["layers"]["phone_occupancy"].get("count_mismatch", False):
        flags.append("phone_occupancy_regions_mismatch")
    if row["layers"]["speech_visible"].get("count_mismatch", False):
        flags.append("visible_speech_regions_mismatch")
    if pause.get("sentence_region_count_mismatch", False):
        flags.append("sentence_regions_mismatch")
    if pause.get("clause_region_count_mismatch", False):
        flags.append("clause_regions_mismatch")
    flag_text = f" flags={','.join(flags)}" if flags else ""
    direct = row.get("direct_aligner", {})
    audio_boundary = row.get("audio_word_boundaries", {})
    audio_boundary_text = ""
    if audio_boundary.get("available", False):
        audio_boundary_text = (
            f" audio_word_boundary_selected={audio_boundary.get('selected_count', 0)}/{audio_boundary.get('reference_count', 0)}"
            f" audio_word_boundary_f1_100={audio_boundary.get('f1_100', 0.0):.3f}"
            f" audio_word_boundary_recall_100={audio_boundary.get('recall_100', 0.0):.3f}"
        )
    direct_text = ""
    if direct.get("available", False):
        direct_text = (
            f" direct_aligner_matched={direct.get('matched_count', 0)}/{direct.get('reference_count', 0)}"
            f" direct_aligner_center_ms={direct.get('mean_abs_center_error_ms', 0.0)}"
            f" direct_aligner_start_ms={direct.get('mean_abs_start_error_ms', 0.0)}"
        )
    print(
        f"{row['case']}: gold={'yes' if row['gold_available'] else 'no'} "
        f"matched={row['matched_count']}/{row['reference_count']}"
        f"{direct_text} "
        f"mean_ms={row['mean_abs_center_error_ms']} median_ms={row.get('median_abs_center_error_ms', 0.0)} "
        f"p90_ms={row['p90_abs_center_error_ms']} "
        f"start_ms={row['mean_abs_start_error_ms']} end_ms={row['mean_abs_end_error_ms']} "
        f"speech_start_ms={row['layers']['speech'].get('mean_abs_start_error_ms', 0.0):.3f} "
        f"speech_end_ms={row['layers']['speech'].get('mean_abs_end_error_ms', 0.0):.3f} "
        f"speech_tail_ms={row['layers']['speech'].get('tail_leakage_ms', 0.0):.3f} "
        f"phone_occupancy_start_ms={row['layers']['phone_occupancy'].get('mean_abs_start_error_ms', 0.0):.3f} "
        f"phone_occupancy_end_ms={row['layers']['phone_occupancy'].get('mean_abs_end_error_ms', 0.0):.3f} "
        f"visible_speech_start_ms={row['layers']['speech_visible'].get('mean_abs_start_error_ms', 0.0):.3f} "
        f"visible_speech_end_ms={row['layers']['speech_visible'].get('mean_abs_end_error_ms', 0.0):.3f} "
        f"sentence_start_ms={pause.get('mean_abs_sentence_region_start_error_ms', 0.0)} "
        f"word_onset_ms={word.get('mean_abs_start_error_ms', 0.0)} "
        f"word_onset_median_ms={word.get('median_abs_start_error_ms', 0.0)} "
        f"word_duration_ms={row['layers']['words'].get('mean_abs_duration_error_ms', 0.0):.3f} "
        f"word_duration_norm={row['layers']['words'].get('mean_abs_duration_error_norm', 0.0):.4f} "
        f"word_duration_l1_norm={row['layers']['words'].get('duration_l1_norm', 0.0):.4f} "
        f"word_stretch_abs_log2={row['layers']['words'].get('mean_abs_stretch_log2', 0.0):.4f} "
        f"intra_word_ms={intra.get('mean_abs_center_error_ms', 0.0)} "
        f"intra_word_median_ms={intra.get('median_abs_center_error_ms', 0.0)} "
        f"speech_f1={row['layers']['speech']['f1']:.3f} "
        f"word_f1={row['layers']['words']['f1']:.3f} "
        f"phoneme_cov={row['layers']['phonemes']['coverage_rate']:.3f}"
        f"{audio_boundary_text}"
        f"{flag_text}"
    )

if rows:
    summary = compute_summary(rows, graded, ungraded)
    print(
        f"SUMMARY cases={summary['cases']} graded_cases={summary['graded_cases']} "
        f"ungraded_cases={summary['ungraded_cases']} "
        f"matched={summary['matched_count']}/{summary['reference_count']} "
        f"match_rate={summary['match_rate']:.4f} "
        f"mean_center_ms={summary['mean_center_ms']:.3f} "
        f"median_center_ms={summary['median_center_ms']:.3f} "
        f"p90_center_ms={summary['p90_center_ms']:.3f} "
        f"mean_start_ms={summary['mean_start_ms']:.3f} "
        f"mean_end_ms={summary['mean_end_ms']:.3f} "
        f"mean_duration_ms={summary['mean_duration_ms']:.3f} "
        f"mean_sentence_start_ms={summary['mean_sentence_start_ms']:.3f} "
        f"median_sentence_start_ms={summary['median_sentence_start_ms']:.3f} "
        f"p90_sentence_start_ms={summary['p90_sentence_start_ms']:.3f} "
        f"mean_clause_start_ms={summary['mean_clause_start_ms']:.3f} "
        f"median_clause_start_ms={summary['median_clause_start_ms']:.3f} "
        f"mean_word_onset_ms={summary['mean_word_onset_ms']:.3f} "
        f"median_word_onset_ms={summary['median_word_onset_ms']:.3f} "
        f"p90_word_onset_ms={summary['p90_word_onset_ms']:.3f} "
        f"mean_intra_word_ms={summary['mean_intra_word_ms']:.3f} "
        f"median_intra_word_ms={summary['median_intra_word_ms']:.3f} "
        f"order_fail_cases={summary['order_fail_cases']} "
        f"degenerate_cases={summary['degenerate_cases']} "
        f"speech_region_mismatches={summary['speech_region_count_mismatch_cases']} "
        f"phone_occupancy_region_mismatches={summary['phone_occupancy_region_count_mismatch_cases']} "
        f"visible_speech_region_mismatches={summary['visible_speech_region_count_mismatch_cases']} "
        f"sentence_region_mismatches={summary['sentence_region_count_mismatch_cases']} "
        f"clause_region_mismatches={summary['clause_region_count_mismatch_cases']}"
    )
    print(
        f"HIERARCHY speech_precision={summary['speech_precision']:.4f} "
        f"speech_recall={summary['speech_recall']:.4f} speech_f1={summary['speech_f1']:.4f} "
        f"speech_boundary_start_ms={summary['speech_boundary_start_ms']:.3f} "
        f"speech_boundary_start_median_ms={summary['speech_boundary_start_median_ms']:.3f} "
        f"speech_boundary_end_ms={summary['speech_boundary_end_ms']:.3f} "
        f"speech_tail_leakage_ms={summary['speech_tail_leakage_ms']:.3f} "
        f"phone_occupancy_f1={summary['phone_occupancy_f1']:.4f} "
        f"phone_occupancy_start_ms={summary['phone_occupancy_boundary_start_ms']:.3f} "
        f"phone_occupancy_end_ms={summary['phone_occupancy_boundary_end_ms']:.3f} "
        f"phone_occupancy_tail_leakage_ms={summary['phone_occupancy_tail_leakage_ms']:.3f} "
        f"visible_speech_f1={summary['visible_speech_f1']:.4f} "
        f"visible_speech_start_ms={summary['visible_speech_boundary_start_ms']:.3f} "
        f"visible_speech_end_ms={summary['visible_speech_boundary_end_ms']:.3f} "
        f"visible_speech_tail_leakage_ms={summary['visible_speech_tail_leakage_ms']:.3f} "
        f"word_precision={summary['word_precision']:.4f} "
        f"word_recall={summary['word_recall']:.4f} word_f1={summary['word_f1']:.4f} "
        f"word_start_ms={summary['word_start_ms']:.3f} "
        f"word_start_median_ms={summary['word_start_median_ms']:.3f} "
        f"word_start_p90_ms={summary['word_start_p90_ms']:.3f} "
        f"word_end_ms={summary['word_end_ms']:.3f} "
        f"word_duration_ms={summary['word_duration_ms']:.3f} "
        f"word_duration_median_ms={summary.get('word_duration_median_ms', 0.0):.3f} "
        f"word_duration_p90_ms={summary.get('word_duration_p90_ms', 0.0):.3f} "
        f"word_duration_norm={summary.get('word_duration_norm', 0.0):.4f} "
        f"word_duration_l1_norm={summary.get('word_duration_l1_norm', 0.0):.4f} "
        f"word_stretch_abs_log2={summary.get('word_stretch_abs_log2', 0.0):.4f} "
        f"word_head_start_ms={summary['word_head_start_ms']:.3f} "
        f"word_head_start_median_ms={summary['word_head_start_median_ms']:.3f} "
        f"word_head_start_p90_ms={summary['word_head_start_p90_ms']:.3f} "
        f"phoneme_coverage={summary['phoneme_coverage_rate']:.4f} "
        f"phoneme_center_ms={summary['phoneme_center_ms']:.3f} "
        f"phoneme_center_median_ms={summary['phoneme_center_median_ms']:.3f} "
        f"phoneme_center_p90_ms={summary['phoneme_center_p90_ms']:.3f} "
        f"phoneme_start_ms={summary['phoneme_start_ms']:.3f} "
        f"phoneme_end_ms={summary['phoneme_end_ms']:.3f} "
        f"intra_word_coverage={summary['intra_word_coverage_rate']:.4f} "
        f"intra_word_center_ms={summary['intra_word_center_ms']:.3f}"
    )
    print(
        f"DIRECT_ALIGNER cases={summary.get('direct_aligner_available_cases', 0)} "
        f"match_rate={summary.get('direct_aligner_match_rate', 0.0):.4f} "
        f"center_ms={summary.get('direct_aligner_center_ms', 0.0):.3f} "
        f"center_median_ms={summary.get('direct_aligner_center_median_ms', 0.0):.3f} "
        f"center_p90_ms={summary.get('direct_aligner_center_p90_ms', 0.0):.3f} "
        f"start_ms={summary.get('direct_aligner_start_ms', 0.0):.3f} "
        f"end_ms={summary.get('direct_aligner_end_ms', 0.0):.3f} "
        f"duration_ms={summary.get('direct_aligner_duration_ms', 0.0):.3f} "
        f"word_onset_ms={summary.get('direct_aligner_word_onset_ms', 0.0):.3f} "
        f"intra_word_coverage={summary.get('direct_aligner_intra_word_coverage_rate', 0.0):.4f} "
        f"intra_word_center_ms={summary.get('direct_aligner_intra_word_center_ms', 0.0):.3f}"
    )

if rows:
    summary = compute_summary(rows, graded, ungraded)
    print(
        f"AUDIO_WORD_BOUNDARY cases={summary.get('audio_word_boundary_available_cases', 0)} "
        f"refs={summary.get('audio_word_boundary_reference_count', 0)} "
        f"candidates={summary.get('audio_word_boundary_candidate_count', 0)} "
        f"selected={summary.get('audio_word_boundary_selected_count', 0)} "
        f"precision_50={summary.get('audio_word_boundary_precision_50', 0.0):.4f} "
        f"recall_50={summary.get('audio_word_boundary_recall_50', 0.0):.4f} "
        f"f1_50={summary.get('audio_word_boundary_f1_50', 0.0):.4f} "
        f"precision_100={summary.get('audio_word_boundary_precision_100', 0.0):.4f} "
        f"recall_100={summary.get('audio_word_boundary_recall_100', 0.0):.4f} "
        f"f1_100={summary.get('audio_word_boundary_f1_100', 0.0):.4f} "
        f"precision_150={summary.get('audio_word_boundary_precision_150', 0.0):.4f} "
        f"recall_150={summary.get('audio_word_boundary_recall_150', 0.0):.4f} "
        f"f1_150={summary.get('audio_word_boundary_f1_150', 0.0):.4f} "
        f"ref_nearest_median_ms={summary.get('audio_word_boundary_ref_nearest_median_ms', 0.0):.3f} "
        f"ref_nearest_p90_ms={summary.get('audio_word_boundary_ref_nearest_p90_ms', 0.0):.3f} "
        f"candidate_nearest_median_ms={summary.get('audio_word_boundary_candidate_nearest_median_ms', 0.0):.3f} "
        f"candidate_nearest_p90_ms={summary.get('audio_word_boundary_candidate_nearest_p90_ms', 0.0):.3f} "
        f"raw_recall_100={summary.get('audio_word_boundary_raw_recall_100', 0.0):.4f} "
        f"top1_100={summary.get('audio_word_boundary_top1_recall_100', 0.0):.4f} "
        f"top2_100={summary.get('audio_word_boundary_top2_recall_100', 0.0):.4f} "
        f"top3_100={summary.get('audio_word_boundary_top3_recall_100', 0.0):.4f} "
        f"top5_100={summary.get('audio_word_boundary_top5_recall_100', 0.0):.4f} "
        f"score_sep={summary.get('audio_word_boundary_score_separation', 0.0):.4f}"
    )


if rows:
    reason_analysis = analyze_audio_word_boundary_reasons(root)
    if reason_analysis.get("candidate_count", 0) > 0:
        csv_path, json_path = write_audio_word_boundary_reason_analysis(root, reason_analysis)
        print(
            f"AUDIO_WORD_BOUNDARY_REASON_ANALYSIS candidates={reason_analysis.get('candidate_count', 0)} "
            f"refs={reason_analysis.get('reference_count', 0)} "
            f"base_precision_100={reason_analysis.get('base_precision_100', 0.0):.4f} "
            f"csv={csv_path.name} weights={json_path.name}"
        )
        top_single = reason_analysis.get("top_single", [])[:6]
        if top_single:
            parts = []
            for row in top_single:
                parts.append(
                    f"{row['reason']}:p100={row['precision_100']:.3f},n={row['count']},w={row['log_odds_weight_100']:.2f}"
                )
            print("AUDIO_WORD_BOUNDARY_TOP_REASONS " + " | ".join(parts))
        top_combo = reason_analysis.get("top_combo", [])[:6]
        if top_combo:
            parts = []
            for row in top_combo:
                parts.append(
                    f"{row['reason']}:p100={row['precision_100']:.3f},n={row['count']},w={row['log_odds_weight_100']:.2f}"
                )
            print("AUDIO_WORD_BOUNDARY_TOP_COMBOS " + " | ".join(parts))
