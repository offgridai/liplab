import json
import pathlib
import sys

from grade_summary import compute_summary, load_case_grades, write_summary


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "outputs" / "runs" / "latest"
    gold_root = pathlib.Path(__file__).resolve().parents[1] / "inputs" / "gold"
    rows, graded, ungraded = load_case_grades(root, gold_root)
    summary = compute_summary(rows, graded, ungraded)
    write_summary(root, summary)

    for row in graded:
        case = row.get("case", "")
        aps = row.get("audio_progress_summary", {})
        srd = row.get("speech_region_diagnostics", {})
        gcs = row.get("gap_candidate_summary", {})
        pcs = row.get("phone_class_summary", {})
        worst = "none"
        if pcs:
            worst_name, worst_stats = max(
                pcs.items(),
                key=lambda item: (float(item[1].get("center_ms", 0.0)), int(item[1].get("unmapped", 0))),
            )
            worst = (
                f"{worst_name}:n={int(worst_stats.get('n', 0))} "
                f"center_ms={float(worst_stats.get('center_ms', 0.0)):.1f} "
                f"unmapped={int(worst_stats.get('unmapped', 0))}"
            )
        print(
            f"CASE_REGION_DIAG {case} "
            f"gold_regions={int(srd.get('gold_count', 0))} "
            f"pred_regions={int(srd.get('predicted_count', 0))} "
            f"count_delta={int(srd.get('count_delta', 0))} "
            f"indexed_start_ms={float(srd.get('indexed', {}).get('mean_start_error_ms', 0.0)):.1f} "
            f"indexed_end_ms={float(srd.get('indexed', {}).get('mean_end_error_ms', 0.0)):.1f} "
            f"indexed_overlap={float(srd.get('indexed', {}).get('mean_overlap_ratio', 0.0)):.3f} "
            f"overlap_start_ms={float(srd.get('overlap', {}).get('mean_start_error_ms', 0.0)):.1f} "
            f"overlap_end_ms={float(srd.get('overlap', {}).get('mean_end_error_ms', 0.0)):.1f} "
            f"overlap_ratio={float(srd.get('overlap', {}).get('mean_overlap_ratio', 0.0)):.3f} "
            f"overlap_matched={int(srd.get('overlap', {}).get('overlap_matched_count', 0))} "
            f"nearest_matched={int(srd.get('overlap', {}).get('nearest_matched_count', 0))} "
            f"pred_only={int(srd.get('overlap', {}).get('predicted_only_count', 0))} "
            f"gold_only={int(srd.get('overlap', {}).get('gold_only_count', 0))}"
        )
        print(
            f"CASE_DIAG {case} "
            f"progress_rows={int(aps.get('rows', 0))} "
            f"progress_mae01={float(aps.get('mae01', 0.0)):.4f} "
            f"progress_mae_ms={float(aps.get('mae_ms', 0.0)):.1f} "
            f"progress_bias01={float(aps.get('bias01', 0.0)):.4f} "
            f"progress_corr={float(aps.get('corr', 0.0)):.3f} "
            f"anchor_phone={float(aps.get('anchor_mean_phone_probability', 0.0)):.2f} "
            f"anchor_boundary={float(aps.get('anchor_mean_boundary_probability', 0.0)):.2f} "
            f"anchor_speech={float(aps.get('anchor_mean_speech_probability', 0.0)):.2f} "
            f"pause50={float(aps.get('micro_pause_50_count', 0.0)):.1f} "
            f"drift_abs_ms={float(aps.get('retrospective_drift_abs_ms', 0.0)):.1f} "
            f"drift_rate={float(aps.get('drift_mean_playrate', 1.0)):.3f} "
            f"would_advance={int(aps.get('would_advance_rows', 0))} "
            f"advance_reasons={json.dumps(aps.get('advance_reasons', {}), sort_keys=True, separators=(',', ':'))} "
            f"worst_phone_class={worst}"
        )
        if gcs:
            print(
                f"CASE_GAPS {case} "
                f"count={int(gcs.get('count', 0))} "
                f"bridged={int(gcs.get('bridged', 0))} "
                f"split={int(gcs.get('split', 0))} "
                f"mean_gap_ms={float(gcs.get('mean_gap_ms', 0.0)):.1f} "
                f"ambiguous={int(gcs.get('ambiguous_count', 0))} "
                f"ambiguous_bridged={int(gcs.get('ambiguous_bridged', 0))} "
                f"ambiguous_split={int(gcs.get('ambiguous_split', 0))} "
                f"decisions={json.dumps(gcs.get('decision_counts', {}), sort_keys=True, separators=(',', ':'))}"
            )

    print(
        f"CASES total={summary['total_cases']} graded={summary['graded_cases']} "
        f"ungraded={summary['ungraded_cases']} degenerate={summary['degenerate_cases']}"
    )
    print(
        f"GRADE center_ms={summary['phoneme_center_ms']:.3f} start_ms={summary['phoneme_start_ms']:.3f} "
        f"end_ms={summary['phoneme_end_ms']:.3f} order_fail_cases={summary['order_fail_cases']}"
    )
    print(
        f"GRADE_STATS "
        f"center_mean_ms={summary.get('phoneme_center_mean_ms', 0.0):.3f} "
        f"center_median_ms={summary.get('phoneme_center_median_ms', 0.0):.3f} "
        f"center_p90_ms={summary.get('phoneme_center_p90_ms', 0.0):.3f} "
        f"center_max_ms={summary.get('phoneme_center_max_ms', 0.0):.3f} "
        f"center_stddev_ms={summary.get('phoneme_center_stddev_ms', 0.0):.3f} "
        f"start_mean_ms={summary.get('phoneme_start_mean_ms', 0.0):.3f} "
        f"start_median_ms={summary.get('phoneme_start_median_ms', 0.0):.3f} "
        f"start_p90_ms={summary.get('phoneme_start_p90_ms', 0.0):.3f} "
        f"end_mean_ms={summary.get('phoneme_end_mean_ms', 0.0):.3f} "
        f"end_median_ms={summary.get('phoneme_end_median_ms', 0.0):.3f} "
        f"end_p90_ms={summary.get('phoneme_end_p90_ms', 0.0):.3f}"
    )
    print(
        f"REGIONS speech_start_ms={summary['speech_boundary_start_ms']:.3f} "
        f"speech_end_ms={summary['speech_boundary_end_ms']:.3f} "
        f"sentence_mismatch={summary['sentence_region_count_mismatch_cases']} "
        f"clause_mismatch={summary['clause_region_count_mismatch_cases']}"
    )
    print(
        f"REGIONS_INDEXED gold_count={summary.get('speech_region_gold_count', 0.0):.3f} "
        f"pred_count={summary.get('speech_region_predicted_count', 0.0):.3f} "
        f"count_delta={summary.get('speech_region_count_delta', 0.0):.3f} "
        f"start_ms={summary.get('speech_region_indexed_start_ms', 0.0):.3f} "
        f"end_ms={summary.get('speech_region_indexed_end_ms', 0.0):.3f} "
        f"overlap_ratio={summary.get('speech_region_indexed_overlap_ratio', 0.0):.3f}"
    )
    print(
        f"REGIONS_OVERLAP gold_count={summary.get('speech_region_gold_count', 0.0):.3f} "
        f"pred_count={summary.get('speech_region_predicted_count', 0.0):.3f} "
        f"overlap_matched={summary.get('speech_region_overlap_matched_count', 0.0):.3f} "
        f"nearest_matched={summary.get('speech_region_nearest_matched_count', 0.0):.3f} "
        f"pred_only={summary.get('speech_region_predicted_only_count', 0.0):.3f} "
        f"gold_only={summary.get('speech_region_gold_only_count', 0.0):.3f} "
        f"start_ms={summary.get('speech_region_overlap_start_ms', 0.0):.3f} "
        f"end_ms={summary.get('speech_region_overlap_end_ms', 0.0):.3f} "
        f"overlap_ratio={summary.get('speech_region_overlap_ratio', 0.0):.3f}"
    )
    print(
        f"GAPS count={summary.get('gap_candidate_count', 0)} "
        f"bridged={summary.get('gap_candidate_bridged', 0)} "
        f"split={summary.get('gap_candidate_split', 0)} "
        f"mean_gap_ms={summary.get('gap_candidate_mean_gap_ms', 0.0):.3f} "
        f"ambiguous={summary.get('gap_candidate_ambiguous_count', 0)} "
        f"ambiguous_bridged={summary.get('gap_candidate_ambiguous_bridged', 0)} "
        f"ambiguous_split={summary.get('gap_candidate_ambiguous_split', 0)} "
        f"ambiguous_mean_gap_ms={summary.get('gap_candidate_ambiguous_mean_gap_ms', 0.0):.3f}"
    )
    print(
        "GAP_DECISIONS "
        + json.dumps(summary.get("gap_decision_counts", {}), sort_keys=True, separators=(",", ":"))
    )
    print(
        f"WORDS assignment_rate={summary.get('word_assignment_rate', summary['word_f1']):.4f} detected_start_ms={summary['detected_word_onset_ms']:.3f} "
        f"intra_word_coverage={summary['intra_word_coverage_rate']:.4f} "
        f"intra_word_center_ms={summary['intra_word_center_ms']:.3f}"
    )
    print(
        f"WORD_STATS "
        f"detected_start_mean_ms={summary.get('detected_word_onset_mean_ms', 0.0):.3f} "
        f"detected_start_median_ms={summary.get('detected_word_onset_median_ms', 0.0):.3f} "
        f"detected_start_p90_ms={summary.get('detected_word_onset_p90_ms', 0.0):.3f} "
        f"detected_start_max_ms={summary.get('detected_word_onset_max_ms', 0.0):.3f} "
        f"detected_start_stddev_ms={summary.get('detected_word_onset_stddev_ms', 0.0):.3f} "
        f"case_median_median_ms={summary.get('detected_word_onset_event_median_median_ms', 0.0):.3f} "
        f"case_median_p90_ms={summary.get('detected_word_onset_event_median_p90_ms', 0.0):.3f} "
        f"duration_mean_ms={summary.get('word_duration_mean_ms', 0.0):.3f} "
        f"duration_median_ms={summary.get('word_duration_median_ms', 0.0):.3f} "
        f"duration_p90_ms={summary.get('word_duration_p90_ms', 0.0):.3f}"
    )
    print(
        f"INTRA_WORD_STATS "
        f"center_mean_ms={summary.get('intra_word_center_mean_ms', 0.0):.3f} "
        f"center_median_ms={summary.get('intra_word_center_median_ms', 0.0):.3f} "
        f"center_p90_ms={summary.get('intra_word_center_p90_ms', 0.0):.3f} "
        f"center_max_ms={summary.get('intra_word_center_max_ms', 0.0):.3f} "
        f"center_stddev_ms={summary.get('intra_word_center_stddev_ms', 0.0):.3f}"
    )
    print(
        f"DIRECT_ALIGNER match_rate={summary['direct_aligner_match_rate']:.4f} "
        f"center_ms={summary['direct_aligner_center_ms']:.3f}"
    )
    print(
        f"DIRECT_ALIGNER_STATS "
        f"center_mean_ms={summary.get('direct_aligner_center_mean_ms', 0.0):.3f} "
        f"center_median_ms={summary.get('direct_aligner_center_median_ms', 0.0):.3f} "
        f"center_p90_ms={summary.get('direct_aligner_center_p90_ms', 0.0):.3f} "
        f"center_max_ms={summary.get('direct_aligner_center_max_ms', 0.0):.3f} "
        f"center_stddev_ms={summary.get('direct_aligner_center_stddev_ms', 0.0):.3f}"
    )
    print(f"AUDIO_PROGRESS rows={summary['audio_progress_rows']}")
    print(
        f"AUDIO_PROGRESS_MFA rows={summary['audio_progress_mfa_rows']} "
        f"would_advance={summary.get('audio_progress_would_advance_rows', 0)} "
        f"mae01={summary['audio_progress_mfa_mae01']:.4f} "
        f"median01={summary['audio_progress_mfa_median01']:.4f} "
        f"mae_ms={summary['audio_progress_mfa_mae_ms']:.1f} "
        f"p90_ms={summary['audio_progress_mfa_p90_ms']:.1f} "
        f"bias01={summary['audio_progress_mfa_bias01']:.4f} "
        f"corr={summary['audio_progress_mfa_corr']:.3f} "
        f"component_mae01=time:{summary['audio_progress_time_prior_mae01']:.4f},"
        f"density:{summary['audio_progress_density_mae01']:.4f},"
        f"boundary:{summary['audio_progress_boundary_mae01']:.4f},"
        f"phone:{summary['audio_progress_phone_mae01']:.4f}"
    )
    print(
        "ADVANCE_REASONS "
        + json.dumps(summary.get("advance_reason_counts", {}), sort_keys=True, separators=(",", ":"))
    )
    print(
        f"TIMING_PIVOT micro_pause_50={summary.get('audio_progress_micro_pause_50_count', 0.0):.3f} "
        f"micro_pause_75={summary.get('audio_progress_micro_pause_75_count', 0.0):.3f} "
        f"micro_pause_120={summary.get('audio_progress_micro_pause_120_count', 0.0):.3f} "
        f"drift_abs01={summary.get('audio_progress_retrospective_drift_abs01', 0.0):.4f} "
        f"drift_abs_ms={summary.get('audio_progress_retrospective_drift_abs_ms', 0.0):.1f} "
        f"drift_bias01={summary.get('audio_progress_retrospective_drift_bias01', 0.0):.4f} "
        f"drift_conf={summary.get('audio_progress_retrospective_drift_confidence', 0.0):.3f} "
        f"drift_rate={summary.get('audio_progress_drift_mean_playrate', 1.0):.3f} "
        f"pll_rate={summary.get('audio_progress_filtered_pll_playrate', 1.0):.3f} "
        f"pll_tempo={summary.get('audio_progress_pll_tempo_playrate', 1.0):.3f} "
        f"pll_phase={summary.get('audio_progress_pll_phase_playrate', 1.0):.3f} "
        f"pll_conf={summary.get('audio_progress_filtered_pll_confidence', 0.0):.3f} "
        f"warp_rate={summary.get('audio_progress_timing_warp_rate', 1.0):.3f} "
        f"warp_conf={summary.get('audio_progress_timing_warp_confidence', 0.0):.3f} "
        f"warp_err_ms={summary.get('audio_progress_timing_warp_error_ms', 0.0):.1f}"
    )
    phone_chunks = []
    for klass, stats in sorted(summary.get('phone_class_errors', {}).items()):
        phone_chunks.append(
            f"{klass}:n={int(stats.get('n', 0))},matched={int(stats.get('matched', 0))},"
            f"unmapped={int(stats.get('unmapped', 0))},center_ms={float(stats.get('center_ms', 0.0)):.1f},"
            f"start_ms={float(stats.get('start_ms', 0.0)):.1f},end_ms={float(stats.get('end_ms', 0.0)):.1f}"
        )
    print("PHONE_CLASS_ERRORS " + " | ".join(phone_chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
