import pathlib

from grade_summary import compute_summary, load_case_grades

root = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "runs" / "latest"
gold_root = pathlib.Path(__file__).resolve().parents[1] / "inputs" / "gold"
rows, graded, ungraded = load_case_grades(root, gold_root)
for row in rows:
    print(
        f"{row['case']}: gold={'yes' if row['gold_available'] else 'no'} "
        f"matched={row['matched_count']}/{row['reference_count']} "
        f"mean_ms={row['mean_abs_center_error_ms']} p90_ms={row['p90_abs_center_error_ms']} "
        f"start_ms={row['mean_abs_start_error_ms']} end_ms={row['mean_abs_end_error_ms']} "
        f"sentence_start_ms={row['pause_alignment'].get('mean_abs_sentence_region_start_error_ms', 0.0)} "
        f"word_onset_ms={row['word_onset_alignment'].get('mean_abs_start_error_ms', 0.0)} "
        f"intra_word_ms={row['intra_word_alignment'].get('mean_abs_center_error_ms', 0.0)} "
        f"speech_f1={row['layers']['speech']['f1']:.3f} "
        f"word_f1={row['layers']['words']['f1']:.3f} "
        f"phoneme_cov={row['layers']['phonemes']['coverage_rate']:.3f}"
    )

if rows:
    summary = compute_summary(rows, graded, ungraded)
    print(
        f"SUMMARY cases={summary['cases']} graded_cases={summary['graded_cases']} "
        f"ungraded_cases={summary['ungraded_cases']} "
        f"matched={summary['matched_count']}/{summary['reference_count']} "
        f"match_rate={summary['match_rate']:.4f} "
        f"mean_center_ms={summary['mean_center_ms']:.3f} mean_start_ms={summary['mean_start_ms']:.3f} "
        f"mean_end_ms={summary['mean_end_ms']:.3f} mean_duration_ms={summary['mean_duration_ms']:.3f} "
        f"mean_sentence_start_ms={summary['mean_sentence_start_ms']:.3f} "
        f"mean_sentence_end_ms={summary['mean_sentence_end_ms']:.3f} "
        f"mean_clause_start_ms={summary['mean_clause_start_ms']:.3f} "
        f"mean_clause_end_ms={summary['mean_clause_end_ms']:.3f} "
        f"mean_word_onset_ms={summary['mean_word_onset_ms']:.3f} "
        f"mean_intra_word_ms={summary['mean_intra_word_ms']:.3f} "
        f"order_fail_cases={summary['order_fail_cases']}"
    )
    print(
        f"HIERARCHY speech_precision={summary['speech_precision']:.4f} "
        f"speech_recall={summary['speech_recall']:.4f} speech_f1={summary['speech_f1']:.4f} "
        f"speech_boundary_start_ms={summary['speech_boundary_start_ms']:.3f} "
        f"speech_boundary_end_ms={summary['speech_boundary_end_ms']:.3f} "
        f"word_precision={summary['word_precision']:.4f} "
        f"word_recall={summary['word_recall']:.4f} word_f1={summary['word_f1']:.4f} "
        f"word_start_ms={summary['word_start_ms']:.3f} word_end_ms={summary['word_end_ms']:.3f} "
        f"word_duration_ms={summary['word_duration_ms']:.3f} "
        f"word_head_start_ms={summary['word_head_start_ms']:.3f} "
        f"phoneme_coverage={summary['phoneme_coverage_rate']:.4f} "
        f"phoneme_center_ms={summary['phoneme_center_ms']:.3f} "
        f"phoneme_start_ms={summary['phoneme_start_ms']:.3f} "
        f"phoneme_end_ms={summary['phoneme_end_ms']:.3f} "
        f"intra_word_coverage={summary['intra_word_coverage_rate']:.4f} "
        f"intra_word_center_ms={summary['intra_word_center_ms']:.3f}"
    )
