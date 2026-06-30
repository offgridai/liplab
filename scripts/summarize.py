import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1] / 'outputs' / 'runs' / 'latest'
grades = []
for grade in sorted(root.glob('*/grade.json')):
    data = json.loads(grade.read_text())
    pause = data.get('pause_alignment', {})
    word_onset = data.get('word_onset_alignment', {})
    intra_word = data.get('intra_word_alignment', {})
    grades.append((grade.parent.name, data))
    print(
        f"{grade.parent.name}: matched={data['matched_count']}/{data['reference_count']} "
        f"mean_ms={data['mean_abs_center_error_ms']} p90_ms={data['p90_abs_center_error_ms']} "
        f"start_ms={data.get('mean_abs_start_error_ms', 0.0)} end_ms={data.get('mean_abs_end_error_ms', 0.0)} "
        f"sentence_start_ms={pause.get('mean_abs_sentence_region_start_error_ms', 0.0)} "
        f"word_onset_ms={word_onset.get('mean_abs_start_error_ms', 0.0)} "
        f"intra_word_ms={intra_word.get('mean_abs_center_error_ms', 0.0)}"
    )

if grades:
    total_matched = sum(data['matched_count'] for _, data in grades)
    total_reference = sum(data['reference_count'] for _, data in grades)
    mean_center = sum(data['mean_abs_center_error_ms'] for _, data in grades) / len(grades)
    mean_start = sum(data.get('mean_abs_start_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_end = sum(data.get('mean_abs_end_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_duration = sum(data.get('mean_abs_duration_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_sentence_start = sum(data.get('pause_alignment', {}).get('mean_abs_sentence_region_start_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_sentence_end = sum(data.get('pause_alignment', {}).get('mean_abs_sentence_region_end_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_clause_start = sum(data.get('pause_alignment', {}).get('mean_abs_clause_region_start_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_clause_end = sum(data.get('pause_alignment', {}).get('mean_abs_clause_region_end_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_word_onset = sum(data.get('word_onset_alignment', {}).get('mean_abs_start_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_intra_word = sum(data.get('intra_word_alignment', {}).get('mean_abs_center_error_ms', 0.0) for _, data in grades) / len(grades)
    order_fail_cases = sum(1 for _, data in grades if data.get('order_violations', 0) > 0)
    print(
        f"SUMMARY cases={len(grades)} matched={total_matched}/{total_reference} "
        f"match_rate={total_matched / max(total_reference, 1):.4f} "
        f"mean_center_ms={mean_center:.3f} mean_start_ms={mean_start:.3f} "
        f"mean_end_ms={mean_end:.3f} mean_duration_ms={mean_duration:.3f} "
        f"mean_sentence_start_ms={mean_sentence_start:.3f} mean_sentence_end_ms={mean_sentence_end:.3f} "
        f"mean_clause_start_ms={mean_clause_start:.3f} mean_clause_end_ms={mean_clause_end:.3f} "
        f"mean_word_onset_ms={mean_word_onset:.3f} mean_intra_word_ms={mean_intra_word:.3f} "
        f"order_fail_cases={order_fail_cases}"
    )
