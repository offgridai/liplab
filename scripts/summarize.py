import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1] / 'outputs' / 'runs' / 'latest'
grades = []
for grade in sorted(root.glob('*/grade.json')):
    data = json.loads(grade.read_text())
    grades.append((grade.parent.name, data))
    print(
        f"{grade.parent.name}: matched={data['matched_count']}/{data['reference_count']} "
        f"mean_ms={data['mean_abs_center_error_ms']} p90_ms={data['p90_abs_center_error_ms']} "
        f"start_ms={data.get('mean_abs_start_error_ms', 0.0)} end_ms={data.get('mean_abs_end_error_ms', 0.0)}"
    )

if grades:
    total_matched = sum(data['matched_count'] for _, data in grades)
    total_reference = sum(data['reference_count'] for _, data in grades)
    mean_center = sum(data['mean_abs_center_error_ms'] for _, data in grades) / len(grades)
    mean_start = sum(data.get('mean_abs_start_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_end = sum(data.get('mean_abs_end_error_ms', 0.0) for _, data in grades) / len(grades)
    mean_duration = sum(data.get('mean_abs_duration_error_ms', 0.0) for _, data in grades) / len(grades)
    order_fail_cases = sum(1 for _, data in grades if data.get('order_violations', 0) > 0)
    print(
        f"SUMMARY cases={len(grades)} matched={total_matched}/{total_reference} "
        f"match_rate={total_matched / max(total_reference, 1):.4f} "
        f"mean_center_ms={mean_center:.3f} mean_start_ms={mean_start:.3f} "
        f"mean_end_ms={mean_end:.3f} mean_duration_ms={mean_duration:.3f} "
        f"order_fail_cases={order_fail_cases}"
    )
