import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1] / 'outputs' / 'runs' / 'latest'
for grade in sorted(root.glob('*/grade.json')):
    data = json.loads(grade.read_text())
    print(f"{grade.parent.name}: matched={data['matched_count']}/{data['reference_count']} mean_ms={data['mean_abs_center_error_ms']}")
