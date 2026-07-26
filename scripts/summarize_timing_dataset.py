import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"
SPLIT_PATH = ROOT / "inputs" / "speech_library" / "timing_split_v1.json"


def main() -> int:
    split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_by_case = {row["case_id"]: row["split"] for row in split_payload["cases"]}
    aggregates: dict[str, dict[str, float | int]] = {}
    missing: list[str] = []
    for case_id, split in split_by_case.items():
        csv_path = RUN_ROOT / case_id / "timing_training_rows.csv"
        if not csv_path.exists():
            missing.append(case_id)
            continue
        aggregate = aggregates.setdefault(
            split,
            {
                "case_count": 0,
                "row_count": 0,
                "actionable_count": 0,
                "clamped_count": 0,
                "absolute_target_sum_ms": 0.0,
                "absolute_clamped_target_sum_ms": 0.0,
            },
        )
        aggregate["case_count"] += 1
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_target = float(row["target_shift_ms"])
                clamped_target = float(row["target_shift_clamped_100_ms"])
                aggregate["row_count"] += 1
                aggregate["actionable_count"] += int(row["target_actionable"])
                aggregate["clamped_count"] += int(abs(raw_target) > 100.0001)
                aggregate["absolute_target_sum_ms"] += abs(raw_target)
                aggregate["absolute_clamped_target_sum_ms"] += abs(clamped_target)

    output_splits: dict[str, object] = {}
    total_rows = 0
    for split, aggregate in sorted(aggregates.items()):
        row_count = int(aggregate["row_count"])
        total_rows += row_count
        output_splits[split] = {
            "case_count": int(aggregate["case_count"]),
            "row_count": row_count,
            "actionable_count": int(aggregate["actionable_count"]),
            "actionable_rate": int(aggregate["actionable_count"]) / row_count if row_count else 0.0,
            "clamped_count": int(aggregate["clamped_count"]),
            "clamped_rate": int(aggregate["clamped_count"]) / row_count if row_count else 0.0,
            "mean_abs_target_ms": (
                float(aggregate["absolute_target_sum_ms"]) / row_count if row_count else 0.0
            ),
            "mean_abs_clamped_target_ms": (
                float(aggregate["absolute_clamped_target_sum_ms"]) / row_count
                if row_count
                else 0.0
            ),
        }

    payload = {
        "schema_version": 2,
        "sample_point": "deterministic proposal immediately before immutable commit",
        "source_split": SPLIT_PATH.relative_to(ROOT).as_posix(),
        "case_count": sum(int(row["case_count"]) for row in output_splits.values()),
        "row_count": total_rows,
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "splits": output_splits,
    }
    output_path = RUN_ROOT / "timing_dataset_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for split, row in output_splits.items():
        print(
            f"{split}: cases={row['case_count']} rows={row['row_count']} "
            f"actionable={row['actionable_rate']:.3f} clamped={row['clamped_rate']:.3f} "
            f"target_mae_ms={row['mean_abs_target_ms']:.1f}"
        )
    if missing:
        raise SystemExit(f"missing timing rows for {len(missing)} cases")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
