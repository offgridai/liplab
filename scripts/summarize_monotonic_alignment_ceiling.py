import argparse
import csv
import json
import statistics
from pathlib import Path


MATCH_TOLERANCE_MS = 180.0
DEFAULT_DECISION_LATENCY_MS = 40.0
PREROLL_POINTS_MS = (100.0, 150.0, 200.0, 250.0, 300.0, 350.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def lcs_alignment(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    previous = [0] * (len(right) + 1)
    rows = [previous]
    for left_value in left:
        current = [0] * (len(right) + 1)
        for column, right_value in enumerate(right, 1):
            if left_value == right_value:
                current[column] = previous[column - 1] + 1
            else:
                current[column] = max(previous[column], current[column - 1])
        rows.append(current)
        previous = current

    alignment: list[tuple[int, int]] = []
    left_index = len(left)
    right_index = len(right)
    while left_index and right_index:
        if left[left_index - 1] == right[right_index - 1]:
            alignment.append((left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
        elif rows[left_index - 1][right_index] >= rows[left_index][right_index - 1]:
            left_index -= 1
        else:
            right_index -= 1
    alignment.reverse()
    return alignment


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the identity and streaming-latency ceiling of a monotonic CMU scheduler."
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs/latest"))
    parser.add_argument("--decision-latency-ms", type=float, default=DEFAULT_DECISION_LATENCY_MS)
    args = parser.parse_args()

    case_count = 0
    gold_count = 0
    planned_count = 0
    aligned_count = 0
    durations_ms: list[float] = []
    cases_without_full_recall: list[str] = []
    oracle_predictions: list[dict[str, object]] = []

    for case_dir in sorted(path for path in args.run_root.iterdir() if path.is_dir()):
        gold_path = case_dir / "gold_visible_visemes.csv"
        planned_path = case_dir / "planned.csv"
        if not gold_path.exists() or not planned_path.exists():
            continue
        gold = read_csv(gold_path)
        planned = [row for row in read_csv(planned_path) if row.get("renderable") == "1"]
        gold_poses = [row["pose"] for row in gold]
        planned_poses = [row["pose"] for row in planned]
        alignment = lcs_alignment(gold_poses, planned_poses)
        if len(alignment) != len(gold):
            cases_without_full_recall.append(case_dir.name)
        durations_ms.extend(
            1000.0 * (float(row["end"]) - float(row["start"])) for row in gold
        )
        case_count += 1
        gold_count += len(gold)
        planned_count += len(planned)
        aligned_count += len(alignment)

        gold_by_key = {
            (row["word_index"], row["source_phone_word_index"], row["pose"]): row
            for row in gold
        }
        centers: list[float | None] = []
        durations: list[float | None] = []
        for row in planned:
            match = gold_by_key.get((row["word_index"], row["source_phone_index"], row["pose"]))
            centers.append(
                None if match is None else 0.5 * (float(match["start"]) + float(match["end"]))
            )
            durations.append(
                None if match is None else float(match["end"]) - float(match["start"])
            )
        for index, center in enumerate(centers):
            if center is not None:
                continue
            before = next((i for i in range(index - 1, -1, -1) if centers[i] is not None), None)
            after = next((i for i in range(index + 1, len(centers)) if centers[i] is not None), None)
            if before is not None and after is not None:
                fraction = (index - before) / (after - before)
                centers[index] = centers[before] + fraction * (centers[after] - centers[before])
            elif before is not None:
                centers[index] = centers[before] + 0.030 * (index - before)
            elif after is not None:
                centers[index] = max(0.0, centers[after] - 0.030 * (after - index))
            else:
                centers[index] = 0.030 * index
            durations[index] = 0.020
        last_center = 0.0
        for row, center, duration in zip(planned, centers, durations):
            last_center = max(last_center, float(center))
            oracle_predictions.append(
                {
                    "case_id": case_dir.name,
                    "event_index": int(row["index"]),
                    "pose": row["pose"],
                    "center_sec": f"{last_center:.6f}",
                    "duration_sec": f"{float(duration):.6f}",
                    "confidence": "1.000000",
                    "fallback_used": 0,
                }
            )

    if not case_count or not gold_count:
        raise RuntimeError(f"no complete cases found under {args.run_root}")

    latency = {}
    for preroll_ms in PREROLL_POINTS_MS:
        schedulable = sum(
            duration / 2.0 + args.decision_latency_ms <= preroll_ms
            for duration in durations_ms
        )
        latency[str(int(preroll_ms))] = {
            "schedulable_count": schedulable,
            "schedulable_rate": schedulable / gold_count,
        }

    report = {
        "schema_version": 1,
        "experiment": "monotonic_transcript_lattice_ceiling",
        "case_count": case_count,
        "gold_visible_count": gold_count,
        "planned_renderable_count": planned_count,
        "ordered_pose_lcs_count": aligned_count,
        "ordered_gold_recall_ceiling": aligned_count / gold_count,
        "ordered_plan_precision": aligned_count / planned_count,
        "additional_transcript_pose_count": planned_count - aligned_count,
        "cases_without_full_ordered_gold_recall": cases_without_full_recall,
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
        "decision_latency_ms": args.decision_latency_ms,
        "phone_duration_ms": {
            "median": statistics.median(durations_ms),
            "p90": percentile(durations_ms, 0.90),
            "p99": percentile(durations_ms, 0.99),
            "maximum": max(durations_ms),
        },
        "causal_center_schedulability_by_preroll_ms": latency,
        "target_0_9_identity_feasible": aligned_count / gold_count >= 0.9,
        "notes": [
            "This is a sequence/latency ceiling, not a trained-model score.",
            "LCS measures whether the transcript plan contains every graded pose in order.",
            "Schedulability assumes a transition is recognized at phone end plus decision latency.",
            "The earlier timing oracle retained deterministic commits and is therefore a lower ceiling.",
        ],
    }
    output_path = args.run_root / "monotonic_alignment_ceiling.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    oracle_path = args.run_root / "monotonic_oracle_track.csv"
    with oracle_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oracle_predictions[0]))
        writer.writeheader()
        writer.writerows(oracle_predictions)
    print(
        f"Monotonic identity ceiling: {aligned_count}/{gold_count} "
        f"({aligned_count / gold_count:.3f}); plan precision={aligned_count / planned_count:.3f}"
    )
    for preroll, row in latency.items():
        print(f"preroll {preroll} ms: causal-center schedulable={row['schedulable_rate']:.3f}")
    print(f"Wrote {output_path.resolve()}")
    print(f"Wrote {oracle_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
