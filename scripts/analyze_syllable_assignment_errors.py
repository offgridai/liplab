#!/usr/bin/env python3
"""Report signed monotonic syllable-assignment errors and local oracle ceilings."""

import collections
import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "latest"
GOLD = ROOT / "inputs" / "gold"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    distribution = collections.Counter()
    adjacent_context = collections.Counter()
    adjacent_phone_pairs = collections.Counter()
    assignment_count = target_count = timing_matched = 0
    for case_dir in sorted(path for path in RUN.iterdir() if path.is_dir()):
        grade_path = case_dir / "grade.json"
        positions_path = case_dir / "streaming_syllable_positions.csv"
        gold_path = GOLD / case_dir.name / "phones.csv"
        if not grade_path.exists() or not positions_path.exists() or not gold_path.exists():
            continue
        grade = json.loads(grade_path.read_text(encoding="utf-8"))
        if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", False):
            continue
        gold = rows(gold_path)
        syllable_rows = rows(case_dir / "planned_syllables.csv")
        targets = []
        for syllable in syllable_rows:
            nucleus = int(syllable["nucleus_phone_index"])
            targets.append(
                (float(gold[nucleus]["start"]), float(gold[nucleus]["end"]))
                if 0 <= nucleus < len(gold) else None
            )
        target_count += sum(target is not None for target in targets)
        used = [False] * len(targets)
        assignments = rows(positions_path)
        assignment_count += len(assignments)
        for assignment in assignments:
            center = float(assignment["audio_center"])
            actual = None
            best_error = 0.100001
            for index, target in enumerate(targets):
                if used[index] or target is None:
                    continue
                start, end = target
                error = max(start - center, center - end, 0.0)
                if error < best_error:
                    actual = index
                    best_error = error
            if actual is None:
                distribution["unmatched_timing"] += 1
                continue
            used[actual] = True
            timing_matched += 1
            delta = int(assignment["syllable_index"]) - actual
            distribution[delta] += 1
            if abs(delta) == 1:
                predicted = int(assignment["syllable_index"])
                actual_row = syllable_rows[actual]
                predicted_row = syllable_rows[predicted]
                if actual_row["speech_region_index"] != predicted_row["speech_region_index"]:
                    adjacent_context["cross_region"] += 1
                elif actual_row["word_index"] != predicted_row["word_index"]:
                    adjacent_context["cross_word"] += 1
                else:
                    adjacent_context["same_word"] += 1
                adjacent_phone_pairs[
                    f"{actual_row['nucleus_base_phone']}->{predicted_row['nucleus_base_phone']}"
                ] += 1

    exact = distribution[0]
    within_one = sum(distribution[delta] for delta in (-1, 0, 1))
    fixed_shift = {}
    for adjustment in (-1, 0, 1):
        corrected = sum(count for delta, count in distribution.items()
                        if isinstance(delta, int) and delta + adjustment == 0)
        fixed_shift[str(adjustment)] = {
            "exact_precision": corrected / assignment_count,
            "exact_recall": corrected / target_count,
        }
    print(json.dumps({
        "assignment_count": assignment_count,
        "target_count": target_count,
        "timing_matched_count": timing_matched,
        "signed_error_counts": {str(key): value for key, value in sorted(
            distribution.items(), key=lambda item: str(item[0]))},
        "actual_before_guess_adjacent": distribution[1],
        "actual_after_guess_adjacent": distribution[-1],
        "adjacent_error_balance": (
            distribution[1] / max(distribution[1] + distribution[-1], 1)
        ),
        "adjacent_context": adjacent_context,
        "adjacent_nucleus_pairs": adjacent_phone_pairs.most_common(20),
        "fixed_global_adjustments": fixed_shift,
        "three_choice_oracle": {
            "exact_precision": within_one / assignment_count,
            "exact_recall": within_one / target_count,
            "exact_count": within_one,
        },
        "current": {
            "exact_precision": exact / assignment_count,
            "exact_recall": exact / target_count,
            "exact_count": exact,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
