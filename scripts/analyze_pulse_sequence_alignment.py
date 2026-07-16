import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
OUT_ROOT = Path("outputs/analysis")
TOLERANCE_SEC = 0.100
INTER_WORD_SEC = 0.020


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@dataclass
class ExpectedPulse:
    index: int
    word: int
    local: int
    prior: float
    stress: int
    region: int


@dataclass
class GoldPulse:
    center: float
    expected: int


@dataclass
class ObservedPulse:
    center: float
    score: float
    target: int = -1
    region: int = -1


def expected_pulses(case_dir: Path) -> list[ExpectedPulse]:
    phone_rows = rows(case_dir / "expected_phones.csv")
    centers: dict[int, float] = {}
    cursor = 0.0
    previous_word = None
    for phone in phone_rows:
        word = int(phone["word_index"])
        if previous_word is not None and word != previous_word:
            cursor += INTER_WORD_SEC
        duration = max(float(phone["weight_seconds"]), 0.025)
        centers[int(phone["phone_index"])] = cursor + duration * 0.5
        cursor += duration
        previous_word = word

    result = []
    for syllable in rows(case_dir / "planned_syllables.csv"):
        nucleus = syllable["nucleus_phone"]
        result.append(
            ExpectedPulse(
                index=int(syllable["syllable_index"]),
                word=int(syllable["word_index"]),
                local=int(syllable["word_syllable_index"]),
                prior=centers[int(syllable["nucleus_phone_index"])],
                stress=int(nucleus[-1]) if nucleus[-1:].isdigit() else 0,
                region=int(syllable["speech_region_index"]),
            )
        )
    return result


def gold_pulses(case_dir: Path, expected: list[ExpectedPulse]) -> list[GoldPulse]:
    expected_by_word_local = {(pulse.word, pulse.local): pulse.index for pulse in expected}
    local_by_word: dict[int, int] = {}
    result = []
    for phone in rows(GOLD_ROOT / case_dir.name / "phones.csv"):
        label = phone["phone"]
        if not label or not label[-1].isdigit():
            continue
        word = int(phone["word_index"])
        local = local_by_word.get(word, 0)
        local_by_word[word] = local + 1
        expected_index = expected_by_word_local.get((word, local), -1)
        if expected_index >= 0:
            result.append(
                GoldPulse(
                    center=(float(phone["start"]) + float(phone["end"])) * 0.5,
                    expected=expected_index,
                )
            )
    return result


def observed_pulses(case_dir: Path, gold: list[GoldPulse]) -> list[ObservedPulse]:
    observed = [
        ObservedPulse(float(row["center"]), float(row["score"]))
        for row in rows(case_dir / "streaming_evidence_observations.csv")
        if row["type"] == "pulse"
    ]
    speech_regions = rows(case_dir / "speech_regions.csv")
    for pulse in observed:
        for region in speech_regions:
            if float(region["start"]) - 0.001 <= pulse.center <= float(region["end"]) + 0.001:
                pulse.region = int(region["index"])
                break
    pairs = sorted(
        (
            (abs(pulse.center - target.center), pulse_index, target_index)
            for pulse_index, pulse in enumerate(observed)
            for target_index, target in enumerate(gold)
            if abs(pulse.center - target.center) <= TOLERANCE_SEC
        ),
        key=lambda item: item[0],
    )
    used_observed: set[int] = set()
    used_gold: set[int] = set()
    for _, pulse_index, target_index in pairs:
        if pulse_index in used_observed or target_index in used_gold:
            continue
        observed[pulse_index].target = gold[target_index].expected
        used_observed.add(pulse_index)
        used_gold.add(target_index)
    return observed


def normalized(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0] * len(values)
    span = max(values[-1] - values[0], 1e-6)
    return [(value - values[0]) / span for value in values]


def ordinal_assign(expected: list[ExpectedPulse], observed: list[ObservedPulse]) -> list[int]:
    if not expected or not observed:
        return []
    if len(observed) == 1:
        return [0]
    return [
        round(index * (len(expected) - 1) / (len(observed) - 1))
        for index in range(len(observed))
    ]


def nearest_progress_assign(
    expected: list[ExpectedPulse], observed: list[ObservedPulse]
) -> list[int]:
    expected_progress = normalized([pulse.prior for pulse in expected])
    observed_progress = normalized([pulse.center for pulse in observed])
    result = []
    minimum = 0
    for progress in observed_progress:
        best = min(
            range(minimum, len(expected)),
            key=lambda index: abs(expected_progress[index] - progress),
        )
        result.append(best)
        minimum = min(best + 1, len(expected) - 1)
    return result


def dp_assign(
    expected: list[ExpectedPulse],
    observed: list[ObservedPulse],
    skip_expected: float,
    skip_observed: float,
    progress_weight: float,
    interval_weight: float,
    stress_weight: float,
) -> list[int]:
    n, m = len(expected), len(observed)
    if not n or not m:
        return []
    expected_progress = normalized([pulse.prior for pulse in expected])
    observed_progress = normalized([pulse.center for pulse in observed])
    # Global prominence is deliberately weak: detector scores are locally useful
    # but not calibrated as absolute lexical-stress probabilities.
    scores = [pulse.score for pulse in observed]
    low, high = min(scores), max(scores)
    prominence = [(score - low) / max(high - low, 1e-6) for score in scores]

    neg = -1e30
    table = [[neg] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    table[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            current = table[i][j]
            if current <= neg / 2:
                continue
            if i < n and current - skip_expected > table[i + 1][j]:
                table[i + 1][j] = current - skip_expected
                back[i + 1][j] = (i, j, "expected")
            if j < m and current - skip_observed > table[i][j + 1]:
                table[i][j + 1] = current - skip_observed
                back[i][j + 1] = (i, j, "observed")
            if i < n and j < m:
                progress_error = abs(expected_progress[i] - observed_progress[j])
                local = 1.0 - progress_weight * progress_error
                expected_prominence = 1.0 if expected[i].stress == 1 else 0.55 if expected[i].stress == 2 else 0.25
                local -= stress_weight * abs(expected_prominence - prominence[j])
                if i > 0 and j > 0:
                    expected_gap = expected_progress[i] - expected_progress[i - 1]
                    observed_gap = observed_progress[j] - observed_progress[j - 1]
                    local -= interval_weight * abs(expected_gap - observed_gap)
                if current + local > table[i + 1][j + 1]:
                    table[i + 1][j + 1] = current + local
                    back[i + 1][j + 1] = (i, j, "match")

    assignments = [-1] * m
    i, j = n, m
    while i or j:
        previous = back[i][j]
        if previous is None:
            break
        pi, pj, action = previous
        if action == "match":
            assignments[pj] = pi
        i, j = pi, pj
    return assignments


def region_reset_assign(expected, observed, local_assigner) -> list[int]:
    result = [-1] * len(observed)
    regions = sorted({pulse.region for pulse in observed if pulse.region >= 0})
    for region in regions:
        expected_indices = [index for index, pulse in enumerate(expected) if pulse.region == region]
        observed_indices = [index for index, pulse in enumerate(observed) if pulse.region == region]
        if not expected_indices or not observed_indices:
            continue
        local_expected = [expected[index] for index in expected_indices]
        local_observed = [observed[index] for index in observed_indices]
        local = local_assigner(local_expected, local_observed)
        for observed_index, prediction in zip(observed_indices, local):
            if prediction >= 0:
                result[observed_index] = expected_indices[prediction]
    return result


def count_partition_assign(expected, observed, local_assigner, proportional: bool) -> list[int]:
    result = [-1] * len(observed)
    region_ids = sorted({pulse.region for pulse in observed if pulse.region >= 0})
    total_observed = sum(pulse.region >= 0 for pulse in observed)
    observed_so_far = 0
    expected_begin = 0
    for region_offset, region in enumerate(region_ids):
        observed_indices = [index for index, pulse in enumerate(observed) if pulse.region == region]
        observed_so_far += len(observed_indices)
        if region_offset + 1 == len(region_ids):
            expected_end = len(expected)
        elif proportional:
            expected_end = round(observed_so_far * len(expected) / max(total_observed, 1))
        else:
            expected_end = min(expected_begin + len(observed_indices), len(expected))
        expected_end = max(expected_end, expected_begin)
        expected_indices = list(range(expected_begin, expected_end))
        expected_begin = expected_end
        if not expected_indices or not observed_indices:
            continue
        local = local_assigner(
            [expected[index] for index in expected_indices],
            [observed[index] for index in observed_indices],
        )
        for observed_index, prediction in zip(observed_indices, local):
            if prediction >= 0:
                result[observed_index] = expected_indices[prediction]
    return result


def score(cases: list[tuple[list[ExpectedPulse], list[ObservedPulse]]], assigner) -> tuple[float, float, float, int]:
    assigned = exact = within = 0
    targets: set[tuple[int, int]] = set()
    exact_targets: set[tuple[int, int]] = set()
    for case_index, (expected, observed) in enumerate(cases):
        assignments = assigner(expected, observed)
        for pulse in observed:
            if pulse.target >= 0:
                targets.add((case_index, pulse.target))
        for pulse, prediction in zip(observed, assignments):
            if prediction < 0:
                continue
            assigned += 1
            if pulse.target == prediction:
                exact += 1
                exact_targets.add((case_index, pulse.target))
            if pulse.target >= 0 and abs(pulse.target - prediction) <= 1:
                within += 1
    return (
        exact / max(assigned, 1),
        len(exact_targets) / max(len(targets), 1),
        within / max(assigned, 1),
        assigned,
    )


cases = []
case_names: list[str] = []
case_gold: list[list[GoldPulse]] = []
for case_dir in sorted(RUN_ROOT.iterdir()):
    if not case_dir.is_dir() or not (GOLD_ROOT / case_dir.name / "phones.csv").exists():
        continue
    grade = json.loads((case_dir / "grade.json").read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    expected = expected_pulses(case_dir)
    gold = gold_pulses(case_dir, expected)
    observed = observed_pulses(case_dir, gold)
    if expected and observed:
        cases.append((expected, observed))
        case_names.append(case_dir.name)
        case_gold.append(gold)


def source_group(case_name: str) -> str:
    match = re.search(r"Alfie_Line_([A-F0-9]+)", case_name)
    return match.group(1) if match else case_name


evaluation_indices = [
    index
    for index, name in enumerate(case_names)
    if int(hashlib.sha1(source_group(name).encode()).hexdigest()[:8], 16) % 5 == 0
]
training_indices = [index for index in range(len(cases)) if index not in set(evaluation_indices)]
training_cases = [cases[index] for index in training_indices]
evaluation_cases = [cases[index] for index in evaluation_indices]


def show(name: str, result: tuple[float, float, float, int]) -> None:
    precision, recall, within, assigned = result
    print(
        f"{name:24s} precision={precision:.3f} recall={recall:.3f} "
        f"within1={within:.3f} assigned={assigned}"
    )


print(
    f"qualified_cases={len(cases)} training={len(training_cases)} "
    f"held_out={len(evaluation_cases)}"
)
show("ordinal", score(cases, ordinal_assign))
show("nearest_progress", score(cases, nearest_progress_assign))
show(
    "region_ordinal",
    score(cases, lambda expected, observed: region_reset_assign(expected, observed, ordinal_assign)),
)
show(
    "count_partition",
    score(
        cases,
        lambda expected, observed: count_partition_assign(
            expected, observed, ordinal_assign, False
        ),
    ),
)
show(
    "proportional_partition",
    score(
        cases,
        lambda expected, observed: count_partition_assign(
            expected, observed, ordinal_assign, True
        ),
    ),
)

best = None
for skip_expected in (0.25, 0.50, 0.75, 1.00):
    for skip_observed in (0.25, 0.50, 0.75, 1.00):
        for progress_weight in (0.5, 1.0, 2.0, 3.0):
            for interval_weight in (0.0, 0.5, 1.0, 2.0):
                for stress_weight in (0.0, 0.15, 0.30):
                    parameters = (
                        skip_expected,
                        skip_observed,
                        progress_weight,
                        interval_weight,
                        stress_weight,
                    )
                    result = score(
                        training_cases,
                        lambda expected, observed, p=parameters: dp_assign(
                            expected, observed, *p
                        ),
                    )
                    objective = result[0] + result[1] + 0.25 * result[2]
                    if best is None or objective > best[0]:
                        best = (objective, parameters, result)

assert best is not None
print(f"best_parameters={best[1]}")
show(
    "pulse_sequence_dp_train",
    best[2],
)
held_out_result = score(
    evaluation_cases,
    lambda expected, observed, p=best[1]: dp_assign(expected, observed, *p),
)
full_result = score(
    cases,
    lambda expected, observed, p=best[1]: dp_assign(expected, observed, *p),
)
show("pulse_sequence_dp_test", held_out_result)
show("pulse_sequence_dp_full", full_result)
show(
    "region_pulse_dp",
    score(
        cases,
        lambda expected, observed, p=best[1]: region_reset_assign(
            expected,
            observed,
            lambda local_expected, local_observed: dp_assign(
                local_expected, local_observed, *p
            ),
        ),
    ),
)
show(
    "proportional_pulse_dp",
    score(
        cases,
        lambda expected, observed, p=best[1]: count_partition_assign(
            expected,
            observed,
            lambda local_expected, local_observed: dp_assign(
                local_expected, local_observed, *p
            ),
            True,
        ),
    ),
)

print("-- pulse confidence filtering --")
for threshold in (0.80, 0.90, 1.00, 1.025, 1.05):
    filtered_cases = [
        (expected, [pulse for pulse in observed if pulse.score >= threshold])
        for expected, observed in cases
    ]
    filtered_cases = [case for case in filtered_cases if case[1]]
    show(
        f"dp_score>={threshold:.3f}",
        score(
            filtered_cases,
            lambda expected, observed, p=best[1]: dp_assign(expected, observed, *p),
        ),
    )


OUT_ROOT.mkdir(parents=True, exist_ok=True)
triplet_rows = []
normalized_errors = []
interval_errors = []
affine_center_errors_ms = []
affine_interval_errors_ms = []
runtime_matched = 0
runtime_observations = 0
runtime_targets = 0
for case_index, ((expected, observed), gold) in enumerate(zip(cases, case_gold)):
    predictions = dp_assign(expected, observed, *best[1])
    runtime_observations += len(observed)
    runtime_targets += len(gold)
    runtime_matched += sum(pulse.target >= 0 for pulse in observed)

    expected_progress = normalized([pulse.prior for pulse in expected])
    gold_progress = normalized([pulse.center for pulse in gold])
    gold_by_expected = {pulse.expected: progress for pulse, progress in zip(gold, gold_progress)}
    gold_center_by_expected = {pulse.expected: pulse.center for pulse in gold}
    mapped_expected = [pulse for pulse in expected if pulse.index in gold_center_by_expected]
    if len(mapped_expected) >= 2:
        expected_span = max(mapped_expected[-1].prior - mapped_expected[0].prior, 1e-6)
        gold_span = max(
            gold_center_by_expected[mapped_expected[-1].index]
            - gold_center_by_expected[mapped_expected[0].index],
            1e-6,
        )
        scale = gold_span / expected_span
        previous_prediction = previous_center = None
        for pulse in mapped_expected:
            prediction = gold_center_by_expected[mapped_expected[0].index] + (
                pulse.prior - mapped_expected[0].prior
            ) * scale
            center = gold_center_by_expected[pulse.index]
            affine_center_errors_ms.append(abs(prediction - center) * 1000.0)
            if previous_prediction is not None:
                affine_interval_errors_ms.append(
                    abs(
                        (prediction - previous_prediction)
                        - (center - previous_center)
                    )
                    * 1000.0
                )
            previous_prediction = prediction
            previous_center = center
    previous_expected = previous_gold = None
    for pulse in expected:
        if pulse.index not in gold_by_expected:
            continue
        expected_position = expected_progress[pulse.index]
        gold_position = gold_by_expected[pulse.index]
        normalized_errors.append(abs(expected_position - gold_position))
        if previous_expected is not None:
            interval_errors.append(
                abs(
                    (expected_position - previous_expected)
                    - (gold_position - previous_gold)
                )
            )
        previous_expected = expected_position
        previous_gold = gold_position

    for observed_index, (pulse, prediction) in enumerate(zip(observed, predictions)):
        triplet_rows.append(
            {
                "case": case_names[case_index],
                "observed_index": observed_index,
                "runtime_center": pulse.center,
                "runtime_score": pulse.score,
                "mfa_expected_pulse": pulse.target,
                "predicted_transcript_pulse": prediction,
                "signed_pulse_error": prediction - pulse.target
                if prediction >= 0 and pulse.target >= 0
                else "",
                "exact": prediction >= 0 and prediction == pulse.target,
                "within_one": prediction >= 0
                and pulse.target >= 0
                and abs(prediction - pulse.target) <= 1,
            }
        )

with (OUT_ROOT / "pulse_alignment_triplets.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(triplet_rows[0]))
    writer.writeheader()
    writer.writerows(triplet_rows)

grade = {
    "qualified_cases": len(cases),
    "training_cases": len(training_cases),
    "held_out_cases": len(evaluation_cases),
    "parameters": {
        "skip_expected": best[1][0],
        "skip_observed": best[1][1],
        "progress_weight": best[1][2],
        "interval_weight": best[1][3],
        "stress_weight": best[1][4],
    },
    "transcript_mfa_normalized_position_mae": sum(normalized_errors)
    / max(len(normalized_errors), 1),
    "transcript_mfa_normalized_interval_mae": sum(interval_errors)
    / max(len(interval_errors), 1),
    "transcript_mfa_affine_center_mae_ms": sum(affine_center_errors_ms)
    / max(len(affine_center_errors_ms), 1),
    "transcript_mfa_affine_interval_mae_ms": sum(affine_interval_errors_ms)
    / max(len(affine_interval_errors_ms), 1),
    "runtime_mfa_precision": runtime_matched / max(runtime_observations, 1),
    "runtime_mfa_recall": runtime_matched / max(runtime_targets, 1),
    "pulse_matcher_train": {
        "precision": best[2][0],
        "recall": best[2][1],
        "within_one": best[2][2],
        "assignments": best[2][3],
    },
    "pulse_matcher_held_out": {
        "precision": held_out_result[0],
        "recall": held_out_result[1],
        "within_one": held_out_result[2],
        "assignments": held_out_result[3],
    },
    "pulse_matcher_full": {
        "precision": full_result[0],
        "recall": full_result[1],
        "within_one": full_result[2],
        "assignments": full_result[3],
    },
}
(OUT_ROOT / "pulse_alignment_grade.json").write_text(
    json.dumps(grade, indent=2) + "\n", encoding="utf-8"
)
print(
    f"transcript_mfa_position_mae={grade['transcript_mfa_normalized_position_mae']:.3f} "
    f"interval_mae={grade['transcript_mfa_normalized_interval_mae']:.3f} "
    f"affine_center_ms={grade['transcript_mfa_affine_center_mae_ms']:.1f} "
    f"affine_interval_ms={grade['transcript_mfa_affine_interval_mae_ms']:.1f}"
)
print(
    f"runtime_mfa_p={grade['runtime_mfa_precision']:.3f} "
    f"runtime_mfa_r={grade['runtime_mfa_recall']:.3f}"
)
