import csv
import itertools
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "latest"
GOLD = ROOT / "inputs" / "gold"
FAMILIES = {
    "mbp": {"M", "B", "P"},
    "fv": {"F", "V"},
    "glide": {"W", "Y"},
    "sibilant": {"S", "Z", "SH", "ZH", "CH", "JH"},
    "open": {"AA", "AE", "AH", "AW", "AY"},
    "front": {"EH", "ER", "EY", "IH", "IY"},
    "round": {"AO", "OW", "OY", "UH", "UW"},
}
RELIABILITY = {
    "mbp": 0.58, "fv": 0.66, "glide": 0.77, "sibilant": 0.92,
    "open": 0.58, "front": 0.58, "round": 0.66,
}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family_for(phone):
    for family, phones in FAMILIES.items():
        if phone in phones:
            return family
    return None


def group_for(name):
    match = re.search(r"_Line_([0-9A-F]+)_", name)
    return match.group(1) if match else name


def load_cases():
    cases = []
    for case_dir in sorted(path for path in RUN.iterdir() if path.is_dir()):
        grade_path = case_dir / "grade.json"
        gold_path = GOLD / case_dir.name / "phones.csv"
        if not grade_path.exists() or not gold_path.exists():
            continue
        grade = json.loads(grade_path.read_text(encoding="utf-8"))
        if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", False):
            continue
        expected = rows(case_dir / "expected_phones.csv")
        syllables = rows(case_dir / "planned_syllables.csv")
        gold = rows(gold_path)
        phone_centers = []
        elapsed = 0.0
        for phone in expected:
            weight = float(phone["weight_seconds"])
            phone_centers.append(elapsed + weight * 0.5)
            elapsed += weight
            phone_index = len(phone_centers) - 1
            if (phone_index + 1 >= len(expected) or (
                expected[phone_index + 1]["word_index"] != phone["word_index"]
            )):
                elapsed += float(phone["pause_seconds_after_word"])
        targets = []
        for syllable in syllables:
            nucleus = int(syllable["nucleus_phone_index"])
            if nucleus < 0 or nucleus >= len(expected) or nucleus >= len(gold):
                continue
            targets.append({
                "nucleus": nucleus,
                "word": int(syllable["word_index"]),
                "word_syllable": int(syllable["word_syllable_index"]),
                "region": int(syllable["speech_region_index"]),
                "stress": int(expected[nucleus]["phone"][-1]) if expected[nucleus]["phone"][-1:].isdigit() else 0,
                "prior": phone_centers[nucleus],
                "gold_start": float(gold[nucleus]["start"]),
                "gold_end": float(gold[nucleus]["end"]),
                "families": [],
            })
        for phone_index, phone in enumerate(expected):
            family = family_for(phone["base_phone"])
            if family is None:
                continue
            word = int(phone["word_index"])
            choices = [
                (abs(target["nucleus"] - phone_index), index)
                for index, target in enumerate(targets) if target["word"] == word
            ]
            if choices:
                target_index = min(choices)[1]
                targets[target_index]["families"].append(
                    (family, phone_centers[phone_index] - targets[target_index]["prior"])
                )
        observations = rows(case_dir / "streaming_permissive_evidence_candidates.csv")
        pulses = [item for item in observations if item["type"] == "pulse"]
        phones = [item for item in observations if item["type"] in FAMILIES]
        if not targets or not pulses:
            continue
        prior_span = max(targets[-1]["prior"] - targets[0]["prior"], 0.001)
        audio_span = max(float(pulses[-1]["center"]) - float(pulses[0]["center"]), 0.001)
        components = []
        for target in targets:
            row_components = []
            for pulse in pulses:
                pulse_center = float(pulse["center"])
                pulse_decision = float(pulse["decision"])
                prior_progress = (target["prior"] - targets[0]["prior"]) / prior_span
                audio_progress = (pulse_center - float(pulses[0]["center"])) / audio_span
                timing = max(0.0, 1.0 - abs(prior_progress - audio_progress) / 0.35)
                reliability_sum = evidence_sum = 0.0
                expected_families = {family for family, _ in target["families"]}
                for family, offset in target["families"]:
                    reliability = RELIABILITY[family]
                    expected_center = pulse_center + offset
                    best = 0.0
                    for candidate in phones:
                        if candidate["type"] != family or float(candidate["decision"]) > pulse_decision + 0.001:
                            continue
                        proximity = max(0.0, 1.0 - abs(float(candidate["center"]) - expected_center) / 0.120)
                        best = max(best, float(candidate["score"]) * proximity)
                    evidence_sum += reliability * best
                    reliability_sum += reliability
                evidence = evidence_sum / max(reliability_sum, 1.0)
                unexpected = 0.0
                for candidate in phones:
                    if float(candidate["decision"]) > pulse_decision + 0.001:
                        continue
                    if abs(float(candidate["center"]) - pulse_center) > 0.160:
                        continue
                    if candidate["type"] not in expected_families:
                        unexpected = max(
                            unexpected,
                            RELIABILITY[candidate["type"]] * float(candidate["score"]),
                        )
                row_components.append((timing, evidence, unexpected))
            components.append(row_components)
        cases.append((case_dir.name, targets, pulses, components))
    return cases


def assign(targets, pulses, components, parameters):
    target_skip, pulse_skip, base, timing_weight, evidence_weight, unexpected_weight = parameters
    rows_count, columns_count = len(targets) + 1, len(pulses) + 1
    negative = -1.0e30
    scores = [[negative] * columns_count for _ in range(rows_count)]
    back = [[0] * columns_count for _ in range(rows_count)]
    scores[0][0] = 0.0
    for row in range(rows_count):
        for column in range(columns_count):
            current = scores[row][column]
            if current <= negative / 2:
                continue
            if row < len(targets) and current - target_skip > scores[row + 1][column]:
                scores[row + 1][column] = current - target_skip
                back[row + 1][column] = 1
            if column < len(pulses) and current - pulse_skip > scores[row][column + 1]:
                scores[row][column + 1] = current - pulse_skip
                back[row][column + 1] = 2
            if row < len(targets) and column < len(pulses):
                timing, evidence, unexpected = components[row][column]
                local = base + timing_weight * timing + evidence_weight * evidence - unexpected_weight * unexpected
                if current + local > scores[row + 1][column + 1]:
                    scores[row + 1][column + 1] = current + local
                    back[row + 1][column + 1] = 3
    assignments = []
    row, column = len(targets), len(pulses)
    while row > 0 or column > 0:
        step = back[row][column]
        if step == 3:
            assignments.append((column - 1, row - 1))
            row -= 1
            column -= 1
        elif step == 1:
            row -= 1
        elif step == 2:
            column -= 1
        else:
            break
    return list(reversed(assignments))


def assign_segmental(targets, pulses, components, parameters):
    (
        target_skip,
        pulse_skip,
        base,
        timing_weight,
        evidence_weight,
        unexpected_weight,
        duration_weight,
        duration_floor,
        duration_scale,
        max_step,
        pulse_score_weight,
        rate_normalized,
    ) = parameters
    rows_count, columns_count = len(targets), len(pulses)
    negative = -1.0e30
    scores = [[negative] * columns_count for _ in range(rows_count)]
    back = [[None] * columns_count for _ in range(rows_count)]
    prior_span = max(targets[-1]["prior"] - targets[0]["prior"], 0.001)
    audio_span = max(float(pulses[-1]["center"]) - float(pulses[0]["center"]), 0.001)
    line_rate = audio_span / prior_span
    for row in range(rows_count):
        for column in range(columns_count):
            timing, evidence, unexpected = components[row][column]
            local = (
                base
                + timing_weight * timing
                + evidence_weight * evidence
                - unexpected_weight * unexpected
                + pulse_score_weight * float(pulses[column]["score"])
            )
            best = local - target_skip * row - pulse_skip * column
            best_back = None
            for previous_row in range(max(0, row - max_step), row):
                for previous_column in range(max(0, column - max_step), column):
                    previous = scores[previous_row][previous_column]
                    if previous <= negative / 2:
                        continue
                    expected_gap = targets[row]["prior"] - targets[previous_row]["prior"]
                    if rate_normalized:
                        expected_gap *= line_rate
                    observed_gap = float(pulses[column]["center"]) - float(pulses[previous_column]["center"])
                    denominator = max(duration_floor, expected_gap * duration_scale)
                    duration_error = min(abs(observed_gap - expected_gap) / denominator, 3.0)
                    candidate = (
                        previous
                        + local
                        - target_skip * (row - previous_row - 1)
                        - pulse_skip * (column - previous_column - 1)
                        - duration_weight * duration_error
                    )
                    if candidate > best:
                        best = candidate
                        best_back = (previous_row, previous_column)
            scores[row][column] = best
            back[row][column] = best_back

    best_score = negative
    cursor = None
    for row in range(rows_count):
        for column in range(columns_count):
            final = (
                scores[row][column]
                - target_skip * (rows_count - row - 1)
                - pulse_skip * (columns_count - column - 1)
            )
            if final > best_score:
                best_score = final
                cursor = (row, column)
    assignments = []
    while cursor is not None:
        row, column = cursor
        assignments.append((column, row))
        cursor = back[row][column]
    return list(reversed(assignments))


def score(cases, parameters):
    target_count = assignment_count = exact = within_one = 0
    for _, targets, pulses, components in cases:
        target_count += len(targets)
        assignments = assign(targets, pulses, components, parameters)
        assignment_count += len(assignments)
        used = [False] * len(targets)
        for pulse_index, predicted in assignments:
            center = float(pulses[pulse_index]["center"])
            actual = None
            best_error = 0.100001
            for index, target in enumerate(targets):
                if used[index]:
                    continue
                error = max(target["gold_start"] - center, center - target["gold_end"], 0.0)
                if error < best_error:
                    actual = index
                    best_error = error
            if actual is None:
                continue
            used[actual] = True
            error = abs(predicted - actual)
            exact += error == 0
            within_one += error <= 1
    precision = exact / assignment_count if assignment_count else 0.0
    recall = exact / target_count if target_count else 0.0
    within = within_one / assignment_count if assignment_count else 0.0
    f1 = 2 * exact / (assignment_count + target_count) if assignment_count + target_count else 0.0
    return precision, recall, within, f1, assignment_count, target_count, exact


def score_segmental(cases, parameters):
    target_count = assignment_count = exact = within_one = 0
    for _, targets, pulses, components in cases:
        target_count += len(targets)
        assignments = assign_segmental(targets, pulses, components, parameters)
        assignment_count += len(assignments)
        used = [False] * len(targets)
        for pulse_index, predicted in assignments:
            center = float(pulses[pulse_index]["center"])
            actual = None
            best_error = 0.100001
            for index, target in enumerate(targets):
                if used[index]:
                    continue
                error = max(target["gold_start"] - center, center - target["gold_end"], 0.0)
                if error < best_error:
                    actual = index
                    best_error = error
            if actual is None:
                continue
            used[actual] = True
            error = abs(predicted - actual)
            exact += error == 0
            within_one += error <= 1
    precision = exact / assignment_count if assignment_count else 0.0
    recall = exact / target_count if target_count else 0.0
    within = within_one / assignment_count if assignment_count else 0.0
    f1 = 2 * exact / (assignment_count + target_count) if assignment_count + target_count else 0.0
    return precision, recall, within, f1, assignment_count, target_count, exact


def main():
    cases = load_cases()
    groups = sorted({group_for(name) for name, *_ in cases})
    heldout_groups = set(groups[::5])
    train = [case for case in cases if group_for(case[0]) not in heldout_groups]
    heldout = [case for case in cases if group_for(case[0]) in heldout_groups]
    stage_one = []
    for timing_weight, evidence_weight, unexpected_weight in itertools.product(
        (0.10, 0.20, 0.40, 0.60),
        (0.00, 0.50, 1.00, 1.50),
        (0.00, 0.10, 0.20, 0.40),
    ):
        parameters = (0.35, 0.25, 0.45, timing_weight, evidence_weight, unexpected_weight)
        result = score(train, parameters)
        stage_one.append((result[3], min(result[0], result[1]), parameters, result))
    first = max(stage_one)
    stage_two = []
    for target_skip, pulse_skip, base in itertools.product(
        (0.20, 0.35, 0.50),
        (0.15, 0.25, 0.35),
        (0.25, 0.45, 0.65),
    ):
        parameters = (target_skip, pulse_skip, base, *first[2][3:])
        result = score(train, parameters)
        stage_two.append((result[3], min(result[0], result[1]), parameters, result))
    best = max(stage_two)
    stage_three = []
    for target_skip, pulse_skip, base, timing_weight, evidence_weight in itertools.product(
        (0.10, 0.15, 0.20, 0.25),
        (0.20, 0.25, 0.30),
        (0.10, 0.20, 0.25, 0.30),
        (0.40, 0.60, 0.80, 1.00),
        (1.50, 2.00, 2.50),
    ):
        parameters = (target_skip, pulse_skip, base, timing_weight, evidence_weight, 0.0)
        result = score(train, parameters)
        stage_three.append((result[3], min(result[0], result[1]), parameters, result))
    best = max(stage_three)
    stage_four = []
    for parameters in itertools.product(
        (0.20, 0.25, 0.30),
        (0.10, 0.15, 0.20, 0.25),
        (0.00, 0.05, 0.10, 0.15),
        (0.50, 0.60, 0.70),
        (1.75, 2.00, 2.25),
    ):
        parameters = (*parameters, 0.0)
        result = score(train, parameters)
        stage_four.append((result[3], min(result[0], result[1]), parameters, result))
    best = max(stage_four)
    segmental = []
    for duration_weight, duration_floor, duration_scale, max_step in itertools.product(
        (0.05, 0.10, 0.20, 0.40, 0.80),
        (0.080, 0.120, 0.180),
        (0.50, 0.75, 1.00),
        (2, 3, 4, 6),
    ):
        parameters = (*best[2], duration_weight, duration_floor, duration_scale, max_step, 0.0, False)
        result = score_segmental(train, parameters)
        segmental.append((result[3], min(result[0], result[1]), parameters, result))
    best_segmental = max(segmental)
    segmental_refinement = []
    for duration_weight, duration_floor, pulse_score_weight, rate_normalized in itertools.product(
        (0.40, 0.60, 0.80, 1.00, 1.20),
        (0.080, 0.120, 0.160),
        (0.00, 0.10, 0.20, 0.40),
        (False, True),
    ):
        parameters = (
            *best[2],
            duration_weight,
            duration_floor,
            1.0,
            2,
            pulse_score_weight,
            rate_normalized,
        )
        result = score_segmental(train, parameters)
        segmental_refinement.append((result[3], min(result[0], result[1]), parameters, result))
    best_segmental = max(best_segmental, max(segmental_refinement))
    segmental_local = []
    for timing_weight, evidence_weight, unexpected_weight, pulse_score_weight in itertools.product(
        (0.30, 0.50, 0.70, 1.00),
        (1.50, 2.25, 3.00, 4.00),
        (0.00, 0.10, 0.20, 0.40),
        (0.20, 0.40, 0.60),
    ):
        parameters = (
            0.20,
            0.10,
            0.00,
            timing_weight,
            evidence_weight,
            unexpected_weight,
            0.80,
            0.080,
            1.0,
            2,
            pulse_score_weight,
            False,
        )
        result = score_segmental(train, parameters)
        segmental_local.append((result[3], min(result[0], result[1]), parameters, result))
    best_local = max(segmental_local)
    segmental_penalties = []
    for target_skip, pulse_skip, duration_weight, duration_floor in itertools.product(
        (0.10, 0.15, 0.20, 0.25, 0.30),
        (0.05, 0.10, 0.15),
        (0.40, 0.60, 0.80, 1.00, 1.20),
        (0.060, 0.080, 0.100),
    ):
        parameters = (
            target_skip,
            pulse_skip,
            best_local[2][2],
            best_local[2][3],
            best_local[2][4],
            best_local[2][5],
            duration_weight,
            duration_floor,
            1.0,
            2,
            best_local[2][10],
            False,
        )
        result = score_segmental(train, parameters)
        segmental_penalties.append((result[3], min(result[0], result[1]), parameters, result))
    best_segmental = max(best_segmental, best_local, max(segmental_penalties))
    print(json.dumps({
        "case_count": len(cases),
        "train_cases": len(train),
        "heldout_cases": len(heldout),
        "baseline": {
            "parameters": [0.35, 0.25, 0.45, 0.20, 1.00, 0.20],
            "train": score(train, (0.35, 0.25, 0.45, 0.20, 1.00, 0.20)),
            "heldout": score(heldout, (0.35, 0.25, 0.45, 0.20, 1.00, 0.20)),
        },
        "selected": {
            "parameters": best[2],
            "train": score(train, best[2]),
            "heldout": score(heldout, best[2]),
            "all": score(cases, best[2]),
        },
        "segmental": {
            "parameters": best_segmental[2],
            "train": score_segmental(train, best_segmental[2]),
            "heldout": score_segmental(heldout, best_segmental[2]),
            "all": score_segmental(cases, best_segmental[2]),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
