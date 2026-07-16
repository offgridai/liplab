#!/usr/bin/env python3
"""Measure transcript-count-conditioned selection of runtime pulse candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
TOLERANCE = 0.100


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def triangular(values: list[float], half_width: int) -> list[float]:
    result = []
    for index in range(len(values)):
        weighted = weight_sum = 0.0
        for offset in range(-half_width, half_width + 1):
            source = index + offset
            if 0 <= source < len(values):
                weight = half_width + 1 - abs(offset)
                weighted += values[source] * weight
                weight_sum += weight
        result.append(weighted / weight_sum if weight_sum else 0.0)
    return result


@dataclass
class Candidate:
    center: float
    prominence: float
    vowel_open: float
    vowel_front: float
    vowel_round: float


def candidates(case_dir: Path, threshold: float, minimum_spacing: float) -> list[Candidate]:
    occupancy = read_csv(case_dir / "occupancy_frames.csv")
    classes = read_csv(case_dir / "phone_class_frames.csv")
    frames = [
        {
            "center": float(o["center"]),
            "energy": float(o["rms_norm"])
            * (0.45 + float(c["vowel"]) * 0.40 + float(o["periodicity"]) * 0.15),
            "vowel": float(c["vowel"]) * 0.60 + float(o["periodicity"]) * 0.40,
            "vowel_open": float(c["vowel_open"]),
            "vowel_front": float(c["vowel_front"]),
            "vowel_round": float(c["vowel_round"]),
            "speech": float(o["evidence"]),
        }
        for o, c in zip(occupancy, classes)
    ]
    medium = triangular([frame["energy"] for frame in frames], 4)
    slow = triangular([frame["energy"] for frame in frames], 6)
    fast = triangular([frame["energy"] for frame in frames], 2)
    sonority = triangular([frame["vowel"] for frame in frames], 2)
    result: list[Candidate] = []
    context = 10
    for index in range(context, len(frames) - context):
        def prominence(envelope: list[float], inner: int) -> float:
            center = envelope[index]
            if center < envelope[index - 1] or center < envelope[index + 1]:
                return 0.0
            left = min(envelope[index - context : index - inner + 1])
            right = min(envelope[index + inner : index + context + 1])
            return min(max(min(center - left, center - right) / 0.12, 0.0), 1.0)

        broad_score = max(prominence(medium, 4), prominence(slow, 6) * 1.05)
        fast_peak = fast[index] >= fast[index - 1] and fast[index] >= fast[index + 1]
        sonority_peak = sonority[index] >= sonority[index - 1] and sonority[index] >= sonority[index + 1]
        fast_contrast = max(
            fast[index] - min(fast[index - 4 : index]),
            fast[index] - min(fast[index + 1 : index + 5]),
        )
        sonority_contrast = max(
            sonority[index] - min(sonority[index - 4 : index]),
            sonority[index] - min(sonority[index + 1 : index + 5]),
        )
        score = max(
            broad_score,
            min(max(fast_contrast / 0.08, 0.0), 1.0) if fast_peak else 0.0,
            min(max(sonority_contrast / 0.20, 0.0), 1.0) if sonority_peak else 0.0,
        )
        if score < threshold or frames[index]["vowel"] < 0.12 or frames[index]["speech"] < 0.12:
            continue
        candidate = Candidate(
            frames[index]["center"], score,
            frames[index]["vowel_open"], frames[index]["vowel_front"], frames[index]["vowel_round"],
        )
        if result and candidate.center - result[-1].center < minimum_spacing:
            if candidate.prominence > result[-1].prominence:
                result[-1] = candidate
            continue
        result.append(candidate)
    return result


@dataclass
class Expected:
    index: int
    word: int
    local: int
    prior: float
    vowel_family: int
    landmarks: tuple[tuple[str, float], ...]


def expected(case_dir: Path) -> list[Expected]:
    centers = {}
    cursor = 0.0
    previous_word = None
    for phone in read_csv(case_dir / "expected_phones.csv"):
        word = int(phone["word_index"])
        if previous_word is not None and word != previous_word:
            cursor += 0.020
        duration = max(float(phone["weight_seconds"]), 0.025)
        centers[int(phone["phone_index"])] = cursor + duration * 0.5
        cursor += duration
        previous_word = word
    def family(phone: str) -> int:
        base = phone.rstrip("012")
        if base in {"EH", "EY", "IH", "IY", "ER"}:
            return 1
        if base in {"AO", "AW", "OW", "OY", "UH", "UW"}:
            return 2
        return 0

    phones = {int(row["phone_index"]): row for row in read_csv(case_dir / "expected_phones.csv")}
    def landmark(base: str) -> str | None:
        if base in {"M", "B", "P"}: return "mbp"
        if base in {"F", "V"}: return "fv"
        if base in {"W", "Y"}: return "glide"
        if base in {"S", "Z", "SH", "ZH", "CH", "JH"}: return "sibilant"
        if base in {"AO", "OW", "OY", "UH", "UW"}: return "round"
        return None
    result = []
    for row in read_csv(case_dir / "planned_syllables.csv"):
        nucleus_index = int(row["nucleus_phone_index"])
        marks = []
        for phone_index in range(int(row["phone_begin_index"]), int(row["phone_end_index"])):
            mark = landmark(phones[phone_index]["base_phone"])
            if mark:
                marks.append((mark, centers[phone_index] - centers[nucleus_index]))
        result.append(Expected(
            int(row["syllable_index"]), int(row["word_index"]),
            int(row["word_syllable_index"]), centers[nucleus_index],
            family(row["nucleus_phone"]), tuple(marks),
        ))
    return result


def normalized(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.5] * len(values)
    # Preserve edge breathing room instead of pinning first/last nuclei to the
    # acoustic region boundaries.
    gaps = [max(values[i + 1] - values[i], 0.025) for i in range(len(values) - 1)]
    left = gaps[0] * 0.5
    right = gaps[-1] * 0.5
    low = values[0] - left
    span = max(values[-1] + right - low, 1e-6)
    return [(value - low) / span for value in values]


def select_track(expected_pulses: list[Expected], audio: list[Candidate], landmarks: list[tuple[str, float, float]], start: float, end: float, progress_weight: float, score_weight: float, family_weight: float, interval_weight: float, landmark_weight: float) -> list[Candidate]:
    n, m = len(expected_pulses), len(audio)
    if not n or m < n:
        return []
    expected_progress = normalized([pulse.prior for pulse in expected_pulses])
    audio_progress = [(pulse.center - start) / max(end - start, 1e-6) for pulse in audio]
    def local_cost(i: int, j: int) -> float:
            family_scores = (audio[j].vowel_open, audio[j].vowel_front, audio[j].vowel_round)
            family_support = family_scores[expected_pulses[i].vowel_family]
            family_margin = family_support - max(
                score for index, score in enumerate(family_scores)
                if index != expected_pulses[i].vowel_family
            )
            landmark_support = 0.0
            for kind, offset in expected_pulses[i].landmarks:
                best = max(
                    (score * max(1.0 - abs(center - (audio[j].center + offset)) / 0.120, 0.0)
                     for observed_kind, center, score in landmarks if observed_kind == kind),
                    default=0.0,
                )
                landmark_support += best
            landmark_support /= max(len(expected_pulses[i].landmarks), 1)
            return (
                progress_weight * abs(expected_progress[i] - audio_progress[j])
                - score_weight * audio[j].prominence
                - family_weight * family_margin
                - landmark_weight * landmark_support
            )
    inf = 1e30
    table = [[inf] * m for _ in range(n)]
    back = [[-1] * m for _ in range(n)]
    for j in range(m):
        table[0][j] = local_cost(0, j)
    for i in range(1, n):
        expected_gap = expected_progress[i] - expected_progress[i - 1]
        for j in range(i, m):
            local = local_cost(i, j)
            for previous in range(i - 1, j):
                observed_gap = audio_progress[j] - audio_progress[previous]
                cost = table[i - 1][previous] + local + interval_weight * abs(expected_gap - observed_gap)
                if cost < table[i][j]:
                    table[i][j] = cost
                    back[i][j] = previous
    finish = min(range(n - 1, m), key=lambda j: table[n - 1][j])
    indices = [finish]
    for i in range(n - 1, 0, -1):
        finish = back[i][finish]
        if finish < 0:
            return []
        indices.append(finish)
    return [audio[index] for index in reversed(indices)]


def source_group(name: str) -> str:
    match = re.search(r"Alfie_Line_([A-F0-9]+)", name)
    return match.group(1) if match else name


cases = []
for case_dir in sorted(RUN_ROOT.iterdir()):
    gold_dir = GOLD_ROOT / case_dir.name
    required = [case_dir / "grade.json", case_dir / "expected_phones.csv", case_dir / "planned_syllables.csv", case_dir / "occupancy_frames.csv", case_dir / "phone_class_frames.csv", gold_dir / "phones.csv", gold_dir / "speech.csv"]
    if not all(path.exists() for path in required):
        continue
    grade = json.loads(required[0].read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    planned = expected(case_dir)
    gold_phones = read_csv(gold_dir / "phones.csv")
    gold_regions = read_csv(gold_dir / "speech.csv")
    region_data = []
    for region in gold_regions:
        first_word, last_word = int(region["word_start_index"]), int(region["word_end_index"])
        planned_region = [pulse for pulse in planned if first_word <= pulse.word <= last_word]
        gold_by_key = {}
        local_by_word = {}
        for phone in gold_phones:
            if phone["phone"].rstrip("012") not in VOWELS:
                continue
            word = int(phone["word_index"])
            local = local_by_word.get(word, 0)
            local_by_word[word] = local + 1
            gold_by_key[(word, local)] = (float(phone["start"]) + float(phone["end"])) * 0.5
        gold_centers = [gold_by_key.get((pulse.word, pulse.local)) for pulse in planned_region]
        region_data.append((float(region["start"]), float(region["end"]), planned_region, gold_centers))
    held_out = int(hashlib.sha1(source_group(case_dir.name).encode()).hexdigest()[:8], 16) % 5 == 0
    cases.append((case_dir, region_data, held_out))


candidate_cache = {}
landmark_cache = {}
def evaluate(parameters, held_out):
    threshold, spacing, progress_weight, score_weight, family_weight, interval_weight, landmark_weight = parameters
    assigned = correct = target_count = exact_count = region_count = missing_candidates = 0
    center_errors = []
    for case_dir, regions, is_held_out in cases:
        if is_held_out != held_out:
            continue
        key = (case_dir.name, threshold, spacing)
        if key not in candidate_cache:
            candidate_cache[key] = candidates(case_dir, threshold, spacing)
        all_candidates = candidate_cache[key]
        if case_dir.name not in landmark_cache:
            landmark_rows = read_csv(case_dir / "streaming_permissive_evidence_candidates.csv")
            landmark_cache[case_dir.name] = [
                (row["type"], float(row["center"]), float(row["score"]))
                for row in landmark_rows if row["type"] in {"mbp", "fv", "glide", "sibilant", "round"}
            ]
        landmarks = landmark_cache[case_dir.name]
        for start, end, planned, gold in regions:
            audio = [candidate for candidate in all_candidates if start <= candidate.center <= end]
            selected = select_track(planned, audio, landmarks, start, end, progress_weight, score_weight, family_weight, interval_weight, landmark_weight)
            target_count += sum(center is not None for center in gold)
            region_count += 1
            exact_count += len(selected) == len(planned)
            missing_candidates += len(audio) < len(planned)
            for candidate, gold_center in zip(selected, gold):
                if gold_center is None:
                    continue
                assigned += 1
                error = abs(candidate.center - gold_center)
                center_errors.append(error * 1000.0)
                correct += error <= TOLERANCE
    precision = correct / max(assigned, 1)
    recall = correct / max(target_count, 1)
    return precision, recall, exact_count / max(region_count, 1), sum(center_errors) / max(len(center_errors), 1), missing_candidates


grid = [
    (0.02, 0.05, progress, score, 0.0, interval, landmark)
    for progress in (0.25, 0.5, 1.0, 2.0, 4.0)
    for score in (0.0, 0.05, 0.10, 0.20, 0.40)
    for interval in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    for landmark in (0.0, 0.10, 0.20, 0.40, 0.80)
]
best = None
for parameters in grid:
    result = evaluate(parameters, False)
    objective = result[0] + result[1] + 0.25 * result[2] - result[3] / 1000.0
    if best is None or objective > best[0]:
        best = objective, parameters, result
assert best is not None
print(f"cases={len(cases)} train={sum(not case[-1] for case in cases)} held_out={sum(case[-1] for case in cases)}")
print("best", best[1])
print("train precision recall exact_count center_mae_ms insufficient_candidates", best[2])
print("test  precision recall exact_count center_mae_ms insufficient_candidates", evaluate(best[1], True))
