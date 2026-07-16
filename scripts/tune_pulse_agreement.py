#!/usr/bin/env python3
"""Tune causal pulse extraction for transcript/runtime count and location agreement."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
TOLERANCE_SEC = 0.100


@dataclass(frozen=True)
class Parameters:
    medium_half_width: int
    slow_half_width: int
    context_frames: int
    prominence_threshold: float
    vowel_threshold: float
    minimum_spacing_sec: float
    distinct_valley: float


def rows(path: Path) -> list[dict[str, str]]:
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


def extract_pulses(frames: list[dict[str, float]], p: Parameters) -> list[float]:
    energy = [
        frame["rms_norm"]
        * (0.45 + frame["vowel"] * 0.40 + frame["periodicity"] * 0.15)
        for frame in frames
    ]
    medium = triangular(energy, p.medium_half_width)
    slow = triangular(energy, p.slow_half_width)
    pulses: list[tuple[int, float]] = []

    def scale_prominence(envelope: list[float], index: int, inner: int) -> float:
        center = envelope[index]
        if center < envelope[index - 1] or center < envelope[index + 1]:
            return 0.0
        left = min(envelope[index - p.context_frames : index - inner + 1])
        right = min(envelope[index + inner : index + p.context_frames + 1])
        return min(max(min(center - left, center - right) / 0.12, 0.0), 1.0)

    for index in range(p.context_frames, len(frames) - p.context_frames):
        prominence = max(
            scale_prominence(medium, index, min(4, p.context_frames)),
            scale_prominence(slow, index, min(6, p.context_frames)) * 1.05,
        )
        vowel_support = min(max(frames[index]["vowel"] * 0.60 + frames[index]["periodicity"] * 0.40, 0.0), 1.0)
        if prominence < p.prominence_threshold or frames[index]["evidence"] < 0.18 or vowel_support < p.vowel_threshold:
            continue
        center = frames[index]["center"]
        if pulses:
            last_index, last_score = pulses[-1]
            spacing = center - frames[last_index]["center"]
            floor = min(medium[last_index : index + 1])
            lower_peak = min(medium[last_index], medium[index])
            distinct = spacing >= 0.060 and lower_peak - floor >= p.distinct_valley
            if spacing < p.minimum_spacing_sec and not distinct:
                if prominence > last_score:
                    pulses[-1] = (index, prominence)
                continue
        pulses.append((index, prominence))
    return [frames[index]["center"] for index, _ in pulses]


def monotonic_matches(observed: list[float], targets: list[float]) -> int:
    candidates = sorted(
        (abs(observation - target), oi, ti)
        for oi, observation in enumerate(observed)
        for ti, target in enumerate(targets)
        if abs(observation - target) <= TOLERANCE_SEC
    )
    used_observed: set[int] = set()
    used_targets: set[int] = set()
    matched = 0
    for _, oi, ti in candidates:
        if oi not in used_observed and ti not in used_targets:
            used_observed.add(oi)
            used_targets.add(ti)
            matched += 1
    return matched


def source_group(name: str) -> str:
    match = re.search(r"Alfie_Line_([A-F0-9]+)", name)
    return match.group(1) if match else name


cases = []
for case_dir in sorted(RUN_ROOT.iterdir()):
    occupancy_path = case_dir / "occupancy_frames.csv"
    phone_class_path = case_dir / "phone_class_frames.csv"
    grade_path = case_dir / "grade.json"
    gold_phone_path = GOLD_ROOT / case_dir.name / "phones.csv"
    gold_speech_path = GOLD_ROOT / case_dir.name / "speech.csv"
    if not all(path.exists() for path in (occupancy_path, phone_class_path, grade_path, gold_phone_path, gold_speech_path)):
        continue
    grade = json.loads(grade_path.read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    occupancy = rows(occupancy_path)
    phone_class = rows(phone_class_path)
    if len(occupancy) != len(phone_class):
        continue
    frames = [
        {
            "center": float(o["center"]),
            "rms_norm": float(o["rms_norm"]),
            "periodicity": float(o["periodicity"]),
            "evidence": float(o["evidence"]),
            "vowel": float(c["vowel"]),
        }
        for o, c in zip(occupancy, phone_class)
    ]
    gold_phones = rows(gold_phone_path)
    gold_regions = rows(gold_speech_path)
    region_targets: list[list[float]] = []
    for region in gold_regions:
        start, end = float(region["start"]), float(region["end"])
        region_targets.append([
            (float(phone["start"]) + float(phone["end"])) * 0.5
            for phone in gold_phones
            if phone["phone"].rstrip("012") in VOWELS
            and start - 0.001 <= (float(phone["start"]) + float(phone["end"])) * 0.5 <= end + 0.001
        ])
    evaluation = int(hashlib.sha1(source_group(case_dir.name).encode()).hexdigest()[:8], 16) % 5 == 0
    cases.append((case_dir.name, frames, gold_regions, region_targets, evaluation))


def evaluate(parameters: Parameters, evaluation_only: bool) -> tuple[float, float, float, float, float, int]:
    observations = targets = matched = exact_regions = regions = 0
    absolute_count_error = 0
    for _, frames, gold_regions, region_targets, is_evaluation in cases:
        if is_evaluation != evaluation_only:
            continue
        pulses = extract_pulses(frames, parameters)
        for region, expected in zip(gold_regions, region_targets):
            start, end = float(region["start"]), float(region["end"])
            observed = [center for center in pulses if start - 0.001 <= center <= end + 0.001]
            observations += len(observed)
            targets += len(expected)
            matched += monotonic_matches(observed, expected)
            exact_regions += len(observed) == len(expected)
            absolute_count_error += abs(len(observed) - len(expected))
            regions += 1
    precision = matched / max(observations, 1)
    recall = matched / max(targets, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, exact_regions / max(regions, 1), absolute_count_error / max(regions, 1), regions


baseline = Parameters(4, 6, 12, 0.16, 0.25, 0.120, 0.030)
print(f"cases={len(cases)} train={sum(not case[-1] for case in cases)} held_out={sum(case[-1] for case in cases)}")
print("baseline_train", evaluate(baseline, False))
print("baseline_test ", evaluate(baseline, True))

best = None
for tail in itertools.product(
    (0.10, 0.13, 0.16, 0.19, 0.22),
    (0.15, 0.25, 0.35, 0.45),
    (0.100, 0.120, 0.140, 0.160),
    (0.015, 0.025, 0.035, 0.045),
):
    parameters = Parameters(4, 6, 12, *tail)
    result = evaluate(parameters, False)
    objective = result[2] + 0.20 * result[3] - 0.10 * result[4]
    if best is None or objective > best[0]:
        best = (objective, parameters, result)

assert best is not None
for head in itertools.product((3, 4, 5), (5, 6, 7, 8), (10, 12, 14, 16)):
    tail = (
        best[1].prominence_threshold,
        best[1].vowel_threshold,
        best[1].minimum_spacing_sec,
        best[1].distinct_valley,
    )
    parameters = Parameters(*head, *tail)
    result = evaluate(parameters, False)
    objective = result[2] + 0.20 * result[3] - 0.10 * result[4]
    if objective > best[0]:
        best = (objective, parameters, result)
print("best", best[1])
print("best_train", best[2])
print("best_test ", evaluate(best[1], True))
