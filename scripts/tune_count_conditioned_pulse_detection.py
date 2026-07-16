#!/usr/bin/env python3
"""Select an acoustic pulse inventory using only transcript regional counts."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from pathlib import Path


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
TOLERANCE_SEC = 0.100


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_group(name: str) -> str:
    match = re.search(r"Alfie_Line_([A-F0-9]+)", name)
    return match.group(1) if match else name


def interval_error(center: float, target: tuple[float, float, float]) -> float:
    if center < target[0]:
        return target[0] - center
    if center > target[1]:
        return center - target[1]
    return 0.0


def triangular(values: list[float], half_width: int) -> list[float]:
    result = []
    for index in range(len(values)):
        total = weight_sum = 0.0
        for offset in range(-half_width, half_width + 1):
            source = index + offset
            if 0 <= source < len(values):
                weight = half_width + 1 - abs(offset)
                total += values[source] * weight
                weight_sum += weight
        result.append(total / weight_sum if weight_sum else 0.0)
    return result


def dense_acoustic_candidates(
    occupancy: list[dict[str, str]],
    classes: list[dict[str, str]],
    landmarks: list[dict[str, str]],
) -> list[tuple[float, float]]:
    rms = [float(row["rms_norm"]) for row in occupancy]
    speech = [float(row["evidence"]) for row in occupancy]
    periodicity = [float(row["periodicity"]) for row in occupancy]
    vowel = [float(row["vowel"]) for row in classes]
    sonority = triangular([
        rms[index] * (0.35 + vowel[index] * 0.45 + periodicity[index] * 0.20)
        for index in range(len(rms))
    ], 2)
    band_sonority = triangular([
        rms[index]
        * (0.15 + float(landmarks[index]["rich_low"]) * 0.45 + float(landmarks[index]["rich_mid"]) * 0.40)
        * (0.45 + periodicity[index] * 0.55)
        for index in range(len(rms))
    ], 2)
    sonority_rate = triangular([
        max(sonority[index] - sonority[max(index - 2, 0)], 0.0) / 0.08
        for index in range(len(rms))
    ], 1)
    band_rate = triangular([
        max(band_sonority[index] - band_sonority[max(index - 2, 0)], 0.0) / 0.07
        for index in range(len(rms))
    ], 1)
    flux = triangular([
        min(float(landmarks[index]["rich_flux"]) / 0.30, 1.0)
        * max(vowel[index], periodicity[index])
        for index in range(len(rms))
    ], 1)
    candidates = []
    for index in range(2, len(rms) - 2):
        support = vowel[index] * 0.60 + periodicity[index] * 0.40
        if speech[index] < 0.12 or support < 0.12:
            continue
        peak_scores = []
        for values, weight in ((sonority, 1.0), (sonority_rate, 0.88), (band_rate, 0.82), (flux, 0.72)):
            if values[index] >= max(values[index - 2 : index + 3]):
                peak_scores.append(min(values[index], 1.0) * weight)
        if not peak_scores or max(peak_scores) < 0.12:
            continue
        candidates.append((float(occupancy[index]["center"]), max(peak_scores)))
    # Collapse only coincident 10 ms-frame alternatives. The count-conditioned
    # selector, not this candidate stage, owns wider pulse separation.
    deduplicated: list[tuple[float, float]] = []
    for center, score in candidates:
        if deduplicated and center - deduplicated[-1][0] < 0.020:
            if score > deduplicated[-1][1]:
                deduplicated[-1] = (center, score)
        else:
            deduplicated.append((center, score))
    return deduplicated


def select_inventory(
    candidates: list[tuple[float, float]],
    count: int,
    start: float,
    end: float,
    score_weight: float,
    progress_weight: float,
    interval_weight: float,
) -> list[float]:
    if count <= 0 or not candidates:
        return []
    candidates = sorted(candidates)
    if len(candidates) <= count:
        return [center for center, _ in candidates]
    duration = max(end - start, 0.001)
    expected_interval = duration / max(count, 1)
    # dp[k][j] is the best acoustic inventory ending with candidate j as pulse k.
    dp = [[-1.0e30] * len(candidates) for _ in range(count)]
    previous = [[-1] * len(candidates) for _ in range(count)]
    for pulse_index in range(count):
        expected_progress = (pulse_index + 0.5) / count
        for candidate_index, (center, score) in enumerate(candidates):
            progress = (center - start) / duration
            local = score_weight * score - progress_weight * abs(progress - expected_progress)
            if pulse_index == 0:
                dp[pulse_index][candidate_index] = local
                continue
            for prior_index in range(candidate_index):
                interval = center - candidates[prior_index][0]
                if interval < 0.050:
                    continue
                utility = (
                    dp[pulse_index - 1][prior_index]
                    + local
                    - interval_weight * abs(interval - expected_interval) / max(expected_interval, 0.050)
                )
                if utility > dp[pulse_index][candidate_index]:
                    dp[pulse_index][candidate_index] = utility
                    previous[pulse_index][candidate_index] = prior_index
    final_index = max(range(len(candidates)), key=lambda index: dp[count - 1][index])
    if dp[count - 1][final_index] < -1.0e20:
        return [center for center, _ in candidates[:count]]
    chosen = []
    for pulse_index in range(count - 1, -1, -1):
        chosen.append(candidates[final_index][0])
        final_index = previous[pulse_index][final_index]
    return list(reversed(chosen))


cases = []
for case_dir in sorted(RUN_ROOT.iterdir()):
    required = {
        "grade": case_dir / "grade.json",
        "occupancy": case_dir / "occupancy_frames.csv",
        "classes": case_dir / "phone_class_frames.csv",
        "landmarks": case_dir / "audio_landmark_frames.csv",
        "strict": case_dir / "streaming_evidence_observations.csv",
        "regions": case_dir / "speech_regions.csv",
        "syllables": case_dir / "planned_syllables.csv",
        "gold_phones": GOLD_ROOT / case_dir.name / "phones.csv",
        "gold_speech": GOLD_ROOT / case_dir.name / "speech.csv",
    }
    if not all(path.exists() for path in required.values()):
        continue
    grade = json.loads(required["grade"].read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    observed_regions = rows(required["regions"])
    gold_regions = rows(required["gold_speech"])
    if len(observed_regions) != len(gold_regions):
        continue
    counts: dict[int, int] = {}
    for syllable in rows(required["syllables"]):
        region_index = int(syllable["speech_region_index"])
        counts[region_index] = counts.get(region_index, 0) + 1
    occupancy = rows(required["occupancy"])
    classes = rows(required["classes"])
    landmarks = rows(required["landmarks"])
    if not (len(occupancy) == len(classes) == len(landmarks)):
        continue
    pulse_candidates = dense_acoustic_candidates(occupancy, classes, landmarks)
    strict_pulses = [
        (float(row["center"]), 2.0 + float(row["score"]))
        for row in rows(required["strict"])
        if row["type"] == "pulse"
    ]
    gold_phones = rows(required["gold_phones"])
    regions = []
    for index, (observed, gold) in enumerate(zip(observed_regions, gold_regions)):
        gold_start, gold_end = float(gold["start"]), float(gold["end"])
        targets = [
            (float(phone["start"]), float(phone["end"]), (float(phone["start"]) + float(phone["end"])) * 0.5)
            for phone in gold_phones
            if phone["phone"].rstrip("012") in VOWELS
            and gold_start - 0.001 <= (float(phone["start"]) + float(phone["end"])) * 0.5 <= gold_end + 0.001
        ]
        start, end = float(observed["start"]), float(observed["end"])
        candidates = [(center, score) for center, score in pulse_candidates if start <= center <= end]
        for strict_center, strict_score in strict_pulses:
            if not start <= strict_center <= end:
                continue
            nearby = [
                candidate_index
                for candidate_index, (center, _) in enumerate(candidates)
                if abs(center - strict_center) <= 0.025
            ]
            if nearby:
                best_nearby = min(nearby, key=lambda candidate_index: abs(candidates[candidate_index][0] - strict_center))
                candidates[best_nearby] = (strict_center, strict_score)
            else:
                candidates.append((strict_center, strict_score))
        regions.append((start, end, counts.get(index, 0), candidates, targets))
    evaluation = int(hashlib.sha1(source_group(case_dir.name).encode()).hexdigest()[:8], 16) % 5 == 0
    cases.append((regions, evaluation))


def evaluate(parameters: tuple[float, float, float], evaluation_only: bool) -> dict[str, float]:
    score_weight, progress_weight, interval_weight = parameters
    observations = targets = matched = exact_counts = regions = count_error = 0
    clustered_targets = clustered_matches = 0
    for case_regions, evaluation in cases:
        if evaluation != evaluation_only:
            continue
        for start, end, expected_count, candidates, gold_targets in case_regions:
            selected = select_inventory(
                candidates,
                expected_count,
                start,
                end,
                score_weight,
                progress_weight,
                interval_weight,
            )
            pairs = sorted(
                (interval_error(center, target), selected_index, target_index)
                for selected_index, center in enumerate(selected)
                for target_index, target in enumerate(gold_targets)
                if interval_error(center, target) <= TOLERANCE_SEC
            )
            used_selected: set[int] = set()
            used_targets: set[int] = set()
            for _, selected_index, target_index in pairs:
                if selected_index in used_selected or target_index in used_targets:
                    continue
                used_selected.add(selected_index)
                used_targets.add(target_index)
            clustered = {
                index
                for index, target in enumerate(gold_targets)
                if (index > 0 and target[2] - gold_targets[index - 1][2] <= 0.150)
                or (index + 1 < len(gold_targets) and gold_targets[index + 1][2] - target[2] <= 0.150)
            }
            observations += len(selected)
            targets += len(gold_targets)
            matched += len(used_targets)
            clustered_targets += len(clustered)
            clustered_matches += len(clustered & used_targets)
            exact_counts += len(selected) == len(gold_targets)
            count_error += abs(len(selected) - len(gold_targets))
            regions += 1
    precision = matched / max(observations, 1)
    recall = matched / max(targets, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "count_ratio": observations / max(targets, 1),
        "exact_count": exact_counts / max(regions, 1),
        "count_mae": count_error / max(regions, 1),
        "cluster150_recall": clustered_matches / max(clustered_targets, 1),
    }


print(f"cases={len(cases)}")
best = None
for parameters in itertools.product((0.0, 0.25, 0.50, 1.0), (0.0, 0.25, 0.50, 1.0, 2.0), (0.0, 0.25, 0.50, 1.0, 2.0)):
    result = evaluate(parameters, False)
    objective = result["f1"] + 0.25 * result["cluster150_recall"] + 0.15 * result["exact_count"] - 0.10 * result["count_mae"]
    if best is None or objective > best[0]:
        best = (objective, parameters, result)
assert best is not None
print("best parameters", best[1])
print("train", best[2])
print("test ", evaluate(best[1], True))
