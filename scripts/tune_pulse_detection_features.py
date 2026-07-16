#!/usr/bin/env python3
"""Evaluate acoustic-only pulse detectors, including clustered-nucleus metrics."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
TOLERANCE_SEC = 0.100


@dataclass(frozen=True)
class Detector:
    name: str
    prominence: float = 0.10
    vowel_support: float = 0.35
    spacing_sec: float = 0.100
    valley: float = 0.035
    onset_threshold: float = 2.0
    flux_threshold: float = 2.0
    band_threshold: float = 2.0
    periodicity_threshold: float = 2.0
    subdivide_threshold: float = 2.0
    subdivide_min_gap_sec: float = 0.180


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def triangular(values: list[float], half_width: int) -> list[float]:
    result: list[float] = []
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


def local_maximum(values: list[float], index: int, radius: int = 1) -> bool:
    return all(
        values[index] >= values[other]
        for other in range(max(0, index - radius), min(len(values), index + radius + 1))
        if other != index
    )


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def positive_rate(values: list[float], lag: int = 2, scale: float = 0.10) -> list[float]:
    return [
        clamp01((values[index] - values[max(index - lag, 0)]) / scale)
        for index in range(len(values))
    ]


FEATURE_CACHE: dict[int, tuple[list[float], list[float], list[float], list[float], list[float], list[float]]] = {}


def extract_pulses(frames: list[dict[str, float]], detector: Detector) -> list[float]:
    if len(frames) < 24:
        return []
    cache_key = id(frames)
    cached = FEATURE_CACHE.get(cache_key)
    if cached is None:
        energy = [
            frame["rms"] * (0.45 + frame["vowel"] * 0.40 + frame["periodicity"] * 0.15)
            for frame in frames
        ]
        medium = triangular(energy, 4)
        slow = triangular(energy, 6)
        sonority = triangular([
            frame["rms"] * (0.35 + frame["vowel"] * 0.45 + frame["periodicity"] * 0.20)
            for frame in frames
        ], 2)
        band_sonority = triangular([
            frame["rms"]
            * (0.15 + frame["rich_low"] * 0.45 + frame["rich_mid"] * 0.40)
            * (0.45 + frame["periodicity"] * 0.55)
            for frame in frames
        ], 2)
        envelope_rate = triangular(positive_rate(sonority, 2, 0.08), 1)
        band_rate = triangular(positive_rate(band_sonority, 2, 0.07), 1)
        flux_onset = triangular([
            clamp01(frame["rich_flux"] / 0.30)
            * max(frame["vowel"], frame["periodicity"])
            for frame in frames
        ], 1)
        periodicity_peak = triangular([frame["periodicity"] for frame in frames], 2)
        cached = (medium, slow, envelope_rate, band_rate, flux_onset, periodicity_peak)
        FEATURE_CACHE[cache_key] = cached
    medium, slow, envelope_rate, band_rate, flux_onset, periodicity_peak = cached

    def prominence(envelope: list[float], index: int, inner: int) -> float:
        center = envelope[index]
        if center < envelope[index - 1] or center < envelope[index + 1]:
            return 0.0
        left = min(envelope[index - 10 : index - inner + 1])
        right = min(envelope[index + inner : index + 11])
        return clamp01(min(center - left, center - right) / 0.12)

    raw: list[tuple[int, float, str]] = []
    for index in range(10, len(frames) - 10):
        frame = frames[index]
        support = clamp01(frame["vowel"] * 0.60 + frame["periodicity"] * 0.40)
        base_prominence = max(prominence(medium, index, 4), prominence(slow, index, 6) * 1.05)
        if base_prominence >= detector.prominence and frame["speech"] >= 0.18 and support >= detector.vowel_support:
            raw.append((index, base_prominence, "peak"))

        future_support = max(
            clamp01(frames[next_index]["vowel"] * 0.60 + frames[next_index]["periodicity"] * 0.40)
            for next_index in range(index, min(index + 7, len(frames)))
        )
        if frame["speech"] < 0.14 or future_support < 0.30:
            continue
        if detector.onset_threshold <= 1.0 and local_maximum(envelope_rate, index, 2):
            if envelope_rate[index] >= detector.onset_threshold:
                raw.append((index, envelope_rate[index] * 0.88, "peak_rate"))
        if detector.flux_threshold <= 1.0 and local_maximum(flux_onset, index, 2):
            if flux_onset[index] >= detector.flux_threshold:
                raw.append((index, flux_onset[index] * 0.78, "flux"))
        if detector.band_threshold <= 1.0 and local_maximum(band_rate, index, 2):
            if band_rate[index] >= detector.band_threshold:
                raw.append((index, band_rate[index] * 0.82, "band_rate"))
        if detector.periodicity_threshold <= 1.0 and local_maximum(periodicity_peak, index, 2):
            left = min(periodicity_peak[index - 8 : index - 2])
            right = min(periodicity_peak[index + 3 : index + 9])
            periodicity_prominence = clamp01(min(
                periodicity_peak[index] - left,
                periodicity_peak[index] - right,
            ) / 0.20)
            if periodicity_prominence >= detector.periodicity_threshold:
                raw.append((index, periodicity_prominence * 0.86, "periodicity"))

    raw.sort(key=lambda item: (item[0], -item[1]))
    pulses: list[tuple[int, float, str]] = []
    for index, score, source in raw:
        if pulses and index == pulses[-1][0]:
            if score > pulses[-1][1]:
                pulses[-1] = (index, score, source)
            continue
        if pulses:
            prior_index, prior_score, _ = pulses[-1]
            spacing = frames[index]["center"] - frames[prior_index]["center"]
            floor = min(medium[prior_index : index + 1])
            distinct = spacing >= 0.060 and min(medium[prior_index], medium[index]) - floor >= detector.valley
            transition_support = max(
                max(envelope_rate[sample], flux_onset[sample], band_rate[sample])
                for sample in range(prior_index + 1, index + 1)
            )
            distinct = distinct or (spacing >= 0.070 and transition_support >= 0.48)
            if spacing < detector.spacing_sec and not distinct:
                if score > prior_score:
                    pulses[-1] = (index, score, source)
                continue
        pulses.append((index, score, source))

    if detector.subdivide_threshold <= 1.0 and len(pulses) >= 2:
        split_signal = [
            math.sqrt(band_rate[index] * max(envelope_rate[index], flux_onset[index]))
            for index in range(len(frames))
        ]
        changed = True
        while changed:
            changed = False
            for pulse_index in range(len(pulses) - 1):
                left = pulses[pulse_index][0]
                right = pulses[pulse_index + 1][0]
                if frames[right]["center"] - frames[left]["center"] < detector.subdivide_min_gap_sec:
                    continue
                search_start = left + 7
                search_end = right - 6
                if search_start >= search_end:
                    continue
                best = max(range(search_start, search_end + 1), key=lambda index: split_signal[index])
                support = clamp01(frames[best]["vowel"] * 0.60 + frames[best]["periodicity"] * 0.40)
                if (split_signal[best] < detector.subdivide_threshold
                    or not local_maximum(split_signal, best, 2)
                    or frames[best]["speech"] < 0.18
                    or support < 0.30):
                    continue
                pulses.insert(pulse_index + 1, (best, split_signal[best] * 0.85, "subdivision"))
                changed = True
                break
    return [frames[index]["center"] for index, _, _ in pulses]


def source_group(name: str) -> str:
    match = re.search(r"Alfie_Line_([A-F0-9]+)", name)
    return match.group(1) if match else name


def interval_error(center: float, target: tuple[float, float, float]) -> float:
    start, end, _ = target
    if center < start:
        return start - center
    if center > end:
        return center - end
    return 0.0


cases: list[dict[str, object]] = []
for case_dir in sorted(RUN_ROOT.iterdir()):
    paths = {
        "occupancy": case_dir / "occupancy_frames.csv",
        "classes": case_dir / "phone_class_frames.csv",
        "landmarks": case_dir / "audio_landmark_frames.csv",
        "grade": case_dir / "grade.json",
        "phones": GOLD_ROOT / case_dir.name / "phones.csv",
        "speech": GOLD_ROOT / case_dir.name / "speech.csv",
    }
    if not all(path.exists() for path in paths.values()):
        continue
    grade = json.loads(paths["grade"].read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    occupancy = read_rows(paths["occupancy"])
    classes = read_rows(paths["classes"])
    landmarks = read_rows(paths["landmarks"])
    if not (len(occupancy) == len(classes) == len(landmarks)):
        continue
    frames = []
    for occupancy_row, class_row, landmark_row in zip(occupancy, classes, landmarks):
        frames.append({
            "center": float(occupancy_row["center"]),
            "rms": float(occupancy_row["rms_norm"]),
            "periodicity": float(occupancy_row["periodicity"]),
            "speech": float(occupancy_row["evidence"]),
            "vowel": float(class_row["vowel"]),
            "rich_low": float(landmark_row["rich_low"]),
            "rich_mid": float(landmark_row["rich_mid"]),
            "rich_flux": float(landmark_row["rich_flux"]),
        })
    phones = read_rows(paths["phones"])
    speech = read_rows(paths["speech"])
    regions = []
    all_targets: list[tuple[float, float, float]] = []
    for region in speech:
        start, end = float(region["start"]), float(region["end"])
        targets = []
        for phone in phones:
            phone_start, phone_end = float(phone["start"]), float(phone["end"])
            center = (phone_start + phone_end) * 0.5
            if phone["phone"].rstrip("012") in VOWELS and start - 0.001 <= center <= end + 0.001:
                targets.append((phone_start, phone_end, center))
                all_targets.append((phone_start, phone_end, center))
        regions.append((start, end, targets))
    evaluation = int(hashlib.sha1(source_group(case_dir.name).encode()).hexdigest()[:8], 16) % 5 == 0
    cases.append({"frames": frames, "regions": regions, "speech": speech, "targets": all_targets, "evaluation": evaluation})


def evaluate(detector: Detector, evaluation_only: bool) -> dict[str, float]:
    observations = targets = matched = 0
    exact_regions = regions = count_error = 0
    outside_speech = 0
    cluster_targets = {0.120: 0, 0.150: 0, 0.200: 0}
    cluster_matches = {0.120: 0, 0.150: 0, 0.200: 0}
    for case in cases:
        if bool(case["evaluation"]) != evaluation_only:
            continue
        pulses = extract_pulses(case["frames"], detector)  # type: ignore[arg-type]
        speech = case["speech"]  # type: ignore[assignment]
        outside_speech += sum(
            not any(float(region["start"]) - 0.030 <= pulse <= float(region["end"]) + 0.030 for region in speech)
            for pulse in pulses
        )
        for start, end, expected in case["regions"]:  # type: ignore[assignment]
            observed = [pulse for pulse in pulses if start - 0.001 <= pulse <= end + 0.001]
            candidates = sorted(
                (interval_error(pulse, target), pulse_index, target_index)
                for pulse_index, pulse in enumerate(observed)
                for target_index, target in enumerate(expected)
                if interval_error(pulse, target) <= TOLERANCE_SEC
            )
            used_pulses: set[int] = set()
            used_targets: set[int] = set()
            for _, pulse_index, target_index in candidates:
                if pulse_index in used_pulses or target_index in used_targets:
                    continue
                used_pulses.add(pulse_index)
                used_targets.add(target_index)
            for threshold in cluster_targets:
                clustered = {
                    index
                    for index, target in enumerate(expected)
                    if (index > 0 and target[2] - expected[index - 1][2] <= threshold)
                    or (index + 1 < len(expected) and expected[index + 1][2] - target[2] <= threshold)
                }
                cluster_targets[threshold] += len(clustered)
                cluster_matches[threshold] += len(clustered & used_targets)
            observations += len(observed)
            targets += len(expected)
            matched += len(used_targets)
            exact_regions += len(observed) == len(expected)
            count_error += abs(len(observed) - len(expected))
            regions += 1
    precision = matched / max(observations, 1)
    recall = matched / max(targets, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-9),
        "count_ratio": observations / max(targets, 1),
        "exact_count": exact_regions / max(regions, 1),
        "count_mae": count_error / max(regions, 1),
        "cluster120_recall": cluster_matches[0.120] / max(cluster_targets[0.120], 1),
        "cluster150_recall": cluster_matches[0.150] / max(cluster_targets[0.150], 1),
        "cluster200_recall": cluster_matches[0.200] / max(cluster_targets[0.200], 1),
        "outside_speech_rate": outside_speech / max(observations + outside_speech, 1),
    }


variants = [
    Detector("baseline"),
    Detector("peak_rate", onset_threshold=0.46),
    Detector("spectral_flux", flux_threshold=0.46),
    Detector("multiband_rate", band_threshold=0.46),
    Detector("periodicity_peak", periodicity_threshold=0.40),
    Detector("combined", onset_threshold=0.46, flux_threshold=0.46, band_threshold=0.46),
    Detector("guarded_subdivision", subdivide_threshold=0.46),
]
print(f"cases={len(cases)} train={sum(not bool(case['evaluation']) for case in cases)} heldout={sum(bool(case['evaluation']) for case in cases)}")
for detector in variants:
    print(detector.name, "train", evaluate(detector, False))
    print(detector.name, "test ", evaluate(detector, True))

best: tuple[float, Detector, dict[str, float]] | None = None
for onset, flux, band, spacing in itertools.product(
    (0.38, 0.46, 0.54, 2.0),
    (0.38, 0.46, 0.54, 2.0),
    (0.38, 0.46, 0.54, 2.0),
    (0.080, 0.090, 0.100),
):
    if onset > 1.0 and flux > 1.0 and band > 1.0:
        continue
    detector = Detector("search", spacing_sec=spacing, onset_threshold=onset, flux_threshold=flux, band_threshold=band)
    result = evaluate(detector, False)
    objective = (
        result["f1"]
        + 0.25 * result["cluster150_recall"]
        + 0.15 * result["exact_count"]
        - 0.10 * result["count_mae"]
        - 0.25 * result["outside_speech_rate"]
    )
    if best is None or objective > best[0]:
        best = (objective, detector, result)

assert best is not None
print("best", best[1])
print("best train", best[2])
print("best test ", evaluate(best[1], True))

best_subdivision: tuple[float, Detector, dict[str, float]] | None = None
for threshold, spacing, minimum_gap in itertools.product(
    (0.64, 0.70, 0.76, 0.82, 0.88),
    (0.090, 0.100, 0.110),
    (0.180, 0.220, 0.260, 0.300),
):
    detector = Detector(
        "subdivision_search",
        spacing_sec=spacing,
        subdivide_threshold=threshold,
        subdivide_min_gap_sec=minimum_gap,
    )
    result = evaluate(detector, False)
    objective = (
        result["f1"]
        + 0.30 * result["cluster150_recall"]
        + 0.20 * result["exact_count"]
        - 0.12 * result["count_mae"]
        - 0.25 * result["outside_speech_rate"]
    )
    if best_subdivision is None or objective > best_subdivision[0]:
        best_subdivision = (objective, detector, result)

assert best_subdivision is not None
print("best subdivision", best_subdivision[1])
print("best subdivision train", best_subdivision[2])
print("best subdivision test ", evaluate(best_subdivision[1], True))

best_core: tuple[float, Detector, dict[str, float]] | None = None
for prominence, vowel_support, spacing, valley in itertools.product(
    (0.04, 0.06, 0.08, 0.10, 0.12),
    (0.25, 0.30, 0.35, 0.40),
    (0.075, 0.085, 0.095, 0.105),
    (0.025, 0.035, 0.045),
):
    detector = Detector(
        "core_search",
        prominence=prominence,
        vowel_support=vowel_support,
        spacing_sec=spacing,
        valley=valley,
    )
    result = evaluate(detector, False)
    objective = (
        result["f1"]
        + 0.20 * result["cluster150_recall"]
        + 0.20 * result["exact_count"]
        - 0.12 * result["count_mae"]
        - 0.25 * result["outside_speech_rate"]
    )
    if best_core is None or objective > best_core[0]:
        best_core = (objective, detector, result)

assert best_core is not None
print("best core", best_core[1])
print("best core train", best_core[2])
print("best core test ", evaluate(best_core[1], True))
