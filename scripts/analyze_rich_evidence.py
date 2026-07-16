#!/usr/bin/env python3
"""Fit small held-out linear probes to the advisory acoustic feature surface."""

from __future__ import annotations

import csv
import math
import pathlib
import random
import re


FEATURES = (
    "mbp",
    "fv",
    "w",
    "chjjsh",
    "round",
    "rich_low",
    "rich_mid",
    "rich_high",
    "rich_centroid",
    "rich_rolloff",
    "rich_flatness",
    "rich_flux",
    "rich_periodicity",
)
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
FV = {"F", "V"}
SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}
MBP = {"M", "B", "P"}
GLIDES = {"W", "Y"}
ROUND = {"AO", "OW", "OY", "UH", "UW"}


def phone_base(phone: str) -> str:
    return re.sub(r"\d+$", "", phone.upper())


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def load_rows(root: pathlib.Path, family: set[str]) -> tuple[list[tuple[list[float], int]], list[tuple[list[float], int]]]:
    train: list[tuple[list[float], int]] = []
    test: list[tuple[list[float], int]] = []
    output_root = root / "outputs" / "runs" / "latest"
    gold_root = root / "inputs" / "gold"
    for case_dir in sorted(output_root.glob("case_*")):
        frames_path = case_dir / "audio_landmark_frames.csv"
        phones_path = gold_root / case_dir.name / "phones.csv"
        if not frames_path.exists() or not phones_path.exists():
            continue
        match = re.match(r"case_(\d+)", case_dir.name)
        if not match:
            continue
        is_test = int(match.group(1)) % 5 == 0
        phones = []
        with phones_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                phones.append((float(row["start"]), float(row["end"]), phone_base(row["phone"])))
        phone_index = 0
        with frames_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                center = float(row["center"])
                while phone_index + 1 < len(phones) and center >= phones[phone_index][1]:
                    phone_index += 1
                if phone_index >= len(phones):
                    continue
                start, end, phone = phones[phone_index]
                if center < start or center >= end:
                    continue
                values = [float(row[name]) for name in FEATURES]
                (test if is_test else train).append((values, int(phone in family)))
    return train, test


def balanced_sample(rows: list[tuple[list[float], int]], seed: int) -> list[tuple[list[float], int]]:
    positives = [row for row in rows if row[1]]
    negatives = [row for row in rows if not row[1]]
    random.Random(seed).shuffle(negatives)
    return positives + negatives[: min(len(negatives), len(positives) * 4)]


def fit_probe(rows: list[tuple[list[float], int]]) -> tuple[list[float], list[float], list[float]]:
    count = len(rows)
    means = [sum(row[0][i] for row in rows) / count for i in range(len(FEATURES))]
    scales = [
        max(math.sqrt(sum((row[0][i] - means[i]) ** 2 for row in rows) / count), 1.0e-4)
        for i in range(len(FEATURES))
    ]
    weights = [0.0] * (len(FEATURES) + 1)
    positive_count = sum(row[1] for row in rows)
    negative_count = count - positive_count
    for iteration in range(180):
        gradient = [0.0] * len(weights)
        learning_rate = 0.18 / (1.0 + iteration * 0.012)
        for values, label in rows:
            normalized = [(values[i] - means[i]) / scales[i] for i in range(len(FEATURES))]
            probability = sigmoid(weights[0] + sum(weights[i + 1] * normalized[i] for i in range(len(FEATURES))))
            class_weight = count / (2.0 * (positive_count if label else negative_count))
            error = (probability - label) * class_weight
            gradient[0] += error
            for i, value in enumerate(normalized):
                gradient[i + 1] += error * value
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradient[i] / count
    return weights, means, scales


def score_rows(rows, weights, means, scales, threshold):
    tp = fp = fn = 0
    for values, label in rows:
        normalized = [(values[i] - means[i]) / scales[i] for i in range(len(FEATURES))]
        probability = sigmoid(weights[0] + sum(weights[i + 1] * normalized[i] for i in range(len(FEATURES))))
        predicted = probability >= threshold
        tp += int(predicted and label)
        fp += int(predicted and not label)
        fn += int(not predicted and label)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-9)
    return precision, recall, f1


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    for name, family in (
        ("fv", FV),
        ("sibilant", SIBILANTS),
        ("mbp", MBP),
        ("glide", GLIDES),
        ("round", ROUND),
        ("vowel", VOWELS),
    ):
        raw_train, raw_test = load_rows(root, family)
        train = balanced_sample(raw_train, 17)
        test = balanced_sample(raw_test, 23)
        weights, means, scales = fit_probe(train)
        best_threshold = max(
            (i / 100.0 for i in range(10, 91)),
            key=lambda threshold: score_rows(train, weights, means, scales, threshold)[2],
        )
        precision, recall, f1 = score_rows(test, weights, means, scales, best_threshold)
        print(f"{name}: threshold={best_threshold:.2f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
        ranked = sorted(zip(FEATURES, weights[1:]), key=lambda item: abs(item[1]), reverse=True)
        print("  " + " ".join(f"{feature}={weight:+.3f}" for feature, weight in ranked))
    for name, family in (("fv", FV), ("sibilant", SIBILANTS)):
        results = []
        for threshold_index in range(58, 96):
            threshold = threshold_index / 100.0
            target_count = observation_count = matched_count = 0
            for case_dir in sorted((root / "outputs" / "runs" / "latest").glob("case_*")):
                phones_path = root / "inputs" / "gold" / case_dir.name / "phones.csv"
                observations_path = case_dir / "streaming_evidence_observations.csv"
                if not phones_path.exists() or not observations_path.exists():
                    continue
                with phones_path.open(newline="", encoding="utf-8") as handle:
                    targets = [
                        (float(row["start"]), float(row["end"]))
                        for row in csv.DictReader(handle)
                        if phone_base(row["phone"]) in family
                    ]
                target_count += len(targets)
                used = [False] * len(targets)
                with observations_path.open(newline="", encoding="utf-8") as handle:
                    observations = [
                        float(row["center"])
                        for row in csv.DictReader(handle)
                        if row["type"] == name and float(row["score"]) >= threshold
                    ]
                observation_count += len(observations)
                for center in observations:
                    best_index = -1
                    best_error = 0.100001
                    for index, (start, end) in enumerate(targets):
                        if used[index]:
                            continue
                        error = start - center if center < start else center - end if center > end else 0.0
                        if error < best_error:
                            best_index = index
                            best_error = error
                    if best_index >= 0:
                        used[best_index] = True
                        matched_count += 1
            precision = matched_count / max(observation_count, 1)
            recall = matched_count / max(target_count, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-9)
            results.append((f1, threshold, precision, recall))
        f1, threshold, precision, recall = max(results)
        print(f"{name} event sweep: threshold={threshold:.2f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
