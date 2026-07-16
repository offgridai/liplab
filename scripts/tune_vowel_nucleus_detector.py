#!/usr/bin/env python3
"""Measure broad vowel-nucleus classification at detected pulse centers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
TOLERANCE_SEC = 0.100
VOWEL_FAMILY = {
    "AA": "open", "AE": "open", "AH": "open", "AW": "open", "AY": "open",
    "EH": "front", "ER": "front", "EY": "front", "IH": "front", "IY": "front",
    "AO": "round", "OW": "round", "OY": "round", "UH": "round", "UW": "round",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_group(name: str) -> str:
    match = re.search(r"Alfie_Line_([A-F0-9]+)", name)
    return match.group(1) if match else name


def is_held_out(name: str) -> bool:
    digest = int(hashlib.sha1(source_group(name).encode()).hexdigest()[:8], 16)
    return digest % 5 == 0


FEATURES = (
    "vowel_open", "vowel_front", "vowel_round", "vowel", "voiced", "sonorant",
    "low_tilt", "high_tilt", "spectral_change", "energy_change",
    "rich_low", "rich_mid", "rich_high", "rich_centroid", "rich_rolloff",
    "rich_flatness", "rich_flux", "rich_periodicity", "rms_norm", "periodicity",
    "low_band", "mid_band", "high_band", "centroid",
)
GOERTZEL_FREQUENCIES = np.asarray(
    [250.0, 400.0, 630.0, 1000.0, 1400.0, 2000.0, 2800.0, 3600.0, 4500.0, 5400.0, 6300.0, 7200.0]
)
BASE_FEATURE_COUNT = len(FEATURES) * 4
SPECTRAL_FEATURE_COUNT = len(GOERTZEL_FREQUENCIES) * 4
MODEL_PATH = Path("offgrid_dropin/Private/Lipsync/OffgridAIVowelNucleusModel.inl")


def spectral_window_features(samples: np.ndarray, sample_rate: int, center: float) -> list[float]:
    distributions = []
    window_size = max(round(0.030 * sample_rate), 16)
    for offset in (-0.030, -0.010, 0.010, 0.030):
        midpoint = round((center + offset) * sample_rate)
        begin = midpoint - window_size // 2
        segment = np.zeros(window_size, dtype=float)
        source_begin = max(begin, 0)
        source_end = min(begin + window_size, len(samples))
        if source_end > source_begin:
            segment[source_begin - begin:source_end - begin] = samples[source_begin:source_end]
        segment *= np.hamming(window_size)
        spectrum = np.abs(np.fft.rfft(segment)) ** 2
        bins = np.minimum(
            np.rint(GOERTZEL_FREQUENCIES * window_size / sample_rate).astype(int),
            len(spectrum) - 1,
        )
        energy = spectrum[bins] + 1.0e-12
        distributions.append(energy / np.sum(energy))
    matrix = np.asarray(distributions)
    return np.concatenate((
        matrix.mean(axis=0), matrix.max(axis=0), matrix.min(axis=0), matrix[-1] - matrix[0]
    )).tolist()


def aggregate_window(
    center: float,
    phone_frames: list[dict[str, str]],
    landmark_frames: list[dict[str, str]],
    occupancy_frames: list[dict[str, str]],
) -> list[float]:
    indices = [
        index for index, frame in enumerate(phone_frames)
        if abs(float(frame["center"]) - center) <= 0.045
    ]
    if not indices:
        indices = [min(range(len(phone_frames)), key=lambda i: abs(float(phone_frames[i]["center"]) - center))]
    feature_rows: list[list[float]] = []
    for index in indices:
        combined = dict(phone_frames[index])
        combined.update(landmark_frames[index])
        combined.update(occupancy_frames[index])
        feature_rows.append([float(combined[name]) for name in FEATURES])
    matrix = np.asarray(feature_rows, dtype=float)
    # Midpoint shape, local extrema, and within-pulse trajectory are all cheap streaming features.
    return np.concatenate((
        matrix.mean(axis=0),
        matrix.max(axis=0),
        matrix.min(axis=0),
        matrix[-1] - matrix[0],
    )).tolist()


def load_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, float]]]:
    feature_rows: list[list[float]] = []
    labels: list[str] = []
    held_out: list[bool] = []
    metadata: list[tuple[str, float]] = []
    for case_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        paths = {
            "grade": case_dir / "grade.json",
            "pulses": case_dir / "streaming_permissive_evidence_candidates.csv",
            "classes": case_dir / "phone_class_frames.csv",
            "landmarks": case_dir / "audio_landmark_frames.csv",
            "occupancy": case_dir / "occupancy_frames.csv",
            "phones": GOLD_ROOT / case_dir.name / "phones.csv",
            "wav": Path("inputs/wav") / f"{case_dir.name}.wav",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        grade = json.loads(paths["grade"].read_text(encoding="utf-8"))
        if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
            continue
        phones = rows(paths["phones"])
        targets = []
        for phone in phones:
            base = phone["phone"].rstrip("012")
            if base in VOWEL_FAMILY:
                targets.append((float(phone["start"]), float(phone["end"]), VOWEL_FAMILY[base]))
        pulse_rows = [row for row in rows(paths["pulses"]) if row["type"] == "pulse"]
        phone_frames = rows(paths["classes"])
        landmark_frames = rows(paths["landmarks"])
        occupancy_frames = rows(paths["occupancy"])
        sample_rate, wav_samples = wavfile.read(paths["wav"])
        wav_samples = np.asarray(wav_samples, dtype=float)
        if wav_samples.ndim > 1:
            wav_samples = wav_samples.mean(axis=1)
        peak = np.max(np.abs(wav_samples))
        if peak > 0.0:
            wav_samples /= peak
        if not (len(phone_frames) == len(landmark_frames) == len(occupancy_frames)):
            continue
        candidates = sorted(
            (max(start - float(pulse["center"]), float(pulse["center"]) - end, 0.0), pulse_index, target_index)
            for pulse_index, pulse in enumerate(pulse_rows)
            for target_index, (start, end, _) in enumerate(targets)
            if max(start - float(pulse["center"]), float(pulse["center"]) - end, 0.0) <= TOLERANCE_SEC
        )
        used_pulses: set[int] = set()
        used_targets: set[int] = set()
        for _, pulse_index, target_index in candidates:
            if pulse_index in used_pulses or target_index in used_targets:
                continue
            used_pulses.add(pulse_index)
            used_targets.add(target_index)
            center = float(pulse_rows[pulse_index]["center"])
            feature_rows.append(
                aggregate_window(center, phone_frames, landmark_frames, occupancy_frames)
                + spectral_window_features(wav_samples, sample_rate, center)
            )
            labels.append(targets[target_index][2])
            held_out.append(is_held_out(case_dir.name))
            metadata.append((case_dir.name, center))
    return np.asarray(feature_rows), np.asarray(labels), np.asarray(held_out), metadata


def score(name: str, truth: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    report = classification_report(truth, predicted, output_dict=True, zero_division=0)
    return {
        "name": name,
        "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "by_class": {label: report[label] for label in sorted(VOWEL_FAMILY.values())},
        "confusion": confusion_matrix(truth, predicted, labels=["open", "front", "round"]).tolist(),
    }


def raw_linear_model(model) -> tuple[np.ndarray, np.ndarray]:
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    weights = logistic.coef_ / scaler.scale_[None, :]
    intercepts = logistic.intercept_ - np.sum(logistic.coef_ * scaler.mean_[None, :] / scaler.scale_[None, :], axis=1)
    return weights, intercepts


def write_model(model) -> None:
    weights, intercepts = raw_linear_model(model)
    classes = list(model.named_steps["logisticregression"].classes_)
    lines = [
        "// Generated by scripts/tune_vowel_nucleus_detector.py --write-model.",
        f"static constexpr int32 GOffgridAIVowelFeatureCount = {weights.shape[1]};",
        "static constexpr float GOffgridAIVowelIntercepts[3] = {",
        "    " + ", ".join(f"{intercepts[classes.index(label)]:.9g}f" for label in ("open", "front", "round")),
        "};",
        "static constexpr float GOffgridAIVowelWeights[3][GOffgridAIVowelFeatureCount] = {",
    ]
    for label in ("open", "front", "round"):
        row = weights[classes.index(label)]
        lines.append("    {")
        for begin in range(0, len(row), 8):
            lines.append("        " + ", ".join(f"{value:.9g}f" for value in row[begin:begin + 8]) + ",")
        lines.append("    },")
    lines.extend(("};", ""))
    MODEL_PATH.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-model", action="store_true")
    args = parser.parse_args()
    features, labels, held_out, _ = load_samples()
    train_x, test_x = features[~held_out], features[held_out]
    train_y, test_y = labels[~held_out], labels[held_out]

    baseline_scores = features[:, :len(FEATURES)]
    baseline_labels = np.asarray(["open", "front", "round"])[
        np.argmax(baseline_scores[:, [0, 1, 2]], axis=1)
    ]
    score_columns = np.asarray([
        block * len(FEATURES) + feature
        for block in range(4)
        for feature in (0, 1, 2)
    ])
    compact_features = (0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23)
    compact_columns = np.asarray([
        block * len(FEATURES) + feature
        for block in range(4)
        for feature in compact_features
    ])
    spectral = features[:, BASE_FEATURE_COUNT:].reshape((-1, 4, len(GOERTZEL_FREQUENCIES)))
    grouped_spectral = np.stack((
        spectral[:, :, 0:2].sum(axis=2),
        spectral[:, :, 2],
        spectral[:, :, 3:5].sum(axis=2),
        spectral[:, :, 5],
        spectral[:, :, 6:8].sum(axis=2),
        spectral[:, :, 8:].sum(axis=2),
    ), axis=2).reshape((len(features), -1))
    variant_features = {
        "three_scores": features[:, score_columns],
        "compact_window": features[:, compact_columns],
        "existing_window": features[:, :BASE_FEATURE_COUNT],
        "goertzel_window": features[:, BASE_FEATURE_COUNT:],
        "six_bands": grouped_spectral,
        "existing_plus_six_bands": np.concatenate((features[:, :BASE_FEATURE_COUNT], grouped_spectral), axis=1),
        "combined_window": features,
    }
    variants = {}
    for name, variant in variant_features.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, class_weight="balanced"),
        )
        model.fit(variant[~held_out], train_y)
        variants[name] = score(name, test_y, model.predict(variant[held_out]))
    result = {
        "samples": len(labels),
        "training_samples": int(np.sum(~held_out)),
        "held_out_samples": int(np.sum(held_out)),
        "baseline": score("existing_argmax", test_y, baseline_labels[held_out]),
        "variants": variants,
    }
    if args.write_model:
        final_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, class_weight="balanced"),
        )
        final_model.fit(features, labels)
        write_model(final_model)
        result["model_path"] = str(MODEL_PATH)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
