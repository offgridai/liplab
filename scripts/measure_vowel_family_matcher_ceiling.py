#!/usr/bin/env python3
"""Measure the syllable-matcher ceiling from perfect broad vowel identity."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import tune_syllable_matcher_dp as matcher
import tune_vowel_nucleus_detector as vowel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
VOWEL_FAMILY = {
    "AA": "open", "AE": "open", "AH": "open", "AW": "open", "AY": "open",
    "EH": "front", "ER": "front", "EY": "front", "IH": "front", "IY": "front",
    "AO": "round", "OW": "round", "OY": "round", "UH": "round", "UW": "round",
}
PARAMETERS = (0.25, 0.15, 0.05, 0.60, 2.00, 0.0)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def augment(cases, weight: float, probabilities=None):
    result = copy.deepcopy(cases)
    for case_index, (name, targets, pulses, components) in enumerate(result):
        expected = rows(ROOT / "outputs" / "runs" / "latest" / name / "expected_phones.csv")
        for target in targets:
            target["vowel_family"] = VOWEL_FAMILY.get(expected[target["nucleus"]]["base_phone"], "other")
        observed_families = []
        for pulse in pulses:
            center = float(pulse["center"])
            if probabilities is not None:
                observed_families.append(probabilities.get((name, round(center, 6))))
                continue
            actual = None
            best_error = 0.100001
            for target in targets:
                error = max(target["gold_start"] - center, center - target["gold_end"], 0.0)
                if error < best_error:
                    actual = target["vowel_family"]
                    best_error = error
            observed_families.append(actual)
        for target_index, target in enumerate(targets):
            for pulse_index, actual in enumerate(observed_families):
                timing, evidence, unexpected = components[target_index][pulse_index]
                if isinstance(actual, dict):
                    support = actual.get(target["vowel_family"], 0.0)
                    conflict = max(
                        (value for family, value in actual.items() if family != target["vowel_family"]),
                        default=0.0,
                    )
                else:
                    support = 1.0 if actual == target["vowel_family"] else 0.0
                    conflict = 1.0 if actual is not None and actual != target["vowel_family"] else 0.0
                components[target_index][pulse_index] = (
                    timing,
                    evidence + weight * support,
                    unexpected + weight * conflict,
                )
        result[case_index] = (name, targets, pulses, components)
    return result


def main() -> None:
    cases = matcher.load_cases()
    features, labels, held_out_samples, metadata = vowel.load_samples()
    probability_sets = {}
    for model_name, columns in {
        "existing": slice(0, vowel.BASE_FEATURE_COUNT),
        "combined": slice(0, vowel.BASE_FEATURE_COUNT + vowel.SPECTRAL_FEATURE_COUNT),
    }.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=2000, class_weight="balanced"),
        )
        model.fit(features[~held_out_samples, columns], labels[~held_out_samples])
        probability_sets[model_name] = {
            (name, round(center, 6)): dict(zip(model.classes_, values))
            for (name, center), values in zip(metadata, model.predict_proba(features[:, columns]))
        }
    report = {}
    for weight in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5):
        adjusted = augment(cases, weight)
        heldout = [case for case in adjusted if vowel.is_held_out(case[0])]
        report[str(weight)] = matcher.score(heldout, PARAMETERS)
    report["learned"] = {}
    for model_name, probabilities in probability_sets.items():
        learned = {}
        for weight in (0.20, 0.40, 0.60, 0.80):
            adjusted = augment(cases, weight, probabilities)
            heldout = [case for case in adjusted if vowel.is_held_out(case[0])]
            learned[str(weight)] = matcher.score(heldout, PARAMETERS)
        report["learned"][model_name] = learned
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
