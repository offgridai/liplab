#!/usr/bin/env python3
"""Describe the MFA nuclei missed by the strict runtime pulse detector."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


RUN_ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


targets = matched = clustered = clustered_matched = 0
missed_vowels: Counter[str] = Counter()
missed_pairs: Counter[str] = Counter()
missed_words: Counter[str] = Counter()
matched_vowels: Counter[str] = Counter()

for case_dir in sorted(RUN_ROOT.iterdir()):
    gold_dir = GOLD_ROOT / case_dir.name
    required = [case_dir / "grade.json", case_dir / "streaming_evidence_observations.csv", gold_dir / "phones.csv"]
    if not all(path.exists() for path in required):
        continue
    grade = json.loads(required[0].read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    phones = rows(required[2])
    nuclei = []
    for phone in phones:
        base = phone["phone"].rstrip("012")
        if base not in VOWELS:
            continue
        start, end = float(phone["start"]), float(phone["end"])
        nuclei.append((start, end, 0.5 * (start + end), base, phone.get("word", "")))
    pulses = [
        float(row["center"])
        for row in rows(required[1])
        if row["type"] == "pulse"
    ]
    candidates = sorted(
        (0.0 if start <= pulse <= end else min(abs(pulse - start), abs(pulse - end)), pulse_index, target_index)
        for pulse_index, pulse in enumerate(pulses)
        for target_index, (start, end, _, _, _) in enumerate(nuclei)
        if start - 0.100 <= pulse <= end + 0.100
    )
    used_pulses: set[int] = set()
    used_targets: set[int] = set()
    for _, pulse_index, target_index in candidates:
        if pulse_index in used_pulses or target_index in used_targets:
            continue
        used_pulses.add(pulse_index)
        used_targets.add(target_index)
    for index, (_, _, center, base, word) in enumerate(nuclei):
        is_clustered = (
            (index > 0 and center - nuclei[index - 1][2] <= 0.150)
            or (index + 1 < len(nuclei) and nuclei[index + 1][2] - center <= 0.150)
        )
        targets += 1
        matched += index in used_targets
        if not is_clustered:
            continue
        clustered += 1
        if index in used_targets:
            clustered_matched += 1
            matched_vowels[base] += 1
            continue
        missed_vowels[base] += 1
        missed_words[word.lower()] += 1
        neighbor = min(
            (other for other in (index - 1, index + 1) if 0 <= other < len(nuclei)),
            key=lambda other: abs(nuclei[other][2] - center),
        )
        missed_pairs[f"{nuclei[neighbor][3]}-{base}"] += 1

print(f"all recall={matched / targets:.3f} ({matched}/{targets})")
print(f"clustered150 recall={clustered_matched / clustered:.3f} ({clustered_matched}/{clustered})")
print("missed vowels", missed_vowels.most_common(12))
print("missed neighbor pairs", missed_pairs.most_common(15))
print("missed words", missed_words.most_common(15))
