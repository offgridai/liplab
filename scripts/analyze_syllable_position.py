#!/usr/bin/env python3
"""Evaluate monotonic transcript-syllable position matching from streamed evidence."""

from __future__ import annotations

import csv
import itertools
import pathlib
import re


FAMILIES = {
    "mbp": {"M", "B", "P"},
    "fv": {"F", "V"},
    "glide": {"W", "Y"},
    "sibilant": {"S", "Z", "SH", "ZH", "CH", "JH"},
    "round": {"AO", "OW", "OY", "UH", "UW"},
}
RELIABILITY = {"mbp": 0.58, "fv": 0.66, "glide": 0.77, "sibilant": 0.92, "round": 0.66}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def phone_family(phone: str) -> str | None:
    for family, phones in FAMILIES.items():
        if phone in phones:
            return family
    return None


def load_case(case_dir: pathlib.Path, gold_root: pathlib.Path) -> tuple[list[dict], list[dict], list[dict]]:
    phones = read_csv(case_dir / "expected_phones.csv")
    # Gold phone_index is word-local; row order is the global CMU phone chain.
    gold = {index: row for index, row in enumerate(read_csv(gold_root / case_dir.name / "phones.csv"))}
    evidence_path = case_dir / "streaming_permissive_evidence_candidates.csv"
    if not evidence_path.exists():
        evidence_path = case_dir / "streaming_conditioned_evidence_observations.csv"
    evidence = read_csv(evidence_path)
    pulses = [row for row in evidence if row["type"] == "pulse"]
    landmarks = [row for row in evidence if row["type"] in RELIABILITY]

    targets: list[dict] = []
    phone_centers: dict[int, float] = {}
    elapsed = 0.0
    for phone_row_index, row in enumerate(phones):
        region = int(row["speech_region_index"])
        duration = float(row["weight_seconds"])
        start = elapsed
        elapsed += duration
        phone_centers[int(row["phone_index"])] = start + duration * 0.5
        next_word = phones[phone_row_index + 1]["word_index"] if phone_row_index + 1 < len(phones) else None
        if next_word != row["word_index"]:
            elapsed += float(row["pause_seconds_after_word"])
        if row["is_vowel"] != "1":
            continue
        index = int(row["phone_index"])
        gold_row = gold.get(index)
        if not gold_row:
            continue
        targets.append({
            "index": len(targets), "phone_index": index, "word_index": int(row["word_index"]),
            "region": region, "prior": start + duration * 0.5, "families": set(), "offsets": {},
            "gold_start": float(gold_row["start"]), "gold_end": float(gold_row["end"]),
            "gold_center": (float(gold_row["start"]) + float(gold_row["end"])) * 0.5,
        })

    # Attach each detectable phone to the nearest nucleus in its word.
    by_word: dict[int, list[dict]] = {}
    for target in targets:
        by_word.setdefault(target["word_index"], []).append(target)
    for row in phones:
        family = phone_family(row["base_phone"])
        word_targets = by_word.get(int(row["word_index"]), [])
        if family and word_targets:
            nearest = min(word_targets, key=lambda target: abs(target["phone_index"] - int(row["phone_index"])))
            nearest["families"].add(family)
            nearest["offsets"].setdefault(family, []).append(
                phone_centers[int(row["phone_index"])] - nearest["prior"]
            )

    observations = [{
        "center": float(row["center"]), "decision": float(row["decision"]), "score": float(row["score"]),
        "nearby": {
            family: max((float(item["score"]) for item in landmarks
                         if item["type"] == family and abs(float(item["center"]) - float(row["center"])) <= 0.160),
                        default=0.0)
            for family in RELIABILITY
        },
        "candidates": {
            family: [(float(item["center"]), float(item["score"])) for item in landmarks
                     if item["type"] == family and abs(float(item["center"]) - float(row["center"])) <= 0.350]
            for family in RELIABILITY
        },
    } for row in pulses]
    regions = read_csv(case_dir / "speech_regions.csv")
    return targets, observations, regions


def match_region(targets: list[dict], observations: list[dict], timing_weight: float,
                 landmark_weight: float, skip_target: float, skip_pulse: float) -> list[tuple[int, int, float]]:
    if not targets or not observations:
        return []
    prior_first, prior_last = targets[0]["prior"], targets[-1]["prior"]
    audio_first, audio_last = observations[0]["center"], observations[-1]["center"]
    prior_span = max(prior_last - prior_first, 0.001)
    audio_span = max(audio_last - audio_first, 0.001)
    rows, cols = len(targets) + 1, len(observations) + 1
    dp = [[-1.0e9] * cols for _ in range(rows)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * cols for _ in range(rows)]
    dp[0][0] = 0.0
    for i in range(rows):
        for j in range(cols):
            score = dp[i][j]
            if score <= -1.0e8:
                continue
            if i < len(targets) and score - skip_target > dp[i + 1][j]:
                dp[i + 1][j] = score - skip_target
                back[i + 1][j] = (i, j, "target")
            if j < len(observations) and score - skip_pulse > dp[i][j + 1]:
                dp[i][j + 1] = score - skip_pulse
                back[i][j + 1] = (i, j, "pulse")
            if i < len(targets) and j < len(observations):
                target, observation = targets[i], observations[j]
                prior_progress = (target["prior"] - prior_first) / prior_span
                audio_progress = (observation["center"] - audio_first) / audio_span
                timing = max(0.0, 1.0 - abs(prior_progress - audio_progress) / 0.35)
                expected = target["families"]
                evidence = 0.0
                for family in expected:
                    family_score = 0.0
                    for offset in target["offsets"].get(family, (0.0,)):
                        expected_center = observation["center"] + offset
                        for center, candidate_score in observation["candidates"][family]:
                            proximity = max(0.0, 1.0 - abs(center - expected_center) / 0.120)
                            family_score = max(family_score, candidate_score * proximity)
                    evidence += RELIABILITY[family] * family_score
                evidence /= max(sum(RELIABILITY[family] for family in expected), 1.0)
                unexpected = max((RELIABILITY[family] * value for family, value in observation["nearby"].items()
                                  if family not in expected), default=0.0)
                match = 0.45 + timing_weight * timing + landmark_weight * (evidence - unexpected * 0.20)
                if score + match > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = score + match
                    back[i + 1][j + 1] = (i, j, "match")
    i, j = len(targets), len(observations)
    matches = []
    while i or j:
        step = back[i][j]
        if step is None:
            break
        pi, pj, kind = step
        if kind == "match":
            matches.append((pi, pj, dp[i][j] - dp[pi][pj]))
        i, j = pi, pj
    return list(reversed(matches))


def evaluate(cases: list[tuple[list[dict], list[dict], list[dict]]], params: tuple[float, float, float, float]) -> dict[str, float]:
    assigned = correct = target_count = exact = within_one = word_correct = 0
    position_error = 0.0
    eligible_cases = 0
    for targets, observations, observed_regions in cases:
        del observed_regions
        eligible_cases += 1
        target_count += len(targets)
        for ti, oi, _ in match_region(targets, observations, *params):
            target, observation = targets[ti], observations[oi]
            assigned += 1
            error = 0.0 if target["gold_start"] <= observation["center"] <= target["gold_end"] else min(
                abs(observation["center"] - target["gold_start"]),
                abs(observation["center"] - target["gold_end"]),
            )
            correct += int(error <= 0.100)
            actual = min(targets, key=lambda candidate: abs(candidate["gold_center"] - observation["center"]))
            delta = abs(target["index"] - actual["index"])
            position_error += delta
            exact += int(delta == 0)
            within_one += int(delta <= 1)
            word_correct += int(target["word_index"] == actual["word_index"])
    return {
        "precision": correct / max(assigned, 1), "recall": correct / max(target_count, 1),
        "exact": exact / max(assigned, 1), "within1": within_one / max(assigned, 1),
        "word": word_correct / max(assigned, 1), "mae": position_error / max(assigned, 1),
        "assigned": assigned, "targets": target_count, "eligible_cases": eligible_cases,
    }


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    run_root, gold_root = root / "outputs" / "runs" / "latest", root / "inputs" / "gold"
    train, test = [], []
    for case_dir in sorted(run_root.glob("case_*")):
        if not (case_dir / "expected_phones.csv").exists() or not (gold_root / case_dir.name / "phones.csv").exists():
            continue
        match = re.match(r"case_(\d+)", case_dir.name)
        case = load_case(case_dir, gold_root)
        (test if match and int(match.group(1)) % 5 == 0 else train).append(case)
    candidates = list(itertools.product((0.0, 0.20, 0.40), (0.0, 0.35, 0.70, 1.0), (0.35, 0.55, 0.75), (0.25, 0.45, 0.65)))
    best = max(candidates, key=lambda params: evaluate(train, params)["exact"] + evaluate(train, params)["recall"] * 0.5)
    print("params", best)
    print("train", evaluate(train, best))
    print("test ", evaluate(test, best))
    pulse_only = (best[0], 0.0, best[2], best[3])
    print("pulse-only test", evaluate(test, pulse_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
