#!/usr/bin/env python3
"""
Score text-viseme planner quality from a completed LipLab run.

This scorer uses CMUdict pronunciations when available. It compares planned
poses against the spoken phoneme sequence and stress pattern rather than
guessing expected visemes from spelling.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


VOWEL_POSES = {"03_Ee", "04_Ih", "05_Ay", "06_Eh", "07_Aa", "08_Ah", "09_Oh", "10_Or", "11_Oo", "18_Uh"}
SUPPORT_POSES = {"14_ChJjSh", "24_Tongue_Th", "04_Ih", "06_Eh", "18_Uh"}
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "in", "is", "it", "its", "of", "on", "or", "our", "she", "that", "the",
    "their", "them", "there", "they", "this", "to", "was", "we", "were", "with", "you", "your",
}

VOWEL_PHONES = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}

OVERRIDES = {
    "weve": ["W", "V"],
    "we've": ["W", "V"],
    "we'll": ["W", "IY1", "L"],
    "whatll": ["W", "AH1", "T", "AH0", "L"],
    "what'll": ["W", "AH1", "T", "AH0", "L"],
}


def norm_word(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def base_phone(phone: str) -> str:
    return re.sub(r"\d", "", phone)


def stress(phone: str) -> int | None:
    m = re.search(r"[012]", phone)
    return int(m.group(0)) if m else None


def is_vowel(phone: str) -> bool:
    return base_phone(phone) in VOWEL_PHONES


def load_cmudict(repo_root: Path) -> dict[str, list[str]]:
    path = repo_root / "third_party" / "cmudict" / "cmudict.dict"
    lexicon: dict[str, list[str]] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        embedded_path = repo_root / "core" / "src" / "Lipsync" / "OffgridAICmudictData.cpp"
        if not embedded_path.exists():
            return lexicon
        embedded = embedded_path.read_text(encoding="latin-1")
        chunks = re.findall(r'R"CMU\((.*?)\)CMU"', embedded, flags=re.S)
        if not chunks:
            return lexicon
        text = "".join(chunks)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0].lower())
        lexicon.setdefault(word, parts[1:])
    return lexicon


def lookup_pron(word: str, lexicon: dict[str, list[str]]) -> list[str] | None:
    w = norm_word(word)
    if w in OVERRIDES:
        return OVERRIDES[w]
    if w in lexicon:
        return lexicon[w]
    if w.endswith("'s") and w[:-2] in lexicon:
        return lexicon[w[:-2]] + ["Z"]
    if len(w) > 1 and w.endswith("s") and w[:-1] in lexicon:
        return lexicon[w[:-1]] + ["Z"]
    return None


def vowel_pose(phone: str) -> str | None:
    b = base_phone(phone)
    s = stress(phone)
    if b == "IY":
        return "03_Ee"
    if b == "IH":
        return "04_Ih"
    if b == "EH":
        return "06_Eh"
    if b in {"AE", "AA"}:
        return "07_Aa"
    if b == "AH":
        return "18_Uh" if s == 0 else "07_Aa"
    if b in {"AO", "OW"}:
        return "09_Oh"
    if b == "UH":
        return "18_Uh" if s == 0 else "11_Oo"
    if b == "UW":
        return "11_Oo"
    if b == "ER":
        return "10_Or"
    if b in {"AY", "EY"}:
        return "05_Ay"
    if b == "AW":
        return "11_Oo"
    if b == "OY":
        return "05_Ay"
    return None


def consonant_pose(phone: str) -> str | None:
    b = base_phone(phone)
    if b in {"P", "B", "M"}:
        return "22_MBP"
    if b in {"F", "V"}:
        return "20_FV"
    if b == "W":
        return "12_Ww-Oo-"
    if b == "Y":
        return "16_Ww-Ew-"
    if b in {"SH", "CH", "JH", "ZH", "S", "Z"}:
        return "14_ChJjSh"
    if b in {"TH", "DH"}:
        return "24_Tongue_Th"
    return None


def phone_pose_sequence(phones: list[str], include_unstressed_vowels: bool) -> list[str]:
    out: list[str] = []
    for i, phone in enumerate(phones):
        b = base_phone(phone)
        if b == "Y" and i + 1 < len(phones) and base_phone(phones[i + 1]) == "UW":
            out.append("16_Ww-Ew-")
            continue
        if b == "AW":
            out.extend(["07_Aa", "11_Oo"])
            continue
        if b == "OY":
            out.extend(["09_Oh", "05_Ay"])
            continue
        if is_vowel(phone):
            if include_unstressed_vowels or stress(phone) in {1, 2} or b == "ER":
                pose = vowel_pose(phone)
                if pose:
                    out.append(pose)
            continue
        pose = consonant_pose(phone)
        if pose:
            out.append(pose)
    return collapse_adjacent(out)


def primary_vowel_pose(phones: list[str]) -> str | None:
    vowels = [(i, p) for i, p in enumerate(phones) if is_vowel(p)]
    for _, p in vowels:
        if stress(p) == 1:
            return vowel_pose(p)
    for _, p in vowels:
        if stress(p) == 2:
            return vowel_pose(p)
    return vowel_pose(vowels[0][1]) if vowels else None


def unstressed_vowel_poses(phones: list[str]) -> set[str]:
    return {pose for p in phones if is_vowel(p) and stress(p) == 0 for pose in [vowel_pose(p)] if pose}


def collapse_adjacent(poses: list[str]) -> list[str]:
    out: list[str] = []
    for pose in poses:
        if not out or out[-1] != pose:
            out.append(pose)
    return out


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    j = 0
    for pose in actual:
        if pose == expected[j]:
            j += 1
            if j == len(expected):
                return True
    return False


def strongest_vowel(poses: list[str], strengths: list[float]) -> str | None:
    best_pose = None
    best_strength = -1.0
    for pose, strength_value in zip(poses, strengths):
        if pose in VOWEL_POSES and strength_value > best_strength:
            best_pose = pose
            best_strength = strength_value
    return best_pose


def read_plan(case_dir: Path) -> dict[tuple[int, str], list[dict[str, str]]]:
    path = case_dir / "planned_events.csv"
    if not path.exists():
        return {}
    words: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            words[(int(row["WordIndex"]), norm_word(row["SourceWord"]))].append(row)
    for evs in words.values():
        evs.sort(key=lambda r: int(r["EventIndex"]))
    return words


def score_word(word: str, evs: list[dict[str, str]], lexicon: dict[str, list[str]]) -> dict[str, float]:
    poses = [r["PoseID"] for r in evs]
    strengths = [float(r["Strength"]) for r in evs]
    pron = lookup_pron(word, lexicon)
    if not pron:
        return {
            "phoneme_missing_visible_cues": 0,
            "dominant_vowel_mismatches": 0,
            "stress_salience_violations": 0,
            "phoneme_order_errors": 0,
            "unsupported_strong_poses": 0,
            "unstressed_vowel_overpromotions": 0,
            "diphthong_shape_errors": 0,
            "pronunciation_fallback_words": 1,
            "repeated_pose_pairs": sum(1 for a, b in zip(poses, poses[1:]) if a == b),
            "busy_short_words": 1 if len(word) <= 6 and len(evs) >= 4 else 0,
            "support_dominance_violations": sum(1 for p, s in zip(poses, strengths) if p in SUPPORT_POSES and s > 0.72),
            "function_overarticulation": 1 if word in FUNCTION_WORDS and len(evs) > 2 else 0,
        }

    required = set(phone_pose_sequence(pron, include_unstressed_vowels=False))
    allowed = set(phone_pose_sequence(pron, include_unstressed_vowels=True))
    expected_order = phone_pose_sequence(pron, include_unstressed_vowels=False)
    actual_order = collapse_adjacent(poses)
    expected_primary = primary_vowel_pose(pron)
    actual_primary = strongest_vowel(poses, strengths)
    unstressed = unstressed_vowel_poses(pron)

    missing = sum(1 for pose in required if pose not in poses)
    dominant_mismatch = 1 if expected_primary and actual_primary and expected_primary != actual_primary else 0
    if expected_primary and actual_primary is None:
        dominant_mismatch = 1

    dominant_strength = max((s for p, s in zip(poses, strengths) if p == expected_primary), default=0.0)
    strongest_secondary_vowel = max((s for p, s in zip(poses, strengths) if p in VOWEL_POSES and p != expected_primary), default=0.0)
    stress_violation = 1 if expected_primary and (dominant_strength < 0.58 or strongest_secondary_vowel > dominant_strength + 0.12) else 0

    order_error = 0 if is_subsequence(expected_order, actual_order) else 1
    unsupported = sum(1 for p, s in zip(poses, strengths) if s >= 0.72 and p not in allowed)
    overpromoted_unstressed = sum(1 for p, s in zip(poses, strengths) if p in unstressed and p != expected_primary and s >= 0.72)
    support_dominance = sum(
        1
        for p, s in zip(poses, strengths)
        if p != expected_primary and p in SUPPORT_POSES and s > 0.72
    )

    diphthong_error = 0
    for phone in pron:
        b = base_phone(phone)
        if b in {"AY", "EY"} and "05_Ay" not in poses:
            diphthong_error += 1
        elif b == "AW" and not ({"07_Aa", "11_Oo"} & set(poses)):
            diphthong_error += 1
        elif b == "OY" and not ({"09_Oh", "05_Ay"} & set(poses)):
            diphthong_error += 1

    return {
        "phoneme_missing_visible_cues": missing,
        "dominant_vowel_mismatches": dominant_mismatch,
        "stress_salience_violations": stress_violation,
        "phoneme_order_errors": order_error,
        "unsupported_strong_poses": unsupported,
        "unstressed_vowel_overpromotions": overpromoted_unstressed,
        "diphthong_shape_errors": diphthong_error,
        "pronunciation_fallback_words": 0,
        "repeated_pose_pairs": sum(1 for a, b in zip(poses, poses[1:]) if a == b),
        "busy_short_words": 1 if len(word) <= 6 and len(evs) >= 4 else 0,
        "support_dominance_violations": support_dominance,
        "function_overarticulation": 1 if word in FUNCTION_WORDS and len(evs) > 2 else 0,
    }


def summarize(values: list[float]) -> dict[str, str]:
    if not values:
        return {"mean": "", "median": "", "p90": "", "max": ""}
    sv = sorted(values)
    return {
        "mean": f"{mean(values):.6g}",
        "median": f"{median(values):.6g}",
        "p90": f"{sv[min(len(sv) - 1, math.ceil(len(sv) * 0.90) - 1)]:.6g}",
        "max": f"{sv[-1]:.6g}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory with per_case outputs")
    args = ap.parse_args()

    run = Path(args.run)
    repo_root = Path.cwd()
    per_case = run / "per_case"
    if not per_case.exists():
        raise SystemExit(f"Missing per_case directory: {per_case}")

    lexicon = load_cmudict(repo_root)
    metric_names = [
        "phoneme_missing_visible_cues",
        "dominant_vowel_mismatches",
        "stress_salience_violations",
        "phoneme_order_errors",
        "unsupported_strong_poses",
        "unstressed_vowel_overpromotions",
        "diphthong_shape_errors",
        "pronunciation_fallback_words",
        "repeated_pose_pairs",
        "busy_short_words",
        "support_dominance_violations",
        "function_overarticulation",
    ]

    rows: list[dict[str, str]] = []
    worst_words: list[dict[str, str]] = []
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        words = read_plan(case_dir)
        totals = {k: 0.0 for k in metric_names}
        word_count = 0
        event_count = 0
        for (_, word), evs in words.items():
            if not word:
                continue
            word_count += 1
            event_count += len(evs)
            scores = score_word(word, evs, lexicon)
            for key, value in scores.items():
                totals[key] += value
            severity = sum(scores.values())
            if severity > 0:
                worst_words.append({
                    "case_id": case_dir.name,
                    "word": word,
                    "pronunciation": " ".join(lookup_pron(word, lexicon) or []),
                    "severity": f"{severity:.0f}",
                    "poses": "|".join(r["PoseID"] for r in evs),
                    **{k: f"{v:.0f}" for k, v in scores.items()},
                })

        total_errors = sum(totals.values())
        row = {
            "case_id": case_dir.name,
            "word_count": str(word_count),
            "event_count": str(event_count),
            "planner_quality_error_count": f"{total_errors:.0f}",
            "planner_quality_error_rate": f"{(total_errors / max(1, word_count)):.6g}",
        }
        row.update({k: f"{v:.0f}" for k, v in totals.items()})
        rows.append(row)

    fieldnames = ["case_id", "word_count", "event_count", "planner_quality_error_count", "planner_quality_error_rate"] + metric_names
    with (run / "planner_quality_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (run / "planner_quality_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "median", "p90", "max"])
        writer.writeheader()
        for metric in fieldnames[1:]:
            writer.writerow({"metric": metric, **summarize([float(r[metric]) for r in rows])})

    worst_words.sort(key=lambda r: (-float(r["severity"]), r["case_id"], r["word"]))
    with (run / "planner_quality_worst_words.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["case_id", "word", "pronunciation", "severity", "poses"] + metric_names
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(worst_words[:500])

    print(f"Wrote: {run / 'planner_quality_summary.csv'}")
    print(f"Wrote: {run / 'planner_quality_stats.csv'}")
    print(f"Wrote: {run / 'planner_quality_worst_words.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
