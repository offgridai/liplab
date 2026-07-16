#!/usr/bin/env python3
"""Build portable CMU pronunciation preferences from approved MFA alignments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path


MIN_OBSERVATIONS = 4
MIN_PREFERRED_SHARE = 0.45
MIN_WIN_OVER_DEFAULT = 2
MIN_CONTEXT_OBSERVATIONS = 2
MIN_CONTEXT_PREFERRED_SHARE = 0.67


def normalize_word(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum() or ch == "'")


def base_sequence(phones: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(re.sub(r"\d", "", phone) for phone in phones)


def read_cmudict(source: Path) -> dict[str, list[tuple[str, ...]]]:
    text = source.read_text(encoding="utf-8")
    raw = "\n".join(re.findall(r'R"CMU\((.*?)\)CMU"', text, re.S))
    variants: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for raw_line in raw.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = normalize_word(re.sub(r"\(\d+\)$", "", parts[0]))
        pronunciation = tuple(parts[1:])
        if pronunciation not in variants[word]:
            variants[word].append(pronunciation)
    return dict(variants)


def read_gold_occurrences(
    gold_root: Path,
    variants: dict[str, list[tuple[str, ...]]],
) -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    occurrences: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    for case_dir in sorted(path for path in gold_root.iterdir() if path.is_dir()):
        phones_path = case_dir / "phones.csv"
        if not phones_path.exists():
            continue
        by_word: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
        with phones_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("word_index", ""), normalize_word(row.get("word", "")))
                by_word.setdefault(key, []).append(row.get("phone", ""))
        ordered_words = list(by_word.items())
        for word_position, ((_, word), observed_list) in enumerate(ordered_words):
            observed = tuple(observed_list)
            exact_variants = [
                candidate
                for candidate in variants.get(word, [])
                if candidate == observed
            ]
            matching_variants = exact_variants or [
                candidate
                for candidate in variants.get(word, [])
                if base_sequence(candidate) == base_sequence(observed)
            ]
            if matching_variants:
                previous_word = ordered_words[word_position - 1][0][1] if word_position > 0 else "<s>"
                next_word = ordered_words[word_position + 1][0][1] if word_position + 1 < len(ordered_words) else "</s>"
                occurrences.append((case_dir.name, word, previous_word, next_word, matching_variants[0]))
    return occurrences


def choose_preferences(
    occurrences: list[tuple[str, str, str, str, tuple[str, ...]]],
    variants: dict[str, list[tuple[str, ...]]],
) -> dict[str, dict[str, object]]:
    counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for _, word, _, _, pronunciation in occurrences:
        counts[word][pronunciation] += 1

    preferences: dict[str, dict[str, object]] = {}
    for word, word_counts in counts.items():
        total = sum(word_counts.values())
        preferred, preferred_count = word_counts.most_common(1)[0]
        default = variants[word][0]
        default_count = word_counts[default]
        preferred_share = preferred_count / total
        if preferred == default:
            continue
        if total < MIN_OBSERVATIONS or preferred_share < MIN_PREFERRED_SHARE:
            continue
        if preferred_count < default_count + MIN_WIN_OVER_DEFAULT:
            continue
        preferences[word] = {
            "phones": preferred,
            "observations": total,
            "preferred_count": preferred_count,
            "preferred_share": preferred_share,
            "default_phones": default,
            "default_count": default_count,
            "variant_index": variants[word].index(preferred),
            "variant_count": len(variants[word]),
        }
    return preferences


def choose_context_preferences(
    occurrences: list[tuple[str, str, str, str, tuple[str, ...]]],
    variants: dict[str, list[tuple[str, ...]]],
    global_preferences: dict[str, dict[str, object]],
) -> dict[tuple[str, str, bool], dict[str, object]]:
    counts: dict[tuple[str, str, bool], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for _, word, previous_word, next_word, pronunciation in occurrences:
        counts[(word, previous_word, True)][pronunciation] += 1
        counts[(word, next_word, False)][pronunciation] += 1

    preferences: dict[tuple[str, str, bool], dict[str, object]] = {}
    for key, context_counts in counts.items():
        word, _, _ = key
        total = sum(context_counts.values())
        preferred, preferred_count = context_counts.most_common(1)[0]
        fallback = global_preferences.get(word, {}).get("phones", variants[word][0])
        preferred_share = preferred_count / total
        if preferred == fallback:
            continue
        if total < MIN_CONTEXT_OBSERVATIONS or preferred_share < MIN_CONTEXT_PREFERRED_SHARE:
            continue
        preferences[key] = {
            "phones": preferred,
            "observations": total,
            "preferred_count": preferred_count,
            "preferred_share": preferred_share,
        }
    return preferences


def heldout_report(
    occurrences: list[tuple[str, str, str, str, tuple[str, ...]]],
    variants: dict[str, list[tuple[str, ...]]],
) -> dict[str, object]:
    train = [
        row for row in occurrences
        if int(hashlib.sha1(row[0].encode("utf-8")).hexdigest()[:8], 16) % 5 != 0
    ]
    test = [
        row for row in occurrences
        if int(hashlib.sha1(row[0].encode("utf-8")).hexdigest()[:8], 16) % 5 == 0
    ]
    preferences = choose_preferences(train, variants)
    context_preferences = choose_context_preferences(train, variants, preferences)

    def accuracy(use_preferences: bool, exact: bool) -> float:
        matches = 0
        for _, word, previous_word, next_word, observed in test:
            selected = variants[word][0]
            if use_preferences and word in preferences:
                selected = preferences[word]["phones"]  # type: ignore[assignment]
            if use_preferences:
                candidates = [
                    context_preferences[key]
                    for key in ((word, previous_word, True), (word, next_word, False))
                    if key in context_preferences
                ]
                if candidates:
                    selected = max(
                        candidates,
                        key=lambda row: (row["preferred_share"], row["preferred_count"]),
                    )["phones"]  # type: ignore[assignment]
            matches += selected == observed if exact else base_sequence(selected) == base_sequence(observed)
        return matches / max(len(test), 1)

    return {
        "train_occurrences": len(train),
        "test_occurrences": len(test),
        "learned_preference_count": len(preferences),
        "learned_context_preference_count": len(context_preferences),
        "default_exact_pronunciation_accuracy": accuracy(False, True),
        "calibrated_exact_pronunciation_accuracy": accuracy(True, True),
        "default_base_pronunciation_accuracy": accuracy(False, False),
        "calibrated_base_pronunciation_accuracy": accuracy(True, False),
    }


def write_cpp(
    path: Path,
    preferences: dict[str, dict[str, object]],
    context_preferences: dict[tuple[str, str, bool], dict[str, object]],
) -> None:
    lines = [
        "// Generated by scripts/build_tts_pronunciation_preferences.py.",
        "// Existing CMU variants preferred by the corpus TTS voice; do not edit manually.",
        "struct FOffgridAITtsPronunciationPreferenceRow",
        "{",
        "    const TCHAR* Word;",
        "    const TCHAR* Phones;",
        "    int32 ObservationCount;",
        "    float PreferredShare;",
        "};",
        "",
        "static const FOffgridAITtsPronunciationPreferenceRow GTtsPronunciationPreferences[] =",
        "{",
    ]
    for word, data in sorted(preferences.items()):
        phones = " ".join(data["phones"])  # type: ignore[arg-type]
        lines.append(
            f'    {{TEXT("{word}"), TEXT("{phones}"), '
            f'{data["observations"]}, {float(data["preferred_share"]):.6f}f}},'
        )
    lines.extend([
        "};",
        "",
        "struct FOffgridAITtsContextPronunciationPreferenceRow",
        "{",
        "    const TCHAR* Word;",
        "    const TCHAR* NeighborWord;",
        "    bool bPreviousNeighbor;",
        "    const TCHAR* Phones;",
        "    int32 ObservationCount;",
        "    float PreferredShare;",
        "};",
        "",
        "static const FOffgridAITtsContextPronunciationPreferenceRow GTtsContextPronunciationPreferences[] =",
        "{",
    ])
    for (word, neighbor, is_previous), data in sorted(context_preferences.items()):
        phones = " ".join(data["phones"])  # type: ignore[arg-type]
        lines.append(
            f'    {{TEXT("{word}"), TEXT("{neighbor}"), '
            f'{str(is_previous).lower()}, TEXT("{phones}"), '
            f'{data["observations"]}, {float(data["preferred_share"]):.6f}f}},'
        )
    lines.extend(["};", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    variants = read_cmudict(root / "offgrid_dropin/Private/Lipsync/OffgridAICmudictData.cpp")
    occurrences = read_gold_occurrences(root / "inputs/gold", variants)
    preferences = choose_preferences(occurrences, variants)
    context_preferences = choose_context_preferences(occurrences, variants, preferences)
    heldout = heldout_report(occurrences, variants)

    write_cpp(
        root / "offgrid_dropin/Private/Lipsync/OffgridAITtsPronunciationPreferences.inl",
        preferences,
        context_preferences,
    )
    report = {
        "criteria": {
            "minimum_observations": MIN_OBSERVATIONS,
            "minimum_preferred_share": MIN_PREFERRED_SHARE,
            "minimum_win_over_default": MIN_WIN_OVER_DEFAULT,
            "minimum_context_observations": MIN_CONTEXT_OBSERVATIONS,
            "minimum_context_preferred_share": MIN_CONTEXT_PREFERRED_SHARE,
        },
        "matched_gold_occurrences": len(occurrences),
        "preference_count": len(preferences),
        "context_preference_count": len(context_preferences),
        "heldout": heldout,
        "preferences": {
            word: {
                **{key: value for key, value in data.items() if key not in {"phones", "default_phones"}},
                "phones": list(data["phones"]),
                "default_phones": list(data["default_phones"]),
            }
            for word, data in sorted(preferences.items())
        },
        "context_preferences": [
            {
                "word": word,
                "neighbor": neighbor,
                "previous_neighbor": is_previous,
                **{key: value for key, value in data.items() if key != "phones"},
                "phones": list(data["phones"]),
            }
            for (word, neighbor, is_previous), data in sorted(context_preferences.items())
        ],
    }
    report_path = root / "docs/tts_pronunciation_preferences.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(
        f"Generated {len(preferences)} global and {len(context_preferences)} context preferences "
        f"from {len(occurrences)} matched occurrences; "
        f"heldout exact {heldout['default_exact_pronunciation_accuracy']:.3f} -> "
        f"{heldout['calibrated_exact_pronunciation_accuracy']:.3f}; "
        f"base {heldout['default_base_pronunciation_accuracy']:.3f} -> "
        f"{heldout['calibrated_base_pronunciation_accuracy']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
