#!/usr/bin/env python3
"""Compare Offgrid lipsync logs with MFA word timing from the same WAVs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from praatio import textgrid


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def normalized_word(word: str) -> str:
    return "".join(ch for ch in word.casefold() if ch.isalnum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("mfa_root", type=Path)
    parser.add_argument("index_csv", type=Path)
    args = parser.parse_args()

    index_rows = read_csv(args.index_csv)
    all_word_rows: list[dict[str, object]] = []
    pause_expected = 0
    pause_detected = 0
    pause_matched = 0
    for index_row in index_rows:
        log_dir = Path(index_row["source_dir"])
        case_id = index_row["case_id"]
        grid_path = args.mfa_root / f"{case_id}.TextGrid"
        if not grid_path.exists() or not log_dir.exists():
            continue
        grid = textgrid.openTextgrid(str(grid_path), includeEmptyIntervals=True)
        mfa_words = [entry for entry in grid.getTier("words").entries if entry.label]
        integrity = sorted(
            read_csv(log_dir / "word_region_integrity.csv"),
            key=lambda row: i(row, "WordIndex"),
        )
        pacing = {
            i(row, "WordIndex"): row for row in read_csv(log_dir / "word_pacing.csv")
        }
        event_word_index = {
            i(row, "EventIndex"): i(row, "WordIndex")
            for row in read_csv(log_dir / "runtime_commit_events_final.csv")
        }
        visibility = read_csv(log_dir / "event_visibility_summary.csv")
        regions = sorted(
            read_csv(log_dir / "speech_regions.csv"),
            key=lambda row: i(row, "RuntimeRegionIndex"),
        )
        visible_by_word: dict[int, list[dict[str, str]]] = {}
        for row in visibility:
            event_index = i(row, "EventIndex")
            word_index = event_word_index.get(event_index, -1)
            visible_by_word.setdefault(word_index, []).append(row)

        print(f"\n=== {log_dir.name}")
        print(read_metadata(log_dir / "line_metadata.txt").get("Dialogue", ""))
        mfa_gaps = [
            (float(left.end), float(right.start))
            for left, right in zip(mfa_words, mfa_words[1:])
            if float(right.start) - float(left.end) >= 0.120
        ]
        runtime_gaps = [
            (f(left, "EndSec"), f(right, "StartSec"))
            for left, right in zip(regions, regions[1:])
            if f(right, "StartSec") - f(left, "EndSec") >= 0.080
        ]
        matched_runtime: set[int] = set()
        line_matches = 0
        for expected_start, expected_end in mfa_gaps:
            expected_center = 0.5 * (expected_start + expected_end)
            candidates = [
                (abs(0.5 * (start + end) - expected_center), index)
                for index, (start, end) in enumerate(runtime_gaps)
                if index not in matched_runtime
                and (min(expected_end, end) > max(expected_start, start)
                     or abs(0.5 * (start + end) - expected_center) <= 0.180)
            ]
            if candidates:
                _, best = min(candidates)
                matched_runtime.add(best)
                line_matches += 1
        pause_expected += len(mfa_gaps)
        pause_detected += len(runtime_gaps)
        pause_matched += line_matches
        print(
            f"pauses: MFA={len(mfa_gaps)} runtime={len(runtime_gaps)} "
            f"matched={line_matches} false={len(runtime_gaps) - line_matches}"
        )
        print("idx word       mfa_start anchor   visible  onset_err  committed last_vs_end status")
        for position, word_row in enumerate(integrity):
            runtime_word = word_row.get("Word", "")
            mfa = mfa_words[position] if position < len(mfa_words) else None
            name_matches = bool(
                mfa
                and normalized_word(runtime_word) == normalized_word(mfa.label)
            )
            word_index = i(word_row, "WordIndex")
            pace = pacing.get(word_index, {})
            visible = visible_by_word.get(word_index, [])
            first_visible = min(
                (f(row, "FirstFaceDriverSec", 1e9) for row in visible),
                default=1e9,
            )
            last_visible = max(
                (f(row, "LastFaceDriverSec", -1.0) for row in visible),
                default=-1.0,
            )
            mfa_start = float(mfa.start) if mfa else -1.0
            mfa_end = float(mfa.end) if mfa else -1.0
            onset_error_ms = (
                (first_visible - mfa_start) * 1000.0
                if first_visible < 1e8 and mfa else None
            )
            last_vs_end_ms = (
                (last_visible - mfa_end) * 1000.0
                if last_visible >= 0.0 and mfa else None
            )
            committed = i(word_row, "CommittedVisemeCount", 0)
            planned = i(word_row, "PlannedVisemeCount", 0)
            status = word_row.get("IntegrityStatus", "")
            if mfa and not name_matches:
                status += f"/mfa={mfa.label}"
            print(
                f"{word_index:>3} {runtime_word[:10]:<10} "
                f"{mfa_start:>8.3f} {f(pace, 'WordStartSec', -1):>7.3f} "
                f"{(first_visible if first_visible < 1e8 else -1):>8.3f} "
                f"{(f'{onset_error_ms:+.0f}ms' if onset_error_ms is not None else 'missing'):>9} "
                f"{committed:>2}/{planned:<2} "
                f"{(f'{last_vs_end_ms:+.0f}ms' if last_vs_end_ms is not None else 'missing'):>10} "
                f"{status}"
            )
            all_word_rows.append(
                {
                    "line": log_dir.name,
                    "word": runtime_word,
                    "onset_error_ms": onset_error_ms,
                    "last_vs_end_ms": last_vs_end_ms,
                    "complete": committed == planned and planned > 0,
                }
            )

    aligned = [row for row in all_word_rows if row["onset_error_ms"] is not None]
    errors = [float(row["onset_error_ms"]) for row in aligned]
    late = [row for row in aligned if float(row["onset_error_ms"]) > 80.0]
    missing = [row for row in all_word_rows if row["onset_error_ms"] is None]
    incomplete = [row for row in all_word_rows if not row["complete"]]
    print("\n=== aggregate")
    print(f"words={len(all_word_rows)} visible={len(aligned)} missing={len(missing)} incomplete={len(incomplete)}")
    print(
        f"pauses_expected={pause_expected} detected={pause_detected} "
        f"matched={pause_matched} missed={pause_expected - pause_matched} "
        f"false={pause_detected - pause_matched}"
    )
    if errors:
        absolute_errors = [abs(error) for error in errors]
        print(
            f"onset_error_signed_mean_ms={statistics.mean(errors):.1f} "
            f"mean_abs_ms={statistics.mean(absolute_errors):.1f} "
            f"median_abs_ms={statistics.median(absolute_errors):.1f} "
            f"late_over_80ms={len(late)}/{len(aligned)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
