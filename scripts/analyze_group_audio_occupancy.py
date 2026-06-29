#!/usr/bin/env python3
"""Audit whether text prosody groups correlate with observed audio occupancy.

This is an offline experiment. It does not change runtime behavior.

Input: a liplab run directory containing per_case/<case_id>/committed_events.csv
and speech_islands.csv. Optional transcript text is read from --inputs.

Output directory defaults to <run>/group_audio_occupancy and contains:
  - group_audio_alignment.csv
  - group_boundary_audio_gaps.csv
  - group_audio_alignment_summary.csv
  - interesting_failures.csv

The audit treats punctuation as a prior for group boundaries, not as the core
state machine. Text groups are matched monotonically to detected speech islands;
multiple adjacent groups may share one island when the audio contains no usable
silence, and one group may be flagged as split when an internal audio gap exists.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


PUNCT_RE = re.compile(r"([A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?)([^A-Za-z0-9\s]*)")
HARD_PUNCT = {".", "?", "!"}
SOFT_PUNCT = {",", ";", ":"}


@dataclass
class Event:
    event_index: int
    pose_id: str
    word: str
    word_index: int
    phrase_index: int
    group_index: int
    text_center: float
    committed_center: float
    strength: float


@dataclass
class Group:
    case_id: str
    group_id: int
    source_group_index: int
    first_event_index: int
    last_event_index: int
    first_word_index: int
    last_word_index: int
    words: list[str]
    terminal_punct: str
    phrase_indices: list[int]
    event_count: int
    strength_max: float
    text_start: float
    text_end: float
    text_center: float
    committed_start: float
    committed_end: float

    @property
    def text_duration(self) -> float:
        return max(0.0, self.text_end - self.text_start)

    @property
    def text(self) -> str:
        return " ".join(self.words)


@dataclass
class Island:
    island_index: int
    start: float
    last_speech: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.last_speech - self.start)


@dataclass
class Gap:
    before_island: int
    after_island: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Match:
    group: Group
    island: Island | None
    match_type: str
    start_error: float
    end_error: float
    duration_ratio: float
    confidence: float
    launch_shift: float


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def i(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except ValueError:
        return default


def canonical_word(s: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", s.lower().replace("’", "'"))


def transcript_terminal_punct(transcript_path: Path | None) -> dict[int, str]:
    if transcript_path is None or not transcript_path.exists():
        return {}
    text = transcript_path.read_text(encoding="utf-8", errors="replace")
    out: dict[int, str] = {}
    word_idx = 0
    for match in PUNCT_RE.finditer(text):
        word = canonical_word(match.group(1))
        if not word:
            continue
        punct = match.group(2) or ""
        punct = "".join(ch for ch in punct if ch in HARD_PUNCT or ch in SOFT_PUNCT)
        if punct:
            out[word_idx] = punct[-1]
        word_idx += 1
    return out


def load_events(case_dir: Path) -> list[Event]:
    rows = read_csv(case_dir / "committed_events.csv")
    events: list[Event] = []
    for row in rows:
        group_idx = i(row, "PlannerProsodyGroupIndex", -1)
        if group_idx < 0:
            group_idx = i(row, "PhraseIndex", 0)
        text_center = f(row, "TextCenterSec", math.nan)
        if math.isnan(text_center):
            text_center = f(row, "PlannerEventPredictedOffsetSec", 0.0)
        committed = f(row, "CommittedPlaybackCenterSec", text_center)
        events.append(
            Event(
                event_index=i(row, "EventIndex", len(events)),
                pose_id=row.get("PoseID", ""),
                word=row.get("SourceWord", ""),
                word_index=i(row, "WordIndex", -1),
                phrase_index=i(row, "PhraseIndex", 0),
                group_index=group_idx,
                text_center=text_center,
                committed_center=committed,
                strength=f(row, "Strength", 0.0),
            )
        )
    events.sort(key=lambda e: (e.text_center, e.event_index))
    return events


def build_groups(case_id: str, events: list[Event], punct_by_word: dict[int, str]) -> list[Group]:
    grouped: list[list[Event]] = []
    current_key: tuple[int, int] | None = None
    for e in events:
        key = (e.group_index, e.phrase_index if e.group_index < 0 else e.group_index)
        if current_key is None or key != current_key:
            grouped.append([])
            current_key = key
        grouped[-1].append(e)

    groups: list[Group] = []
    for gid, evs in enumerate(grouped):
        if not evs:
            continue
        word_pairs: list[tuple[int, str]] = []
        seen_words: set[int] = set()
        for e in evs:
            if e.word_index >= 0 and e.word_index not in seen_words:
                seen_words.add(e.word_index)
                word_pairs.append((e.word_index, e.word))
        word_pairs.sort(key=lambda x: x[0])
        words = [w for _, w in word_pairs if w]
        first_word = min((idx for idx, _ in word_pairs), default=-1)
        last_word = max((idx for idx, _ in word_pairs), default=-1)
        terminal = punct_by_word.get(last_word, "")
        text_centers = [e.text_center for e in evs]
        committed_centers = [e.committed_center for e in evs]
        groups.append(
            Group(
                case_id=case_id,
                group_id=gid,
                source_group_index=evs[0].group_index,
                first_event_index=min(e.event_index for e in evs),
                last_event_index=max(e.event_index for e in evs),
                first_word_index=first_word,
                last_word_index=last_word,
                words=words,
                terminal_punct=terminal,
                phrase_indices=sorted({e.phrase_index for e in evs}),
                event_count=len(evs),
                strength_max=max((e.strength for e in evs), default=0.0),
                text_start=min(text_centers),
                text_end=max(text_centers),
                text_center=sum(text_centers) / max(1, len(text_centers)),
                committed_start=min(committed_centers),
                committed_end=max(committed_centers),
            )
        )
    return groups


def load_islands(case_dir: Path) -> list[Island]:
    islands: list[Island] = []
    for row in read_csv(case_dir / "speech_islands.csv"):
        islands.append(
            Island(
                island_index=i(row, "IslandIndex", len(islands)),
                start=f(row, "AudioBufferStartSec", 0.0),
                last_speech=f(row, "AudioBufferLastSpeechSec", f(row, "AudioBufferEndSec", 0.0)),
                end=f(row, "AudioBufferEndSec", 0.0),
            )
        )
    islands.sort(key=lambda x: x.start)
    return islands


def audio_gaps(islands: list[Island], min_gap_sec: float) -> list[Gap]:
    gaps: list[Gap] = []
    for a, b in zip(islands, islands[1:]):
        gap = Gap(a.island_index, b.island_index, a.last_speech, b.start)
        if gap.duration >= min_gap_sec:
            gaps.append(gap)
    return gaps


def gap_after_island(islands: list[Island], island_idx: int) -> float:
    for pos, island in enumerate(islands[:-1]):
        if island.island_index == island_idx:
            return max(0.0, islands[pos + 1].start - island.last_speech)
    return 0.0


def classify_punct(p: str) -> str:
    if not p:
        return "none"
    if any(ch in HARD_PUNCT for ch in p):
        return "hard"
    if any(ch in SOFT_PUNCT for ch in p):
        return "soft"
    return "other"


def match_groups_to_islands(groups: list[Group], islands: list[Island], min_gap_sec: float, boundary_window_sec: float) -> list[Match]:
    if not groups:
        return []
    if not islands:
        return [
            Match(
                group=g,
                island=None,
                match_type="missing_audio_island",
                start_error=0.0,
                end_error=0.0,
                duration_ratio=0.0,
                confidence=0.0,
                launch_shift=0.0,
            )
            for g in groups
        ]

    # Align the text clock to the first observed onset. This preserves text spacing
    # while asking whether the audio occupancy rhythm naturally supports group starts.
    launch_shift = islands[0].start - groups[0].text_start
    matches: list[Match] = []
    island_pos = 0
    groups_per_island: dict[int, int] = {}

    gaps = audio_gaps(islands, min_gap_sec)

    for idx, group in enumerate(groups):
        expected_start = group.text_start + launch_shift
        expected_end = group.text_end + launch_shift

        # Advance only when the next island is a better causal fit for this group.
        while island_pos + 1 < len(islands):
            cur = islands[island_pos]
            nxt = islands[island_pos + 1]
            cur_error = abs(expected_start - cur.start)
            nxt_error = abs(expected_start - nxt.start)
            if expected_start > cur.last_speech + min_gap_sec and nxt_error + 0.050 < cur_error:
                island_pos += 1
            else:
                break

        island = islands[island_pos]
        groups_per_island[island.island_index] = groups_per_island.get(island.island_index, 0) + 1

        internal_gap = any(g.start > expected_start and g.end < expected_end for g in gaps)
        if internal_gap:
            match_type = "split_audio_inside_text_group"
        elif groups_per_island[island.island_index] > 1:
            match_type = "merged_text_groups_to_one_audio_island"
        else:
            match_type = "exact_1_to_1"

        start_error = island.start - expected_start
        end_error = island.last_speech - expected_end
        duration_ratio = island.duration / max(0.001, group.text_duration)
        confidence = max(0.0, 1.0 - min(1.0, abs(start_error) / max(0.150, boundary_window_sec)))
        matches.append(
            Match(
                group=group,
                island=island,
                match_type=match_type,
                start_error=start_error,
                end_error=end_error,
                duration_ratio=duration_ratio,
                confidence=confidence,
                launch_shift=launch_shift,
            )
        )

    # Reclassify any group mapped to an island used by multiple groups.
    counts: dict[int, int] = {}
    for match in matches:
        if match.island is not None:
            counts[match.island.island_index] = counts.get(match.island.island_index, 0) + 1
    fixed: list[Match] = []
    for match in matches:
        if (
            match.island is not None
            and counts.get(match.island.island_index, 0) > 1
            and match.match_type == "exact_1_to_1"
        ):
            match.match_type = "merged_text_groups_to_one_audio_island"
        fixed.append(match)
    return fixed


def boundary_rows(groups: list[Group], islands: list[Island], min_gap_sec: float, boundary_window_sec: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if len(groups) < 2:
        return rows
    gaps = audio_gaps(islands, min_gap_sec)
    launch_shift = islands[0].start - groups[0].text_start if islands else 0.0
    for left, right in zip(groups, groups[1:]):
        expected = 0.5 * (left.text_end + right.text_start) + launch_shift
        best: Gap | None = None
        best_abs = math.inf
        for gap in gaps:
            center = 0.5 * (gap.start + gap.end)
            err = abs(center - expected)
            if err < best_abs:
                best_abs = err
                best = gap
        matched = best is not None and best_abs <= boundary_window_sec
        rows.append(
            {
                "CaseId": left.case_id,
                "BoundaryAfterGroupId": left.group_id,
                "LeftText": left.text,
                "RightText": right.text,
                "TerminalPunct": left.terminal_punct,
                "PunctClass": classify_punct(left.terminal_punct),
                "ExpectedBoundarySec": f"{expected:.6f}",
                "MatchedAudioGap": 1 if matched else 0,
                "AudioGapBeforeIsland": best.before_island if matched and best else "",
                "AudioGapAfterIsland": best.after_island if matched and best else "",
                "AudioGapStartSec": f"{best.start:.6f}" if matched and best else "",
                "AudioGapEndSec": f"{best.end:.6f}" if matched and best else "",
                "AudioGapDurationSec": f"{best.duration:.6f}" if matched and best else "",
                "BoundaryErrorMs": f"{(0.5 * (best.start + best.end) - expected) * 1000.0:.3f}" if matched and best else "",
            }
        )
    return rows


def iter_case_dirs(run_dir: Path) -> Iterable[Path]:
    per_case = run_dir / "per_case"
    root = per_case if per_case.exists() else run_dir
    for child in sorted(root.iterdir() if root.exists() else []):
        if child.is_dir() and (child / "committed_events.csv").exists():
            yield child


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="liplab run directory containing per_case/")
    ap.add_argument("--inputs", default=None, help="optional inputs directory containing transcripts/")
    ap.add_argument("--out", default=None, help="output directory; default: <run>/group_audio_occupancy")
    ap.add_argument("--min-gap-sec", type=float, default=0.120, help="minimum silence gap considered an occupancy pause")
    ap.add_argument("--boundary-window-sec", type=float, default=0.450, help="max distance for matching expected text boundary to observed audio gap")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    inputs_dir = Path(args.inputs).resolve() if args.inputs else None
    out_dir = Path(args.out).resolve() if args.out else run_dir / "group_audio_occupancy"

    alignment: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    cases_seen = 0
    for case_dir in iter_case_dirs(run_dir):
        case_id = case_dir.name
        events = load_events(case_dir)
        islands = load_islands(case_dir)
        transcript_path = inputs_dir / "transcripts" / f"{case_id}.txt" if inputs_dir else None
        groups = build_groups(case_id, events, transcript_terminal_punct(transcript_path))
        if not groups:
            continue
        cases_seen += 1
        matches = match_groups_to_islands(groups, islands, args.min_gap_sec, args.boundary_window_sec)
        for match in matches:
            group = match.group
            island = match.island
            row = {
                "CaseId": case_id,
                "GroupId": group.group_id,
                "SourceGroupIndex": group.source_group_index,
                "Text": group.text,
                "FirstWordIndex": group.first_word_index,
                "LastWordIndex": group.last_word_index,
                "TerminalPunct": group.terminal_punct,
                "PunctClass": classify_punct(group.terminal_punct),
                "PhraseIndices": ";".join(str(x) for x in group.phrase_indices),
                "EventCount": group.event_count,
                "TextStartSec": f"{group.text_start:.6f}",
                "TextEndSec": f"{group.text_end:.6f}",
                "TextDurationSec": f"{group.text_duration:.6f}",
                "LaunchShiftSec": f"{match.launch_shift:.6f}",
                "AudioIslandId": island.island_index if island else "",
                "AudioStartSec": f"{island.start:.6f}" if island else "",
                "AudioEndSec": f"{island.last_speech:.6f}" if island else "",
                "AudioDurationSec": f"{island.duration:.6f}" if island else "",
                "AudioGapAfterSec": f"{gap_after_island(islands, island.island_index):.6f}" if island else "",
                "StartErrorMs": f"{match.start_error * 1000.0:.3f}",
                "EndErrorMs": f"{match.end_error * 1000.0:.3f}",
                "DurationRatio": f"{match.duration_ratio:.6f}",
                "MatchType": match.match_type,
                "Confidence": f"{match.confidence:.6f}",
            }
            alignment.append(row)
            if match.match_type != "exact_1_to_1" or abs(match.start_error) > args.boundary_window_sec:
                failures.append(row)
        boundaries.extend(boundary_rows(groups, islands, args.min_gap_sec, args.boundary_window_sec))

    alignment_fields = [
        "CaseId", "GroupId", "SourceGroupIndex", "Text", "FirstWordIndex", "LastWordIndex",
        "TerminalPunct", "PunctClass", "PhraseIndices", "EventCount", "TextStartSec", "TextEndSec",
        "TextDurationSec", "LaunchShiftSec", "AudioIslandId", "AudioStartSec", "AudioEndSec",
        "AudioDurationSec", "AudioGapAfterSec", "StartErrorMs", "EndErrorMs", "DurationRatio",
        "MatchType", "Confidence",
    ]
    boundary_fields = [
        "CaseId", "BoundaryAfterGroupId", "LeftText", "RightText", "TerminalPunct", "PunctClass",
        "ExpectedBoundarySec", "MatchedAudioGap", "AudioGapBeforeIsland", "AudioGapAfterIsland",
        "AudioGapStartSec", "AudioGapEndSec", "AudioGapDurationSec", "BoundaryErrorMs",
    ]
    write_csv(out_dir / "group_audio_alignment.csv", alignment, alignment_fields)
    write_csv(out_dir / "group_boundary_audio_gaps.csv", boundaries, boundary_fields)
    write_csv(out_dir / "interesting_failures.csv", failures[:250], alignment_fields)

    total = len(alignment)
    exact = sum(1 for r in alignment if r["MatchType"] == "exact_1_to_1")
    merged = sum(1 for r in alignment if r["MatchType"] == "merged_text_groups_to_one_audio_island")
    split = sum(1 for r in alignment if r["MatchType"] == "split_audio_inside_text_group")
    missing = sum(1 for r in alignment if r["MatchType"] == "missing_audio_island")
    comma_boundaries = [r for r in boundaries if r["TerminalPunct"] == ","]
    hard_boundaries = [r for r in boundaries if r["PunctClass"] == "hard"]
    none_boundaries = [r for r in boundaries if r["PunctClass"] == "none"]
    start_errors = [abs(float(r["StartErrorMs"])) for r in alignment if r["StartErrorMs"] != ""]
    boundary_errors = [abs(float(r["BoundaryErrorMs"])) for r in boundaries if r["BoundaryErrorMs"] != ""]
    summary = [
        {"Metric": "CasesSeen", "Value": cases_seen},
        {"Metric": "Groups", "Value": total},
        {"Metric": "ExactOneToOneGroups", "Value": exact},
        {"Metric": "ExactOneToOneRatePct", "Value": f"{pct(exact, total):.3f}"},
        {"Metric": "MergedGroups", "Value": merged},
        {"Metric": "SplitGroups", "Value": split},
        {"Metric": "MissingAudioGroups", "Value": missing},
        {"Metric": "MedianAbsGroupStartErrorMs", "Value": f"{median(start_errors):.3f}" if start_errors else ""},
        {"Metric": "Boundaries", "Value": len(boundaries)},
        {"Metric": "MatchedBoundaries", "Value": sum(1 for r in boundaries if r["MatchedAudioGap"] == 1)},
        {"Metric": "MatchedBoundaryRatePct", "Value": f"{pct(sum(1 for r in boundaries if r['MatchedAudioGap'] == 1), len(boundaries)):.3f}"},
        {"Metric": "CommaBoundaries", "Value": len(comma_boundaries)},
        {"Metric": "CommaBoundaryMatchedRatePct", "Value": f"{pct(sum(1 for r in comma_boundaries if r['MatchedAudioGap'] == 1), len(comma_boundaries)):.3f}"},
        {"Metric": "HardPunctBoundaries", "Value": len(hard_boundaries)},
        {"Metric": "HardPunctBoundaryMatchedRatePct", "Value": f"{pct(sum(1 for r in hard_boundaries if r['MatchedAudioGap'] == 1), len(hard_boundaries)):.3f}"},
        {"Metric": "NoPunctBoundaries", "Value": len(none_boundaries)},
        {"Metric": "NoPunctBoundaryMatchedRatePct", "Value": f"{pct(sum(1 for r in none_boundaries if r['MatchedAudioGap'] == 1), len(none_boundaries)):.3f}"},
        {"Metric": "MedianAbsBoundaryErrorMs", "Value": f"{median(boundary_errors):.3f}" if boundary_errors else ""},
        {"Metric": "MinGapSec", "Value": f"{args.min_gap_sec:.3f}"},
        {"Metric": "BoundaryWindowSec", "Value": f"{args.boundary_window_sec:.3f}"},
    ]
    write_csv(out_dir / "group_audio_alignment_summary.csv", summary, ["Metric", "Value"])

    print(f"Wrote {out_dir}")
    print(f"Cases: {cases_seen}; groups: {total}; exact 1:1: {exact} ({pct(exact, total):.1f}%)")
    print(f"Comma boundary gap match rate: {pct(sum(1 for r in comma_boundaries if r['MatchedAudioGap'] == 1), len(comma_boundaries)):.1f}%")
    print(f"Hard-punctuation boundary gap match rate: {pct(sum(1 for r in hard_boundaries if r['MatchedAudioGap'] == 1), len(hard_boundaries)):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
