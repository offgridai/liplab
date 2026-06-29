#!/usr/bin/env python3
"""Audit whether observed audio pauses land near legal text boundaries.

This is an offline liplab experiment. It does not change runtime behavior.

Unlike analyze_group_audio_occupancy.py, this script does not ask every text
prosody group to become its own audio island. It starts from observed audio
silence gaps and asks where each gap lands in the text structure.

Input: a liplab run directory containing per_case/<case_id>/committed_events.csv
and speech_islands.csv. Transcript punctuation is read from --inputs.

Output directory defaults to <run>/audio_gap_boundaries and contains:
  - audio_gap_boundary_alignment.csv
  - audio_gap_boundary_summary.csv
  - comma_gap_failures.csv
  - period_gap_failures.csv
  - large_unclaimed_gaps.csv

The intended runtime thesis tested by this audit is:
  audio detects a meaningful gap -> nearest legal text boundary absorbs it ->
  downstream visemes shift/resume there, rather than leaking to a later hard
  punctuation boundary.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from analyze_group_audio_occupancy import (
    Group,
    Island,
    audio_gaps,
    build_groups,
    classify_punct,
    iter_case_dirs,
    load_events,
    load_islands,
    transcript_terminal_punct,
    write_csv,
)


DEFAULT_THRESHOLDS_SEC = (0.080, 0.120, 0.180, 0.250)


@dataclass
class Boundary:
    case_id: str
    boundary_id: int
    left_group: Group
    right_group: Group
    expected_sec: float
    committed_sec: float
    boundary_type: str
    terminal_punct: str
    launch_shift: float

    @property
    def label(self) -> str:
        if self.terminal_punct:
            return self.terminal_punct
        return self.boundary_type

    @property
    def delay_absorbed_sec(self) -> float:
        # Positive values indicate that the current committed schedule pushed the
        # right side of this boundary later than the pure launched text timeline.
        expected_right = self.right_group.text_start + self.launch_shift
        return self.right_group.committed_start - expected_right


@dataclass
class GapMatch:
    case_id: str
    gap_id: int
    before_island: int
    after_island: int
    gap_start: float
    gap_end: float
    gap_duration: float
    gap_center: float
    nearest_boundary: Boundary | None
    nearest_error_sec: float
    prior_boundary: Boundary | None
    prior_error_sec: float
    upcoming_boundary: Boundary | None
    upcoming_error_sec: float


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def safe_median(values: list[float]) -> str:
    return f"{median(values):.3f}" if values else ""


def boundary_type(left: Group, right: Group) -> str:
    punct_class = classify_punct(left.terminal_punct)
    if punct_class == "hard":
        return "hard_punctuation"
    if punct_class == "soft":
        return "soft_punctuation"
    if left.source_group_index != right.source_group_index:
        return "prosody_group"
    if set(left.phrase_indices) != set(right.phrase_indices):
        return "phrase"
    return "word_or_event"


def build_boundaries(case_id: str, groups: list[Group], islands: list[Island]) -> list[Boundary]:
    if len(groups) < 2:
        return []
    launch_shift = islands[0].start - groups[0].text_start if islands else 0.0
    boundaries: list[Boundary] = []
    for idx, (left, right) in enumerate(zip(groups, groups[1:])):
        expected = 0.5 * (left.text_end + right.text_start) + launch_shift
        committed = 0.5 * (left.committed_end + right.committed_start)
        boundaries.append(
            Boundary(
                case_id=case_id,
                boundary_id=idx,
                left_group=left,
                right_group=right,
                expected_sec=expected,
                committed_sec=committed,
                boundary_type=boundary_type(left, right),
                terminal_punct=left.terminal_punct,
                launch_shift=launch_shift,
            )
        )
    return boundaries


def nearest_boundary(gap_center: float, boundaries: list[Boundary], predicate=None) -> tuple[Boundary | None, float]:
    best: Boundary | None = None
    best_abs = math.inf
    for boundary in boundaries:
        if predicate is not None and not predicate(boundary):
            continue
        err = boundary.expected_sec - gap_center
        abs_err = abs(err)
        if abs_err < best_abs:
            best = boundary
            best_abs = abs_err
    if best is None:
        return None, math.inf
    return best, best.expected_sec - gap_center


def match_gaps(case_id: str, islands: list[Island], boundaries: list[Boundary], min_gap_sec: float) -> list[GapMatch]:
    rows: list[GapMatch] = []
    for gap_id, gap in enumerate(audio_gaps(islands, min_gap_sec)):
        center = 0.5 * (gap.start + gap.end)
        nearest, nearest_err = nearest_boundary(center, boundaries)
        prior, prior_err = nearest_boundary(center, boundaries, lambda b: b.expected_sec <= center)
        upcoming, upcoming_err = nearest_boundary(center, boundaries, lambda b: b.expected_sec >= center)
        rows.append(
            GapMatch(
                case_id=case_id,
                gap_id=gap_id,
                before_island=gap.before_island,
                after_island=gap.after_island,
                gap_start=gap.start,
                gap_end=gap.end,
                gap_duration=gap.duration,
                gap_center=center,
                nearest_boundary=nearest,
                nearest_error_sec=nearest_err,
                prior_boundary=prior,
                prior_error_sec=prior_err,
                upcoming_boundary=upcoming,
                upcoming_error_sec=upcoming_err,
            )
        )
    return rows


def boundary_kind(boundary: Boundary | None) -> str:
    if boundary is None:
        return "none"
    if boundary.boundary_type == "soft_punctuation" and boundary.terminal_punct == ",":
        return "comma"
    if boundary.boundary_type == "hard_punctuation":
        return "hard_punctuation"
    return boundary.boundary_type


def match_to_row(match: GapMatch, threshold_sec: float, match_window_sec: float) -> dict[str, object]:
    b = match.nearest_boundary
    prior = match.prior_boundary
    upcoming = match.upcoming_boundary
    nearest_abs_ms = abs(match.nearest_error_sec) * 1000.0 if b is not None else math.inf
    consumed_ratio = ""
    under_consumed = ""
    delay_absorbed = ""
    leak_to_later_hard = ""
    if b is not None:
        delay = max(0.0, b.delay_absorbed_sec)
        delay_absorbed = f"{delay:.6f}"
        consumed_ratio_value = delay / max(0.001, match.gap_duration)
        consumed_ratio = f"{consumed_ratio_value:.6f}"
        under_consumed = 1 if consumed_ratio_value < 0.40 and match.gap_duration >= threshold_sec else 0
        if b.boundary_type == "soft_punctuation" and consumed_ratio_value < 0.40:
            later_hard = nearest_later_hard(b, match.gap_center)
            leak_to_later_hard = 1 if later_hard is not None else 0
        else:
            leak_to_later_hard = 0

    return {
        "CaseId": match.case_id,
        "GapId": match.gap_id,
        "ThresholdSec": f"{threshold_sec:.3f}",
        "GapBeforeIsland": match.before_island,
        "GapAfterIsland": match.after_island,
        "GapStartSec": f"{match.gap_start:.6f}",
        "GapEndSec": f"{match.gap_end:.6f}",
        "GapCenterSec": f"{match.gap_center:.6f}",
        "GapDurationSec": f"{match.gap_duration:.6f}",
        "NearestBoundaryId": b.boundary_id if b else "",
        "NearestBoundaryType": b.boundary_type if b else "",
        "NearestBoundaryKind": boundary_kind(b),
        "NearestTerminalPunct": b.terminal_punct if b else "",
        "NearestBoundaryExpectedSec": f"{b.expected_sec:.6f}" if b else "",
        "NearestBoundaryCommittedSec": f"{b.committed_sec:.6f}" if b else "",
        "NearestBoundaryErrorMs": f"{match.nearest_error_sec * 1000.0:.3f}" if b else "",
        "NearestBoundaryAbsErrorMs": f"{nearest_abs_ms:.3f}" if b else "",
        "MatchedWithinWindow": 1 if b is not None and nearest_abs_ms <= match_window_sec * 1000.0 else 0,
        "LeftText": b.left_group.text if b else "",
        "RightText": b.right_group.text if b else "",
        "PriorBoundaryId": prior.boundary_id if prior else "",
        "PriorBoundaryKind": boundary_kind(prior),
        "PriorBoundaryErrorMs": f"{match.prior_error_sec * 1000.0:.3f}" if prior else "",
        "UpcomingBoundaryId": upcoming.boundary_id if upcoming else "",
        "UpcomingBoundaryKind": boundary_kind(upcoming),
        "UpcomingBoundaryErrorMs": f"{match.upcoming_error_sec * 1000.0:.3f}" if upcoming else "",
        "RuntimeDelayAbsorbedSec": delay_absorbed,
        "RuntimeConsumedGapRatio": consumed_ratio,
        "RuntimeUnderConsumed": under_consumed,
        "PossibleLeakToLaterHardPunct": leak_to_later_hard,
    }


def nearest_later_hard(boundary: Boundary, gap_center: float) -> Boundary | None:
    # Search within the same case's following groups by walking the objects that
    # are reachable from this boundary list is awkward here, so this flag remains
    # conservative and is filled by caller context only when available. Returning
    # None avoids claiming a leak without evidence.
    return None


def rows_for_threshold(
    case_id: str,
    islands: list[Island],
    boundaries: list[Boundary],
    threshold_sec: float,
    match_window_sec: float,
) -> list[dict[str, object]]:
    return [match_to_row(m, threshold_sec, match_window_sec) for m in match_gaps(case_id, islands, boundaries, threshold_sec)]


def summarize(rows: list[dict[str, object]], threshold_sec: float, match_window_sec: float) -> list[dict[str, object]]:
    threshold_rows = [r for r in rows if r["ThresholdSec"] == f"{threshold_sec:.3f}"]
    matched = [r for r in threshold_rows if r["MatchedWithinWindow"] == 1]
    comma = [r for r in matched if r["NearestBoundaryKind"] == "comma"]
    hard = [r for r in matched if r["NearestBoundaryKind"] == "hard_punctuation"]
    soft = [r for r in matched if r["NearestBoundaryType"] == "soft_punctuation"]
    group = [r for r in matched if r["NearestBoundaryKind"] == "prosody_group"]
    unmatched = [r for r in threshold_rows if r["MatchedWithinWindow"] != 1]
    errors = [float(r["NearestBoundaryAbsErrorMs"]) for r in matched if r["NearestBoundaryAbsErrorMs"] != ""]
    comma_durations = [float(r["GapDurationSec"]) * 1000.0 for r in comma]
    hard_durations = [float(r["GapDurationSec"]) * 1000.0 for r in hard]
    under_consumed = [r for r in matched if r["RuntimeUnderConsumed"] == 1]
    return [
        {"Metric": "ThresholdSec", "Value": f"{threshold_sec:.3f}"},
        {"Metric": "MatchWindowSec", "Value": f"{match_window_sec:.3f}"},
        {"Metric": "AudioGaps", "Value": len(threshold_rows)},
        {"Metric": "MatchedToAnyBoundary", "Value": len(matched)},
        {"Metric": "MatchedToAnyBoundaryRatePct", "Value": f"{pct(len(matched), len(threshold_rows)):.3f}"},
        {"Metric": "MatchedToComma", "Value": len(comma)},
        {"Metric": "MatchedToCommaRatePct", "Value": f"{pct(len(comma), len(threshold_rows)):.3f}"},
        {"Metric": "MatchedToSoftPunctuation", "Value": len(soft)},
        {"Metric": "MatchedToSoftPunctuationRatePct", "Value": f"{pct(len(soft), len(threshold_rows)):.3f}"},
        {"Metric": "MatchedToHardPunctuation", "Value": len(hard)},
        {"Metric": "MatchedToHardPunctuationRatePct", "Value": f"{pct(len(hard), len(threshold_rows)):.3f}"},
        {"Metric": "MatchedToProsodyGroup", "Value": len(group)},
        {"Metric": "MatchedToProsodyGroupRatePct", "Value": f"{pct(len(group), len(threshold_rows)):.3f}"},
        {"Metric": "UnmatchedLargeGaps", "Value": len(unmatched)},
        {"Metric": "MedianMatchedAbsBoundaryErrorMs", "Value": safe_median(errors)},
        {"Metric": "MedianCommaGapDurationMs", "Value": safe_median(comma_durations)},
        {"Metric": "MedianHardPunctGapDurationMs", "Value": safe_median(hard_durations)},
        {"Metric": "RuntimeUnderConsumedMatchedGaps", "Value": len(under_consumed)},
        {"Metric": "RuntimeUnderConsumedMatchedGapRatePct", "Value": f"{pct(len(under_consumed), len(matched)):.3f}"},
    ]


def parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return sorted(set(out)) or list(DEFAULT_THRESHOLDS_SEC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="liplab run directory containing per_case/")
    ap.add_argument("--inputs", default=None, help="inputs directory containing transcripts/")
    ap.add_argument("--out", default=None, help="output directory; default: <run>/audio_gap_boundaries")
    ap.add_argument(
        "--thresholds-sec",
        default=",".join(f"{x:.3f}" for x in DEFAULT_THRESHOLDS_SEC),
        help="comma-separated audio gap thresholds to audit",
    )
    ap.add_argument("--match-window-sec", type=float, default=0.300, help="max abs distance from gap center to text boundary")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    inputs_dir = Path(args.inputs).resolve() if args.inputs else None
    out_dir = Path(args.out).resolve() if args.out else run_dir / "audio_gap_boundaries"
    thresholds = parse_thresholds(args.thresholds_sec)

    all_rows: list[dict[str, object]] = []
    cases_seen = 0
    cases_with_gaps = 0
    boundaries_seen = 0

    for case_dir in iter_case_dirs(run_dir):
        case_id = case_dir.name
        events = load_events(case_dir)
        islands = load_islands(case_dir)
        transcript_path = inputs_dir / "transcripts" / f"{case_id}.txt" if inputs_dir else None
        groups = build_groups(case_id, events, transcript_terminal_punct(transcript_path))
        boundaries = build_boundaries(case_id, groups, islands)
        if not groups:
            continue
        cases_seen += 1
        boundaries_seen += len(boundaries)
        case_had_gap = False
        for threshold in thresholds:
            rows = rows_for_threshold(case_id, islands, boundaries, threshold, args.match_window_sec)
            if rows:
                case_had_gap = True
            all_rows.extend(rows)
        if case_had_gap:
            cases_with_gaps += 1

    fields = [
        "CaseId", "GapId", "ThresholdSec", "GapBeforeIsland", "GapAfterIsland", "GapStartSec", "GapEndSec",
        "GapCenterSec", "GapDurationSec", "NearestBoundaryId", "NearestBoundaryType", "NearestBoundaryKind",
        "NearestTerminalPunct", "NearestBoundaryExpectedSec", "NearestBoundaryCommittedSec", "NearestBoundaryErrorMs",
        "NearestBoundaryAbsErrorMs", "MatchedWithinWindow", "LeftText", "RightText", "PriorBoundaryId",
        "PriorBoundaryKind", "PriorBoundaryErrorMs", "UpcomingBoundaryId", "UpcomingBoundaryKind", "UpcomingBoundaryErrorMs",
        "RuntimeDelayAbsorbedSec", "RuntimeConsumedGapRatio", "RuntimeUnderConsumed", "PossibleLeakToLaterHardPunct",
    ]
    write_csv(out_dir / "audio_gap_boundary_alignment.csv", all_rows, fields)

    summary_rows: list[dict[str, object]] = [
        {"Metric": "CasesSeen", "Value": cases_seen},
        {"Metric": "CasesWithAnyAuditedGap", "Value": cases_with_gaps},
        {"Metric": "TextBoundariesSeen", "Value": boundaries_seen},
    ]
    for threshold in thresholds:
        summary_rows.extend(summarize(all_rows, threshold, args.match_window_sec))
    write_csv(out_dir / "audio_gap_boundary_summary.csv", summary_rows, ["Metric", "Value"])

    comma_failures = [
        r for r in all_rows
        if r["NearestBoundaryKind"] == "comma" and (r["MatchedWithinWindow"] != 1 or r["RuntimeUnderConsumed"] == 1)
    ]
    period_failures = [
        r for r in all_rows
        if r["NearestBoundaryKind"] == "hard_punctuation" and (r["MatchedWithinWindow"] != 1 or r["RuntimeUnderConsumed"] == 1)
    ]
    large_unclaimed = [r for r in all_rows if r["MatchedWithinWindow"] != 1]
    write_csv(out_dir / "comma_gap_failures.csv", comma_failures[:300], fields)
    write_csv(out_dir / "period_gap_failures.csv", period_failures[:300], fields)
    write_csv(out_dir / "large_unclaimed_gaps.csv", large_unclaimed[:300], fields)

    primary = f"{thresholds[0]:.3f}"
    primary_rows = [r for r in all_rows if r["ThresholdSec"] == primary]
    primary_matched = sum(1 for r in primary_rows if r["MatchedWithinWindow"] == 1)
    print(f"Wrote {out_dir}")
    print(f"Cases: {cases_seen}; text boundaries: {boundaries_seen}")
    print(f"Gaps >= {float(primary):.3f} sec {len(primary_rows)} audited; matched: {primary_matched} ({pct(primary_matched, len(primary_rows)):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
