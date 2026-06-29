#!/usr/bin/env python3
"""
Offline policy sweep for streamed landmark-driven local timing correction.

This script does not change runtime behavior.  It reads an existing liplab run
and replays multiple hypothetical landmark-warp policies against the per-case
CSV diagnostics.  The goal is to answer, from the 401-case corpus, which local
rules are worth trying in C++ before shipping another Unreal patch.

Expected input layout:
  <run>/per_case/case_XXXXXX/nudge_diagnostics.csv
  <run>/per_case/case_XXXXXX/timing_diagnostics.csv

Useful output:
  <out>/landmark_policy_sweep_summary.csv
  <out>/landmark_policy_sweep_by_case.csv
  <out>/landmark_policy_sweep_events.csv    (optional, --write-events)

The sweep treats text/prosody timing as the master schedule and streamed audio
landmarks as sparse anchors.  It evaluates families of local elastic policies:
  - event_only: anchors move only themselves
  - same_word: anchors influence events from the same source word only
  - same_phrase_*: anchors influence nearby events only inside the same phrase
  - local_*: anchors influence nearby events inside a time window, optionally
    blocked by phrase boundaries

This is intentionally offline and diagnostic-only.  It is not a new runtime
aligner and should not be ported to Unreal wholesale.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable


@dataclass(frozen=True)
class Event:
    case_id: str
    event_index: int
    pose_id: str
    source_word: str
    word_index: int
    phrase_index: int
    scheduled_sec: float
    current_sec: float
    evidence_sec: float | None
    evidence_confidence: float
    evidence_fallback: bool
    landmark_class: str
    nudge_accepted: bool
    anchor_target_sec: float | None
    anchor_confidence: float


@dataclass(frozen=True)
class Policy:
    name: str
    radius_ms: float
    anchor_weight: float
    neighbor_weight: float
    same_word_only: bool = False
    same_phrase_only: bool = True
    forward_only: bool = False
    asymmetric_delay_bias: float = 1.0
    max_anchor_shift_ms: float = 90.0
    max_neighbor_shift_ms: float = 45.0
    min_gap_ms: float = 6.0


@dataclass
class PolicyStats:
    policy: str
    case_count: int = 0
    event_count: int = 0
    scored_event_count: int = 0
    anchor_count: int = 0
    changed_event_count: int = 0
    abs_error_ms: list[float] | None = None
    signed_error_ms: list[float] | None = None
    shifts_ms: list[float] | None = None
    compressed_gap_under_16: int = 0
    compressed_gap_under_33: int = 0
    min_gap_ms: float = math.inf

    def __post_init__(self) -> None:
        self.abs_error_ms = []
        self.signed_error_ms = []
        self.shifts_ms = []


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_bool01(value: str | None) -> bool:
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def discover_case_dirs(run_dir: Path) -> list[Path]:
    per_case = run_dir / "per_case"
    if per_case.exists():
        return sorted([p for p in per_case.iterdir() if p.is_dir()])
    # Also support unpacked Unreal-style debug dirs for quick experiments.
    return sorted([p for p in run_dir.rglob("runtime_commit_events.csv") if p.parent.is_dir()], key=lambda p: str(p.parent))


def case_id_from_dir(case_dir_or_file: Path) -> str:
    p = case_dir_or_file.parent if case_dir_or_file.name == "runtime_commit_events.csv" else case_dir_or_file
    return p.name


def load_events(case_dir_or_file: Path) -> list[Event]:
    case_dir = case_dir_or_file.parent if case_dir_or_file.name == "runtime_commit_events.csv" else case_dir_or_file
    case_id = case_id_from_dir(case_dir)

    nudge_rows = read_csv_dicts(case_dir / "nudge_diagnostics.csv")
    timing_rows = read_csv_dicts(case_dir / "timing_diagnostics.csv")
    commit_rows = read_csv_dicts(case_dir / "runtime_commit_events.csv")

    timing_by_index: dict[int, dict[str, str]] = {
        parse_int(r.get("EventIndex"), -1): r for r in timing_rows
    }

    events: list[Event] = []
    if nudge_rows:
        for r in nudge_rows:
            idx = parse_int(r.get("EventIndex"), -1)
            if idx < 0:
                continue
            t = timing_by_index.get(idx, {})
            scheduled = parse_float(r.get("NudgeScheduledCenterSec"), parse_float(r.get("PlannedCenterSec")))
            current = parse_float(r.get("FinalCenterSec"), scheduled)
            candidate = parse_float(r.get("NudgeCandidateAppliedCenterSec"), 0.0)
            raw_candidate = parse_float(r.get("NudgeCandidateRawCenterSec"), 0.0)
            target = candidate if candidate > 0.0 else (raw_candidate if raw_candidate > 0.0 else None)
            ev_conf = parse_float(t.get("EvidenceConfidence"), 0.0)
            ev_center = parse_float(t.get("EvidenceCenterSec"), 0.0)
            events.append(Event(
                case_id=case_id,
                event_index=idx,
                pose_id=r.get("PoseID", ""),
                source_word=r.get("SourceWord", ""),
                word_index=parse_int(r.get("WordIndex"), -1),
                phrase_index=parse_int(r.get("PhraseIndex"), 0),
                scheduled_sec=scheduled,
                current_sec=current,
                evidence_sec=ev_center if ev_center > 0.0 else None,
                evidence_confidence=ev_conf,
                evidence_fallback=parse_bool01(t.get("EvidenceFallback")),
                landmark_class=r.get("LandmarkClass", ""),
                nudge_accepted=parse_bool01(r.get("NudgeAccepted")),
                anchor_target_sec=target,
                anchor_confidence=parse_float(r.get("NudgeCandidateConfidence"), 0.0),
            ))
    elif commit_rows:
        # Fallback for Unreal debug dirs lacking nudge/timing diagnostics.  This
        # path cannot score against oracle evidence, but it still lets the script
        # report topology/compression on the committed track.
        for r in commit_rows:
            idx = parse_int(r.get("EventIndex"), -1)
            center = parse_float(r.get("FinalRenderCenterSec"), 0.0)
            events.append(Event(
                case_id=case_id,
                event_index=idx,
                pose_id=r.get("PoseID", ""),
                source_word=r.get("SourceWord", ""),
                word_index=-1,
                phrase_index=parse_int(r.get("TextIslandIndex"), 0),
                scheduled_sec=center,
                current_sec=center,
                evidence_sec=None,
                evidence_confidence=0.0,
                evidence_fallback=False,
                landmark_class="",
                nudge_accepted=False,
                anchor_target_sec=None,
                anchor_confidence=0.0,
            ))
    return sorted(events, key=lambda e: e.event_index)


def default_policies() -> list[Policy]:
    return [
        Policy("scheduled_no_audio", radius_ms=0, anchor_weight=0.0, neighbor_weight=0.0, same_phrase_only=False),
        Policy("event_only", radius_ms=0, anchor_weight=1.0, neighbor_weight=0.0, same_phrase_only=False),
        Policy("same_word_soft", radius_ms=180, anchor_weight=1.0, neighbor_weight=0.45, same_word_only=True, same_phrase_only=False, max_neighbor_shift_ms=30),
        Policy("same_phrase_150", radius_ms=150, anchor_weight=1.0, neighbor_weight=0.45, same_phrase_only=True, max_neighbor_shift_ms=30),
        Policy("same_phrase_250", radius_ms=250, anchor_weight=1.0, neighbor_weight=0.55, same_phrase_only=True, max_neighbor_shift_ms=40),
        Policy("same_phrase_350", radius_ms=350, anchor_weight=1.0, neighbor_weight=0.60, same_phrase_only=True, max_neighbor_shift_ms=50),
        Policy("same_phrase_forward_250", radius_ms=250, anchor_weight=1.0, neighbor_weight=0.55, same_phrase_only=True, forward_only=True, max_neighbor_shift_ms=40),
        Policy("same_phrase_delay_biased_250", radius_ms=250, anchor_weight=1.0, neighbor_weight=0.55, same_phrase_only=True, asymmetric_delay_bias=1.35, max_neighbor_shift_ms=45),
        Policy("local_no_phrase_barrier_150", radius_ms=150, anchor_weight=1.0, neighbor_weight=0.35, same_phrase_only=False, max_neighbor_shift_ms=25),
    ]


def compatible(anchor: Event, event: Event, policy: Policy) -> bool:
    if policy.same_word_only and anchor.source_word != event.source_word:
        return False
    if policy.same_phrase_only and anchor.phrase_index != event.phrase_index:
        return False
    if policy.forward_only and event.scheduled_sec < anchor.scheduled_sec:
        return False
    return True


def apply_policy(events: list[Event], policy: Policy) -> list[float]:
    centers = [e.scheduled_sec for e in events]
    anchors = [e for e in events if e.nudge_accepted and e.anchor_target_sec is not None]

    if policy.name == "scheduled_no_audio":
        return repair_monotonic(centers, policy.min_gap_ms * 0.001)

    # Event-only and elastic variants start from scheduled centers, not the
    # already-mutated C++ result.  This lets policies compete fairly offline.
    shifts = [0.0 for _ in events]
    weights = [0.0 for _ in events]
    index_by_event = {e.event_index: i for i, e in enumerate(events)}

    for anchor in anchors:
        ai = index_by_event.get(anchor.event_index)
        if ai is None:
            continue
        raw_shift = (anchor.anchor_target_sec or anchor.scheduled_sec) - anchor.scheduled_sec
        if raw_shift >= 0.0:
            raw_shift *= policy.asymmetric_delay_bias
        anchor_shift = max(-policy.max_anchor_shift_ms * 0.001, min(policy.max_anchor_shift_ms * 0.001, raw_shift))
        for i, event in enumerate(events):
            if not compatible(anchor, event, policy):
                continue
            dist_ms = abs((event.scheduled_sec - anchor.scheduled_sec) * 1000.0)
            if i != ai and (policy.radius_ms <= 0.0 or dist_ms > policy.radius_ms):
                continue
            if i == ai:
                weight = policy.anchor_weight
                capped = anchor_shift
            else:
                # Smooth triangular falloff avoids long-range warps and makes the
                # sweep easy to reason about.
                falloff = max(0.0, 1.0 - dist_ms / max(1.0, policy.radius_ms))
                weight = policy.neighbor_weight * falloff
                cap = policy.max_neighbor_shift_ms * 0.001
                capped = max(-cap, min(cap, anchor_shift))
            if weight <= 0.0:
                continue
            shifts[i] += capped * weight
            weights[i] += weight

    for i in range(len(events)):
        if weights[i] > 0.0:
            centers[i] += shifts[i] / weights[i]
    return repair_monotonic(centers, policy.min_gap_ms * 0.001)


def repair_monotonic(centers: list[float], min_gap_sec: float) -> list[float]:
    out = list(centers)
    prev: float | None = None
    for i, center in enumerate(out):
        if prev is not None and center < prev + min_gap_sec:
            center = prev + min_gap_sec
            out[i] = center
        prev = center
    return out


def score_case(events: list[Event], centers: list[float], policy_name: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    abs_errors: list[float] = []
    signed_errors: list[float] = []
    shifts: list[float] = []
    scored = 0
    changed = 0
    anchors = 0
    gaps_under_16 = 0
    gaps_under_33 = 0
    min_gap_ms = math.inf
    event_rows: list[dict[str, str]] = []

    for i, event in enumerate(events):
        new_center = centers[i]
        shift_ms = (new_center - event.scheduled_sec) * 1000.0
        shifts.append(abs(shift_ms))
        if abs(shift_ms) > 0.5:
            changed += 1
        if event.nudge_accepted:
            anchors += 1
        if event.evidence_sec is not None and not event.evidence_fallback:
            err_ms = (new_center - event.evidence_sec) * 1000.0
            abs_errors.append(abs(err_ms))
            signed_errors.append(err_ms)
            scored += 1
            event_rows.append({
                "policy": policy_name,
                "case_id": event.case_id,
                "event_index": str(event.event_index),
                "pose_id": event.pose_id,
                "source_word": event.source_word,
                "phrase_index": str(event.phrase_index),
                "scheduled_sec": f"{event.scheduled_sec:.6f}",
                "policy_center_sec": f"{new_center:.6f}",
                "evidence_sec": f"{event.evidence_sec:.6f}",
                "error_ms": f"{err_ms:.3f}",
                "shift_ms": f"{shift_ms:.3f}",
                "nudge_accepted": "1" if event.nudge_accepted else "0",
                "landmark_class": event.landmark_class,
            })

    for a, b in zip(centers, centers[1:]):
        gap_ms = (b - a) * 1000.0
        min_gap_ms = min(min_gap_ms, gap_ms)
        if gap_ms < 16.0:
            gaps_under_16 += 1
        if gap_ms < 33.0:
            gaps_under_33 += 1

    row = {
        "policy": policy_name,
        "case_id": events[0].case_id if events else "",
        "event_count": str(len(events)),
        "scored_event_count": str(scored),
        "anchor_count": str(anchors),
        "changed_event_count": str(changed),
        "mae_ms": f"{mean(abs_errors):.3f}" if abs_errors else "",
        "median_abs_error_ms": f"{median(abs_errors):.3f}" if abs_errors else "",
        "p90_abs_error_ms": f"{percentile(abs_errors, 0.90):.3f}" if abs_errors else "",
        "lead_bias_ms": f"{mean(signed_errors):.3f}" if signed_errors else "",
        "max_abs_shift_ms": f"{max(shifts):.3f}" if shifts else "0.000",
        "mean_abs_shift_ms": f"{mean(shifts):.3f}" if shifts else "0.000",
        "gap_under_16_count": str(gaps_under_16),
        "gap_under_33_count": str(gaps_under_33),
        "min_gap_ms": f"{min_gap_ms:.3f}" if min_gap_ms < math.inf else "",
    }
    return row, event_rows


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_policy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)

    out: list[dict[str, str]] = []
    for policy, group in sorted(by_policy.items()):
        weighted_abs_sum = 0.0
        weighted_count = 0
        changed = 0
        anchors = 0
        events = 0
        gap16 = 0
        gap33 = 0
        min_gap = math.inf
        case_maes: list[float] = []
        p90s: list[float] = []
        shifts: list[float] = []
        for row in group:
            scored = parse_int(row["scored_event_count"])
            mae = parse_float(row["mae_ms"], math.nan)
            if scored > 0 and not math.isnan(mae):
                weighted_abs_sum += mae * scored
                weighted_count += scored
                case_maes.append(mae)
            p90 = parse_float(row["p90_abs_error_ms"], math.nan)
            if not math.isnan(p90):
                p90s.append(p90)
            changed += parse_int(row["changed_event_count"])
            anchors += parse_int(row["anchor_count"])
            events += parse_int(row["event_count"])
            gap16 += parse_int(row["gap_under_16_count"])
            gap33 += parse_int(row["gap_under_33_count"])
            mg = parse_float(row["min_gap_ms"], math.inf)
            min_gap = min(min_gap, mg)
            shifts.append(parse_float(row["mean_abs_shift_ms"], 0.0))

        out.append({
            "policy": policy,
            "case_count": str(len(group)),
            "event_count": str(events),
            "scored_event_count": str(weighted_count),
            "anchor_count": str(anchors),
            "changed_event_count": str(changed),
            "corpus_mae_ms": f"{(weighted_abs_sum / weighted_count):.3f}" if weighted_count else "",
            "case_mae_mean_ms": f"{mean(case_maes):.3f}" if case_maes else "",
            "case_mae_median_ms": f"{median(case_maes):.3f}" if case_maes else "",
            "case_p90_mean_ms": f"{mean(p90s):.3f}" if p90s else "",
            "mean_abs_shift_ms": f"{mean(shifts):.3f}" if shifts else "0.000",
            "gap_under_16_count": str(gap16),
            "gap_under_33_count": str(gap33),
            "min_gap_ms": f"{min_gap:.3f}" if min_gap < math.inf else "",
        })
    out.sort(key=lambda r: (parse_float(r.get("corpus_mae_ms"), math.inf), parse_int(r.get("gap_under_33_count"), 10**9)))
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Existing liplab run directory, or unpacked Unreal debug directory")
    ap.add_argument("--out", default=None, help="Output directory. Defaults to <run>/policy_sweeps")
    ap.add_argument("--write-events", action="store_true", help="Write per-event rows for every policy; large on full corpus")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "policy_sweeps"
    case_entries = discover_case_dirs(run_dir)
    if not case_entries:
        raise SystemExit(f"No case directories found under {run_dir}")

    policies = default_policies()
    by_case_rows: list[dict[str, str]] = []
    event_rows_all: list[dict[str, str]] = []
    skipped = 0

    for entry in case_entries:
        events = load_events(entry)
        if not events:
            skipped += 1
            continue
        for policy in policies:
            centers = apply_policy(events, policy)
            row, event_rows = score_case(events, centers, policy.name)
            by_case_rows.append(row)
            if args.write_events:
                event_rows_all.extend(event_rows)

    summary_rows = aggregate(by_case_rows)
    write_csv(out_dir / "landmark_policy_sweep_summary.csv", summary_rows)
    write_csv(out_dir / "landmark_policy_sweep_by_case.csv", by_case_rows)
    if args.write_events:
        write_csv(out_dir / "landmark_policy_sweep_events.csv", event_rows_all)

    print(f"Cases read: {len(case_entries) - skipped}/{len(case_entries)}")
    print(f"Policies: {len(policies)}")
    print(f"Output: {out_dir}")
    if summary_rows:
        best = summary_rows[0]
        print(f"Best corpus MAE: {best.get('policy')} {best.get('corpus_mae_ms')} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
