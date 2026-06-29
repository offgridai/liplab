#!/usr/bin/env python3
"""Offline punctuation occupancy-servo policy sweep.

This is a research tool only.  It does not change runtime behavior.

The script uses the D01 punctuation resume oracle boundary table as a compact
representation of continuous audio occupancy around transcript punctuation:
text gives the expected boundary, and the oracle gives the acoustic
pause/resume estimate from the WAV.  It then simulates a family of bounded
"playback-rate servo" policies that would slow/speed the phrase restart toward
that acoustic resume without editing the whole timeline or stalling forever.

Typical usage after running D01:

    python scripts/sweep_occupancy_servo_policies.py --run experiments/runs/foo

or directly:

    python scripts/sweep_occupancy_servo_policies.py \
        --boundaries experiments/runs/foo/punctuation_oracle/punctuation_resume_oracle_boundaries.csv

Outputs are written to <run>/occupancy_servo_sweeps/ when --run is supplied,
or next to the boundary CSV otherwise.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ServoPolicy:
    name: str
    visual_lead_ms: float
    early_gain: float
    late_gain: float
    max_delay_ms: float
    max_advance_ms: float
    deadband_ms: float
    max_shift_ms: float


POLICIES: Sequence[ServoPolicy] = (
    # Baselines.
    ServoPolicy("planner_only", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ServoPolicy("lead60_planner_only", 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),

    # Gentle PLL-like controllers.  These intentionally do not fully snap.
    ServoPolicy("servo_gentle_60_120_80", 60.0, 0.35, 0.25, 120.0, 80.0, 35.0, 140.0),
    ServoPolicy("servo_gentle_60_160_100", 60.0, 0.40, 0.30, 160.0, 100.0, 30.0, 180.0),
    ServoPolicy("servo_balanced_60_220_140", 60.0, 0.55, 0.40, 220.0, 140.0, 25.0, 240.0),
    ServoPolicy("servo_balanced_40_220_140", 40.0, 0.55, 0.40, 220.0, 140.0, 25.0, 240.0),
    ServoPolicy("servo_balanced_80_220_140", 80.0, 0.55, 0.40, 220.0, 140.0, 25.0, 240.0),

    # More assertive, but still bounded.  These approximate a speed-controller
    # that is allowed to absorb larger punctuation-pause residuals.
    ServoPolicy("servo_assertive_60_300_180", 60.0, 0.75, 0.55, 300.0, 180.0, 20.0, 320.0),
    ServoPolicy("servo_assertive_40_300_180", 40.0, 0.75, 0.55, 300.0, 180.0, 20.0, 320.0),
    ServoPolicy("servo_delay_biased_60_320_120", 60.0, 0.85, 0.35, 320.0, 120.0, 20.0, 340.0),
    ServoPolicy("servo_delay_biased_80_320_120", 80.0, 0.85, 0.35, 320.0, 120.0, 20.0, 340.0),

    # Near-snap reference.  Useful for bounding the best case, but riskier for
    # runtime because it effectively lets occupancy dominate phrase starts.
    ServoPolicy("near_snap_60_360_220", 60.0, 0.95, 0.75, 360.0, 220.0, 10.0, 380.0),
)


REQUIRED_COLUMNS = {
    "CaseID",
    "BoundaryIndex",
    "Punctuation",
    "PunctuationKind",
    "Detected",
    "ResumeSec",
    "NextFirstCenterSec",
}


def _float(row: Dict[str, str], name: str, default: float = math.nan) -> float:
    value = row.get(name, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: Dict[str, str], name: str, default: int = 0) -> int:
    value = row.get(name, "")
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return statistics.median(values)


def mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return statistics.fmean(values)


def pctl(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def simulate(row: Dict[str, str], policy: ServoPolicy) -> Dict[str, float | str | int]:
    planned_sec = _float(row, "NextFirstCenterSec")
    resume_sec = _float(row, "ResumeSec")
    detected = _int(row, "Detected") == 1 and math.isfinite(resume_sec)

    if not math.isfinite(planned_sec):
        return {
            "Detected": int(detected),
            "Usable": 0,
            "PlannedStartSec": planned_sec,
            "TargetStartSec": math.nan,
            "CorrectedStartSec": planned_sec,
            "RawErrorMs": math.nan,
            "CorrectedErrorMs": math.nan,
            "AppliedShiftMs": 0.0,
            "Reason": "missing_planned_start",
        }

    if not detected:
        return {
            "Detected": 0,
            "Usable": 0,
            "PlannedStartSec": planned_sec,
            "TargetStartSec": math.nan,
            "CorrectedStartSec": planned_sec,
            "RawErrorMs": math.nan,
            "CorrectedErrorMs": math.nan,
            "AppliedShiftMs": 0.0,
            "Reason": row.get("RejectReason") or "not_detected",
        }

    target_sec = resume_sec - policy.visual_lead_ms / 1000.0
    raw_error_ms = (planned_sec - target_sec) * 1000.0

    # Negative raw error means planner would launch before the acoustic target;
    # a servo should slow down / delay.  Positive means planner is late;
    # the servo may speed up / advance.
    needed_shift_ms = -raw_error_ms
    if abs(needed_shift_ms) <= policy.deadband_ms:
        applied_ms = 0.0
        reason = "deadband"
    elif needed_shift_ms > 0.0:
        applied_ms = min(policy.max_delay_ms, needed_shift_ms * policy.early_gain)
        reason = "delay_toward_resume"
    else:
        applied_ms = -min(policy.max_advance_ms, abs(needed_shift_ms) * policy.late_gain)
        reason = "advance_toward_resume"

    applied_ms = clamp(applied_ms, -policy.max_shift_ms, policy.max_shift_ms)
    corrected_sec = planned_sec + applied_ms / 1000.0
    corrected_error_ms = (corrected_sec - target_sec) * 1000.0

    return {
        "Detected": 1,
        "Usable": 1,
        "PlannedStartSec": planned_sec,
        "TargetStartSec": target_sec,
        "CorrectedStartSec": corrected_sec,
        "RawErrorMs": raw_error_ms,
        "CorrectedErrorMs": corrected_error_ms,
        "AppliedShiftMs": applied_ms,
        "Reason": reason,
    }


def read_boundaries(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(policy_name: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    usable = [r for r in rows if int(r["Usable"]) == 1]
    raw_abs = [abs(float(r["RawErrorMs"])) for r in usable]
    corr_abs = [abs(float(r["CorrectedErrorMs"])) for r in usable]
    shifts = [float(r["AppliedShiftMs"]) for r in usable]
    corr = [float(r["CorrectedErrorMs"]) for r in usable]
    raw = [float(r["RawErrorMs"]) for r in usable]

    return {
        "Policy": policy_name,
        "BoundaryCount": len(rows),
        "UsableDetectedCount": len(usable),
        "UsableDetectedRate": len(usable) / len(rows) if rows else math.nan,
        "RawAbsErrorMeanMs": mean(raw_abs),
        "RawAbsErrorMedianMs": median(raw_abs),
        "CorrectedAbsErrorMeanMs": mean(corr_abs),
        "CorrectedAbsErrorMedianMs": median(corr_abs),
        "CorrectedAbsErrorP90Ms": pctl(corr_abs, 0.90),
        "ImprovementMeanMs": mean(raw_abs) - mean(corr_abs) if usable else math.nan,
        "SignedErrorMeanMs": mean(corr),
        "RawSignedErrorMeanMs": mean(raw),
        "AppliedShiftAbsMeanMs": mean([abs(s) for s in shifts]),
        "AppliedShiftP90AbsMs": pctl([abs(s) for s in shifts], 0.90),
        "AppliedDelayCount": sum(1 for s in shifts if s > 1e-6),
        "AppliedAdvanceCount": sum(1 for s in shifts if s < -1e-6),
        "BigShiftGE250msCount": sum(1 for s in shifts if abs(s) >= 250.0),
        "BigShiftGE350msCount": sum(1 for s in shifts if abs(s) >= 350.0),
        "StillEarlyGT150msCount": sum(1 for e in corr if e < -150.0),
        "StillLateGT150msCount": sum(1 for e in corr if e > 150.0),
    }


def group_summary(policy_name: str, rows: Sequence[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    out = []
    for name in sorted(groups):
        s = summarize(policy_name, groups[name])
        s[key] = name
        out.append(s)
    return out


def find_boundary_csv(args: argparse.Namespace) -> Path:
    if args.boundaries:
        return Path(args.boundaries)
    if not args.run:
        raise SystemExit("Provide --run or --boundaries")
    run = Path(args.run)
    candidate = run / "punctuation_oracle" / "punctuation_resume_oracle_boundaries.csv"
    if not candidate.exists():
        raise SystemExit(f"Could not find D01 boundary CSV at {candidate}")
    return candidate


def output_dir(args: argparse.Namespace, boundary_csv: Path) -> Path:
    if args.output:
        return Path(args.output)
    if args.run:
        return Path(args.run) / "occupancy_servo_sweeps"
    return boundary_csv.parent / "occupancy_servo_sweeps"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="LipLab run directory containing punctuation_oracle outputs")
    parser.add_argument("--boundaries", help="Path to punctuation_resume_oracle_boundaries.csv")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--write-by-boundary", action="store_true", help="Write per-boundary policy rows")
    args = parser.parse_args(argv)

    boundary_csv = find_boundary_csv(args)
    out_dir = output_dir(args, boundary_csv)
    boundaries = read_boundaries(boundary_csv)

    all_policy_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    by_kind_rows: List[Dict[str, object]] = []
    by_punc_rows: List[Dict[str, object]] = []

    for policy in POLICIES:
        policy_rows: List[Dict[str, object]] = []
        for row in boundaries:
            sim = simulate(row, policy)
            out_row: Dict[str, object] = {
                "Policy": policy.name,
                "CaseID": row.get("CaseID", ""),
                "BoundaryIndex": row.get("BoundaryIndex", ""),
                "Punctuation": row.get("Punctuation", ""),
                "PunctuationKind": row.get("PunctuationKind", ""),
                "PrevWord": row.get("PrevWord", ""),
                "NextWord": row.get("NextWord", ""),
                "ResumeSec": row.get("ResumeSec", ""),
                **sim,
            }
            policy_rows.append(out_row)
        all_policy_rows.extend(policy_rows)
        summary_rows.append(summarize(policy.name, policy_rows))
        by_kind_rows.extend(group_summary(policy.name, policy_rows, "PunctuationKind"))
        by_punc_rows.extend(group_summary(policy.name, policy_rows, "Punctuation"))

    summary_rows.sort(key=lambda r: (float(r["CorrectedAbsErrorMeanMs"]), float(r["AppliedShiftAbsMeanMs"])))

    summary_fields = [
        "Policy", "BoundaryCount", "UsableDetectedCount", "UsableDetectedRate",
        "RawAbsErrorMeanMs", "RawAbsErrorMedianMs",
        "CorrectedAbsErrorMeanMs", "CorrectedAbsErrorMedianMs", "CorrectedAbsErrorP90Ms",
        "ImprovementMeanMs", "RawSignedErrorMeanMs", "SignedErrorMeanMs",
        "AppliedShiftAbsMeanMs", "AppliedShiftP90AbsMs",
        "AppliedDelayCount", "AppliedAdvanceCount",
        "BigShiftGE250msCount", "BigShiftGE350msCount",
        "StillEarlyGT150msCount", "StillLateGT150msCount",
    ]
    write_csv(out_dir / "occupancy_servo_policy_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "occupancy_servo_policy_by_kind.csv", by_kind_rows, ["PunctuationKind"] + summary_fields)
    write_csv(out_dir / "occupancy_servo_policy_by_punctuation.csv", by_punc_rows, ["Punctuation"] + summary_fields)

    if args.write_by_boundary:
        by_boundary_fields = [
            "Policy", "CaseID", "BoundaryIndex", "Punctuation", "PunctuationKind",
            "PrevWord", "NextWord", "Detected", "Usable", "ResumeSec",
            "PlannedStartSec", "TargetStartSec", "CorrectedStartSec",
            "RawErrorMs", "CorrectedErrorMs", "AppliedShiftMs", "Reason",
        ]
        write_csv(out_dir / "occupancy_servo_policy_by_boundary.csv", all_policy_rows, by_boundary_fields)

    review = out_dir / "occupancy_servo_policy_review.md"
    with review.open("w", encoding="utf-8") as f:
        f.write("# Occupancy Servo Policy Sweep\n\n")
        f.write(f"Input boundary CSV: `{boundary_csv}`\n\n")
        f.write("This offline tool estimates whether continuous audio occupancy would be better used as a bounded playback-rate servo rather than as a timeline editor. It uses D01 detected punctuation resume points as the acoustic reference and simulates small speed-up/slow-down corrections of the next phrase start.\n\n")
        f.write("## Top policies\n\n")
        f.write("| Policy | Mean abs error ms | Median abs error ms | Mean improvement ms | Mean abs shift ms | Big shifts >=250ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows[:8]:
            f.write(
                f"| {row['Policy']} | {float(row['CorrectedAbsErrorMeanMs']):.1f} | "
                f"{float(row['CorrectedAbsErrorMedianMs']):.1f} | "
                f"{float(row['ImprovementMeanMs']):.1f} | "
                f"{float(row['AppliedShiftAbsMeanMs']):.1f} | "
                f"{int(row['BigShiftGE250msCount'])} |\n"
            )
        f.write("\n## Interpretation notes\n\n")
        f.write("- `planner_only` is the no-servo baseline against the same D01 resume targets.\n")
        f.write("- Policies with large improvements but many >=250ms shifts may be risky in runtime.\n")
        f.write("- A good runtime candidate should improve boundary error while keeping shifts bounded and avoiding many still-early/still-late residuals.\n")

    print(f"Wrote occupancy servo sweep outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
