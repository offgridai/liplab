#!/usr/bin/env python3
"""
D02 offline punctuation prior review.

This is a research/planner-calibration tool only.  It consumes the D01
punctuation resume oracle output and estimates useful planner priors for
punctuation gaps.  It does not change runtime behavior.

The key question is:

    Given a punctuation boundary and the detected acoustic resume, how much gap
    should the text/prosody planner allocate before the next phrase starts?

Inputs:
  <run>/punctuation_oracle/punctuation_resume_oracle_boundaries.csv

Outputs:
  <out>/punctuation_prior_by_class.csv
  <out>/punctuation_prior_by_punctuation.csv
  <out>/punctuation_prior_policy_summary.csv
  <out>/punctuation_prior_recommendations.csv
  <out>/punctuation_prior_review.md

Usage:
  python scripts/review_punctuation_priors.py --run experiments/runs/b09

or:
  python scripts/review_punctuation_priors.py \
      --boundaries experiments/runs/b09/punctuation_oracle/punctuation_resume_oracle_boundaries.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

POLICY_PREFIXES = [
    "loose_30pct_40ms",
    "loose_35pct_50ms",
    "balanced_40pct_60ms",
    "balanced_45pct_80ms",
    "strict_50pct_100ms",
]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        s = str(row.get(key, "")).strip()
        return float(s) if s else default
    except Exception:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        s = str(row.get(key, "")).strip()
        return int(float(s)) if s else default
    except Exception:
        return default


def finite(v: float) -> bool:
    return math.isfinite(v)


def pct(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if finite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def fmt_ms(v: float) -> str:
    if not finite(v):
        return ""
    return f"{v:.1f}"


def round_nearest(v: float, step: float = 10.0) -> float:
    if not finite(v):
        return math.nan
    return round(v / step) * step


def robust_stats(values: Iterable[float]) -> dict[str, float]:
    vals = [v for v in values if finite(v)]
    if not vals:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "trimmed_mean": math.nan,
        }
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    lo = int(n * 0.10)
    hi = max(lo + 1, int(math.ceil(n * 0.90)))
    trimmed = vals_sorted[lo:hi]
    return {
        "count": n,
        "mean": mean(vals_sorted),
        "median": median(vals_sorted),
        "p10": pct(vals_sorted, 0.10),
        "p25": pct(vals_sorted, 0.25),
        "p75": pct(vals_sorted, 0.75),
        "p90": pct(vals_sorted, 0.90),
        "trimmed_mean": mean(trimmed) if trimmed else math.nan,
    }


def derive_boundary_measure(row: dict[str, str], resume_sec: float) -> dict[str, float]:
    prev_last = as_float(row, "PrevLastCenterSec")
    next_first = as_float(row, "NextFirstCenterSec")
    current_gap_ms = as_float(row, "CommittedCenterGapMs")
    raw_gap_ms = as_float(row, "RawCenterGapMs")
    if not finite(current_gap_ms) and finite(prev_last) and finite(next_first):
        current_gap_ms = (next_first - prev_last) * 1000.0
    acoustic_gap_ms = (resume_sec - prev_last) * 1000.0 if finite(resume_sec) and finite(prev_last) else math.nan
    next_start_error_ms = (next_first - resume_sec) * 1000.0 if finite(resume_sec) and finite(next_first) else math.nan
    additional_delay_ms = -next_start_error_ms if finite(next_start_error_ms) else math.nan
    return {
        "current_gap_ms": current_gap_ms,
        "raw_gap_ms": raw_gap_ms,
        "acoustic_gap_ms": acoustic_gap_ms,
        "next_start_error_ms": next_start_error_ms,
        "additional_delay_ms": additional_delay_ms,
    }


def summarize_group(rows: list[dict[str, object]], label: str, total_boundaries: int) -> dict[str, object]:
    detected = [r for r in rows if r.get("detected")]
    acoustic_gap = [float(r["acoustic_gap_ms"]) for r in detected]
    additional = [float(r["additional_delay_ms"]) for r in detected]
    current_gap = [float(r["current_gap_ms"]) for r in detected]
    err = [float(r["next_start_error_ms"]) for r in detected]
    gap_stats = robust_stats(acoustic_gap)
    add_stats = robust_stats(additional)
    cur_stats = robust_stats(current_gap)
    err_stats = robust_stats(err)
    detection_rate = len(detected) / total_boundaries if total_boundaries else 0.0
    recommended_gap_ms = round_nearest(gap_stats["median"], 10.0)
    # Planner priors should be conservative: we want to remove the systematic early-start bias,
    # not fully chase long breaths/tails.  Median acoustic gap is the main recommendation;
    # P25/P75 define the expected runtime search window around it.
    return {
        "Class": label,
        "BoundaryCount": total_boundaries,
        "DetectedCount": len(detected),
        "DetectionRate": detection_rate,
        "CurrentGapMedianMs": cur_stats["median"],
        "AcousticResumeGapMedianMs": gap_stats["median"],
        "AcousticResumeGapP25Ms": gap_stats["p25"],
        "AcousticResumeGapP75Ms": gap_stats["p75"],
        "AcousticResumeGapP10Ms": gap_stats["p10"],
        "AcousticResumeGapP90Ms": gap_stats["p90"],
        "AdditionalDelayMedianMs": add_stats["median"],
        "AdditionalDelayP25Ms": add_stats["p25"],
        "AdditionalDelayP75Ms": add_stats["p75"],
        "NextStartErrorMedianMs": err_stats["median"],
        "NextStartErrorP25Ms": err_stats["p25"],
        "NextStartErrorP75Ms": err_stats["p75"],
        "RecommendedPlannerGapMs": recommended_gap_ms,
        "SuggestedRuntimeWindowBeforeMs": max(0.0, round_nearest(recommended_gap_ms - gap_stats["p25"], 10.0)) if finite(recommended_gap_ms) and finite(gap_stats["p25"]) else math.nan,
        "SuggestedRuntimeWindowAfterMs": max(0.0, round_nearest(gap_stats["p75"] - recommended_gap_ms, 10.0)) if finite(recommended_gap_ms) and finite(gap_stats["p75"]) else math.nan,
    }


def policy_rows(boundary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for policy in POLICY_PREFIXES:
        accepted_rows: list[dict[str, object]] = []
        total = 0
        by_kind_total: dict[str, int] = defaultdict(int)
        by_kind_detected: dict[str, int] = defaultdict(int)
        for row in boundary_rows:
            total += 1
            kind = row.get("PunctuationKind", "unknown") or "unknown"
            by_kind_total[kind] += 1
            detected = as_int(row, f"{policy}_Detected") == 1
            if not detected:
                continue
            by_kind_detected[kind] += 1
            resume = as_float(row, f"{policy}_ResumeSec")
            m = derive_boundary_measure(row, resume)
            accepted_rows.append({"detected": True, **m})
        summary = summarize_group(accepted_rows, policy, total)
        summary["CommaDetectionRate"] = by_kind_detected.get("comma", 0) / by_kind_total.get("comma", 1) if by_kind_total.get("comma", 0) else math.nan
        summary["SentenceDetectionRate"] = by_kind_detected.get("sentence", 0) / by_kind_total.get("sentence", 1) if by_kind_total.get("sentence", 0) else math.nan
        out.append(summary)
    return out


def class_rows(boundary_rows: list[dict[str, str]], policy: str | None) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    punct_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    punct_totals: dict[str, int] = defaultdict(int)

    for row in boundary_rows:
        kind = row.get("PunctuationKind", "unknown") or "unknown"
        punct = row.get("Punctuation", "") or ""
        class_key = "sentence" if as_int(row, "HardSentenceBreak") == 1 else kind
        totals[class_key] += 1
        punct_totals[punct] += 1

        if policy:
            detected = as_int(row, f"{policy}_Detected") == 1
            resume = as_float(row, f"{policy}_ResumeSec")
        else:
            detected = as_int(row, "Detected") == 1
            resume = as_float(row, "ResumeSec")
        if not detected:
            continue
        m = derive_boundary_measure(row, resume)
        rec = {
            "detected": True,
            "CaseID": row.get("CaseID", ""),
            "BoundaryIndex": as_int(row, "BoundaryIndex"),
            "Punctuation": punct,
            "PunctuationKind": kind,
            "HardSentenceBreak": as_int(row, "HardSentenceBreak"),
            "PrevWord": row.get("PrevWord", ""),
            "NextWord": row.get("NextWord", ""),
            **m,
        }
        groups[class_key].append(rec)
        punct_groups[punct].append(rec)

    class_out = [summarize_group(groups[k], k, totals[k]) for k in sorted(totals.keys())]
    punct_out = [summarize_group(punct_groups[k], k, punct_totals[k]) for k in sorted(punct_totals.keys())]

    # D03-friendly recommendation table: only common classes with stable samples.
    recommendations: list[dict[str, object]] = []
    for row in class_out:
        count = int(row["DetectedCount"])
        if count < 10:
            continue
        recommendations.append({
            "PunctuationClass": row["Class"],
            "RecommendedPlannerGapSec": float(row["RecommendedPlannerGapMs"]) / 1000.0 if finite(float(row["RecommendedPlannerGapMs"])) else math.nan,
            "RecommendedPlannerGapMs": row["RecommendedPlannerGapMs"],
            "RuntimeSearchBeforeMs": row["SuggestedRuntimeWindowBeforeMs"],
            "RuntimeSearchAfterMs": row["SuggestedRuntimeWindowAfterMs"],
            "DetectionRate": row["DetectionRate"],
            "SampleCount": row["DetectedCount"],
            "Notes": "median acoustic resume gap from previous visible center to next speech resume",
        })
    return class_out, punct_out, recommendations


def write_markdown(path: Path, policy: str | None, class_summary: list[dict[str, object]], policy_summary: list[dict[str, object]], recommendations: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# D02 Punctuation Prior Review")
    lines.append("")
    lines.append("This is an offline planner-calibration report generated from the D01 punctuation resume oracle. It estimates how much spacing the text/prosody planner should allocate around punctuation before runtime resume detection tries to correct the residual.")
    lines.append("")
    lines.append(f"Selected policy: `{policy or 'BestPolicy columns'}`")
    lines.append("")
    lines.append("## Recommended planner priors")
    lines.append("")
    if recommendations:
        lines.append("| Class | Gap ms | Search before ms | Search after ms | Detection rate | Samples |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in recommendations:
            lines.append(
                f"| {row['PunctuationClass']} | {fmt_ms(float(row['RecommendedPlannerGapMs']))} | "
                f"{fmt_ms(float(row['RuntimeSearchBeforeMs']))} | {fmt_ms(float(row['RuntimeSearchAfterMs']))} | "
                f"{float(row['DetectionRate']):.3f} | {int(row['SampleCount'])} |"
            )
    else:
        lines.append("No class had enough detected samples for a recommendation.")
    lines.append("")
    lines.append("## Class summary")
    lines.append("")
    lines.append("| Class | Boundaries | Detected | Rate | Current gap med | Acoustic gap med | Add delay med | Next-start error med |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in class_summary:
        lines.append(
            f"| {row['Class']} | {int(row['BoundaryCount'])} | {int(row['DetectedCount'])} | "
            f"{float(row['DetectionRate']):.3f} | {fmt_ms(float(row['CurrentGapMedianMs']))} | "
            f"{fmt_ms(float(row['AcousticResumeGapMedianMs']))} | {fmt_ms(float(row['AdditionalDelayMedianMs']))} | "
            f"{fmt_ms(float(row['NextStartErrorMedianMs']))} |"
        )
    lines.append("")
    lines.append("## Policy comparison")
    lines.append("")
    lines.append("| Policy | Detected | Rate | Acoustic gap med | Add delay med | Comma rate | Sentence rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in policy_summary:
        lines.append(
            f"| {row['Class']} | {int(row['DetectedCount'])} | {float(row['DetectionRate']):.3f} | "
            f"{fmt_ms(float(row['AcousticResumeGapMedianMs']))} | {fmt_ms(float(row['AdditionalDelayMedianMs']))} | "
            f"{fmt_ms(float(row['CommaDetectionRate']))} | {fmt_ms(float(row['SentenceDetectionRate']))} |"
        )
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `Current gap med` is the current scheduled gap between the last pre-punctuation visible center and the first post-punctuation visible center.")
    lines.append("- `Acoustic gap med` is the detected WAV resume time minus the last pre-punctuation visible center; this is the main candidate planner prior.")
    lines.append("- `Add delay med` is how much later the next phrase would need to start to hit the detected resume under the current planner.")
    lines.append("- Runtime should still detect the actual resume; these priors are meant to shrink the correction from hundreds of milliseconds to a small residual window.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Review D01 punctuation oracle results and derive planner punctuation priors.")
    ap.add_argument("--run", type=Path, help="LipLab run directory containing punctuation_oracle output")
    ap.add_argument("--boundaries", type=Path, help="Explicit punctuation_resume_oracle_boundaries.csv path")
    ap.add_argument("--out", type=Path, help="Output directory. Defaults to <run>/punctuation_priors or sibling punctuation_priors.")
    ap.add_argument("--policy", default="loose_30pct_40ms", help="D01 policy columns to use, or 'best' to use BestPolicy/Detected columns. Default: loose_30pct_40ms")
    args = ap.parse_args()

    if args.boundaries:
        boundaries_path = args.boundaries
    elif args.run:
        boundaries_path = args.run / "punctuation_oracle" / "punctuation_resume_oracle_boundaries.csv"
    else:
        ap.error("Provide --run or --boundaries")

    if not boundaries_path.exists():
        raise SystemExit(f"Missing D01 boundary CSV: {boundaries_path}")

    if args.out:
        out_dir = args.out
    elif args.run:
        out_dir = args.run / "punctuation_priors"
    else:
        out_dir = boundaries_path.parent.parent / "punctuation_priors"

    rows = read_csv_dicts(boundaries_path)
    selected_policy = None if args.policy.lower() == "best" else args.policy
    if selected_policy and selected_policy not in POLICY_PREFIXES:
        raise SystemExit(f"Unknown policy '{selected_policy}'. Known: {', '.join(POLICY_PREFIXES)} or 'best'.")

    pol_rows = policy_rows(rows)
    class_summary, punct_summary, recommendations = class_rows(rows, selected_policy)

    numeric_fields = [
        "DetectionRate", "CurrentGapMedianMs", "AcousticResumeGapMedianMs", "AcousticResumeGapP25Ms", "AcousticResumeGapP75Ms",
        "AcousticResumeGapP10Ms", "AcousticResumeGapP90Ms", "AdditionalDelayMedianMs", "AdditionalDelayP25Ms",
        "AdditionalDelayP75Ms", "NextStartErrorMedianMs", "NextStartErrorP25Ms", "NextStartErrorP75Ms",
        "RecommendedPlannerGapMs", "SuggestedRuntimeWindowBeforeMs", "SuggestedRuntimeWindowAfterMs",
        "CommaDetectionRate", "SentenceDetectionRate",
    ]
    for table in (pol_rows, class_summary, punct_summary, recommendations):
        for row in table:
            for key, val in list(row.items()):
                if key in numeric_fields or isinstance(val, float):
                    if isinstance(val, float) and finite(val):
                        row[key] = round(val, 6)

    summary_fields = [
        "Class", "BoundaryCount", "DetectedCount", "DetectionRate", "CurrentGapMedianMs", "AcousticResumeGapMedianMs",
        "AcousticResumeGapP25Ms", "AcousticResumeGapP75Ms", "AcousticResumeGapP10Ms", "AcousticResumeGapP90Ms",
        "AdditionalDelayMedianMs", "AdditionalDelayP25Ms", "AdditionalDelayP75Ms", "NextStartErrorMedianMs",
        "NextStartErrorP25Ms", "NextStartErrorP75Ms", "RecommendedPlannerGapMs", "SuggestedRuntimeWindowBeforeMs",
        "SuggestedRuntimeWindowAfterMs",
    ]
    write_csv(out_dir / "punctuation_prior_by_class.csv", class_summary, summary_fields)
    write_csv(out_dir / "punctuation_prior_by_punctuation.csv", punct_summary, summary_fields)
    write_csv(out_dir / "punctuation_prior_policy_summary.csv", pol_rows, summary_fields + ["CommaDetectionRate", "SentenceDetectionRate"])
    write_csv(out_dir / "punctuation_prior_recommendations.csv", recommendations, [
        "PunctuationClass", "RecommendedPlannerGapSec", "RecommendedPlannerGapMs", "RuntimeSearchBeforeMs", "RuntimeSearchAfterMs", "DetectionRate", "SampleCount", "Notes"
    ])
    write_markdown(out_dir / "punctuation_prior_review.md", selected_policy, class_summary, pol_rows, recommendations)
    print(f"Wrote D02 punctuation prior review to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
