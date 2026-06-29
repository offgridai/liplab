#!/usr/bin/env python3
"""V13 diagnostics for text-island duration and streaming handoff.

This script is intentionally diagnostic-only. It consumes the V9/V10/V12
runtime island alignment summary and compares candidate duration priors against
multiple observed targets:

  * detector audio span        (often too brittle; useful as a warning signal)
  * evidence center span       (event-center-derived spoken occupancy)
  * evidence render span       (event-envelope-derived spoken occupancy)
  * committed center/render spans (what runtime actually emitted)

The purpose is to explain mismatches like V11 preferring a smaller syllable
constant offline while live V12 became perceptually/timing worse.

Typical use:

  python scripts/analyze_duration_handoff_v13.py \
      --run experiments/runs/local_run \
      --out experiments/runs/local_run/v13_duration_handoff_diagnostics

Outputs:

  v13_formula_target_stats.csv
  v13_formula_target_ranked.md
  v13_island_diagnostics.csv
  v13_worst_underfilled_islands.csv
  v13_handoff_gap_stats.csv
  v13_notes.md
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def find_alignment_csv(run: Path) -> Path:
    candidates = [
        run / "corpus_runtime_island_alignment.csv",
        run / "summary" / "corpus_runtime_island_alignment.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find corpus_runtime_island_alignment.csv. "
        "Run scripts/summarize_run.py --run <run> first."
    )


@dataclass(frozen=True)
class Formula:
    name: str
    syllable_sec: float | None
    use_existing_floor: bool
    use_current_logged: bool = False

    def predict(self, row: dict[str, str]) -> float:
        if self.use_current_logged:
            return as_float(row, "PlannerDurationSec", 0.0)
        syllables = as_int(row, "PlannerSyllableCount", 0)
        raw = max(0.0, syllables * float(self.syllable_sec or 0.0))
        if self.use_existing_floor:
            existing = max(
                as_float(row, "PlannerSpeechMaterialSec", 0.0) + as_float(row, "PlannerPunctuationSec", 0.0),
                as_float(row, "PlannerShortUtteranceFloorSec", 0.0),
            )
            raw = max(raw, existing)
        return max(0.18, min(8.0, raw))


FORMULAS: list[Formula] = [
    Formula("current_logged_runtime", None, False, True),
    Formula("max_existing_syllable_155_v12", 0.155, True),
    Formula("max_existing_syllable_175_v10", 0.175, True),
    Formula("max_existing_syllable_165_midpoint", 0.165, True),
    Formula("pure_syllable_155", 0.155, False),
    Formula("pure_syllable_175", 0.175, False),
]

TARGETS = [
    ("detector_audio", "DetectorAudioDurationSec"),
    ("evidence_center", "EvidenceCenterSpanSec"),
    ("evidence_render", "EvidenceRenderSpanSec"),
    ("committed_center", "CommittedCenterSpanSec"),
    ("committed_render", "CommittedRenderSpanSec"),
    ("mapped_audio", "MappedAudioDurationSec"),
]


def score_formula(rows: list[dict[str, str]], formula: Formula, target_key: str) -> dict[str, object]:
    ratios: list[float] = []
    abs_errors_ms: list[float] = []
    under: list[float] = []
    over: list[float] = []
    count = 0
    for row in rows:
        pred = formula.predict(row)
        target = as_float(row, target_key, 0.0)
        if pred <= 0.0 or target <= 0.05:
            continue
        ratio = pred / target
        ratios.append(ratio)
        abs_errors_ms.append(abs(pred - target) * 1000.0)
        if ratio < 1.0:
            under.append(1.0 - ratio)
        else:
            over.append(ratio - 1.0)
        count += 1

    return {
        "Formula": formula.name,
        "Target": target_key,
        "Count": count,
        "MeanRatio": f"{mean(ratios):.6f}",
        "MedianRatio": f"{median(ratios):.6f}",
        "P10Ratio": f"{percentile(ratios, 0.10):.6f}",
        "P90Ratio": f"{percentile(ratios, 0.90):.6f}",
        "MeanAbsErrorMs": f"{mean(abs_errors_ms):.3f}",
        "MedianAbsErrorMs": f"{median(abs_errors_ms):.3f}",
        "P90AbsErrorMs": f"{percentile(abs_errors_ms, 0.90):.3f}",
        "UnderRate": f"{len(under) / count if count else 0.0:.6f}",
        "SevereUnderRate": f"{sum(1 for r in ratios if r < 0.80) / count if count else 0.0:.6f}",
        "OverRate": f"{len(over) / count if count else 0.0:.6f}",
        "SevereOverRate": f"{sum(1 for r in ratios if r > 1.25) / count if count else 0.0:.6f}",
    }


def build_island_diagnostics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        syllables = as_int(row, "PlannerSyllableCount", 0)
        if syllables <= 0:
            continue
        evidence_render = as_float(row, "EvidenceRenderSpanSec", 0.0)
        evidence_center = as_float(row, "EvidenceCenterSpanSec", 0.0)
        current = as_float(row, "PlannerDurationSec", 0.0)
        v10 = FORMULAS[2].predict(row)
        v12 = FORMULAS[1].predict(row)
        out.append({
            "case_id": row.get("case_id", ""),
            "TextIslandIndex": row.get("TextIslandIndex", ""),
            "AudioIslandIndex": row.get("AudioIslandIndex", ""),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EventCount": row.get("EventCount", ""),
            "EvidenceEventCount": row.get("EvidenceEventCount", ""),
            "Syllables": syllables,
            "PlannerSpeechMaterialSec": row.get("PlannerSpeechMaterialSec", ""),
            "PlannerPunctuationSec": row.get("PlannerPunctuationSec", ""),
            "CurrentPlannerDurationSec": f"{current:.6f}",
            "V10MaxExistingSyllable175Sec": f"{v10:.6f}",
            "V12MaxExistingSyllable155Sec": f"{v12:.6f}",
            "EvidenceCenterSpanSec": f"{evidence_center:.6f}",
            "EvidenceRenderSpanSec": f"{evidence_render:.6f}",
            "DetectorAudioDurationSec": row.get("DetectorAudioDurationSec", ""),
            "CommittedCenterSpanSec": row.get("CommittedCenterSpanSec", ""),
            "CommittedRenderSpanSec": row.get("CommittedRenderSpanSec", ""),
            "V10ToEvidenceRenderRatio": f"{(v10 / evidence_render) if evidence_render > 0.05 else 0.0:.6f}",
            "V12ToEvidenceRenderRatio": f"{(v12 / evidence_render) if evidence_render > 0.05 else 0.0:.6f}",
            "V12MinusV10Ms": f"{(v12 - v10) * 1000.0:.3f}",
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            "PrevEvidenceCenterGapSec": row.get("PrevEvidenceCenterGapSec", ""),
            "PrevCommittedCenterGapSec": row.get("PrevCommittedCenterGapSec", ""),
            "PrevGapDeltaCommittedVsEvidenceMs": row.get("PrevGapDeltaCommittedVsEvidenceMs", ""),
        })
    return out


def build_handoff_gap_stats(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    def collect(key: str, pred=lambda r: True) -> list[float]:
        return [as_float(r, key, 0.0) for r in rows if pred(r) and as_float(r, key, 0.0) > 0.0]

    predicates = {
        "all_non_first": lambda r: as_int(r, "TextIslandIndex", 0) > 0,
        "mapped_audio_missing": lambda r: as_int(r, "AudioIslandIndex", -1) < 0,
        "large_gap_error_gt_500ms": lambda r: abs(as_float(r, "PrevGapDeltaCommittedVsEvidenceMs", 0.0)) > 500.0,
    }
    out: list[dict[str, object]] = []
    for group_name, pred in predicates.items():
        for key in ["PrevEvidenceCenterGapSec", "PrevCommittedCenterGapSec", "PrevGapDeltaCommittedVsEvidenceMs", "MeanStreamingAbsErrorMs"]:
            values = collect(key, pred)
            out.append({
                "Group": group_name,
                "Metric": key,
                "Count": len(values),
                "Mean": f"{mean(values):.6f}",
                "Median": f"{median(values):.6f}",
                "P90": f"{percentile(values, 0.90):.6f}",
                "P95": f"{percentile(values, 0.95):.6f}",
                "Max": f"{max(values) if values else 0.0:.6f}",
            })
    return out


def write_ranked_md(path: Path, stats: list[dict[str, object]]) -> None:
    # Rank primarily by evidence_render because it corresponds to visible occupancy,
    # while detector_audio is often a detector segmentation artifact.
    preferred = [r for r in stats if r["Target"] == "EvidenceRenderSpanSec"]
    preferred.sort(key=lambda r: (float(r["MeanAbsErrorMs"]), float(r["SevereUnderRate"])))
    lines = [
        "# V13 duration / handoff diagnostics",
        "",
        "Primary ranking target: `EvidenceRenderSpanSec`.",
        "This target tracks the span occupied by evidence-backed event envelopes and is usually a better visible-duration target than raw detector island duration.",
        "",
        "| Rank | Formula | Mean ratio | Median ratio | MAE ms | Severe under | Severe over |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(preferred, start=1):
        lines.append(
            f"| {idx} | `{row['Formula']}` | {float(row['MeanRatio']):.3f} | "
            f"{float(row['MedianRatio']):.3f} | {float(row['MeanAbsErrorMs']):.1f} | "
            f"{float(row['SevereUnderRate']):.3f} | {float(row['SevereOverRate']):.3f} |"
        )
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "If `detector_audio` prefers a much shorter formula than `evidence_render`, treat that as a detector segmentation warning, not automatically as a better duration model.",
        "V12 was a likely example: it followed the simpler detector-span sweep but worsened live streaming MAE and visible gaps.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--alignment-csv", type=Path, default=None)
    args = parser.parse_args()

    run = args.run
    out_dir = args.out or (run / "v13_duration_handoff_diagnostics")
    alignment_csv = args.alignment_csv or find_alignment_csv(run)
    rows = read_csv(alignment_csv)
    rows = [r for r in rows if as_int(r, "PlannerSyllableCount", 0) > 0]

    stats: list[dict[str, object]] = []
    for formula in FORMULAS:
        for _target_name, target_key in TARGETS:
            stats.append(score_formula(rows, formula, target_key))

    stats_fields = [
        "Formula", "Target", "Count", "MeanRatio", "MedianRatio", "P10Ratio", "P90Ratio",
        "MeanAbsErrorMs", "MedianAbsErrorMs", "P90AbsErrorMs", "UnderRate",
        "SevereUnderRate", "OverRate", "SevereOverRate",
    ]
    write_csv(out_dir / "v13_formula_target_stats.csv", stats, stats_fields)
    write_ranked_md(out_dir / "v13_formula_target_ranked.md", stats)

    island_rows = build_island_diagnostics(rows)
    island_fields = list(island_rows[0].keys()) if island_rows else []
    write_csv(out_dir / "v13_island_diagnostics.csv", island_rows, island_fields)

    worst = sorted(
        island_rows,
        key=lambda r: (
            float(r.get("V12ToEvidenceRenderRatio", 1.0)),
            -float(r.get("MeanStreamingAbsErrorMs", 0.0) or 0.0),
        ),
    )[:50]
    write_csv(out_dir / "v13_worst_underfilled_islands.csv", worst, island_fields)

    handoff_rows = build_handoff_gap_stats(rows)
    write_csv(out_dir / "v13_handoff_gap_stats.csv", handoff_rows, ["Group", "Metric", "Count", "Mean", "Median", "P90", "P95", "Max"])

    notes = [
        "# V13 notes",
        "",
        "Runtime change in V13 reverts the V12 syllable floor from 0.155s back to 0.175s.",
        "Diagnostics compare formulas against detector, evidence, committed, and mapped spans so we do not overfit to one brittle target.",
        "",
        "Recommended files to send back:",
        "",
        "```text",
        str(out_dir / "v13_formula_target_ranked.md"),
        str(out_dir / "v13_formula_target_stats.csv"),
        str(out_dir / "v13_worst_underfilled_islands.csv"),
        str(out_dir / "v13_handoff_gap_stats.csv"),
        "```",
    ]
    (out_dir / "v13_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote V13 diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
