#!/usr/bin/env python3
"""V14 diagnostics for runtime island evidence semantics.

This script is intentionally diagnostic-only. It consumes the V9/V10/V13
runtime island alignment summary and asks whether the *evidence island* spans
are semantically clean enough to use as duration ground truth.

It does not change runtime behavior. It classifies each text island by likely
failure mode:

  CLEAN              evidence and committed/planner spans broadly agree
  TAIL_HEAVY         evidence render span has a large envelope tail beyond centers
  BLEED_FORWARD      evidence begins before the committed/planner island by a lot
  BLEED_BACKWARD     evidence extends after the committed/planner island by a lot
  STALL_FALLBACK     non-first island committed gap greatly exceeds evidence gap
  NO_AUDIO_MAPPING   text island had no mapped detector/audio island
  DETECTOR_MISMATCH  detector island span disagrees strongly with evidence span
  UNDERFILLED        committed render span is much shorter than evidence render span
  OVERFILLED         committed render span is much longer than evidence render span

Typical use:

  python scripts/analyze_island_evidence_v14.py \
      --run experiments/runs/local_run \
      --out experiments/runs/local_run/v14_island_evidence_diagnostics

Outputs:

  v14_island_evidence_diagnostics.csv
  v14_evidence_class_summary.csv
  v14_evidence_metric_stats.csv
  v14_worst_evidence_bleed.csv
  v14_worst_handoff_stalls.csv
  v14_notes.md
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def fmt(value: float) -> str:
    return f"{value:.6f}"


def classify(row: dict[str, str], metrics: dict[str, float]) -> list[str]:
    labels: list[str] = []
    text_island = as_int(row, "TextIslandIndex", 0)
    audio_island = as_int(row, "AudioIslandIndex", -1)

    if audio_island < 0:
        labels.append("NO_AUDIO_MAPPING")

    if text_island > 0 and metrics["prev_gap_delta_committed_vs_evidence_ms"] > 500.0:
        labels.append("STALL_FALLBACK")

    if metrics["evidence_tail_sec"] > max(0.180, metrics["evidence_center_span_sec"] * 0.22):
        labels.append("TAIL_HEAVY")

    if metrics["evidence_lead_before_committed_ms"] > 160.0:
        labels.append("BLEED_FORWARD")

    if metrics["evidence_trail_after_committed_ms"] > 220.0:
        labels.append("BLEED_BACKWARD")

    detector = metrics["detector_duration_sec"]
    evidence_render = metrics["evidence_render_span_sec"]
    if detector > 0.050 and evidence_render > 0.050:
        ratio = detector / evidence_render
        if ratio < 0.55 or ratio > 1.45:
            labels.append("DETECTOR_MISMATCH")
    elif detector <= 0.050 and evidence_render > 0.250:
        labels.append("DETECTOR_MISMATCH")

    committed_render = metrics["committed_render_span_sec"]
    if evidence_render > 0.050 and committed_render > 0.050:
        ratio = committed_render / evidence_render
        if ratio < 0.78:
            labels.append("UNDERFILLED")
        elif ratio > 1.22:
            labels.append("OVERFILLED")

    if not labels:
        labels.append("CLEAN")
    return labels


def build_diagnostics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        evidence_center_span = as_float(row, "EvidenceCenterSpanSec", 0.0)
        evidence_render_span = as_float(row, "EvidenceRenderSpanSec", 0.0)
        committed_center_span = as_float(row, "CommittedCenterSpanSec", 0.0)
        committed_render_span = as_float(row, "CommittedRenderSpanSec", 0.0)
        planner_duration = as_float(row, "PlannerDurationSec", 0.0)
        mapped_audio_duration = as_float(row, "MappedAudioDurationSec", 0.0)
        detector_duration = as_float(row, "DetectorAudioDurationSec", 0.0)

        evidence_first_start = as_float(row, "EvidenceFirstStartSec", 0.0)
        evidence_first_center = as_float(row, "EvidenceFirstCenterSec", 0.0)
        evidence_last_center = as_float(row, "EvidenceLastCenterSec", 0.0)
        evidence_last_end = as_float(row, "EvidenceLastEndSec", 0.0)
        committed_first_start = as_float(row, "CommittedFirstRenderStartSec", 0.0)
        committed_last_end = as_float(row, "CommittedLastRenderEndSec", 0.0)
        committed_first_center = as_float(row, "CommittedFirstCenterSec", 0.0)
        committed_last_center = as_float(row, "CommittedLastCenterSec", 0.0)

        evidence_lead = max(0.0, evidence_first_center - evidence_first_start)
        evidence_trail = max(0.0, evidence_last_end - evidence_last_center)
        evidence_tail = max(0.0, evidence_render_span - evidence_center_span)
        committed_tail = max(0.0, committed_render_span - committed_center_span)

        evidence_lead_before_committed_ms = max(0.0, committed_first_start - evidence_first_start) * 1000.0
        evidence_trail_after_committed_ms = max(0.0, evidence_last_end - committed_last_end) * 1000.0
        evidence_first_center_to_committed_ms = (committed_first_center - evidence_first_center) * 1000.0
        evidence_last_center_to_committed_ms = (committed_last_center - evidence_last_center) * 1000.0

        prev_evidence_gap = as_float(row, "PrevEvidenceCenterGapSec", 0.0)
        prev_committed_gap = as_float(row, "PrevCommittedCenterGapSec", 0.0)
        prev_detector_gap = as_float(row, "PrevDetectorAudioGapSec", 0.0)
        prev_gap_delta = as_float(row, "PrevGapDeltaCommittedVsEvidenceMs", 0.0)

        metrics = {
            "evidence_center_span_sec": evidence_center_span,
            "evidence_render_span_sec": evidence_render_span,
            "committed_center_span_sec": committed_center_span,
            "committed_render_span_sec": committed_render_span,
            "planner_duration_sec": planner_duration,
            "mapped_audio_duration_sec": mapped_audio_duration,
            "detector_duration_sec": detector_duration,
            "evidence_tail_sec": evidence_tail,
            "evidence_lead_before_committed_ms": evidence_lead_before_committed_ms,
            "evidence_trail_after_committed_ms": evidence_trail_after_committed_ms,
            "prev_gap_delta_committed_vs_evidence_ms": prev_gap_delta,
        }
        labels = classify(row, metrics)

        out.append({
            "case_id": row.get("case_id", ""),
            "TextIslandIndex": row.get("TextIslandIndex", ""),
            "AudioIslandIndex": row.get("AudioIslandIndex", ""),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EventCount": row.get("EventCount", ""),
            "EvidenceEventCount": row.get("EvidenceEventCount", ""),
            "Classification": ";".join(labels),
            "PrimaryClassification": labels[0],
            "PlannerDurationSec": fmt(planner_duration),
            "MappedAudioDurationSec": fmt(mapped_audio_duration),
            "DetectorAudioDurationSec": fmt(detector_duration),
            "CommittedCenterSpanSec": fmt(committed_center_span),
            "CommittedRenderSpanSec": fmt(committed_render_span),
            "EvidenceCenterSpanSec": fmt(evidence_center_span),
            "EvidenceRenderSpanSec": fmt(evidence_render_span),
            "EvidenceLeadSec": fmt(evidence_lead),
            "EvidenceTrailSec": fmt(evidence_trail),
            "EvidenceTailSec": fmt(evidence_tail),
            "CommittedTailSec": fmt(committed_tail),
            "EvidenceTailShareOfRender": fmt(evidence_tail / evidence_render_span if evidence_render_span > 0.05 else 0.0),
            "CommittedTailShareOfRender": fmt(committed_tail / committed_render_span if committed_render_span > 0.05 else 0.0),
            "CommittedRenderToEvidenceRenderRatio": fmt(committed_render_span / evidence_render_span if evidence_render_span > 0.05 else 0.0),
            "PlannerToEvidenceRenderRatio": fmt(planner_duration / evidence_render_span if evidence_render_span > 0.05 else 0.0),
            "DetectorToEvidenceRenderRatio": fmt(detector_duration / evidence_render_span if evidence_render_span > 0.05 else 0.0),
            "EvidenceLeadBeforeCommittedMs": f"{evidence_lead_before_committed_ms:.3f}",
            "EvidenceTrailAfterCommittedMs": f"{evidence_trail_after_committed_ms:.3f}",
            "CommittedFirstMinusEvidenceFirstCenterMs": f"{evidence_first_center_to_committed_ms:.3f}",
            "CommittedLastMinusEvidenceLastCenterMs": f"{evidence_last_center_to_committed_ms:.3f}",
            "PrevEvidenceCenterGapSec": fmt(prev_evidence_gap),
            "PrevCommittedCenterGapSec": fmt(prev_committed_gap),
            "PrevDetectorAudioGapSec": fmt(prev_detector_gap),
            "PrevGapDeltaCommittedVsEvidenceMs": f"{prev_gap_delta:.3f}",
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
        })
    return out


def metric_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        "EvidenceTailSec",
        "EvidenceTailShareOfRender",
        "CommittedRenderToEvidenceRenderRatio",
        "PlannerToEvidenceRenderRatio",
        "DetectorToEvidenceRenderRatio",
        "EvidenceLeadBeforeCommittedMs",
        "EvidenceTrailAfterCommittedMs",
        "CommittedFirstMinusEvidenceFirstCenterMs",
        "CommittedLastMinusEvidenceLastCenterMs",
        "PrevGapDeltaCommittedVsEvidenceMs",
        "MeanStreamingAbsErrorMs",
    ]
    out: list[dict[str, object]] = []
    for key in numeric_keys:
        vals: list[float] = []
        for row in rows:
            try:
                v = float(row.get(key, 0.0))
                if math.isfinite(v):
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        out.append({
            "Metric": key,
            "Count": len(vals),
            "Mean": f"{mean(vals):.6f}",
            "Median": f"{median(vals):.6f}",
            "P10": f"{percentile(vals, 0.10):.6f}",
            "P90": f"{percentile(vals, 0.90):.6f}",
            "P95": f"{percentile(vals, 0.95):.6f}",
            "Max": f"{max(vals) if vals else 0.0:.6f}",
        })
    return out


def class_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = len(rows)
    counts: Counter[str] = Counter()
    exclusive: Counter[str] = Counter()
    streaming: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        labels = str(row.get("Classification", "")).split(";")
        exclusive[str(row.get("PrimaryClassification", ""))] += 1
        for label in labels:
            if not label:
                continue
            counts[label] += 1
            try:
                streaming[label].append(float(row.get("MeanStreamingAbsErrorMs", 0.0)))
            except (TypeError, ValueError):
                pass
    out: list[dict[str, object]] = []
    for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        vals = streaming[label]
        out.append({
            "Classification": label,
            "IslandCount": count,
            "IslandRate": f"{count / total if total else 0.0:.6f}",
            "PrimaryCount": exclusive[label],
            "MeanStreamingAbsErrorMs": f"{mean(vals):.3f}",
            "MedianStreamingAbsErrorMs": f"{median(vals):.3f}",
            "P90StreamingAbsErrorMs": f"{percentile(vals, 0.90):.3f}",
        })
    return out


def write_notes(path: Path, rows: list[dict[str, object]], class_rows: list[dict[str, object]], stat_rows: list[dict[str, object]]) -> None:
    def stat(metric: str, field: str) -> str:
        for row in stat_rows:
            if row.get("Metric") == metric:
                return str(row.get(field, "0"))
        return "0"

    top_classes = ", ".join(
        f"{r['Classification']}={r['IslandRate']}" for r in class_rows[:6]
    )
    text = f"""# V14 Island Evidence Diagnostics

Rows analyzed: {len(rows)}

Top classifications: {top_classes}

Key distribution checks:

| Metric | Median | P90 | P95 |
|---|---:|---:|---:|
| Evidence tail share of render | {stat('EvidenceTailShareOfRender', 'Median')} | {stat('EvidenceTailShareOfRender', 'P90')} | {stat('EvidenceTailShareOfRender', 'P95')} |
| Committed render / evidence render | {stat('CommittedRenderToEvidenceRenderRatio', 'Median')} | {stat('CommittedRenderToEvidenceRenderRatio', 'P90')} | {stat('CommittedRenderToEvidenceRenderRatio', 'P95')} |
| Planner / evidence render | {stat('PlannerToEvidenceRenderRatio', 'Median')} | {stat('PlannerToEvidenceRenderRatio', 'P90')} | {stat('PlannerToEvidenceRenderRatio', 'P95')} |
| Evidence lead before committed ms | {stat('EvidenceLeadBeforeCommittedMs', 'Median')} | {stat('EvidenceLeadBeforeCommittedMs', 'P90')} | {stat('EvidenceLeadBeforeCommittedMs', 'P95')} |
| Evidence trail after committed ms | {stat('EvidenceTrailAfterCommittedMs', 'Median')} | {stat('EvidenceTrailAfterCommittedMs', 'P90')} | {stat('EvidenceTrailAfterCommittedMs', 'P95')} |
| Prev committed-vs-evidence gap delta ms | {stat('PrevGapDeltaCommittedVsEvidenceMs', 'Median')} | {stat('PrevGapDeltaCommittedVsEvidenceMs', 'P90')} | {stat('PrevGapDeltaCommittedVsEvidenceMs', 'P95')} |

Interpretation hints:

* High `TAIL_HEAVY` means evidence render spans include envelope lead/release mass, so
  evidence-render duration is not a clean speech-duration target.
* High `BLEED_FORWARD` or `BLEED_BACKWARD` means evidence is leaking across island
  boundaries, so per-island duration comparisons are contaminated by neighbors.
* High `STALL_FALLBACK` means bounded handoff is still arriving later than the
  evidence-center transition for some non-first islands.
* High `DETECTOR_MISMATCH` means detector audio islands are too brittle to use as
  the sole duration target.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="V14 island evidence semantics diagnostics")
    parser.add_argument("--run", required=True, type=Path, help="Run directory containing corpus_runtime_island_alignment.csv")
    parser.add_argument("--out", type=Path, help="Output directory")
    args = parser.parse_args()

    run = args.run
    out_dir = args.out or (run / "v14_island_evidence_diagnostics")
    align_path = find_alignment_csv(run)
    source_rows = read_csv(align_path)
    rows = build_diagnostics(source_rows)

    island_fields = [
        "case_id", "TextIslandIndex", "AudioIslandIndex", "FirstWord", "LastWord", "EventCount", "EvidenceEventCount",
        "Classification", "PrimaryClassification",
        "PlannerDurationSec", "MappedAudioDurationSec", "DetectorAudioDurationSec",
        "CommittedCenterSpanSec", "CommittedRenderSpanSec", "EvidenceCenterSpanSec", "EvidenceRenderSpanSec",
        "EvidenceLeadSec", "EvidenceTrailSec", "EvidenceTailSec", "CommittedTailSec",
        "EvidenceTailShareOfRender", "CommittedTailShareOfRender",
        "CommittedRenderToEvidenceRenderRatio", "PlannerToEvidenceRenderRatio", "DetectorToEvidenceRenderRatio",
        "EvidenceLeadBeforeCommittedMs", "EvidenceTrailAfterCommittedMs",
        "CommittedFirstMinusEvidenceFirstCenterMs", "CommittedLastMinusEvidenceLastCenterMs",
        "PrevEvidenceCenterGapSec", "PrevCommittedCenterGapSec", "PrevDetectorAudioGapSec", "PrevGapDeltaCommittedVsEvidenceMs",
        "FirstCenterErrorMs", "LastCenterErrorMs", "MeanStreamingAbsErrorMs",
    ]
    write_csv(out_dir / "v14_island_evidence_diagnostics.csv", rows, island_fields)

    classes = class_summary(rows)
    write_csv(out_dir / "v14_evidence_class_summary.csv", classes, [
        "Classification", "IslandCount", "IslandRate", "PrimaryCount",
        "MeanStreamingAbsErrorMs", "MedianStreamingAbsErrorMs", "P90StreamingAbsErrorMs",
    ])

    stats = metric_stats(rows)
    write_csv(out_dir / "v14_evidence_metric_stats.csv", stats, [
        "Metric", "Count", "Mean", "Median", "P10", "P90", "P95", "Max",
    ])

    def f(row: dict[str, object], key: str) -> float:
        try:
            return float(row.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    worst_bleed = sorted(
        rows,
        key=lambda r: max(f(r, "EvidenceLeadBeforeCommittedMs"), f(r, "EvidenceTrailAfterCommittedMs"), f(r, "EvidenceTailSec") * 1000.0),
        reverse=True,
    )[:50]
    write_csv(out_dir / "v14_worst_evidence_bleed.csv", worst_bleed, island_fields)

    worst_stalls = sorted(
        [r for r in rows if int(float(str(r.get("TextIslandIndex", "0") or "0"))) > 0],
        key=lambda r: f(r, "PrevGapDeltaCommittedVsEvidenceMs"),
        reverse=True,
    )[:50]
    write_csv(out_dir / "v14_worst_handoff_stalls.csv", worst_stalls, island_fields)

    write_notes(out_dir / "v14_notes.md", rows, classes, stats)
    print(f"Wrote V14 island evidence diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
