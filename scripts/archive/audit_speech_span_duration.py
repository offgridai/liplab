#!/usr/bin/env python3
"""D09 offline speech-span duration auditor.

This script consumes D08 phase timing diagnostics and the standard LipLab summary
CSV, then answers a narrow question:

    Where is the committed/planned animation span longer or shorter than the
    detected speech envelope?

It does not change runtime behavior. It is intended to guide duration/planner
work before adding any more online control logic.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


OUT_DIR_NAME = "speech_span_audit"


def _find_file(run: Path, relative: str) -> Path:
    p = run / relative
    if not p.exists():
        raise FileNotFoundError(f"missing required file: {p}")
    return p


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - user-facing failure path
        raise RuntimeError(f"failed reading {path}: {exc}") from exc


def _safe_num(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _case_key(df: pd.DataFrame) -> pd.Series:
    if "CaseDir" in df.columns:
        return df["CaseDir"].astype(str)
    if "case_id" in df.columns:
        return df["case_id"].astype(str)
    if "LineID" in df.columns:
        return df["LineID"].astype(str)
    raise ValueError("could not find case key column; expected CaseDir, case_id, or LineID")


def _bucket_numeric(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels, include_lowest=True)


def _agg_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_col, dropna=False, observed=False)
    out = g.agg(
        CaseCount=("CaseDir", "count"),
        EventCountMean=("EventCount", "mean"),
        SpeechChunkCountMean=("SpeechChunkCount", "mean"),
        CommittedSpanMeanSec=("CommittedCenterSpanSec", "mean"),
        SpeechEnvelopeMeanSec=("SpeechChunkEnvelopeSec", "mean"),
        SpanExcessMeanMs=("SpanExcessMs", "mean"),
        SpanExcessMedianMs=("SpanExcessMs", "median"),
        SpanExcessP90Ms=("SpanExcessMs", lambda s: s.quantile(0.90)),
        SpanRatioMean=("CommittedSpanToSpeechEnvelopeRatio", "mean"),
        FirstCenterMinusSpeechStartMeanMs=("FirstCenterMinusSpeechStartMs", "mean"),
        LastCenterMinusSpeechEndMeanMs=("LastCenterMinusSpeechEndMs", "mean"),
        OverrunGE250Rate=("OverrunGE250", "mean"),
        OverrunGE500Rate=("OverrunGE500", "mean"),
        UnderrunGE150Rate=("UnderrunGE150", "mean"),
    ).reset_index()
    return out.sort_values(["SpanExcessMeanMs", "CaseCount"], ascending=[False, False])


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _summary_row(audit: pd.DataFrame) -> pd.DataFrame:
    def q(col: str, p: float) -> float:
        return float(pd.to_numeric(audit[col], errors="coerce").quantile(p)) if col in audit.columns else math.nan

    n = len(audit)
    return pd.DataFrame([
        {
            "CaseCount": n,
            "TotalEventCount": int(pd.to_numeric(audit.get("EventCount", pd.Series(dtype=float)), errors="coerce").sum()) if n else 0,
            "SpanExcessMeanMs": float(audit["SpanExcessMs"].mean()) if n else math.nan,
            "SpanExcessMedianMs": float(audit["SpanExcessMs"].median()) if n else math.nan,
            "SpanExcessP10Ms": q("SpanExcessMs", 0.10),
            "SpanExcessP90Ms": q("SpanExcessMs", 0.90),
            "SpanRatioMean": float(audit["CommittedSpanToSpeechEnvelopeRatio"].mean()) if n else math.nan,
            "SpanRatioMedian": float(audit["CommittedSpanToSpeechEnvelopeRatio"].median()) if n else math.nan,
            "StartErrorMeanMs": float(audit["FirstCenterMinusSpeechStartMs"].mean()) if n else math.nan,
            "StartErrorMedianMs": float(audit["FirstCenterMinusSpeechStartMs"].median()) if n else math.nan,
            "EndErrorMeanMs": float(audit["LastCenterMinusSpeechEndMs"].mean()) if n else math.nan,
            "EndErrorMedianMs": float(audit["LastCenterMinusSpeechEndMs"].median()) if n else math.nan,
            "OverrunGE250Rate": float(audit["OverrunGE250"].mean()) if n else math.nan,
            "OverrunGE500Rate": float(audit["OverrunGE500"].mean()) if n else math.nan,
            "UnderrunGE150Rate": float(audit["UnderrunGE150"].mean()) if n else math.nan,
        }
    ])


def _top_cases(audit: pd.DataFrame, *, largest: bool, n: int) -> pd.DataFrame:
    cols = [
        "CaseDir",
        "EventCount",
        "SpeechChunkCount",
        "SoftPunctuationBoundaryCount",
        "TextPhraseCount",
        "AudioDurationSec",
        "CommittedCenterSpanSec",
        "SpeechChunkEnvelopeSec",
        "SpanExcessMs",
        "CommittedSpanToSpeechEnvelopeRatio",
        "FirstCenterMinusSpeechStartMs",
        "LastCenterMinusSpeechEndMs",
        "SpeechAttributionPlannerSpeechToDetectedSpeechRatio",
        "SpeechAttributionPlannerPunctuationSec",
        "SpeechAttributionPlannerSpeechMaterialSec",
        "SpeechAttributionDetectedSpeechMaterialSec",
    ]
    cols = [c for c in cols if c in audit.columns]
    return audit.sort_values("SpanExcessMs", ascending=not largest)[cols].head(n)


def _event_pose_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "PoseID" not in events.columns:
        return pd.DataFrame()
    df = events.copy()
    df["CenterMinusSpeechStartAbsMs"] = _safe_num(df, "CenterMinusMatchedSpeechStartMs").abs()
    df["CenterMinusSpeechEndAbsMs"] = _safe_num(df, "CenterMinusMatchedSpeechEndMs").abs()
    g = df.groupby("PoseID", dropna=False)
    return g.agg(
        EventCount=("PoseID", "count"),
        MeanAbsStartMs=("CenterMinusSpeechStartAbsMs", "mean"),
        MedianAbsStartMs=("CenterMinusSpeechStartAbsMs", "median"),
        MeanAbsEndMs=("CenterMinusSpeechEndAbsMs", "mean"),
        MedianAbsEndMs=("CenterMinusSpeechEndAbsMs", "median"),
        MeanFacePeakMinusCenterMs=("FacePeakMinusCenterMs", "mean") if "FacePeakMinusCenterMs" in df.columns else ("PoseID", "count"),
    ).reset_index().sort_values("MeanAbsStartMs", ascending=False)


def _review_md(summary: pd.DataFrame, tables: dict[str, pd.DataFrame], top_over: pd.DataFrame, top_under: pd.DataFrame) -> str:
    row = summary.iloc[0].to_dict() if not summary.empty else {}

    def fmt(v: object, unit: str = "") -> str:
        try:
            f = float(v)
        except Exception:
            return "n/a"
        if math.isnan(f):
            return "n/a"
        return f"{f:.1f}{unit}"

    lines: list[str] = []
    lines.append("# D09 Speech Span Duration Audit")
    lines.append("")
    lines.append("This report compares the committed viseme center span against the detected speech envelope from D08 phase timing diagnostics.")
    lines.append("")
    lines.append("## Corpus headline")
    lines.append("")
    lines.append(f"- Cases: {int(row.get('CaseCount', 0) or 0)}")
    lines.append(f"- Events: {int(row.get('TotalEventCount', 0) or 0)}")
    lines.append(f"- Mean span excess: {fmt(row.get('SpanExcessMeanMs'), ' ms')}")
    lines.append(f"- Median span excess: {fmt(row.get('SpanExcessMedianMs'), ' ms')}")
    lines.append(f"- Mean committed/speech-envelope ratio: {fmt(row.get('SpanRatioMean'))}")
    lines.append(f"- Mean first-center start error: {fmt(row.get('StartErrorMeanMs'), ' ms')}")
    lines.append(f"- Mean last-center end error: {fmt(row.get('EndErrorMeanMs'), ' ms')}")
    lines.append(f"- Overrun >=250ms rate: {fmt((row.get('OverrunGE250Rate') or 0) * 100.0, '%')}")
    lines.append(f"- Overrun >=500ms rate: {fmt((row.get('OverrunGE500Rate') or 0) * 100.0, '%')}")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- Positive span excess means the animation center span outlasts the detected speech envelope.")
    lines.append("- Negative span excess means the animation center span ends before the detected speech envelope.")
    lines.append("- Start error near zero with large positive end error means launch is fine but playback/prosody duration is too long.")
    lines.append("")
    for name, df in tables.items():
        if df.empty:
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append(df.head(12).to_markdown(index=False))
        lines.append("")
    if not top_over.empty:
        lines.append("## Worst overrun cases")
        lines.append("")
        lines.append(top_over.head(15).to_markdown(index=False))
        lines.append("")
    if not top_under.empty:
        lines.append("## Worst underrun cases")
        lines.append("")
        lines.append(top_under.head(15).to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def build_audit(run: Path, phase_dir: Optional[Path], out_dir: Optional[Path], top_n: int) -> Path:
    run = run.resolve()
    phase = phase_dir.resolve() if phase_dir else run / "phase_timing_diagnostics"
    out = out_dir.resolve() if out_dir else run / OUT_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)

    case_phase = _read_csv(_find_file(phase, "case_phase_timing.csv"))
    event_phase_path = phase / "event_phase_timing.csv"
    event_phase = _read_csv(event_phase_path) if event_phase_path.exists() else pd.DataFrame()

    audit = case_phase.copy()
    audit["CaseDir"] = _case_key(audit)
    audit["SpanExcessMs"] = (_safe_num(audit, "CommittedCenterSpanSec") - _safe_num(audit, "SpeechChunkEnvelopeSec")) * 1000.0
    audit["OverrunGE250"] = audit["SpanExcessMs"] >= 250.0
    audit["OverrunGE500"] = audit["SpanExcessMs"] >= 500.0
    audit["UnderrunGE150"] = audit["SpanExcessMs"] <= -150.0

    summary_path = run / "summary.csv"
    if summary_path.exists():
        summary = _read_csv(summary_path).copy()
        summary["CaseDir"] = _case_key(summary)
        wanted = [
            "CaseDir",
            "AudioDurationSec",
            "EstimatedTextDurationSec",
            "RuntimeCommittedDurationSec",
            "RuntimeCommittedDurationRatioToAudio",
            "DurationRatioTextToAudio",
            "SoftPunctuationBoundaryCount",
            "TextPhraseCount",
            "SpeechAttributionPlannerDurationSec",
            "SpeechAttributionPlannerDurationToAudioRatio",
            "SpeechAttributionPlannerSpeechMaterialSec",
            "SpeechAttributionDetectedSpeechMaterialSec",
            "SpeechAttributionPlannerSpeechMinusDetectedSpeechSec",
            "SpeechAttributionPlannerSpeechToDetectedSpeechRatio",
            "SpeechAttributionPlannerPunctuationSec",
            "SpeechAttributionPlannedPunctuationToPlannerDurationRatio",
            "SpeechAttributionDetectedGapsAndSilenceToAudioRatio",
            "SpeechAttributionDetectedSpeechToAudioRatio",
        ]
        wanted = [c for c in wanted if c in summary.columns]
        audit = audit.merge(summary[wanted], on="CaseDir", how="left")

    # Derived buckets.
    audit["EventCountBucket"] = _bucket_numeric(
        _safe_num(audit, "EventCount"),
        [-1, 8, 16, 24, 32, 48, 10_000],
        ["00-08", "09-16", "17-24", "25-32", "33-48", "49+"],
    )
    audit["SpeechEnvelopeBucketSec"] = _bucket_numeric(
        _safe_num(audit, "SpeechChunkEnvelopeSec"),
        [-0.01, 1.0, 2.0, 3.0, 4.0, 6.0, 10_000],
        ["0-1s", "1-2s", "2-3s", "3-4s", "4-6s", "6s+"],
    )
    audit["SpeechChunkCountBucket"] = _bucket_numeric(
        _safe_num(audit, "SpeechChunkCount"),
        [-1, 1, 2, 3, 4, 10_000],
        ["1", "2", "3", "4", "5+"],
    )
    audit["SoftPunctuationBucket"] = _bucket_numeric(
        _safe_num(audit, "SoftPunctuationBoundaryCount", 0.0),
        [-1, 0, 1, 2, 3, 10_000],
        ["0", "1", "2", "3", "4+"],
    )
    audit["PlannerSpeechToDetectedBucket"] = _bucket_numeric(
        _safe_num(audit, "SpeechAttributionPlannerSpeechToDetectedSpeechRatio"),
        [-10_000, 0.75, 1.0, 1.25, 1.5, 2.0, 10_000],
        ["<=0.75", "0.75-1.00", "1.00-1.25", "1.25-1.50", "1.50-2.00", "2.00+"],
    )

    _write(audit, out / "case_speech_span_audit.csv")

    corpus = _summary_row(audit)
    _write(corpus, out / "speech_span_audit_summary.csv")

    tables = {
        "By event count": _agg_table(audit, "EventCountBucket"),
        "By speech envelope": _agg_table(audit, "SpeechEnvelopeBucketSec"),
        "By speech chunk count": _agg_table(audit, "SpeechChunkCountBucket"),
        "By soft punctuation count": _agg_table(audit, "SoftPunctuationBucket"),
        "By planner speech/detected speech ratio": _agg_table(audit, "PlannerSpeechToDetectedBucket"),
    }
    for fname, df in [
        ("speech_span_by_event_count.csv", tables["By event count"]),
        ("speech_span_by_speech_envelope.csv", tables["By speech envelope"]),
        ("speech_span_by_speech_chunk_count.csv", tables["By speech chunk count"]),
        ("speech_span_by_soft_punctuation_count.csv", tables["By soft punctuation count"]),
        ("speech_span_by_planner_speech_ratio.csv", tables["By planner speech/detected speech ratio"]),
    ]:
        if not df.empty:
            _write(df, out / fname)

    top_over = _top_cases(audit, largest=True, n=top_n)
    top_under = _top_cases(audit, largest=False, n=top_n)
    _write(top_over, out / "worst_overrun_cases.csv")
    _write(top_under, out / "worst_underrun_cases.csv")

    pose_summary = _event_pose_summary(event_phase)
    if not pose_summary.empty:
        _write(pose_summary, out / "event_pose_phase_summary.csv")

    review = _review_md(corpus, tables, top_over, top_under)
    (out / "speech_span_audit_review.md").write_text(review, encoding="utf-8")
    return out


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="D09 audit committed viseme span vs detected speech span")
    parser.add_argument("--run", type=Path, required=True, help="LipLab run directory containing summary.csv and phase_timing_diagnostics/")
    parser.add_argument("--phase-dir", type=Path, help="Optional explicit D08 phase_timing_diagnostics directory")
    parser.add_argument("--out", type=Path, help="Optional output directory; default <run>/speech_span_audit")
    parser.add_argument("--top", type=int, default=25, help="Number of worst overrun/underrun cases to write")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = build_audit(args.run, args.phase_dir, args.out, args.top)
    print(f"wrote D09 speech span audit to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
