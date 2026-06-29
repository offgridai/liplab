#!/usr/bin/env python3
"""D10 offline planner duration decomposition auditor.

This script consumes a LipLab run summary (and D09 speech-span audit when
available) and answers a narrow question:

    Which planner duration terms explain the overlong committed speech span?

It does not change runtime behavior.  It is intentionally corpus-level: it
looks for systematic bias in the text/prosody duration model before we add more
online control logic.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUT_DIR_NAME = "planner_duration_decomposition"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - user-facing failure path
        raise RuntimeError(f"failed reading {path}: {exc}") from exc


def _find_existing(candidates: Iterable[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _safe_num(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _case_key(df: pd.DataFrame) -> pd.Series:
    for col in ("CaseDir", "case_id", "LineID"):
        if col in df.columns:
            return df[col].astype(str)
    raise ValueError("could not find case key column; expected CaseDir, case_id, or LineID")


def _bucket(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels, include_lowest=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _agg(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_col, dropna=False, observed=False)
    return g.agg(
        CaseCount=("CaseDir", "count"),
        EventCountMean=("EventCount", "mean"),
        SpeechEnvelopeMeanSec=("SpeechChunkEnvelopeSec", "mean"),
        PlannerSpeechMeanSec=("PlannerSpeechMaterialSec", "mean"),
        DetectedSpeechMeanSec=("DetectedSpeechMaterialSec", "mean"),
        PlannerPunctuationMeanSec=("PlannerPunctuationSec", "mean"),
        PlannerSpeechToDetectedMean=("PlannerSpeechToDetectedRatio", "mean"),
        SpanExcessMeanMs=("SpanExcessMs", "mean"),
        EndErrorMeanMs=("LastCenterMinusSpeechEndMs", "mean"),
        PlannerSpeechMinusDetectedMeanMs=("PlannerSpeechMinusDetectedMs", "mean"),
        PlannerPunctuationRatioMean=("PlannerPunctuationToDurationRatio", "mean"),
        DetectedSecPerEventMean=("DetectedSecPerEvent", "mean"),
        PlannerSecPerEventMean=("PlannerSecPerEvent", "mean"),
        ExcessPerEventMeanMs=("ExcessPerEventMs", "mean"),
        OverrunGE250Rate=("OverrunGE250", "mean"),
        OverrunGE500Rate=("OverrunGE500", "mean"),
    ).reset_index().sort_values(["SpanExcessMeanMs", "CaseCount"], ascending=[False, False])


def _corr_table(df: pd.DataFrame, target: str, cols: list[str]) -> pd.DataFrame:
    y = pd.to_numeric(df[target], errors="coerce") if target in df.columns else pd.Series(dtype=float)
    rows: list[dict[str, object]] = []
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3 or x[valid].std() == 0 or y[valid].std() == 0:
            corr = math.nan
        else:
            corr = float(x[valid].corr(y[valid]))
        rows.append({"Feature": col, "Target": target, "PearsonR": corr, "SampleCount": int(valid.sum())})
    return pd.DataFrame(rows).sort_values("PearsonR", key=lambda s: s.abs(), ascending=False)


def _fit_linear_model(df: pd.DataFrame, target: str, features: list[str], name: str) -> dict[str, object]:
    cols = [f for f in features if f in df.columns]
    if target not in df.columns or not cols:
        return {"Model": name, "FeatureCount": len(cols), "SampleCount": 0, "MAEMs": math.nan, "BiasMs": math.nan, "RMSEMs": math.nan, "Coefficients": ""}
    work = df[[target, *cols]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < max(5, len(cols) + 2):
        return {"Model": name, "FeatureCount": len(cols), "SampleCount": len(work), "MAEMs": math.nan, "BiasMs": math.nan, "RMSEMs": math.nan, "Coefficients": ""}
    y = work[target].to_numpy(dtype=float)
    x = work[cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    err_ms = (pred - y) * 1000.0
    coefs = [f"Intercept={beta[0]:.6g}"] + [f"{c}={b:.6g}" for c, b in zip(cols, beta[1:])]
    return {
        "Model": name,
        "FeatureCount": len(cols),
        "SampleCount": len(work),
        "MAEMs": float(np.mean(np.abs(err_ms))),
        "BiasMs": float(np.mean(err_ms)),
        "RMSEMs": float(np.sqrt(np.mean(err_ms * err_ms))),
        "Coefficients": "; ".join(coefs),
    }


def _summary(decomp: pd.DataFrame) -> pd.DataFrame:
    def mean(col: str) -> float:
        return float(pd.to_numeric(decomp.get(col, pd.Series(dtype=float)), errors="coerce").mean()) if len(decomp) else math.nan

    def median(col: str) -> float:
        return float(pd.to_numeric(decomp.get(col, pd.Series(dtype=float)), errors="coerce").median()) if len(decomp) else math.nan

    return pd.DataFrame([
        {
            "CaseCount": len(decomp),
            "TotalEventCount": int(pd.to_numeric(decomp.get("EventCount", pd.Series(dtype=float)), errors="coerce").sum()) if len(decomp) else 0,
            "SpanExcessMeanMs": mean("SpanExcessMs"),
            "SpanExcessMedianMs": median("SpanExcessMs"),
            "EndErrorMeanMs": mean("LastCenterMinusSpeechEndMs"),
            "EndErrorMedianMs": median("LastCenterMinusSpeechEndMs"),
            "PlannerSpeechMinusDetectedMeanMs": mean("PlannerSpeechMinusDetectedMs"),
            "PlannerSpeechMinusDetectedMedianMs": median("PlannerSpeechMinusDetectedMs"),
            "PlannerSpeechToDetectedMean": mean("PlannerSpeechToDetectedRatio"),
            "PlannerSpeechToDetectedMedian": median("PlannerSpeechToDetectedRatio"),
            "PlannerSecPerEventMean": mean("PlannerSecPerEvent"),
            "DetectedSecPerEventMean": mean("DetectedSecPerEvent"),
            "ExcessPerEventMeanMs": mean("ExcessPerEventMs"),
            "PlannerPunctuationMeanSec": mean("PlannerPunctuationSec"),
            "PlannerPunctuationToDurationRatioMean": mean("PlannerPunctuationToDurationRatio"),
            "OverrunGE250Rate": mean("OverrunGE250"),
            "OverrunGE500Rate": mean("OverrunGE500"),
        }
    ])


def _top(df: pd.DataFrame, col: str, n: int, ascending: bool = False) -> pd.DataFrame:
    cols = [
        "CaseDir", "EventCount", "TextPhraseCount", "SoftPunctuationBoundaryCount", "SpeechChunkCount",
        "CommittedCenterSpanSec", "SpeechChunkEnvelopeSec", "SpanExcessMs", "LastCenterMinusSpeechEndMs",
        "PlannerSpeechMaterialSec", "DetectedSpeechMaterialSec", "PlannerSpeechMinusDetectedMs",
        "PlannerSpeechToDetectedRatio", "PlannerPunctuationSec", "PlannerSecPerEvent", "DetectedSecPerEvent",
        "ExcessPerEventMs", "EstimatedTextDurationSec", "RuntimeCommittedDurationSec", "AudioDurationSec",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(col, ascending=ascending)[cols].head(n)


def _fmt(v: object, unit: str = "") -> str:
    try:
        f = float(v)
    except Exception:
        return "n/a"
    if math.isnan(f):
        return "n/a"
    return f"{f:.1f}{unit}"


def _review_md(summary: pd.DataFrame, corr: pd.DataFrame, models: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> str:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    lines: list[str] = []
    lines.append("# D10 Planner Duration Decomposition")
    lines.append("")
    lines.append("This report decomposes planner speech duration versus detected speech material. It is diagnostic only; no runtime behavior changes are implied.")
    lines.append("")
    lines.append("## Corpus headline")
    lines.append("")
    lines.append(f"- Cases: {int(row.get('CaseCount', 0) or 0)}")
    lines.append(f"- Events: {int(row.get('TotalEventCount', 0) or 0)}")
    lines.append(f"- Mean span excess: {_fmt(row.get('SpanExcessMeanMs'), ' ms')}")
    lines.append(f"- Mean final-center end error: {_fmt(row.get('EndErrorMeanMs'), ' ms')}")
    lines.append(f"- Mean planner speech minus detected speech: {_fmt(row.get('PlannerSpeechMinusDetectedMeanMs'), ' ms')}")
    lines.append(f"- Mean planner/detected speech ratio: {_fmt(row.get('PlannerSpeechToDetectedMean'))}")
    lines.append(f"- Mean planner seconds/event: {_fmt(row.get('PlannerSecPerEventMean'), ' sec')}")
    lines.append(f"- Mean detected seconds/event: {_fmt(row.get('DetectedSecPerEventMean'), ' sec')}")
    lines.append(f"- Mean excess/event: {_fmt(row.get('ExcessPerEventMeanMs'), ' ms')}")
    lines.append(f"- Mean planner punctuation seconds: {_fmt(row.get('PlannerPunctuationMeanSec'), ' sec')}")
    lines.append("")
    if not corr.empty:
        lines.append("## Strongest correlations with span excess")
        lines.append("")
        for _, r in corr.head(8).iterrows():
            lines.append(f"- {r['Feature']}: r={_fmt(r['PearsonR'])} (n={int(r['SampleCount'])})")
        lines.append("")
    if not models.empty:
        lines.append("## Simple detected-speech model fits")
        lines.append("")
        for _, r in models.sort_values("MAEMs").iterrows():
            lines.append(f"- {r['Model']}: MAE={_fmt(r['MAEMs'], ' ms')}, bias={_fmt(r['BiasMs'], ' ms')}, n={int(r['SampleCount'])}")
        lines.append("")
        lines.append("Lower MAE here means the features explain detected speech material better. This is not a final duration model; it identifies which simple predictors deserve D11 fitting.")
        lines.append("")
    for name, table in tables.items():
        if table.empty:
            continue
        lines.append(f"## {name}")
        lines.append("")
        cols = [c for c in table.columns if c != name][:8]
        for _, r in table.head(8).iterrows():
            label = r.get(name, r.iloc[0])
            pieces = [f"{c}={_fmt(r.get(c))}" for c in cols if c in table.columns and c != "CaseCount"]
            if "CaseCount" in table.columns:
                pieces.insert(0, f"n={int(r.get('CaseCount', 0) or 0)}")
            lines.append(f"- {label}: " + ", ".join(pieces[:7]))
        lines.append("")
    lines.append("## What to look for")
    lines.append("")
    lines.append("- If planner seconds/event is far above detected seconds/event, the per-viseme or prosody material budget is too slow.")
    lines.append("- If span excess scales mainly with punctuation seconds, punctuation priors are dominating.")
    lines.append("- If event-count models fit detected speech poorly, visible-viseme count is not a sufficient duration predictor.")
    lines.append("- D11 should fit the simplest duration prior that reduces planner/detected speech ratio without reintroducing early cutoffs.")
    lines.append("")
    return "\n".join(lines)


def build_decomposition(run: Path, summary_path: Path | None = None, speech_span_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if summary_path is None:
        summary_path = _find_existing([run / "summary.csv"])
    if summary_path is None:
        raise FileNotFoundError("missing summary.csv; pass --summary or --run pointing at a LipLab run")
    summary = _read_csv(summary_path)
    summary["CaseDir"] = _case_key(summary)

    if speech_span_path is None:
        speech_span_path = _find_existing([
            run / "speech_span_audit" / "case_speech_span_audit.csv",
            run / "case_speech_span_audit.csv",
        ])
    span = _read_csv(speech_span_path) if speech_span_path and speech_span_path.exists() else pd.DataFrame()
    if not span.empty:
        span["CaseDir"] = _case_key(span)
        # Prefer D09-derived columns for case envelope/span calculations.
        merge_cols = [c for c in span.columns if c == "CaseDir" or c not in summary.columns]
        # If both have a column, keep the D09 value under its normal name by dropping the summary duplicate first.
        for c in span.columns:
            if c != "CaseDir" and c in summary.columns:
                summary = summary.drop(columns=[c])
                merge_cols.append(c)
        df = summary.merge(span[merge_cols], on="CaseDir", how="left")
    else:
        df = summary.copy()

    out = pd.DataFrame()
    out["CaseDir"] = df["CaseDir"]
    for col in ["NPCID", "LineID", "case_id"]:
        if col in df.columns:
            out[col] = df[col]
    out["EventCount"] = _safe_num(df, "EventCount", default=np.nan)
    if out["EventCount"].isna().all():
        out["EventCount"] = _safe_num(df, "AU41EventCount", default=np.nan)
    out["TextPhraseCount"] = _safe_num(df, "TextPhraseCount", default=0)
    out["SoftPunctuationBoundaryCount"] = _safe_num(df, "SoftPunctuationBoundaryCount", default=0)
    out["SpeechChunkCount"] = _safe_num(df, "SpeechChunkCount", default=np.nan)
    out["AudioDurationSec"] = _safe_num(df, "AudioDurationSec", default=np.nan)
    out["EstimatedTextDurationSec"] = _safe_num(df, "EstimatedTextDurationSec", default=np.nan)
    out["RuntimeCommittedDurationSec"] = _safe_num(df, "RuntimeCommittedDurationSec", default=np.nan)
    out["CommittedCenterSpanSec"] = _safe_num(df, "CommittedCenterSpanSec", default=np.nan)
    out["SpeechChunkEnvelopeSec"] = _safe_num(df, "SpeechChunkEnvelopeSec", default=np.nan)
    out["LastCenterMinusSpeechEndMs"] = _safe_num(df, "LastCenterMinusSpeechEndMs", default=np.nan)
    out["FirstCenterMinusSpeechStartMs"] = _safe_num(df, "FirstCenterMinusSpeechStartMs", default=np.nan)

    out["PlannerDurationSec"] = _safe_num(df, "SpeechAttributionPlannerDurationSec", default=np.nan)
    out["PlannerSpeechMaterialSec"] = _safe_num(df, "SpeechAttributionPlannerSpeechMaterialSec", default=np.nan)
    out["DetectedSpeechMaterialSec"] = _safe_num(df, "SpeechAttributionDetectedSpeechMaterialSec", default=np.nan)
    out["PlannerPunctuationSec"] = _safe_num(df, "SpeechAttributionPlannerPunctuationSec", default=np.nan)
    out["DetectedGapAndSilenceRatio"] = _safe_num(df, "SpeechAttributionDetectedGapsAndSilenceToAudioRatio", default=np.nan)
    out["DetectedSpeechToAudioRatio"] = _safe_num(df, "SpeechAttributionDetectedSpeechToAudioRatio", default=np.nan)
    out["PlannerSpeechToDetectedRatio"] = _safe_num(df, "SpeechAttributionPlannerSpeechToDetectedSpeechRatio", default=np.nan)

    out["SpanExcessMs"] = _safe_num(df, "SpanExcessMs", default=np.nan)
    if out["SpanExcessMs"].isna().all():
        out["SpanExcessMs"] = (out["CommittedCenterSpanSec"] - out["SpeechChunkEnvelopeSec"]) * 1000.0
    out["CommittedSpanToSpeechEnvelopeRatio"] = _safe_num(df, "CommittedSpanToSpeechEnvelopeRatio", default=np.nan)
    if out["CommittedSpanToSpeechEnvelopeRatio"].isna().all():
        out["CommittedSpanToSpeechEnvelopeRatio"] = out["CommittedCenterSpanSec"] / out["SpeechChunkEnvelopeSec"].replace(0, np.nan)

    out["PlannerSpeechMinusDetectedSec"] = out["PlannerSpeechMaterialSec"] - out["DetectedSpeechMaterialSec"]
    out["PlannerSpeechMinusDetectedMs"] = out["PlannerSpeechMinusDetectedSec"] * 1000.0
    out["PlannerPunctuationToDurationRatio"] = out["PlannerPunctuationSec"] / out["PlannerDurationSec"].replace(0, np.nan)
    out["PlannerSpeechToPlannerDurationRatio"] = out["PlannerSpeechMaterialSec"] / out["PlannerDurationSec"].replace(0, np.nan)
    out["PlannerSecPerEvent"] = out["PlannerSpeechMaterialSec"] / out["EventCount"].replace(0, np.nan)
    out["DetectedSecPerEvent"] = out["DetectedSpeechMaterialSec"] / out["EventCount"].replace(0, np.nan)
    out["ExcessPerEventMs"] = (out["PlannerSecPerEvent"] - out["DetectedSecPerEvent"]) * 1000.0
    out["PunctuationSecPerBoundary"] = out["PlannerPunctuationSec"] / out["SoftPunctuationBoundaryCount"].replace(0, np.nan)
    out["OverrunGE250"] = out["SpanExcessMs"] >= 250.0
    out["OverrunGE500"] = out["SpanExcessMs"] >= 500.0

    out["EventCountBucket"] = _bucket(out["EventCount"], [0, 8, 16, 24, 32, 48, 10_000], ["1-8", "9-16", "17-24", "25-32", "33-48", "49+"])
    out["SoftPunctuationBucket"] = _bucket(out["SoftPunctuationBoundaryCount"], [-1, 0, 1, 2, 3, 10_000], ["0", "1", "2", "3", "4+"])
    out["PlannerSpeechToDetectedBucket"] = _bucket(out["PlannerSpeechToDetectedRatio"], [0, 0.75, 1.0, 1.25, 1.5, 2.0, 10_000], ["<0.75", "0.75-1.00", "1.00-1.25", "1.25-1.50", "1.50-2.00", "2.00+"])
    out["DetectedSecPerEventBucket"] = _bucket(out["DetectedSecPerEvent"], [0, .04, .06, .08, .10, .12, 10], ["<40ms", "40-60ms", "60-80ms", "80-100ms", "100-120ms", "120ms+"])

    tables = {
        "by_event_count": _agg(out, "EventCountBucket"),
        "by_punctuation_count": _agg(out, "SoftPunctuationBucket"),
        "by_planner_speech_ratio": _agg(out, "PlannerSpeechToDetectedBucket"),
        "by_detected_sec_per_event": _agg(out, "DetectedSecPerEventBucket"),
    }
    return out, tables


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run) if args.run else Path(".")
    summary_path = Path(args.summary) if args.summary else None
    span_path = Path(args.speech_span_audit) if args.speech_span_audit else None
    out_dir = Path(args.output) if args.output else run_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    decomp, tables = build_decomposition(run_dir, summary_path=summary_path, speech_span_path=span_path)
    summary = _summary(decomp)
    corr_features = [
        "EventCount", "TextPhraseCount", "SoftPunctuationBoundaryCount", "SpeechChunkCount",
        "PlannerDurationSec", "PlannerSpeechMaterialSec", "DetectedSpeechMaterialSec", "PlannerPunctuationSec",
        "PlannerSpeechMinusDetectedMs", "PlannerSpeechToDetectedRatio", "PlannerSecPerEvent",
        "DetectedSecPerEvent", "ExcessPerEventMs", "PlannerPunctuationToDurationRatio",
        "DetectedGapAndSilenceRatio", "DetectedSpeechToAudioRatio",
    ]
    corr = _corr_table(decomp, "SpanExcessMs", corr_features)

    models = pd.DataFrame([
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["EventCount"], "event_count_only"),
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["PlannerSpeechMaterialSec"], "planner_speech_only"),
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["EstimatedTextDurationSec"], "estimated_text_duration_only"),
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["EventCount", "SoftPunctuationBoundaryCount"], "event_count_plus_punctuation"),
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["EventCount", "TextPhraseCount", "SoftPunctuationBoundaryCount"], "event_phrase_punctuation"),
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["PlannerSpeechMaterialSec", "PlannerPunctuationSec"], "planner_speech_plus_punctuation"),
        _fit_linear_model(decomp, "DetectedSpeechMaterialSec", ["EventCount", "PlannerPunctuationSec", "TextPhraseCount"], "event_count_planner_punctuation_phrase"),
    ]).sort_values("MAEMs")

    _write(decomp, out_dir / "case_planner_duration_decomposition.csv")
    _write(summary, out_dir / "planner_duration_decomposition_summary.csv")
    _write(corr, out_dir / "planner_duration_correlations.csv")
    _write(models, out_dir / "detected_speech_model_fits.csv")
    _write(_top(decomp, "SpanExcessMs", args.top_n, ascending=False), out_dir / "worst_span_overrun_cases.csv")
    _write(_top(decomp, "SpanExcessMs", args.top_n, ascending=True), out_dir / "worst_span_underrun_cases.csv")
    for name, table in tables.items():
        _write(table, out_dir / f"planner_duration_{name}.csv")

    review = _review_md(
        summary,
        corr,
        models,
        {
            "by_planner_speech_ratio": tables.get("by_planner_speech_ratio", pd.DataFrame()),
            "by_event_count": tables.get("by_event_count", pd.DataFrame()),
            "by_punctuation_count": tables.get("by_punctuation_count", pd.DataFrame()),
            "by_detected_sec_per_event": tables.get("by_detected_sec_per_event", pd.DataFrame()),
        },
    )
    (out_dir / "planner_duration_decomposition_review.md").write_text(review, encoding="utf-8")

    print(f"Wrote D10 planner duration decomposition to {out_dir}")
    print(review.split("\n\n", 2)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="D10 planner duration decomposition auditor")
    parser.add_argument("--run", default=".", help="LipLab run directory containing summary.csv and optionally speech_span_audit/")
    parser.add_argument("--summary", default=None, help="Explicit summary.csv path")
    parser.add_argument("--speech-span-audit", default=None, help="Explicit D09 case_speech_span_audit.csv path")
    parser.add_argument("--output", default=None, help="Output directory; defaults to <run>/planner_duration_decomposition")
    parser.add_argument("--top-n", type=int, default=40, help="Number of worst overrun/underrun cases to write")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
