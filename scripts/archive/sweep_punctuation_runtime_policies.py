#!/usr/bin/env python3
"""
Sweep punctuation-boundary runtime policies against the offline punctuation oracle.

This is an offline research tool. It does not change runtime behavior.

Input is the D01 oracle boundary CSV, normally:
  <run>/punctuation_oracle/punctuation_resume_oracle_boundaries.csv

The tool asks a deliberately narrow question:

  If the planner used a given punctuation pause prior, and runtime were allowed
  to make only a bounded correction when a detected acoustic resume falls in a
  local window, how close would the next phrase start land to the oracle resume?

It is intended to test candidate priors/windows before implementing runtime
punctuation correction in the shared lipsync library.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Boundary:
    case_id: str
    boundary_index: int
    punctuation: str
    kind: str
    hard_sentence: bool
    prev_word: str
    next_word: str
    prev_last_center_sec: float
    current_next_first_sec: float
    raw_next_first_sec: float
    detected: bool
    resume_sec: Optional[float]
    confidence: float

    @property
    def boundary_class(self) -> str:
        if self.kind:
            return self.kind
        if self.punctuation in {".", "?", "!"}:
            return "sentence"
        if self.punctuation == ",":
            return "comma"
        if self.punctuation in {"-", "—", "–"}:
            return "dash"
        if self.punctuation in {";", ":"}:
            return "clause"
        return "other"


@dataclass(frozen=True)
class PriorPolicy:
    name: str
    comma_ms: float
    sentence_ms: float
    dash_ms: float
    clause_ms: float
    other_ms: float

    def prior_for(self, b: Boundary) -> float:
        cls = b.boundary_class
        if cls == "comma":
            return self.comma_ms
        if cls == "sentence":
            return self.sentence_ms
        if cls == "dash":
            return self.dash_ms
        if cls in {"semicolon", "colon", "clause"}:
            return self.clause_ms
        return self.other_ms


@dataclass(frozen=True)
class RuntimePolicy:
    name: str
    visual_lead_ms: float
    window_before_ms: float
    window_after_ms: float
    max_advance_ms: float
    max_delay_ms: float
    min_confidence: float
    class_window_scale_sentence: float = 1.0
    class_window_scale_comma: float = 1.0

    def scaled_window(self, b: Boundary) -> Tuple[float, float]:
        scale = 1.0
        if b.boundary_class == "sentence":
            scale = self.class_window_scale_sentence
        elif b.boundary_class == "comma":
            scale = self.class_window_scale_comma
        return self.window_before_ms * scale, self.window_after_ms * scale


@dataclass
class EvalRow:
    policy: str
    prior_policy: str
    runtime_policy: str
    case_id: str
    boundary_index: int
    punctuation: str
    boundary_class: str
    detected: bool
    used_detection: bool
    confidence: float
    prior_ms: float
    planned_start_sec: float
    target_start_sec: Optional[float]
    final_start_sec: float
    current_next_first_sec: float
    raw_next_first_sec: float
    error_ms: Optional[float]
    current_error_ms: Optional[float]
    raw_error_ms: Optional[float]
    correction_ms: float
    reject_reason: str


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    text = (row.get(key) or "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _int(row: Dict[str, str], key: str, default: int = 0) -> int:
    text = (row.get(key) or "").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _bool(row: Dict[str, str], key: str) -> bool:
    text = (row.get(key) or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_boundaries(path: Path) -> List[Boundary]:
    out: List[Boundary] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resume_text = (row.get("ResumeSec") or "").strip()
            resume = None
            if resume_text:
                try:
                    resume = float(resume_text)
                except ValueError:
                    resume = None
            out.append(
                Boundary(
                    case_id=row.get("CaseID", ""),
                    boundary_index=_int(row, "BoundaryIndex"),
                    punctuation=row.get("Punctuation", ""),
                    kind=row.get("PunctuationKind", "") or row.get("Class", ""),
                    hard_sentence=_bool(row, "HardSentenceBreak"),
                    prev_word=row.get("PrevWord", ""),
                    next_word=row.get("NextWord", ""),
                    prev_last_center_sec=_float(row, "PrevLastCenterSec"),
                    current_next_first_sec=_float(row, "NextFirstCenterSec"),
                    raw_next_first_sec=_float(row, "RawNextFirstCenterSec", _float(row, "NextFirstCenterSec")),
                    detected=_bool(row, "Detected"),
                    resume_sec=resume,
                    confidence=_float(row, "Confidence"),
                )
            )
    return out


def default_prior_policies() -> List[PriorPolicy]:
    return [
        PriorPolicy("legacy_175", comma_ms=175, sentence_ms=175, dash_ms=196, clause_ms=220, other_ms=175),
        PriorPolicy("d03_470", comma_ms=470, sentence_ms=475, dash_ms=525, clause_ms=500, other_ms=470),
        PriorPolicy("moderate_300_375", comma_ms=300, sentence_ms=375, dash_ms=450, clause_ms=400, other_ms=325),
        PriorPolicy("compact_250_325", comma_ms=250, sentence_ms=325, dash_ms=400, clause_ms=350, other_ms=275),
        PriorPolicy("long_sentence_300_450", comma_ms=300, sentence_ms=450, dash_ms=525, clause_ms=450, other_ms=350),
        PriorPolicy("list_fast_220_sentence_390", comma_ms=220, sentence_ms=390, dash_ms=475, clause_ms=380, other_ms=300),
        PriorPolicy("d02_p25ish", comma_ms=290, sentence_ms=320, dash_ms=310, clause_ms=330, other_ms=300),
        PriorPolicy("d02_medianish", comma_ms=470, sentence_ms=470, dash_ms=520, clause_ms=500, other_ms=470),
    ]


def default_runtime_policies() -> List[RuntimePolicy]:
    return [
        RuntimePolicy("no_runtime", visual_lead_ms=0, window_before_ms=0, window_after_ms=0, max_advance_ms=0, max_delay_ms=0, min_confidence=2.0),
        RuntimePolicy("bounded_80_160_lead50", visual_lead_ms=50, window_before_ms=160, window_after_ms=220, max_advance_ms=80, max_delay_ms=160, min_confidence=0.65),
        RuntimePolicy("bounded_120_220_lead50", visual_lead_ms=50, window_before_ms=220, window_after_ms=280, max_advance_ms=120, max_delay_ms=220, min_confidence=0.65),
        RuntimePolicy("comma_tight_sentence_wide", visual_lead_ms=50, window_before_ms=180, window_after_ms=240, max_advance_ms=100, max_delay_ms=220, min_confidence=0.65, class_window_scale_sentence=1.35, class_window_scale_comma=0.85),
        RuntimePolicy("delay_only_220_lead50", visual_lead_ms=50, window_before_ms=80, window_after_ms=300, max_advance_ms=0, max_delay_ms=220, min_confidence=0.65),
        RuntimePolicy("loose_160_300_lead60", visual_lead_ms=60, window_before_ms=300, window_after_ms=360, max_advance_ms=160, max_delay_ms=300, min_confidence=0.55),
        RuntimePolicy("very_safe_60_120_lead40", visual_lead_ms=40, window_before_ms=120, window_after_ms=180, max_advance_ms=60, max_delay_ms=120, min_confidence=0.75),
    ]


def evaluate(boundaries: Sequence[Boundary], prior: PriorPolicy, runtime: RuntimePolicy) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for b in boundaries:
        prior_ms = prior.prior_for(b)
        planned = b.prev_last_center_sec + prior_ms / 1000.0
        target = None
        err = None
        current_err = None
        raw_err = None
        final = planned
        used = False
        reason = "no_detection"
        if b.detected and b.resume_sec is not None and b.confidence >= runtime.min_confidence:
            target = b.resume_sec - runtime.visual_lead_ms / 1000.0
            before_ms, after_ms = runtime.scaled_window(b)
            window_start = planned - before_ms / 1000.0
            window_end = planned + after_ms / 1000.0
            if target < window_start:
                reason = "target_before_window"
            elif target > window_end:
                reason = "target_after_window"
            else:
                desired_delta_ms = (target - planned) * 1000.0
                clamped_delta_ms = max(-runtime.max_advance_ms, min(runtime.max_delay_ms, desired_delta_ms))
                final = planned + clamped_delta_ms / 1000.0
                used = abs(clamped_delta_ms) > 1e-6
                reason = "accepted" if used else "already_aligned"
        elif b.detected and b.resume_sec is not None:
            reason = "confidence_too_low"

        if target is not None:
            err = (final - target) * 1000.0
            current_err = (b.current_next_first_sec - target) * 1000.0
            raw_err = (b.raw_next_first_sec - target) * 1000.0

        rows.append(
            EvalRow(
                policy=f"{prior.name}+{runtime.name}",
                prior_policy=prior.name,
                runtime_policy=runtime.name,
                case_id=b.case_id,
                boundary_index=b.boundary_index,
                punctuation=b.punctuation,
                boundary_class=b.boundary_class,
                detected=b.detected,
                used_detection=used,
                confidence=b.confidence,
                prior_ms=prior_ms,
                planned_start_sec=planned,
                target_start_sec=target,
                final_start_sec=final,
                current_next_first_sec=b.current_next_first_sec,
                raw_next_first_sec=b.raw_next_first_sec,
                error_ms=err,
                current_error_ms=current_err,
                raw_error_ms=raw_err,
                correction_ms=(final - planned) * 1000.0,
                reject_reason=reason,
            )
        )
    return rows


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize(rows: Sequence[EvalRow], group_key: Optional[str] = None) -> List[Dict[str, object]]:
    groups: Dict[str, List[EvalRow]] = {}
    for r in rows:
        if group_key == "class":
            key = f"{r.policy}|{r.boundary_class}"
        elif group_key == "punctuation":
            key = f"{r.policy}|{r.punctuation}"
        else:
            key = r.policy
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, rs in sorted(groups.items()):
        usable = [r for r in rs if r.error_ms is not None]
        abs_err = [abs(r.error_ms or 0.0) for r in usable]
        signed_err = [r.error_ms or 0.0 for r in usable]
        current_abs = [abs(r.current_error_ms or 0.0) for r in usable if r.current_error_ms is not None]
        raw_abs = [abs(r.raw_error_ms or 0.0) for r in usable if r.raw_error_ms is not None]
        corrections = [r.correction_ms for r in rs]
        accepted = [r for r in rs if r.used_detection]
        base = rs[0]
        rec: Dict[str, object] = {
            "Policy": key,
            "PriorPolicy": base.prior_policy,
            "RuntimePolicy": base.runtime_policy,
            "BoundaryClass": base.boundary_class if group_key == "class" else "",
            "Punctuation": base.punctuation if group_key == "punctuation" else "",
            "BoundaryCount": len(rs),
            "OracleDetectedCount": sum(1 for r in rs if r.detected),
            "OracleDetectedRate": sum(1 for r in rs if r.detected) / len(rs) if rs else 0.0,
            "EvaluatedCount": len(usable),
            "RuntimeAcceptedCount": len(accepted),
            "RuntimeAcceptedRateOfDetected": len(accepted) / len(usable) if usable else 0.0,
            "FinalAbsErrorMeanMs": mean(abs_err) if abs_err else "",
            "FinalAbsErrorMedianMs": median(abs_err) if abs_err else "",
            "FinalAbsErrorP75Ms": percentile(abs_err, 0.75) if abs_err else "",
            "FinalAbsErrorP90Ms": percentile(abs_err, 0.90) if abs_err else "",
            "FinalSignedErrorMeanMs": mean(signed_err) if signed_err else "",
            "FinalSignedErrorMedianMs": median(signed_err) if signed_err else "",
            "CurrentAbsErrorMeanMs": mean(current_abs) if current_abs else "",
            "RawAbsErrorMeanMs": mean(raw_abs) if raw_abs else "",
            "ImprovementVsCurrentMeanMs": (mean(current_abs) - mean(abs_err)) if current_abs and abs_err else "",
            "ImprovementVsRawMeanMs": (mean(raw_abs) - mean(abs_err)) if raw_abs and abs_err else "",
            "Within50MsRate": sum(1 for e in abs_err if e <= 50.0) / len(abs_err) if abs_err else "",
            "Within100MsRate": sum(1 for e in abs_err if e <= 100.0) / len(abs_err) if abs_err else "",
            "VeryEarlyOver150MsRate": sum(1 for e in signed_err if e < -150.0) / len(signed_err) if signed_err else "",
            "VeryLateOver150MsRate": sum(1 for e in signed_err if e > 150.0) / len(signed_err) if signed_err else "",
            "CorrectionAbsMeanMs": mean([abs(c) for c in corrections]) if corrections else "",
            "CorrectionAbsP90Ms": percentile([abs(c) for c in corrections], 0.90) if corrections else "",
        }
        reasons: Dict[str, int] = {}
        for r in rs:
            reasons[r.reject_reason] = reasons.get(r.reject_reason, 0) + 1
        for reason, count in sorted(reasons.items()):
            rec[f"Reason_{reason}"] = count
        out.append(rec)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_eval_rows(path: Path, rows: Sequence[EvalRow]) -> None:
    dicts: List[Dict[str, object]] = []
    for r in rows:
        dicts.append({
            "Policy": r.policy,
            "PriorPolicy": r.prior_policy,
            "RuntimePolicy": r.runtime_policy,
            "CaseID": r.case_id,
            "BoundaryIndex": r.boundary_index,
            "Punctuation": r.punctuation,
            "BoundaryClass": r.boundary_class,
            "OracleDetected": int(r.detected),
            "RuntimeUsedDetection": int(r.used_detection),
            "Confidence": r.confidence,
            "PriorMs": r.prior_ms,
            "PlannedStartSec": r.planned_start_sec,
            "TargetStartSec": "" if r.target_start_sec is None else r.target_start_sec,
            "FinalStartSec": r.final_start_sec,
            "CurrentNextFirstSec": r.current_next_first_sec,
            "RawNextFirstSec": r.raw_next_first_sec,
            "ErrorMs": "" if r.error_ms is None else r.error_ms,
            "CurrentErrorMs": "" if r.current_error_ms is None else r.current_error_ms,
            "RawErrorMs": "" if r.raw_error_ms is None else r.raw_error_ms,
            "CorrectionMs": r.correction_ms,
            "RejectReason": r.reject_reason,
        })
    write_csv(path, dicts)


def infer_boundaries_path(run: Optional[Path], boundaries: Optional[Path]) -> Path:
    if boundaries:
        return boundaries
    if not run:
        raise SystemExit("Provide --run or --boundaries")
    candidate = run / "punctuation_oracle" / "punctuation_resume_oracle_boundaries.csv"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Could not find D01 boundary CSV at {candidate}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep punctuation runtime policies against D01 oracle boundaries.")
    parser.add_argument("--run", type=Path, help="LipLab run directory containing punctuation_oracle output")
    parser.add_argument("--boundaries", type=Path, help="Path to punctuation_resume_oracle_boundaries.csv")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to <run>/punctuation_policy_sweeps")
    parser.add_argument("--write-by-boundary", action="store_true", help="Also write the large per-boundary policy table")
    args = parser.parse_args()

    boundaries_path = infer_boundaries_path(args.run, args.boundaries)
    out_dir = args.out
    if out_dir is None:
        if args.run:
            out_dir = args.run / "punctuation_policy_sweeps"
        else:
            out_dir = boundaries_path.parent / ".." / "punctuation_policy_sweeps"
    out_dir = out_dir.resolve()

    boundaries = load_boundaries(boundaries_path)
    priors = default_prior_policies()
    runtimes = default_runtime_policies()

    all_rows: List[EvalRow] = []
    for prior in priors:
        for runtime in runtimes:
            all_rows.extend(evaluate(boundaries, prior, runtime))

    summary = summarize(all_rows)
    by_class = summarize(all_rows, group_key="class")
    by_punctuation = summarize(all_rows, group_key="punctuation")

    write_csv(out_dir / "punctuation_policy_sweep_summary.csv", summary)
    write_csv(out_dir / "punctuation_policy_sweep_by_class.csv", by_class)
    write_csv(out_dir / "punctuation_policy_sweep_by_punctuation.csv", by_punctuation)
    if args.write_by_boundary:
        write_eval_rows(out_dir / "punctuation_policy_sweep_by_boundary.csv", all_rows)

    ranked = sorted(
        summary,
        key=lambda r: (
            float(r["FinalAbsErrorMeanMs"]) if r.get("FinalAbsErrorMeanMs") != "" else 1e9,
            float(r["FinalAbsErrorP90Ms"]) if r.get("FinalAbsErrorP90Ms") != "" else 1e9,
        ),
    )
    md = [
        "# Punctuation Runtime Policy Sweep", "",
        f"Input: `{boundaries_path}`", "",
        "This offline sweep tests planner punctuation priors plus bounded runtime resume correction policies against the D01 oracle boundaries.", "",
        "## Top policies by mean absolute target-start error", "",
        "| Rank | Policy | Mean abs error | P90 abs error | Accepted/detected | vs current |", 
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(ranked[:12], 1):
        md.append(
            f"| {i} | {row['Policy']} | {float(row['FinalAbsErrorMeanMs']):.1f} ms | "
            f"{float(row['FinalAbsErrorP90Ms']):.1f} ms | {float(row['RuntimeAcceptedRateOfDetected']):.3f} | "
            f"{float(row['ImprovementVsCurrentMeanMs']):.1f} ms |"
        )
    md.extend([
        "", "## Interpretation guide", "",
        "- `no_runtime` rows estimate how good a planner prior is without any resume correction.",
        "- Runtime policies only use an oracle resume when it falls inside a bounded window around the planner prior.",
        "- `visual_lead` models launching the visible phrase slightly before the acoustic restart.",
        "- High `Reason_target_before_window` or `Reason_target_after_window` means the prior/window misses real resumes.",
    ])
    (out_dir / "punctuation_policy_sweep_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Loaded {len(boundaries)} boundaries from {boundaries_path}")
    print(f"Wrote {out_dir}")
    if ranked:
        best = ranked[0]
        print(
            "Best: "
            f"{best['Policy']} mean_abs={float(best['FinalAbsErrorMeanMs']):.1f}ms "
            f"p90={float(best['FinalAbsErrorP90Ms']):.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
