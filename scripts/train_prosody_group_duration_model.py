#!/usr/bin/env python3
"""Train/assess simple prosody-group duration priors from alpha13 diagnostics.

This is intentionally small and dependency-free.  It consumes
corpus_prosody_group_timing_diagnostics.csv emitted by summarize_run.py and
writes candidate model diagnostics that can be inspected before any runtime
constants are changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Iterable


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_int(value: str | None) -> int | None:
    f = parse_float(value)
    if f is None:
        return None
    return int(f)


def fmt(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.9g}"


def safe_div(a: float, b: float) -> float | None:
    if abs(b) <= 1e-9:
        return None
    return a / b


def stable_split(case_id: str, valid_percent: int) -> str:
    h = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "valid" if bucket < valid_percent else "train"


def event_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2_3"
    if n <= 7:
        return "4_7"
    return "8_plus"


def median_or_default(values: Iterable[float], default: float) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return median(vals) if vals else default


class Model:
    def __init__(self, name: str, predict: Callable[[dict], float], coefficients: list[dict[str, str]]):
        self.name = name
        self.predict = predict
        self.coefficients = coefficients


def load_rows(path: Path, valid_percent: int, min_evidence_events: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            evidence = parse_float(raw.get("EvidenceGroupCenterSpanSec"))
            planner = parse_float(raw.get("PlannerGroupAllocatedSec"))
            event_count = parse_int(raw.get("EventCount")) or 0
            evidence_event_count = parse_int(raw.get("EvidenceEventCount")) or 0
            if evidence is None or evidence <= 1e-6:
                continue
            if planner is None or planner < 0.0:
                continue
            if event_count <= 0 or evidence_event_count < min_evidence_events:
                continue
            if evidence_event_count < event_count:
                # Do not train on partially observed groups; keep model targets crisp.
                continue
            row = dict(raw)
            row["_evidence"] = evidence
            row["_planner"] = planner
            row["_event_count"] = event_count
            row["_role"] = (raw.get("PlannerProsodyRole") or "unknown").strip() or "unknown"
            row["_event_bucket"] = event_bucket(event_count)
            row["_is_initial_text_island"] = "1" if (parse_int(raw.get("TextIslandIndex")) or 0) == 0 else "0"
            row["_is_initial_phrase"] = "1" if (parse_int(raw.get("PhraseIndex")) or 0) == 0 else "0"
            row["_split"] = stable_split(raw.get("case_id", ""), valid_percent)
            rows.append(row)
    return rows


def fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (0.0, median(ys) if ys else 0.0)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 1e-12:
        return (0.0, my)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / var
    intercept = my - slope * mx
    return (slope, intercept)


def build_models(train: list[dict]) -> list[Model]:
    global_median = median_or_default((r["_evidence"] for r in train), 0.25)
    planner_ratios = [r["_evidence"] / r["_planner"] for r in train if r["_planner"] > 1e-6]
    planner_scale = median_or_default(planner_ratios, 1.0)
    event_rate = median_or_default((r["_evidence"] / r["_event_count"] for r in train if r["_event_count"] > 0), 0.12)
    slope, intercept = fit_affine([r["_planner"] for r in train], [r["_evidence"] for r in train])

    role_median: dict[str, float] = {}
    role_event_rate: dict[str, float] = {}
    role_bucket_median: dict[tuple[str, str], float] = {}
    role_initial_bucket_median: dict[tuple[str, str, str], float] = {}
    by_role: dict[str, list[dict]] = defaultdict(list)
    by_role_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_role_initial_bucket: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in train:
        by_role[r["_role"]].append(r)
        by_role_bucket[(r["_role"], r["_event_bucket"])].append(r)
        by_role_initial_bucket[(r["_role"], r["_is_initial_text_island"], r["_event_bucket"])].append(r)
    for role, rs in by_role.items():
        role_median[role] = median_or_default((r["_evidence"] for r in rs), global_median)
        role_event_rate[role] = median_or_default((r["_evidence"] / r["_event_count"] for r in rs if r["_event_count"] > 0), event_rate)
    for key, rs in by_role_bucket.items():
        role_bucket_median[key] = median_or_default((r["_evidence"] for r in rs), global_median)
    for key, rs in by_role_initial_bucket.items():
        role_initial_bucket_median[key] = median_or_default((r["_evidence"] for r in rs), global_median)

    models: list[Model] = []
    models.append(Model("current_planner_allocated", lambda r: r["_planner"], []))
    models.append(Model("fixed_global_median", lambda r, gm=global_median: gm, [{"model":"fixed_global_median","feature":"global_median_sec","value":fmt(global_median)}]))
    models.append(Model("planner_scaled_median_ratio", lambda r, s=planner_scale: r["_planner"] * s, [{"model":"planner_scaled_median_ratio","feature":"median_evidence_to_planner_ratio","value":fmt(planner_scale)}]))
    models.append(Model("planner_affine", lambda r, a=slope, b=intercept: max(0.02, b + a * r["_planner"]), [
        {"model":"planner_affine","feature":"slope","value":fmt(slope)},
        {"model":"planner_affine","feature":"intercept_sec","value":fmt(intercept)},
    ]))
    models.append(Model("event_count_rate", lambda r, er=event_rate: r["_event_count"] * er, [{"model":"event_count_rate","feature":"median_sec_per_event","value":fmt(event_rate)}]))
    models.append(Model("role_median", lambda r: role_median.get(r["_role"], global_median), [
        {"model":"role_median","feature":role,"value":fmt(v)} for role, v in sorted(role_median.items())
    ]))
    models.append(Model("role_event_count_rate", lambda r: r["_event_count"] * role_event_rate.get(r["_role"], event_rate), [
        {"model":"role_event_count_rate","feature":role,"value":fmt(v)} for role, v in sorted(role_event_rate.items())
    ]))
    models.append(Model("role_event_bucket_median", lambda r: role_bucket_median.get((r["_role"], r["_event_bucket"]), role_median.get(r["_role"], global_median)), [
        {"model":"role_event_bucket_median","feature":f"{role}|{bucket}","value":fmt(v)} for (role, bucket), v in sorted(role_bucket_median.items())
    ]))
    models.append(Model("role_initial_event_bucket_median", lambda r: role_initial_bucket_median.get((r["_role"], r["_is_initial_text_island"], r["_event_bucket"]), role_bucket_median.get((r["_role"], r["_event_bucket"]), role_median.get(r["_role"], global_median))), [
        {"model":"role_initial_event_bucket_median","feature":f"{role}|initial={initial}|{bucket}","value":fmt(v)} for (role, initial, bucket), v in sorted(role_initial_bucket_median.items())
    ]))
    return models


def summarize_errors(rows: list[dict], model: Model) -> dict[str, str]:
    errors = []
    abs_errors = []
    sq_errors = []
    ratios = []
    pred_total = 0.0
    evidence_total = 0.0
    for r in rows:
        p = max(0.0, model.predict(r))
        e = r["_evidence"]
        err = p - e
        errors.append(err)
        abs_errors.append(abs(err))
        sq_errors.append(err * err)
        if e > 1e-9:
            ratios.append(p / e)
        pred_total += p
        evidence_total += e
    if not rows:
        return {"count":"0"}
    abs_sorted = sorted(abs_errors)
    return {
        "count": str(len(rows)),
        "predicted_total_sec": fmt(pred_total),
        "evidence_total_sec": fmt(evidence_total),
        "predicted_to_evidence_ratio": fmt(pred_total / evidence_total) if evidence_total > 1e-9 else "",
        "mae_sec": fmt(sum(abs_errors) / len(abs_errors)),
        "median_abs_error_sec": fmt(median(abs_errors)),
        "p90_abs_error_sec": fmt(abs_sorted[int(0.9 * (len(abs_sorted)-1))]),
        "rmse_sec": fmt(math.sqrt(sum(sq_errors) / len(sq_errors))),
        "bias_sec": fmt(sum(errors) / len(errors)),
        "median_ratio": fmt(median(ratios)) if ratios else "",
    }


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    train = [r for r in rows if r["_split"] == "train"]
    valid = [r for r in rows if r["_split"] == "valid"]
    models = build_models(train or rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "corpus_prosody_group_duration_model_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "split", "count", "predicted_total_sec", "evidence_total_sec", "predicted_to_evidence_ratio", "mae_sec", "median_abs_error_sec", "p90_abs_error_sec", "rmse_sec", "bias_sec", "median_ratio"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model in models:
            for split_name, split_rows in [("train", train), ("valid", valid), ("all", rows)]:
                rec = {"model": model.name, "split": split_name}
                rec.update(summarize_errors(split_rows, model))
                w.writerow(rec)

    with (out_dir / "corpus_prosody_group_duration_model_coefficients.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "feature", "value"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model in models:
            for rec in model.coefficients:
                w.writerow({k: rec.get(k, "") for k in fields})

    with (out_dir / "corpus_prosody_group_duration_model_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        model_names = [m.name for m in models]
        fields = [
            "case_id", "split", "TextIslandIndex", "PhraseIndex", "PlannerProsodyGroupIndex",
            "PlannerProsodyRole", "EventCount", "EventCountBucket", "FirstWord", "LastWord",
            "EvidenceGroupCenterSpanSec", "PlannerGroupAllocatedSec",
        ]
        for name in model_names:
            fields.extend([f"{name}_PredSec", f"{name}_ErrSec", f"{name}_AbsErrSec"])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = {
                "case_id": r.get("case_id", ""),
                "split": r["_split"],
                "TextIslandIndex": r.get("TextIslandIndex", ""),
                "PhraseIndex": r.get("PhraseIndex", ""),
                "PlannerProsodyGroupIndex": r.get("PlannerProsodyGroupIndex", ""),
                "PlannerProsodyRole": r.get("PlannerProsodyRole", ""),
                "EventCount": r.get("EventCount", ""),
                "EventCountBucket": r["_event_bucket"],
                "FirstWord": r.get("FirstWord", ""),
                "LastWord": r.get("LastWord", ""),
                "EvidenceGroupCenterSpanSec": fmt(r["_evidence"]),
                "PlannerGroupAllocatedSec": fmt(r["_planner"]),
            }
            for model in models:
                p = max(0.0, model.predict(r))
                err = p - r["_evidence"]
                rec[f"{model.name}_PredSec"] = fmt(p)
                rec[f"{model.name}_ErrSec"] = fmt(err)
                rec[f"{model.name}_AbsErrSec"] = fmt(abs(err))
            w.writerow(rec)

    # Compact recommendation table sorted by validation MAE, then all MAE.
    candidate_rows = []
    for model in models:
        valid_stats = summarize_errors(valid, model)
        all_stats = summarize_errors(rows, model)
        candidate_rows.append((
            parse_float(valid_stats.get("mae_sec")) or 999.0,
            parse_float(all_stats.get("mae_sec")) or 999.0,
            model.name,
            valid_stats,
            all_stats,
        ))
    candidate_rows.sort()
    with (out_dir / "corpus_prosody_group_duration_model_recommendation.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["rank", "model", "valid_mae_sec", "valid_median_abs_error_sec", "valid_bias_sec", "valid_predicted_to_evidence_ratio", "all_mae_sec", "all_predicted_to_evidence_ratio", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, (_vmae, _amae, name, vs, alls) in enumerate(candidate_rows, start=1):
            note = "diagnostic_candidate_only"
            if name == "current_planner_allocated":
                note = "baseline_current_runtime_prior"
            w.writerow({
                "rank": str(rank),
                "model": name,
                "valid_mae_sec": vs.get("mae_sec", ""),
                "valid_median_abs_error_sec": vs.get("median_abs_error_sec", ""),
                "valid_bias_sec": vs.get("bias_sec", ""),
                "valid_predicted_to_evidence_ratio": vs.get("predicted_to_evidence_ratio", ""),
                "all_mae_sec": alls.get("mae_sec", ""),
                "all_predicted_to_evidence_ratio": alls.get("predicted_to_evidence_ratio", ""),
                "note": note,
            })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Run/summary directory containing corpus_prosody_group_timing_diagnostics.csv")
    ap.add_argument("--valid-percent", type=int, default=20, help="deterministic validation split percent by case id")
    ap.add_argument("--min-evidence-events", type=int, default=1, help="minimum evidence events required per group")
    args = ap.parse_args()
    src = args.run_dir / "corpus_prosody_group_timing_diagnostics.csv"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    rows = load_rows(src, args.valid_percent, args.min_evidence_events)
    if not rows:
        raise SystemExit("no trainable prosody group rows found")
    write_outputs(rows, args.run_dir)
    print(f"Wrote prosody group duration model diagnostics for {len(rows)} rows to {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
