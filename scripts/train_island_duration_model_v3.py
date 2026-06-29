#!/usr/bin/env python3
"""Train a conservative, componentized island-duration model from LipLab artifacts.

This is research/tooling only. It does not change runtime behavior.

The model intentionally avoids phrase-specific hacks.  It learns systemic duration
mass from:

    speech material/prosody groups
  + non-terminal punctuation inside a sentence island
  + terminal cadence punctuation
  + explicit global calibration

The default objective is quantile loss (`--target-quantile 0.80`), because the
runtime consequence of underestimating a sentence island is usually worse than
slightly overestimating it.  This trains a conservative duration prior rather
than an unbiased mean predictor.

Inputs expected from a LipLab run:
  <run>/per_case/<case>/committed_events.csv
  <run>/per_case/<case>/island_launch_diagnostics.csv
  <run>/per_case/<case>/speech_islands.csv

Optional transcript inputs for punctuation features:
  --inputs <root> with <root>/transcripts/<case>.txt or case_000000.txt, etc.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_int(value: str | None) -> int | None:
    v = parse_float(value)
    return None if v is None else int(v)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(len(s) * p)) - 1))
    return s[idx]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "p90": math.nan, "p95": math.nan}
    return {
        "count": float(len(values)),
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def is_word_char(c: str) -> bool:
    return c.isalnum() or c in {"'", "-"}


def punctuation_by_word_index(text: str) -> dict[int, list[str]]:
    """Return punctuation marks that appear after each zero-based word index."""
    out: dict[int, list[str]] = {}
    word_index = 0
    inside_word = False
    i = 0
    while i < len(text):
        c = text[i]
        if is_word_char(c):
            inside_word = True
            i += 1
            continue
        if inside_word:
            word_index += 1
            inside_word = False
        if word_index > 0:
            mark: str | None = None
            if c in ",;:.!?":
                mark = c
            elif c in "—–":
                mark = "-"
            elif c == "." and text[i : i + 3] == "...":
                mark = "…"
                i += 2
            if mark:
                out.setdefault(word_index - 1, []).append(mark)
        i += 1
    return out


def load_transcript(inputs: Path | None, case_id: str) -> str:
    if inputs is None:
        return ""
    candidates = [
        inputs / "transcripts" / f"{case_id}.txt",
        inputs / "transcripts" / f"{case_id}.transcript.txt",
        inputs / "transcripts" / f"{case_id}.csv",
        inputs / f"{case_id}.txt",
    ]
    # Also tolerate case_000000 -> 000000 style names.
    if case_id.startswith("case_"):
        short = case_id.removeprefix("case_")
        candidates.extend([
            inputs / "transcripts" / f"{short}.txt",
            inputs / f"{short}.txt",
        ])
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8-sig", errors="replace").strip()
    return ""


PARAM_BOUNDS: list[tuple[str, float, float]] = [
    # Explicit calibration. These are not hidden runtime scalers; if used, they
    # must be logged and ported as named constants.
    ("GlobalScale", 0.85, 1.65),
    ("GlobalBiasSec", 0.000, 0.450),

    # Speech material / prosody groups.
    ("ConsonantTransitionGroupSec", 0.015, 0.120),
    ("WeakPickupGroupSec", 0.035, 0.180),
    ("ConnectiveGroupSec", 0.045, 0.210),
    ("StressBeatGroupSec", 0.080, 0.300),
    ("CadenceBearingGroupSec", 0.090, 0.420),
    ("UnknownRoleGroupSec", 0.050, 0.240),

    # Punctuation duration mass inside islands. Terminal punctuation is cadence
    # mass for the current island; it is not an extra launch island.
    ("CommaPauseSec", 0.000, 0.120),
    ("ListCommaExtraSec", 0.000, 0.120),
    ("SemicolonPauseSec", 0.080, 0.500),
    ("ColonPauseSec", 0.050, 0.420),
    ("DashPauseSec", 0.030, 0.360),
    ("EllipsisPauseSec", 0.080, 0.650),
    ("PeriodCadenceSec", 0.040, 0.340),
    ("QuestionCadenceSec", 0.040, 0.360),
    ("ExclamationCadenceSec", 0.040, 0.360),

    # Generic material fallback. Useful if group extraction undercounts dense text.
    ("UniqueWordSec", 0.000, 0.065),
    ("VisibleEventSec", 0.000, 0.030),
]

PARAM_NAMES = [name for name, _, _ in PARAM_BOUNDS]


@dataclass
class IslandRecord:
    case_id: str
    text_island_index: int
    audio_island_index: int
    target_duration_sec: float
    current_duration_sec: float
    features: dict[str, float]
    transcript_span_words: tuple[int, int] | None


def rows_by_int(rows: list[dict[str, str]], key: str) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        idx = parse_int(row.get(key))
        if idx is not None:
            out.setdefault(idx, []).append(row)
    return out


def first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        v = parse_float(row.get(key))
        if v is not None:
            return v
    return None


def collect_records(run_dir: Path, inputs: Path | None, include_fallback: bool) -> list[IslandRecord]:
    per_case = run_dir / "per_case"
    if not per_case.exists():
        raise SystemExit(f"Missing per-case directory: {per_case}")

    records: list[IslandRecord] = []
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        case_id = case_dir.name
        committed = read_csv_rows(case_dir / "committed_events.csv")
        launches = read_csv_rows(case_dir / "island_launch_diagnostics.csv")
        speech = read_csv_rows(case_dir / "speech_islands.csv")
        if not committed or not launches or not speech:
            continue

        speech_by_index: dict[int, dict[str, str]] = {}
        for row in speech:
            idx = parse_int(row.get("IslandIndex") or row.get("AudioIslandIndex"))
            if idx is not None:
                speech_by_index[idx] = row

        committed_by_text_island = rows_by_int(committed, "TextIslandIndex")
        punct = punctuation_by_word_index(load_transcript(inputs, case_id))

        for launch in launches:
            if not include_fallback and launch.get("LaunchReason", "") != "detected_audio_start":
                continue
            text_idx = parse_int(launch.get("TextIslandIndex"))
            audio_idx = parse_int(launch.get("AudioIslandIndex"))
            if text_idx is None or audio_idx is None or audio_idx < 0:
                continue
            speech_row = speech_by_index.get(audio_idx)
            if speech_row is None:
                continue
            audio_start = first_float(speech_row, ["AudioBufferStartSec", "AudioStartSec", "StartSec"])
            audio_end = first_float(speech_row, ["AudioBufferLastSpeechSec", "AudioBufferEndSec", "AudioEndSec", "EndSec"])
            if audio_start is None or audio_end is None:
                continue
            target = audio_end - audio_start
            if target <= 0.04 or target > 30.0:
                continue

            rows = committed_by_text_island.get(text_idx, [])
            if not rows:
                continue

            current = first_float(rows[0], ["PlannerIslandPredictedDurationSeconds", "IslandAudioSpanSeconds"])
            if current is None:
                # Fall back to committed span if older logs lack explicit planner duration.
                centers = [parse_float(r.get("FinalRenderCenterSeconds")) for r in rows]
                centers = [v for v in centers if v is not None]
                current = max(0.05, max(centers) - min(centers)) if centers else 0.0

            group_keys: set[tuple[int, str]] = set()
            unique_words: set[int] = set()
            event_count = 0
            min_word: int | None = None
            max_word: int | None = None
            for row in rows:
                event_count += 1
                group_index = parse_int(row.get("PlannerProsodyGroupIndex"))
                role = (row.get("PlannerProsodyRole") or row.get("ProsodyGroupRole") or "unknown").strip() or "unknown"
                if group_index is not None:
                    group_keys.add((group_index, role))
                word_idx = parse_int(row.get("WordIndex"))
                if word_idx is not None and word_idx >= 0:
                    unique_words.add(word_idx)
                    min_word = word_idx if min_word is None else min(min_word, word_idx)
                    max_word = word_idx if max_word is None else max(max_word, word_idx)

            features: dict[str, float] = {name: 0.0 for name in PARAM_NAMES}
            features["UniqueWordSec"] = float(len(unique_words))
            features["VisibleEventSec"] = float(event_count)
            for _, role in group_keys:
                normalized = role.lower()
                if normalized == "consonant_transition":
                    features["ConsonantTransitionGroupSec"] += 1.0
                elif normalized == "weak_pickup":
                    features["WeakPickupGroupSec"] += 1.0
                elif normalized == "connective":
                    features["ConnectiveGroupSec"] += 1.0
                elif normalized == "stress_beat":
                    features["StressBeatGroupSec"] += 1.0
                elif normalized == "cadence_bearing":
                    features["CadenceBearingGroupSec"] += 1.0
                else:
                    features["UnknownRoleGroupSec"] += 1.0

            # Punctuation belongs to the current island if it follows one of the
            # island's words.  Commas in a run are treated as ordinary comma plus
            # additional list-cadence mass from the second comma onward.
            comma_count = 0
            if unique_words:
                for word_idx in sorted(unique_words):
                    for mark in punct.get(word_idx, []):
                        if mark == ",":
                            comma_count += 1
                            features["CommaPauseSec"] += 1.0
                            if comma_count >= 2:
                                features["ListCommaExtraSec"] += 1.0
                        elif mark == ";":
                            features["SemicolonPauseSec"] += 1.0
                        elif mark == ":":
                            features["ColonPauseSec"] += 1.0
                        elif mark == "-":
                            features["DashPauseSec"] += 1.0
                        elif mark == "…":
                            features["EllipsisPauseSec"] += 1.0
                        elif mark == ".":
                            features["PeriodCadenceSec"] += 1.0
                        elif mark == "?":
                            features["QuestionCadenceSec"] += 1.0
                        elif mark == "!":
                            features["ExclamationCadenceSec"] += 1.0

            records.append(IslandRecord(
                case_id=case_id,
                text_island_index=text_idx,
                audio_island_index=audio_idx,
                target_duration_sec=target,
                current_duration_sec=current,
                features=features,
                transcript_span_words=(min_word, max_word) if min_word is not None and max_word is not None else None,
            ))
    return records


def raw_component_seconds(features: dict[str, float], params: dict[str, float]) -> float:
    total = 0.0
    for name in PARAM_NAMES:
        if name in {"GlobalScale", "GlobalBiasSec"}:
            continue
        total += features.get(name, 0.0) * params[name]
    return max(0.035, total)


def predict_seconds(features: dict[str, float], params: dict[str, float]) -> float:
    raw = raw_component_seconds(features, params)
    return max(0.05, raw * params["GlobalScale"] + params["GlobalBiasSec"])


def quantile_loss(pred: float, target: float, q: float) -> float:
    if pred < target:
        return q * (target - pred)
    return (1.0 - q) * (pred - target)


def objective(records: list[IslandRecord], params: dict[str, float], q: float) -> float:
    if not records:
        return math.inf
    return mean(quantile_loss(predict_seconds(r.features, params), r.target_duration_sec, q) for r in records)


def midpoint_params() -> dict[str, float]:
    return {name: (lo + hi) * 0.5 for name, lo, hi in PARAM_BOUNDS}


def random_params(rng: random.Random) -> dict[str, float]:
    return {name: rng.uniform(lo, hi) for name, lo, hi in PARAM_BOUNDS}


def clamp_param(name: str, value: float) -> float:
    for bound_name, lo, hi in PARAM_BOUNDS:
        if name == bound_name:
            return min(hi, max(lo, value))
    return value


def refine(records: list[IslandRecord], params: dict[str, float], q: float, passes: int) -> tuple[dict[str, float], float]:
    best = dict(params)
    best_score = objective(records, best, q)
    steps = {name: (hi - lo) * 0.20 for name, lo, hi in PARAM_BOUNDS}
    for _ in range(passes):
        improved = False
        for name, _, _ in PARAM_BOUNDS:
            step = steps[name]
            if step < 0.00025:
                continue
            for direction in (-1.0, 1.0):
                cand = dict(best)
                cand[name] = clamp_param(name, cand[name] + direction * step)
                score = objective(records, cand, q)
                if score + 1e-12 < best_score:
                    best = cand
                    best_score = score
                    improved = True
        if not improved:
            for key in steps:
                steps[key] *= 0.5
    return best, best_score


def fit(records: list[IslandRecord], iterations: int, seed: int, q: float, trace_path: Path | None, verbose: bool, progress_every: int) -> tuple[dict[str, float], list[dict[str, str]]]:
    rng = random.Random(seed)
    candidates = [midpoint_params()]
    candidates.extend(random_params(rng) for _ in range(max(0, iterations)))
    if trace_path and trace_path.exists():
        trace_path.unlink()
    trace: list[dict[str, str]] = []
    best: dict[str, float] | None = None
    best_score = math.inf
    fields = ["iteration", "stage", "candidate_loss_ms", "best_loss_ms", "improved", *PARAM_NAMES]

    def append(row: dict[str, str]) -> None:
        trace.append(row)
        if trace_path:
            exists = trace_path.exists()
            with trace_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    w.writeheader()
                w.writerow(row)

    for i, params in enumerate(candidates):
        params, score = refine(records, params, q, passes=4 if i else 8)
        improved = score < best_score
        if improved:
            best = params
            best_score = score
        row = {
            "iteration": str(i),
            "stage": "candidate_refined",
            "candidate_loss_ms": f"{score * 1000.0:.9g}",
            "best_loss_ms": f"{best_score * 1000.0:.9g}",
            "improved": "1" if improved else "0",
            **{name: f"{params[name]:.9g}" for name in PARAM_NAMES},
        }
        append(row)
        if verbose and (i == 0 or improved or i == len(candidates) - 1 or i % max(1, progress_every) == 0):
            print(f"[v3] {i + 1}/{len(candidates)} candidate_loss={score*1000:.2f}ms best={best_score*1000:.2f}ms" + (" improved" if improved else ""), flush=True)

    assert best is not None
    best, best_score = refine(records, best, q, passes=14)
    append({
        "iteration": "final_refine",
        "stage": "final_refine",
        "candidate_loss_ms": f"{best_score * 1000.0:.9g}",
        "best_loss_ms": f"{best_score * 1000.0:.9g}",
        "improved": "1",
        **{name: f"{best[name]:.9g}" for name in PARAM_NAMES},
    })
    return best, trace


def evaluate(records: list[IslandRecord], params: dict[str, float]) -> dict[str, float]:
    current_abs = []
    model_abs = []
    current_ratios = []
    model_ratios = []
    for r in records:
        pred = predict_seconds(r.features, params)
        current_abs.append(abs(r.current_duration_sec - r.target_duration_sec) * 1000.0)
        model_abs.append(abs(pred - r.target_duration_sec) * 1000.0)
        if r.target_duration_sec > 0:
            current_ratios.append(r.current_duration_sec / r.target_duration_sec)
            model_ratios.append(pred / r.target_duration_sec)
    cur = stats(current_abs)
    mod = stats(model_abs)
    return {
        "record_count": float(len(records)),
        "current_mae_ms": cur["mean"],
        "current_median_ae_ms": cur["median"],
        "current_p90_ae_ms": cur["p90"],
        "model_mae_ms": mod["mean"],
        "model_median_ae_ms": mod["median"],
        "model_p90_ae_ms": mod["p90"],
        "current_ratio_mean": mean(current_ratios) if current_ratios else math.nan,
        "current_ratio_median": median(current_ratios) if current_ratios else math.nan,
        "model_ratio_mean": mean(model_ratios) if model_ratios else math.nan,
        "model_ratio_median": median(model_ratios) if model_ratios else math.nan,
        "current_under_60_count": float(sum(1 for v in current_ratios if v < 0.60)),
        "current_under_75_count": float(sum(1 for v in current_ratios if v < 0.75)),
        "current_under_90_count": float(sum(1 for v in current_ratios if v < 0.90)),
        "model_under_60_count": float(sum(1 for v in model_ratios if v < 0.60)),
        "model_under_75_count": float(sum(1 for v in model_ratios if v < 0.75)),
        "model_under_90_count": float(sum(1 for v in model_ratios if v < 0.90)),
        "model_over_125_count": float(sum(1 for v in model_ratios if v > 1.25)),
        "model_over_150_count": float(sum(1 for v in model_ratios if v > 1.50)),
    }


def cross_validate(records: list[IslandRecord], folds: int, iterations: int, seed: int, q: float) -> dict[str, float]:
    if len(records) < folds:
        return {}
    model_errors = []
    current_errors = []
    under60 = under75 = under90 = over125 = 0
    fold_iterations = max(20, iterations // max(1, folds))
    for fold in range(folds):
        train = [r for i, r in enumerate(records) if i % folds != fold]
        test = [r for i, r in enumerate(records) if i % folds == fold]
        params, _ = fit(train, fold_iterations, seed + fold * 7919, q, None, False, 25)
        for r in test:
            pred = predict_seconds(r.features, params)
            model_errors.append(abs(pred - r.target_duration_sec) * 1000.0)
            current_errors.append(abs(r.current_duration_sec - r.target_duration_sec) * 1000.0)
            ratio = pred / r.target_duration_sec if r.target_duration_sec > 0 else 1.0
            under60 += int(ratio < 0.60)
            under75 += int(ratio < 0.75)
            under90 += int(ratio < 0.90)
            over125 += int(ratio > 1.25)
    mod = stats(model_errors)
    cur = stats(current_errors)
    return {
        "cv_current_mae_ms": cur["mean"],
        "cv_current_median_ae_ms": cur["median"],
        "cv_current_p90_ae_ms": cur["p90"],
        "cv_model_mae_ms": mod["mean"],
        "cv_model_median_ae_ms": mod["median"],
        "cv_model_p90_ae_ms": mod["p90"],
        "cv_model_under_60_count": float(under60),
        "cv_model_under_75_count": float(under75),
        "cv_model_under_90_count": float(under90),
        "cv_model_over_125_count": float(over125),
    }


def write_predictions(records: list[IslandRecord], params: dict[str, float], path: Path) -> None:
    rows = []
    for r in records:
        raw = raw_component_seconds(r.features, params)
        pred = predict_seconds(r.features, params)
        rows.append({
            "case_id": r.case_id,
            "text_island_index": r.text_island_index,
            "audio_island_index": r.audio_island_index,
            "target_duration_sec": f"{r.target_duration_sec:.9g}",
            "current_duration_sec": f"{r.current_duration_sec:.9g}",
            "model_raw_component_sec": f"{raw:.9g}",
            "model_duration_sec": f"{pred:.9g}",
            "current_ratio": f"{r.current_duration_sec / r.target_duration_sec:.9g}",
            "model_ratio": f"{pred / r.target_duration_sec:.9g}",
            "current_abs_error_ms": f"{abs(r.current_duration_sec - r.target_duration_sec) * 1000.0:.9g}",
            "model_abs_error_ms": f"{abs(pred - r.target_duration_sec) * 1000.0:.9g}",
            **{name: f"{r.features.get(name, 0.0):.9g}" for name in PARAM_NAMES if name not in {"GlobalScale", "GlobalBiasSec"}},
        })
    rows.sort(key=lambda row: float(row["model_abs_error_ms"]), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["case_id"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_correlations(records: list[IslandRecord], path: Path) -> None:
    ys = [r.target_duration_sec for r in records]
    if not ys:
        return
    y_mean = mean(ys)
    y_var = sum((v - y_mean) ** 2 for v in ys)
    rows = []
    for name in PARAM_NAMES:
        if name in {"GlobalScale", "GlobalBiasSec"}:
            continue
        xs = [r.features.get(name, 0.0) for r in records]
        x_mean = mean(xs)
        x_var = sum((v - x_mean) ** 2 for v in xs)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        corr = cov / math.sqrt(x_var * y_var) if x_var > 1e-12 and y_var > 1e-12 else 0.0
        rows.append({"feature": name, "mean": f"{x_mean:.9g}", "target_corr": f"{corr:.9g}"})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "mean", "target_corr"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory, e.g. experiments/runs/local_run")
    ap.add_argument("--inputs", default=None, help="Optional inputs root containing transcripts/")
    ap.add_argument("--out", default=None, help="Output directory; defaults to run directory")
    ap.add_argument("--target-quantile", type=float, default=0.80, help="Quantile target for conservative duration training. Use 0.75-0.85 for underprediction-averse fitting.")
    ap.add_argument("--search-iterations", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--include-fallback", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    q = min(0.95, max(0.50, args.target_quantile))
    run_dir = Path(args.run)
    inputs = Path(args.inputs) if args.inputs else None
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(run_dir, inputs, include_fallback=args.include_fallback)
    if not records:
        raise SystemExit("No usable island records found. Run LipLab with committed_events.csv, island_launch_diagnostics.csv, and speech_islands.csv enabled.")

    trace_path = out_dir / "island_duration_model_v3_search_trace.csv"
    params, _ = fit(records, max(0, args.search_iterations), args.seed, q, trace_path, args.verbose, args.progress_every)
    train_stats = evaluate(records, params)
    cv_stats = {} if args.no_cv else cross_validate(records, max(2, args.folds), max(0, args.search_iterations), args.seed, q)

    payload = {
        "record_count": len(records),
        "target": "speech_islands observed duration",
        "objective": "quantile_loss",
        "target_quantile": q,
        "excluded_fallback_unmapped_islands": not args.include_fallback,
        "bounds_seconds": {name: [lo, hi] for name, lo, hi in PARAM_BOUNDS},
        "constants_seconds": params,
        "train_stats": train_stats,
        "cross_validation_stats": cv_stats,
    }
    (out_dir / "island_duration_model_v3_constants.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_predictions(records, params, out_dir / "island_duration_model_v3_predictions.csv")
    write_correlations(records, out_dir / "island_duration_model_v3_feature_correlations.csv")

    lines = [
        "LipLab island duration model v3 report",
        "======================================",
        "",
        f"records: {len(records)}",
        f"target quantile: {q:.3f}",
        f"fallback/unmapped included: {args.include_fallback}",
        "",
        "Current runtime duration error",
        "------------------------------",
        f"train MAE ms: {train_stats['current_mae_ms']:.3f}",
        f"train median AE ms: {train_stats['current_median_ae_ms']:.3f}",
        f"train p90 AE ms: {train_stats['current_p90_ae_ms']:.3f}",
        f"train ratio mean: {train_stats['current_ratio_mean']:.3f}",
        f"train ratio median: {train_stats['current_ratio_median']:.3f}",
        f"train under 60% count: {train_stats['current_under_60_count']:.0f}",
        f"train under 75% count: {train_stats['current_under_75_count']:.0f}",
        f"train under 90% count: {train_stats['current_under_90_count']:.0f}",
        "",
        "Conservative componentized model",
        "--------------------------------",
        f"train MAE ms: {train_stats['model_mae_ms']:.3f}",
        f"train median AE ms: {train_stats['model_median_ae_ms']:.3f}",
        f"train p90 AE ms: {train_stats['model_p90_ae_ms']:.3f}",
        f"train ratio mean: {train_stats['model_ratio_mean']:.3f}",
        f"train ratio median: {train_stats['model_ratio_median']:.3f}",
        f"train under 60% count: {train_stats['model_under_60_count']:.0f}",
        f"train under 75% count: {train_stats['model_under_75_count']:.0f}",
        f"train under 90% count: {train_stats['model_under_90_count']:.0f}",
        f"train over 125% count: {train_stats['model_over_125_count']:.0f}",
        f"train over 150% count: {train_stats['model_over_150_count']:.0f}",
        f"cv MAE ms: {cv_stats.get('cv_model_mae_ms', math.nan):.3f}",
        f"cv median AE ms: {cv_stats.get('cv_model_median_ae_ms', math.nan):.3f}",
        f"cv p90 AE ms: {cv_stats.get('cv_model_p90_ae_ms', math.nan):.3f}",
        f"cv under 75% count: {cv_stats.get('cv_model_under_75_count', math.nan):.0f}",
        f"cv over 125% count: {cv_stats.get('cv_model_over_125_count', math.nan):.0f}",
        "",
        "Constants seconds",
        "-----------------",
    ]
    for name in PARAM_NAMES:
        lines.append(f"{name}: {params[name]:.9g}")
    (out_dir / "island_duration_model_v3_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cpp_lines = [
        "// Candidate constants from scripts/train_island_duration_model_v3.py.",
        "// Conservative quantile-trained island duration model; diagnostic until ported.",
        f"// target quantile: {q:.3f}",
    ]
    for name in PARAM_NAMES:
        cpp_name = "k" + name
        cpp_lines.append(f"static constexpr float {cpp_name} = {params[name]:.9g}f;")
    (out_dir / "island_duration_model_v3_cpp_constants.txt").write_text("\n".join(cpp_lines) + "\n", encoding="utf-8")

    print(f"records: {len(records)}")
    print(f"target quantile: {q:.3f}")
    print(f"current MAE ms: {train_stats['current_mae_ms']:.3f}")
    print(f"model train MAE ms: {train_stats['model_mae_ms']:.3f}")
    print(f"model CV MAE ms: {cv_stats.get('cv_model_mae_ms', math.nan):.3f}")
    print(f"model under 75% count: {train_stats['model_under_75_count']:.0f}")
    print(f"model over 125% count: {train_stats['model_over_125_count']:.0f}")
    print(f"Wrote: {out_dir / 'island_duration_model_v3_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
