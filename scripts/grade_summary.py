import csv
import json
import pathlib
import statistics
from collections import Counter
from typing import Any


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = sorted(float(v) for v in values if v is not None)
    return statistics.median(values) if values else 0.0


def _p90(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return 0.0
    return values[int(0.9 * (len(values) - 1))]


def _max(values):
    values = [float(v) for v in values if v is not None]
    return max(values) if values else 0.0


def _stddev(values):
    values = [float(v) for v in values if v is not None]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _stats(prefix: str, values):
    vals = [float(v) for v in values if v is not None]
    return {
        f"{prefix}_mean_ms": _mean(vals),
        f"{prefix}_median_ms": _median(vals),
        f"{prefix}_p90_ms": _p90(vals),
        f"{prefix}_max_ms": _max(vals),
        f"{prefix}_stddev_ms": _stddev(vals),
    }


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def load_case_grades(root: pathlib.Path, gold_root: pathlib.Path | None = None):
    rows = []
    graded = []
    ungraded = []
    if not root.exists():
        return rows, graded, ungraded
    for grade_path in sorted(root.glob("*/grade.json")):
        grade = read_json(grade_path)
        case_dir = grade_path.parent
        direct = read_json(case_dir / "direct_aligner_grade.json")
        runtime_phone_rows = read_csv_rows(case_dir / "runtime_phone_alignment.csv")
        progress_rows = read_csv_rows(case_dir / "audio_progress_measurements.csv")
        committed_rows = read_csv_rows(case_dir / "committed.csv")
        region_drop_rows = read_csv_rows(case_dir / "region_drop_diagnostics.csv")
        predicted_speech_rows = read_csv_rows(case_dir / "speech_regions.csv")
        gap_candidate_rows = read_csv_rows(case_dir / "gap_candidates.csv")
        landmark_audit = read_json(case_dir / "landmark_audit.json")
        conditioned_landmark_audit = read_json(case_dir / "conditioned_landmark_audit.json")
        gold_speech_rows = read_csv_rows(gold_root / case_dir.name / "speech.csv") if gold_root else []
        gold_viseme_rows = _gold_viseme_rows_for_case(gold_root, case_dir.name)
        row = {
            "case": case_dir.name,
            "grade": grade,
            "direct_aligner": direct,
            "runtime_phone_rows": runtime_phone_rows,
            "audio_progress_rows": progress_rows,
            "committed_rows": committed_rows,
            "region_drop_rows": region_drop_rows,
            "predicted_speech_rows": predicted_speech_rows,
            "gap_candidate_rows": gap_candidate_rows,
            "landmark_audit": landmark_audit,
            "conditioned_landmark_audit": conditioned_landmark_audit,
            "gold_speech_rows": gold_speech_rows,
            "gold_viseme_rows": gold_viseme_rows,
        }
        row["audio_progress_summary"] = summarize_audio_progress(row)
        row["speech_region_diagnostics"] = summarize_speech_regions(row)
        row["gap_candidate_summary"] = summarize_gap_candidates(row)
        row["phone_class_summary"] = summarize_phone_classes(row)
        rows.append(row)
        if grade.get("gold_available", True):
            graded.append(row)
        else:
            ungraded.append(row)
    return rows, graded, ungraded


def _metric(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(row.get("grade", {}).get(name, default))
    except Exception:
        return default


def _nested(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = row.get("grade", {})
    try:
        for key in keys:
            cur = cur.get(key, {})
        return float(cur)
    except Exception:
        return default


def _pause_metric(row: dict[str, Any], prefix: str, field: str, default: float = 0.0) -> float:
    pause = row.get("grade", {}).get("pause_alignment", {})
    prefix_aliases = {
        "speech_region": ["speech_region"],
        "text_sentence_span": ["text_sentence_span", "sentence_region"],
        "text_sentence_region_subspan": ["text_sentence_region_subspan", "clause_region"],
        "sentence_region": ["sentence_region", "text_sentence_span"],
        "clause_region": ["clause_region", "text_sentence_region_subspan"],
    }
    field_templates = {
        "reference_count": "{prefix}_reference_count",
        "predicted_count": "{prefix}_predicted_count",
        "matched_count": "{prefix}_matched_count",
        "missing_count": "{prefix}_missing_count",
        "extra_count": "{prefix}_extra_count",
        "start_error_ms": "mean_abs_{prefix}_start_error_ms",
        "end_error_ms": "mean_abs_{prefix}_end_error_ms",
        "tail_out_ms": "mean_{prefix}_tail_out_ms",
        "lead_in_ms": "mean_{prefix}_lead_in_ms",
    }
    template = field_templates.get(field, "{prefix}_" + field)
    candidate_keys = [
        template.format(prefix=alias)
        for alias in prefix_aliases.get(prefix, [prefix])
    ]
    try:
        for key in candidate_keys:
            if key in pause:
                return float(pause.get(key, default))
        return float(default)
    except Exception:
        return default


def _pause_flag(row: dict[str, Any], prefix: str, field: str, default: bool = False) -> bool:
    pause = row.get("grade", {}).get("pause_alignment", {})
    prefix_aliases = {
        "speech_region": ["speech_region"],
        "text_sentence_span": ["text_sentence_span", "sentence_region"],
        "text_sentence_region_subspan": ["text_sentence_region_subspan", "clause_region"],
        "sentence_region": ["sentence_region", "text_sentence_span"],
        "clause_region": ["clause_region", "text_sentence_region_subspan"],
    }
    value = default
    for alias in prefix_aliases.get(prefix, [prefix]):
        key = f"{alias}_{field}"
        if key in pause:
            value = pause.get(key, default)
            break
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _direct(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(row.get("direct_aligner", {}).get(name, default))
    except Exception:
        return default



def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _find_gold_speech_progress(gold_speech_rows: list[dict[str, str]], t: float):
    best = None
    best_dist = float("inf")
    for r in gold_speech_rows:
        start = _safe_float(r, "start")
        end = _safe_float(r, "end")
        if end <= start:
            continue
        if start <= t <= end:
            return ((t - start) / (end - start), end - start)
        dist = min(abs(t - start), abs(t - end))
        if dist < best_dist:
            best_dist = dist
            best = (start, end)
    if best and best[1] > best[0] and best_dist <= 0.200:
        start, end = best
        return (max(0.0, min(1.0, (t - start) / (end - start))), end - start)
    return None


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((vx * vy) ** 0.5)


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _region_rows_to_spans(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for row in rows:
        start = _safe_float(row, "start")
        end = _safe_float(row, "end")
        if end > start:
            spans.append((start, end))
    spans.sort()
    return spans


def _pair_regions_by_index(gold: list[tuple[float, float]], predicted: list[tuple[float, float]]) -> dict[str, Any]:
    matched = min(len(gold), len(predicted))
    if matched <= 0:
        return {
            "matched_count": 0,
            "mean_start_error_ms": 0.0,
            "mean_end_error_ms": 0.0,
            "mean_overlap_ratio": 0.0,
        }

    start_errors = []
    end_errors = []
    overlap_ratios = []
    for i in range(matched):
        gold_start, gold_end = gold[i]
        pred_start, pred_end = predicted[i]
        start_errors.append(abs(pred_start - gold_start) * 1000.0)
        end_errors.append(abs(pred_end - gold_end) * 1000.0)
        overlap = _overlap_seconds(gold_start, gold_end, pred_start, pred_end)
        union = max(gold_end, pred_end) - min(gold_start, pred_start)
        overlap_ratios.append(overlap / union if union > 1e-9 else 0.0)
    return {
        "matched_count": matched,
        "mean_start_error_ms": _mean(start_errors),
        "mean_end_error_ms": _mean(end_errors),
        "mean_overlap_ratio": _mean(overlap_ratios),
    }


def _pair_regions_by_overlap_or_nearest(gold: list[tuple[float, float]], predicted: list[tuple[float, float]]) -> dict[str, Any]:
    pairs: list[tuple[float, int, int]] = []
    for pi, (pred_start, pred_end) in enumerate(predicted):
        for gi, (gold_start, gold_end) in enumerate(gold):
            overlap = _overlap_seconds(pred_start, pred_end, gold_start, gold_end)
            if overlap > 0.0:
                pairs.append((overlap, pi, gi))
    pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    matches: list[tuple[int, int, float, str]] = []
    for overlap, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gold:
            continue
        matched_pred.add(pi)
        matched_gold.add(gi)
        matches.append((pi, gi, overlap, "overlap"))

    unmatched_pred = [pi for pi in range(len(predicted)) if pi not in matched_pred]
    unmatched_gold = [gi for gi in range(len(gold)) if gi not in matched_gold]
    for pi in unmatched_pred:
        if not unmatched_gold:
            break
        pred_start, pred_end = predicted[pi]
        pred_center = 0.5 * (pred_start + pred_end)
        best_idx = -1
        best_dist = float("inf")
        for idx, gi in enumerate(unmatched_gold):
            gold_start, gold_end = gold[gi]
            gold_center = 0.5 * (gold_start + gold_end)
            dist = abs(pred_center - gold_center)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            gi = unmatched_gold.pop(best_idx)
            matched_pred.add(pi)
            matched_gold.add(gi)
            matches.append((pi, gi, 0.0, "nearest"))

    if not matches:
        return {
            "matched_count": 0,
            "overlap_matched_count": 0,
            "nearest_matched_count": 0,
            "mean_start_error_ms": 0.0,
            "mean_end_error_ms": 0.0,
            "mean_overlap_ratio": 0.0,
            "predicted_only_count": len(predicted),
            "gold_only_count": len(gold),
        }

    start_errors = []
    end_errors = []
    overlap_ratios = []
    overlap_matched_count = 0
    nearest_matched_count = 0
    for pi, gi, overlap, mode in matches:
        pred_start, pred_end = predicted[pi]
        gold_start, gold_end = gold[gi]
        start_errors.append(abs(pred_start - gold_start) * 1000.0)
        end_errors.append(abs(pred_end - gold_end) * 1000.0)
        union = max(gold_end, pred_end) - min(gold_start, pred_start)
        overlap_ratios.append(overlap / union if union > 1e-9 else 0.0)
        if mode == "overlap":
            overlap_matched_count += 1
        else:
            nearest_matched_count += 1
    return {
        "matched_count": len(matches),
        "overlap_matched_count": overlap_matched_count,
        "nearest_matched_count": nearest_matched_count,
        "mean_start_error_ms": _mean(start_errors),
        "mean_end_error_ms": _mean(end_errors),
        "mean_overlap_ratio": _mean(overlap_ratios),
        "predicted_only_count": max(0, len(predicted) - len(matches)),
        "gold_only_count": max(0, len(gold) - len(matches)),
    }


def summarize_speech_regions(row: dict[str, Any]) -> dict[str, Any]:
    gold = _region_rows_to_spans(row.get("gold_speech_rows", []))
    predicted = _region_rows_to_spans(row.get("predicted_speech_rows", []))
    indexed = _pair_regions_by_index(gold, predicted)
    overlap = _pair_regions_by_overlap_or_nearest(gold, predicted)
    return {
        "gold_count": len(gold),
        "predicted_count": len(predicted),
        "count_delta": len(predicted) - len(gold),
        "indexed": indexed,
        "overlap": overlap,
    }


def summarize_gap_candidates(row: dict[str, Any]) -> dict[str, Any]:
    gap_rows = row.get("gap_candidate_rows", [])
    if not gap_rows:
        return {}

    decision_counts: Counter[str] = Counter()
    bridged = 0
    split = 0
    ambiguous = 0
    ambiguous_bridged = 0
    ambiguous_split = 0
    durations_ms: list[float] = []
    ambiguous_durations_ms: list[float] = []
    for r in gap_rows:
        decision = (r.get("decision_class") or "").strip()
        if decision:
            decision_counts[decision] += 1
        is_bridged = _safe_int(r, "bridged", 0) == 1
        bridged += 1 if is_bridged else 0
        split += 0 if is_bridged else 1
        duration_ms = _safe_float(r, "gap_duration") * 1000.0
        durations_ms.append(duration_ms)
        if 90.0 <= duration_ms <= 350.0:
            ambiguous += 1
            ambiguous_durations_ms.append(duration_ms)
            if is_bridged:
                ambiguous_bridged += 1
            else:
                ambiguous_split += 1

    return {
        "count": len(gap_rows),
        "bridged": bridged,
        "split": split,
        "mean_gap_ms": _mean(durations_ms),
        "ambiguous_count": ambiguous,
        "ambiguous_bridged": ambiguous_bridged,
        "ambiguous_split": ambiguous_split,
        "ambiguous_mean_gap_ms": _mean(ambiguous_durations_ms),
        "decision_counts": dict(sorted(decision_counts.items())),
    }


def summarize_audio_progress(row: dict[str, Any]) -> dict[str, Any]:
    gold_speech = row.get("gold_speech_rows", [])
    progress_rows = row.get("audio_progress_rows", [])
    errors01 = []
    errors_ms = []
    signed_errors01 = []
    ests = []
    mfas = []
    by_component: dict[str, list[float]] = {
        "time_prior": [],
        "audio_density": [],
        "boundary": [],
        "phone_expectation": [],
    }
    playrates = []
    confidences = []
    anchor_phone = []
    anchor_boundary = []
    anchor_speech = []
    micro_pause_50 = []
    micro_pause_75 = []
    micro_pause_120 = []
    retrospective_drift_abs01 = []
    retrospective_drift_abs_ms = []
    retrospective_drift_signed01 = []
    retrospective_drift_confidence = []
    drift_playrates = []
    filtered_pll_rates = []
    filtered_pll_conf = []
    pll_tempo_rates = []
    pll_tempo_conf = []
    pll_phase_rates = []
    timing_warp_rates = []
    timing_warp_conf = []
    timing_warp_error_ms = []
    would_advance_rows = 0
    advance_reasons: Counter[str] = Counter()
    for r in progress_rows:
        if _safe_int(r, "would_advance", 0) == 1:
            would_advance_rows += 1
        reason = (r.get("advance_reason") or "").strip()
        if reason:
            advance_reasons[reason] += 1
        t = _safe_float(r, "current_playback", _safe_float(r, "CurrentPlaybackSec"))
        mfa = _find_gold_speech_progress(gold_speech, t)
        if not mfa:
            continue
        mfa_progress, region_duration = mfa
        est = _safe_float(r, "estimated_region_progress_01")
        err = est - mfa_progress
        errors01.append(abs(err))
        errors_ms.append(abs(err) * region_duration * 1000.0)
        signed_errors01.append(err)
        ests.append(est)
        mfas.append(mfa_progress)
        components = {
            "time_prior": _safe_float(r, "time_prior_progress_01"),
            "audio_density": _safe_float(r, "audio_density_progress_01"),
            "boundary": _safe_float(r, "boundary_evidence_progress_01"),
            "phone_expectation": _safe_float(r, "phone_expectation_progress_01"),
        }
        for k, v in components.items():
            by_component[k].append(abs(v - mfa_progress))
        playrates.append(_safe_float(r, "suggested_play_rate", 1.0))
        confidences.append(_safe_float(r, "progress_confidence"))
        anchor_phone.append(_safe_float(r, "anchor_mean_phone_probability", 0.0))
        anchor_boundary.append(_safe_float(r, "anchor_mean_boundary_probability", 0.0))
        anchor_speech.append(_safe_float(r, "anchor_mean_speech_probability", 0.0))
        micro_pause_50.append(_safe_float(r, "micro_pause_count_50ms", 0.0))
        micro_pause_75.append(_safe_float(r, "micro_pause_count_75ms", 0.0))
        micro_pause_120.append(_safe_float(r, "micro_pause_count_120ms", 0.0))
        drift01 = _safe_float(r, "retrospective_drift_01", 0.0)
        retrospective_drift_abs01.append(abs(drift01))
        retrospective_drift_abs_ms.append(abs(_safe_float(r, "retrospective_drift_sec", 0.0)) * 1000.0)
        retrospective_drift_signed01.append(drift01)
        retrospective_drift_confidence.append(_safe_float(r, "retrospective_drift_confidence", 0.0))
        drift_playrates.append(_safe_float(r, "drift_suggested_play_rate", 1.0))
        filtered_pll_rates.append(_safe_float(r, "filtered_pll_play_rate", 1.0))
        filtered_pll_conf.append(_safe_float(r, "filtered_pll_confidence", 0.0))
        pll_tempo_rates.append(_safe_float(r, "pll_tempo_rate", 1.0))
        pll_tempo_conf.append(_safe_float(r, "pll_tempo_confidence", 0.0))
        pll_phase_rates.append(_safe_float(r, "pll_phase_rate", 1.0))
        timing_warp_rates.append(_safe_float(r, "timing_warp_rate", 1.0))
        timing_warp_conf.append(_safe_float(r, "timing_warp_confidence", 0.0))
        timing_warp_error_ms.append(abs(_safe_float(r, "timing_warp_error_sec", 0.0)) * 1000.0)
    return {
        "rows": len(errors01),
        "mae01": _mean(errors01),
        "median01": _median(errors01),
        "p90_ms": sorted(errors_ms)[int(0.9 * (len(errors_ms) - 1))] if errors_ms else 0.0,
        "mae_ms": _mean(errors_ms),
        "bias01": _mean(signed_errors01),
        "corr": _corr(ests, mfas),
        "time_prior_mae01": _mean(by_component["time_prior"]),
        "audio_density_mae01": _mean(by_component["audio_density"]),
        "boundary_mae01": _mean(by_component["boundary"]),
        "phone_expectation_mae01": _mean(by_component["phone_expectation"]),
        "mean_playrate": _mean(playrates),
        "mean_confidence": _mean(confidences),
        "anchor_mean_phone_probability": _mean(anchor_phone),
        "anchor_mean_boundary_probability": _mean(anchor_boundary),
        "anchor_mean_speech_probability": _mean(anchor_speech),
        "micro_pause_50_count": _mean(micro_pause_50),
        "micro_pause_75_count": _mean(micro_pause_75),
        "micro_pause_120_count": _mean(micro_pause_120),
        "retrospective_drift_abs01": _mean(retrospective_drift_abs01),
        "retrospective_drift_abs_ms": _mean(retrospective_drift_abs_ms),
        "retrospective_drift_bias01": _mean(retrospective_drift_signed01),
        "retrospective_drift_confidence": _mean(retrospective_drift_confidence),
        "drift_mean_playrate": _mean(drift_playrates),
        "filtered_pll_playrate": _mean(filtered_pll_rates),
        "filtered_pll_confidence": _mean(filtered_pll_conf),
        "pll_tempo_playrate": _mean(pll_tempo_rates),
        "pll_tempo_confidence": _mean(pll_tempo_conf),
        "pll_phase_playrate": _mean(pll_phase_rates),
        "timing_warp_rate": _mean(timing_warp_rates),
        "timing_warp_confidence": _mean(timing_warp_conf),
        "timing_warp_error_ms": _mean(timing_warp_error_ms),
        "would_advance_rows": would_advance_rows,
        "advance_reasons": dict(sorted(advance_reasons.items())),
    }


def _gold_viseme_rows_for_case(gold_root: pathlib.Path | None, case_name: str) -> list[dict[str, str]]:
    if not gold_root:
        return []
    return read_csv_rows(gold_root / case_name / "visemes.csv")


def summarize_phone_classes(row: dict[str, Any]) -> dict[str, Any]:
    committed = row.get("committed_rows", [])
    gold = row.get("gold_viseme_rows", [])
    by_class: dict[str, dict[str, Any]] = {}
    used_gold: set[int] = set()
    for c in committed:
        klass = c.get("source_phone_class") or c.get("pose") or "Unknown"
        c_center = _safe_float(c, "center", _safe_float(c, "render_center"))
        c_start = _safe_float(c, "start", _safe_float(c, "render_start"))
        c_end = _safe_float(c, "end", _safe_float(c, "render_end"))
        c_pose = c.get("pose", "")
        c_word_index = _safe_int(c, "word_index", -9999)
        best_i = -1
        best_d = float("inf")
        for i, g in enumerate(gold):
            if i in used_gold:
                continue
            if c_pose and g.get("pose", "") and c_pose != g.get("pose", ""):
                continue
            if c_word_index != -9999 and "word_index" in g and _safe_int(g, "word_index", -9998) != c_word_index:
                continue
            g_center = (_safe_float(g, "start") + _safe_float(g, "end")) * 0.5
            d = abs(c_center - g_center)
            if d < best_d:
                best_d = d
                best_i = i
        bucket = by_class.setdefault(klass, {"n": 0, "matched": 0, "center": [], "start": [], "end": [], "unmapped": 0})
        bucket["n"] += 1
        if str(c.get("mapped_to_observed_speech", "1")) != "1":
            bucket["unmapped"] += 1
        if best_i >= 0 and best_d <= 0.250:
            used_gold.add(best_i)
            g = gold[best_i]
            bucket["matched"] += 1
            bucket["center"].append(best_d * 1000.0)
            bucket["start"].append(abs(c_start - _safe_float(g, "start")) * 1000.0)
            bucket["end"].append(abs(c_end - _safe_float(g, "end")) * 1000.0)
    out = {}
    for klass, b in by_class.items():
        out[klass] = {
            "n": b["n"],
            "matched": b["matched"],
            "unmapped": b["unmapped"],
            "center_ms": _mean(b["center"]),
            "start_ms": _mean(b["start"]),
            "end_ms": _mean(b["end"]),
        }
    return out

def compute_summary(rows, graded, ungraded):
    graded_cases = len(graded)
    qualified = [row for row in graded if not _pause_flag(row, "speech_region", "count_mismatch")]
    qualified_cases = len(qualified)
    phoneme_coverage = []
    runtime_phone_available = any(row.get("runtime_phone_rows") for row in rows)
    direct_aligner_available = any(bool(row.get("direct_aligner")) for row in rows)
    audio_progress_available = any(row.get("audio_progress_rows") for row in rows)
    landmark_audit_available = any(row.get("landmark_audit", {}).get("available") for row in rows)
    conditioned_landmark_audit_available = any(row.get("conditioned_landmark_audit", {}).get("available") for row in rows)
    for row in qualified:
        phone_rows = row.get("runtime_phone_rows", [])
        if phone_rows:
            mapped = sum(1 for r in phone_rows if str(r.get("mapped_to_observed_speech", "0")) == "1")
            phoneme_coverage.append(mapped / max(len(phone_rows), 1))
        else:
            reference_count = _metric(row, "reference_count")
            matched_count = _metric(row, "matched_count")
            if reference_count > 0.0:
                phoneme_coverage.append(matched_count / reference_count)

    direct_refs = sum(int(row.get("direct_aligner", {}).get("reference_count", 0)) for row in qualified)
    direct_matches = sum(int(row.get("direct_aligner", {}).get("matched_count", 0)) for row in qualified)

    progress_summaries = [row.get("audio_progress_summary", {}) for row in qualified]
    speech_region_summaries = [row.get("speech_region_diagnostics", {}) for row in qualified]
    gap_candidate_summaries = [row.get("gap_candidate_summary", {}) for row in qualified]
    advance_reason_counts: Counter[str] = Counter()
    gap_decision_counts: Counter[str] = Counter()
    for progress_summary in progress_summaries:
        advance_reason_counts.update(progress_summary.get("advance_reasons", {}))
    for gap_summary in gap_candidate_summaries:
        gap_decision_counts.update(gap_summary.get("decision_counts", {}))
    phone_class_rollup: dict[str, dict[str, Any]] = {}
    for row in qualified:
        for klass, stats in row.get("phone_class_summary", {}).items():
            bucket = phone_class_rollup.setdefault(klass, {"n": 0, "matched": 0, "unmapped": 0, "center": [], "start": [], "end": []})
            n = int(stats.get("n", 0))
            matched = int(stats.get("matched", 0))
            bucket["n"] += n
            bucket["matched"] += matched
            bucket["unmapped"] += int(stats.get("unmapped", 0))
            if matched > 0:
                bucket["center"].extend([float(stats.get("center_ms", 0.0))] * matched)
                bucket["start"].extend([float(stats.get("start_ms", 0.0))] * matched)
                bucket["end"].extend([float(stats.get("end_ms", 0.0))] * matched)
    phone_class_errors = {
        klass: {
            "n": b["n"],
            "matched": b["matched"],
            "unmapped": b["unmapped"],
            "center_ms": _mean(b["center"]),
            "start_ms": _mean(b["start"]),
            "end_ms": _mean(b["end"]),
        }
        for klass, b in sorted(phone_class_rollup.items())
    }

    landmark_types = ["mbp", "fv", "w", "chjjsh", "round", "comma_lull"]
    landmark_rollup: dict[str, dict[str, Any]] = {}
    conditioned_landmark_rollup: dict[str, dict[str, Any]] = {}
    for landmark_type in landmark_types:
        landmark_rollup[landmark_type] = {
            "target_count": 0,
            "observation_count": 0,
            "matched_target_count": 0,
            "matched_observation_count": 0,
            "center_error_weighted_sum": 0.0,
            "window_peak_weighted_sum": 0.0,
            "matched_score_weighted_sum": 0.0,
            "window_hit_weighted_sum": 0.0,
        }
        conditioned_landmark_rollup[landmark_type] = {
            "target_count": 0,
            "observation_count": 0,
            "matched_target_count": 0,
            "matched_observation_count": 0,
            "center_error_weighted_sum": 0.0,
            "window_peak_weighted_sum": 0.0,
            "matched_score_weighted_sum": 0.0,
            "window_hit_weighted_sum": 0.0,
        }

    for row in qualified:
        audit = row.get("landmark_audit", {})
        if not audit.get("available", False):
            continue
        type_map = audit.get("types", {})
        for landmark_type in landmark_types:
            stats = type_map.get(landmark_type, {})
            bucket = landmark_rollup[landmark_type]
            target_count = int(stats.get("target_count", 0))
            observation_count = int(stats.get("observation_count", 0))
            matched_target_count = int(stats.get("matched_target_count", 0))
            matched_observation_count = int(stats.get("matched_observation_count", 0))
            bucket["target_count"] += target_count
            bucket["observation_count"] += observation_count
            bucket["matched_target_count"] += matched_target_count
            bucket["matched_observation_count"] += matched_observation_count
            bucket["center_error_weighted_sum"] += float(stats.get("mean_abs_center_error_ms", 0.0)) * matched_target_count
            bucket["window_peak_weighted_sum"] += float(stats.get("mean_target_window_peak_score", 0.0)) * target_count
            bucket["matched_score_weighted_sum"] += float(stats.get("mean_matched_observation_score", 0.0)) * matched_observation_count
            bucket["window_hit_weighted_sum"] += float(stats.get("target_window_hit_rate", 0.0)) * target_count

        conditioned_audit = row.get("conditioned_landmark_audit", {})
        if conditioned_audit.get("available", False):
            conditioned_type_map = conditioned_audit.get("types", {})
            for landmark_type in landmark_types:
                stats = conditioned_type_map.get(landmark_type, {})
                bucket = conditioned_landmark_rollup[landmark_type]
                target_count = int(stats.get("target_count", 0))
                observation_count = int(stats.get("observation_count", 0))
                matched_target_count = int(stats.get("matched_target_count", 0))
                matched_observation_count = int(stats.get("matched_observation_count", 0))
                bucket["target_count"] += target_count
                bucket["observation_count"] += observation_count
                bucket["matched_target_count"] += matched_target_count
                bucket["matched_observation_count"] += matched_observation_count
                bucket["center_error_weighted_sum"] += float(stats.get("mean_abs_center_error_ms", 0.0)) * matched_target_count
                bucket["window_peak_weighted_sum"] += float(stats.get("mean_target_window_peak_score", 0.0)) * target_count
                bucket["matched_score_weighted_sum"] += float(stats.get("mean_matched_observation_score", 0.0)) * matched_observation_count
                bucket["window_hit_weighted_sum"] += float(stats.get("target_window_hit_rate", 0.0)) * target_count

    region_drop_rows = [drop_row for row in qualified for drop_row in row.get("region_drop_rows", [])]
    drop_regions_with_drops = [r for r in region_drop_rows if _safe_int(r, "dropped_event_count", 0) > 0]
    drop_regions_without_observed_region = [
        r for r in drop_regions_with_drops
        if _safe_float(r, "region_start", -1.0) < 0.0 or _safe_float(r, "region_end", -1.0) < 0.0
    ]
    drop_regions_with_observed_region = [
        r for r in drop_regions_with_drops
        if _safe_float(r, "region_start", -1.0) >= 0.0 and _safe_float(r, "region_end", -1.0) >= 0.0
    ]
    drop_regions_with_commits = [
        r for r in drop_regions_with_observed_region
        if _safe_int(r, "committed_event_count", 0) > 0
    ]
    underrun_regions = [
        r for r in region_drop_rows
        if _safe_float(r, "underrun_idle_tail_ms", 0.0) > 100.0
    ]

    detected_word_onset_values = [_nested(row, "word_onset_alignment", "mean_abs_start_error_ms") for row in qualified]
    detected_word_onset_event_medians = [_nested(row, "word_onset_alignment", "median_abs_start_error_ms") for row in qualified]
    word_duration_values = [_metric(row, "mean_abs_duration_error_ms") for row in qualified]
    phoneme_center_values = [_metric(row, "mean_abs_center_error_ms") for row in qualified]
    phoneme_start_values = [_metric(row, "mean_abs_start_error_ms") for row in qualified]
    phoneme_end_values = [_metric(row, "mean_abs_end_error_ms") for row in qualified]
    intra_word_center_values = [_nested(row, "intra_word_alignment", "mean_abs_center_error_ms") for row in qualified]
    direct_center_values = [_direct(row, "mean_abs_center_error_ms") for row in qualified if row.get("direct_aligner")]

    summary = {
        "total_cases": len(rows),
        "graded_cases": graded_cases,
        "qualified_cases": qualified_cases,
        "disqualified_speech_region_mismatch_cases": graded_cases - qualified_cases,
        "ungraded_cases": len(ungraded),
        "degenerate_cases": sum(1 for row in qualified if _metric(row, "committed_count") <= 0),
        "order_fail_cases": sum(1 for row in qualified if _metric(row, "order_violations") > 0),
        "speech_region_count_mismatch_cases": sum(1 for row in graded if _pause_flag(row, "speech_region", "count_mismatch")),
        "phone_occupancy_region_count_mismatch_cases": 0,
        "visible_speech_region_count_mismatch_cases": 0,
        "text_sentence_span_count_mismatch_cases": sum(1 for row in qualified if _pause_flag(row, "text_sentence_span", "count_mismatch")),
        "text_sentence_region_subspan_count_mismatch_cases": sum(1 for row in qualified if _pause_flag(row, "text_sentence_region_subspan", "count_mismatch")),
        "runtime_phone_alignment_available": runtime_phone_available,
        "direct_aligner_available": direct_aligner_available,
        "audio_progress_available": audio_progress_available,
        "landmark_audit_available": landmark_audit_available,
        "conditioned_landmark_audit_available": conditioned_landmark_audit_available,
        "speech_f1": _mean(
            _pause_metric(row, "speech_region", "matched_count")
            / max(_pause_metric(row, "speech_region", "reference_count"), 1.0)
            for row in qualified
        ),
        "speech_boundary_start_ms": _mean(_pause_metric(row, "speech_region", "start_error_ms") for row in qualified),
        "speech_boundary_end_ms": _mean(_pause_metric(row, "speech_region", "end_error_ms") for row in qualified),
        "speech_tail_leakage_ms": _mean(_pause_metric(row, "speech_region", "tail_out_ms") for row in qualified),
        "speech_region_containment_invalid_cases": sum(
            1 for row in qualified if _nested(row, "speech_region_containment", "events_with_invalid_region") > 0
        ),
        "speech_region_containment_leak_cases": sum(
            1 for row in qualified if _nested(row, "speech_region_containment", "events_outside_region") > 0
        ),
        "speech_region_containment_invalid_events": sum(
            _nested(row, "speech_region_containment", "events_with_invalid_region") for row in qualified
        ),
        "speech_region_containment_leak_events": sum(
            _nested(row, "speech_region_containment", "events_outside_region") for row in qualified
        ),
        "speech_region_containment_early_entry_events": sum(
            _nested(row, "speech_region_containment", "events_starting_before_region") for row in qualified
        ),
        "speech_region_containment_late_tail_events": sum(
            _nested(row, "speech_region_containment", "events_ending_after_region") for row in qualified
        ),
        "speech_region_containment_early_leak_ms": _mean(
            _nested(row, "speech_region_containment", "mean_early_start_leak_ms") for row in qualified
        ),
        "speech_region_containment_late_leak_ms": _mean(
            _nested(row, "speech_region_containment", "mean_late_end_leak_ms") for row in qualified
        ),
        "dropped_viseme_cases": sum(
            1 for row in qualified if _nested(row, "dropped_visemes", "dropped_count") > 0
        ),
        "dropped_viseme_count": sum(
            _nested(row, "dropped_visemes", "dropped_count") for row in qualified
        ),
        "dropped_region_closed_count": sum(
            _nested(row, "dropped_visemes", "dropped_region_closed_count") for row in qualified
        ),
        "dropped_missing_region_count": sum(
            _nested(row, "dropped_visemes", "dropped_missing_region_count") for row in qualified
        ),
        "dropped_strong_visible_count": sum(
            _nested(row, "dropped_visemes", "dropped_strong_visible_count") for row in qualified
        ),
        "dropped_viseme_rate": _mean(
            _nested(row, "dropped_visemes", "dropped_rate") for row in qualified
        ),
        "drop_region_count": len(region_drop_rows),
        "drop_regions_with_drops": len(drop_regions_with_drops),
        "drop_regions_without_observed_region": len(drop_regions_without_observed_region),
        "drop_regions_with_observed_region": len(drop_regions_with_observed_region),
        "drop_regions_without_any_commits": sum(
            1 for r in drop_regions_with_observed_region if _safe_int(r, "committed_event_count", 0) <= 0
        ),
        "drop_aligner_tail_pinned_committed_count": sum(
            _safe_int(r, "tail_pinned_aligner_committed_count", 0) for r in region_drop_rows
        ),
        "drop_first_commit_lag_ms": _mean(
            _safe_float(r, "first_commit_lag_ms", 0.0) for r in drop_regions_with_commits
        ),
        "drop_last_commit_to_region_end_ms": _mean(
            _safe_float(r, "last_commit_to_region_end_ms", 0.0) for r in drop_regions_with_commits
        ),
        "drop_required_deficit_mean_ms": _mean(
            _safe_float(r, "dropped_required_deficit_mean_ms", 0.0) for r in drop_regions_with_commits
        ),
        "drop_required_deficit_max_ms": _mean(
            _safe_float(r, "dropped_required_deficit_max_ms", 0.0) for r in drop_regions_with_commits
        ),
        "underrun_region_count": len(underrun_regions),
        "underrun_idle_tail_ms": _mean(
            _safe_float(r, "underrun_idle_tail_ms", 0.0) for r in underrun_regions
        ),
        "underrun_idle_tail_max_ms": max(
            (_safe_float(r, "underrun_idle_tail_ms", 0.0) for r in underrun_regions),
            default=0.0,
        ),
        "phone_occupancy_boundary_end_ms": 0.0,
        "visible_speech_boundary_end_ms": 0.0,
        "speech_region_gold_count": _mean(s.get("gold_count", 0) for s in speech_region_summaries),
        "speech_region_predicted_count": _mean(s.get("predicted_count", 0) for s in speech_region_summaries),
        "speech_region_count_delta": _mean(s.get("count_delta", 0) for s in speech_region_summaries),
        "speech_region_indexed_start_ms": _mean(s.get("indexed", {}).get("mean_start_error_ms", 0.0) for s in speech_region_summaries),
        "speech_region_indexed_end_ms": _mean(s.get("indexed", {}).get("mean_end_error_ms", 0.0) for s in speech_region_summaries),
        "speech_region_indexed_overlap_ratio": _mean(s.get("indexed", {}).get("mean_overlap_ratio", 0.0) for s in speech_region_summaries),
        "speech_region_overlap_start_ms": _mean(s.get("overlap", {}).get("mean_start_error_ms", 0.0) for s in speech_region_summaries),
        "speech_region_overlap_end_ms": _mean(s.get("overlap", {}).get("mean_end_error_ms", 0.0) for s in speech_region_summaries),
        "speech_region_overlap_ratio": _mean(s.get("overlap", {}).get("mean_overlap_ratio", 0.0) for s in speech_region_summaries),
        "speech_region_overlap_matched_count": _mean(s.get("overlap", {}).get("overlap_matched_count", 0) for s in speech_region_summaries),
        "speech_region_nearest_matched_count": _mean(s.get("overlap", {}).get("nearest_matched_count", 0) for s in speech_region_summaries),
        "speech_region_predicted_only_count": _mean(s.get("overlap", {}).get("predicted_only_count", 0) for s in speech_region_summaries),
        "speech_region_gold_only_count": _mean(s.get("overlap", {}).get("gold_only_count", 0) for s in speech_region_summaries),
        "text_sentence_span_start_ms": _mean(_pause_metric(row, "text_sentence_span", "start_error_ms") for row in qualified),
        "text_sentence_span_end_ms": _mean(_pause_metric(row, "text_sentence_span", "end_error_ms") for row in qualified),
        "text_sentence_span_tail_out_ms": _mean(_pause_metric(row, "text_sentence_span", "tail_out_ms") for row in qualified),
        "text_sentence_region_subspan_start_ms": _mean(_pause_metric(row, "text_sentence_region_subspan", "start_error_ms") for row in qualified),
        "text_sentence_region_subspan_end_ms": _mean(_pause_metric(row, "text_sentence_region_subspan", "end_error_ms") for row in qualified),
        "text_sentence_region_subspan_tail_out_ms": _mean(_pause_metric(row, "text_sentence_region_subspan", "tail_out_ms") for row in qualified),
        "gap_candidate_count": sum(int(s.get("count", 0)) for s in gap_candidate_summaries),
        "gap_candidate_bridged": sum(int(s.get("bridged", 0)) for s in gap_candidate_summaries),
        "gap_candidate_split": sum(int(s.get("split", 0)) for s in gap_candidate_summaries),
        "gap_candidate_mean_gap_ms": _mean(s.get("mean_gap_ms", 0.0) for s in gap_candidate_summaries if s.get("count", 0)),
        "gap_candidate_ambiguous_count": sum(int(s.get("ambiguous_count", 0)) for s in gap_candidate_summaries),
        "gap_candidate_ambiguous_bridged": sum(int(s.get("ambiguous_bridged", 0)) for s in gap_candidate_summaries),
        "gap_candidate_ambiguous_split": sum(int(s.get("ambiguous_split", 0)) for s in gap_candidate_summaries),
        "gap_candidate_ambiguous_mean_gap_ms": _mean(s.get("ambiguous_mean_gap_ms", 0.0) for s in gap_candidate_summaries if s.get("ambiguous_count", 0)),
        "gap_decision_counts": dict(sorted(gap_decision_counts.items())),
        "word_f1": _mean((_nested(row, "word_onset_alignment", "matched_count") / max(_nested(row, "word_onset_alignment", "reference_count"), 1.0)) for row in qualified),
        "word_assignment_rate": _mean((_nested(row, "word_onset_alignment", "matched_count") / max(_nested(row, "word_onset_alignment", "reference_count"), 1.0)) for row in qualified),
        "detected_word_onset_ms": _mean(_nested(row, "word_onset_alignment", "mean_abs_start_error_ms") for row in qualified),
        "word_duration_ms": _mean(_metric(row, "mean_abs_duration_error_ms") for row in qualified),
        "detected_word_onset_median_ms": _median(_nested(row, "word_onset_alignment", "median_abs_start_error_ms") for row in qualified),
        "phoneme_coverage_rate": _mean(phoneme_coverage),
        "phoneme_center_ms": _mean(_metric(row, "mean_abs_center_error_ms") for row in qualified),
        "phoneme_start_ms": _mean(_metric(row, "mean_abs_start_error_ms") for row in qualified),
        "phoneme_end_ms": _mean(_metric(row, "mean_abs_end_error_ms") for row in qualified),
        "intra_word_coverage_rate": _mean((_nested(row, "intra_word_alignment", "matched_count") / max(_nested(row, "intra_word_alignment", "reference_count"), 1.0)) for row in qualified),
        "intra_word_center_ms": _mean(_nested(row, "intra_word_alignment", "mean_abs_center_error_ms") for row in qualified),
        "advance_reason_counts": dict(sorted(advance_reason_counts.items())),
        "phone_class_errors": phone_class_errors,
    }
    if direct_aligner_available:
        summary["direct_aligner_match_rate"] = direct_matches / max(direct_refs, 1)
        summary["direct_aligner_center_ms"] = _mean(_direct(row, "mean_abs_center_error_ms") for row in qualified if row.get("direct_aligner"))
    if landmark_audit_available:
        total_landmark_targets = 0
        total_landmark_observations = 0
        total_landmark_matches = 0
        for landmark_type, bucket in landmark_rollup.items():
            target_count = bucket["target_count"]
            observation_count = bucket["observation_count"]
            matched_target_count = bucket["matched_target_count"]
            matched_observation_count = bucket["matched_observation_count"]
            summary[f"landmark_{landmark_type}_target_count"] = target_count
            summary[f"landmark_{landmark_type}_observation_count"] = observation_count
            summary[f"landmark_{landmark_type}_matched_target_count"] = matched_target_count
            summary[f"landmark_{landmark_type}_matched_observation_count"] = matched_observation_count
            summary[f"landmark_{landmark_type}_recall"] = (
                matched_target_count / target_count if target_count else 0.0
            )
            summary[f"landmark_{landmark_type}_precision"] = (
                matched_observation_count / observation_count if observation_count else 0.0
            )
            summary[f"landmark_{landmark_type}_mean_abs_center_error_ms"] = (
                bucket["center_error_weighted_sum"] / matched_target_count if matched_target_count else 0.0
            )
            summary[f"landmark_{landmark_type}_mean_target_window_peak_score"] = (
                bucket["window_peak_weighted_sum"] / target_count if target_count else 0.0
            )
            summary[f"landmark_{landmark_type}_mean_matched_observation_score"] = (
                bucket["matched_score_weighted_sum"] / matched_observation_count if matched_observation_count else 0.0
            )
            summary[f"landmark_{landmark_type}_target_window_hit_rate"] = (
                bucket["window_hit_weighted_sum"] / target_count if target_count else 0.0
            )
            total_landmark_targets += target_count
            total_landmark_observations += observation_count
            total_landmark_matches += matched_target_count
        summary["landmark_target_count"] = total_landmark_targets
        summary["landmark_observation_count"] = total_landmark_observations
        summary["landmark_matched_target_count"] = total_landmark_matches
        summary["landmark_recall"] = (
            total_landmark_matches / total_landmark_targets if total_landmark_targets else 0.0
        )
    if conditioned_landmark_audit_available:
        total_landmark_targets = 0
        total_landmark_observations = 0
        total_landmark_matches = 0
        for landmark_type, bucket in conditioned_landmark_rollup.items():
            target_count = bucket["target_count"]
            observation_count = bucket["observation_count"]
            matched_target_count = bucket["matched_target_count"]
            matched_observation_count = bucket["matched_observation_count"]
            summary[f"conditioned_landmark_{landmark_type}_target_count"] = target_count
            summary[f"conditioned_landmark_{landmark_type}_observation_count"] = observation_count
            summary[f"conditioned_landmark_{landmark_type}_matched_target_count"] = matched_target_count
            summary[f"conditioned_landmark_{landmark_type}_matched_observation_count"] = matched_observation_count
            summary[f"conditioned_landmark_{landmark_type}_recall"] = (
                matched_target_count / target_count if target_count else 0.0
            )
            summary[f"conditioned_landmark_{landmark_type}_precision"] = (
                matched_observation_count / observation_count if observation_count else 0.0
            )
            summary[f"conditioned_landmark_{landmark_type}_mean_abs_center_error_ms"] = (
                bucket["center_error_weighted_sum"] / matched_target_count if matched_target_count else 0.0
            )
            summary[f"conditioned_landmark_{landmark_type}_mean_target_window_peak_score"] = (
                bucket["window_peak_weighted_sum"] / target_count if target_count else 0.0
            )
            summary[f"conditioned_landmark_{landmark_type}_mean_matched_observation_score"] = (
                bucket["matched_score_weighted_sum"] / matched_observation_count if matched_observation_count else 0.0
            )
            summary[f"conditioned_landmark_{landmark_type}_target_window_hit_rate"] = (
                bucket["window_hit_weighted_sum"] / target_count if target_count else 0.0
            )
            total_landmark_targets += target_count
            total_landmark_observations += observation_count
            total_landmark_matches += matched_target_count
        summary["conditioned_landmark_target_count"] = total_landmark_targets
        summary["conditioned_landmark_observation_count"] = total_landmark_observations
        summary["conditioned_landmark_matched_target_count"] = total_landmark_matches
        summary["conditioned_landmark_recall"] = (
            total_landmark_matches / total_landmark_targets if total_landmark_targets else 0.0
        )
    if audio_progress_available:
        summary.update({
            "audio_progress_rows": sum(len(row.get("audio_progress_rows", [])) for row in rows),
            "audio_progress_mfa_rows": sum(int(s.get("rows", 0)) for s in progress_summaries),
            "audio_progress_would_advance_rows": sum(int(s.get("would_advance_rows", 0)) for s in progress_summaries),
            "audio_progress_mfa_mae01": _mean(s.get("mae01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_mfa_median01": _median(s.get("median01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_mfa_mae_ms": _mean(s.get("mae_ms", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_mfa_p90_ms": _mean(s.get("p90_ms", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_mfa_bias01": _mean(s.get("bias01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_mfa_corr": _mean(s.get("corr", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_time_prior_mae01": _mean(s.get("time_prior_mae01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_density_mae01": _mean(s.get("audio_density_mae01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_boundary_mae01": _mean(s.get("boundary_mae01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_phone_mae01": _mean(s.get("phone_expectation_mae01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_anchor_mean_phone_probability": _mean(s.get("anchor_mean_phone_probability", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_anchor_mean_boundary_probability": _mean(s.get("anchor_mean_boundary_probability", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_anchor_mean_speech_probability": _mean(s.get("anchor_mean_speech_probability", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_micro_pause_50_count": _mean(s.get("micro_pause_50_count", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_micro_pause_75_count": _mean(s.get("micro_pause_75_count", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_micro_pause_120_count": _mean(s.get("micro_pause_120_count", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_retrospective_drift_abs01": _mean(s.get("retrospective_drift_abs01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_retrospective_drift_abs_ms": _mean(s.get("retrospective_drift_abs_ms", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_retrospective_drift_bias01": _mean(s.get("retrospective_drift_bias01", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_retrospective_drift_confidence": _mean(s.get("retrospective_drift_confidence", 0.0) for s in progress_summaries if s.get("rows", 0)),
            "audio_progress_drift_mean_playrate": _mean(s.get("drift_mean_playrate", 1.0) for s in progress_summaries if s.get("rows", 0)),
        })
    summary.update(_stats("detected_word_onset", detected_word_onset_values))
    summary.update(_stats("detected_word_onset_event_median", detected_word_onset_event_medians))
    summary.update(_stats("word_duration", word_duration_values))
    summary.update(_stats("phoneme_center", phoneme_center_values))
    summary.update(_stats("phoneme_start", phoneme_start_values))
    summary.update(_stats("phoneme_end", phoneme_end_values))
    summary.update(_stats("intra_word_center", intra_word_center_values))
    if direct_aligner_available:
        summary.update(_stats("direct_aligner_center", direct_center_values))
    # Keep legacy headline names stable for existing scripts/checks.
    summary["detected_word_onset_ms"] = summary["detected_word_onset_mean_ms"]
    summary["detected_word_onset_median_ms"] = summary["detected_word_onset_event_median_median_ms"]
    summary["word_start_ms"] = summary["detected_word_onset_mean_ms"]
    summary["word_head_start_ms"] = summary["detected_word_onset_mean_ms"]
    summary["word_head_start_median_ms"] = summary["detected_word_onset_event_median_median_ms"]
    summary["word_start_mean_ms"] = summary["detected_word_onset_mean_ms"]
    summary["word_start_median_ms"] = summary["detected_word_onset_median_ms"]
    summary["word_start_p90_ms"] = summary["detected_word_onset_p90_ms"]
    summary["word_start_max_ms"] = summary["detected_word_onset_max_ms"]
    summary["word_start_stddev_ms"] = summary["detected_word_onset_stddev_ms"]
    summary["word_start_event_median_mean_ms"] = summary["detected_word_onset_event_median_mean_ms"]
    summary["word_start_event_median_median_ms"] = summary["detected_word_onset_event_median_median_ms"]
    summary["word_start_event_median_p90_ms"] = summary["detected_word_onset_event_median_p90_ms"]
    summary["word_start_event_median_max_ms"] = summary["detected_word_onset_event_median_max_ms"]
    summary["word_start_event_median_stddev_ms"] = summary["detected_word_onset_event_median_stddev_ms"]
    summary["word_duration_ms"] = summary["word_duration_mean_ms"]
    summary["phoneme_center_ms"] = summary["phoneme_center_mean_ms"]
    summary["phoneme_start_ms"] = summary["phoneme_start_mean_ms"]
    summary["phoneme_end_ms"] = summary["phoneme_end_mean_ms"]
    summary["intra_word_center_ms"] = summary["intra_word_center_mean_ms"]
    summary["sentence_region_count_mismatch_cases"] = summary["text_sentence_span_count_mismatch_cases"]
    summary["clause_region_count_mismatch_cases"] = summary["text_sentence_region_subspan_count_mismatch_cases"]
    summary["sentence_region_start_ms"] = summary["text_sentence_span_start_ms"]
    summary["sentence_region_end_ms"] = summary["text_sentence_span_end_ms"]
    summary["sentence_region_tail_out_ms"] = summary["text_sentence_span_tail_out_ms"]
    summary["clause_region_start_ms"] = summary["text_sentence_region_subspan_start_ms"]
    summary["clause_region_end_ms"] = summary["text_sentence_region_subspan_end_ms"]
    summary["clause_region_tail_out_ms"] = summary["text_sentence_region_subspan_tail_out_ms"]
    if direct_aligner_available:
        summary["direct_aligner_center_ms"] = summary["direct_aligner_center_mean_ms"]
    return summary


def write_summary(root: pathlib.Path, summary: dict[str, Any]) -> pathlib.Path:
    path = root / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--gold-root", type=pathlib.Path, default=None)
    args = parser.parse_args()
    rows, graded, ungraded = load_case_grades(args.root, args.gold_root)
    summary = compute_summary(rows, graded, ungraded)
    write_summary(args.root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
