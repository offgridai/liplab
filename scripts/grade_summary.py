import csv
import json
import pathlib
import statistics
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
        boundary_filter_rows = read_csv_rows(case_dir / "boundary_filter_training.csv")
        committed_rows = read_csv_rows(case_dir / "committed.csv")
        gold_speech_rows = read_csv_rows(gold_root / case_dir.name / "speech.csv") if gold_root else []
        gold_viseme_rows = _gold_viseme_rows_for_case(gold_root, case_dir.name)
        row = {
            "case": case_dir.name,
            "grade": grade,
            "direct_aligner": direct,
            "runtime_phone_rows": runtime_phone_rows,
            "audio_progress_rows": progress_rows,
            "boundary_filter_rows": boundary_filter_rows,
            "committed_rows": committed_rows,
            "gold_speech_rows": gold_speech_rows,
            "gold_viseme_rows": gold_viseme_rows,
        }
        row["audio_progress_summary"] = summarize_audio_progress(row)
        row["boundary_filter_summary"] = summarize_boundary_filter(row)
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
    for r in progress_rows:
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
    }


def summarize_boundary_filter(row: dict[str, Any], threshold: float = 0.35) -> dict[str, Any]:
    rows = row.get("boundary_filter_rows", [])
    expected = 0
    for r in rows:
        expected = max(expected, _safe_int(r, "expected_boundary_count"))
    raw_tp = raw_fp = filt_tp = filt_fp = 0
    raw_matched = set()
    filt_matched = set()
    fp_sustained = fp_flux_only = fp_flat = 0
    for r in rows:
        label = _safe_int(r, "label_true") == 1
        nearest = round(_safe_float(r, "nearest_mfa_boundary"), 3)
        raw_on = _safe_float(r, "raw_confidence") >= threshold
        filt_on = _safe_float(r, "filtered_confidence") >= threshold
        if raw_on and label:
            raw_tp += 1; raw_matched.add(nearest)
        elif raw_on:
            raw_fp += 1
        if filt_on and label:
            filt_tp += 1; filt_matched.add(nearest)
        elif filt_on:
            filt_fp += 1
            red = _safe_float(r, "red_herring_probability")
            rms = _safe_float(r, "rms_norm")
            flux = _safe_float(r, "flux")
            valley = _safe_int(r, "local_rms_valley") == 1
            periodic = _safe_float(r, "periodicity")
            if rms > 0.20 and red > 0.35:
                fp_sustained += 1
            elif flux > 0.30 and not valley:
                fp_flux_only += 1
            elif periodic < 0.20 and rms < 0.08:
                fp_flat += 1
    raw_p = raw_tp / max(raw_tp + raw_fp, 1)
    filt_p = filt_tp / max(filt_tp + filt_fp, 1)
    return {
        "candidates": len(rows),
        "expected": expected,
        "raw_precision": raw_p,
        "raw_recall": len(raw_matched) / max(expected, 1),
        "raw_tp": raw_tp,
        "raw_fp": raw_fp,
        "filtered_precision": filt_p,
        "filtered_recall": len(filt_matched) / max(expected, 1),
        "filtered_tp": filt_tp,
        "filtered_fp": filt_fp,
        "filtered_fp_sustained": fp_sustained,
        "filtered_fp_flux_only": fp_flux_only,
        "filtered_fp_flat": fp_flat,
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
    phoneme_coverage = []
    for row in rows:
        phone_rows = row.get("runtime_phone_rows", [])
        if phone_rows:
            mapped = sum(1 for r in phone_rows if str(r.get("mapped_to_observed_speech", "0")) == "1")
            phoneme_coverage.append(mapped / max(len(phone_rows), 1))

    direct_refs = sum(int(row.get("direct_aligner", {}).get("reference_count", 0)) for row in graded)
    direct_matches = sum(int(row.get("direct_aligner", {}).get("matched_count", 0)) for row in graded)

    progress_summaries = [row.get("audio_progress_summary", {}) for row in graded]
    boundary_summaries = [row.get("boundary_filter_summary", {}) for row in graded]
    phone_class_rollup: dict[str, dict[str, Any]] = {}
    for row in graded:
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

    word_start_values = [_nested(row, "word_onset_alignment", "mean_abs_start_error_ms") for row in graded]
    word_start_event_medians = [_nested(row, "word_onset_alignment", "median_abs_start_error_ms") for row in graded]
    word_duration_values = [_metric(row, "mean_abs_duration_error_ms") for row in graded]
    phoneme_center_values = [_metric(row, "mean_abs_center_error_ms") for row in graded]
    phoneme_start_values = [_metric(row, "mean_abs_start_error_ms") for row in graded]
    phoneme_end_values = [_metric(row, "mean_abs_end_error_ms") for row in graded]
    intra_word_center_values = [_nested(row, "intra_word_alignment", "mean_abs_center_error_ms") for row in graded]
    direct_center_values = [_direct(row, "mean_abs_center_error_ms") for row in graded]

    summary = {
        "total_cases": len(rows),
        "graded_cases": graded_cases,
        "ungraded_cases": len(ungraded),
        "degenerate_cases": sum(1 for row in graded if _metric(row, "committed_count") <= 0),
        "order_fail_cases": sum(1 for row in graded if _metric(row, "order_violations") > 0),
        "speech_region_count_mismatch_cases": sum(1 for row in graded if bool(row.get("grade", {}).get("pause_alignment", {}).get("speech_regions", {}).get("count_mismatch", False))),
        "phone_occupancy_region_count_mismatch_cases": 0,
        "visible_speech_region_count_mismatch_cases": 0,
        "sentence_region_count_mismatch_cases": sum(1 for row in graded if bool(row.get("grade", {}).get("pause_alignment", {}).get("sentence_regions", {}).get("count_mismatch", False))),
        "clause_region_count_mismatch_cases": sum(1 for row in graded if bool(row.get("grade", {}).get("pause_alignment", {}).get("clause_regions", {}).get("count_mismatch", False))),
        "speech_f1": _mean((_nested(row, "pause_alignment", "speech_regions", "matched_count") / max(_nested(row, "pause_alignment", "speech_regions", "reference_count"), 1.0)) for row in graded),
        "speech_boundary_start_ms": _mean(_nested(row, "pause_alignment", "speech_regions", "mean_abs_start_error_ms") for row in graded),
        "speech_boundary_end_ms": _mean(_nested(row, "pause_alignment", "speech_regions", "mean_abs_end_error_ms") for row in graded),
        "speech_tail_leakage_ms": _mean(_nested(row, "pause_alignment", "speech_regions", "mean_tail_out_ms") for row in graded),
        "phone_occupancy_boundary_end_ms": 0.0,
        "visible_speech_boundary_end_ms": 0.0,
        "word_f1": _mean((_nested(row, "word_onset_alignment", "matched_count") / max(_nested(row, "word_onset_alignment", "reference_count"), 1.0)) for row in graded),
        "word_start_ms": _mean(_nested(row, "word_onset_alignment", "mean_abs_start_error_ms") for row in graded),
        "word_end_ms": 0.0,
        "word_duration_ms": _mean(_metric(row, "mean_abs_duration_error_ms") for row in graded),
        "word_duration_l1_norm": 0.0,
        "word_stretch_abs_log2": 0.0,
        "word_head_start_ms": _mean(_nested(row, "word_onset_alignment", "mean_abs_start_error_ms") for row in graded),
        "word_head_start_median_ms": _median(_nested(row, "word_onset_alignment", "median_abs_start_error_ms") for row in graded),
        "phoneme_coverage_rate": _mean(phoneme_coverage),
        "phoneme_center_ms": _mean(_metric(row, "mean_abs_center_error_ms") for row in graded),
        "phoneme_start_ms": _mean(_metric(row, "mean_abs_start_error_ms") for row in graded),
        "phoneme_end_ms": _mean(_metric(row, "mean_abs_end_error_ms") for row in graded),
        "intra_word_coverage_rate": _mean((_nested(row, "intra_word_alignment", "matched_count") / max(_nested(row, "intra_word_alignment", "reference_count"), 1.0)) for row in graded),
        "intra_word_center_ms": _mean(_nested(row, "intra_word_alignment", "mean_abs_center_error_ms") for row in graded),
        "direct_aligner_match_rate": direct_matches / max(direct_refs, 1),
        "direct_aligner_center_ms": _mean(_direct(row, "mean_abs_center_error_ms") for row in graded),
        "audio_progress_rows": sum(len(row.get("audio_progress_rows", [])) for row in rows),
        "audio_progress_mfa_rows": sum(int(s.get("rows", 0)) for s in progress_summaries),
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
        "boundary_filter_candidates": sum(int(s.get("candidates", 0)) for s in boundary_summaries),
        "boundary_filter_expected": sum(int(s.get("expected", 0)) for s in boundary_summaries),
        "boundary_filter_raw_precision": _mean(s.get("raw_precision", 0.0) for s in boundary_summaries if s.get("candidates", 0)),
        "boundary_filter_raw_recall": _mean(s.get("raw_recall", 0.0) for s in boundary_summaries if s.get("candidates", 0)),
        "boundary_filter_filtered_precision": _mean(s.get("filtered_precision", 0.0) for s in boundary_summaries if s.get("candidates", 0)),
        "boundary_filter_filtered_recall": _mean(s.get("filtered_recall", 0.0) for s in boundary_summaries if s.get("candidates", 0)),
        "boundary_filter_filtered_fp": sum(int(s.get("filtered_fp", 0)) for s in boundary_summaries),
        "boundary_filter_filtered_fp_sustained": sum(int(s.get("filtered_fp_sustained", 0)) for s in boundary_summaries),
        "boundary_filter_filtered_fp_flux_only": sum(int(s.get("filtered_fp_flux_only", 0)) for s in boundary_summaries),
        "boundary_filter_filtered_fp_flat": sum(int(s.get("filtered_fp_flat", 0)) for s in boundary_summaries),
        "phone_class_errors": phone_class_errors,
    }
    summary.update(_stats("word_start", word_start_values))
    summary.update(_stats("word_start_event_median", word_start_event_medians))
    summary.update(_stats("word_duration", word_duration_values))
    summary.update(_stats("phoneme_center", phoneme_center_values))
    summary.update(_stats("phoneme_start", phoneme_start_values))
    summary.update(_stats("phoneme_end", phoneme_end_values))
    summary.update(_stats("intra_word_center", intra_word_center_values))
    summary.update(_stats("direct_aligner_center", direct_center_values))
    # Keep legacy headline names stable for existing scripts/checks.
    summary["word_start_ms"] = summary["word_start_mean_ms"]
    summary["word_head_start_ms"] = summary["word_start_mean_ms"]
    summary["word_head_start_median_ms"] = summary["word_start_event_median_median_ms"]
    summary["word_duration_ms"] = summary["word_duration_mean_ms"]
    summary["phoneme_center_ms"] = summary["phoneme_center_mean_ms"]
    summary["phoneme_start_ms"] = summary["phoneme_start_mean_ms"]
    summary["phoneme_end_ms"] = summary["phoneme_end_mean_ms"]
    summary["intra_word_center_ms"] = summary["intra_word_center_mean_ms"]
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
