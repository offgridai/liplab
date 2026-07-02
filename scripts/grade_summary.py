import csv
import json
import pathlib
import statistics
import math
from typing import Iterable


def read_csv_rows(path: pathlib.Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def span_duration_seconds(spans):
    return sum(max(0.0, end - start) for start, end in spans)


def merge_spans(spans, gap_seconds=0.120):
    merged = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + gap_seconds:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def overlap_duration_seconds(lhs, rhs):
    total = 0.0
    i = 0
    j = 0
    while i < len(lhs) and j < len(rhs):
        start = max(lhs[i][0], rhs[j][0])
        end = min(lhs[i][1], rhs[j][1])
        if end > start:
            total += end - start
        if lhs[i][1] <= rhs[j][1]:
            i += 1
        else:
            j += 1
    return total



def span_boundary_metrics(reference_spans, predicted_spans, gap_seconds=0.120):
    """Return occupancy and boundary metrics for two span sets.

    Boundary errors are computed only for temporally-overlapping matched regions.
    This prevents one split/merge from poisoning every later region by index.
    Count mismatch remains explicit via missing_count/extra_count/count_mismatch.
    """
    reference_spans = merge_spans(reference_spans, gap_seconds)
    predicted_spans = merge_spans(predicted_spans, gap_seconds)
    reference_ms = span_duration_seconds(reference_spans) * 1000.0
    predicted_ms = span_duration_seconds(predicted_spans) * 1000.0
    overlap_ms = overlap_duration_seconds(reference_spans, predicted_spans) * 1000.0
    precision = overlap_ms / predicted_ms if predicted_ms > 0.0 else 0.0
    recall = overlap_ms / reference_ms if reference_ms > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision > 0.0 or recall > 0.0) else 0.0

    used_pred = set()
    start_errors = []
    end_errors = []
    for ref_start, ref_end in reference_spans:
        best_i = None
        best_overlap = 0.0
        for i, (pred_start, pred_end) in enumerate(predicted_spans):
            if i in used_pred:
                continue
            ov = max(0.0, min(ref_end, pred_end) - max(ref_start, pred_start))
            if ov > best_overlap:
                best_overlap = ov
                best_i = i
        if best_i is None or best_overlap <= 0.0:
            continue
        used_pred.add(best_i)
        pred_start, pred_end = predicted_spans[best_i]
        start_errors.append(abs(pred_start - ref_start) * 1000.0)
        end_errors.append(abs(pred_end - ref_end) * 1000.0)

    matched_count = len(start_errors)
    return {
        "reference_count": len(reference_spans),
        "predicted_count": len(predicted_spans),
        "matched_count": matched_count,
        "missing_count": max(0, len(reference_spans) - matched_count),
        "extra_count": max(0, len(predicted_spans) - matched_count),
        "count_mismatch": len(reference_spans) != len(predicted_spans),
        "reference_ms": reference_ms,
        "predicted_ms": predicted_ms,
        "overlap_ms": overlap_ms,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "start_errors_ms": start_errors,
        "end_errors_ms": end_errors,
        "region_start_error_sum_ms": sum(start_errors),
        "region_end_error_sum_ms": sum(end_errors),
        "mean_abs_start_error_ms": _mean(start_errors),
        "median_abs_start_error_ms": _median(start_errors),
        "p90_abs_start_error_ms": _percentile(start_errors, 0.90),
        "mean_abs_end_error_ms": _mean(end_errors),
        "median_abs_end_error_ms": _median(end_errors),
        "p90_abs_end_error_ms": _percentile(end_errors, 0.90),
        "tail_leakage_ms": tail_leakage_ms(reference_spans, predicted_spans),
    }


def empty_speech_layer():
    return span_boundary_metrics([], [])


def tail_leakage_ms(reference_spans, predicted_spans, gap_seconds=0.120):
    reference_spans = merge_spans(reference_spans, gap_seconds)
    predicted_spans = merge_spans(predicted_spans, gap_seconds)
    if not reference_spans or not predicted_spans:
        return 0.0

    total_ms = 0.0
    last_reference_end = reference_spans[-1][1]
    for pred_start, pred_end in predicted_spans:
        best_ref = None
        best_overlap = 0.0
        for ref_start, ref_end in reference_spans:
            overlap = max(0.0, min(ref_end, pred_end) - max(ref_start, pred_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_ref = (ref_start, ref_end)

        if best_ref is not None and best_overlap > 0.0:
            total_ms += max(0.0, pred_end - best_ref[1]) * 1000.0
        elif pred_start >= last_reference_end:
            total_ms += max(0.0, pred_end - pred_start) * 1000.0

    return total_ms


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: Iterable[float], q: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    t = pos - lo
    return values[lo] * (1.0 - t) + values[hi] * t


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _valid_span(item, start_key="start", end_key="end"):
    start = _safe_float(item.get(start_key), -1.0)
    end = _safe_float(item.get(end_key), -1.0)
    if start < 0.0 or end <= start:
        return None
    return (start, end)


def extract_runtime_phone_spans(rows):
    spans = []
    for item in rows:
        observed = _valid_span(item, "observed_start", "observed_end")
        if observed is not None:
            spans.append(observed)
            continue

        span = _valid_span(item, "start", "end")
        if span is None:
            continue

        # Older harness exports included fallback-paced phone timings in start/end
        # plus a mapped_to_observed_speech bit. Prefer acoustic/observed spans only.
        if ("observed_start" in item or "observed_end" in item) and _safe_float(item.get("mapped_to_observed_speech"), 0.0) <= 0.5:
            continue

        spans.append(span)
    return spans


def build_case_layers(case_dir: pathlib.Path, gold_case_dir: pathlib.Path | None, row):
    layers = {
        # Primary speech layer is detector-owned speech occupancy. Phone-derived
        # and visible-viseme envelopes remain separate diagnostics.
        "speech": empty_speech_layer(),
        "phone_occupancy": empty_speech_layer(),
        "speech_visible": empty_speech_layer(),
        "words": {
            "reference_count": 0,
            "matched_count": 0,
            "missing_count": 0,
            "coverage_rate": 0.0,
            "reference_ms": 0.0,
            "predicted_ms": 0.0,
            "overlap_ms": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "start_error_sum_ms": 0.0,
            "end_error_sum_ms": 0.0,
            "duration_error_sum_ms": 0.0,
            "start_errors_ms": [],
            "end_errors_ms": [],
            "duration_errors_ms": [],
            "duration_norm_errors": [],
            "stretch_abs_log2_errors": [],
            "duration_l1_all_ms": 0.0,
            "duration_l1_norm": 0.0,
            "matched_reference_ms": 0.0,
            "matched_predicted_ms": 0.0,
            "missing_reference_ms": 0.0,
            "extra_predicted_ms": 0.0,
            "mean_abs_start_error_ms": 0.0,
            "median_abs_start_error_ms": 0.0,
            "p90_abs_start_error_ms": 0.0,
            "mean_abs_end_error_ms": 0.0,
            "median_abs_end_error_ms": 0.0,
            "p90_abs_end_error_ms": 0.0,
            "mean_abs_duration_error_ms": 0.0,
            "median_abs_duration_error_ms": 0.0,
            "p90_abs_duration_error_ms": 0.0,
            "mean_abs_duration_error_norm": 0.0,
            "median_abs_duration_error_norm": 0.0,
            "p90_abs_duration_error_norm": 0.0,
            "mean_abs_stretch_log2": 0.0,
            "median_abs_stretch_log2": 0.0,
            "p90_abs_stretch_log2": 0.0,
            "head_reference_count": 0,
            "head_matched_count": 0,
            "head_start_error_sum_ms": 0.0,
            "head_start_errors_ms": [],
            "mean_head_start_error_ms": 0.0,
            "median_head_start_error_ms": 0.0,
            "p90_head_start_error_ms": 0.0,
        },
        "phonemes": {
            "reference_count": row["reference_count"],
            "matched_count": row["matched_count"],
            "missing_count": row["raw"].get("missing_count", 0),
            "extra_count": row["raw"].get("extra_count", 0),
            "coverage_rate": row["matched_count"] / max(row["reference_count"], 1),
            "center_error_sum_ms": row["mean_abs_center_error_ms"] * row["matched_count"],
            "start_error_sum_ms": row["mean_abs_start_error_ms"] * row["matched_count"],
            "end_error_sum_ms": row["mean_abs_end_error_ms"] * row["matched_count"],
            "mean_abs_center_error_ms": row["mean_abs_center_error_ms"],
            "median_abs_center_error_ms": row["median_abs_center_error_ms"],
            "p90_abs_center_error_ms": row["p90_abs_center_error_ms"],
            "mean_abs_start_error_ms": row["mean_abs_start_error_ms"],
            "mean_abs_end_error_ms": row["mean_abs_end_error_ms"],
            "intra_reference_count": row["intra_word_alignment"].get("reference_count", 0),
            "intra_matched_count": row["intra_word_alignment"].get("matched_count", 0),
            "intra_missing_count": row["intra_word_alignment"].get("missing_count", 0),
            "intra_extra_count": row["intra_word_alignment"].get("extra_count", 0),
            "intra_coverage_rate": row["intra_word_alignment"].get("matched_count", 0) / max(row["intra_word_alignment"].get("reference_count", 0), 1),
            "intra_center_error_sum_ms": row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0) * row["intra_word_alignment"].get("matched_count", 0),
            "intra_mean_abs_center_error_ms": row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0),
            "intra_median_abs_center_error_ms": row["intra_word_alignment"].get("median_abs_center_error_ms", 0.0),
            "intra_p90_abs_center_error_ms": row["intra_word_alignment"].get("p90_abs_center_error_ms", 0.0),
        },
    }

    words = layers["words"]
    word_onset = row["word_onset_alignment"]
    words["head_reference_count"] = word_onset.get("reference_count", 0)
    words["head_matched_count"] = word_onset.get("matched_count", 0)
    words["head_start_error_sum_ms"] = word_onset.get("mean_abs_start_error_ms", 0.0) * words["head_matched_count"]
    # Per-case median/p90 are diagnostic only; pooled median is computed from CSV below when possible.
    words["mean_head_start_error_ms"] = word_onset.get("mean_abs_start_error_ms", 0.0)
    words["median_head_start_error_ms"] = word_onset.get("median_abs_start_error_ms", words["mean_head_start_error_ms"])
    words["p90_head_start_error_ms"] = word_onset.get("p90_abs_start_error_ms", words["mean_head_start_error_ms"])

    if gold_case_dir is None or not gold_case_dir.exists():
        return layers

    gold_speech_rows = read_csv_rows(gold_case_dir / "speech.csv")
    gold_word_rows = read_csv_rows(gold_case_dir / "words.csv")
    committed_rows = read_csv_rows(case_dir / "committed.csv")
    runtime_alignment_rows = read_csv_rows(case_dir / "runtime_phone_alignment.csv")
    speech_region_rows = read_csv_rows(case_dir / "speech_regions.csv")
    onset_rows = read_csv_rows(case_dir / "word_onset_diagnostics.csv")

    gold_speech = [
        (_safe_float(item.get("start")), _safe_float(item.get("end")))
        for item in gold_speech_rows
    ]
    visible_speech = [
        (_safe_float(item.get("start")), _safe_float(item.get("end")))
        for item in committed_rows
        if _safe_float(item.get("end")) > _safe_float(item.get("start"))
    ]
    runtime_phone_speech = extract_runtime_phone_spans(runtime_alignment_rows)
    detector_speech = [
        (_safe_float(item.get("start")), _safe_float(item.get("end")))
        for item in speech_region_rows
        if _safe_float(item.get("end")) > _safe_float(item.get("start"))
    ]

    layers["phone_occupancy"] = span_boundary_metrics(gold_speech, runtime_phone_speech)
    layers["speech_visible"] = span_boundary_metrics(gold_speech, visible_speech)
    if detector_speech:
        layers["speech"] = span_boundary_metrics(gold_speech, detector_speech)
    elif runtime_phone_speech:
        # Backward compatibility for older runs before speech_regions.csv existed.
        layers["speech"] = layers["phone_occupancy"]
    else:
        # Oldest runs may only have committed viseme spans.
        layers["speech"] = layers["speech_visible"]

    predicted_word_spans = {}
    for item in committed_rows:
        word_index = int(item.get("word_index", -1))
        if word_index < 0:
            continue
        start = _safe_float(item.get("start"))
        end = _safe_float(item.get("end"))
        if word_index not in predicted_word_spans:
            predicted_word_spans[word_index] = [start, end]
        else:
            predicted_word_spans[word_index][0] = min(predicted_word_spans[word_index][0], start)
            predicted_word_spans[word_index][1] = max(predicted_word_spans[word_index][1], end)

    words["reference_count"] = len(gold_word_rows)
    gold_word_indices = set()
    word_duration_diag_rows = []
    for item in gold_word_rows:
        word_index = int(item.get("word_index", -1))
        if word_index >= 0:
            gold_word_indices.add(word_index)
        gold_start = _safe_float(item.get("start"))
        gold_end = _safe_float(item.get("end"))
        gold_duration_ms = max(0.0, gold_end - gold_start) * 1000.0
        words["reference_ms"] += gold_duration_ms
        predicted = predicted_word_spans.get(word_index)
        diag = {
            "word_index": word_index,
            "word": item.get("word") or item.get("text") or item.get("label") or "",
            "gold_start": gold_start,
            "gold_end": gold_end,
            "gold_duration_ms": gold_duration_ms,
            "predicted_start": "",
            "predicted_end": "",
            "predicted_duration_ms": 0.0,
            "overlap_ms": 0.0,
            "start_error_ms": "",
            "end_error_ms": "",
            "duration_error_ms": gold_duration_ms,
            "duration_error_norm": 1.0 if gold_duration_ms > 0.0 else 0.0,
            "stretch_factor": "",
            "stretch_abs_log2": "",
            "missing_predicted": 1,
            "extra_predicted": 0,
        }
        if predicted is None:
            words["missing_reference_ms"] += gold_duration_ms
            words["duration_l1_all_ms"] += gold_duration_ms
            word_duration_diag_rows.append(diag)
            continue
        words["matched_count"] += 1
        predicted_duration_ms = max(0.0, predicted[1] - predicted[0]) * 1000.0
        overlap_ms = max(0.0, min(predicted[1], gold_end) - max(predicted[0], gold_start)) * 1000.0
        start_error = abs(predicted[0] - gold_start) * 1000.0
        end_error = abs(predicted[1] - gold_end) * 1000.0
        duration_error = abs(predicted_duration_ms - gold_duration_ms)
        duration_norm = duration_error / gold_duration_ms if gold_duration_ms > 0.0 else 0.0
        stretch_factor = predicted_duration_ms / gold_duration_ms if gold_duration_ms > 0.0 else 0.0
        stretch_abs_log2 = abs(math.log(stretch_factor, 2.0)) if stretch_factor > 0.0 else 0.0
        words["predicted_ms"] += predicted_duration_ms
        words["matched_reference_ms"] += gold_duration_ms
        words["matched_predicted_ms"] += predicted_duration_ms
        words["overlap_ms"] += overlap_ms
        words["start_error_sum_ms"] += start_error
        words["end_error_sum_ms"] += end_error
        words["duration_error_sum_ms"] += duration_error
        words["duration_l1_all_ms"] += duration_error
        words["start_errors_ms"].append(start_error)
        words["end_errors_ms"].append(end_error)
        words["duration_errors_ms"].append(duration_error)
        words["duration_norm_errors"].append(duration_norm)
        words["stretch_abs_log2_errors"].append(stretch_abs_log2)
        diag.update({
            "predicted_start": predicted[0],
            "predicted_end": predicted[1],
            "predicted_duration_ms": predicted_duration_ms,
            "overlap_ms": overlap_ms,
            "start_error_ms": start_error,
            "end_error_ms": end_error,
            "duration_error_ms": duration_error,
            "duration_error_norm": duration_norm,
            "stretch_factor": stretch_factor,
            "stretch_abs_log2": stretch_abs_log2,
            "missing_predicted": 0,
        })
        word_duration_diag_rows.append(diag)

    for word_index, predicted in sorted(predicted_word_spans.items()):
        if word_index in gold_word_indices:
            continue
        predicted_duration_ms = max(0.0, predicted[1] - predicted[0]) * 1000.0
        words["predicted_ms"] += predicted_duration_ms
        words["extra_predicted_ms"] += predicted_duration_ms
        words["duration_l1_all_ms"] += predicted_duration_ms
        word_duration_diag_rows.append({
            "word_index": word_index,
            "word": "",
            "gold_start": "",
            "gold_end": "",
            "gold_duration_ms": 0.0,
            "predicted_start": predicted[0],
            "predicted_end": predicted[1],
            "predicted_duration_ms": predicted_duration_ms,
            "overlap_ms": 0.0,
            "start_error_ms": "",
            "end_error_ms": "",
            "duration_error_ms": predicted_duration_ms,
            "duration_error_norm": "",
            "stretch_factor": "",
            "stretch_abs_log2": "",
            "missing_predicted": 0,
            "extra_predicted": 1,
        })

    if word_duration_diag_rows:
        diag_path = case_dir / "word_duration_diagnostics.csv"
        with diag_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(word_duration_diag_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(word_duration_diag_rows)

    if onset_rows:
        words["head_start_errors_ms"] = [
            _safe_float(item.get("error_ms"))
            for item in onset_rows
            if str(item.get("missing_event", "")).lower() != "true"
        ]
        words["head_start_error_sum_ms"] = sum(words["head_start_errors_ms"])
        words["head_matched_count"] = len(words["head_start_errors_ms"])

    words["missing_count"] = max(0, words["reference_count"] - words["matched_count"])
    words["coverage_rate"] = words["matched_count"] / max(words["reference_count"], 1)
    if words["predicted_ms"] > 0.0:
        words["precision"] = words["overlap_ms"] / words["predicted_ms"]
    if words["reference_ms"] > 0.0:
        words["recall"] = words["overlap_ms"] / words["reference_ms"]
    if words["precision"] > 0.0 or words["recall"] > 0.0:
        words["f1"] = 2.0 * words["precision"] * words["recall"] / (words["precision"] + words["recall"])
    if words["matched_count"] > 0:
        words["mean_abs_start_error_ms"] = words["start_error_sum_ms"] / words["matched_count"]
        words["mean_abs_end_error_ms"] = words["end_error_sum_ms"] / words["matched_count"]
        words["mean_abs_duration_error_ms"] = words["duration_error_sum_ms"] / words["matched_count"]
        words["median_abs_start_error_ms"] = _median(words["start_errors_ms"])
        words["p90_abs_start_error_ms"] = _percentile(words["start_errors_ms"], 0.90)
        words["median_abs_end_error_ms"] = _median(words["end_errors_ms"])
        words["p90_abs_end_error_ms"] = _percentile(words["end_errors_ms"], 0.90)
        words["median_abs_duration_error_ms"] = _median(words["duration_errors_ms"])
        words["p90_abs_duration_error_ms"] = _percentile(words["duration_errors_ms"], 0.90)
        words["mean_abs_duration_error_norm"] = _mean(words["duration_norm_errors"])
        words["median_abs_duration_error_norm"] = _median(words["duration_norm_errors"])
        words["p90_abs_duration_error_norm"] = _percentile(words["duration_norm_errors"], 0.90)
        words["mean_abs_stretch_log2"] = _mean(words["stretch_abs_log2_errors"])
        words["median_abs_stretch_log2"] = _median(words["stretch_abs_log2_errors"])
        words["p90_abs_stretch_log2"] = _percentile(words["stretch_abs_log2_errors"], 0.90)
    words["duration_l1_norm"] = words["duration_l1_all_ms"] / words["reference_ms"] if words["reference_ms"] > 0.0 else 0.0
    if words["head_matched_count"] > 0:
        words["mean_head_start_error_ms"] = words["head_start_error_sum_ms"] / words["head_matched_count"]
        words["median_head_start_error_ms"] = _median(words["head_start_errors_ms"])
        words["p90_head_start_error_ms"] = _percentile(words["head_start_errors_ms"], 0.90)
    return layers



def load_direct_aligner_grade(case_dir: pathlib.Path):
    path = case_dir / "direct_aligner_grade.json"
    if not path.exists():
        return {
            "available": False,
            "reference_count": 0,
            "matched_count": 0,
            "committed_count": 0,
            "coverage_rate": 0.0,
            "mean_abs_center_error_ms": 0.0,
            "median_abs_center_error_ms": 0.0,
            "p90_abs_center_error_ms": 0.0,
            "mean_abs_start_error_ms": 0.0,
            "mean_abs_end_error_ms": 0.0,
            "mean_abs_duration_error_ms": 0.0,
            "word_onset_alignment": {},
            "intra_word_alignment": {},
        }
    data = json.loads(path.read_text())
    reference = int(data.get("reference_count", 0))
    matched = int(data.get("matched_count", 0))
    data["available"] = bool(data.get("gold_available", True))
    data["coverage_rate"] = matched / max(reference, 1)
    return data

def load_case_grades(root: pathlib.Path, gold_root: pathlib.Path | None = None):
    rows = []
    graded = []
    ungraded = 0
    for grade in sorted(root.glob("*/grade.json")):
        data = json.loads(grade.read_text())
        gold_available = data.get("gold_available", True)
        pause = data.get("pause_alignment", {})
        word_onset = data.get("word_onset_alignment", {})
        intra_word = data.get("intra_word_alignment", {})
        row = {
            "case": grade.parent.name,
            "gold_available": gold_available,
            "reference_count": data.get("reference_count", 0),
            "matched_count": data.get("matched_count", 0),
            "mean_abs_center_error_ms": data.get("mean_abs_center_error_ms", 0.0),
            "median_abs_center_error_ms": data.get("median_abs_center_error_ms", data.get("mean_abs_center_error_ms", 0.0)),
            "p90_abs_center_error_ms": data.get("p90_abs_center_error_ms", 0.0),
            "mean_abs_start_error_ms": data.get("mean_abs_start_error_ms", 0.0),
            "mean_abs_end_error_ms": data.get("mean_abs_end_error_ms", 0.0),
            "mean_abs_duration_error_ms": data.get("mean_abs_duration_error_ms", 0.0),
            "order_violations": data.get("order_violations", 0),
            "pause_alignment": pause,
            "word_onset_alignment": word_onset,
            "intra_word_alignment": intra_word,
            "raw": data,
            "direct_aligner": load_direct_aligner_grade(grade.parent),
        }
        row["layers"] = build_case_layers(
            grade.parent,
            None if gold_root is None else gold_root / grade.parent.name,
            row,
        )
        rows.append(row)
        if gold_available:
            graded.append(row)
        else:
            ungraded += 1
    return rows, graded, ungraded


def _empty_summary(cases=0, graded_cases=0, ungraded_cases=0):
    keys = [
        "matched_count", "reference_count", "match_rate",
        "mean_center_ms", "median_center_ms", "p90_center_ms",
        "mean_start_ms", "mean_end_ms", "mean_duration_ms",
        "mean_sentence_start_ms", "median_sentence_start_ms", "p90_sentence_start_ms",
        "mean_sentence_end_ms", "mean_clause_start_ms", "median_clause_start_ms", "p90_clause_start_ms",
        "mean_clause_end_ms", "mean_word_onset_ms", "median_word_onset_ms", "p90_word_onset_ms",
        "mean_intra_word_ms", "median_intra_word_ms", "p90_intra_word_ms",
        "order_fail_cases", "degenerate_cases", "speech_region_count_mismatch_cases",
        "sentence_region_count_mismatch_cases", "clause_region_count_mismatch_cases",
        "speech_precision", "speech_recall", "speech_f1",
        "speech_boundary_start_ms", "speech_boundary_start_median_ms", "speech_boundary_start_p90_ms",
        "speech_boundary_end_ms", "speech_boundary_end_median_ms", "speech_boundary_end_p90_ms",
        "speech_tail_leakage_ms",
        "phone_occupancy_precision", "phone_occupancy_recall", "phone_occupancy_f1",
        "phone_occupancy_boundary_start_ms", "phone_occupancy_boundary_end_ms",
        "phone_occupancy_tail_leakage_ms",
        "visible_speech_precision", "visible_speech_recall", "visible_speech_f1",
        "visible_speech_boundary_start_ms", "visible_speech_boundary_end_ms",
        "visible_speech_tail_leakage_ms",
        "word_coverage_rate", "word_precision", "word_recall", "word_f1",
        "word_start_ms", "word_start_median_ms", "word_start_p90_ms",
        "word_end_ms", "word_duration_ms", "word_duration_median_ms", "word_duration_p90_ms",
        "word_duration_norm", "word_duration_norm_median", "word_duration_norm_p90",
        "word_duration_l1_norm", "word_missing_duration_ms", "word_extra_duration_ms",
        "word_stretch_abs_log2", "word_stretch_abs_log2_p90",
        "word_head_start_ms", "word_head_start_median_ms", "word_head_start_p90_ms",
        "phoneme_coverage_rate", "phoneme_center_ms", "phoneme_center_median_ms", "phoneme_center_p90_ms",
        "phoneme_start_ms", "phoneme_end_ms", "intra_word_coverage_rate", "intra_word_center_ms",
        "direct_aligner_match_rate", "direct_aligner_center_ms", "direct_aligner_center_median_ms",
        "direct_aligner_center_p90_ms", "direct_aligner_start_ms", "direct_aligner_end_ms",
        "direct_aligner_duration_ms", "direct_aligner_word_onset_ms",
        "direct_aligner_intra_word_coverage_rate", "direct_aligner_intra_word_center_ms",
        "direct_aligner_available_cases",
    ]
    out = {"cases": cases, "graded_cases": graded_cases, "ungraded_cases": ungraded_cases}
    out.update({key: 0.0 for key in keys})
    out["matched_count"] = 0
    out["reference_count"] = 0
    out["order_fail_cases"] = 0
    out["degenerate_cases"] = 0
    out["speech_region_count_mismatch_cases"] = 0
    out["phone_occupancy_region_count_mismatch_cases"] = 0
    out["visible_speech_region_count_mismatch_cases"] = 0
    out["sentence_region_count_mismatch_cases"] = 0
    out["clause_region_count_mismatch_cases"] = 0
    return out



def aggregate_speech_layer(graded, layer_name):
    reference_ms = sum(row["layers"].get(layer_name, empty_speech_layer())["reference_ms"] for row in graded)
    predicted_ms = sum(row["layers"].get(layer_name, empty_speech_layer())["predicted_ms"] for row in graded)
    overlap_ms = sum(row["layers"].get(layer_name, empty_speech_layer())["overlap_ms"] for row in graded)
    matched = sum(row["layers"].get(layer_name, empty_speech_layer())["matched_count"] for row in graded)
    start_errors = [v for row in graded for v in row["layers"].get(layer_name, empty_speech_layer()).get("start_errors_ms", [])]
    end_errors = [v for row in graded for v in row["layers"].get(layer_name, empty_speech_layer()).get("end_errors_ms", [])]
    precision = overlap_ms / predicted_ms if predicted_ms > 0.0 else 0.0
    recall = overlap_ms / reference_ms if reference_ms > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision > 0.0 or recall > 0.0) else 0.0
    return {
        "reference_ms": reference_ms,
        "predicted_ms": predicted_ms,
        "overlap_ms": overlap_ms,
        "matched_count": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "boundary_start_ms": _mean(start_errors),
        "boundary_start_median_ms": _median(start_errors),
        "boundary_start_p90_ms": _percentile(start_errors, 0.90),
        "boundary_end_ms": _mean(end_errors),
        "boundary_end_median_ms": _median(end_errors),
        "boundary_end_p90_ms": _percentile(end_errors, 0.90),
        "tail_leakage_ms": _mean(
            row["layers"].get(layer_name, empty_speech_layer()).get("tail_leakage_ms", 0.0)
            for row in graded
        ),
        "count_mismatch_cases": sum(1 for row in graded if row["layers"].get(layer_name, empty_speech_layer()).get("count_mismatch", False)),
    }


def compute_summary(rows, graded, ungraded):
    if not rows:
        return _empty_summary()
    if not graded:
        return _empty_summary(len(rows), 0, ungraded)

    total_matched = sum(row["matched_count"] for row in graded)
    total_reference = sum(row["reference_count"] for row in graded)
    graded_count = len(graded)

    speech_stats = aggregate_speech_layer(graded, "speech")
    phone_occupancy_stats = aggregate_speech_layer(graded, "phone_occupancy")
    visible_speech_stats = aggregate_speech_layer(graded, "speech_visible")

    word_reference_total = sum(row["layers"]["words"]["reference_count"] for row in graded)
    word_matched_total = sum(row["layers"]["words"]["matched_count"] for row in graded)
    word_reference_ms = sum(row["layers"]["words"]["reference_ms"] for row in graded)
    word_predicted_ms = sum(row["layers"]["words"]["predicted_ms"] for row in graded)
    word_overlap_ms = sum(row["layers"]["words"]["overlap_ms"] for row in graded)
    word_start_sum_ms = sum(row["layers"]["words"]["start_error_sum_ms"] for row in graded)
    word_end_sum_ms = sum(row["layers"]["words"]["end_error_sum_ms"] for row in graded)
    word_duration_sum_ms = sum(row["layers"]["words"]["duration_error_sum_ms"] for row in graded)
    word_duration_l1_all_ms = sum(row["layers"]["words"].get("duration_l1_all_ms", 0.0) for row in graded)
    word_missing_reference_ms = sum(row["layers"]["words"].get("missing_reference_ms", 0.0) for row in graded)
    word_extra_predicted_ms = sum(row["layers"]["words"].get("extra_predicted_ms", 0.0) for row in graded)
    word_head_matched_total = sum(row["layers"]["words"]["head_matched_count"] for row in graded)
    word_head_start_sum_ms = sum(row["layers"]["words"]["head_start_error_sum_ms"] for row in graded)

    phoneme_matched_total = sum(row["layers"]["phonemes"]["matched_count"] for row in graded)
    phoneme_start_sum_ms = sum(row["layers"]["phonemes"]["start_error_sum_ms"] for row in graded)
    phoneme_end_sum_ms = sum(row["layers"]["phonemes"]["end_error_sum_ms"] for row in graded)
    intra_reference_total = sum(row["layers"]["phonemes"]["intra_reference_count"] for row in graded)
    intra_matched_total = sum(row["layers"]["phonemes"]["intra_matched_count"] for row in graded)
    intra_center_sum_ms = sum(row["layers"]["phonemes"]["intra_center_error_sum_ms"] for row in graded)

    word_precision = word_overlap_ms / word_predicted_ms if word_predicted_ms > 0.0 else 0.0
    word_recall = word_overlap_ms / word_reference_ms if word_reference_ms > 0.0 else 0.0
    word_f1 = 2.0 * word_precision * word_recall / (word_precision + word_recall) if (word_precision > 0.0 or word_recall > 0.0) else 0.0

    word_start_errors = [v for row in graded for v in row["layers"]["words"]["start_errors_ms"]]
    word_duration_errors = [v for row in graded for v in row["layers"]["words"].get("duration_errors_ms", [])]
    word_duration_norm_errors = [v for row in graded for v in row["layers"]["words"].get("duration_norm_errors", [])]
    word_stretch_abs_log2_errors = [v for row in graded for v in row["layers"]["words"].get("stretch_abs_log2_errors", [])]
    word_head_errors = [v for row in graded for v in row["layers"]["words"]["head_start_errors_ms"]]

    direct_available = [row.get("direct_aligner", {}) for row in graded if row.get("direct_aligner", {}).get("available", False)]
    direct_reference_total = sum(int(item.get("reference_count", 0)) for item in direct_available)
    direct_matched_total = sum(int(item.get("matched_count", 0)) for item in direct_available)
    direct_intra_reference_total = sum(int(item.get("intra_word_alignment", {}).get("reference_count", 0)) for item in direct_available)
    direct_intra_matched_total = sum(int(item.get("intra_word_alignment", {}).get("matched_count", 0)) for item in direct_available)
    direct_intra_center_sum_ms = sum(
        float(item.get("intra_word_alignment", {}).get("mean_abs_center_error_ms", 0.0))
        * int(item.get("intra_word_alignment", {}).get("matched_count", 0))
        for item in direct_available
    )

    return {
        "cases": len(rows),
        "graded_cases": graded_count,
        "ungraded_cases": ungraded,
        "matched_count": total_matched,
        "reference_count": total_reference,
        "match_rate": total_matched / max(total_reference, 1),
        "mean_center_ms": sum(row["mean_abs_center_error_ms"] for row in graded) / graded_count,
        "median_center_ms": _median(row["median_abs_center_error_ms"] for row in graded),
        "p90_center_ms": _percentile((row["p90_abs_center_error_ms"] for row in graded), 0.90),
        "mean_start_ms": sum(row["mean_abs_start_error_ms"] for row in graded) / graded_count,
        "mean_end_ms": sum(row["mean_abs_end_error_ms"] for row in graded) / graded_count,
        "mean_duration_ms": sum(row["mean_abs_duration_error_ms"] for row in graded) / graded_count,
        "mean_sentence_start_ms": _mean(row["pause_alignment"].get("mean_abs_sentence_region_start_error_ms", 0.0) for row in graded),
        "median_sentence_start_ms": _median(row["pause_alignment"].get("median_abs_sentence_region_start_error_ms", row["pause_alignment"].get("mean_abs_sentence_region_start_error_ms", 0.0)) for row in graded),
        "p90_sentence_start_ms": _percentile((row["pause_alignment"].get("p90_abs_sentence_region_start_error_ms", row["pause_alignment"].get("mean_abs_sentence_region_start_error_ms", 0.0)) for row in graded), 0.90),
        "mean_sentence_end_ms": _mean(row["pause_alignment"].get("mean_abs_sentence_region_end_error_ms", 0.0) for row in graded),
        "mean_clause_start_ms": _mean(row["pause_alignment"].get("mean_abs_clause_region_start_error_ms", 0.0) for row in graded),
        "median_clause_start_ms": _median(row["pause_alignment"].get("median_abs_clause_region_start_error_ms", row["pause_alignment"].get("mean_abs_clause_region_start_error_ms", 0.0)) for row in graded),
        "p90_clause_start_ms": _percentile((row["pause_alignment"].get("p90_abs_clause_region_start_error_ms", row["pause_alignment"].get("mean_abs_clause_region_start_error_ms", 0.0)) for row in graded), 0.90),
        "mean_clause_end_ms": _mean(row["pause_alignment"].get("mean_abs_clause_region_end_error_ms", 0.0) for row in graded),
        "mean_word_onset_ms": _mean(row["word_onset_alignment"].get("mean_abs_start_error_ms", 0.0) for row in graded),
        "median_word_onset_ms": _median(row["word_onset_alignment"].get("median_abs_start_error_ms", row["word_onset_alignment"].get("mean_abs_start_error_ms", 0.0)) for row in graded),
        "p90_word_onset_ms": _percentile((row["word_onset_alignment"].get("p90_abs_start_error_ms", row["word_onset_alignment"].get("mean_abs_start_error_ms", 0.0)) for row in graded), 0.90),
        "mean_intra_word_ms": _mean(row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0) for row in graded),
        "median_intra_word_ms": _median(row["intra_word_alignment"].get("median_abs_center_error_ms", row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0)) for row in graded),
        "p90_intra_word_ms": _percentile((row["intra_word_alignment"].get("p90_abs_center_error_ms", row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0)) for row in graded), 0.90),
        "order_fail_cases": sum(1 for row in graded if row["order_violations"] > 0),
        "degenerate_cases": sum(1 for row in graded if row["reference_count"] > 0 and row["matched_count"] == 0),
        "speech_region_count_mismatch_cases": speech_stats["count_mismatch_cases"],
        "phone_occupancy_region_count_mismatch_cases": phone_occupancy_stats["count_mismatch_cases"],
        "visible_speech_region_count_mismatch_cases": visible_speech_stats["count_mismatch_cases"],
        "sentence_region_count_mismatch_cases": sum(1 for row in graded if row["pause_alignment"].get("sentence_region_count_mismatch", False)),
        "clause_region_count_mismatch_cases": sum(1 for row in graded if row["pause_alignment"].get("clause_region_count_mismatch", False)),
        # Primary speech metrics come from detector-owned speech occupancy.
        "speech_precision": speech_stats["precision"],
        "speech_recall": speech_stats["recall"],
        "speech_f1": speech_stats["f1"],
        "speech_boundary_start_ms": speech_stats["boundary_start_ms"],
        "speech_boundary_start_median_ms": speech_stats["boundary_start_median_ms"],
        "speech_boundary_start_p90_ms": speech_stats["boundary_start_p90_ms"],
        "speech_boundary_end_ms": speech_stats["boundary_end_ms"],
        "speech_boundary_end_median_ms": speech_stats["boundary_end_median_ms"],
        "speech_boundary_end_p90_ms": speech_stats["boundary_end_p90_ms"],
        "speech_tail_leakage_ms": speech_stats["tail_leakage_ms"],
        # Diagnostic occupancy layers.
        "phone_occupancy_precision": phone_occupancy_stats["precision"],
        "phone_occupancy_recall": phone_occupancy_stats["recall"],
        "phone_occupancy_f1": phone_occupancy_stats["f1"],
        "phone_occupancy_boundary_start_ms": phone_occupancy_stats["boundary_start_ms"],
        "phone_occupancy_boundary_end_ms": phone_occupancy_stats["boundary_end_ms"],
        "phone_occupancy_tail_leakage_ms": phone_occupancy_stats["tail_leakage_ms"],
        "visible_speech_precision": visible_speech_stats["precision"],
        "visible_speech_recall": visible_speech_stats["recall"],
        "visible_speech_f1": visible_speech_stats["f1"],
        "visible_speech_boundary_start_ms": visible_speech_stats["boundary_start_ms"],
        "visible_speech_boundary_end_ms": visible_speech_stats["boundary_end_ms"],
        "visible_speech_tail_leakage_ms": visible_speech_stats["tail_leakage_ms"],
        "word_coverage_rate": word_matched_total / max(word_reference_total, 1),
        "word_precision": word_precision,
        "word_recall": word_recall,
        "word_f1": word_f1,
        "word_start_ms": word_start_sum_ms / max(word_matched_total, 1),
        "word_start_median_ms": _median(word_start_errors),
        "word_start_p90_ms": _percentile(word_start_errors, 0.90),
        "word_end_ms": word_end_sum_ms / max(word_matched_total, 1),
        "word_duration_ms": word_duration_sum_ms / max(word_matched_total, 1),
        "word_duration_median_ms": _median(word_duration_errors),
        "word_duration_p90_ms": _percentile(word_duration_errors, 0.90),
        "word_duration_norm": _mean(word_duration_norm_errors),
        "word_duration_norm_median": _median(word_duration_norm_errors),
        "word_duration_norm_p90": _percentile(word_duration_norm_errors, 0.90),
        "word_duration_l1_norm": word_duration_l1_all_ms / max(word_reference_ms, 1.0),
        "word_missing_duration_ms": word_missing_reference_ms,
        "word_extra_duration_ms": word_extra_predicted_ms,
        "word_stretch_abs_log2": _mean(word_stretch_abs_log2_errors),
        "word_stretch_abs_log2_p90": _percentile(word_stretch_abs_log2_errors, 0.90),
        "word_head_start_ms": word_head_start_sum_ms / max(word_head_matched_total, 1),
        "word_head_start_median_ms": _median(word_head_errors),
        "word_head_start_p90_ms": _percentile(word_head_errors, 0.90),
        "phoneme_coverage_rate": total_matched / max(total_reference, 1),
        "phoneme_center_ms": sum(row["layers"]["phonemes"]["center_error_sum_ms"] for row in graded) / max(phoneme_matched_total, 1),
        "phoneme_center_median_ms": _median(row["layers"]["phonemes"].get("median_abs_center_error_ms", 0.0) for row in graded),
        "phoneme_center_p90_ms": _percentile((row["layers"]["phonemes"].get("p90_abs_center_error_ms", 0.0) for row in graded), 0.90),
        "phoneme_start_ms": phoneme_start_sum_ms / max(phoneme_matched_total, 1),
        "phoneme_end_ms": phoneme_end_sum_ms / max(phoneme_matched_total, 1),
        "intra_word_coverage_rate": intra_matched_total / max(intra_reference_total, 1),
        "intra_word_center_ms": intra_center_sum_ms / max(intra_matched_total, 1),
        "direct_aligner_available_cases": len(direct_available),
        "direct_aligner_match_rate": direct_matched_total / max(direct_reference_total, 1),
        "direct_aligner_center_ms": (
            sum(float(item.get("mean_abs_center_error_ms", 0.0)) * int(item.get("matched_count", 0)) for item in direct_available)
            / max(direct_matched_total, 1)
        ),
        "direct_aligner_center_median_ms": _median(item.get("median_abs_center_error_ms", 0.0) for item in direct_available),
        "direct_aligner_center_p90_ms": _percentile((item.get("p90_abs_center_error_ms", 0.0) for item in direct_available), 0.90),
        "direct_aligner_start_ms": (
            sum(float(item.get("mean_abs_start_error_ms", 0.0)) * int(item.get("matched_count", 0)) for item in direct_available)
            / max(direct_matched_total, 1)
        ),
        "direct_aligner_end_ms": (
            sum(float(item.get("mean_abs_end_error_ms", 0.0)) * int(item.get("matched_count", 0)) for item in direct_available)
            / max(direct_matched_total, 1)
        ),
        "direct_aligner_duration_ms": (
            sum(float(item.get("mean_abs_duration_error_ms", 0.0)) * int(item.get("matched_count", 0)) for item in direct_available)
            / max(direct_matched_total, 1)
        ),
        "direct_aligner_word_onset_ms": _mean(
            item.get("word_onset_alignment", {}).get("mean_abs_start_error_ms", 0.0)
            for item in direct_available
        ),
        "direct_aligner_intra_word_coverage_rate": direct_intra_matched_total / max(direct_intra_reference_total, 1),
        "direct_aligner_intra_word_center_ms": direct_intra_center_sum_ms / max(direct_intra_matched_total, 1),
    }
