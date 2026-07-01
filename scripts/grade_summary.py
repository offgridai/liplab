import json
import pathlib
import csv


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


def build_case_layers(case_dir: pathlib.Path, gold_case_dir: pathlib.Path | None, row):
    layers = {
        "speech": {
            "reference_count": 0,
            "predicted_count": 0,
            "matched_count": 0,
            "reference_ms": 0.0,
            "predicted_ms": 0.0,
            "overlap_ms": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "region_start_error_sum_ms": 0.0,
            "region_end_error_sum_ms": 0.0,
        },
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
            "mean_abs_start_error_ms": 0.0,
            "mean_abs_end_error_ms": 0.0,
            "mean_abs_duration_error_ms": 0.0,
            "head_reference_count": 0,
            "head_matched_count": 0,
            "head_start_error_sum_ms": 0.0,
            "mean_head_start_error_ms": 0.0,
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
            "mean_abs_start_error_ms": row["mean_abs_start_error_ms"],
            "mean_abs_end_error_ms": row["mean_abs_end_error_ms"],
            "intra_reference_count": row["intra_word_alignment"].get("reference_count", 0),
            "intra_matched_count": row["intra_word_alignment"].get("matched_count", 0),
            "intra_missing_count": row["intra_word_alignment"].get("missing_count", 0),
            "intra_extra_count": row["intra_word_alignment"].get("extra_count", 0),
            "intra_coverage_rate": row["intra_word_alignment"].get("matched_count", 0) / max(row["intra_word_alignment"].get("reference_count", 0), 1),
            "intra_center_error_sum_ms": row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0) * row["intra_word_alignment"].get("matched_count", 0),
            "intra_mean_abs_center_error_ms": row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0),
        },
    }

    pause = row["pause_alignment"]
    speech = layers["speech"]
    speech["reference_count"] = pause.get("speech_region_reference_count", 0)
    speech["predicted_count"] = pause.get("speech_region_predicted_count", 0)
    speech["matched_count"] = pause.get("speech_region_matched_count", 0)
    speech["region_start_error_sum_ms"] = pause.get("mean_abs_speech_region_start_error_ms", 0.0) * speech["matched_count"]
    speech["region_end_error_sum_ms"] = pause.get("mean_abs_speech_region_end_error_ms", 0.0) * speech["matched_count"]

    words = layers["words"]
    word_onset = row["word_onset_alignment"]
    words["head_reference_count"] = word_onset.get("reference_count", 0)
    words["head_matched_count"] = word_onset.get("matched_count", 0)
    words["head_start_error_sum_ms"] = word_onset.get("mean_abs_start_error_ms", 0.0) * words["head_matched_count"]
    if words["head_matched_count"] > 0:
        words["mean_head_start_error_ms"] = words["head_start_error_sum_ms"] / words["head_matched_count"]

    if gold_case_dir is None or not gold_case_dir.exists():
        return layers

    gold_speech_rows = read_csv_rows(gold_case_dir / "speech.csv")
    gold_word_rows = read_csv_rows(gold_case_dir / "words.csv")
    committed_rows = read_csv_rows(case_dir / "committed.csv")

    gold_speech = [
        (float(item["start"]), float(item["end"]))
        for item in gold_speech_rows
    ]
    predicted_speech = merge_spans([
        (float(item["start"]), float(item["end"]))
        for item in committed_rows
    ])

    speech["reference_ms"] = span_duration_seconds(gold_speech) * 1000.0
    speech["predicted_ms"] = span_duration_seconds(predicted_speech) * 1000.0
    speech["overlap_ms"] = overlap_duration_seconds(gold_speech, predicted_speech) * 1000.0
    if speech["predicted_ms"] > 0.0:
        speech["precision"] = speech["overlap_ms"] / speech["predicted_ms"]
    if speech["reference_ms"] > 0.0:
        speech["recall"] = speech["overlap_ms"] / speech["reference_ms"]
    if speech["precision"] > 0.0 or speech["recall"] > 0.0:
        speech["f1"] = 2.0 * speech["precision"] * speech["recall"] / (speech["precision"] + speech["recall"])

    predicted_word_spans = {}
    for item in committed_rows:
        word_index = int(item["word_index"])
        if word_index < 0:
            continue
        start = float(item["start"])
        end = float(item["end"])
        if word_index not in predicted_word_spans:
            predicted_word_spans[word_index] = [start, end]
        else:
            predicted_word_spans[word_index][0] = min(predicted_word_spans[word_index][0], start)
            predicted_word_spans[word_index][1] = max(predicted_word_spans[word_index][1], end)

    words["reference_count"] = len(gold_word_rows)
    for item in gold_word_rows:
        word_index = int(item["word_index"])
        gold_start = float(item["start"])
        gold_end = float(item["end"])
        words["reference_ms"] += max(0.0, gold_end - gold_start) * 1000.0
        predicted = predicted_word_spans.get(word_index)
        if predicted is None:
            continue
        words["matched_count"] += 1
        predicted_duration_ms = max(0.0, predicted[1] - predicted[0]) * 1000.0
        overlap_ms = max(0.0, min(predicted[1], gold_end) - max(predicted[0], gold_start)) * 1000.0
        words["predicted_ms"] += predicted_duration_ms
        words["overlap_ms"] += overlap_ms
        words["start_error_sum_ms"] += abs(predicted[0] - gold_start) * 1000.0
        words["end_error_sum_ms"] += abs(predicted[1] - gold_end) * 1000.0
        words["duration_error_sum_ms"] += abs((predicted[1] - predicted[0]) - (gold_end - gold_start)) * 1000.0

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
    return layers


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
            "p90_abs_center_error_ms": data.get("p90_abs_center_error_ms", 0.0),
            "mean_abs_start_error_ms": data.get("mean_abs_start_error_ms", 0.0),
            "mean_abs_end_error_ms": data.get("mean_abs_end_error_ms", 0.0),
            "mean_abs_duration_error_ms": data.get("mean_abs_duration_error_ms", 0.0),
            "order_violations": data.get("order_violations", 0),
            "pause_alignment": pause,
            "word_onset_alignment": word_onset,
            "intra_word_alignment": intra_word,
            "raw": data,
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


def compute_summary(rows, graded, ungraded):
    if not rows:
        return {
            "cases": 0,
            "graded_cases": 0,
            "ungraded_cases": 0,
            "matched_count": 0,
            "reference_count": 0,
            "match_rate": 0.0,
            "mean_center_ms": 0.0,
            "mean_start_ms": 0.0,
            "mean_end_ms": 0.0,
            "mean_duration_ms": 0.0,
            "mean_sentence_start_ms": 0.0,
            "mean_sentence_end_ms": 0.0,
            "mean_clause_start_ms": 0.0,
            "mean_clause_end_ms": 0.0,
            "mean_word_onset_ms": 0.0,
            "mean_intra_word_ms": 0.0,
            "order_fail_cases": 0,
            "speech_precision": 0.0,
            "speech_recall": 0.0,
            "speech_f1": 0.0,
            "speech_boundary_start_ms": 0.0,
            "speech_boundary_end_ms": 0.0,
            "word_coverage_rate": 0.0,
            "word_precision": 0.0,
            "word_recall": 0.0,
            "word_f1": 0.0,
            "word_start_ms": 0.0,
            "word_end_ms": 0.0,
            "word_duration_ms": 0.0,
            "word_head_start_ms": 0.0,
            "phoneme_coverage_rate": 0.0,
            "phoneme_center_ms": 0.0,
            "phoneme_start_ms": 0.0,
            "phoneme_end_ms": 0.0,
            "intra_word_coverage_rate": 0.0,
            "intra_word_center_ms": 0.0,
        }

    if not graded:
        return {
            "cases": len(rows),
            "graded_cases": 0,
            "ungraded_cases": ungraded,
            "matched_count": 0,
            "reference_count": 0,
            "match_rate": 0.0,
            "mean_center_ms": 0.0,
            "mean_start_ms": 0.0,
            "mean_end_ms": 0.0,
            "mean_duration_ms": 0.0,
            "mean_sentence_start_ms": 0.0,
            "mean_sentence_end_ms": 0.0,
            "mean_clause_start_ms": 0.0,
            "mean_clause_end_ms": 0.0,
            "mean_word_onset_ms": 0.0,
            "mean_intra_word_ms": 0.0,
            "order_fail_cases": 0,
            "speech_precision": 0.0,
            "speech_recall": 0.0,
            "speech_f1": 0.0,
            "speech_boundary_start_ms": 0.0,
            "speech_boundary_end_ms": 0.0,
            "word_coverage_rate": 0.0,
            "word_precision": 0.0,
            "word_recall": 0.0,
            "word_f1": 0.0,
            "word_start_ms": 0.0,
            "word_end_ms": 0.0,
            "word_duration_ms": 0.0,
            "word_head_start_ms": 0.0,
            "phoneme_coverage_rate": 0.0,
            "phoneme_center_ms": 0.0,
            "phoneme_start_ms": 0.0,
            "phoneme_end_ms": 0.0,
            "intra_word_coverage_rate": 0.0,
            "intra_word_center_ms": 0.0,
        }

    total_matched = sum(row["matched_count"] for row in graded)
    total_reference = sum(row["reference_count"] for row in graded)
    graded_count = len(graded)
    speech_reference_ms = sum(row["layers"]["speech"]["reference_ms"] for row in graded)
    speech_predicted_ms = sum(row["layers"]["speech"]["predicted_ms"] for row in graded)
    speech_overlap_ms = sum(row["layers"]["speech"]["overlap_ms"] for row in graded)
    speech_region_matched = sum(row["layers"]["speech"]["matched_count"] for row in graded)
    speech_start_sum_ms = sum(row["layers"]["speech"]["region_start_error_sum_ms"] for row in graded)
    speech_end_sum_ms = sum(row["layers"]["speech"]["region_end_error_sum_ms"] for row in graded)
    word_reference_total = sum(row["layers"]["words"]["reference_count"] for row in graded)
    word_matched_total = sum(row["layers"]["words"]["matched_count"] for row in graded)
    word_reference_ms = sum(row["layers"]["words"]["reference_ms"] for row in graded)
    word_predicted_ms = sum(row["layers"]["words"]["predicted_ms"] for row in graded)
    word_overlap_ms = sum(row["layers"]["words"]["overlap_ms"] for row in graded)
    word_start_sum_ms = sum(row["layers"]["words"]["start_error_sum_ms"] for row in graded)
    word_end_sum_ms = sum(row["layers"]["words"]["end_error_sum_ms"] for row in graded)
    word_duration_sum_ms = sum(row["layers"]["words"]["duration_error_sum_ms"] for row in graded)
    word_head_matched_total = sum(row["layers"]["words"]["head_matched_count"] for row in graded)
    word_head_start_sum_ms = sum(row["layers"]["words"]["head_start_error_sum_ms"] for row in graded)
    phoneme_matched_total = sum(row["layers"]["phonemes"]["matched_count"] for row in graded)
    phoneme_start_sum_ms = sum(row["layers"]["phonemes"]["start_error_sum_ms"] for row in graded)
    phoneme_end_sum_ms = sum(row["layers"]["phonemes"]["end_error_sum_ms"] for row in graded)
    intra_reference_total = sum(row["layers"]["phonemes"]["intra_reference_count"] for row in graded)
    intra_matched_total = sum(row["layers"]["phonemes"]["intra_matched_count"] for row in graded)
    intra_center_sum_ms = sum(row["layers"]["phonemes"]["intra_center_error_sum_ms"] for row in graded)
    speech_precision = speech_overlap_ms / speech_predicted_ms if speech_predicted_ms > 0.0 else 0.0
    speech_recall = speech_overlap_ms / speech_reference_ms if speech_reference_ms > 0.0 else 0.0
    speech_f1 = 0.0
    if speech_precision > 0.0 or speech_recall > 0.0:
        speech_f1 = 2.0 * speech_precision * speech_recall / (speech_precision + speech_recall)
    word_precision = word_overlap_ms / word_predicted_ms if word_predicted_ms > 0.0 else 0.0
    word_recall = word_overlap_ms / word_reference_ms if word_reference_ms > 0.0 else 0.0
    word_f1 = 0.0
    if word_precision > 0.0 or word_recall > 0.0:
        word_f1 = 2.0 * word_precision * word_recall / (word_precision + word_recall)
    return {
        "cases": len(rows),
        "graded_cases": graded_count,
        "ungraded_cases": ungraded,
        "matched_count": total_matched,
        "reference_count": total_reference,
        "match_rate": total_matched / max(total_reference, 1),
        "mean_center_ms": sum(row["mean_abs_center_error_ms"] for row in graded) / graded_count,
        "mean_start_ms": sum(row["mean_abs_start_error_ms"] for row in graded) / graded_count,
        "mean_end_ms": sum(row["mean_abs_end_error_ms"] for row in graded) / graded_count,
        "mean_duration_ms": sum(row["mean_abs_duration_error_ms"] for row in graded) / graded_count,
        "mean_sentence_start_ms": sum(row["pause_alignment"].get("mean_abs_sentence_region_start_error_ms", 0.0) for row in graded) / graded_count,
        "mean_sentence_end_ms": sum(row["pause_alignment"].get("mean_abs_sentence_region_end_error_ms", 0.0) for row in graded) / graded_count,
        "mean_clause_start_ms": sum(row["pause_alignment"].get("mean_abs_clause_region_start_error_ms", 0.0) for row in graded) / graded_count,
        "mean_clause_end_ms": sum(row["pause_alignment"].get("mean_abs_clause_region_end_error_ms", 0.0) for row in graded) / graded_count,
        "mean_word_onset_ms": sum(row["word_onset_alignment"].get("mean_abs_start_error_ms", 0.0) for row in graded) / graded_count,
        "mean_intra_word_ms": sum(row["intra_word_alignment"].get("mean_abs_center_error_ms", 0.0) for row in graded) / graded_count,
        "order_fail_cases": sum(1 for row in graded if row["order_violations"] > 0),
        "speech_precision": speech_precision,
        "speech_recall": speech_recall,
        "speech_f1": speech_f1,
        "speech_boundary_start_ms": sum(row["pause_alignment"].get("mean_abs_sentence_region_start_error_ms", 0.0) for row in graded) / graded_count,
        "speech_boundary_end_ms": sum(row["pause_alignment"].get("mean_abs_sentence_region_end_error_ms", 0.0) for row in graded) / graded_count,
        "word_coverage_rate": word_matched_total / max(word_reference_total, 1),
        "word_precision": word_precision,
        "word_recall": word_recall,
        "word_f1": word_f1,
        "word_start_ms": word_start_sum_ms / max(word_matched_total, 1),
        "word_end_ms": word_end_sum_ms / max(word_matched_total, 1),
        "word_duration_ms": word_duration_sum_ms / max(word_matched_total, 1),
        "word_head_start_ms": word_head_start_sum_ms / max(word_head_matched_total, 1),
        "phoneme_coverage_rate": total_matched / max(total_reference, 1),
        "phoneme_center_ms": sum(row["layers"]["phonemes"]["center_error_sum_ms"] for row in graded) / max(phoneme_matched_total, 1),
        "phoneme_start_ms": phoneme_start_sum_ms / max(phoneme_matched_total, 1),
        "phoneme_end_ms": phoneme_end_sum_ms / max(phoneme_matched_total, 1),
        "intra_word_coverage_rate": intra_matched_total / max(intra_reference_total, 1),
        "intra_word_center_ms": intra_center_sum_ms / max(intra_matched_total, 1),
    }
