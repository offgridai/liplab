import csv
import json
import pathlib
import statistics
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(0.9 * (len(ordered) - 1))]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _nested(dct: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = dct
    try:
        for key in keys:
            cur = cur[key]
        return float(cur)
    except Exception:
        return default


def _flag(dct: dict[str, Any], *keys: str, default: bool = False) -> bool:
    cur: Any = dct
    try:
        for key in keys:
            cur = cur[key]
    except Exception:
        return default
    if isinstance(cur, bool):
        return cur
    if isinstance(cur, (int, float)):
        return bool(cur)
    if isinstance(cur, str):
        return cur.strip().lower() in {"1", "true", "yes"}
    return bool(cur)


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(value)
    except Exception:
        return default


def _decode_boundary_mark(value: str) -> str:
    if value == "":
        return ""
    try:
        codepoint = int(value)
        return chr(codepoint) if codepoint > 0 else ""
    except Exception:
        return value


def _normalize_word(word: str) -> str:
    return (word or "").strip().lower()


def _transcript_landmark_type_for_phone_base(phone_base: str) -> str | None:
    base = (phone_base or "").strip().upper()
    if base in {"M", "B", "P"}:
        return "mbp"
    if base in {"F", "V"}:
        return "fv"
    if base == "W":
        return "w"
    if base in {"CH", "JH", "SH", "ZH"}:
        return "chjjsh"
    if base in {"UW", "UH", "OW", "OY", "AO"}:
        return "round"
    return None


def _gold_pause_class(boundary_row: dict[str, str]) -> str:
    pause_class = (boundary_row.get("pause_class") or "").strip().lower()
    split_applied = (boundary_row.get("split_applied") or "").strip() not in {"", "0", "false", "False"}
    if split_applied or pause_class in {"hard_pause", "region_break"}:
        return "hard_break_pause"
    if pause_class == "soft_pause":
        return "soft_list_pause"
    return "none"


def _build_gold_landmark_refs(
    gold_phones: list[dict[str, str]],
    gold_words: list[dict[str, str]],
    gold_boundaries: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    refs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for ordinal, row in enumerate(gold_phones):
        phone_index = _safe_int(row, "global_phone_index", ordinal)
        phone_base = "".join(ch for ch in (row.get("phone") or "") if not ch.isdigit())
        landmark_type = _transcript_landmark_type_for_phone_base(phone_base)
        if landmark_type is None or phone_index < 0:
            continue
        start = _safe_float(row, "start")
        end = _safe_float(row, "end")
        refs[(landmark_type, "phone", phone_index)] = {
            "type": landmark_type,
            "kind": "phone",
            "index": phone_index,
            "center": 0.5 * (start + end),
            "start": start,
            "end": end,
        }

    words_by_index = {_safe_int(row, "word_index"): row for row in gold_words}
    for row in gold_boundaries:
        word_index = _safe_int(row, "word_index")
        next_word_index = _safe_int(row, "next_word_index")
        if word_index < 0 or next_word_index < 0:
            continue
        if _gold_pause_class(row) == "none":
            continue
        left = words_by_index.get(word_index)
        right = words_by_index.get(next_word_index)
        if not left or not right:
            continue
        start = _safe_float(left, "end")
        end = _safe_float(right, "start")
        if end <= start:
            continue
        refs[("pause_lull", "boundary", word_index)] = {
            "type": "pause_lull",
            "kind": "boundary",
            "index": word_index,
            "center": 0.5 * (start + end),
            "start": start,
            "end": end,
        }
    return refs


def _compute_text_plan_grade(case_dir: pathlib.Path, gold_case_dir: pathlib.Path) -> dict[str, Any]:
    planned_words = _read_csv_rows(case_dir / "planned_words.csv")
    planned_boundaries = _read_csv_rows(case_dir / "planned_boundaries.csv")
    expected_phones = _read_csv_rows(case_dir / "expected_phones.csv")
    transcript_landmarks = _read_csv_rows(case_dir / "transcript_landmarks.csv")
    gold_words = _read_csv_rows(gold_case_dir / "words.csv")
    gold_phones = _read_csv_rows(gold_case_dir / "phones.csv")
    gold_boundaries = _read_csv_rows(gold_case_dir / "boundaries.csv")

    if not planned_words or not expected_phones or not gold_words or not gold_phones or not gold_boundaries:
        return {"available": False}

    pred_words = sorted(planned_words, key=lambda row: _safe_int(row, "word_index"))
    ref_words = sorted(gold_words, key=lambda row: _safe_int(row, "word_index"))
    pred_phones = sorted(expected_phones, key=lambda row: _safe_int(row, "phone_index"))
    ref_phones = list(gold_phones)
    pred_boundaries = sorted(planned_boundaries, key=lambda row: _safe_int(row, "word_index"))
    ref_boundaries = sorted(gold_boundaries, key=lambda row: _safe_int(row, "word_index"))

    phone_prior_centers: list[float] = []
    word_prior_starts: dict[int, float] = {}
    cursor = 0.0
    for index, phone in enumerate(pred_phones):
        word_index = _safe_int(phone, "word_index")
        duration = max(_safe_float(phone, "weight_seconds") * 0.90, 0.018)
        start = cursor
        end = start + duration
        center = 0.5 * (start + end)
        phone_prior_centers.append(center)
        if word_index not in word_prior_starts:
            word_prior_starts[word_index] = start
        cursor = end

        next_word_index = _safe_int(pred_phones[index + 1], "word_index") if index + 1 < len(pred_phones) else None
        if next_word_index is not None and next_word_index != word_index:
            boundary_row = next((row for row in pred_boundaries if _safe_int(row, "word_index") == word_index), None)
            pause_seconds = _safe_float(boundary_row or {}, "pause_seconds", 0.0)
            cursor += 0.020 + pause_seconds

    pred_word_sequence = [_normalize_word(row.get("word", "")) for row in pred_words]
    ref_word_sequence = [_normalize_word(row.get("word", "")) for row in ref_words]

    word_exact_matches = sum(
        1
        for pred, ref in zip(pred_words, ref_words)
        if _normalize_word(pred.get("word", "")) == _normalize_word(ref.get("word", ""))
    )
    phone_exact_matches = sum(
        1
        for pred, ref in zip(pred_phones, ref_phones)
        if (pred.get("phone", "") == ref.get("phone", ""))
    )
    phone_base_matches = sum(
        1
        for pred, ref in zip(pred_phones, ref_phones)
        if (pred.get("base_phone", "") == "".join(ch for ch in (ref.get("phone", "") or "") if not ch.isdigit()))
    )

    boundary_mark_matches = 0
    pause_class_matches = 0
    predicted_break_count = 0
    gold_break_count = 0
    matched_break_count = 0
    for pred, ref in zip(pred_boundaries, ref_boundaries):
        pred_mark = _decode_boundary_mark(pred.get("boundary_mark", ""))
        gold_mark = ref.get("mark", "")
        pred_class = pred.get("pause_class", "none")
        gold_class = _gold_pause_class(ref)
        if pred_mark == gold_mark:
            boundary_mark_matches += 1
        if pred_class == gold_class:
            pause_class_matches += 1
        pred_break = pred_class == "hard_break_pause"
        gold_break = gold_class == "hard_break_pause"
        predicted_break_count += int(pred_break)
        gold_break_count += int(gold_break)
        matched_break_count += int(pred_break and gold_break)

    phone_prior_center_errors_ms = [
        abs(pred_center - 0.5 * (_safe_float(ref, "start") + _safe_float(ref, "end"))) * 1000.0
        for pred_center, ref in zip(phone_prior_centers, ref_phones)
    ]
    word_prior_start_errors_ms = [
        abs(word_prior_starts.get(_safe_int(ref, "word_index"), 0.0) - _safe_float(ref, "start")) * 1000.0
        for ref in ref_words
        if _safe_int(ref, "word_index") in word_prior_starts
    ]

    predicted_landmark_refs: dict[tuple[str, str, int], dict[str, Any]] = {}
    landmark_prior_errors_ms: list[float] = []
    for row in transcript_landmarks:
        landmark_type = row.get("type", "")
        if not landmark_type:
            continue
        if landmark_type == "pause_lull":
            key = (landmark_type, "boundary", _safe_int(row, "word_index"))
        else:
            key = (landmark_type, "phone", _safe_int(row, "global_phone_index"))
        predicted_landmark_refs[key] = row
        if (row.get("has_gold_reference") or "") not in {"", "0", "false", "False"}:
            landmark_prior_errors_ms.append(
                abs(_safe_float(row, "prior_center") - _safe_float(row, "gold_center")) * 1000.0
            )

    gold_landmark_refs = _build_gold_landmark_refs(ref_phones, ref_words, ref_boundaries)
    matched_landmark_count = sum(1 for key in predicted_landmark_refs if key in gold_landmark_refs)

    by_type: dict[str, dict[str, float]] = {}
    for landmark_type in sorted({key[0] for key in predicted_landmark_refs} | {key[0] for key in gold_landmark_refs}):
        predicted_count = sum(1 for key in predicted_landmark_refs if key[0] == landmark_type)
        gold_count = sum(1 for key in gold_landmark_refs if key[0] == landmark_type)
        matched_count = sum(1 for key in predicted_landmark_refs if key[0] == landmark_type and key in gold_landmark_refs)
        by_type[landmark_type] = {
            "predicted_count": predicted_count,
            "reference_count": gold_count,
            "matched_count": matched_count,
            "precision": matched_count / predicted_count if predicted_count else 0.0,
            "recall": matched_count / gold_count if gold_count else 0.0,
        }

    return {
        "available": True,
        "predicted_word_count": len(pred_words),
        "reference_word_count": len(ref_words),
        "word_exact_match_rate": word_exact_matches / max(len(ref_words), 1),
        "word_sequence_exact": pred_word_sequence == ref_word_sequence,
        "predicted_phone_count": len(pred_phones),
        "reference_phone_count": len(ref_phones),
        "phone_exact_match_rate": phone_exact_matches / max(len(ref_phones), 1),
        "phone_base_match_rate": phone_base_matches / max(len(ref_phones), 1),
        "predicted_boundary_count": len(pred_boundaries),
        "reference_boundary_count": len(ref_boundaries),
        "boundary_mark_accuracy": boundary_mark_matches / max(len(ref_boundaries), 1),
        "pause_class_accuracy": pause_class_matches / max(len(ref_boundaries), 1),
        "predicted_region_break_count": predicted_break_count,
        "reference_region_break_count": gold_break_count,
        "matched_region_break_count": matched_break_count,
        "region_break_precision": matched_break_count / max(predicted_break_count, 1),
        "region_break_recall": matched_break_count / max(gold_break_count, 1),
        "predicted_landmark_count": len(predicted_landmark_refs),
        "reference_landmark_count": len(gold_landmark_refs),
        "matched_landmark_count": matched_landmark_count,
        "landmark_precision": matched_landmark_count / max(len(predicted_landmark_refs), 1),
        "landmark_recall": matched_landmark_count / max(len(gold_landmark_refs), 1),
        "mean_abs_word_prior_start_error_ms": _mean(word_prior_start_errors_ms),
        "median_abs_word_prior_start_error_ms": _median(word_prior_start_errors_ms),
        "mean_abs_phone_prior_center_error_ms": _mean(phone_prior_center_errors_ms),
        "median_abs_phone_prior_center_error_ms": _median(phone_prior_center_errors_ms),
        "mean_abs_landmark_prior_center_error_ms": _mean(landmark_prior_errors_ms),
        "median_abs_landmark_prior_center_error_ms": _median(landmark_prior_errors_ms),
        "by_type": by_type,
    }


def load_case_grades(root: pathlib.Path, gold_root: pathlib.Path | None = None):
    rows: list[dict[str, Any]] = []
    graded: list[dict[str, Any]] = []
    ungraded: list[dict[str, Any]] = []
    if not root.exists():
        return rows, graded, ungraded

    for grade_path in sorted(root.glob("*/grade.json")):
        case_dir = grade_path.parent
        grade = _read_json(grade_path)
        text_plan_grade: dict[str, Any] = {}
        if gold_root is not None:
            gold_case_dir = gold_root / case_dir.name
            text_plan_grade = _compute_text_plan_grade(case_dir, gold_case_dir)
            if text_plan_grade.get("available"):
                _write_json(case_dir / "text_plan_grade.json", text_plan_grade)
        if not text_plan_grade:
            text_plan_grade = _read_json(case_dir / "text_plan_grade.json")

        row = {
            "case": case_dir.name,
            "grade": grade,
            "text_plan_grade": text_plan_grade,
            "streaming_landmark_grade": _read_json(case_dir / "streaming_landmark_grade.json"),
            "committed_rows": _read_csv_rows(case_dir / "committed.csv"),
            "commit_rows": _read_csv_rows(case_dir / "commit_decisions.csv"),
            "speech_rows": _read_csv_rows(case_dir / "speech_regions.csv"),
            "runtime_region_rows": _read_csv_rows(case_dir / "runtime_speech_regions.csv"),
            "occupancy_rows": _read_csv_rows(case_dir / "occupancy_frames.csv"),
            "gap_rows": _read_csv_rows(case_dir / "gap_candidates.csv"),
            "word_onset_rows": _read_csv_rows(case_dir / "word_onset_diagnostics.csv"),
        }
        rows.append(row)
        if grade.get("gold_available", True):
            graded.append(row)
        else:
            ungraded.append(row)
    return rows, graded, ungraded


def compute_summary(rows: list[dict[str, Any]], graded: list[dict[str, Any]], ungraded: list[dict[str, Any]]) -> dict[str, Any]:
    def speech_mismatch(row: dict[str, Any]) -> bool:
        return _flag(row["grade"], "pause_alignment", "speech_region_count_mismatch")

    qualified = [row for row in graded if not speech_mismatch(row)]
    bulk = qualified if qualified else graded

    speech_start_values = [_nested(row["grade"], "pause_alignment", "mean_abs_speech_region_start_error_ms") for row in qualified]
    speech_end_values = [_nested(row["grade"], "pause_alignment", "mean_abs_speech_region_end_error_ms") for row in qualified]
    speech_lead_values = [_nested(row["grade"], "pause_alignment", "mean_speech_region_lead_in_ms") for row in qualified]
    speech_tail_values = [_nested(row["grade"], "pause_alignment", "mean_speech_region_tail_out_ms") for row in qualified]
    word_start_values = [_nested(row["grade"], "word_onset_alignment", "mean_abs_start_error_ms") for row in bulk]
    word_start_median_values = [_nested(row["grade"], "word_onset_alignment", "median_abs_start_error_ms") for row in bulk]
    word_duration_values = [_nested(row["grade"], "mean_abs_duration_error_ms") for row in bulk]
    phoneme_center_values = [_nested(row["grade"], "mean_abs_center_error_ms") for row in bulk]
    phoneme_start_values = [_nested(row["grade"], "mean_abs_start_error_ms") for row in bulk]
    phoneme_end_values = [_nested(row["grade"], "mean_abs_end_error_ms") for row in bulk]
    intra_word_center_values = [_nested(row["grade"], "intra_word_alignment", "mean_abs_center_error_ms") for row in bulk]
    underrun_tail_values = [_nested(row["grade"], "speech_region_containment", "mean_late_end_leak_ms") for row in bulk]
    overrun_lead_values = [_nested(row["grade"], "speech_region_containment", "mean_early_start_leak_ms") for row in bulk]

    speech_refs = sum(int(_nested(row["grade"], "pause_alignment", "speech_region_reference_count")) for row in qualified)
    speech_matches = sum(int(_nested(row["grade"], "pause_alignment", "speech_region_matched_count")) for row in qualified)
    word_refs = sum(int(_nested(row["grade"], "word_onset_alignment", "reference_count")) for row in bulk)
    word_matches = sum(int(_nested(row["grade"], "word_onset_alignment", "matched_count")) for row in bulk)
    phoneme_refs = sum(int(row["grade"].get("reference_count", 0)) for row in bulk)
    phoneme_matches = sum(int(row["grade"].get("matched_count", 0)) for row in bulk)
    intra_word_refs = sum(int(_nested(row["grade"], "intra_word_alignment", "reference_count")) for row in bulk)
    intra_word_matches = sum(int(_nested(row["grade"], "intra_word_alignment", "matched_count")) for row in bulk)

    text_available = [row for row in bulk if row["text_plan_grade"].get("available")]
    landmark_available = [row for row in bulk if row["streaming_landmark_grade"].get("available")]

    text_by_type: dict[str, dict[str, float]] = {}
    text_predicted_break_count = sum(int(row["text_plan_grade"].get("predicted_region_break_count", 0)) for row in text_available)
    text_reference_break_count = sum(int(row["text_plan_grade"].get("reference_region_break_count", 0)) for row in text_available)
    text_matched_break_count = sum(int(row["text_plan_grade"].get("matched_region_break_count", 0)) for row in text_available)
    text_predicted_landmark_count = sum(int(row["text_plan_grade"].get("predicted_landmark_count", 0)) for row in text_available)
    text_reference_landmark_count = sum(int(row["text_plan_grade"].get("reference_landmark_count", 0)) for row in text_available)
    text_matched_landmark_count = sum(int(row["text_plan_grade"].get("matched_landmark_count", 0)) for row in text_available)
    text_reference_phone_count = sum(int(row["text_plan_grade"].get("reference_phone_count", 0)) for row in text_available)
    text_reference_word_count = sum(int(row["text_plan_grade"].get("reference_word_count", 0)) for row in text_available)
    text_phone_exact_count = sum(
        int(round(float(row["text_plan_grade"].get("phone_exact_match_rate", 0.0)) * max(int(row["text_plan_grade"].get("reference_phone_count", 0)), 0)))
        for row in text_available
    )
    text_phone_base_count = sum(
        int(round(float(row["text_plan_grade"].get("phone_base_match_rate", 0.0)) * max(int(row["text_plan_grade"].get("reference_phone_count", 0)), 0)))
        for row in text_available
    )
    text_word_exact_count = sum(
        int(round(float(row["text_plan_grade"].get("word_exact_match_rate", 0.0)) * max(int(row["text_plan_grade"].get("reference_word_count", 0)), 0)))
        for row in text_available
    )
    all_text_types = sorted(
        {
            key
            for row in text_available
            for key in row["text_plan_grade"].get("by_type", {}).keys()
        }
    )
    for landmark_type in all_text_types:
        predicted = sum(int(row["text_plan_grade"].get("by_type", {}).get(landmark_type, {}).get("predicted_count", 0)) for row in text_available)
        reference = sum(int(row["text_plan_grade"].get("by_type", {}).get(landmark_type, {}).get("reference_count", 0)) for row in text_available)
        matched = sum(int(row["text_plan_grade"].get("by_type", {}).get(landmark_type, {}).get("matched_count", 0)) for row in text_available)
        text_by_type[landmark_type] = {
            "predicted_count": predicted,
            "reference_count": reference,
            "matched_count": matched,
            "precision": matched / predicted if predicted else 0.0,
            "recall": matched / reference if reference else 0.0,
        }

    streaming_by_type: dict[str, dict[str, float]] = {}
    all_streaming_types = sorted(
        {
            key
            for row in landmark_available
            for key in row["streaming_landmark_grade"].get("types", {}).keys()
        }
    )
    for landmark_type in all_streaming_types:
        target_count = sum(int(row["streaming_landmark_grade"].get("types", {}).get(landmark_type, {}).get("target_count", 0)) for row in landmark_available)
        observation_count = sum(int(row["streaming_landmark_grade"].get("types", {}).get(landmark_type, {}).get("observation_count", 0)) for row in landmark_available)
        matched_target_count = sum(int(row["streaming_landmark_grade"].get("types", {}).get(landmark_type, {}).get("matched_target_count", 0)) for row in landmark_available)
        matched_observation_count = sum(int(row["streaming_landmark_grade"].get("types", {}).get(landmark_type, {}).get("matched_observation_count", 0)) for row in landmark_available)
        weighted_errors = []
        for row in landmark_available:
            stats = row["streaming_landmark_grade"].get("types", {}).get(landmark_type, {})
            matched_count = int(stats.get("matched_target_count", 0))
            if matched_count > 0:
                weighted_errors.extend([float(stats.get("mean_abs_center_error_ms", 0.0))] * matched_count)
        streaming_by_type[landmark_type] = {
            "target_count": target_count,
            "observation_count": observation_count,
            "matched_target_count": matched_target_count,
            "matched_observation_count": matched_observation_count,
            "precision": matched_observation_count / observation_count if observation_count else 0.0,
            "recall": matched_target_count / target_count if target_count else 0.0,
            "mean_abs_center_error_ms": _mean(weighted_errors),
        }

    streaming_target_count = sum(int(row["streaming_landmark_grade"].get("target_count", 0)) for row in landmark_available)
    streaming_observation_count = sum(int(row["streaming_landmark_grade"].get("observation_count", 0)) for row in landmark_available)
    streaming_matched_target_count = sum(int(row["streaming_landmark_grade"].get("matched_target_count", 0)) for row in landmark_available)
    streaming_matched_observation_count = sum(int(row["streaming_landmark_grade"].get("matched_observation_count", 0)) for row in landmark_available)
    streaming_center_error_values = []
    for row in landmark_available:
        grade = row["streaming_landmark_grade"]
        for landmark_type, stats in grade.get("types", {}).items():
            matched_count = int(stats.get("matched_target_count", 0))
            if matched_count > 0:
                streaming_center_error_values.extend([float(stats.get("mean_abs_center_error_ms", 0.0))] * matched_count)

    playback_summary = {
        "cases": len(rows),
        "graded_cases": len(graded),
        "ungraded_cases": len(ungraded),
        "qualified_cases": len(qualified),
        "bulk_cases": len(bulk),
        "degenerate_cases": sum(1 for row in graded if int(row["grade"].get("matched_count", 0)) == 0),
        "order_fail_cases": sum(1 for row in graded if int(row["grade"].get("order_violations", 0)) > 0),
        "speech_region_count_mismatch_cases": sum(1 for row in graded if speech_mismatch(row)),
        "visible_speech_region_count_mismatch_cases": 0,
        "text_sentence_span_count_mismatch_cases": sum(1 for row in graded if _flag(row["grade"], "pause_alignment", "text_sentence_span_count_mismatch")),
        "text_sentence_region_subspan_count_mismatch_cases": sum(1 for row in graded if _flag(row["grade"], "pause_alignment", "text_sentence_region_subspan_count_mismatch")),
        "sentence_region_count_mismatch_cases": 0,
        "clause_region_count_mismatch_cases": 0,
        "speech_match_rate": speech_matches / max(speech_refs, 1),
        "speech_boundary_start_ms": _mean(speech_start_values),
        "speech_boundary_end_ms": _mean(speech_end_values),
        "speech_boundary_start_median_ms": _median(speech_start_values),
        "speech_boundary_end_median_ms": _median(speech_end_values),
        "speech_lead_in_ms": _mean(speech_lead_values),
        "speech_tail_leakage_ms": _mean(speech_tail_values),
        "word_match_rate": word_matches / max(word_refs, 1),
        "word_start_ms": _mean(word_start_values),
        "word_head_start_ms": _mean(word_start_values),
        "word_head_start_median_ms": _median(word_start_median_values),
        "word_duration_ms": _mean(word_duration_values),
        "phoneme_coverage_rate": phoneme_matches / max(phoneme_refs, 1),
        "phoneme_center_ms": _mean(phoneme_center_values),
        "phoneme_start_ms": _mean(phoneme_start_values),
        "phoneme_end_ms": _mean(phoneme_end_values),
        "intra_word_coverage_rate": intra_word_matches / max(intra_word_refs, 1),
        "intra_word_center_ms": _mean(intra_word_center_values),
        "region_overrun_ms": _mean(underrun_tail_values),
        "region_early_leak_ms": _mean(overrun_lead_values),
        "direct_aligner_available": False,
    }

    text_plan_summary = {
        "text_plan_available_cases": len(text_available),
        "text_word_sequence_exact_rate": text_word_exact_count / max(text_reference_word_count, 1),
        "text_phone_exact_rate": text_phone_exact_count / max(text_reference_phone_count, 1),
        "text_phone_base_rate": text_phone_base_count / max(text_reference_phone_count, 1),
        "text_pause_class_accuracy": _mean([float(row["text_plan_grade"].get("pause_class_accuracy", 0.0)) for row in text_available]),
        "text_region_break_precision": text_matched_break_count / max(text_predicted_break_count, 1),
        "text_region_break_recall": text_matched_break_count / max(text_reference_break_count, 1),
        "text_landmark_precision": text_matched_landmark_count / max(text_predicted_landmark_count, 1),
        "text_landmark_recall": text_matched_landmark_count / max(text_reference_landmark_count, 1),
        "text_word_prior_start_ms": _mean([float(row["text_plan_grade"].get("mean_abs_word_prior_start_error_ms", 0.0)) for row in text_available]),
        "text_phone_prior_center_ms": _mean([float(row["text_plan_grade"].get("mean_abs_phone_prior_center_error_ms", 0.0)) for row in text_available]),
        "text_landmark_prior_center_ms": _mean([float(row["text_plan_grade"].get("mean_abs_landmark_prior_center_error_ms", 0.0)) for row in text_available]),
        "text_landmark_by_type": text_by_type,
    }

    runtime_detector_summary = {
        "available_cases": len(landmark_available),
        "planned_landmark_target_count": streaming_target_count,
        "observed_landmark_count": streaming_observation_count,
        "planned_landmark_selection_rate": streaming_observation_count / max(streaming_target_count, 1),
        "planned_landmark_precision": streaming_matched_observation_count / max(streaming_observation_count, 1),
        "planned_landmark_recall": streaming_matched_target_count / max(streaming_target_count, 1),
        "planned_landmark_center_ms": _mean(streaming_center_error_values),
        "planned_landmark_center_median_ms": _median(streaming_center_error_values),
        "planned_landmark_center_p90_ms": _p90(streaming_center_error_values),
        "planned_landmark_by_type": streaming_by_type,
    }

    summary = {
        **playback_summary,
        **text_plan_summary,
        "runtime_detector_available_cases": runtime_detector_summary["available_cases"],
        "runtime_detector_target_count": runtime_detector_summary["planned_landmark_target_count"],
        "runtime_detector_observation_count": runtime_detector_summary["observed_landmark_count"],
        "runtime_detector_planned_landmark_selection_rate": runtime_detector_summary["planned_landmark_selection_rate"],
        "runtime_detector_planned_landmark_precision": runtime_detector_summary["planned_landmark_precision"],
        "runtime_detector_planned_landmark_recall": runtime_detector_summary["planned_landmark_recall"],
        "runtime_detector_planned_landmark_center_ms": runtime_detector_summary["planned_landmark_center_ms"],
        "runtime_detector_planned_landmark_center_median_ms": runtime_detector_summary["planned_landmark_center_median_ms"],
        "runtime_detector_planned_landmark_center_p90_ms": runtime_detector_summary["planned_landmark_center_p90_ms"],
        "runtime_detector_planned_landmark_by_type": runtime_detector_summary["planned_landmark_by_type"],
        "streaming_landmark_available_cases": runtime_detector_summary["available_cases"],
        "streaming_landmark_target_count": runtime_detector_summary["planned_landmark_target_count"],
        "streaming_landmark_observation_count": runtime_detector_summary["observed_landmark_count"],
        "streaming_landmark_selection_rate": runtime_detector_summary["planned_landmark_selection_rate"],
        "streaming_landmark_precision": runtime_detector_summary["planned_landmark_precision"],
        "streaming_landmark_recall": runtime_detector_summary["planned_landmark_recall"],
        "streaming_landmark_center_ms": runtime_detector_summary["planned_landmark_center_ms"],
        "streaming_landmark_center_median_ms": runtime_detector_summary["planned_landmark_center_median_ms"],
        "streaming_landmark_center_p90_ms": runtime_detector_summary["planned_landmark_center_p90_ms"],
        "streaming_landmark_by_type": runtime_detector_summary["planned_landmark_by_type"],
        "playback": playback_summary,
        "text_plan_vs_mfa": text_plan_summary,
        "runtime_detector_vs_text_plan": runtime_detector_summary,
    }

    summary["speech_f1"] = summary["speech_match_rate"]
    summary["word_f1"] = summary["word_match_rate"]
    summary["sentence_region_count_mismatch_cases"] = summary["text_sentence_span_count_mismatch_cases"]
    summary["clause_region_count_mismatch_cases"] = summary["text_sentence_region_subspan_count_mismatch_cases"]
    return summary
