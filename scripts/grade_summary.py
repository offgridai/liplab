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


def load_case_grades(root: pathlib.Path, gold_root: pathlib.Path | None = None):
    rows: list[dict[str, Any]] = []
    graded: list[dict[str, Any]] = []
    ungraded: list[dict[str, Any]] = []
    if not root.exists():
        return rows, graded, ungraded

    for grade_path in sorted(root.glob("*/grade.json")):
        case_dir = grade_path.parent
        grade = _read_json(grade_path)
        row = {
            "case": case_dir.name,
            "grade": grade,
            "committed_rows": _read_csv_rows(case_dir / "committed.csv"),
            "commit_rows": _read_csv_rows(case_dir / "commit_decisions.csv"),
            "speech_rows": _read_csv_rows(case_dir / "speech_regions.csv"),
            "runtime_region_rows": _read_csv_rows(case_dir / "runtime_speech_regions.csv"),
            "occupancy_rows": _read_csv_rows(case_dir / "occupancy_frames.csv"),
            "gap_rows": _read_csv_rows(case_dir / "gap_candidates.csv"),
            "word_onset_rows": _read_csv_rows(case_dir / "word_onset_diagnostics.csv"),
            "streaming_detector_summary": _read_json(case_dir / "streaming_detector_summary.json"),
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

    speech_start_values = [
        _nested(row["grade"], "pause_alignment", "mean_abs_speech_region_start_error_ms")
        for row in qualified
    ]
    speech_end_values = [
        _nested(row["grade"], "pause_alignment", "mean_abs_speech_region_end_error_ms")
        for row in qualified
    ]
    speech_lead_values = [
        _nested(row["grade"], "pause_alignment", "mean_speech_region_lead_in_ms")
        for row in qualified
    ]
    speech_tail_values = [
        _nested(row["grade"], "pause_alignment", "mean_speech_region_tail_out_ms")
        for row in qualified
    ]
    word_start_values = [
        _nested(row["grade"], "word_onset_alignment", "mean_abs_start_error_ms")
        for row in bulk
    ]
    word_start_median_values = [
        _nested(row["grade"], "word_onset_alignment", "median_abs_start_error_ms")
        for row in bulk
    ]
    word_duration_values = [
        _nested(row["grade"], "mean_abs_duration_error_ms")
        for row in bulk
    ]
    phoneme_center_values = [
        _nested(row["grade"], "mean_abs_center_error_ms")
        for row in bulk
    ]
    phoneme_start_values = [
        _nested(row["grade"], "mean_abs_start_error_ms")
        for row in bulk
    ]
    phoneme_end_values = [
        _nested(row["grade"], "mean_abs_end_error_ms")
        for row in bulk
    ]
    intra_word_center_values = [
        _nested(row["grade"], "intra_word_alignment", "mean_abs_center_error_ms")
        for row in bulk
    ]
    underrun_tail_values = [
        _nested(row["grade"], "speech_region_containment", "mean_late_end_leak_ms")
        for row in bulk
    ]
    overrun_lead_values = [
        _nested(row["grade"], "speech_region_containment", "mean_early_start_leak_ms")
        for row in bulk
    ]

    speech_refs = sum(int(_nested(row["grade"], "pause_alignment", "speech_region_reference_count")) for row in qualified)
    speech_matches = sum(int(_nested(row["grade"], "pause_alignment", "speech_region_matched_count")) for row in qualified)
    word_refs = sum(int(_nested(row["grade"], "word_onset_alignment", "reference_count")) for row in bulk)
    word_matches = sum(int(_nested(row["grade"], "word_onset_alignment", "matched_count")) for row in bulk)
    phoneme_refs = sum(int(row["grade"].get("reference_count", 0)) for row in bulk)
    phoneme_matches = sum(int(row["grade"].get("matched_count", 0)) for row in bulk)
    intra_word_refs = sum(int(_nested(row["grade"], "intra_word_alignment", "reference_count")) for row in bulk)
    intra_word_matches = sum(int(_nested(row["grade"], "intra_word_alignment", "matched_count")) for row in bulk)

    summary = {
        "cases": len(rows),
        "graded_cases": len(graded),
        "ungraded_cases": len(ungraded),
        "qualified_cases": len(qualified),
        "bulk_cases": len(bulk),
        "degenerate_cases": sum(1 for row in graded if int(row["grade"].get("matched_count", 0)) == 0),
        "order_fail_cases": sum(1 for row in graded if int(row["grade"].get("order_violations", 0)) > 0),
        "speech_region_count_mismatch_cases": sum(1 for row in graded if speech_mismatch(row)),
        "visible_speech_region_count_mismatch_cases": 0,
        "text_sentence_span_count_mismatch_cases": sum(
            1 for row in graded if _flag(row["grade"], "pause_alignment", "text_sentence_span_count_mismatch")
        ),
        "text_sentence_region_subspan_count_mismatch_cases": sum(
            1 for row in graded if _flag(row["grade"], "pause_alignment", "text_sentence_region_subspan_count_mismatch")
        ),
        "sentence_region_count_mismatch_cases": 0,
        "clause_region_count_mismatch_cases": 0,
        "speech_f1": speech_matches / max(speech_refs, 1),
        "speech_boundary_start_ms": _mean(speech_start_values),
        "speech_boundary_end_ms": _mean(speech_end_values),
        "speech_boundary_start_median_ms": _median(speech_start_values),
        "speech_boundary_end_median_ms": _median(speech_end_values),
        "speech_lead_in_ms": _mean(speech_lead_values),
        "speech_tail_leakage_ms": _mean(speech_tail_values),
        "word_f1": word_matches / max(word_refs, 1),
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
        "streaming_pause_top1_accuracy": _mean([
            float(row["streaming_detector_summary"].get("pause_top1_accuracy", 0.0)) for row in rows if row["streaming_detector_summary"]
        ]),
        "streaming_word_boundary_f1": _mean([
            float(row["streaming_detector_summary"].get("word_boundary_f1", 0.0)) for row in rows if row["streaming_detector_summary"]
        ]),
    }

    summary["sentence_region_count_mismatch_cases"] = summary["text_sentence_span_count_mismatch_cases"]
    summary["clause_region_count_mismatch_cases"] = summary["text_sentence_region_subspan_count_mismatch_cases"]
    return summary
