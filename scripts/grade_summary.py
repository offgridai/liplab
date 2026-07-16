import csv
import json
import pathlib
import statistics
from typing import Any

_VOWEL_PHONE_BASES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}

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


def _write_csv_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _renderable_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # Legacy logs have no renderable column and all of their rows are visible.
    return [row for row in rows if row.get("renderable", "1") != "0"]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_overlap_seconds(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    total = 0.0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _longest_uncovered_seconds(
    containers: list[tuple[float, float]], coverage: list[tuple[float, float]]
) -> float:
    longest = 0.0
    for container_start, container_end in containers:
        cursor = container_start
        for covered_start, covered_end in coverage:
            if covered_end <= cursor or covered_start >= container_end:
                continue
            longest = max(longest, max(0.0, covered_start - cursor))
            cursor = max(cursor, min(covered_end, container_end))
            if cursor >= container_end:
                break
        longest = max(longest, max(0.0, container_end - cursor))
    return longest


def _compute_playback_health(
    commit_rows: list[dict[str, str]], speech_rows: list[dict[str, str]]
) -> dict[str, Any]:
    if not commit_rows:
        return {"available": False}

    usable_intervals: list[tuple[float, float]] = []
    late_start_count = 0
    late_center_count = 0
    expired_count = 0
    commits_by_tick: dict[int, list[dict[str, str]]] = {}
    for row in commit_rows:
        render_start = _safe_float(row, "render_start")
        render_center = _safe_float(row, "render_center")
        render_end = _safe_float(row, "render_end")
        commit_time = _safe_float(row, "commit_playback")
        late_start_count += int(commit_time > render_start)
        late_center_count += int(commit_time > render_center)
        expired_count += int(commit_time >= render_end)
        if commit_time < render_end:
            usable_intervals.append((max(render_start, commit_time), render_end))
        commits_by_tick.setdefault(round(commit_time * 1000.0), []).append(row)

    late_burst_count = 0
    late_burst_event_count = 0
    for tick_rows in commits_by_tick.values():
        late_rows = [row for row in tick_rows if _safe_float(row, "commit_lead") < 0.0]
        if len(late_rows) >= 2:
            late_burst_count += 1
            late_burst_event_count += len(late_rows)

    compressed_cadence_count = 0
    longest_compressed_cadence_run = 0
    compressed_run = 0
    for previous, current in zip(commit_rows, commit_rows[1:]):
        planned_delta = (
            _safe_float(current, "text_diagnostic_center")
            - _safe_float(previous, "text_diagnostic_center")
        )
        committed_delta = (
            _safe_float(current, "render_center")
            - _safe_float(previous, "render_center")
        )
        # Ignore naturally dense phones. Flag only cadence that was materially
        # compressed relative to the transcript prior by runtime scheduling.
        if planned_delta >= 0.075 and committed_delta <= planned_delta * 0.60:
            compressed_cadence_count += 1
            compressed_run += 1
            longest_compressed_cadence_run = max(longest_compressed_cadence_run, compressed_run)
        else:
            compressed_run = 0

    speech_intervals = _merge_intervals(
        [(_safe_float(row, "start"), _safe_float(row, "end")) for row in speech_rows]
    )
    rendered_intervals = _merge_intervals(usable_intervals)
    speech_seconds = sum(end - start for start, end in speech_intervals)
    covered_seconds = _interval_overlap_seconds(speech_intervals, rendered_intervals)
    uncovered_seconds = max(0.0, speech_seconds - covered_seconds)
    last_speech_end = speech_intervals[-1][1] if speech_intervals else 0.0
    scheduled_after_speech_count = sum(
        1 for row in commit_rows if _safe_float(row, "render_center") > last_speech_end + 0.050
    ) if speech_intervals else 0
    scheduled_tail_overrun_ms = max(
        [_safe_float(row, "render_end") - last_speech_end for row in commit_rows] + [0.0]
    ) * 1000.0 if speech_intervals else 0.0

    return {
        "available": True,
        "event_count": len(commit_rows),
        "late_start_count": late_start_count,
        "late_center_count": late_center_count,
        "expired_count": expired_count,
        "late_burst_count": late_burst_count,
        "late_burst_event_count": late_burst_event_count,
        "compressed_cadence_count": compressed_cadence_count,
        "longest_compressed_cadence_run": longest_compressed_cadence_run,
        "scheduled_after_speech_count": scheduled_after_speech_count,
        "scheduled_tail_overrun_ms": scheduled_tail_overrun_ms,
        "renderable_event_rate": (len(commit_rows) - expired_count) / len(commit_rows),
        "speech_animation_coverage_rate": covered_seconds / speech_seconds if speech_seconds else 1.0,
        "speech_animation_dropout_ms": uncovered_seconds * 1000.0,
        "longest_speech_animation_dropout_ms": _longest_uncovered_seconds(
            speech_intervals, rendered_intervals
        ) * 1000.0,
    }


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


def _gold_pause_class(boundary_row: dict[str, str]) -> str:
    pause_class = (boundary_row.get("pause_class") or "").strip().lower()
    split_applied = (boundary_row.get("split_applied") or "").strip() not in {"", "0", "false", "False"}
    if split_applied or pause_class in {"hard_pause", "region_break"}:
        return "hard_break_pause"
    if pause_class == "soft_pause":
        return "soft_list_pause"
    return "none"


def _compute_text_plan_grade(case_dir: pathlib.Path, gold_case_dir: pathlib.Path) -> dict[str, Any]:
    planned_words = _read_csv_rows(case_dir / "planned_words.csv")
    planned_boundaries = _read_csv_rows(case_dir / "planned_boundaries.csv")
    expected_phones = _read_csv_rows(case_dir / "expected_phones.csv")
    planned_syllables = _read_csv_rows(case_dir / "planned_syllables.csv")
    gold_words = _read_csv_rows(gold_case_dir / "words.csv")
    gold_phones = _read_csv_rows(gold_case_dir / "phones.csv")
    gold_boundaries = _read_csv_rows(gold_case_dir / "boundaries.csv")

    if not planned_words or not expected_phones or not planned_syllables or not gold_words or not gold_phones or not gold_boundaries:
        return {"available": False}

    pred_words = sorted(planned_words, key=lambda row: _safe_int(row, "word_index"))
    ref_words = sorted(gold_words, key=lambda row: _safe_int(row, "word_index"))
    pred_phones = sorted(expected_phones, key=lambda row: _safe_int(row, "phone_index"))
    ref_phones = list(gold_phones)
    pred_boundaries = sorted(planned_boundaries, key=lambda row: _safe_int(row, "word_index"))
    ref_boundaries = sorted(gold_boundaries, key=lambda row: _safe_int(row, "word_index"))
    pred_boundaries_by_word = {_safe_int(row, "word_index"): row for row in pred_boundaries}
    boundary_pairs = [
        (pred_boundaries_by_word[_safe_int(ref, "word_index")], ref)
        for ref in ref_boundaries
        if _safe_int(ref, "word_index") in pred_boundaries_by_word
    ]

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
    def monotonic_phone_pairs(exact: bool) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        word_indices = sorted(
            {_safe_int(row, "word_index") for row in pred_phones}
            | {_safe_int(row, "word_index") for row in ref_phones}
        )
        for word_index in word_indices:
            pred_indices = [
                index for index, row in enumerate(pred_phones)
                if _safe_int(row, "word_index") == word_index
            ]
            ref_indices = [
                index for index, row in enumerate(ref_phones)
                if _safe_int(row, "word_index") == word_index
            ]
            n = len(pred_indices)
            m = len(ref_indices)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                pred = pred_phones[pred_indices[i - 1]]
                pred_value = pred.get("phone" if exact else "base_phone", "")
                for j in range(1, m + 1):
                    ref_phone = ref_phones[ref_indices[j - 1]].get("phone", "") or ""
                    ref_value = ref_phone if exact else "".join(ch for ch in ref_phone if not ch.isdigit())
                    if pred_value == ref_value:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            i = n
            j = m
            word_pairs: list[tuple[int, int]] = []
            while i > 0 and j > 0:
                pred = pred_phones[pred_indices[i - 1]]
                pred_value = pred.get("phone" if exact else "base_phone", "")
                ref_phone = ref_phones[ref_indices[j - 1]].get("phone", "") or ""
                ref_value = ref_phone if exact else "".join(ch for ch in ref_phone if not ch.isdigit())
                if pred_value == ref_value and dp[i][j] == dp[i - 1][j - 1] + 1:
                    word_pairs.append((pred_indices[i - 1], ref_indices[j - 1]))
                    i -= 1
                    j -= 1
                elif dp[i - 1][j] >= dp[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            pairs.extend(reversed(word_pairs))
        return pairs

    exact_phone_pairs = monotonic_phone_pairs(exact=True)
    base_phone_pairs = monotonic_phone_pairs(exact=False)
    phone_exact_matches = len(exact_phone_pairs)
    phone_base_matches = len(base_phone_pairs)

    spn_word_indices = {
        _safe_int(row, "word_index")
        for row in ref_phones
        if (row.get("phone", "") or "").lower() == "spn"
    }
    pred_syllables = [
        row
        for row in sorted(planned_syllables, key=lambda row: _safe_int(row, "syllable_index"))
        if _safe_int(row, "word_index") not in spn_word_indices
    ]
    ref_nuclei = [
        row for row in ref_phones
        if _safe_int(row, "word_index") not in spn_word_indices
        and "".join(ch for ch in (row.get("phone", "") or "") if not ch.isdigit()) in _VOWEL_PHONE_BASES
    ]

    def monotonic_nucleus_matches(exact: bool) -> int:
        matched = 0
        word_indices = sorted(
            {_safe_int(row, "word_index") for row in pred_syllables}
            | {_safe_int(row, "word_index") for row in ref_nuclei}
        )
        for word_index in word_indices:
            predicted = [row for row in pred_syllables if _safe_int(row, "word_index") == word_index]
            reference = [row for row in ref_nuclei if _safe_int(row, "word_index") == word_index]
            n = len(predicted)
            m = len(reference)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                pred_value = predicted[i - 1].get("nucleus_phone" if exact else "nucleus_base_phone", "")
                for j in range(1, m + 1):
                    ref_phone = reference[j - 1].get("phone", "") or ""
                    ref_value = ref_phone if exact else "".join(ch for ch in ref_phone if not ch.isdigit())
                    dp[i][j] = dp[i - 1][j - 1] + 1 if pred_value == ref_value else max(dp[i - 1][j], dp[i][j - 1])
            matched += dp[n][m]
        return matched

    syllable_count_matches = 0
    for word_index in sorted(
        {_safe_int(row, "word_index") for row in pred_syllables}
        | {_safe_int(row, "word_index") for row in ref_nuclei}
    ):
        pred_count = sum(_safe_int(row, "word_index") == word_index for row in pred_syllables)
        ref_count = sum(_safe_int(row, "word_index") == word_index for row in ref_nuclei)
        syllable_count_matches += min(pred_count, ref_count)
    nucleus_exact_matches = monotonic_nucleus_matches(exact=True)
    nucleus_base_matches = monotonic_nucleus_matches(exact=False)

    matched_pred_base = {pred_index for pred_index, _ in base_phone_pairs}
    matched_ref_base = {ref_index for _, ref_index in base_phone_pairs}
    unmatched_predicted_base_phones: dict[str, int] = {}
    unmatched_reference_base_phones: dict[str, int] = {}
    for index, row in enumerate(pred_phones):
        if index in matched_pred_base:
            continue
        phone = row.get("base_phone", "") or "<empty>"
        unmatched_predicted_base_phones[phone] = unmatched_predicted_base_phones.get(phone, 0) + 1
    for index, row in enumerate(ref_phones):
        if index in matched_ref_base:
            continue
        phone = "".join(ch for ch in (row.get("phone", "") or "") if not ch.isdigit()) or "<empty>"
        unmatched_reference_base_phones[phone] = unmatched_reference_base_phones.get(phone, 0) + 1

    base_matches_by_word: dict[int, int] = {}
    for pred_index, _ in base_phone_pairs:
        word_index = _safe_int(pred_phones[pred_index], "word_index")
        base_matches_by_word[word_index] = base_matches_by_word.get(word_index, 0) + 1
    phone_mismatches_by_word: dict[str, dict[str, int]] = {}
    word_indices = sorted(
        {_safe_int(row, "word_index") for row in pred_phones}
        | {_safe_int(row, "word_index") for row in ref_phones}
    )
    for word_index in word_indices:
        predicted = [row for row in pred_phones if _safe_int(row, "word_index") == word_index]
        reference = [row for row in ref_phones if _safe_int(row, "word_index") == word_index]
        matched = base_matches_by_word.get(word_index, 0)
        false_positive = len(predicted) - matched
        false_negative = len(reference) - matched
        if false_positive <= 0 and false_negative <= 0:
            continue
        word = _normalize_word(
            (predicted[0].get("word", "") if predicted else "")
            or (reference[0].get("word", "") if reference else "")
        ) or "<empty>"
        totals = phone_mismatches_by_word.setdefault(
            word,
            {"occurrences": 0, "false_positive": 0, "false_negative": 0, "spn_reference": 0},
        )
        totals["occurrences"] += 1
        totals["false_positive"] += false_positive
        totals["false_negative"] += false_negative
        totals["spn_reference"] += int(any((row.get("phone", "") or "").lower() == "spn" for row in reference))

    boundary_mark_matches = 0
    pause_class_matches = 0
    predicted_break_count = 0
    gold_break_count = 0
    matched_break_count = 0
    for pred, ref in boundary_pairs:
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
        abs(phone_prior_centers[pred_index] - 0.5 * (
            _safe_float(ref_phones[ref_index], "start")
            + _safe_float(ref_phones[ref_index], "end")
        )) * 1000.0
        for pred_index, ref_index in base_phone_pairs
    ]
    phone_prior_duration_errors_ms = [
        abs(
            max(_safe_float(pred_phones[pred_index], "weight_seconds"), 0.025)
            - max(
                _safe_float(ref_phones[ref_index], "end")
                - _safe_float(ref_phones[ref_index], "start"),
                0.0,
            )
        ) * 1000.0
        for pred_index, ref_index in base_phone_pairs
    ]
    word_prior_start_errors_ms = [
        abs(word_prior_starts.get(_safe_int(ref, "word_index"), 0.0) - _safe_float(ref, "start")) * 1000.0
        for ref in ref_words
        if _safe_int(ref, "word_index") in word_prior_starts
    ]

    return {
        "available": True,
        "predicted_word_count": len(pred_words),
        "reference_word_count": len(ref_words),
        "word_exact_match_rate": word_exact_matches / max(len(ref_words), 1),
        "word_sequence_exact": pred_word_sequence == ref_word_sequence,
        "predicted_phone_count": len(pred_phones),
        "reference_phone_count": len(ref_phones),
        "phone_exact_match_count": phone_exact_matches,
        "phone_base_match_count": phone_base_matches,
        "phone_exact_precision": phone_exact_matches / max(len(pred_phones), 1),
        "phone_exact_recall": phone_exact_matches / max(len(ref_phones), 1),
        "phone_exact_f1": 2.0 * phone_exact_matches / max(len(pred_phones) + len(ref_phones), 1),
        "phone_base_precision": phone_base_matches / max(len(pred_phones), 1),
        "phone_base_recall": phone_base_matches / max(len(ref_phones), 1),
        "phone_base_f1": 2.0 * phone_base_matches / max(len(pred_phones) + len(ref_phones), 1),
        "predicted_syllable_count": len(pred_syllables),
        "reference_syllable_count": len(ref_nuclei),
        "matched_syllable_count": syllable_count_matches,
        "syllable_count_ratio": len(pred_syllables) / max(len(ref_nuclei), 1),
        "syllable_count_precision": syllable_count_matches / max(len(pred_syllables), 1),
        "syllable_count_recall": syllable_count_matches / max(len(ref_nuclei), 1),
        "nucleus_exact_match_count": nucleus_exact_matches,
        "nucleus_base_match_count": nucleus_base_matches,
        "nucleus_exact_precision": nucleus_exact_matches / max(len(pred_syllables), 1),
        "nucleus_exact_recall": nucleus_exact_matches / max(len(ref_nuclei), 1),
        "nucleus_base_precision": nucleus_base_matches / max(len(pred_syllables), 1),
        "nucleus_base_recall": nucleus_base_matches / max(len(ref_nuclei), 1),
        "syllable_ungradeable_spn_word_count": len(spn_word_indices),
        "phone_exact_match_rate": phone_exact_matches / max(len(ref_phones), 1),
        "phone_base_match_rate": phone_base_matches / max(len(ref_phones), 1),
        "unmatched_predicted_base_phones": unmatched_predicted_base_phones,
        "unmatched_reference_base_phones": unmatched_reference_base_phones,
        "phone_mismatches_by_word": phone_mismatches_by_word,
        "predicted_boundary_count": len(pred_boundaries),
        "reference_boundary_count": len(ref_boundaries),
        "boundary_mark_accuracy": boundary_mark_matches / max(len(ref_boundaries), 1),
        "pause_class_accuracy": pause_class_matches / max(len(ref_boundaries), 1),
        "predicted_pause_hint_count": sum(
            (pred.get("pause_class", "none") or "none") != "none"
            for pred, _ in boundary_pairs
        ),
        "reference_pause_hint_count": sum(_gold_pause_class(row) != "none" for row in ref_boundaries),
        "matched_pause_hint_count": sum(
            (pred.get("pause_class", "none") or "none") != "none"
            and _gold_pause_class(ref) != "none"
            for pred, ref in boundary_pairs
        ),
        "predicted_region_break_count": predicted_break_count,
        "reference_region_break_count": gold_break_count,
        "matched_region_break_count": matched_break_count,
        "region_break_precision": matched_break_count / max(predicted_break_count, 1),
        "region_break_recall": matched_break_count / max(gold_break_count, 1),
        "mean_abs_word_prior_start_error_ms": _mean(word_prior_start_errors_ms),
        "median_abs_word_prior_start_error_ms": _median(word_prior_start_errors_ms),
        "mean_abs_phone_prior_center_error_ms": _mean(phone_prior_center_errors_ms),
        "median_abs_phone_prior_center_error_ms": _median(phone_prior_center_errors_ms),
        "mean_abs_phone_prior_duration_error_ms": _mean(phone_prior_duration_errors_ms),
        "median_abs_phone_prior_duration_error_ms": _median(phone_prior_duration_errors_ms),
        "phone_prior_duration_match_count": len(phone_prior_duration_errors_ms),
    }


def _compute_runtime_health(
    case_dir: pathlib.Path,
    gold_case_dir: pathlib.Path | None,
) -> dict[str, Any]:
    planned = _renderable_rows(_read_csv_rows(case_dir / "planned.csv"))
    committed = _renderable_rows(_read_csv_rows(case_dir / "committed.csv"))
    predicted_regions = _read_csv_rows(case_dir / "speech_regions.csv")
    boundary_rows = [
        row for row in _read_csv_rows(case_dir / "runtime_boundary_state.csv")
        if _safe_int(row, "final_replay", 0) == 0
    ]

    committed_indices = {_safe_int(row, "index", -1) for row in committed}
    max_committed_index = max(committed_indices, default=-1)
    uncommitted = [row for row in planned if _safe_int(row, "index", -1) not in committed_indices]
    uncommitted_suffix = [
        row for row in uncommitted
        if _safe_int(row, "index", -1) > max_committed_index
    ]

    word_regions: dict[int, set[int]] = {}
    for row in committed:
        word_index = _safe_int(row, "word_index", -1)
        region_index = _safe_int(row, "speech_region_index", -1)
        if word_index >= 0 and region_index >= 0:
            word_regions.setdefault(word_index, set()).add(region_index)
    split_word_indices = sorted(
        word_index for word_index, regions in word_regions.items() if len(regions) > 1
    )

    report: dict[str, Any] = {
        "available": bool(planned or committed or boundary_rows),
        "planned_visible_count": len(planned),
        "committed_visible_count": len(committed),
        "uncommitted_visible_count": len(uncommitted),
        "uncommitted_visible_suffix_count": len(uncommitted_suffix),
        "uncommitted_visible_internal_count": len(uncommitted) - len(uncommitted_suffix),
        "cross_region_word_split_count": len(split_word_indices),
        "cross_region_word_indices": split_word_indices,
        "gold_available": False,
    }
    if gold_case_dir is None or not gold_case_dir.exists():
        return report

    gold_speech = _read_csv_rows(gold_case_dir / "speech.csv")
    gold_words = _read_csv_rows(gold_case_dir / "words.csv")
    if not gold_speech:
        return report
    report["gold_available"] = True

    gold_intervals = [
        (_safe_float(row, "start"), _safe_float(row, "end"))
        for row in gold_speech
    ]
    gold_start = gold_intervals[0][0]
    gold_end = gold_intervals[-1][1]
    first_committed_center = min(
        (_safe_float(row, "center") for row in committed),
        default=gold_end,
    )
    first_delay_ms = max(0.0, first_committed_center - gold_start) * 1000.0
    report.update({
        "first_region_animation_delay_ms": first_delay_ms,
        "first_region_starved": first_delay_ms > 250.0,
        "speech_region_count_match": len(predicted_regions) == len(gold_speech),
    })

    gold_word_start_by_region = {
        _safe_int(row, "index", -1): _safe_int(row, "word_start_index", -1)
        for row in gold_speech
    }
    resume_details: list[dict[str, Any]] = []
    if len(predicted_regions) == len(gold_speech):
        for region_index in range(1, len(gold_speech)):
            region_events = sorted(
                (
                    row for row in committed
                    if _safe_int(row, "speech_region_index", -1) == region_index
                ),
                key=lambda row: _safe_float(row, "center"),
            )
            expected_word = gold_word_start_by_region.get(region_index, -1)
            actual_word = _safe_int(region_events[0], "word_index", -1) if region_events else -1
            resume_details.append({
                "region_index": region_index,
                "expected_word_index": expected_word,
                "actual_word_index": actual_word,
                "word_error": actual_word - expected_word if actual_word >= 0 and expected_word >= 0 else None,
                "missing_animation": not region_events,
            })
    valid_resume_errors = [
        int(detail["word_error"])
        for detail in resume_details
        if detail["word_error"] is not None
    ]
    report.update({
        "resume_cursor_target_count": len(resume_details),
        "resume_cursor_observed_count": len(valid_resume_errors),
        "resume_cursor_exact_count": sum(error == 0 for error in valid_resume_errors),
        "resume_cursor_behind_count": sum(error < 0 for error in valid_resume_errors),
        "resume_cursor_ahead_count": sum(error > 0 for error in valid_resume_errors),
        "resume_cursor_mean_abs_word_error": _mean([abs(error) for error in valid_resume_errors]),
        "resume_cursor_details": resume_details,
    })

    gate_closed_speech_sec = 0.0
    gate_open_pause_sec = 0.0
    gold_speech_sec = sum(max(0.0, end - start) for start, end in gold_intervals)
    gold_pause_sec = max(0.0, gold_end - gold_start - gold_speech_sec)
    for index, row in enumerate(boundary_rows[:-1]):
        current = _safe_float(row, "current_playback_sec")
        following = _safe_float(boundary_rows[index + 1], "current_playback_sec")
        duration = max(0.0, min(following - current, 0.100))
        if duration <= 0.0 or current < gold_start or current >= gold_end:
            continue
        midpoint = current + duration * 0.5
        in_gold_speech = any(start <= midpoint < end for start, end in gold_intervals)
        gate_open = _safe_int(row, "b_audio_envelope_gate_open", 0) != 0
        if in_gold_speech and not gate_open:
            gate_closed_speech_sec += duration
        elif not in_gold_speech and gate_open:
            gate_open_pause_sec += duration
    report.update({
        "gold_speech_sec": gold_speech_sec,
        "gold_pause_sec": gold_pause_sec,
        "gate_closed_during_gold_speech_ms": gate_closed_speech_sec * 1000.0,
        "gate_closed_during_gold_speech_rate": gate_closed_speech_sec / gold_speech_sec if gold_speech_sec else 0.0,
        "gate_open_during_gold_pause_ms": gate_open_pause_sec * 1000.0,
        "gate_open_during_gold_pause_rate": gate_open_pause_sec / gold_pause_sec if gold_pause_sec else 0.0,
    })
    return report


def _compute_streaming_region_boundary_grade(
    case_dir: pathlib.Path,
    gold_case_dir: pathlib.Path | None,
    tolerance_seconds: float = 0.100,
) -> dict[str, Any]:
    comparison_tolerance = tolerance_seconds + 1e-6
    predicted_regions = _read_csv_rows(case_dir / "speech_regions.csv")
    gold_regions = (
        _read_csv_rows(gold_case_dir / "speech.csv")
        if gold_case_dir is not None and gold_case_dir.exists()
        else []
    )
    if not gold_regions:
        return {"available": False}

    predicted_pairs = [
        (_safe_float(predicted_regions[index], "end"), _safe_float(predicted_regions[index + 1], "start"))
        for index in range(len(predicted_regions) - 1)
    ]
    reference_pairs = [
        (_safe_float(gold_regions[index], "end"), _safe_float(gold_regions[index + 1], "start"))
        for index in range(len(gold_regions) - 1)
    ]

    gap_candidates = _read_csv_rows(case_dir / "gap_candidates.csv")
    candidate_pairs = [
        (_safe_float(row, "gap_start"), _safe_float(row, "gap_end"))
        for row in gap_candidates
    ]

    def monotonic_pair_matches(
        observations: list[tuple[float, float]],
    ) -> list[tuple[int, int]]:
        n = len(observations)
        m = len(reference_pairs)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            observed_close, observed_resume = observations[i - 1]
            for j in range(1, m + 1):
                reference_close, reference_resume = reference_pairs[j - 1]
                compatible = (
                    abs(observed_close - reference_close) <= comparison_tolerance
                    and abs(observed_resume - reference_resume) <= comparison_tolerance
                )
                dp[i][j] = max(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1] + int(compatible),
                )

        matches: list[tuple[int, int]] = []
        i = n
        j = m
        while i > 0 and j > 0:
            observed_close, observed_resume = observations[i - 1]
            reference_close, reference_resume = reference_pairs[j - 1]
            compatible = (
                abs(observed_close - reference_close) <= comparison_tolerance
                and abs(observed_resume - reference_resume) <= comparison_tolerance
            )
            if compatible and dp[i][j] == dp[i - 1][j - 1] + 1:
                matches.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        matches.reverse()
        return matches

    accepted_matches = monotonic_pair_matches(predicted_pairs)
    candidate_matches = monotonic_pair_matches(candidate_pairs)
    close_errors_ms: list[float] = []
    resume_errors_ms: list[float] = []
    matched_pairs: list[dict[str, Any]] = []
    for predicted_index, reference_index in accepted_matches:
        predicted_close, predicted_resume = predicted_pairs[predicted_index]
        reference_close, reference_resume = reference_pairs[reference_index]
        close_error_ms = (predicted_close - reference_close) * 1000.0
        resume_error_ms = (predicted_resume - reference_resume) * 1000.0
        close_errors_ms.append(abs(close_error_ms))
        resume_errors_ms.append(abs(resume_error_ms))
        matched_pairs.append({
                "predicted_pair_index": predicted_index,
                "reference_pair_index": reference_index,
                "close_error_ms": close_error_ms,
                "resume_error_ms": resume_error_ms,
            })

    n = len(predicted_pairs)
    m = len(reference_pairs)
    matched_count = len(accepted_matches)
    accepted_reference_indices = {reference for _, reference in accepted_matches}
    candidate_reference_indices = {reference for _, reference in candidate_matches}
    adjudication_miss_count = len(candidate_reference_indices - accepted_reference_indices)
    no_candidate_count = m - len(candidate_reference_indices)
    initial_open_error_ms = (
        abs(_safe_float(predicted_regions[0], "start") - _safe_float(gold_regions[0], "start")) * 1000.0
        if predicted_regions else 0.0
    )
    final_close_error_ms = (
        abs(_safe_float(predicted_regions[-1], "end") - _safe_float(gold_regions[-1], "end")) * 1000.0
        if predicted_regions else 0.0
    )
    return {
        "available": True,
        "tolerance_ms": tolerance_seconds * 1000.0,
        "predicted_pair_count": n,
        "reference_pair_count": m,
        "matched_pair_count": matched_count,
        "matched_pairs": matched_pairs,
        "candidate_pair_count": len(candidate_pairs),
        "candidate_matched_pair_count": len(candidate_matches),
        "candidate_oracle_recall": len(candidate_matches) / m if m else 1.0,
        "adjudication_miss_count": adjudication_miss_count,
        "no_candidate_count": no_candidate_count,
        "false_accepted_pair_count": n - matched_count,
        "pair_precision": matched_count / n if n else (1.0 if m == 0 else 0.0),
        "pair_recall": matched_count / m if m else 1.0,
        "pair_f1": 2.0 * matched_count / (n + m) if n + m else 1.0,
        "mean_abs_close_error_ms": _mean(close_errors_ms),
        "mean_abs_resume_error_ms": _mean(resume_errors_ms),
        "initial_open_observed": bool(predicted_regions),
        "initial_open_matched": bool(predicted_regions) and initial_open_error_ms <= comparison_tolerance * 1000.0,
        "initial_open_error_ms": initial_open_error_ms,
        "final_close_observed": bool(predicted_regions),
        "final_close_matched": bool(predicted_regions) and final_close_error_ms <= comparison_tolerance * 1000.0,
        "final_close_error_ms": final_close_error_ms,
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

        commit_rows = _renderable_rows(_read_csv_rows(case_dir / "commit_decisions.csv"))
        committed_rows = _renderable_rows(_read_csv_rows(case_dir / "committed.csv"))
        speech_rows = _read_csv_rows(case_dir / "speech_regions.csv")
        playback_health = _compute_playback_health(commit_rows, speech_rows)
        if playback_health.get("available"):
            _write_json(case_dir / "playback_health.json", playback_health)
        gold_case_dir = gold_root / case_dir.name if gold_root is not None else None
        runtime_health = _compute_runtime_health(case_dir, gold_case_dir)
        if runtime_health.get("available"):
            _write_json(case_dir / "runtime_health.json", runtime_health)
        streaming_region_boundary_grade = _compute_streaming_region_boundary_grade(
            case_dir,
            gold_case_dir,
        )
        if streaming_region_boundary_grade.get("available"):
            _write_json(
                case_dir / "streaming_region_boundary_grade.json",
                streaming_region_boundary_grade,
            )

        row = {
            "case": case_dir.name,
            "grade": grade,
            "text_plan_grade": text_plan_grade,
            "streaming_evidence_grade": _read_json(case_dir / "streaming_evidence_grade.json"),
            "streaming_conditioned_evidence_grade": _read_json(case_dir / "streaming_conditioned_evidence_grade.json"),
            "streaming_intra_envelope_phone_grade": _read_json(case_dir / "streaming_intra_envelope_phone_grade.json"),
            "streaming_syllable_candidate_set_grade": _read_json(case_dir / "streaming_syllable_candidate_set_grade.json"),
            "streaming_syllable_assignment_grade": _read_json(case_dir / "streaming_syllable_assignment_grade.json"),
            "streaming_syllable_position_grade": _read_json(case_dir / "streaming_syllable_position_grade.json"),
            "streaming_historical_anchor_grade": _read_json(case_dir / "streaming_historical_anchor_grade.json"),
            "runtime_syllable_assignment_grade": _read_json(case_dir / "runtime_syllable_assignment_grade.json"),
            "committed_rows": committed_rows,
            "commit_rows": commit_rows,
            "speech_rows": speech_rows,
            "playback_health": playback_health,
            "runtime_health": runtime_health,
            "streaming_region_boundary_grade": streaming_region_boundary_grade,
            "runtime_region_rows": _read_csv_rows(case_dir / "runtime_speech_regions.csv"),
            "occupancy_rows": _read_csv_rows(case_dir / "occupancy_frames.csv"),
            "gap_rows": _read_csv_rows(case_dir / "gap_candidates.csv"),
            "word_onset_rows": _read_csv_rows(case_dir / "word_onset_diagnostics.csv"),
            "planned_word_rows": _read_csv_rows(case_dir / "planned_words.csv"),
            "expected_phone_rows": _read_csv_rows(case_dir / "expected_phones.csv"),
            "planned_event_rows": _read_csv_rows(case_dir / "planned.csv"),
            "gold_visible_rows": _read_csv_rows(case_dir / "gold_visible_visemes.csv"),
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

    visual_role_counts: dict[str, int] = {}
    source_phone_counts: dict[str, dict[str, int]] = {}
    pronunciation_words: set[tuple[str, int]] = set()
    pronunciation_alternate_words: set[tuple[str, int]] = set()
    pronunciation_selected_words: set[tuple[str, int]] = set()
    pronunciation_selected_by_word: dict[str, int] = {}
    for row in rows:
        for phone in row.get("expected_phone_rows", []):
            role = phone.get("visual_role", "legacy_unspecified")
            base = phone.get("base_phone", "")
            visual_role_counts[role] = visual_role_counts.get(role, 0) + 1
            stats = source_phone_counts.setdefault(base, {"phone_count": 0, "visible_count": 0})
            stats["phone_count"] += 1
            stats["visible_count"] += int(phone.get("is_visible_viseme", "0") == "1")
            word_key = (row["case"], _safe_int(phone, "word_index"))
            if word_key not in pronunciation_words:
                pronunciation_words.add(word_key)
                variant_count = _safe_int(phone, "pronunciation_variant_count", 1)
                variant_index = _safe_int(phone, "pronunciation_variant_index", 0)
                if variant_count > 1:
                    pronunciation_alternate_words.add(word_key)
                if variant_index > 0:
                    pronunciation_selected_words.add(word_key)
                    word = _normalize_word(phone.get("word", ""))
                    pronunciation_selected_by_word[word] = pronunciation_selected_by_word.get(word, 0) + 1
    total_phone_plan_count = sum(visual_role_counts.values())
    total_visible_plan_count = sum(
        count for role, count in visual_role_counts.items()
        if role in {"primary_pose", "supporting_pose", "legacy_unspecified"}
    )
    total_renderable_event_count = sum(
        1
        for row in graded
        for event in row.get("planned_event_rows", [])
        if event.get("renderable", "1") == "1"
    )
    total_gold_corresponded_event_count = sum(
        len(row.get("gold_visible_rows", []))
        for row in graded
    )
    correspondence_by_phone: dict[str, dict[str, float]] = {}
    for row in graded:
        for event in row.get("planned_event_rows", []):
            if event.get("renderable", "1") != "1":
                continue
            base = event.get("source_phone", "")
            stats = correspondence_by_phone.setdefault(
                base, {"planned_visible_count": 0, "gold_corresponded_visible_count": 0}
            )
            stats["planned_visible_count"] += 1
        for label in row.get("gold_visible_rows", []):
            base = label.get("source_phone_base", "")
            stats = correspondence_by_phone.setdefault(
                base, {"planned_visible_count": 0, "gold_corresponded_visible_count": 0}
            )
            stats["gold_corresponded_visible_count"] += 1
    for stats in correspondence_by_phone.values():
        stats["correspondence_rate"] = (
            stats["gold_corresponded_visible_count"]
            / max(stats["planned_visible_count"], 1)
        )

    list_word_start_errors: list[float] = []
    list_word_count = 0
    for row in bulk:
        planned_words = row.get("planned_word_rows", [])
        sentence_words: dict[int, list[dict[str, str]]] = {}
        for word in planned_words:
            sentence_words.setdefault(_safe_int(word, "text_sentence_index"), []).append(word)
        list_word_indices: set[int] = set()
        for words in sentence_words.values():
            words.sort(key=lambda item: _safe_int(item, "word_index"))
            list_boundaries = [
                _safe_int(item, "word_index")
                for item in words
                if item.get("pause_class", "").lower() == "soft_list_pause"
            ]
            if len(list_boundaries) < 2:
                continue
            first_item = list_boundaries[0]
            list_word_indices.update(
                _safe_int(item, "word_index")
                for item in words
                if _safe_int(item, "word_index") >= first_item
            )
        onset_by_index = {
            _safe_int(item, "word_index"): item
            for item in row.get("word_onset_rows", [])
        }
        for word_index in list_word_indices:
            onset = onset_by_index.get(word_index)
            if not onset or onset.get("missing_event", "").lower() == "true":
                continue
            list_word_start_errors.append(abs(_safe_float(onset, "error_ms")))
            list_word_count += 1

    # Text planning is independent of runtime speech-region segmentation. Grade
    # every approved case; region-count qualification applies only to timing.
    text_available = [row for row in graded if row["text_plan_grade"].get("available")]
    evidence_available = [row for row in bulk if row["streaming_evidence_grade"].get("available")]
    conditioned_evidence_available = [row for row in bulk if row["streaming_conditioned_evidence_grade"].get("available")]
    intra_envelope_phone_available = [
        row for row in bulk if row["streaming_intra_envelope_phone_grade"].get("available")
    ]
    syllable_candidate_set_available = [
        row for row in bulk if row["streaming_syllable_candidate_set_grade"].get("available")
    ]
    syllable_assignment_available = [
        row for row in bulk if row["streaming_syllable_assignment_grade"].get("available")
    ]
    syllable_position_available = [row for row in bulk if row["streaming_syllable_position_grade"].get("available")]
    historical_anchor_available = [row for row in bulk if row["streaming_historical_anchor_grade"].get("available")]
    runtime_syllable_available = [row for row in bulk if row["runtime_syllable_assignment_grade"].get("available")]
    playback_health_available = [row for row in bulk if row["playback_health"].get("available")]
    text_predicted_break_count = sum(int(row["text_plan_grade"].get("predicted_region_break_count", 0)) for row in text_available)
    text_reference_break_count = sum(int(row["text_plan_grade"].get("reference_region_break_count", 0)) for row in text_available)
    text_matched_break_count = sum(int(row["text_plan_grade"].get("matched_region_break_count", 0)) for row in text_available)
    text_reference_phone_count = sum(int(row["text_plan_grade"].get("reference_phone_count", 0)) for row in text_available)
    text_predicted_phone_count = sum(int(row["text_plan_grade"].get("predicted_phone_count", 0)) for row in text_available)
    text_predicted_syllable_count = sum(int(row["text_plan_grade"].get("predicted_syllable_count", 0)) for row in text_available)
    text_reference_syllable_count = sum(int(row["text_plan_grade"].get("reference_syllable_count", 0)) for row in text_available)
    text_matched_syllable_count = sum(int(row["text_plan_grade"].get("matched_syllable_count", 0)) for row in text_available)
    text_nucleus_exact_count = sum(int(row["text_plan_grade"].get("nucleus_exact_match_count", 0)) for row in text_available)
    text_nucleus_base_count = sum(int(row["text_plan_grade"].get("nucleus_base_match_count", 0)) for row in text_available)
    text_syllable_ungradeable_spn_words = sum(
        int(row["text_plan_grade"].get("syllable_ungradeable_spn_word_count", 0))
        for row in text_available
    )
    text_reference_word_count = sum(int(row["text_plan_grade"].get("reference_word_count", 0)) for row in text_available)
    text_phone_duration_match_count = sum(
        int(row["text_plan_grade"].get("phone_prior_duration_match_count", 0))
        for row in text_available
    )
    text_phone_duration_error_sum = sum(
        float(row["text_plan_grade"].get("mean_abs_phone_prior_duration_error_ms", 0.0))
        * int(row["text_plan_grade"].get("phone_prior_duration_match_count", 0))
        for row in text_available
    )
    text_predicted_pause_hint_count = sum(
        int(row["text_plan_grade"].get("predicted_pause_hint_count", 0))
        for row in text_available
    )
    text_reference_pause_hint_count = sum(
        int(row["text_plan_grade"].get("reference_pause_hint_count", 0))
        for row in text_available
    )
    text_matched_pause_hint_count = sum(
        int(row["text_plan_grade"].get("matched_pause_hint_count", 0))
        for row in text_available
    )
    text_phone_exact_count = sum(
        int(round(float(row["text_plan_grade"].get("phone_exact_match_rate", 0.0)) * max(int(row["text_plan_grade"].get("reference_phone_count", 0)), 0)))
        for row in text_available
    )
    text_phone_base_count = sum(
        int(round(float(row["text_plan_grade"].get("phone_base_match_rate", 0.0)) * max(int(row["text_plan_grade"].get("reference_phone_count", 0)), 0)))
        for row in text_available
    )
    text_unmatched_predicted_phones: dict[str, int] = {}
    text_unmatched_reference_phones: dict[str, int] = {}
    text_phone_mismatches_by_word: dict[str, dict[str, int]] = {}
    for row in text_available:
        for phone, count in row["text_plan_grade"].get("unmatched_predicted_base_phones", {}).items():
            text_unmatched_predicted_phones[phone] = text_unmatched_predicted_phones.get(phone, 0) + int(count)
        for phone, count in row["text_plan_grade"].get("unmatched_reference_base_phones", {}).items():
            text_unmatched_reference_phones[phone] = text_unmatched_reference_phones.get(phone, 0) + int(count)
        for word, values in row["text_plan_grade"].get("phone_mismatches_by_word", {}).items():
            totals = text_phone_mismatches_by_word.setdefault(
                word,
                {"occurrences": 0, "false_positive": 0, "false_negative": 0, "spn_reference": 0},
            )
            for key in totals:
                totals[key] += int(values.get(key, 0))
    text_word_exact_count = sum(
        int(round(float(row["text_plan_grade"].get("word_exact_match_rate", 0.0)) * max(int(row["text_plan_grade"].get("reference_word_count", 0)), 0)))
        for row in text_available
    )
    region_boundary_available = [
        row for row in graded
        if row["streaming_region_boundary_grade"].get("available")
    ]
    region_boundary_predicted = sum(
        int(row["streaming_region_boundary_grade"].get("predicted_pair_count", 0))
        for row in region_boundary_available
    )
    region_boundary_reference = sum(
        int(row["streaming_region_boundary_grade"].get("reference_pair_count", 0))
        for row in region_boundary_available
    )
    region_boundary_matched = sum(
        int(row["streaming_region_boundary_grade"].get("matched_pair_count", 0))
        for row in region_boundary_available
    )
    region_candidate_count = sum(
        int(row["streaming_region_boundary_grade"].get("candidate_pair_count", 0))
        for row in region_boundary_available
    )
    region_candidate_matched = sum(
        int(row["streaming_region_boundary_grade"].get("candidate_matched_pair_count", 0))
        for row in region_boundary_available
    )
    region_adjudication_misses = sum(
        int(row["streaming_region_boundary_grade"].get("adjudication_miss_count", 0))
        for row in region_boundary_available
    )
    region_no_candidate = sum(
        int(row["streaming_region_boundary_grade"].get("no_candidate_count", 0))
        for row in region_boundary_available
    )
    region_false_accepted = sum(
        int(row["streaming_region_boundary_grade"].get("false_accepted_pair_count", 0))
        for row in region_boundary_available
    )
    region_initial_observed = sum(
        bool(row["streaming_region_boundary_grade"].get("initial_open_observed"))
        for row in region_boundary_available
    )
    region_initial_matched = sum(
        bool(row["streaming_region_boundary_grade"].get("initial_open_matched"))
        for row in region_boundary_available
    )
    region_final_observed = sum(
        bool(row["streaming_region_boundary_grade"].get("final_close_observed"))
        for row in region_boundary_available
    )
    region_final_matched = sum(
        bool(row["streaming_region_boundary_grade"].get("final_close_matched"))
        for row in region_boundary_available
    )
    region_case_categories = {
        "perfect": 0,
        "equal_count_mismatch": 0,
        "undersegmented": 0,
        "oversegmented": 0,
    }
    for row in region_boundary_available:
        grade = row["streaming_region_boundary_grade"]
        predicted = int(grade.get("predicted_pair_count", 0))
        reference = int(grade.get("reference_pair_count", 0))
        matched = int(grade.get("matched_pair_count", 0))
        if predicted == reference == matched:
            category = "perfect"
        elif predicted == reference:
            category = "equal_count_mismatch"
        elif predicted < reference:
            category = "undersegmented"
        else:
            category = "oversegmented"
        region_case_categories[category] += 1

    audio_health_rows = [
        row["runtime_health"]
        for row in graded
        if row.get("runtime_health", {}).get("gold_available")
    ]
    resume_cursor_targets = sum(int(row.get("resume_cursor_target_count", 0)) for row in audio_health_rows)
    resume_cursor_observed = sum(int(row.get("resume_cursor_observed_count", 0)) for row in audio_health_rows)
    resume_cursor_exact = sum(int(row.get("resume_cursor_exact_count", 0)) for row in audio_health_rows)
    gate_gold_speech_sec = sum(float(row.get("gold_speech_sec", 0.0)) for row in audio_health_rows)
    gate_gold_pause_sec = sum(float(row.get("gold_pause_sec", 0.0)) for row in audio_health_rows)
    gate_closed_speech_ms = sum(float(row.get("gate_closed_during_gold_speech_ms", 0.0)) for row in audio_health_rows)
    gate_open_pause_ms = sum(float(row.get("gate_open_during_gold_pause_ms", 0.0)) for row in audio_health_rows)
    audio_health_summary = {
        "audio_health_available_cases": len(audio_health_rows),
        "audio_health_cross_region_word_split_cases": sum(
            int(row.get("cross_region_word_split_count", 0) > 0) for row in audio_health_rows
        ),
        "audio_health_cross_region_word_split_count": sum(
            int(row.get("cross_region_word_split_count", 0)) for row in audio_health_rows
        ),
        "audio_health_first_region_starvation_cases": sum(
            int(row.get("first_region_starved", False)) for row in audio_health_rows
        ),
        "audio_health_first_region_animation_delay_ms": _mean([
            float(row.get("first_region_animation_delay_ms", 0.0)) for row in audio_health_rows
        ]),
        "audio_health_first_region_animation_delay_median_ms": _median([
            float(row.get("first_region_animation_delay_ms", 0.0)) for row in audio_health_rows
        ]),
        "audio_health_first_region_animation_delay_p90_ms": _p90([
            float(row.get("first_region_animation_delay_ms", 0.0)) for row in audio_health_rows
        ]),
        "audio_health_resume_cursor_target_count": resume_cursor_targets,
        "audio_health_resume_cursor_observed_count": resume_cursor_observed,
        "audio_health_resume_cursor_exact_rate": resume_cursor_exact / resume_cursor_targets if resume_cursor_targets else 1.0,
        "audio_health_resume_cursor_behind_count": sum(
            int(row.get("resume_cursor_behind_count", 0)) for row in audio_health_rows
        ),
        "audio_health_resume_cursor_ahead_count": sum(
            int(row.get("resume_cursor_ahead_count", 0)) for row in audio_health_rows
        ),
        "audio_health_resume_cursor_mean_abs_word_error": (
            sum(
                float(row.get("resume_cursor_mean_abs_word_error", 0.0))
                * int(row.get("resume_cursor_observed_count", 0))
                for row in audio_health_rows
            ) / resume_cursor_observed
            if resume_cursor_observed else 0.0
        ),
        "audio_health_uncommitted_visible_cases": sum(
            int(row.get("uncommitted_visible_count", 0) > 0) for row in audio_health_rows
        ),
        "audio_health_uncommitted_visible_count": sum(
            int(row.get("uncommitted_visible_count", 0)) for row in audio_health_rows
        ),
        "audio_health_uncommitted_visible_suffix_count": sum(
            int(row.get("uncommitted_visible_suffix_count", 0)) for row in audio_health_rows
        ),
        "audio_health_gate_closed_during_gold_speech_ms": gate_closed_speech_ms,
        "audio_health_gate_closed_during_gold_speech_rate": (
            gate_closed_speech_ms / (gate_gold_speech_sec * 1000.0) if gate_gold_speech_sec else 0.0
        ),
        "audio_health_gate_open_during_gold_pause_ms": gate_open_pause_ms,
        "audio_health_gate_open_during_gold_pause_rate": (
            gate_open_pause_ms / (gate_gold_pause_sec * 1000.0) if gate_gold_pause_sec else 0.0
        ),
    }

    playback_summary = {
        **audio_health_summary,
        "cases": len(rows),
        "graded_cases": len(graded),
        "ungraded_cases": len(ungraded),
        "qualified_cases": len(qualified),
        "bulk_cases": len(bulk),
        "degenerate_cases": sum(1 for row in graded if int(row["grade"].get("matched_count", 0)) == 0),
        "order_fail_cases": sum(1 for row in graded if int(row["grade"].get("order_violations", 0)) > 0),
        "speech_region_count_mismatch_cases": sum(1 for row in graded if speech_mismatch(row)),
        "visible_speech_region_count_mismatch_cases": 0,
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
        "list_word_count": list_word_count,
        "list_word_start_ms": _mean(list_word_start_errors),
        "list_word_start_median_ms": _median(list_word_start_errors),
        "phoneme_coverage_rate": phoneme_matches / max(phoneme_refs, 1),
        "phoneme_center_ms": _mean(phoneme_center_values),
        "phoneme_start_ms": _mean(phoneme_start_values),
        "phoneme_end_ms": _mean(phoneme_end_values),
        # Both names describe projected visible events, not phone recognition.
        "visible_articulation_coverage_rate": phoneme_matches / max(phoneme_refs, 1),
        "visible_articulation_center_ms": _mean(phoneme_center_values),
        "visible_articulation_start_ms": _mean(phoneme_start_values),
        "visible_articulation_end_ms": _mean(phoneme_end_values),
        "intra_word_coverage_rate": intra_word_matches / max(intra_word_refs, 1),
        "intra_word_center_ms": _mean(intra_word_center_values),
        "text_phone_plan_count": total_phone_plan_count,
        "visible_articulation_count": total_visible_plan_count,
        "visible_articulation_rate": total_visible_plan_count / max(total_phone_plan_count, 1),
        "renderable_articulation_event_count": total_renderable_event_count,
        "gold_corresponded_articulation_event_count": total_gold_corresponded_event_count,
        "unmatched_articulation_event_count": max(
            0, total_renderable_event_count - total_gold_corresponded_event_count
        ),
        "visual_gold_correspondence_rate": (
            total_gold_corresponded_event_count / max(total_renderable_event_count, 1)
        ),
        "visual_gold_correspondence_by_phone": correspondence_by_phone,
        "visual_role_counts": visual_role_counts,
        "visual_projection_by_phone": source_phone_counts,
        "pronunciation_word_count": len(pronunciation_words),
        "pronunciation_alternate_available_word_count": len(pronunciation_alternate_words),
        "pronunciation_alternate_selected_word_count": len(pronunciation_selected_words),
        "pronunciation_alternate_selected_by_word": pronunciation_selected_by_word,
        "region_overrun_ms": _mean(underrun_tail_values),
        "region_early_leak_ms": _mean(overrun_lead_values),
        "direct_aligner_available": False,
        "renderable_event_rate": (
            sum(int(row["playback_health"].get("event_count", 0)) - int(row["playback_health"].get("expired_count", 0)) for row in playback_health_available)
            / max(sum(int(row["playback_health"].get("event_count", 0)) for row in playback_health_available), 1)
        ),
        "late_commit_event_count": sum(int(row["playback_health"].get("late_start_count", 0)) for row in playback_health_available),
        "expired_commit_event_count": sum(int(row["playback_health"].get("expired_count", 0)) for row in playback_health_available),
        "scheduled_after_speech_event_count": sum(int(row["playback_health"].get("scheduled_after_speech_count", 0)) for row in playback_health_available),
        "scheduled_tail_overrun_ms": _mean([float(row["playback_health"].get("scheduled_tail_overrun_ms", 0.0)) for row in playback_health_available]),
        "late_commit_burst_count": sum(int(row["playback_health"].get("late_burst_count", 0)) for row in playback_health_available),
        "compressed_cadence_event_count": sum(int(row["playback_health"].get("compressed_cadence_count", 0)) for row in playback_health_available),
        "compressed_cadence_case_count": sum(1 for row in playback_health_available if int(row["playback_health"].get("compressed_cadence_count", 0)) > 0),
        "max_compressed_cadence_run": max([int(row["playback_health"].get("longest_compressed_cadence_run", 0)) for row in playback_health_available] + [0]),
        "speech_animation_coverage_rate": _mean([float(row["playback_health"].get("speech_animation_coverage_rate", 0.0)) for row in playback_health_available]),
        "speech_animation_dropout_ms": _mean([float(row["playback_health"].get("speech_animation_dropout_ms", 0.0)) for row in playback_health_available]),
        "longest_speech_animation_dropout_ms": _mean([float(row["playback_health"].get("longest_speech_animation_dropout_ms", 0.0)) for row in playback_health_available]),
        "playback_health_failure_cases": sum(
            1
            for row in playback_health_available
            if int(row["playback_health"].get("expired_count", 0)) > 0
            or int(row["playback_health"].get("late_burst_count", 0)) > 0
            or int(row["playback_health"].get("scheduled_after_speech_count", 0)) > 0
        ),
    }

    text_plan_summary = {
        "text_plan_available_cases": len(text_available),
        "text_word_sequence_exact_rate": text_word_exact_count / max(text_reference_word_count, 1),
        "text_phone_exact_rate": text_phone_exact_count / max(text_reference_phone_count, 1),
        "text_phone_base_rate": text_phone_base_count / max(text_reference_phone_count, 1),
        "text_phone_exact_precision": text_phone_exact_count / max(text_predicted_phone_count, 1),
        "text_phone_exact_recall": text_phone_exact_count / max(text_reference_phone_count, 1),
        "text_phone_exact_f1": 2.0 * text_phone_exact_count / max(text_predicted_phone_count + text_reference_phone_count, 1),
        "text_phone_base_precision": text_phone_base_count / max(text_predicted_phone_count, 1),
        "text_phone_base_recall": text_phone_base_count / max(text_reference_phone_count, 1),
        "text_phone_base_f1": 2.0 * text_phone_base_count / max(text_predicted_phone_count + text_reference_phone_count, 1),
        "text_phone_predicted_count": text_predicted_phone_count,
        "text_phone_reference_count": text_reference_phone_count,
        "text_syllable_predicted_count": text_predicted_syllable_count,
        "text_syllable_reference_count": text_reference_syllable_count,
        "text_syllable_count_ratio": text_predicted_syllable_count / max(text_reference_syllable_count, 1),
        "text_syllable_count_precision": text_matched_syllable_count / max(text_predicted_syllable_count, 1),
        "text_syllable_count_recall": text_matched_syllable_count / max(text_reference_syllable_count, 1),
        "text_nucleus_exact_precision": text_nucleus_exact_count / max(text_predicted_syllable_count, 1),
        "text_nucleus_exact_recall": text_nucleus_exact_count / max(text_reference_syllable_count, 1),
        "text_nucleus_exact_f1": 2.0 * text_nucleus_exact_count / max(text_predicted_syllable_count + text_reference_syllable_count, 1),
        "text_nucleus_base_precision": text_nucleus_base_count / max(text_predicted_syllable_count, 1),
        "text_nucleus_base_recall": text_nucleus_base_count / max(text_reference_syllable_count, 1),
        "text_nucleus_base_f1": 2.0 * text_nucleus_base_count / max(text_predicted_syllable_count + text_reference_syllable_count, 1),
        "text_syllable_ungradeable_spn_word_count": text_syllable_ungradeable_spn_words,
        "text_phone_unmatched_predicted_by_phone": dict(sorted(text_unmatched_predicted_phones.items(), key=lambda item: (-item[1], item[0]))),
        "text_phone_unmatched_reference_by_phone": dict(sorted(text_unmatched_reference_phones.items(), key=lambda item: (-item[1], item[0]))),
        "text_phone_mismatches_by_word": dict(sorted(
            text_phone_mismatches_by_word.items(),
            key=lambda item: (-(item[1]["false_positive"] + item[1]["false_negative"]), item[0]),
        )),
        "text_pause_class_accuracy": _mean([float(row["text_plan_grade"].get("pause_class_accuracy", 0.0)) for row in text_available]),
        "text_phone_duration_mae_ms": text_phone_duration_error_sum / max(text_phone_duration_match_count, 1),
        "text_phone_duration_match_count": text_phone_duration_match_count,
        "text_pause_hint_predicted_count": text_predicted_pause_hint_count,
        "text_pause_hint_reference_count": text_reference_pause_hint_count,
        "text_pause_hint_matched_count": text_matched_pause_hint_count,
        "text_pause_hint_precision": text_matched_pause_hint_count / max(text_predicted_pause_hint_count, 1),
        "text_pause_hint_recall": text_matched_pause_hint_count / max(text_reference_pause_hint_count, 1),
        "text_pause_hint_f1": 2.0 * text_matched_pause_hint_count / max(
            text_predicted_pause_hint_count + text_reference_pause_hint_count,
            1,
        ),
        "text_region_break_precision": text_matched_break_count / max(text_predicted_break_count, 1),
        "text_region_break_recall": text_matched_break_count / max(text_reference_break_count, 1),
        "text_word_prior_start_ms": _mean([float(row["text_plan_grade"].get("mean_abs_word_prior_start_error_ms", 0.0)) for row in text_available]),
        "text_phone_prior_center_ms": _mean([float(row["text_plan_grade"].get("mean_abs_phone_prior_center_error_ms", 0.0)) for row in text_available]),
    }

    def aggregate_evidence(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
        evidence_types = sorted({
            landmark_type
            for row in rows
            for landmark_type in row[key].get("types", {})
        })
        result: dict[str, dict[str, float]] = {}
        for landmark_type in evidence_types:
            target_count = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("target_count", 0)) for row in rows)
            observation_count = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("observation_count", 0)) for row in rows)
            matched_count = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("matched_count", 0)) for row in rows)
            region_count = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("region_count", 0)) for row in rows)
            exact_region_count = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("exact_region_count", 0)) for row in rows)
            region_count_abs_error = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("region_count_abs_error", 0)) for row in rows)
            clustered_120_targets = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("clustered_120_target_count", 0)) for row in rows)
            clustered_120_matches = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("clustered_120_matched_count", 0)) for row in rows)
            clustered_150_targets = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("clustered_150_target_count", 0)) for row in rows)
            clustered_150_matches = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("clustered_150_matched_count", 0)) for row in rows)
            clustered_200_targets = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("clustered_200_target_count", 0)) for row in rows)
            clustered_200_matches = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("clustered_200_matched_count", 0)) for row in rows)
            outside_speech_count = sum(int(row[key].get("types", {}).get(landmark_type, {}).get("outside_speech_observation_count", 0)) for row in rows)
            weighted_errors = []
            weighted_latencies = []
            for row in rows:
                stats = row[key].get("types", {}).get(landmark_type, {})
                count = int(stats.get("matched_count", 0))
                weighted_errors.extend([float(stats.get("mean_abs_error_ms", 0.0))] * count)
                weighted_latencies.extend([float(stats.get("mean_decision_latency_ms", 0.0))] * count)
            result[landmark_type] = {
                "target_count": target_count,
                "observation_count": observation_count,
                "matched_count": matched_count,
                "precision": matched_count / observation_count if observation_count else 0.0,
                "recall": matched_count / target_count if target_count else 0.0,
                "f1": (
                    2.0 * matched_count / (observation_count + target_count)
                    if observation_count + target_count else 0.0
                ),
                "mean_abs_error_ms": _mean(weighted_errors),
                "mean_decision_latency_ms": _mean(weighted_latencies),
                "count_ratio": observation_count / target_count if target_count else 0.0,
                "exact_region_count_rate": exact_region_count / region_count if region_count else 0.0,
                "region_count_mae": region_count_abs_error / region_count if region_count else 0.0,
                "clustered_120_recall": clustered_120_matches / clustered_120_targets if clustered_120_targets else 0.0,
                "clustered_150_recall": clustered_150_matches / clustered_150_targets if clustered_150_targets else 0.0,
                "clustered_200_recall": clustered_200_matches / clustered_200_targets if clustered_200_targets else 0.0,
                "outside_speech_rate": outside_speech_count / observation_count if observation_count else 0.0,
            }
        return result

    evidence_by_type = aggregate_evidence(evidence_available, "streaming_evidence_grade")
    conditioned_evidence_by_type = aggregate_evidence(
        conditioned_evidence_available,
        "streaming_conditioned_evidence_grade",
    )
    intra_envelope_phone_by_type = aggregate_evidence(
        intra_envelope_phone_available,
        "streaming_intra_envelope_phone_grade",
    )
    syllable_candidate_matched_pulses = sum(
        int(row["streaming_syllable_candidate_set_grade"].get("matched_pulse_count", 0))
        for row in syllable_candidate_set_available
    )
    syllable_candidate_unmatched_pulses = sum(
        int(row["streaming_syllable_candidate_set_grade"].get("unmatched_pulse_count", 0))
        for row in syllable_candidate_set_available
    )
    syllable_candidate_recall_at = {}
    for rank in range(1, 7):
        hit_count = sum(
            int(row["streaming_syllable_candidate_set_grade"]
                .get("recall_at", {}).get(str(rank), {}).get("hit_count", 0))
            for row in syllable_candidate_set_available
        )
        syllable_candidate_recall_at[str(rank)] = {
            "hit_count": hit_count,
            "recall": (
                hit_count / syllable_candidate_matched_pulses
                if syllable_candidate_matched_pulses else 0.0
            ),
        }
    syllable_assignment_targets = sum(
        int(row["streaming_syllable_assignment_grade"].get("target_count", 0))
        for row in syllable_assignment_available
    )
    syllable_assignment_count = sum(
        int(row["streaming_syllable_assignment_grade"].get("assignment_count", 0))
        for row in syllable_assignment_available
    )
    syllable_assignment_exact = sum(
        int(row["streaming_syllable_assignment_grade"].get("exact_count", 0))
        for row in syllable_assignment_available
    )
    syllable_assignment_within_one = sum(
        int(row["streaming_syllable_assignment_grade"].get("within_one_count", 0))
        for row in syllable_assignment_available
    )
    syllable_position_targets = sum(int(row["streaming_syllable_position_grade"].get("target_count", 0)) for row in syllable_position_available)
    syllable_position_estimates = sum(int(row["streaming_syllable_position_grade"].get("estimate_count", 0)) for row in syllable_position_available)
    syllable_position_timing_matches = sum(int(row["streaming_syllable_position_grade"].get("timing_match_count", 0)) for row in syllable_position_available)
    syllable_position_exact = sum(int(row["streaming_syllable_position_grade"].get("exact_position_count", 0)) for row in syllable_position_available)
    syllable_position_within_one = sum(int(row["streaming_syllable_position_grade"].get("within_one_count", 0)) for row in syllable_position_available)
    syllable_position_word_matches = sum(int(row["streaming_syllable_position_grade"].get("word_match_count", 0)) for row in syllable_position_available)
    syllable_position_confident = sum(int(row["streaming_syllable_position_grade"].get("confident_count", 0)) for row in syllable_position_available)
    syllable_position_confident_exact = sum(int(row["streaming_syllable_position_grade"].get("confident_exact_count", 0)) for row in syllable_position_available)
    historical_anchor_targets = sum(int(row["streaming_historical_anchor_grade"].get("target_count", 0)) for row in historical_anchor_available)
    historical_anchor_estimates = sum(int(row["streaming_historical_anchor_grade"].get("estimate_count", 0)) for row in historical_anchor_available)
    historical_anchor_timing = sum(int(row["streaming_historical_anchor_grade"].get("timing_match_count", 0)) for row in historical_anchor_available)
    historical_anchor_exact = sum(int(row["streaming_historical_anchor_grade"].get("exact_position_count", 0)) for row in historical_anchor_available)
    historical_anchor_within_one = sum(int(row["streaming_historical_anchor_grade"].get("within_one_count", 0)) for row in historical_anchor_available)
    runtime_syllable_targets = sum(int(row["runtime_syllable_assignment_grade"].get("target_count", 0)) for row in runtime_syllable_available)
    runtime_syllable_observations = sum(int(row["runtime_syllable_assignment_grade"].get("observation_count", 0)) for row in runtime_syllable_available)
    runtime_syllable_matches = sum(int(row["runtime_syllable_assignment_grade"].get("matched_count", 0)) for row in runtime_syllable_available)
    runtime_syllable_errors = []
    for row in runtime_syllable_available:
        grade = row["runtime_syllable_assignment_grade"]
        count = int(grade.get("matched_count", 0))
        if count > 0:
            runtime_syllable_errors.extend([float(grade.get("mean_abs_error_ms", 0.0))] * count)
    summary = {
        **playback_summary,
        **text_plan_summary,
        "evidence_surface_available_cases": len(evidence_available),
        "evidence_surface_preroll_ms": _mean([float(row["streaming_evidence_grade"].get("preroll_ms", 0.0)) for row in evidence_available]),
        "evidence_surface_postroll_ms": _mean([float(row["streaming_evidence_grade"].get("postroll_ms", 0.0)) for row in evidence_available]),
        "evidence_surface_by_type": evidence_by_type,
        "conditioned_evidence_surface_available_cases": len(conditioned_evidence_available),
        "conditioned_evidence_surface_by_type": conditioned_evidence_by_type,
        "intra_envelope_phone_available_cases": len(intra_envelope_phone_available),
        "intra_envelope_phone_by_type": intra_envelope_phone_by_type,
        "syllable_candidate_set_available_cases": len(syllable_candidate_set_available),
        "syllable_candidate_set_matched_pulses": syllable_candidate_matched_pulses,
        "syllable_candidate_set_unmatched_pulses": syllable_candidate_unmatched_pulses,
        "syllable_candidate_set_recall_at": syllable_candidate_recall_at,
        "syllable_assignment_available_cases": len(syllable_assignment_available),
        "syllable_assignment_target_count": syllable_assignment_targets,
        "syllable_assignment_count": syllable_assignment_count,
        "syllable_assignment_exact_precision": (
            syllable_assignment_exact / syllable_assignment_count
            if syllable_assignment_count else 0.0
        ),
        "syllable_assignment_exact_recall": (
            syllable_assignment_exact / syllable_assignment_targets
            if syllable_assignment_targets else 0.0
        ),
        "syllable_assignment_within_one_accuracy": (
            syllable_assignment_within_one / syllable_assignment_count
            if syllable_assignment_count else 0.0
        ),
        "syllable_position_available_cases": len(syllable_position_available),
        "syllable_position_target_count": syllable_position_targets,
        "syllable_position_estimate_count": syllable_position_estimates,
        "syllable_position_timing_precision": syllable_position_timing_matches / syllable_position_estimates if syllable_position_estimates else 0.0,
        "syllable_position_timing_recall": syllable_position_timing_matches / syllable_position_targets if syllable_position_targets else 0.0,
        "syllable_position_exact_rate": syllable_position_exact / syllable_position_estimates if syllable_position_estimates else 0.0,
        "syllable_position_within_one_rate": syllable_position_within_one / syllable_position_estimates if syllable_position_estimates else 0.0,
        "syllable_position_word_rate": syllable_position_word_matches / syllable_position_estimates if syllable_position_estimates else 0.0,
        "syllable_position_confident_coverage": syllable_position_confident / syllable_position_estimates if syllable_position_estimates else 0.0,
        "syllable_position_confident_exact_rate": syllable_position_confident_exact / syllable_position_confident if syllable_position_confident else 0.0,
        "historical_anchor_available_cases": len(historical_anchor_available),
        "historical_anchor_count": historical_anchor_estimates,
        "historical_anchor_exact_rate": historical_anchor_exact / historical_anchor_estimates if historical_anchor_estimates else 0.0,
        "historical_anchor_within_one_rate": historical_anchor_within_one / historical_anchor_estimates if historical_anchor_estimates else 0.0,
        "historical_anchor_timing_precision": historical_anchor_timing / historical_anchor_estimates if historical_anchor_estimates else 0.0,
        "historical_anchor_timing_recall": historical_anchor_timing / historical_anchor_targets if historical_anchor_targets else 0.0,
        "historical_anchor_median_latency_ms": _mean([float(row["streaming_historical_anchor_grade"].get("median_decision_latency_ms", 0.0)) for row in historical_anchor_available]),
        "historical_anchor_median_gap_ms": _mean([float(row["streaming_historical_anchor_grade"].get("median_anchor_gap_ms", 0.0)) for row in historical_anchor_available]),
        "streaming_region_boundary_available_cases": len(region_boundary_available),
        "streaming_region_boundary_tolerance_ms": 100.0,
        "streaming_region_boundary_predicted_count": region_boundary_predicted,
        "streaming_region_boundary_reference_count": region_boundary_reference,
        "streaming_region_boundary_matched_count": region_boundary_matched,
        "streaming_region_boundary_pair_precision": region_boundary_matched / region_boundary_predicted if region_boundary_predicted else (1.0 if region_boundary_reference == 0 else 0.0),
        "streaming_region_boundary_pair_recall": region_boundary_matched / region_boundary_reference if region_boundary_reference else 1.0,
        "streaming_region_boundary_pair_f1": 2.0 * region_boundary_matched / (region_boundary_predicted + region_boundary_reference) if region_boundary_predicted + region_boundary_reference else 1.0,
        "streaming_region_candidate_count": region_candidate_count,
        "streaming_region_candidate_matched_count": region_candidate_matched,
        "streaming_region_candidate_oracle_recall": region_candidate_matched / region_boundary_reference if region_boundary_reference else 1.0,
        "streaming_region_adjudication_miss_count": region_adjudication_misses,
        "streaming_region_no_candidate_count": region_no_candidate,
        "streaming_region_false_accepted_count": region_false_accepted,
        "streaming_region_boundary_case_categories": region_case_categories,
        "streaming_region_initial_open_precision": region_initial_matched / region_initial_observed if region_initial_observed else 0.0,
        "streaming_region_initial_open_recall": region_initial_matched / len(region_boundary_available) if region_boundary_available else 0.0,
        "streaming_region_final_close_precision": region_final_matched / region_final_observed if region_final_observed else 0.0,
        "streaming_region_final_close_recall": region_final_matched / len(region_boundary_available) if region_boundary_available else 0.0,
        "runtime_syllable_available_cases": len(runtime_syllable_available),
        "runtime_syllable_target_count": runtime_syllable_targets,
        "runtime_syllable_observation_count": runtime_syllable_observations,
        "runtime_syllable_matched_count": runtime_syllable_matches,
        "runtime_syllable_precision": runtime_syllable_matches / runtime_syllable_observations if runtime_syllable_observations else 0.0,
        "runtime_syllable_recall": runtime_syllable_matches / runtime_syllable_targets if runtime_syllable_targets else 0.0,
        "runtime_syllable_error_ms": _mean(runtime_syllable_errors),
        "runtime_syllable_count_ratio": runtime_syllable_observations / runtime_syllable_targets if runtime_syllable_targets else 0.0,
        "playback": playback_summary,
        "text_plan_vs_mfa": text_plan_summary,
    }

    summary["speech_f1"] = summary["speech_match_rate"]
    summary["word_f1"] = summary["word_match_rate"]

    return summary
