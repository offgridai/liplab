import csv
import json
import pathlib
import re
import statistics
import sys


START_TOLERANCE_MS = 100.0
REGION_NUCLEUS_TOLERANCE_MS = 50.0
MATERIAL_PAUSE_LEAKAGE_MS = 50
MAJORITY_HOLD_LOSS_RATE = 0.5
MIN_DISTINCT_VISEME_SPACING_MS = 20.0
MAX_NORMAL_COMMIT_LEAD_MS = 160.0
MATERIAL_PRIOR_SHIFT_MS = 150.0


def ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator else default


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def case_id(path: pathlib.Path) -> str:
    match = re.match(r"case_(\d+)_", path.parent.name)
    return match.group(1) if match else path.parent.name


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def punctuation_class(boundaries: list[dict], index: int) -> str:
    mark = boundaries[index].get("mark", "").strip()
    if mark != ",":
        return {
            ".": "hard_stop",
            "!": "hard_stop",
            "?": "hard_stop",
            ";": "semicolon",
            ":": "colon",
        }.get(mark, "none" if not mark else "other")

    word_index = int(boundaries[index].get("word_index", index))
    for later in boundaries[index + 1:]:
        later_mark = later.get("mark", "").strip()
        if not later_mark:
            continue
        later_word_index = int(later.get("word_index", word_index + 1))
        if later_mark == "," and later_word_index == word_index + 1:
            return "list_comma"
        if later_word_index > word_index + 1:
            return "standard_comma"
        return "other_comma"
    return "standard_comma"


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    cases: list[dict] = []
    words_out: list[dict] = []
    boundaries_out: list[dict] = []
    word_duration_ratios: list[float] = []
    word_duration_abs_errors_ms: list[float] = []
    word_duration_signed_errors_ms: list[float] = []
    word_interval_coverages: list[float] = []
    word_interval_precisions: list[float] = []
    word_interval_ious: list[float] = []
    word_interval_onset_abs_errors_ms: list[float] = []
    word_interval_onset_signed_errors_ms: list[float] = []
    word_interval_exit_abs_errors_ms: list[float] = []
    word_interval_exit_signed_errors_ms: list[float] = []
    bilabial_peak_abs_errors_ms: list[float] = []
    sentence_duration_ratios: list[float] = []
    viseme_run_share_errors: list[float] = []
    viseme_boundary_errors: list[float] = []
    viseme_word_total_variations: list[float] = []
    presentation_visible_events = 0
    presentation_eligible_events = 0
    presentation_visible_boundary_events = 0
    presentation_boundary_events = 0

    for focus_path in sorted(latest.glob("*/focus_alignment_grade.json")):
        ownership_path = focus_path.parent / "region_ownership_grade.json"
        grade_path = focus_path.parent / "grade.json"
        if not ownership_path.exists() or not grade_path.exists():
            continue

        focus = load_json(focus_path)
        ownership = load_json(ownership_path)
        grade = load_json(grade_path)
        delivery_path = focus_path.parent / "runtime_delivery_grade.json"
        delivery = load_json(delivery_path) if delivery_path.exists() else {}
        word_duration_ratios.extend(
            float(value) for value in delivery.get("word_duration_ratios", [])
        )
        word_duration_abs_errors_ms.extend(
            float(value)
            for value in delivery.get("word_duration_abs_errors_ms", [])
        )
        word_duration_signed_errors_ms.extend(
            float(value)
            for value in delivery.get("word_duration_signed_errors_ms", [])
        )
        word_interval_coverages.extend(
            float(value) for value in delivery.get("word_interval_coverages", [])
        )
        word_interval_precisions.extend(
            float(value) for value in delivery.get("word_interval_precisions", [])
        )
        word_interval_ious.extend(
            float(value) for value in delivery.get("word_interval_ious", [])
        )
        word_interval_onset_abs_errors_ms.extend(
            float(value) for value in delivery.get("word_onset_abs_errors_ms", [])
        )
        word_interval_onset_signed_errors_ms.extend(
            float(value) for value in delivery.get("word_onset_signed_errors_ms", [])
        )
        word_interval_exit_abs_errors_ms.extend(
            float(value) for value in delivery.get("word_exit_abs_errors_ms", [])
        )
        word_interval_exit_signed_errors_ms.extend(
            float(value) for value in delivery.get("word_exit_signed_errors_ms", [])
        )
        phonetic_path = focus_path.parent / "phonetic_presentation_grade.json"
        phonetic = load_json(phonetic_path) if phonetic_path.exists() else {}
        bilabial_peak_abs_errors_ms.extend(
            float(value)
            for value in phonetic.get("bilabial_peak_abs_errors_ms", [])
        )
        sentence_duration_ratios.extend(
            float(value) for value in delivery.get("sentence_duration_ratios", [])
        )
        proportion_path = focus_path.parent / "viseme_proportion_grade.json"
        proportion = load_json(proportion_path) if proportion_path.exists() else {}
        viseme_run_share_errors.extend(
            float(value) for value in proportion.get("run_share_abs_errors", [])
        )
        viseme_boundary_errors.extend(
            float(value)
            for value in proportion.get("boundary_position_abs_errors", [])
        )
        viseme_word_total_variations.extend(
            float(value) for value in proportion.get("word_total_variations", [])
        )
        visibility_path = (
            focus_path.parent / "presentation_event_visibility_grade.json"
        )
        visibility = load_json(visibility_path) if visibility_path.exists() else {}
        presentation_visible_events += int(
            visibility.get("robust_visible_event_count", 0)
        )
        presentation_eligible_events += int(
            visibility.get("eligible_event_count", 0)
        )
        presentation_visible_boundary_events += int(
            visibility.get("robust_boundary_event_count", 0)
        )
        presentation_boundary_events += int(
            visibility.get("boundary_event_count", 0)
        )
        current_case_name = focus_path.parent.name
        gold_dir = root / "inputs" / "gold" / current_case_name
        gold_boundaries = load_csv(gold_dir / "boundaries.csv")
        gold_words = {
            int(row["word_index"]): row for row in load_csv(gold_dir / "words.csv")
        }
        gold_speech = {
            int(row["index"]): row for row in load_csv(gold_dir / "speech.csv")
        }
        detected_speech = load_csv(focus_path.parent / "speech_regions.csv")
        committed_rows = load_csv(focus_path.parent / "committed.csv")
        word_onset_rows = load_csv(focus_path.parent / "word_onset_diagnostics.csv")
        runtime_word_rows = {
            int(row["word_index"]): row
            for row in load_csv(focus_path.parent / "runtime_word_delivery.csv")
        }
        ownership_words = {
            int(word["word_index"]): word for word in ownership.get("words", [])
        }
        word_heads = {
            int(word["word_index"]): word for word in focus.get("word_heads", [])
        }
        region_heads = {
            int(row["region_index"]): row for row in focus.get("region_heads", [])
        }
        region_nuclei = focus.get("region_nuclei", [])
        pause_boundaries = {
            (int(row["region_before"]), int(row["region_after"])): row
            for row in focus.get("pause_boundaries", [])
        }

        def detected_region_for_word(word_index: int) -> int:
            word = gold_words.get(word_index)
            if word is None:
                return -1
            center = (float(word["start"]) + float(word["end"])) * 0.5
            matches = [
                int(region["index"])
                for region in detected_speech
                if float(region["start"]) <= center <= float(region["end"]) + 0.001
            ]
            return matches[0] if len(matches) == 1 else -1

        def detected_region_mapped_to_mfa(detected_region_index: int) -> int:
            if detected_region_index < 0:
                return -1
            detected = next(
                (
                    region for region in detected_speech
                    if int(region["index"]) == detected_region_index
                ),
                None,
            )
            if detected is None:
                return -1
            start = float(detected["start"])
            end = float(detected["end"])
            best_index = -1
            best_overlap = 0.0
            for mfa_index, region in gold_speech.items():
                overlap = max(
                    0.0,
                    min(end, float(region["end"]))
                        - max(start, float(region["start"])),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = mfa_index
            return best_index

        # The first transcript word that MFA places in each region owns that
        # region's start. This join makes timing explicitly ownership-aware.
        first_word_by_region: dict[int, int] = {}
        for word_index, word in ownership_words.items():
            region_index = int(word.get("mfa_region_index", -1))
            if region_index >= 0:
                first_word_by_region.setdefault(region_index, word_index)
        region_expected = int(focus.get("expected_region_head_count", 0))
        region_successes = 0
        region_errors: list[float] = []
        for region_index, fallback_word_index in first_word_by_region.items():
            head = region_heads.get(region_index)
            word_index = int(head.get("word_index", fallback_word_index)) if head else fallback_word_index
            owner = ownership_words.get(word_index, {})
            if head is not None:
                error_ms = abs(float(head.get("error_ms", 0.0)))
                region_errors.append(error_ms)
                if owner.get("runtime_mfa_first_event_correct", False) and error_ms <= START_TOLERANCE_MS:
                    region_successes += 1

        nucleus_errors: list[float] = []
        nucleus_successes = 0
        for nucleus in region_nuclei:
            word_index = int(nucleus.get("word_index", -1))
            owner = ownership_words.get(word_index, {})
            error_ms = abs(float(nucleus.get("error_ms", 0.0)))
            nucleus_errors.append(error_ms)
            mfa_region = int(owner.get("mfa_region_index", -1))
            runtime_regions = sorted({
                int(value) for value in owner.get("runtime_regions", [])
            })
            runtime_mapped_regions = sorted({
                int(value)
                for value in owner.get("runtime_mapped_mfa_regions", [])
            })
            physical_runtime_region = detected_region_for_word(word_index)
            physical_runtime_mfa_region = detected_region_mapped_to_mfa(
                physical_runtime_region
            )
            fully_committed = (
                int(owner.get("planned_event_count", 0)) > 0
                and int(owner.get("planned_event_count", 0))
                    == int(owner.get("committed_event_count", 0))
            )
            if (
                fully_committed
                and len(runtime_regions) == 1
                and runtime_mapped_regions == [mfa_region]
                and physical_runtime_mfa_region == mfa_region
                and error_ms <= REGION_NUCLEUS_TOLERANCE_MS
            ):
                nucleus_successes += 1

        word_expected = int(focus.get("expected_word_head_count", 0))
        word_successes = 0
        word_errors: list[float] = []
        assignment_successes = 0
        three_level_word_successes = 0
        early_region_theft_count = 0
        late_region_assignment_count = 0
        materially_early_intact_word_count = 0
        max_intact_word_lead_ms = 0.0
        assigned_last_word_by_runtime_region: dict[int, int] = {}
        mfa_last_word_by_region: dict[int, int] = {}
        for expected_word_index, expected_owner in ownership_words.items():
            expected_region = int(expected_owner.get("mfa_region_index", -1))
            if expected_region >= 0:
                mfa_last_word_by_region[expected_region] = max(
                    mfa_last_word_by_region.get(expected_region, -1),
                    expected_word_index,
                )
        for word_index, owner in ownership_words.items():
            head = word_heads.get(word_index)
            interval = runtime_word_rows.get(word_index, {})
            strict_owner = bool(owner.get("runtime_mfa_strict_correct", False))
            head_owner = bool(owner.get("runtime_mfa_first_event_correct", strict_owner))
            planned = int(owner.get("planned_event_count", 0))
            committed = int(owner.get("committed_event_count", 0))
            fully_committed = planned > 0 and committed == planned
            planned_transcript_region = int(owner.get("transcript_region_index", -1))
            mfa_region = int(owner.get("mfa_region_index", -1))
            runtime_regions = sorted({
                int(value) for value in owner.get("runtime_regions", [])
            })
            runtime_mapped_regions = sorted({
                int(value)
                for value in owner.get("runtime_mapped_mfa_regions", [])
            })
            physical_runtime_region = detected_region_for_word(word_index)
            physical_runtime_mfa_region = detected_region_mapped_to_mfa(
                physical_runtime_region
            )
            transcript_mfa_correct = runtime_mapped_regions == [mfa_region]
            runtime_raw_mfa_correct = runtime_regions == [mfa_region]
            assigned_mfa_region = (
                runtime_mapped_regions[0]
                if len(runtime_mapped_regions) == 1
                else -1
            )
            region_assignment_delta = (
                assigned_mfa_region - mfa_region
                if assigned_mfa_region >= 0 and mfa_region >= 0
                else 0
            )
            if fully_committed and region_assignment_delta < 0:
                early_region_theft_count += 1
            if fully_committed and region_assignment_delta > 0:
                late_region_assignment_count += 1
            if fully_committed and len(runtime_regions) == 1:
                runtime_region = runtime_regions[0]
                assigned_last_word_by_runtime_region[runtime_region] = max(
                    assigned_last_word_by_runtime_region.get(runtime_region, -1),
                    word_index,
                )
            whole_word_three_level_success = (
                fully_committed
                and transcript_mfa_correct
                and len(runtime_regions) == 1
                and physical_runtime_mfa_region == mfa_region
            )
            if whole_word_three_level_success:
                three_level_word_successes += 1
            if fully_committed and strict_owner:
                assignment_successes += 1

            error_ms = None
            start_success = False
            if head is not None:
                signed_error_ms = float(head.get("error_ms", 0.0))
                error_ms = abs(signed_error_ms)
                word_errors.append(error_ms)
                start_success = head_owner and error_ms <= START_TOLERANCE_MS
                if start_success:
                    word_successes += 1
                if fully_committed and signed_error_ms < -START_TOLERANCE_MS:
                    materially_early_intact_word_count += 1
                    max_intact_word_lead_ms = max(
                        max_intact_word_lead_ms,
                        -signed_error_ms,
                    )
            else:
                signed_error_ms = None
            words_out.append({
                "case": case_id(focus_path),
                "case_name": focus_path.parent.name,
                "word_index": word_index,
                "word": owner.get("word", ""),
                "mfa_region_index": owner.get("mfa_region_index", -1),
                "planned_transcript_region_index": planned_transcript_region,
                "transcript_region_index": runtime_regions[0] if len(runtime_regions) == 1 else -1,
                "runtime_region_index": owner.get("runtime_region_index", -1),
                "physical_runtime_region_index": physical_runtime_region,
                "physical_runtime_mfa_region_index": physical_runtime_mfa_region,
                "runtime_regions": "|".join(str(value) for value in runtime_regions),
                "assigned_mfa_region_index": assigned_mfa_region,
                "region_assignment_delta": region_assignment_delta,
                "early_region_theft": fully_committed and region_assignment_delta < 0,
                "late_region_assignment": fully_committed and region_assignment_delta > 0,
                "fully_committed": fully_committed,
                "runtime_region_integrity": bool(owner.get("runtime_region_integrity", False)),
                "region_assignment_success": fully_committed and strict_owner,
                "transcript_mfa_region_correct": transcript_mfa_correct,
                "runtime_raw_mfa_region_correct": physical_runtime_mfa_region == mfa_region,
                "planned_transcript_mfa_region_correct": planned_transcript_region == mfa_region,
                "whole_word_three_level_success": whole_word_three_level_success,
                "start_error_ms": "" if error_ms is None else error_ms,
                "start_signed_error_ms": (
                    "" if signed_error_ms is None else signed_error_ms
                ),
                "materially_early_intact_word": (
                    fully_committed
                    and signed_error_ms is not None
                    and signed_error_ms < -START_TOLERANCE_MS
                ),
                "start_success": start_success,
                "runtime_start_sec": interval.get("runtime_start_sec", ""),
                "runtime_end_sec": interval.get("runtime_end_sec", ""),
                "onset_signed_error_ms": interval.get(
                    "onset_signed_error_ms", ""
                ),
                "exit_signed_error_ms": interval.get(
                    "exit_signed_error_ms", ""
                ),
                "spoken_interval_coverage": interval.get(
                    "spoken_interval_coverage", ""
                ),
                "animation_interval_precision": interval.get(
                    "animation_interval_precision", ""
                ),
                "interval_iou": interval.get("interval_iou", ""),
            })

        region_word_overrun_count = 0
        max_region_word_overrun = 0
        for runtime_region, assigned_last_word in assigned_last_word_by_runtime_region.items():
            mapped_mfa_region = detected_region_mapped_to_mfa(runtime_region)
            expected_last_word = mfa_last_word_by_region.get(mapped_mfa_region, -1)
            if expected_last_word < 0 or assigned_last_word <= expected_last_word:
                continue
            region_word_overrun_count += 1
            max_region_word_overrun = max(
                max_region_word_overrun,
                assigned_last_word - expected_last_word,
            )

        split_word_count = 0
        incomplete_word_count = 0
        for owner in ownership_words.values():
            planned = int(owner.get("planned_event_count", 0))
            committed = int(owner.get("committed_event_count", 0))
            runtime_regions = [int(value) for value in owner.get("runtime_regions", [])]
            if planned > 0 and committed != planned:
                incomplete_word_count += 1
            if len(set(runtime_regions)) > 1:
                split_word_count += 1

        owned_tail_rows = [
            row for row in committed_rows
            if row.get("reason", "") == "owned_word_tail_commit"
        ]
        owned_tail_word_count = len({
            int(row["word_index"]) for row in owned_tail_rows
            if row.get("word_index", "") != ""
        })
        committed_by_index = sorted(
            committed_rows,
            key=lambda row: int(row.get("index", -1)),
        )
        previous_committed: dict[int, dict] = {}
        owned_tail_compressed_transition_count = 0
        owned_tail_excessive_lead_count = 0
        owned_tail_material_prior_shift_count = 0
        for row in committed_by_index:
            word_index = int(row.get("word_index", -1))
            if row.get("reason", "") == "owned_word_tail_commit":
                previous = previous_committed.get(word_index)
                if previous is not None:
                    spacing_ms = (
                        float(row.get("center", 0.0))
                        - float(previous.get("center", 0.0))
                    ) * 1000.0
                    if spacing_ms < MIN_DISTINCT_VISEME_SPACING_MS:
                        owned_tail_compressed_transition_count += 1
                if float(row.get("commit_lead", 0.0)) * 1000.0 > MAX_NORMAL_COMMIT_LEAD_MS:
                    owned_tail_excessive_lead_count += 1
                prior_shift_ms = abs(
                    float(row.get("center", 0.0))
                    - float(row.get("prior_center", row.get("center", 0.0)))
                ) * 1000.0
                if prior_shift_ms >= MATERIAL_PRIOR_SHIFT_MS:
                    owned_tail_material_prior_shift_count += 1
            previous_committed[word_index] = row

        for boundary_index, boundary in enumerate(gold_boundaries):
            before_index = int(boundary.get("word_index", boundary_index))
            after_index = int(boundary.get("next_word_index", before_index + 1))
            before = ownership_words.get(before_index, {})
            after = ownership_words.get(after_index, {})
            before_regions = sorted({int(value) for value in before.get("runtime_regions", [])})
            after_regions = sorted({int(value) for value in after.get("runtime_regions", [])})
            before_complete = (
                int(before.get("planned_event_count", 0)) > 0
                and int(before.get("planned_event_count", 0))
                    == int(before.get("committed_event_count", 0))
            )
            after_complete = (
                int(after.get("planned_event_count", 0)) > 0
                and int(after.get("planned_event_count", 0))
                    == int(after.get("committed_event_count", 0))
            )
            scheduler_boundary_observable = (
                before_complete and after_complete
                and len(before_regions) == 1 and len(after_regions) == 1
            )
            scheduler_runtime_split = (
                scheduler_boundary_observable and before_regions[0] != after_regions[0]
            )
            detected_before = detected_region_for_word(before_index)
            detected_after = detected_region_for_word(after_index)
            boundary_observable = detected_before >= 0 and detected_after >= 0
            runtime_split = (
                boundary_observable and detected_before != detected_after
            )
            region_before = int(boundary.get("speech_region_index_before", -1))
            region_after = int(boundary.get("speech_region_index_after", -1))
            gold_split = region_before >= 0 and region_after >= 0 and region_before != region_after
            planned_transcript_region_before = int(before.get("transcript_region_index", -1))
            planned_transcript_region_after = int(after.get("transcript_region_index", -1))
            planned_transcript_split = (
                planned_transcript_region_before >= 0
                and planned_transcript_region_after >= 0
                and planned_transcript_region_before != planned_transcript_region_after
            )
            transcript_region_before = before_regions[0] if len(before_regions) == 1 else -1
            transcript_region_after = after_regions[0] if len(after_regions) == 1 else -1
            transcript_split = scheduler_runtime_split
            exact_three_level_boundary = (
                boundary_observable
                and scheduler_boundary_observable
                and gold_split == transcript_split
                and gold_split == runtime_split
            )
            pause = pause_boundaries.get((region_before, region_after), {})
            pause_duration_ms = int(pause.get("duration_ms", 0))
            pause_leakage_ms = int(pause.get("animation_leakage_ms", 0))
            leakage_rate = ratio(pause_leakage_ms, pause_duration_ms, 0.0)
            resume_head = region_heads.get(region_after, {}) if gold_split else {}
            resume_early_ms = max(0.0, float(resume_head.get("resume_in_pause_ms", 0.0)))
            pause_start = float(gold_speech.get(region_before, {}).get("end", 0.0))
            before_owned_tail = [
                row for row in owned_tail_rows
                if int(row.get("word_index", -1)) == before_index
            ]
            owned_tail_in_pause = sum(
                1 for row in before_owned_tail
                if float(row.get("center", -1.0)) >= pause_start - 0.0005
            ) if gold_split else 0
            post_word_in_previous_region = (
                gold_split and len(before_regions) == 1 and len(after_regions) == 1
                and before_regions[0] == after_regions[0]
            )
            material_leakage = (
                pause_leakage_ms >= MATERIAL_PAUSE_LEAKAGE_MS
            )
            majority_hold_lost = leakage_rate >= MAJORITY_HOLD_LOSS_RATE
            boundaries_out.append({
                "case": case_id(focus_path),
                "case_name": current_case_name,
                "boundary_index": boundary_index,
                "word_before_index": before_index,
                "word_before": boundary.get("word", ""),
                "word_after_index": after_index,
                "word_after": boundary.get("next_word", ""),
                "mark": boundary.get("mark", ""),
                "punctuation_class": punctuation_class(gold_boundaries, boundary_index),
                "gold_region_before": region_before,
                "gold_region_after": region_after,
                "gold_pause_boundary": gold_split,
                "transcript_region_before": transcript_region_before,
                "transcript_region_after": transcript_region_after,
                "transcript_pause_boundary": transcript_split,
                "planned_transcript_region_before": planned_transcript_region_before,
                "planned_transcript_region_after": planned_transcript_region_after,
                "planned_transcript_pause_boundary": planned_transcript_split,
                "gold_pause_duration_ms": pause_duration_ms,
                "runtime_boundary_observable": boundary_observable,
                "runtime_region_before": detected_before,
                "runtime_region_after": detected_after,
                "runtime_pause_boundary": runtime_split,
                "scheduler_boundary_observable": scheduler_boundary_observable,
                "scheduler_region_before": before_regions[0] if len(before_regions) == 1 else -1,
                "scheduler_region_after": after_regions[0] if len(after_regions) == 1 else -1,
                "scheduler_pause_boundary": scheduler_runtime_split,
                "exact_three_level_boundary": exact_three_level_boundary,
                "transcript_mfa_boundary_mismatch": transcript_split != gold_split,
                "planned_transcript_mfa_boundary_mismatch": planned_transcript_split != gold_split,
                "runtime_mfa_boundary_mismatch": not boundary_observable or runtime_split != gold_split,
                "runtime_transcript_boundary_mismatch": (
                    not boundary_observable
                    or not scheduler_boundary_observable
                    or runtime_split != transcript_split
                ),
                "missed_pause_boundary": gold_split and not runtime_split,
                "extra_pause_boundary": not gold_split and runtime_split,
                "post_word_in_previous_region": post_word_in_previous_region,
                "resume_early_ms": resume_early_ms,
                "pause_animation_leakage_ms": pause_leakage_ms,
                "pause_animation_leakage_rate": leakage_rate,
                "material_pause_leakage": material_leakage,
                "majority_hold_lost": majority_hold_lost,
                "owned_tail_event_count": len(before_owned_tail),
                "owned_tail_events_in_pause": owned_tail_in_pause,
                "adjacent_incomplete_word": not before_complete or not after_complete,
                "adjacent_split_word": len(before_regions) > 1 or len(after_regions) > 1,
                "hold_failure": gold_split and (
                    not runtime_split or material_leakage or resume_early_ms > 0.0
                ),
            })

        pause_duration = int(focus.get("pause_duration_ms", 0))
        pause_leakage = int(focus.get("pause_animation_leakage_ms", 0))
        planned_events = int(focus.get("planned_event_count", 0))
        committed_events = int(focus.get("committed_event_count", 0))
        assignment_expected = int(ownership.get("mfa_word_count", len(ownership_words)))
        animation_onset_expected = len(word_onset_rows)
        animation_onset_matched = 0
        animation_onset_successes = 0
        animation_onset_errors: list[float] = []
        for onset in word_onset_rows:
            if as_bool(onset.get("missing_event", False)):
                continue
            word_index = int(onset.get("word_index", -1))
            owner = ownership_words.get(word_index, {})
            error_ms = abs(float(onset.get("error_ms", 0.0)))
            animation_onset_matched += 1
            animation_onset_errors.append(error_ms)
            if (
                bool(owner.get("runtime_mfa_first_event_correct", False))
                and error_ms <= START_TOLERANCE_MS
            ):
                animation_onset_successes += 1
        case_boundaries = [
            row for row in boundaries_out if row["case_name"] == current_case_name
        ]
        exact_boundary_count = sum(
            int(bool(row["exact_three_level_boundary"])) for row in case_boundaries
        )
        cases.append({
            "case": case_id(focus_path),
            "case_name": focus_path.parent.name,
            "region_start_expected": region_expected,
            "region_start_matched": len(region_errors),
            "region_start_successes": region_successes,
            "region_start_success_rate": ratio(region_successes, region_expected),
            "region_start_mean_abs_error_ms": ratio(sum(region_errors), len(region_errors), 0.0),
            "region_nucleus_expected": len(first_word_by_region),
            "region_nucleus_matched": len(nucleus_errors),
            "region_nucleus_successes": nucleus_successes,
            "region_nucleus_success_rate": ratio(
                nucleus_successes, len(first_word_by_region)
            ),
            "region_nucleus_mean_abs_error_ms": ratio(
                sum(nucleus_errors), len(nucleus_errors), 0.0
            ),
            "pause_duration_ms": pause_duration,
            "pause_leakage_ms": pause_leakage,
            "pause_clean_rate": 1.0 - ratio(pause_leakage, pause_duration, 0.0),
            "word_start_expected": word_expected,
            "word_start_matched": len(word_errors),
            "word_start_successes": word_successes,
            "word_start_success_rate": ratio(word_successes, word_expected),
            "word_start_mean_abs_error_ms": ratio(sum(word_errors), len(word_errors), 0.0),
            "animation_onset_expected": animation_onset_expected,
            "animation_onset_matched": animation_onset_matched,
            "animation_onset_successes": animation_onset_successes,
            "animation_onset_mean_abs_error_ms": ratio(
                sum(animation_onset_errors), animation_onset_matched, 0.0
            ),
            "decoded_viseme_reference": int(
                grade.get("intra_word_alignment", {}).get("reference_count", 0)
            ),
            "decoded_viseme_matched": int(
                grade.get("intra_word_alignment", {}).get("matched_count", 0)
            ),
            "decoded_viseme_missing": int(
                grade.get("intra_word_alignment", {}).get("missing_count", 0)
            ),
            "decoded_viseme_extra": int(
                grade.get("intra_word_alignment", {}).get("extra_count", 0)
            ),
            "decoded_viseme_center_mae_ms": float(
                grade.get("intra_word_alignment", {}).get(
                    "mean_abs_center_error_ms", 0.0
                )
            ),
            "word_region_expected": assignment_expected,
            "word_region_successes": assignment_successes,
            "word_region_assignment_rate": ratio(assignment_successes, assignment_expected),
            "three_level_word_successes": three_level_word_successes,
            "three_level_word_assignment_rate": ratio(
                three_level_word_successes, assignment_expected
            ),
            "early_region_theft_count": early_region_theft_count,
            "late_region_assignment_count": late_region_assignment_count,
            "materially_early_intact_word_count": materially_early_intact_word_count,
            "max_intact_word_lead_ms": max_intact_word_lead_ms,
            "region_word_overrun_count": region_word_overrun_count,
            "max_region_word_overrun": max_region_word_overrun,
            "planned_event_count": planned_events,
            "committed_event_count": committed_events,
            "event_completion_rate": ratio(committed_events, planned_events),
            "delivery_planned_events": int(
                delivery.get("planned_candidate_events", 0)
            ),
            "delivery_committed_events": int(
                delivery.get("committed_candidate_events", 0)
            ),
            "delivered_events": int(delivery.get("delivered_events", 0)),
            "late_after_window_events": int(
                delivery.get("late_after_window_events", 0)
            ),
            "delivery_expected_words": int(delivery.get("expected_words", 0)),
            "delivery_missing_words": int(delivery.get("missing_words", 0)),
            "measured_word_durations": int(
                delivery.get("measured_word_durations", 0)
            ),
            "successful_word_durations": int(
                delivery.get("successful_word_durations", 0)
            ),
            "compressed_words": int(delivery.get("compressed_words", 0)),
            "severely_compressed_words": int(
                delivery.get("severely_compressed_words", 0)
            ),
            "stretched_words": int(delivery.get("stretched_words", 0)),
            "zero_overlap_words": int(delivery.get("zero_overlap_words", 0)),
            "low_coverage_words": int(delivery.get("low_coverage_words", 0)),
            "word_duration_ratio_mean": float(
                delivery.get("word_duration_ratio_mean", 0.0)
            ),
            "word_duration_ratio_median": float(
                delivery.get("word_duration_ratio_median", 0.0)
            ),
            "delivery_expected_speech_regions": int(
                delivery.get("expected_speech_regions", 0)
            ),
            "empty_speech_regions": int(
                delivery.get("empty_speech_regions", 0)
            ),
            "delivery_expected_sentences": int(
                delivery.get("expected_sentences", 0)
            ),
            "delivery_missing_sentences": int(
                delivery.get("missing_sentences", 0)
            ),
            "compressed_sentences": int(
                delivery.get("compressed_sentences", 0)
            ),
            "sentence_duration_ratio_mean": float(
                delivery.get("sentence_duration_ratio_mean", 0.0)
            ),
            "sentence_duration_ratio_median": float(
                delivery.get("sentence_duration_ratio_median", 0.0)
            ),
            "viseme_proportion_planned_words": int(
                proportion.get("planned_multi_viseme_words", 0)
            ),
            "viseme_proportion_gradeable_words": int(
                proportion.get("gradeable_words", 0)
            ),
            "viseme_proportion_ungradeable_words": int(
                proportion.get("ungradeable_words", 0)
            ),
            "viseme_proportion_zero_runtime_words": int(
                proportion.get("zero_runtime_words", 0)
            ),
            "viseme_proportion_severe_words": int(
                proportion.get("severe_words", 0)
            ),
            "phonetic_vowel_frames": int(phonetic.get("vowel_frames", 0)),
            "phonetic_vowel_target_dominant_frames": int(
                phonetic.get("vowel_target_dominant_frames", 0)
            ),
            "phonetic_vowel_same_word_dominant_frames": int(
                phonetic.get("vowel_same_word_dominant_frames", 0)
            ),
            "phonetic_vowel_foreign_word_dominant_frames": int(
                phonetic.get("vowel_foreign_word_dominant_frames", 0)
            ),
            "phonetic_low_dominance_vowels": int(
                phonetic.get("low_dominance_vowels", 0)
            ),
            "phonetic_foreign_intrusion_vowels": int(
                phonetic.get("foreign_intrusion_vowels", 0)
            ),
            "phonetic_bilabial_frames": int(phonetic.get("bilabial_frames", 0)),
            "phonetic_bilabial_target_dominant_frames": int(
                phonetic.get("bilabial_target_dominant_frames", 0)
            ),
            "phonetic_late_bilabial_peaks": int(
                phonetic.get("late_bilabial_peaks", 0)
            ),
            "phonetic_oh_frames": int(phonetic.get("oh_frames", 0)),
            "phonetic_oh_saturated_frames": int(
                phonetic.get("oh_saturated_frames", 0)
            ),
            "phonetic_other_vowel_frames": int(
                phonetic.get("other_vowel_frames", 0)
            ),
            "phonetic_other_vowel_saturated_frames": int(
                phonetic.get("other_vowel_saturated_frames", 0)
            ),
            "order_violations": int(grade.get("order_violations", 0)),
            "region_boundary_count": len(case_boundaries),
            "exact_three_level_boundary_count": exact_boundary_count,
            "exact_three_level_boundary_rate": ratio(
                exact_boundary_count, len(case_boundaries)
            ),
            "exact_three_level_region_segmentation": (
                exact_boundary_count == len(case_boundaries)
            ),
            "transcript_mfa_boundary_mismatch_count": sum(
                int(bool(row["transcript_mfa_boundary_mismatch"]))
                for row in case_boundaries
            ),
            "runtime_mfa_boundary_mismatch_count": sum(
                int(bool(row["runtime_mfa_boundary_mismatch"]))
                for row in case_boundaries
            ),
            "runtime_transcript_boundary_mismatch_count": sum(
                int(bool(row["runtime_transcript_boundary_mismatch"]))
                for row in case_boundaries
            ),
            "split_word_count": split_word_count,
            "incomplete_word_count": incomplete_word_count,
            "owned_tail_event_count": len(owned_tail_rows),
            "owned_tail_word_count": owned_tail_word_count,
            "owned_tail_compressed_transition_count": owned_tail_compressed_transition_count,
            "owned_tail_excessive_lead_count": owned_tail_excessive_lead_count,
            "owned_tail_material_prior_shift_count": owned_tail_material_prior_shift_count,
        })

    if not cases:
        print("No complete alignment grades found. Run the corpus first.", file=sys.stderr)
        return 2

    def totals(name: str) -> int:
        return sum(int(row[name]) for row in cases)

    region_expected = totals("region_start_expected")
    region_successes = totals("region_start_successes")
    nucleus_expected = totals("region_nucleus_expected")
    nucleus_successes = totals("region_nucleus_successes")
    nucleus_matched = totals("region_nucleus_matched")
    word_expected = totals("word_start_expected")
    word_successes = totals("word_start_successes")
    assignment_expected = totals("word_region_expected")
    assignment_successes = totals("word_region_successes")
    three_level_word_successes = totals("three_level_word_successes")
    pause_duration = totals("pause_duration_ms")
    pause_leakage = totals("pause_leakage_ms")
    planned_events = totals("planned_event_count")
    committed_events = totals("committed_event_count")
    delivery_planned_events = totals("delivery_planned_events")
    delivered_events = totals("delivered_events")
    delivery_expected_words = totals("delivery_expected_words")
    delivery_missing_words = totals("delivery_missing_words")
    measured_word_durations = totals("measured_word_durations")
    successful_word_durations = totals("successful_word_durations")
    delivery_expected_regions = totals("delivery_expected_speech_regions")
    empty_speech_regions = totals("empty_speech_regions")
    delivery_expected_sentences = totals("delivery_expected_sentences")
    delivery_missing_sentences = totals("delivery_missing_sentences")
    exact_three_level_boundaries = sum(
        int(bool(row["exact_three_level_boundary"])) for row in boundaries_out
    )
    pause_boundary_rows = [row for row in boundaries_out if row["gold_pause_boundary"]]
    missed_pause_rows = [row for row in pause_boundary_rows if row["missed_pause_boundary"]]
    extra_pause_rows = [row for row in boundaries_out if row["extra_pause_boundary"]]
    material_leakage_rows = [row for row in pause_boundary_rows if row["material_pause_leakage"]]
    majority_hold_loss_rows = [row for row in pause_boundary_rows if row["majority_hold_lost"]]
    early_resume_rows = [row for row in pause_boundary_rows if float(row["resume_early_ms"]) > 0.0]
    owned_tail_pause_rows = [row for row in pause_boundary_rows if int(row["owned_tail_events_in_pause"]) > 0]
    hold_failure_rows = [row for row in pause_boundary_rows if row["hold_failure"]]
    missed_pause_case_names = {row["case_name"] for row in missed_pause_rows}
    owned_tail_case_names = {
        row["case_name"] for row in cases if row["owned_tail_event_count"]
    }

    def affected_cases(rows: list[dict]) -> int:
        return len({row["case_name"] for row in rows})

    punctuation_breakdown: dict[str, dict] = {}
    for row in pause_boundary_rows:
        key = str(row["punctuation_class"])
        stats = punctuation_breakdown.setdefault(key, {
            "expected_boundaries": 0,
            "missed_boundaries": 0,
            "material_leakage_boundaries": 0,
            "majority_hold_lost_boundaries": 0,
            "early_resume_boundaries": 0,
            "owned_tail_overlap_boundaries": 0,
            "hold_failures": 0,
        })
        stats["expected_boundaries"] += 1
        stats["missed_boundaries"] += int(bool(row["missed_pause_boundary"]))
        stats["material_leakage_boundaries"] += int(bool(row["material_pause_leakage"]))
        stats["majority_hold_lost_boundaries"] += int(bool(row["majority_hold_lost"]))
        stats["early_resume_boundaries"] += int(float(row["resume_early_ms"]) > 0.0)
        stats["owned_tail_overlap_boundaries"] += int(
            int(row["owned_tail_events_in_pause"]) > 0
        )
        stats["hold_failures"] += int(bool(row["hold_failure"]))

    matched_region_count = totals("region_start_matched")
    matched_word_count = totals("word_start_matched")
    region_error_total = sum(
        float(row["region_start_mean_abs_error_ms"]) * int(row["region_start_matched"])
        for row in cases
    )
    word_error_total = sum(
        float(row["word_start_mean_abs_error_ms"]) * int(row["word_start_matched"])
        for row in cases
    )
    animation_onset_expected = totals("animation_onset_expected")
    animation_onset_matched = totals("animation_onset_matched")
    animation_onset_successes = totals("animation_onset_successes")
    animation_onset_error_total = sum(
        float(row["animation_onset_mean_abs_error_ms"])
        * int(row["animation_onset_matched"])
        for row in cases
    )
    decoded_viseme_reference = totals("decoded_viseme_reference")
    decoded_viseme_matched = totals("decoded_viseme_matched")
    decoded_viseme_missing = totals("decoded_viseme_missing")
    decoded_viseme_extra = totals("decoded_viseme_extra")
    decoded_viseme_center_error_total = sum(
        float(row["decoded_viseme_center_mae_ms"])
        * int(row["decoded_viseme_matched"])
        for row in cases
    )
    summary = {
        "cases": len(cases),
        "start_tolerance_ms": START_TOLERANCE_MS,
        "region_start": {
            "expected": region_expected,
            "successful": region_successes,
            "success_rate": ratio(region_successes, region_expected),
            "mean_abs_error_ms": ratio(region_error_total, matched_region_count, 0.0),
        },
        "strict_region_nucleus_alignment": {
            "expected": nucleus_expected,
            "matched": nucleus_matched,
            "successful": nucleus_successes,
            "success_rate": ratio(nucleus_successes, nucleus_expected),
            "tolerance_ms": REGION_NUCLEUS_TOLERANCE_MS,
            "mean_abs_error_ms": ratio(
                sum(
                    float(row["region_nucleus_mean_abs_error_ms"])
                    * int(row["region_nucleus_matched"])
                    for row in cases
                ),
                nucleus_matched,
                0.0,
            ),
        },
        "pause": {
            "duration_ms": pause_duration,
            "leakage_ms": pause_leakage,
            "clean_rate": 1.0 - ratio(pause_leakage, pause_duration, 0.0),
        },
        "word_animation_onset": {
            "expected": animation_onset_expected,
            "matched": animation_onset_matched,
            "successful": animation_onset_successes,
            "success_rate": ratio(
                animation_onset_successes, animation_onset_expected
            ),
            "mean_abs_error_ms": ratio(
                animation_onset_error_total, animation_onset_matched, 0.0
            ),
            "tolerance_ms": START_TOLERANCE_MS,
        },
        "word_interval_alignment": {
            "expected": delivery_expected_words,
            "measured": len(word_interval_onset_abs_errors_ms),
            "zero_overlap_words": totals("zero_overlap_words"),
            "low_coverage_words": totals("low_coverage_words"),
            "spoken_coverage_mean": (
                statistics.fmean(word_interval_coverages)
                if word_interval_coverages else 0.0
            ),
            "spoken_coverage_median": (
                statistics.median(word_interval_coverages)
                if word_interval_coverages else 0.0
            ),
            "spoken_coverage_p10": percentile(word_interval_coverages, 0.10),
            "animation_precision_mean": (
                statistics.fmean(word_interval_precisions)
                if word_interval_precisions else 0.0
            ),
            "interval_iou_mean": (
                statistics.fmean(word_interval_ious)
                if word_interval_ious else 0.0
            ),
            "interval_iou_median": (
                statistics.median(word_interval_ious)
                if word_interval_ious else 0.0
            ),
            "onset_mean_abs_error_ms": (
                statistics.fmean(word_interval_onset_abs_errors_ms)
                if word_interval_onset_abs_errors_ms else 0.0
            ),
            "onset_p95_abs_error_ms": percentile(
                word_interval_onset_abs_errors_ms, 0.95
            ),
            "onset_max_abs_error_ms": max(
                word_interval_onset_abs_errors_ms, default=0.0
            ),
            "onsets_over_200ms": sum(
                error > 200.0 for error in word_interval_onset_abs_errors_ms
            ),
            "maximum_early_onset_ms": max(
                (-error for error in word_interval_onset_signed_errors_ms),
                default=0.0,
            ),
            "maximum_late_onset_ms": max(
                word_interval_onset_signed_errors_ms, default=0.0
            ),
            "exit_mean_abs_error_ms": (
                statistics.fmean(word_interval_exit_abs_errors_ms)
                if word_interval_exit_abs_errors_ms else 0.0
            ),
            "exit_p95_abs_error_ms": percentile(
                word_interval_exit_abs_errors_ms, 0.95
            ),
            "exit_max_abs_error_ms": max(
                word_interval_exit_abs_errors_ms, default=0.0
            ),
        },
        "decoded_viseme_alignment": {
            "reference": decoded_viseme_reference,
            "matched": decoded_viseme_matched,
            "missing": decoded_viseme_missing,
            "extra": decoded_viseme_extra,
            "identity_recall": ratio(
                decoded_viseme_matched, decoded_viseme_reference
            ),
            "identity_precision": ratio(
                decoded_viseme_matched,
                decoded_viseme_matched + decoded_viseme_extra,
            ),
            "mean_abs_center_error_ms": ratio(
                decoded_viseme_center_error_total, decoded_viseme_matched, 0.0
            ),
        },
        "word_duration": {
            "expected": delivery_expected_words,
            "measured": measured_word_durations,
            "coverage_rate": ratio(
                measured_word_durations, delivery_expected_words
            ),
            "successful": successful_word_durations,
            "success_rate": ratio(
                successful_word_durations, delivery_expected_words
            ),
            "tolerance_rule": "max(80 ms, 25% of MFA word duration)",
            "missing_words": delivery_missing_words,
            "compressed_words": totals("compressed_words"),
            "severely_compressed_words": totals("severely_compressed_words"),
            "stretched_words": totals("stretched_words"),
            "mean_abs_error_ms": (
                statistics.fmean(word_duration_abs_errors_ms)
                if word_duration_abs_errors_ms else 0.0
            ),
            "median_abs_error_ms": (
                statistics.median(word_duration_abs_errors_ms)
                if word_duration_abs_errors_ms else 0.0
            ),
            "p90_abs_error_ms": percentile(word_duration_abs_errors_ms, 0.90),
            "p95_abs_error_ms": percentile(word_duration_abs_errors_ms, 0.95),
            "max_abs_error_ms": max(word_duration_abs_errors_ms, default=0.0),
            "mean_signed_error_ms": (
                statistics.fmean(word_duration_signed_errors_ms)
                if word_duration_signed_errors_ms else 0.0
            ),
            "median_signed_error_ms": (
                statistics.median(word_duration_signed_errors_ms)
                if word_duration_signed_errors_ms else 0.0
            ),
            "mean_ratio": (
                statistics.fmean(word_duration_ratios)
                if word_duration_ratios else 0.0
            ),
            "median_ratio": (
                statistics.median(word_duration_ratios)
                if word_duration_ratios else 0.0
            ),
            "p10_ratio": percentile(word_duration_ratios, 0.10),
            "p90_ratio": percentile(word_duration_ratios, 0.90),
        },
        "word_region_assignment": {
            "expected": assignment_expected,
            "successful": assignment_successes,
            "success_rate": ratio(assignment_successes, assignment_expected),
        },
        "strict_three_level_word_assignment": {
            "expected": assignment_expected,
            "successful": three_level_word_successes,
            "success_rate": ratio(three_level_word_successes, assignment_expected),
            "failed_cases": sum(
                1 for row in cases
                if row["three_level_word_successes"] < row["word_region_expected"]
            ),
        },
        "word_region_confusion": {
            "early_region_thefts": totals("early_region_theft_count"),
            "early_region_theft_cases": sum(
                1 for row in cases if row["early_region_theft_count"]
            ),
            "late_region_assignments": totals("late_region_assignment_count"),
            "late_region_assignment_cases": sum(
                1 for row in cases if row["late_region_assignment_count"]
            ),
            "materially_early_intact_words": totals(
                "materially_early_intact_word_count"
            ),
            "materially_early_intact_word_cases": sum(
                1 for row in cases if row["materially_early_intact_word_count"]
            ),
            "max_intact_word_lead_ms": max(
                float(row["max_intact_word_lead_ms"]) for row in cases
            ),
            "region_word_overruns": totals("region_word_overrun_count"),
            "region_word_overrun_cases": sum(
                1 for row in cases if row["region_word_overrun_count"]
            ),
            "max_region_word_overrun": max(
                int(row["max_region_word_overrun"]) for row in cases
            ),
        },
        "guardrails": {
            "planned_events": planned_events,
            "committed_events": committed_events,
            "event_completion_rate": ratio(committed_events, planned_events),
            "order_violations": totals("order_violations"),
            "order_violation_cases": sum(1 for row in cases if row["order_violations"]),
        },
        "runtime_delivery": {
            "planned_candidate_events": delivery_planned_events,
            "delivered_events": delivered_events,
            "event_delivery_rate": ratio(
                delivered_events, delivery_planned_events
            ),
            "late_after_window_events": totals("late_after_window_events"),
            "late_after_window_cases": sum(
                1 for row in cases if row["late_after_window_events"]
            ),
            "expected_words": delivery_expected_words,
            "missing_words": delivery_missing_words,
            "missing_word_cases": sum(
                1 for row in cases if row["delivery_missing_words"]
            ),
            "word_delivery_rate": ratio(
                delivery_expected_words - delivery_missing_words,
                delivery_expected_words,
            ),
            "compressed_words": totals("compressed_words"),
            "compressed_word_cases": sum(
                1 for row in cases if row["compressed_words"]
            ),
            "word_duration_ratio_mean": (
                statistics.fmean(word_duration_ratios)
                if word_duration_ratios else 0.0
            ),
            "word_duration_ratio_median": (
                statistics.median(word_duration_ratios)
                if word_duration_ratios else 0.0
            ),
            "expected_speech_regions": delivery_expected_regions,
            "empty_speech_regions": empty_speech_regions,
            "empty_speech_region_cases": sum(
                1 for row in cases if row["empty_speech_regions"]
            ),
            "speech_region_delivery_rate": ratio(
                delivery_expected_regions - empty_speech_regions,
                delivery_expected_regions,
            ),
            "expected_sentences": delivery_expected_sentences,
            "missing_sentences": delivery_missing_sentences,
            "missing_sentence_cases": sum(
                1 for row in cases if row["delivery_missing_sentences"]
            ),
            "sentence_delivery_rate": ratio(
                delivery_expected_sentences - delivery_missing_sentences,
                delivery_expected_sentences,
            ),
            "compressed_sentences": totals("compressed_sentences"),
            "compressed_sentence_cases": sum(
                1 for row in cases if row["compressed_sentences"]
            ),
            "sentence_duration_ratio_mean": (
                statistics.fmean(sentence_duration_ratios)
                if sentence_duration_ratios else 0.0
            ),
            "sentence_duration_ratio_median": (
                statistics.median(sentence_duration_ratios)
                if sentence_duration_ratios else 0.0
            ),
        },
        "viseme_proportion": {
            "planned_multi_viseme_words": totals(
                "viseme_proportion_planned_words"
            ),
            "gradeable_words": totals("viseme_proportion_gradeable_words"),
            "ungradeable_words": totals("viseme_proportion_ungradeable_words"),
            "word_coverage_rate": ratio(
                totals("viseme_proportion_gradeable_words"),
                totals("viseme_proportion_planned_words"),
            ),
            "zero_runtime_words": totals("viseme_proportion_zero_runtime_words"),
            "expected_viseme_runs": len(viseme_run_share_errors),
            "runs_within_10pp": sum(
                error <= 0.10 for error in viseme_run_share_errors
            ),
            "runs_within_10pp_rate": ratio(
                sum(error <= 0.10 for error in viseme_run_share_errors),
                len(viseme_run_share_errors),
            ),
            "run_share_abs_error_mean": (
                statistics.fmean(viseme_run_share_errors)
                if viseme_run_share_errors else 0.0
            ),
            "run_share_abs_error_median": (
                statistics.median(viseme_run_share_errors)
                if viseme_run_share_errors else 0.0
            ),
            "run_share_abs_error_p90": percentile(
                viseme_run_share_errors, 0.90
            ),
            "boundary_position_abs_error_mean": (
                statistics.fmean(viseme_boundary_errors)
                if viseme_boundary_errors else 0.0
            ),
            "boundary_position_abs_error_median": (
                statistics.median(viseme_boundary_errors)
                if viseme_boundary_errors else 0.0
            ),
            "boundary_position_abs_error_p90": percentile(
                viseme_boundary_errors, 0.90
            ),
            "word_total_variation_mean": (
                statistics.fmean(viseme_word_total_variations)
                if viseme_word_total_variations else 0.0
            ),
            "word_total_variation_median": (
                statistics.median(viseme_word_total_variations)
                if viseme_word_total_variations else 0.0
            ),
            "word_total_variation_p90": percentile(
                viseme_word_total_variations, 0.90
            ),
            "severe_words": totals("viseme_proportion_severe_words"),
        },
        "presentation_event_visibility": {
            "eligible_events": presentation_eligible_events,
            "robust_visible_events": presentation_visible_events,
            "robust_visible_event_rate": ratio(
                presentation_visible_events, presentation_eligible_events
            ),
            "boundary_events": presentation_boundary_events,
            "robust_boundary_events": presentation_visible_boundary_events,
            "robust_boundary_event_rate": ratio(
                presentation_visible_boundary_events, presentation_boundary_events
            ),
        },
        "phonetic_presentation": {
            "vowel_frames": totals("phonetic_vowel_frames"),
            "vowel_target_dominance_rate": ratio(
                totals("phonetic_vowel_target_dominant_frames"),
                totals("phonetic_vowel_frames"),
            ),
            "vowel_same_word_dominance_rate": ratio(
                totals("phonetic_vowel_same_word_dominant_frames"),
                totals("phonetic_vowel_frames"),
            ),
            "vowel_foreign_word_dominance_rate": ratio(
                totals("phonetic_vowel_foreign_word_dominant_frames"),
                totals("phonetic_vowel_frames"),
                0.0,
            ),
            "low_dominance_vowels": totals("phonetic_low_dominance_vowels"),
            "foreign_intrusion_vowels": totals(
                "phonetic_foreign_intrusion_vowels"
            ),
            "bilabial_frames": totals("phonetic_bilabial_frames"),
            "bilabial_target_dominance_rate": ratio(
                totals("phonetic_bilabial_target_dominant_frames"),
                totals("phonetic_bilabial_frames"),
            ),
            "bilabial_peak_mean_abs_error_ms": (
                statistics.fmean(bilabial_peak_abs_errors_ms)
                if bilabial_peak_abs_errors_ms else 0.0
            ),
            "bilabial_peak_p95_abs_error_ms": percentile(
                bilabial_peak_abs_errors_ms, 0.95
            ),
            "late_bilabial_peaks": totals("phonetic_late_bilabial_peaks"),
            "oh_saturation_rate": ratio(
                totals("phonetic_oh_saturated_frames"),
                totals("phonetic_oh_frames"),
                0.0,
            ),
            "other_vowel_saturation_rate": ratio(
                totals("phonetic_other_vowel_saturated_frames"),
                totals("phonetic_other_vowel_frames"),
                0.0,
            ),
        },
        "strict_region_segmentation": {
            "boundaries": len(boundaries_out),
            "exact_boundaries": exact_three_level_boundaries,
            "exact_boundary_rate": ratio(
                exact_three_level_boundaries, len(boundaries_out)
            ),
            "perfect_cases": sum(
                1 for row in cases if row["exact_three_level_region_segmentation"]
            ),
            "perfect_case_rate": ratio(
                sum(1 for row in cases if row["exact_three_level_region_segmentation"]),
                len(cases),
            ),
            "transcript_mfa_boundary_mismatches": sum(
                int(bool(row["transcript_mfa_boundary_mismatch"]))
                for row in boundaries_out
            ),
            "runtime_mfa_boundary_mismatches": sum(
                int(bool(row["runtime_mfa_boundary_mismatch"]))
                for row in boundaries_out
            ),
            "runtime_transcript_boundary_mismatches": sum(
                int(bool(row["runtime_transcript_boundary_mismatch"]))
                for row in boundaries_out
            ),
        },
        "failure_classes": {
            "pause_boundaries": {
                "expected": len(pause_boundary_rows),
                "missed": len(missed_pause_rows),
                "miss_rate": ratio(len(missed_pause_rows), len(pause_boundary_rows), 0.0),
                "missed_cases": affected_cases(missed_pause_rows),
                "extra": len(extra_pause_rows),
                "extra_cases": affected_cases(extra_pause_rows),
            },
            "pause_presentation": {
                "material_leakage_threshold_ms": MATERIAL_PAUSE_LEAKAGE_MS,
                "material_leakage_boundaries": len(material_leakage_rows),
                "material_leakage_cases": affected_cases(material_leakage_rows),
                "majority_hold_lost_boundaries": len(majority_hold_loss_rows),
                "majority_hold_lost_cases": affected_cases(majority_hold_loss_rows),
                "early_resume_boundaries": len(early_resume_rows),
                "early_resume_cases": affected_cases(early_resume_rows),
                "hold_failure_boundaries": len(hold_failure_rows),
                "hold_failure_cases": affected_cases(hold_failure_rows),
            },
            "word_ownership": {
                "split_words": totals("split_word_count"),
                "split_word_cases": sum(1 for row in cases if row["split_word_count"]),
                "incomplete_words": totals("incomplete_word_count"),
                "incomplete_word_cases": sum(1 for row in cases if row["incomplete_word_count"]),
            },
            "owned_tail_recovery": {
                "events": totals("owned_tail_event_count"),
                "words": totals("owned_tail_word_count"),
                "cases": sum(1 for row in cases if row["owned_tail_event_count"]),
                "compressed_spacing_threshold_ms": MIN_DISTINCT_VISEME_SPACING_MS,
                "compressed_transitions": totals("owned_tail_compressed_transition_count"),
                "compressed_transition_cases": sum(
                    1 for row in cases if row["owned_tail_compressed_transition_count"]
                ),
                "normal_commit_lead_limit_ms": MAX_NORMAL_COMMIT_LEAD_MS,
                "excessive_lead_events": totals("owned_tail_excessive_lead_count"),
                "excessive_lead_cases": sum(
                    1 for row in cases if row["owned_tail_excessive_lead_count"]
                ),
                "material_prior_shift_threshold_ms": MATERIAL_PRIOR_SHIFT_MS,
                "material_prior_shift_events": totals("owned_tail_material_prior_shift_count"),
                "material_prior_shift_cases": sum(
                    1 for row in cases if row["owned_tail_material_prior_shift_count"]
                ),
                "pause_overlap_boundaries": len(owned_tail_pause_rows),
                "pause_overlap_cases": affected_cases(owned_tail_pause_rows),
            },
            "interactions": {
                "missed_pause_and_owned_tail_cases": len(
                    missed_pause_case_names & owned_tail_case_names
                ),
            },
            "by_punctuation": punctuation_breakdown,
        },
    }

    with (latest / "alignment_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    if words_out:
        with (latest / "alignment_words.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(words_out[0]))
            writer.writeheader()
            writer.writerows(words_out)
    if boundaries_out:
        with (latest / "alignment_boundaries.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(boundaries_out[0]))
            writer.writeheader()
            writer.writerows(boundaries_out)
    summary_path = latest / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Region starts: {summary['region_start']['success_rate']:.3f} "
        f"({summary['region_start']['mean_abs_error_ms']:.1f} ms MAE)"
    )
    print(f"Pause clean: {summary['pause']['clean_rate']:.3f}")
    print(
        f"P1 word animation onset: "
        f"{summary['word_animation_onset']['success_rate']:.3f} "
        f"({summary['word_animation_onset']['mean_abs_error_ms']:.1f} ms MAE)"
    )
    decoded = summary["decoded_viseme_alignment"]
    print(
        "Decoded viseme alignment: "
        f"recall={decoded['identity_recall']:.3f} "
        f"precision={decoded['identity_precision']:.3f} "
        f"center_MAE={decoded['mean_abs_center_error_ms']:.1f}ms "
        f"missing={decoded['missing']} extra={decoded['extra']}"
    )
    word_duration = summary["word_duration"]
    print(
        "Word animation duration: "
        f"success={word_duration['success_rate']:.3f} "
        f"mean={word_duration['mean_abs_error_ms']:.1f}ms "
        f"median={word_duration['median_abs_error_ms']:.1f}ms "
        f"compressed={word_duration['compressed_words']} "
        f"stretched={word_duration['stretched_words']}"
    )
    print(f"Word-region assignment: {summary['word_region_assignment']['success_rate']:.3f}")
    confusion = summary["word_region_confusion"]
    print(
        "Region confusion: "
        f"early_thefts={confusion['early_region_thefts']} "
        f"late_assignments={confusion['late_region_assignments']} "
        f"intact_words_early={confusion['materially_early_intact_words']} "
        f"region_overruns={confusion['region_word_overruns']}"
    )
    print(
        f"Guardrails: completion={summary['guardrails']['event_completion_rate']:.3f} "
        f"order_violations={summary['guardrails']['order_violations']}"
    )
    proportions = summary["viseme_proportion"]
    print(
        "Within-word viseme proportions: "
        f"run_median={proportions['run_share_abs_error_median'] * 100.0:.2f}pp "
        f"boundary_median={proportions['boundary_position_abs_error_median'] * 100.0:.2f}pp "
        f"word_TV_median={proportions['word_total_variation_median'] * 100.0:.2f}% "
        f"coverage={proportions['word_coverage_rate']:.3f}"
    )
    visibility = summary["presentation_event_visibility"]
    print(
        "Presentation event visibility: "
        f"all={visibility['robust_visible_events']}/"
        f"{visibility['eligible_events']} "
        f"({visibility['robust_visible_event_rate']:.3f}) "
        f"region-boundary={visibility['robust_boundary_events']}/"
        f"{visibility['boundary_events']} "
        f"({visibility['robust_boundary_event_rate']:.3f})"
    )
    strict_regions = summary["strict_region_segmentation"]
    print(
        "Strict regions: "
        f"boundaries={strict_regions['exact_boundaries']}/"
        f"{strict_regions['boundaries']} "
        f"perfect_cases={strict_regions['perfect_cases']}/{summary['cases']}"
    )
    failures = summary["failure_classes"]
    print(
        "Failure classes: "
        f"missed_pauses={failures['pause_boundaries']['missed']}/"
        f"{failures['pause_boundaries']['expected']} "
        f"material_leaks={failures['pause_presentation']['material_leakage_boundaries']} "
        f"split_words={failures['word_ownership']['split_words']} "
        f"incomplete_words={failures['word_ownership']['incomplete_words']} "
        f"owned_tail_events={failures['owned_tail_recovery']['events']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
