import csv
import json
import pathlib
import re
import sys


START_TOLERANCE_MS = 100.0
MATERIAL_PAUSE_LEAKAGE_MS = 50
MAJORITY_HOLD_LOSS_RATE = 0.5
MIN_DISTINCT_VISEME_SPACING_MS = 20.0
MAX_NORMAL_COMMIT_LEAD_MS = 160.0
MATERIAL_PRIOR_SHIFT_MS = 150.0


def ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator else default


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

    for focus_path in sorted(latest.glob("case_*/focus_alignment_grade.json")):
        ownership_path = focus_path.parent / "region_ownership_grade.json"
        grade_path = focus_path.parent / "grade.json"
        if not ownership_path.exists() or not grade_path.exists():
            continue

        focus = load_json(focus_path)
        ownership = load_json(ownership_path)
        grade = load_json(grade_path)
        current_case_name = focus_path.parent.name
        gold_dir = root / "inputs" / "gold" / current_case_name
        gold_boundaries = load_csv(gold_dir / "boundaries.csv")
        gold_speech = {
            int(row["index"]): row for row in load_csv(gold_dir / "speech.csv")
        }
        committed_rows = load_csv(focus_path.parent / "committed.csv")
        ownership_words = {
            int(word["word_index"]): word for word in ownership.get("words", [])
        }
        word_heads = {
            int(word["word_index"]): word for word in focus.get("word_heads", [])
        }
        region_heads = {
            int(row["region_index"]): row for row in focus.get("region_heads", [])
        }
        pause_boundaries = {
            (int(row["region_before"]), int(row["region_after"])): row
            for row in focus.get("pause_boundaries", [])
        }

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

        word_expected = int(focus.get("expected_word_head_count", 0))
        word_successes = 0
        word_errors: list[float] = []
        assignment_successes = 0
        for word_index, owner in ownership_words.items():
            head = word_heads.get(word_index)
            strict_owner = bool(owner.get("runtime_mfa_strict_correct", False))
            head_owner = bool(owner.get("runtime_mfa_first_event_correct", strict_owner))
            planned = int(owner.get("planned_event_count", 0))
            committed = int(owner.get("committed_event_count", 0))
            fully_committed = planned > 0 and committed == planned
            if fully_committed and strict_owner:
                assignment_successes += 1

            error_ms = None
            start_success = False
            if head is not None:
                error_ms = abs(float(head.get("error_ms", 0.0)))
                word_errors.append(error_ms)
                start_success = head_owner and error_ms <= START_TOLERANCE_MS
                if start_success:
                    word_successes += 1
            words_out.append({
                "case": case_id(focus_path),
                "case_name": focus_path.parent.name,
                "word_index": word_index,
                "word": owner.get("word", ""),
                "mfa_region_index": owner.get("mfa_region_index", -1),
                "runtime_region_index": owner.get("runtime_region_index", -1),
                "fully_committed": fully_committed,
                "runtime_region_integrity": bool(owner.get("runtime_region_integrity", False)),
                "region_assignment_success": fully_committed and strict_owner,
                "start_error_ms": "" if error_ms is None else error_ms,
                "start_success": start_success,
            })

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
            boundary_observable = (
                before_complete and after_complete
                and len(before_regions) == 1 and len(after_regions) == 1
            )
            runtime_split = (
                boundary_observable and before_regions[0] != after_regions[0]
            )
            region_before = int(boundary.get("speech_region_index_before", -1))
            region_after = int(boundary.get("speech_region_index_after", -1))
            gold_split = region_before >= 0 and region_after >= 0 and region_before != region_after
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
                "gold_pause_duration_ms": pause_duration_ms,
                "runtime_boundary_observable": boundary_observable,
                "runtime_region_before": before_regions[0] if len(before_regions) == 1 else -1,
                "runtime_region_after": after_regions[0] if len(after_regions) == 1 else -1,
                "runtime_pause_boundary": runtime_split,
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
        cases.append({
            "case": case_id(focus_path),
            "case_name": focus_path.parent.name,
            "region_start_expected": region_expected,
            "region_start_matched": len(region_errors),
            "region_start_successes": region_successes,
            "region_start_success_rate": ratio(region_successes, region_expected),
            "region_start_mean_abs_error_ms": ratio(sum(region_errors), len(region_errors), 0.0),
            "pause_duration_ms": pause_duration,
            "pause_leakage_ms": pause_leakage,
            "pause_clean_rate": 1.0 - ratio(pause_leakage, pause_duration, 0.0),
            "word_start_expected": word_expected,
            "word_start_matched": len(word_errors),
            "word_start_successes": word_successes,
            "word_start_success_rate": ratio(word_successes, word_expected),
            "word_start_mean_abs_error_ms": ratio(sum(word_errors), len(word_errors), 0.0),
            "word_region_expected": assignment_expected,
            "word_region_successes": assignment_successes,
            "word_region_assignment_rate": ratio(assignment_successes, assignment_expected),
            "planned_event_count": planned_events,
            "committed_event_count": committed_events,
            "event_completion_rate": ratio(committed_events, planned_events),
            "order_violations": int(grade.get("order_violations", 0)),
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
    word_expected = totals("word_start_expected")
    word_successes = totals("word_start_successes")
    assignment_expected = totals("word_region_expected")
    assignment_successes = totals("word_region_successes")
    pause_duration = totals("pause_duration_ms")
    pause_leakage = totals("pause_leakage_ms")
    planned_events = totals("planned_event_count")
    committed_events = totals("committed_event_count")
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

    summary = {
        "cases": len(cases),
        "start_tolerance_ms": START_TOLERANCE_MS,
        "region_start": {
            "expected": region_expected,
            "successful": region_successes,
            "success_rate": ratio(region_successes, region_expected),
            "mean_abs_error_ms": ratio(region_error_total, matched_region_count, 0.0),
        },
        "pause": {
            "duration_ms": pause_duration,
            "leakage_ms": pause_leakage,
            "clean_rate": 1.0 - ratio(pause_leakage, pause_duration, 0.0),
        },
        "word_start": {
            "expected": word_expected,
            "successful": word_successes,
            "success_rate": ratio(word_successes, word_expected),
            "mean_abs_error_ms": ratio(word_error_total, matched_word_count, 0.0),
        },
        "word_region_assignment": {
            "expected": assignment_expected,
            "successful": assignment_successes,
            "success_rate": ratio(assignment_successes, assignment_expected),
        },
        "guardrails": {
            "planned_events": planned_events,
            "committed_events": committed_events,
            "event_completion_rate": ratio(committed_events, planned_events),
            "order_violations": totals("order_violations"),
            "order_violation_cases": sum(1 for row in cases if row["order_violations"]),
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
        f"P0 region starts: {summary['region_start']['success_rate']:.3f} "
        f"({summary['region_start']['mean_abs_error_ms']:.1f} ms MAE)"
    )
    print(f"P1 pause clean: {summary['pause']['clean_rate']:.3f}")
    print(
        f"P2 word starts: {summary['word_start']['success_rate']:.3f} "
        f"({summary['word_start']['mean_abs_error_ms']:.1f} ms MAE)"
    )
    print(f"Word-region assignment: {summary['word_region_assignment']['success_rate']:.3f}")
    print(
        f"Guardrails: completion={summary['guardrails']['event_completion_rate']:.3f} "
        f"order_violations={summary['guardrails']['order_violations']}"
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
