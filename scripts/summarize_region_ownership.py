import csv
import json
import pathlib
import re
import sys


def ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator else default


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    cases: list[dict] = []
    word_rows: list[dict] = []

    for path in sorted(latest.glob("case_*/region_ownership_grade.json")):
        grade = json.loads(path.read_text(encoding="utf-8"))
        match = re.match(r"case_(\d+)_", path.parent.name)
        case_id = match.group(1) if match else path.parent.name
        case = {"case": case_id, "case_name": path.parent.name}
        case.update({key: value for key, value in grade.items() if key != "words"})
        cases.append(case)
        for word in grade.get("words", []):
            word_rows.append({"case": case_id, "case_name": path.parent.name, **word})

    if not cases:
        print("No region ownership grades found. Run the corpus first.", file=sys.stderr)
        return 2

    total_mfa_words = sum(int(row["mfa_word_count"]) for row in cases)
    total_planned_words = sum(int(row["transcript_planned_word_count"]) for row in cases)
    total_committed_words = sum(int(row["runtime_committed_word_count"]) for row in cases)
    intact_words = sum(
        1 for row in word_rows
        if row.get("committed_event_count", 0) > 0 and row.get("runtime_region_integrity", False)
    )
    strict_correct_words = sum(
        1 for row in word_rows
        if row.get("committed_event_count", 0) > 0 and row.get("runtime_mfa_strict_correct", False)
    )
    majority_correct_words = sum(
        1 for row in word_rows
        if row.get("committed_event_count", 0) > 0 and row.get("runtime_mfa_majority_correct", False)
    )
    transcript_index_correct_words = sum(
        1 for row in word_rows
        if int(row.get("transcript_region_index", -1))
        == int(row.get("mfa_region_index", -2))
    )
    total_planned_events = sum(int(row.get("planned_event_count", 0)) for row in word_rows)
    total_committed_events = sum(int(row.get("committed_event_count", 0)) for row in word_rows)
    fully_committed_words = sum(
        1 for row in word_rows
        if int(row.get("planned_event_count", 0)) > 0
        and int(row.get("committed_event_count", 0)) == int(row.get("planned_event_count", 0))
    )
    fully_committed_and_strict_correct_words = sum(
        1 for row in word_rows
        if int(row.get("planned_event_count", 0)) > 0
        and int(row.get("committed_event_count", 0)) == int(row.get("planned_event_count", 0))
        and row.get("runtime_mfa_strict_correct", False)
    )

    summary = {
        "cases": len(cases),
        "mfa_word_count": total_mfa_words,
        "transcript_planned_word_count": total_planned_words,
        "runtime_committed_word_count": total_committed_words,
        "runtime_word_coverage_rate": ratio(total_committed_words, total_mfa_words),
        "runtime_event_completion_rate": ratio(total_committed_events, total_planned_events),
        "runtime_fully_committed_word_rate": ratio(fully_committed_words, total_mfa_words),
        "runtime_fully_committed_and_strict_correct_word_rate": ratio(
            fully_committed_and_strict_correct_words, total_mfa_words
        ),
        "runtime_word_region_integrity_rate": ratio(intact_words, total_committed_words),
        "runtime_mfa_word_region_strict_accuracy": ratio(strict_correct_words, total_committed_words),
        "runtime_mfa_word_region_majority_accuracy": ratio(majority_correct_words, total_committed_words),
        # Diagnostic only: a single missed/extra text boundary shifts every
        # later ordinal, so boundary agreement is the primary transcript/MFA
        # comparison.
        "transcript_mfa_raw_region_index_accuracy": ratio(
            transcript_index_correct_words, total_planned_words
        ),
    }
    for prefix in ("transcript_mfa", "runtime_mfa", "runtime_transcript"):
        boundary_count = sum(int(row[f"{prefix}_boundary_count"]) for row in cases)
        reference_splits = sum(int(row[f"{prefix}_reference_split_count"]) for row in cases)
        predicted_splits = sum(int(row[f"{prefix}_predicted_split_count"]) for row in cases)
        true_splits = sum(int(row[f"{prefix}_true_split_count"]) for row in cases)
        agreements = sum(
            boundary_count_for_case
            - int(row[f"{prefix}_reference_split_count"])
            - int(row[f"{prefix}_predicted_split_count"])
            + 2 * int(row[f"{prefix}_true_split_count"])
            for row in cases
            for boundary_count_for_case in [int(row[f"{prefix}_boundary_count"])]
        )
        precision = ratio(true_splits, predicted_splits)
        recall = ratio(true_splits, reference_splits)
        summary[f"{prefix}_boundary_agreement_rate"] = ratio(agreements, boundary_count)
        summary[f"{prefix}_split_precision"] = precision
        summary[f"{prefix}_split_recall"] = recall
        summary[f"{prefix}_split_f1"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    failing_words = [
        row for row in word_rows
        if row.get("committed_event_count", 0) == 0
        or not row.get("runtime_region_integrity", False)
        or not row.get("runtime_mfa_strict_correct", False)
    ]
    summary["affected_case_count"] = len({row["case"] for row in failing_words})
    summary["split_word_count"] = sum(
        1 for row in word_rows
        if row.get("committed_event_count", 0) > 0 and not row.get("runtime_region_integrity", False)
    )
    summary["incorrect_or_missing_word_count"] = len(failing_words)

    with (latest / "region_ownership_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    if word_rows:
        with (latest / "region_ownership_words.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(word_rows[0]))
            writer.writeheader()
            writer.writerows(word_rows)
    (latest / "region_ownership_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        f"Region ownership cases={summary['cases']} words={total_mfa_words} "
        f"coverage={summary['runtime_word_coverage_rate']:.3f} "
        f"integrity={summary['runtime_word_region_integrity_rate']:.3f} "
        f"runtime_mfa_strict={summary['runtime_mfa_word_region_strict_accuracy']:.3f} "
        f"runtime_mfa_majority={summary['runtime_mfa_word_region_majority_accuracy']:.3f}"
    )
    print(
        f"Boundary agreement transcript/MFA={summary['transcript_mfa_boundary_agreement_rate']:.3f} "
        f"runtime/MFA={summary['runtime_mfa_boundary_agreement_rate']:.3f} "
        f"runtime/transcript={summary['runtime_transcript_boundary_agreement_rate']:.3f} "
        f"split_words={summary['split_word_count']} affected_cases={summary['affected_case_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
