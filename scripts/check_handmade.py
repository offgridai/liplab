import argparse
import csv
import json
import pathlib
import sys

from handmade_tools import case_stems, handmade_draft_dir, repo_root


def validate_csv(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors

    previous_start = -1.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows):
        try:
            start = float(row["start"])
            end = float(row["end"])
        except (KeyError, ValueError) as exc:
            errors.append(f"{path.name}: row {index + 1} has invalid time field ({exc})")
            continue

        pose = row.get("pose", "").strip()
        if not pose:
            errors.append(f"{path.name}: row {index + 1} has empty pose")
        if end <= start:
            errors.append(f"{path.name}: row {index + 1} has non-positive duration")
        if previous_start > start + 1e-6:
            errors.append(f"{path.name}: row {index + 1} is not start-time monotonic")
        previous_start = start

    return errors


def validate_draft(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors

    payload = json.loads(path.read_text(encoding="utf-8"))
    visemes = payload.get("visemes", [])
    previous_start = -1.0
    for index, viseme in enumerate(visemes):
        start = float(viseme.get("start", 0.0))
        end = float(viseme.get("end", 0.0))
        pose = str(viseme.get("pose", "")).strip()
        if not pose:
            errors.append(f"{path.parent.name}: viseme {index} has empty pose")
        if end <= start:
            errors.append(f"{path.parent.name}: viseme {index} has non-positive duration")
        if previous_start > start + 1e-6:
            errors.append(f"{path.parent.name}: viseme {index} is not start-time monotonic")
        previous_start = start
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate handmade CSVs and optional draft annotation packages.")
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cases = case_stems()
    errors: list[str] = []

    for case_id in cases:
        errors.extend(validate_csv(root / "inputs" / "handmade" / f"{case_id}.csv"))
        if args.include_drafts:
            errors.extend(validate_draft(handmade_draft_dir(case_id) / "draft.annotation.json"))

    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1

    checked = len(cases)
    if args.include_drafts:
        print(f"Handmade CSVs and draft annotations validated for {checked} case(s).")
    else:
        print(f"Handmade CSVs validated for {checked} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
