import argparse
import csv
import pathlib
import sys

from gold_tools import case_stems, gold_draft_dir, gold_manifest_path, read_json, repo_root


def validate_monotonic_rows(path: pathlib.Path, start_key: str, end_key: str, label: str) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors

    previous_start = -1.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows):
        try:
            start = float(row[start_key])
            end = float(row[end_key])
        except (KeyError, ValueError) as exc:
            errors.append(f"{path.name}: row {index + 1} has invalid time field ({exc})")
            continue

        if end <= start:
            errors.append(f"{path.name}: row {index + 1} has non-positive duration")
        if previous_start > start + 1e-6:
            errors.append(f"{path.name}: row {index + 1} is not start-time monotonic")
        previous_start = start

        if label == "viseme" and not row.get("pose", "").strip():
            errors.append(f"{path.name}: row {index + 1} has empty pose")
        if label == "word" and not row.get("word", "").strip():
            errors.append(f"{path.name}: row {index + 1} has empty word")

    return errors


def validate_draft(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors

    payload = read_json(path)
    if payload.get("schema_version") != 1:
        errors.append(f"{path.parent.name}: schema_version must be 1")

    visemes = payload.get("visemes", [])
    previous_viseme_start = -1.0
    for index, viseme in enumerate(visemes):
        start = float(viseme.get("start", 0.0))
        end = float(viseme.get("end", 0.0))
        pose = str(viseme.get("pose", "")).strip()
        if not pose:
            errors.append(f"{path.parent.name}: viseme {index} has empty pose")
        if end <= start:
            errors.append(f"{path.parent.name}: viseme {index} has non-positive duration")
        if previous_viseme_start > start + 1e-6:
            errors.append(f"{path.parent.name}: viseme {index} is not start-time monotonic")
        previous_viseme_start = start

    words = payload.get("gold_words", [])
    previous_word_start = -1.0
    for index, word in enumerate(words):
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        if end <= start:
            errors.append(f"{path.parent.name}: word {index} has non-positive duration")
        if previous_word_start > start + 1e-6:
            errors.append(f"{path.parent.name}: word {index} is not start-time monotonic")
        previous_word_start = start

    speech_regions = payload.get("speech_regions", [])
    previous_region_start = -1.0
    for index, region in enumerate(speech_regions):
        start = float(region.get("start", 0.0))
        end = float(region.get("end", 0.0))
        if end <= start:
            errors.append(f"{path.parent.name}: speech region {index} has non-positive duration")
        if previous_region_start > start + 1e-6:
            errors.append(f"{path.parent.name}: speech region {index} is not start-time monotonic")
        previous_region_start = start

    return errors


def validate_manifest(root: pathlib.Path) -> list[str]:
    errors: list[str] = []
    manifest_path = gold_manifest_path()
    if not manifest_path.exists():
        return errors

    payload = read_json(manifest_path)
    if payload.get("schema_version") != 1:
        errors.append("manifest.json: schema_version must be 1")

    for case in payload.get("cases", []):
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            errors.append("manifest.json: case entry missing case_id")
            continue
        case_dir = root / "inputs" / "gold" / case_id
        for name in ("visemes.csv", "phones.csv", "words.csv", "speech.csv", "boundaries.csv", "metadata.json"):
            if not (case_dir / name).exists():
                errors.append(f"manifest.json: {case_id} missing {name}")
    return errors


def validate_spn_quarantine(gold_root: pathlib.Path) -> tuple[list[str], int]:
    """Reject opaque MFA speech unless it is an explicitly reviewed exception."""

    errors: list[str] = []
    allowlist_path = gold_root / "spn_allowlist.json"
    if not allowlist_path.exists():
        return ["spn_allowlist.json is required"], 0
    payload = read_json(allowlist_path)
    if payload.get("schema_version") != 1:
        errors.append("spn_allowlist.json: schema_version must be 1")
    expected: set[tuple[str, str]] = set()
    for entry in payload.get("entries", []):
        case_id = str(entry.get("case_id", "")).strip()
        word = str(entry.get("word", "")).strip().lower()
        reason = str(entry.get("reason", "")).strip()
        if not case_id or not word or not reason:
            errors.append("spn_allowlist.json: every entry requires case_id, word, and reason")
            continue
        key = (case_id, word)
        if key in expected:
            errors.append(f"spn_allowlist.json: duplicate entry {case_id}/{word}")
        expected.add(key)

    observed: set[tuple[str, str]] = set()
    interval_count = 0
    for case_dir in sorted(path for path in gold_root.iterdir() if path.is_dir()):
        phones_path = case_dir / "phones.csv"
        if not phones_path.exists():
            continue
        with phones_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("phone", "")).strip().lower() != "spn":
                    continue
                key = (case_dir.name, str(row.get("word", "")).strip().lower())
                observed.add(key)
                interval_count += 1
                if key not in expected:
                    errors.append(
                        f"{case_dir.name}/phones.csv: unreviewed spn for {key[1] or '<empty word>'}"
                    )
    for case_id, word in sorted(expected - observed):
        errors.append(f"spn_allowlist.json: stale entry {case_id}/{word}")
    return errors, interval_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate approved gold packages and optional draft annotation packages.")
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    errors: list[str] = []
    approved_count = 0
    draft_count = 0

    gold_root = root / "inputs" / "gold"
    for case_dir in sorted(path for path in gold_root.iterdir() if path.is_dir()) if gold_root.exists() else []:
        errors.extend(validate_monotonic_rows(case_dir / "visemes.csv", "start", "end", "viseme"))
        errors.extend(validate_monotonic_rows(case_dir / "words.csv", "start", "end", "word"))
        errors.extend(validate_monotonic_rows(case_dir / "speech.csv", "start", "end", "speech"))
        approved_count += 1

    spn_errors, spn_count = validate_spn_quarantine(gold_root)
    errors.extend(spn_errors)

    if args.include_drafts:
        for case_id in case_stems():
            draft_path = gold_draft_dir(case_id) / "draft.annotation.json"
            if draft_path.exists():
                errors.extend(validate_draft(draft_path))
                draft_count += 1

    errors.extend(validate_manifest(root))

    if errors:
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
        return 1

    if args.include_drafts:
        print(
            f"Approved gold packages validated for {approved_count} case(s); "
            f"draft packages validated for {draft_count} case(s); "
            f"reviewed spn quarantine contains {spn_count} interval(s)."
        )
    else:
        print(f"Approved gold packages validated for {approved_count} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
