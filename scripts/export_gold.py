import argparse
import csv
import datetime as dt

from gold_tools import (
    approved_gold_case_dir,
    case_stems,
    gold_draft_dir,
    gold_inputs_root,
    gold_manifest_path,
    read_json,
    repo_root,
    write_json,
)


REQUIRED_LAYER_STATUSES = {
    "speech_regions": {"reviewed_boundary", "reviewed_dense", "approved_gold"},
    "word_heads": {"reviewed_boundary", "reviewed_dense", "approved_gold"},
    "dense_visemes": {"reviewed_dense", "approved_gold"},
}


def approval_errors(case_id: str, payload: dict) -> list[str]:
    errors: list[str] = []
    approval = payload.get("approval", {})
    if approval.get("status") != "approved_gold":
        errors.append(f"{case_id}: approval.status must be approved_gold")

    review_layers = payload.get("review_layers", {})
    for layer_name, allowed in REQUIRED_LAYER_STATUSES.items():
        status = review_layers.get(layer_name, {}).get("status", "draft_auto")
        if status not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            errors.append(f"{case_id}: review_layers.{layer_name}.status must be one of {allowed_list}")
    return errors


def _phone_rows(payload: dict) -> list[dict]:
    rows = payload.get("visemes", [])
    return rows if isinstance(rows, list) else []


def export_visemes(case_dir, payload: dict) -> None:
    # Compatibility export: the reviewer currently reads visemes.csv, but these rows
    # are now MFA/CMU phone intervals. pose is intentionally set to the phone label.
    dest = case_dir / "visemes.csv"
    fields = [
        "start",
        "end",
        "pose",
        "phone",
        "word",
        "confidence",
        "word_index",
        "speech_region_index",
        "phrase_index",
        "sentence_index",
        "source_phone",
        "source_phone_index",
        "alignment_reason",
    ]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in _phone_rows(payload):
            phone = row.get("phone") or row.get("source_phone") or row.get("pose", "")
            writer.writerow(
                [
                    f"{float(row.get('start', 0.0)):.6f}",
                    f"{float(row.get('end', 0.0)):.6f}",
                    phone,
                    phone,
                    row.get("word", ""),
                    "1.0",
                    int(row.get("word_index", -1)),
                    int(row.get("speech_region_index", row.get("phrase_index", -1))),
                    int(row.get("phrase_index", -1)),
                    int(row.get("sentence_index", -1)),
                    phone,
                    int(row.get("source_phone_index", -1)),
                    row.get("alignment_reason", "mfa_textgrid_phone"),
                ]
            )


def export_phones(case_dir, payload: dict) -> None:
    # Canonical export for new tooling. Same content as visemes.csv, named plainly.
    dest = case_dir / "phones.csv"
    fields = [
        "start",
        "end",
        "phone",
        "word",
        "confidence",
        "word_index",
        "speech_region_index",
        "phone_index",
        "phrase_index",
        "sentence_index",
        "alignment_reason",
    ]
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in _phone_rows(payload):
            phone = row.get("phone") or row.get("source_phone") or row.get("pose", "")
            writer.writerow(
                [
                    f"{float(row.get('start', 0.0)):.6f}",
                    f"{float(row.get('end', 0.0)):.6f}",
                    phone,
                    row.get("word", ""),
                    "1.0",
                    int(row.get("word_index", -1)),
                    int(row.get("speech_region_index", row.get("phrase_index", -1))),
                    int(row.get("source_phone_index", -1)),
                    int(row.get("phrase_index", -1)),
                    int(row.get("sentence_index", -1)),
                    row.get("alignment_reason", "mfa_textgrid_phone"),
                ]
            )


def export_words(case_dir, payload: dict) -> None:
    dest = case_dir / "words.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word_index", "word", "start", "end", "speech_region_index", "phrase_index", "sentence_index"])
        for word in payload.get("gold_words", []):
            writer.writerow(
                [
                    int(word.get("word_index", -1)),
                    word.get("word", ""),
                    f"{float(word.get('start', 0.0)):.6f}",
                    f"{float(word.get('end', 0.0)):.6f}",
                    int(word.get("speech_region_index", word.get("phrase_index", -1))),
                    int(word.get("phrase_index", -1)),
                    int(word.get("sentence_index", -1)),
                ]
            )


def export_speech(case_dir, payload: dict) -> None:
    dest = case_dir / "speech.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "start", "end", "word_start_index", "word_end_index", "source"])
        for region in payload.get("speech_regions", []):
            writer.writerow(
                [
                    int(region.get("index", -1)),
                    f"{float(region.get('start', 0.0)):.6f}",
                    f"{float(region.get('end', 0.0)):.6f}",
                    int(region.get("word_start_index", -1)),
                    int(region.get("word_end_index", -1)),
                    region.get("source", ""),
                ]
            )


def export_boundaries(case_dir, payload: dict) -> None:
    dest = case_dir / "boundaries.csv"
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "word_index",
                "word",
                "next_word_index",
                "next_word",
                "mark",
                "mark_sequence",
                "pause_class",
                "split_applied",
                "split_reason",
                "direct_word_gap_seconds",
                "phone_gap_seconds",
                "wave_silence_seconds",
                "acoustic_gap_seconds",
                "speech_region_index_before",
                "speech_region_index_after",
            ]
        )
        for boundary in payload.get("pause_boundaries", []):
            writer.writerow(
                [
                    int(boundary.get("word_index", -1)),
                    boundary.get("word", ""),
                    int(boundary.get("next_word_index", -1)),
                    boundary.get("next_word", ""),
                    boundary.get("mark", ""),
                    boundary.get("mark_sequence", ""),
                    boundary.get("pause_class", ""),
                    1 if bool(boundary.get("split_applied")) else 0,
                    boundary.get("split_reason", ""),
                    f"{float(boundary.get('direct_word_gap_seconds', 0.0)):.6f}",
                    f"{float(boundary.get('phone_gap_seconds', 0.0)):.6f}",
                    f"{float(boundary.get('wave_silence_seconds', 0.0)):.6f}",
                    f"{float(boundary.get('acoustic_gap_seconds', 0.0)):.6f}",
                    int(boundary.get("speech_region_index_before", -1)),
                    int(boundary.get("speech_region_index_after", -1)),
                ]
            )


def export_metadata(case_id: str, case_dir, payload: dict) -> None:
    metadata = {
        "schema_version": 1,
        "case_id": case_id,
        "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "approval": payload.get("approval", {}),
        "review_layers": payload.get("review_layers", {}),
        "reviewer": payload.get("approval", {}).get("reviewer", ""),
        "source_draft": gold_draft_dir(case_id).relative_to(repo_root()).as_posix(),
        "summary": payload.get("summary", {}),
    }
    write_json(case_dir / "metadata.json", metadata)


def export_case(case_id: str) -> bool:
    draft_path = gold_draft_dir(case_id) / "draft.annotation.json"
    if not draft_path.exists():
        return False

    payload = read_json(draft_path)
    errors = approval_errors(case_id, payload)
    if errors:
        for error in errors:
            print(f"SKIP {error}")
        return False

    case_dir = approved_gold_case_dir(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    export_visemes(case_dir, payload)
    export_phones(case_dir, payload)
    export_words(case_dir, payload)
    export_speech(case_dir, payload)
    export_boundaries(case_dir, payload)
    export_metadata(case_id, case_dir, payload)
    return True


def write_manifest() -> None:
    root = gold_inputs_root()
    records: list[dict] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = read_json(metadata_path)
        records.append(
            {
                "case_id": metadata.get("case_id", metadata_path.parent.name),
                "approval": metadata.get("approval", {}),
                "review_layers": metadata.get("review_layers", {}),
                "reviewer": metadata.get("reviewer", ""),
                "summary": metadata.get("summary", {}),
                "path": metadata_path.parent.relative_to(repo_root()).as_posix(),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_count": len(records),
        "cases": records,
    }
    write_json(gold_manifest_path(), payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export approved gold drafts into grader-facing inputs/gold packages.")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    args = parser.parse_args()

    exported = 0
    for case_id in args.cases or case_stems():
        if export_case(case_id):
            exported += 1
            print(f"{case_id}: exported")

    write_manifest()
    print(f"Exported {exported} approved gold case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
