import argparse
import csv
import json

from handmade_tools import case_stems, handmade_draft_dir, repo_root


def word_metadata_by_index(payload: dict) -> dict[int, dict]:
    metadata: dict[int, dict] = {}
    for viseme in payload.get("planned_visemes", []):
        word_index = int(viseme.get("word_index", -1))
        if word_index < 0 or word_index in metadata:
            continue
        metadata[word_index] = {
            "word_index": word_index,
            "word": viseme.get("word", ""),
            "phrase_index": int(viseme.get("phrase_index", -1)),
            "sentence_index": int(viseme.get("sentence_index", -1)),
        }
    return metadata


def export_word_timings(root, case_id: str, payload: dict) -> None:
    metadata = word_metadata_by_index(payload)
    dest = root / "inputs" / "handmade_words" / f"{case_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word_index", "word", "start", "end", "phrase_index", "sentence_index"])
        for word in payload.get("mfa_words", []):
            word_index = int(word.get("word_index", -1))
            extra = metadata.get(word_index, {})
            writer.writerow(
                [
                    word_index,
                    word.get("word", extra.get("word", "")),
                    f"{float(word.get('start', 0.0)):.6f}",
                    f"{float(word.get('end', 0.0)):.6f}",
                    extra.get("phrase_index", -1),
                    extra.get("sentence_index", -1),
                ]
            )


def export_speech_regions(root, case_id: str, payload: dict) -> None:
    dest = root / "inputs" / "handmade_speech" / f"{case_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "start", "end", "last_speech", "started", "ended"])
        for region in payload.get("speech_regions", []):
            writer.writerow(
                [
                    int(region.get("index", -1)),
                    f"{float(region.get('start', 0.0)):.6f}",
                    f"{float(region.get('end', 0.0)):.6f}",
                    f"{float(region.get('last_speech', 0.0)):.6f}",
                    "1" if region.get("started", False) else "0",
                    "1" if region.get("ended", False) else "0",
                ]
            )


def should_export(payload: dict, allow_draft: bool) -> bool:
    status = payload.get("approval", {}).get("status", "draft")
    return allow_draft or status == "approved"


def export_case(case_id: str, allow_draft: bool) -> bool:
    root = repo_root()
    draft_path = handmade_draft_dir(case_id) / "draft.annotation.json"
    if not draft_path.exists():
        return False

    payload = json.loads(draft_path.read_text(encoding="utf-8"))
    if not should_export(payload, allow_draft):
        return False

    word_meta = word_metadata_by_index(payload)
    dest = root / "inputs" / "handmade" / f"{case_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start", "end", "pose", "word", "confidence", "word_index", "phrase_index", "sentence_index"])
        for viseme in payload.get("visemes", []):
            word_index = int(viseme.get("word_index", -1))
            extra = word_meta.get(word_index, {})
            writer.writerow(
                [
                    f"{float(viseme['start']):.6f}",
                    f"{float(viseme['end']):.6f}",
                    viseme["pose"],
                    viseme.get("word", ""),
                    "1.0",
                    word_index,
                    extra.get("phrase_index", -1),
                    extra.get("sentence_index", -1),
                ]
            )
    export_word_timings(root, case_id, payload)
    export_speech_regions(root, case_id, payload)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export approved handmade drafts into grader-facing CSV files.")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()

    exported = 0
    for case_id in args.cases or case_stems():
        if export_case(case_id, args.allow_draft):
            exported += 1
            print(f"{case_id}: exported")

    print(f"Exported {exported} handmade CSV file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
