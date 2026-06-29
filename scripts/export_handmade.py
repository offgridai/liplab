import argparse
import csv
import json

from handmade_tools import case_stems, handmade_draft_dir, repo_root


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

    dest = root / "inputs" / "handmade" / f"{case_id}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start", "end", "pose", "word", "confidence"])
        for viseme in payload.get("visemes", []):
            writer.writerow(
                [
                    f"{float(viseme['start']):.6f}",
                    f"{float(viseme['end']):.6f}",
                    viseme["pose"],
                    viseme.get("word", ""),
                    "1.0",
                ]
            )
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

