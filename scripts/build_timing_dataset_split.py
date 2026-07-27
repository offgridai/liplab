import argparse
import csv
import hashlib
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_MANIFEST = ROOT / "inputs" / "corpus.csv"
DEFAULT_OUTPUT = ROOT / "inputs" / "timing_split_v1.json"
NAMESPACE = "liplab-timing-split-v1"
HELD_OUT_SPEAKER = "lana_1_7b"


def normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", value.lower()))


def text_split(text: str) -> tuple[str, str]:
    group_hash = hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()
    bucket = int(
        hashlib.sha256(f"{NAMESPACE}\0{group_hash}".encode("utf-8")).hexdigest()[:8],
        16,
    ) % 100
    if bucket < 75:
        return "train", group_hash
    if bucket < 85:
        return "validation", group_hash
    return "text_test", group_hash


def build_payload() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {
        "train": 0,
        "validation": 0,
        "text_test": 0,
        "speaker_holdout": 0,
    }
    with CORPUS_MANIFEST.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    for case in cases:
        text = (ROOT / case["transcript"]).read_text(encoding="utf-8-sig").strip()
        split, group_hash = text_split(text)
        speaker_id = str(case["speaker_id"])
        speaker_holdout = speaker_id == HELD_OUT_SPEAKER and split == "train"
        effective_split = "speaker_holdout" if speaker_holdout else split
        counts[effective_split] += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "text_group_sha256": group_hash,
                "origin": case["origin"],
                "speaker_id": speaker_id,
                "voice_id": case["voice_id"],
                "split": effective_split,
                "text_split": split,
                "speaker_holdout": speaker_holdout,
                "eligible_for_training": effective_split == "train",
            }
        )
    return {
        "schema_version": 1,
        "split_id": "timing_split_v1",
        "source_manifest": CORPUS_MANIFEST.relative_to(ROOT).as_posix(),
        "assignment": {
            "namespace": NAMESPACE,
            "grouping": "normalized transcript text SHA-256",
            "text_buckets": {"train": 75, "validation": 10, "text_test": 15},
            "held_out_speaker": HELD_OUT_SPEAKER,
            "policy": (
                "Text groups never cross train/validation/text_test. Cases using the held-out "
                "speaker inside train text groups move to speaker_holdout."
            ),
        },
        "counts": counts,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen timing-model dataset split.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_payload(), indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"timing dataset split is stale: {output}")
        print(f"Timing dataset split is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    print(f"Wrote timing dataset split: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
