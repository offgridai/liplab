import argparse
import hashlib
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPES = [
    ROOT / "inputs" / "speech_library" / "existing_corpus_v1.json",
    ROOT / "inputs" / "speech_library" / "neural_scheduler_corpus_v1.json",
]
DEFAULT_OUTPUT = ROOT / "inputs" / "speech_library" / "timing_split_v1.json"
NAMESPACE = "liplab-timing-split-v1"
HELD_OUT_VOICE = "lana"


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
    for recipe_path in RECIPES:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        for case in recipe["cases"]:
            split, group_hash = text_split(case["text"])
            is_held_out_voice = str(case["voice_id"]).startswith(HELD_OUT_VOICE)
            speaker_holdout = is_held_out_voice and split == "train"
            effective_split = "speaker_holdout" if speaker_holdout else split
            counts[effective_split] += 1
            rows.append(
                {
                    "case_id": case["id"],
                    "text_group_sha256": group_hash,
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
        "source_recipes": [path.relative_to(ROOT).as_posix() for path in RECIPES],
        "assignment": {
            "namespace": NAMESPACE,
            "grouping": "normalized transcript text SHA-256",
            "text_buckets": {"train": 75, "validation": 10, "text_test": 15},
            "held_out_voice": HELD_OUT_VOICE,
            "policy": (
                "Text groups never cross train/validation/text_test. Cases using the held-out "
                "voice inside train text groups move to speaker_holdout."
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
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote timing dataset split: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
