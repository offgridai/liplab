import argparse
import csv
import io
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
DEFAULT_OUTPUT = INPUTS / "corpus.csv"
RECORDED_RECIPE = INPUTS / "speech_library" / "existing_corpus_v1.json"
FACTORY_RECIPE = INPUTS / "speech_library" / "neural_scheduler_corpus_v1.json"

FIELDS = [
    "case_id",
    "origin",
    "speaker_id",
    "recipe_id",
    "recipe_role",
    "voice_id",
    "seed",
    "line_id",
    "category",
    "transcript",
    "wav",
    "gold",
    "gold_ready",
    "source_uri",
]


def load_rows(path: pathlib.Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["case_id"]: row
            for row in csv.DictReader(handle)
            if row.get("case_id")
        }


def recorded_speaker(case_id: str) -> str:
    match = re.search(r"_([A-Za-z0-9]+)_Line_", case_id)
    return match.group(1).lower() if match else "unknown"


def recipe_cases(path: pathlib.Path) -> tuple[str, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["recipe_id"]), list(payload["cases"])


def build_rows(
    output: pathlib.Path, source_overrides: dict[str, str] | None = None
) -> list[dict[str, object]]:
    preserved = load_rows(output)

    specifications: list[tuple[str, str, str, dict[str, object]]] = []
    recorded_recipe_id, recorded = recipe_cases(RECORDED_RECIPE)
    factory_recipe_id, factory = recipe_cases(FACTORY_RECIPE)
    specifications.extend(
        ("recorded", recorded_recipe_id, "recreation", case) for case in recorded
    )
    specifications.extend(
        ("data_factory", factory_recipe_id, "source", case) for case in factory
    )

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for origin, recipe_id, recipe_role, case in specifications:
        case_id = str(case["id"])
        if case_id in seen:
            raise SystemExit(f"duplicate corpus case in recipes: {case_id}")
        seen.add(case_id)
        transcript = INPUTS / "transcripts" / f"{case_id}.txt"
        wav = INPUTS / "wav" / f"{case_id}.wav"
        gold = INPUTS / "gold" / case_id
        for required in (transcript, wav):
            if not required.is_file():
                raise SystemExit(f"corpus asset missing: {required}")
        if gold.is_dir():
            for name in ("phones.csv", "words.csv", "speech.csv", "boundaries.csv"):
                if not (gold / name).is_file():
                    raise SystemExit(f"incomplete gold package: {gold / name}")
        actual_text = transcript.read_text(encoding="utf-8-sig").strip()
        if actual_text != str(case["text"]).strip():
            raise SystemExit(f"recipe transcript differs from corpus asset: {case_id}")
        old = preserved.get(case_id, {})
        source_uri = (source_overrides or {}).get(
            case_id, old.get("source_uri", old.get("source_dir", ""))
        )
        speaker_id = (
            recorded_speaker(case_id)
            if origin == "recorded"
            else str(case["voice_id"])
        )
        rows.append(
            {
                "case_id": case_id,
                "origin": origin,
                "speaker_id": speaker_id,
                "recipe_id": recipe_id,
                "recipe_role": recipe_role,
                "voice_id": case["voice_id"],
                "seed": case["seed"],
                "line_id": case.get("line_id", ""),
                "category": case.get("category", ""),
                "transcript": transcript.relative_to(ROOT).as_posix(),
                "wav": wav.relative_to(ROOT).as_posix(),
                "gold": gold.relative_to(ROOT).as_posix() if gold.is_dir() else "",
                "gold_ready": "1" if gold.is_dir() else "0",
                "source_uri": source_uri,
            }
        )

    transcript_ids = {path.stem for path in (INPUTS / "transcripts").glob("*.txt")}
    wav_ids = {path.stem for path in (INPUTS / "wav").glob("*.wav")}
    for label, actual in (("transcript", transcript_ids), ("wav", wav_ids)):
        missing = sorted(actual - seen)
        extra = sorted(seen - actual)
        if missing or extra:
            raise SystemExit(
                f"{label} inventory differs from corpus recipes: "
                f"unlisted={missing[:5]} missing={extra[:5]}"
            )
    return sorted(rows, key=lambda row: str(row["case_id"]))


def render(rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the unified recorded and Data Factory corpus manifest."
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--recorded-source",
        action="append",
        default=[],
        metavar="CASE_ID=URI",
        help="Record source provenance while rebuilding after an Offgrid import.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    source_overrides: dict[str, str] = {}
    for value in args.recorded_source:
        if "=" not in value:
            raise SystemExit("--recorded-source requires CASE_ID=URI")
        case_id, uri = value.split("=", 1)
        source_overrides[case_id] = uri
    rows = build_rows(output, source_overrides)
    rendered = render(rows)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"unified corpus manifest is stale: {output}")
        print(
            f"Unified corpus manifest is current: {len(rows)} cases "
            f"({sum(row['origin'] == 'recorded' for row in rows)} recorded, "
            f"{sum(row['origin'] == 'data_factory' for row in rows)} Data Factory)"
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="")
    print(f"Wrote unified corpus manifest: {output} ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
