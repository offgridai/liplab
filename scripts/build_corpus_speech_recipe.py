import argparse
import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "inputs" / "speech_library" / "existing_corpus_v1.json"
ASSIGNMENT_NAMESPACE = "liplab-existing-corpus-v1"
VOICES = [
    {"id": "alfie", "json": "reference/alfie_0.6b_f16.json"},
    {"id": "lana", "json": "reference/lana_0.6b_f16.json"},
    {"id": "priestley", "json": "reference/priestley_0.6b_f16.json"},
    {"id": "priestley2", "json": "reference/priestley2_0.6b_f16.json"},
]


def assignment(case_id: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{ASSIGNMENT_NAMESPACE}\0{case_id}".encode("utf-8")).digest()
    seed = 1 + int.from_bytes(digest[:4], "big") % 2_147_483_646
    voice_id = VOICES[int.from_bytes(digest[4:8], "big") % len(VOICES)]["id"]
    return seed, voice_id


def build_payload() -> dict[str, object]:
    transcripts_root = ROOT / "inputs" / "transcripts"
    cases: list[dict[str, object]] = []
    for transcript_path in sorted(transcripts_root.glob("case_*.txt")):
        seed, voice_id = assignment(transcript_path.stem)
        cases.append(
            {
                "id": transcript_path.stem,
                "source_transcript": transcript_path.relative_to(ROOT).as_posix(),
                "text": transcript_path.read_text(encoding="utf-8-sig").strip(),
                "seed": seed,
                "voice_id": voice_id,
            }
        )
    if not cases:
        raise SystemExit(f"no corpus transcripts found under {transcripts_root}")
    return {
        "schema_version": 1,
        "recipe_id": "existing_corpus_v1",
        "description": "One reproducible Qwen utterance for every checked-in liplab corpus transcript.",
        "assignment": {
            "algorithm": "sha256 namespace + case id",
            "namespace": ASSIGNMENT_NAMESPACE,
            "seed_range": [1, 2_147_483_646],
            "voice_selection": "digest modulo ordered voices list",
        },
        "voices": VOICES,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic data-factory recipe for the existing corpus."
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = json.dumps(build_payload(), indent=2, ensure_ascii=False) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"corpus speech recipe is stale: {output}")
        print(f"Corpus speech recipe is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(json.loads(rendered)['cases'])} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
