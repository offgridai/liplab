import argparse
import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
LINES_PATH = ROOT / "inputs" / "speech_library" / "neural_scheduler_lines_v1.json"
DEFAULT_OUTPUT = ROOT / "inputs" / "speech_library" / "neural_scheduler_corpus_v1.json"
NAMESPACE = "liplab-neural-scheduler-corpus-v1"
VOICES = [
    {"id": "alfie_1_7b", "json": "reference/alfie_1.7B_f16.json"},
    {"id": "lana_1_7b", "json": "reference/lana_1.7b_f16.json"},
    {"id": "priestley_1_7b", "json": "reference/priestley_1.7b_f16.json"},
    {"id": "priestley2_1_7b", "json": "reference/priestley2_1.7b_f16.json"},
]
SEED_RETRIES = {
    "long_004__priestley2_1_7b": 1,
    "long_list_008__priestley2_1_7b": 1,
    "long_list_009__priestley2_1_7b": 1,
    "long_list_010__lana_1_7b": 2,
    "sentences_004__priestley2_1_7b": 3,
    "sentences_008__lana_1_7b": 1,
    "sentences_008__priestley2_1_7b": 1,
}


def seed_for(line_id: str, voice_id: str) -> int:
    retry = SEED_RETRIES.get(f"{line_id}__{voice_id}", 0)
    material = f"{NAMESPACE}\0{line_id}\0{voice_id}"
    if retry:
        material += f"\0retry={retry}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 2_147_483_646


def load_lines() -> list[dict[str, str]]:
    payload = json.loads(LINES_PATH.read_text(encoding="utf-8"))
    lines = payload.get("lines", [])
    if len(lines) != 100:
        raise SystemExit(f"expected exactly 100 lines, found {len(lines)}")
    ids: set[str] = set()
    texts: set[str] = set()
    for line in lines:
        line_id = str(line.get("id", ""))
        text = str(line.get("text", "")).strip()
        if not line_id or not text or line_id in ids or text.casefold() in texts:
            raise SystemExit(f"invalid or duplicate line: {line!r}")
        ids.add(line_id)
        texts.add(text.casefold())
    return lines


def build_payload(write_transcripts: bool) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    transcript_root = ROOT / "inputs" / "transcripts"
    if write_transcripts:
        transcript_root.mkdir(parents=True, exist_ok=True)
    for line in load_lines():
        for voice in VOICES:
            case_id = f"factory_v1_{line['id']}__{voice['id']}"
            source = transcript_root / f"{case_id}.txt"
            text = str(line["text"]).strip()
            if write_transcripts:
                source.write_text(text + "\n", encoding="utf-8")
            elif not source.exists() or source.read_text(encoding="utf-8-sig").strip() != text:
                raise SystemExit(f"missing or stale generated transcript: {source}")
            cases.append(
                {
                    "id": case_id,
                    "source_transcript": source.relative_to(ROOT).as_posix(),
                    "line_id": line["id"],
                    "category": line["category"],
                    "text": text,
                    "seed": seed_for(str(line["id"]), str(voice["id"])),
                    "voice_id": voice["id"],
                }
            )
    return {
        "schema_version": 1,
        "recipe_id": "neural_scheduler_corpus_v1",
        "description": "One reproducible 1.7B Qwen utterance for every transcript and voice pairing.",
        "assignment": {
            "algorithm": "sha256 namespace + line id + voice id",
            "namespace": NAMESPACE,
            "seed_range": [1, 2_147_483_646],
            "cross_product": "100 transcripts x 4 voices",
            "seed_retries_after_mfa_rejection": SEED_RETRIES,
        },
        "source_lines": LINES_PATH.relative_to(ROOT).as_posix(),
        "voices": VOICES,
        "line_count": 100,
        "case_count": len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic 100-line, four-voice neural corpus recipe."
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_payload(write_transcripts=not args.check)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"neural corpus speech recipe is stale: {output}")
        print(f"Neural corpus speech recipe is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(payload['cases'])} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
