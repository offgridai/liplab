import argparse
import json
import pathlib
import shutil


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a completed Datafactory library into the canonical LipLab corpus."
    )
    parser.add_argument("library_root", type=pathlib.Path)
    args = parser.parse_args()

    library_root = args.library_root.resolve()
    manifest_path = library_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"speech library manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not cases:
        raise SystemExit("speech library manifest contains no cases")

    wav_root = ROOT / "inputs" / "wav"
    transcript_root = ROOT / "inputs" / "transcripts"
    align_root = ROOT / "outputs" / "mfa_align" / "latest"
    wav_root.mkdir(parents=True, exist_ok=True)
    transcript_root.mkdir(parents=True, exist_ok=True)
    align_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_id = str(case["case_id"])
        source_wav = library_root / str(case["wav"])
        alignment = case.get("alignment", {})
        source_grid = library_root / str(alignment.get("textgrid", ""))
        source_transcript = library_root / "transcripts" / f"{case_id}.txt"
        for source in (source_wav, source_grid, source_transcript):
            if not source.is_file():
                raise SystemExit(f"required library artifact missing: {source}")
        shutil.copy2(source_wav, wav_root / f"{case_id}.wav")
        shutil.copy2(source_grid, align_root / f"{case_id}.TextGrid")
        destination_transcript = transcript_root / f"{case_id}.txt"
        expected_text = str(case["transcript"]).strip()
        if destination_transcript.exists():
            actual_text = destination_transcript.read_text(encoding="utf-8-sig").strip()
            if actual_text != expected_text:
                raise SystemExit(f"canonical transcript differs for {case_id}")
        else:
            shutil.copy2(source_transcript, destination_transcript)

    print(f"Imported {len(cases)} speech library cases into the canonical corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
