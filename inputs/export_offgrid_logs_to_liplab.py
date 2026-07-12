#!/usr/bin/env python3
r"""
export_offgrid_logs_to_liplab.py

Convert OffgridAI lipsync debug log folders into liplab input folders.

Expected Offgrid log folder contents:
  line_metadata.txt          # contains Dialogue=...
  tts_line_*.wav             # source TTS waveform

Output:
  <liplab_root>/inputs/transcripts/<case_id>.txt
  <liplab_root>/inputs/wav/<case_id>.wav

Optional:
  <liplab_root>/inputs/index.csv

Usage:
  python export_offgrid_logs_to_liplab.py "C:\path\to\Saved\Logs\OffgridAI\LipsyncDebug" "C:\git\liplab"

Or explicitly:
  python export_offgrid_logs_to_liplab.py --source "C:\logs\LipsyncDebug" --liplab "C:\git\liplab"
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportCase:
    source_dir: Path
    case_id: str
    transcript: str
    wav_path: Path
    transcript_out: Path
    wav_out: Path


def sanitize_token(value: str, fallback: str = "unknown") -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def parse_metadata(metadata_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = metadata_path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = raw.decode(encoding, errors="replace")
    text = text.replace("â€”", "—").replace("â€“", "–")

    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def find_log_dirs(source_root: Path) -> list[Path]:
    """
    Return directories that look like Offgrid lipsync debug line folders.
    A valid folder has line_metadata.txt and at least one .wav.
    """
    candidates: list[Path] = []

    for metadata in source_root.rglob("line_metadata.txt"):
        folder = metadata.parent
        if any(folder.glob("*.wav")):
            candidates.append(folder)

    return sorted(set(candidates), key=lambda p: str(p).lower())


def choose_wav(log_dir: Path) -> Path | None:
    """
    Prefer Offgrid's tts_line_*.wav, then any wav in the folder.
    """
    preferred = sorted(log_dir.glob("tts_line_*.wav"))
    if preferred:
        return preferred[0]

    wavs = sorted(log_dir.glob("*.wav"))
    return wavs[0] if wavs else None


def make_case_id(index: int, log_dir: Path, metadata: dict[str, str]) -> str:
    """
    Stable, readable liplab case ID.

    Example:
      case_0001_20260629_093726_076_Alfie_Line_D80D...
    """
    npc = sanitize_token(metadata.get("NPCID", "npc"))
    line_id = sanitize_token(metadata.get("LineID", log_dir.name))

    # Many Offgrid folder names begin with a timestamp:
    # 20260629_093726_076_Alfie_Line_...
    timestamp_match = re.match(r"^(\d{8}_\d{6}_\d{3})", log_dir.name)
    timestamp = timestamp_match.group(1) if timestamp_match else f"{index:04d}"

    return sanitize_token(f"case_{index:04d}_{timestamp}_{npc}_{line_id}")


def collect_cases(source_root: Path, liplab_root: Path) -> list[ExportCase]:
    transcript_dir = liplab_root / "inputs" / "transcripts"
    wav_dir = liplab_root / "inputs" / "wav"

    existing_indices: list[int] = []
    for existing in list(transcript_dir.glob("case_*.txt")) + list(wav_dir.glob("case_*.wav")):
        match = re.match(r"^case_(\d+)_", existing.name)
        if match:
            existing_indices.append(int(match.group(1)))
    first_index = max(existing_indices, default=0) + 1

    cases: list[ExportCase] = []

    for i, log_dir in enumerate(find_log_dirs(source_root), start=first_index):
        metadata_path = log_dir / "line_metadata.txt"
        metadata = parse_metadata(metadata_path)

        transcript = metadata.get("Dialogue", "").strip()
        if not transcript:
            print(f"SKIP: no Dialogue= in {metadata_path}", file=sys.stderr)
            continue

        wav_path = choose_wav(log_dir)
        if wav_path is None:
            print(f"SKIP: no wav in {log_dir}", file=sys.stderr)
            continue

        case_id = make_case_id(i, log_dir, metadata)

        cases.append(
            ExportCase(
                source_dir=log_dir,
                case_id=case_id,
                transcript=transcript,
                wav_path=wav_path,
                transcript_out=transcript_dir / f"{case_id}.txt",
                wav_out=wav_dir / f"{case_id}.wav",
            )
        )

    return cases


def export_cases(cases: list[ExportCase], liplab_root: Path, overwrite: bool) -> None:
    transcript_dir = liplab_root / "inputs" / "transcripts"
    wav_dir = liplab_root / "inputs" / "wav"

    transcript_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    index_path = liplab_root / "inputs" / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    existing_sources = {str(Path(row["source_dir"]).resolve()).lower() for row in rows}

    for case in cases:
        source_key = str(case.source_dir.resolve()).lower()
        if source_key in existing_sources:
            continue
        if not overwrite and (case.transcript_out.exists() or case.wav_out.exists()):
            print(f"SKIP existing: {case.case_id}", file=sys.stderr)
            continue

        case.transcript_out.write_text(case.transcript + "\n", encoding="utf-8")
        shutil.copy2(case.wav_path, case.wav_out)

        rows.append(
            {
                "case_id": case.case_id,
                "transcript": str(case.transcript_out.relative_to(liplab_root)),
                "wav": str(case.wav_out.relative_to(liplab_root)),
                "source_dir": str(case.source_dir),
                "text": case.transcript,
            }
        )
        existing_sources.add(source_key)

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case_id", "transcript", "wav", "source_dir", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} case(s)")
    print(f"Transcripts: {transcript_dir}")
    print(f"WAVs:        {wav_dir}")
    print(f"Index:       {index_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export OffgridAI LipsyncDebug logs into liplab inputs."
    )
    parser.add_argument(
        "source_pos",
        nargs="?",
        help="Root OffgridAI LipsyncDebug directory, or a parent containing it.",
    )
    parser.add_argument(
        "liplab_pos",
        nargs="?",
        help="liplab repository root.",
    )
    parser.add_argument(
        "--source",
        dest="source_opt",
        help="Root OffgridAI LipsyncDebug directory, or a parent containing it.",
    )
    parser.add_argument(
        "--liplab",
        dest="liplab_opt",
        help="liplab repository root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript/wav files.",
    )

    args = parser.parse_args()

    source_root = Path(args.source_opt or args.source_pos or "").expanduser()
    liplab_root = Path(args.liplab_opt or args.liplab_pos or "").expanduser()

    if not source_root:
        parser.error("missing source root")
    if not liplab_root:
        parser.error("missing liplab root")

    source_root = source_root.resolve()
    liplab_root = liplab_root.resolve()

    if not source_root.exists():
        print(f"ERROR: source root does not exist: {source_root}", file=sys.stderr)
        return 2

    if not liplab_root.exists():
        print(f"ERROR: liplab root does not exist: {liplab_root}", file=sys.stderr)
        return 2

    cases = collect_cases(source_root, liplab_root)
    if not cases:
        print(f"ERROR: no valid Offgrid log folders found under {source_root}", file=sys.stderr)
        return 1

    export_cases(cases, liplab_root, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
