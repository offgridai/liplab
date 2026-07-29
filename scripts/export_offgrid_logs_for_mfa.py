#!/usr/bin/env python3
"""Export Offgrid debug WAVs/transcripts as an MFA corpus plus analysis index."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First numeric case index (default: 1).",
    )
    args = parser.parse_args()

    corpus = args.output_root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for number, log_dir in enumerate(
        sorted(args.log_root.iterdir()), args.start_index
    ):
        if not log_dir.is_dir() or not (log_dir / "line_metadata.txt").exists():
            continue
        values = metadata(log_dir / "line_metadata.txt")
        wavs = sorted(log_dir.glob("tts_line_*.wav"))
        if not wavs or not values.get("Dialogue"):
            continue
        case_id = f"case_{number:04d}_{log_dir.name}"
        shutil.copy2(wavs[0], corpus / f"{case_id}.wav")
        (corpus / f"{case_id}.lab").write_text(
            values["Dialogue"] + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "case_id": case_id,
                "source_dir": str(log_dir),
                "dialogue": values["Dialogue"],
            }
        )
    with (args.output_root / "index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "source_dir", "dialogue"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} log lines to {corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
