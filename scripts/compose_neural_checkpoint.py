#!/usr/bin/env python3
"""Compose independently validated token and speech-region neural heads."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


MAGIC = 0x4C504E53
VERSION = 3
AUDIO_FEATURES = 18
TOKEN_FEATURES = 64 + 32 + 10
HIDDEN = 64
HEADER = struct.Struct("<6I")

MEAN_COUNT = AUDIO_FEATURES
SCALE_COUNT = AUDIO_FEATURES
AUDIO_CONV1 = HIDDEN * AUDIO_FEATURES * 3 + HIDDEN
AUDIO_CONV2 = HIDDEN * HIDDEN * 3 + HIDDEN
TOKEN_LINEAR1 = HIDDEN * TOKEN_FEATURES + HIDDEN
TOKEN_LINEAR2 = HIDDEN * HIDDEN + HIDDEN
REGION_OFFSET = (
    MEAN_COUNT + SCALE_COUNT + AUDIO_CONV1 + AUDIO_CONV2
    + TOKEN_LINEAR1 + TOKEN_LINEAR2
)
REGION_COUNT = AUDIO_CONV1 + AUDIO_CONV2 + HIDDEN + 1
PARAMETER_COUNT = REGION_OFFSET + REGION_COUNT


def read_checkpoint(path: Path) -> tuple[bytes, list[float]]:
    data = path.read_bytes()
    if len(data) != HEADER.size + 4 * PARAMETER_COUNT:
        raise ValueError(f"{path}: unexpected checkpoint size {len(data)}")
    fields = HEADER.unpack_from(data)
    expected = (
        MAGIC, VERSION, AUDIO_FEATURES, TOKEN_FEATURES, HIDDEN, PARAMETER_COUNT
    )
    if fields != expected:
        raise ValueError(f"{path}: incompatible checkpoint header {fields}")
    values = list(struct.unpack_from(f"<{PARAMETER_COUNT}f", data, HEADER.size))
    return data[: HEADER.size], values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-checkpoint", type=Path, required=True)
    parser.add_argument("--region-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    header, token_values = read_checkpoint(args.token_checkpoint)
    _, region_values = read_checkpoint(args.region_checkpoint)
    normalization_count = MEAN_COUNT + SCALE_COUNT
    maximum_normalization_delta = max(
        abs(a - b)
        for a, b in zip(
            token_values[:normalization_count],
            region_values[:normalization_count],
        )
    )
    if maximum_normalization_delta > 1.0e-6:
        raise ValueError(
            "checkpoints use different audio normalization "
            f"(maximum delta {maximum_normalization_delta:.9g})"
        )

    # Normalization is identical. Preserve the sentence-aware token branch and
    # replace only the wholly independent region branch.
    token_values[REGION_OFFSET:] = region_values[REGION_OFFSET:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        header + struct.pack(f"<{PARAMETER_COUNT}f", *token_values)
    )
    print(
        f"Composed {args.output}: token={args.token_checkpoint}, "
        f"region={args.region_checkpoint}, region_offset={REGION_OFFSET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
