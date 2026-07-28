#!/usr/bin/env python3
"""Fail when a CUDA archive is missing required native GPU machine code."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--required-if-supported", action="append", default=[])
    args = parser.parse_args()

    if not args.required and not args.required_if_supported:
        parser.error("at least one architecture requirement is needed")

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise SystemExit("cuobjdump is required to audit CUDA runtime architectures")
    if not args.archive.is_file():
        raise SystemExit(f"CUDA archive not found: {args.archive}")

    required = list(args.required)
    if args.required_if_supported:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise SystemExit("nvcc is required for conditional architecture audits")
        supported = subprocess.run(
            [nvcc, "--list-gpu-code"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        required.extend(
            arch for arch in args.required_if_supported if arch in supported
        )

    result = subprocess.run(
        [cuobjdump, "--list-elf", str(args.archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    missing = [arch for arch in required if f".{arch}.cubin" not in result.stdout]
    if missing:
        raise SystemExit(
            f"{args.archive} is missing native CUDA code for: {', '.join(missing)}"
        )
    print(
        f"CUDA architecture audit passed for {args.archive}: "
        + ", ".join(required)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
