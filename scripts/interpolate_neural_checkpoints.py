import argparse
import pathlib
import struct


HEADER = struct.Struct("<6I")
MAGIC = 0x4C504E53


def read_checkpoint(path: pathlib.Path) -> tuple[bytes, tuple[float, ...]]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"checkpoint is truncated: {path}")
    fields = HEADER.unpack_from(data)
    if fields[0] != MAGIC:
        raise ValueError(f"checkpoint schema mismatch: {path}")
    count = fields[5]
    expected = HEADER.size + count * 4
    if len(data) != expected:
        raise ValueError(f"checkpoint size mismatch: {path}")
    return data[: HEADER.size], struct.unpack_from(f"<{count}f", data, HEADER.size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate two schema-v3 neural streamer checkpoints."
    )
    parser.add_argument("base", type=pathlib.Path)
    parser.add_argument("trained", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between zero and one")

    base_header, base = read_checkpoint(args.base)
    trained_header, trained = read_checkpoint(args.trained)
    if base_header != trained_header or len(base) != len(trained):
        raise ValueError("checkpoint layouts differ")
    values = tuple(
        left + args.alpha * (right - left)
        for left, right in zip(base, trained, strict=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base_header + struct.pack(f"<{len(values)}f", *values))
    print(f"Wrote {args.output} with alpha={args.alpha:.6f}")


if __name__ == "__main__":
    main()
