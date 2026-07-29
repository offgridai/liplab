import argparse
import pathlib
import struct


HEADER = struct.Struct("<6I")
MAGIC = 0x4C504E53


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the schema-v4 pause-boundary feature to a v3 checkpoint."
    )
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    data = args.source.read_bytes()
    magic, version, audio, token, hidden, count = HEADER.unpack_from(data)
    if magic != MAGIC or version != 3 or token != 106 or hidden != 64:
        raise ValueError("source is not the expected schema-v3 checkpoint")
    if len(data) != HEADER.size + count * 4:
        raise ValueError("source checkpoint size mismatch")
    values = list(struct.unpack_from(f"<{count}f", data, HEADER.size))

    token_weight_offset = (
        audio + audio
        + hidden * audio * 3 + hidden
        + hidden * hidden * 3 + hidden
    )
    old_weight_count = hidden * token
    before = values[:token_weight_offset]
    old_weights = values[token_weight_offset : token_weight_offset + old_weight_count]
    after = values[token_weight_offset + old_weight_count :]
    new_weights: list[float] = []
    for row in range(hidden):
        begin = row * token
        new_weights.extend(old_weights[begin : begin + token])
        new_weights.append(0.0)
    upgraded = before + new_weights + after
    header = HEADER.pack(MAGIC, 4, audio, token + 1, hidden, len(upgraded))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + struct.pack(f"<{len(upgraded)}f", *upgraded))
    print(f"Wrote schema-v4 checkpoint: {args.output}")


if __name__ == "__main__":
    main()
