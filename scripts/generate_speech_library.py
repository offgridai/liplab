import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import wave

from gold_tools import (
    build_compound_pronunciation,
    build_unknown_word_pronunciation,
    combine_dictionary_entries,
    find_mfa_env,
    mfa_acoustic_model_path,
    mfa_root_dir,
    normalize_dictionary_word,
    normalize_spoken_numbers,
    parse_long_textgrid,
    transcript_word_candidates,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_LINES = [
    {
        "id": "visible_consonants",
        "text": "Please bring me five blue paper maps.",
    },
    {
        "id": "word_correspondence",
        "text": "We should verify every visible word before the final pause.",
    },
    {
        "id": "careful_tutor",
        "text": "Today the careful tutor compares each spoken sound with the transcript.",
    },
]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_command(repo: pathlib.Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    # The neighboring checkout can have the desktop user's Windows SID while
    # this process runs under the sandbox SID. Trust only the explicitly
    # selected repository for these read-only provenance queries.
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def git_revision(repo: pathlib.Path) -> str:
    result = git_command(repo, "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_dirty(repo: pathlib.Path) -> bool | None:
    result = git_command(repo, "status", "--porcelain")
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def find_qwen_executable(qwen_root: pathlib.Path) -> pathlib.Path:
    executable = "qwen3_streaming_cli.exe" if os.name == "nt" else "qwen3_streaming_cli"
    candidates = [
        qwen_root / "build-ninja-cuda" / "apps" / "streaming_cli" / executable,
        qwen_root / "build-ninja-verify" / "apps" / "streaming_cli" / executable,
        qwen_root / "build" / "apps" / "streaming_cli" / executable,
    ]
    found = next((path for path in candidates if path.exists()), None)
    if found is None:
        raise SystemExit(
            "Qwen streaming CLI not found; pass --qwen-exe or build qwen3_streaming_cli"
        )
    return found.resolve()


def load_lines(path: pathlib.Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_LINES
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("lines", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit("lines JSON must contain a non-empty list")
    lines: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, str):
            row = {"id": f"line_{index:03d}", "text": row}
        if not isinstance(row, dict) or not str(row.get("text", "")).strip():
            raise SystemExit(f"invalid line at index {index}")
        line_id = str(row.get("id", f"line_{index:03d}")).strip()
        if not line_id.replace("_", "").replace("-", "").isalnum():
            raise SystemExit(f"line id must be alphanumeric with '_' or '-': {line_id}")
        lines.append({"id": line_id, "text": str(row["text"]).strip()})
    return lines


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("at least one seed is required")
    return seeds


def load_voices(raw_values: list[str] | None, qwen_root: pathlib.Path) -> list[dict[str, object]]:
    values = raw_values or [str(qwen_root / "reference" / "priestley_0.6b_f16.json")]
    voices: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in values:
        explicit_id, separator, raw_path = raw.partition("=")
        path = pathlib.Path(raw_path if separator else raw).expanduser().resolve()
        voice_id = explicit_id.strip() if separator else path.stem
        if not voice_id or not voice_id.replace("_", "").replace("-", "").isalnum():
            raise SystemExit(
                f"voice id must be alphanumeric with '_' or '-': {voice_id!r}"
            )
        if voice_id in seen:
            raise SystemExit(f"duplicate voice id: {voice_id}")
        if not path.exists():
            raise SystemExit(f"voice JSON not found: {path}")
        seen.add(voice_id)
        voices.append({"id": voice_id, "path": path, "sha256": sha256(path)})
    return voices


def write_named_path_list(
    path: pathlib.Path, rows: list[tuple[str, pathlib.Path]]
) -> None:
    path.write_text(
        "".join(f"{row_id}\t{row_path.resolve()}\n" for row_id, row_path in rows),
        encoding="utf-8",
    )


def wav_metadata(path: pathlib.Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return {
            "sample_rate": rate,
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "frame_count": frames,
            "duration_seconds": frames / rate if rate else 0.0,
        }


def build_dictionary(transcripts: list[str], output_path: pathlib.Path) -> dict[str, object]:
    entries = combine_dictionary_entries()
    selected: dict[str, list[str]] = {}
    fallbacks: dict[str, dict[str, str]] = {}
    unresolved: list[str] = []
    for transcript in transcripts:
        for token in transcript_word_candidates(transcript):
            word = normalize_dictionary_word(token)
            if not word or word in selected:
                continue
            pronunciations = entries.get(word)
            if pronunciations:
                selected[word] = pronunciations
                continue
            compound = build_compound_pronunciation(word, entries)
            if compound:
                selected[word] = [compound]
                fallbacks[word] = {"source": "compound", "pronunciation": compound}
                continue
            pronunciation, source = build_unknown_word_pronunciation(word, entries)
            if pronunciation and source:
                selected[word] = [pronunciation]
                fallbacks[word] = {"source": source, "pronunciation": pronunciation}
            else:
                unresolved.append(word)

    lines: list[str] = []
    for word in sorted(selected):
        for pronunciation in selected[word]:
            lines.append(f"{word}\t{pronunciation}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "word_count": len(selected),
        "fallbacks": fallbacks,
        "unresolved_words": sorted(set(unresolved)),
        "sha256": sha256(output_path),
    }


def mfa_environment() -> tuple[pathlib.Path, dict[str, str]]:
    env_path = find_mfa_env()
    executable = (
        env_path / "Scripts" / "mfa.exe"
        if os.name == "nt"
        else env_path / "bin" / "mfa"
    )
    if not executable.exists():
        raise SystemExit(f"MFA executable not found: {executable}")
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(mfa_root_dir())
    env["CONDA_NO_PLUGINS"] = "true"
    env["PATH"] = os.pathsep.join(
        [
            str(env_path / "Library" / "bin"),
            str(env_path / "Scripts"),
            str(env_path / "bin"),
            env.get("PATH", ""),
        ]
    )
    return executable, env


def run_mfa(
    corpus_root: pathlib.Path,
    wav_root: pathlib.Path,
    dictionary: pathlib.Path,
    align_root: pathlib.Path,
    num_jobs: int,
) -> list[str]:
    executable, env = mfa_environment()
    align_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "align",
        str(corpus_root),
        str(dictionary),
        str(mfa_acoustic_model_path()),
        str(align_root),
        "--audio_directory",
        str(wav_root),
        "--clean",
        "--single_speaker",
        "--num_jobs",
        str(max(num_jobs, 1)),
    ]
    print("Running MFA:", " ".join(command))
    subprocess.run(command, env=env, check=True)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible Qwen TTS speech library and align it with MFA."
    )
    parser.add_argument(
        "--qwen-root",
        type=pathlib.Path,
        default=ROOT.parent / "qwen3-tts-cpp-streaming",
    )
    parser.add_argument("--qwen-exe", type=pathlib.Path)
    parser.add_argument(
        "--voice-json",
        action="append",
        metavar="[ID=]PATH",
        help="voice-clone JSON; repeat for multiple voices (default: Priestley reference)",
    )
    parser.add_argument("--lines-json", type=pathlib.Path)
    parser.add_argument("--seeds", default="41,42")
    parser.add_argument("--model-identifier", default="qwen3-tts-0.6b-f16")
    parser.add_argument("--library-id", default="demo_v1")
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--mfa-num-jobs", type=int, default=2)
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-mfa", action="store_true")
    args = parser.parse_args()

    qwen_root = args.qwen_root.resolve()
    qwen_exe = (args.qwen_exe.resolve() if args.qwen_exe else find_qwen_executable(qwen_root))
    voices = load_voices(args.voice_json, qwen_root)
    models_root = (qwen_root / "models").resolve()
    if not qwen_exe.exists():
        raise SystemExit(f"Qwen executable not found: {qwen_exe}")
    if not models_root.exists():
        raise SystemExit(f"Qwen models directory not found: {models_root}")

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / "outputs" / "speech_library" / args.library_id).resolve()
    )
    wav_root = output_root / "wav"
    transcript_root = output_root / "transcripts"
    corpus_root = output_root / "mfa_corpus"
    align_root = output_root / "mfa_align"
    batch_input_root = output_root / "batch_inputs"
    batch_transcript_root = batch_input_root / "transcripts"
    for path in (
        wav_root,
        transcript_root,
        corpus_root,
        align_root,
        batch_input_root,
        batch_transcript_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    lines = load_lines(args.lines_json)
    seeds = parse_seeds(args.seeds)
    old_manifest_path = output_root / "manifest.json"
    old_manifest = (
        json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old_manifest_path.exists()
        else {}
    )
    old_hashes = {
        row["case_id"]: row.get("wav_sha256")
        for row in old_manifest.get("cases", [])
        if isinstance(row, dict) and row.get("case_id")
    }

    transcript_list_rows: list[tuple[str, pathlib.Path]] = []
    for line in lines:
        batch_transcript_path = batch_transcript_root / f"{line['id']}.txt"
        batch_transcript_path.write_text(line["text"] + "\n", encoding="utf-8")
        transcript_list_rows.append((line["id"], batch_transcript_path))

    transcript_list_path = batch_input_root / "transcripts.tsv"
    voice_list_path = batch_input_root / "voices.tsv"
    write_named_path_list(transcript_list_path, transcript_list_rows)
    write_named_path_list(
        voice_list_path,
        [(str(voice["id"]), pathlib.Path(voice["path"])) for voice in voices],
    )

    batch_command = [
        str(qwen_exe),
        "-m",
        str(models_root),
        "--model-identifier",
        args.model_identifier,
        "--tts-profile",
        "offgrid-callback",
        "--temperature",
        "0.75",
        "--top-k",
        "16",
        "--top-p",
        "0.9",
        "--cb0-top-p",
        "1.0",
        "--repetition-penalty",
        "1.02",
        "--max-frames-per-text-token",
        "5",
        "--min-dynamic-tokens",
        "64",
        "--no-play-streaming",
        "--simulate-stream-callback",
        "--quiet-all",
        "--batch-transcript-list",
        str(transcript_list_path),
        "--batch-voice-list",
        str(voice_list_path),
        "--batch-seeds",
        ",".join(str(seed) for seed in seeds),
        "--batch-output-dir",
        str(wav_root),
    ]
    if not args.skip_synthesis:
        print(
            f"Synthesizing {len(lines)} transcripts x {len(voices)} voices x "
            f"{len(seeds)} seeds"
        )
        subprocess.run(batch_command, cwd=qwen_root, check=True)

    cases: list[dict[str, object]] = []
    for line in lines:
        for voice in voices:
            for seed in seeds:
                case_id = f"{line['id']}__{voice['id']}__seed_{seed}"
                wav_path = wav_root / f"{case_id}.wav"
                transcript_path = transcript_root / f"{case_id}.txt"
                lab_path = corpus_root / f"{case_id}.lab"
                transcript_path.write_text(line["text"] + "\n", encoding="utf-8")
                lab_path.write_text(
                    normalize_spoken_numbers(line["text"]) + "\n", encoding="utf-8"
                )
                if not wav_path.exists():
                    raise SystemExit(f"generated WAV missing: {wav_path}")
                wav_hash = sha256(wav_path)
                case = {
                    "case_id": case_id,
                    "line_id": line["id"],
                    "voice_id": voice["id"],
                    "seed": seed,
                    "transcript": line["text"],
                    "wav": wav_path.relative_to(output_root).as_posix(),
                    "wav_sha256": wav_hash,
                    "matches_previous_wav": old_hashes.get(case_id) == wav_hash
                    if case_id in old_hashes
                    else None,
                    "wav_metadata": wav_metadata(wav_path),
                }
                cases.append(case)

    dictionary_path = output_root / "mfa_dictionary.dict"
    dictionary_report = build_dictionary(
        [str(case["transcript"]) for case in cases], dictionary_path
    )
    if dictionary_report["unresolved_words"]:
        raise SystemExit(
            "unresolved MFA dictionary words: "
            + ", ".join(dictionary_report["unresolved_words"])
        )

    previous_mfa = old_manifest.get("mfa", {})
    mfa_command: list[str] | None = (
        previous_mfa.get("command") if isinstance(previous_mfa, dict) else None
    )
    if not args.skip_mfa:
        mfa_command = run_mfa(
            corpus_root, wav_root, dictionary_path, align_root, args.mfa_num_jobs
        )

    alignment_rows: list[dict[str, object]] = []
    for case in cases:
        textgrid_path = align_root / f"{case['case_id']}.TextGrid"
        if not textgrid_path.exists():
            if args.skip_mfa:
                continue
            raise SystemExit(f"MFA TextGrid missing: {textgrid_path}")
        tiers = parse_long_textgrid(textgrid_path)
        words = [row for row in tiers["words"] if str(row["text"]).strip()]
        phones = [row for row in tiers["phones"] if str(row["text"]).strip()]
        alignment = {
            "case_id": case["case_id"],
            "textgrid": textgrid_path.relative_to(output_root).as_posix(),
            "textgrid_sha256": sha256(textgrid_path),
            "word_intervals": len(words),
            "phone_intervals": len(phones),
            "aligned_start_seconds": words[0]["start"] if words else None,
            "aligned_end_seconds": words[-1]["end"] if words else None,
        }
        alignment_rows.append(alignment)
        case["alignment"] = alignment

    manifest = {
        "schema_version": 2,
        "library_id": args.library_id,
        "qwen": {
            "repo": str(qwen_root),
            "revision": git_revision(qwen_root),
            "dirty": git_dirty(qwen_root),
            "executable": str(qwen_exe),
            "executable_sha256": sha256(qwen_exe),
            "model_identifier": args.model_identifier,
            "batch_command": batch_command,
            "voices": [
                {
                    "id": voice["id"],
                    "json": str(voice["path"]),
                    "json_sha256": voice["sha256"],
                }
                for voice in voices
            ],
        },
        "sampling": {
            "seeds": seeds,
            "temperature": 0.75,
            "top_k": 16,
            "top_p": 0.9,
            "cb0_top_p": 1.0,
            "repetition_penalty": 1.02,
            "max_frames_per_text_token": 5,
            "min_dynamic_tokens": 64,
        },
        "mfa": {
            "acoustic_model": str(mfa_acoustic_model_path()),
            "dictionary": dictionary_path.relative_to(output_root).as_posix(),
            "dictionary_report": dictionary_report,
            "command": mfa_command,
        },
        "case_count": len(cases),
        "all_previous_wavs_matched": bool(cases)
        and all(case["matches_previous_wav"] is True for case in cases),
        "cases": cases,
    }
    write_json(old_manifest_path, manifest)
    write_json(
        output_root / "alignment_summary.json",
        {"case_count": len(alignment_rows), "cases": alignment_rows},
    )
    print(f"Wrote speech library: {output_root}")
    print(f"Cases: {len(cases)}; MFA alignments: {len(alignment_rows)}")
    if old_hashes:
        print(
            "WAV reproducibility: "
            f"{sum(case['matches_previous_wav'] is True for case in cases)}/{len(cases)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
