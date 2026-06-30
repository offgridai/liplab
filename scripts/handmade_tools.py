import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import wave


ROOT = pathlib.Path(__file__).resolve().parents[1]


def repo_root() -> pathlib.Path:
    return ROOT


def find_runner() -> pathlib.Path:
    bin_name = "liplab_runner.exe" if sys.platform.startswith("win") else "liplab_runner"
    candidates = [
        ROOT / "build" / bin_name,
        ROOT / "build" / "Release" / bin_name,
        ROOT / "build" / "Debug" / bin_name,
    ]
    exe = next((path for path in candidates if path.exists()), None)
    if not exe:
        raise SystemExit("liplab_runner not built; run cmake first")
    return exe


def run_runner(buffer_ms: int, chunk_ms: int) -> int:
    exe = find_runner()
    cmd = [
        str(exe),
        str(ROOT),
        "--buffer-ms",
        str(buffer_ms),
        "--chunk-ms",
        str(chunk_ms),
    ]
    return subprocess.call(cmd)


def run_runner_and_capture(buffer_ms: int, chunk_ms: int, capture_root: pathlib.Path) -> int:
    rc = run_runner(buffer_ms, chunk_ms)
    if rc != 0:
        return rc

    latest = ROOT / "outputs" / "runs" / "latest"
    if capture_root.exists():
        shutil.rmtree(capture_root)
    capture_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(latest, capture_root)
    return 0


def find_mfa_env() -> pathlib.Path:
    env = ROOT / ".conda" / "mfa"
    if not env.exists():
        raise SystemExit("MFA conda environment not found at .conda/mfa")
    return env


def mfa_root_dir() -> pathlib.Path:
    return ROOT / ".mfa"


def mfa_dictionary_path() -> pathlib.Path:
    path = mfa_root_dir() / "pretrained_models" / "dictionary" / "english_us_arpa.dict"
    if not path.exists():
        raise SystemExit(f"MFA dictionary not found: {path}")
    return path


def mfa_acoustic_model_path() -> pathlib.Path:
    path = mfa_root_dir() / "pretrained_models" / "acoustic" / "english_us_arpa.zip"
    if not path.exists():
        raise SystemExit(f"MFA acoustic model not found: {path}")
    return path


def mfa_corpus_root() -> pathlib.Path:
    return ROOT / "outputs" / "mfa_corpus" / "latest"


def mfa_align_root() -> pathlib.Path:
    return ROOT / "outputs" / "mfa_align" / "latest"


def stage_mfa_corpus(cases: list[str], corpus_root: pathlib.Path) -> None:
    if corpus_root.exists():
        shutil.rmtree(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    for case_id in cases:
        transcript = read_text(ROOT / "inputs" / "transcripts" / f"{case_id}.txt")
        (corpus_root / f"{case_id}.lab").write_text(transcript + "\n", encoding="utf-8")


def run_mfa_align(cases: list[str], output_root: pathlib.Path, num_jobs: int) -> int:
    env_path = find_mfa_env()
    corpus_root = mfa_corpus_root()
    stage_mfa_corpus(cases, corpus_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(mfa_root_dir())
    env["CONDA_NO_PLUGINS"] = "true"
    cmd = [
        "conda",
        "run",
        "-p",
        str(env_path),
        "mfa",
        "align",
        str(corpus_root),
        str(mfa_dictionary_path()),
        str(mfa_acoustic_model_path()),
        str(output_root),
        "--audio_directory",
        str(ROOT / "inputs" / "wav"),
        "--clean",
        "--single_speaker",
        "--num_jobs",
        str(max(num_jobs, 1)),
    ]
    return subprocess.call(cmd, env=env)


def case_stems() -> list[str]:
    transcripts_dir = ROOT / "inputs" / "transcripts"
    return sorted(path.stem for path in transcripts_dir.glob("*.txt"))


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def wav_metadata(path: pathlib.Path) -> dict[str, float | int | str]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        duration_sec = frame_count / sample_rate if sample_rate > 0 else 0.0
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
        }


def latest_run_case_dir(case_id: str) -> pathlib.Path:
    return ROOT / "outputs" / "runs" / "latest" / case_id


def offline_gold_run_root() -> pathlib.Path:
    return ROOT / "outputs" / "offline_gold" / "latest"


def offline_gold_case_dir(case_id: str) -> pathlib.Path:
    return offline_gold_run_root() / case_id


def handmade_draft_dir(case_id: str) -> pathlib.Path:
    return ROOT / "outputs" / "handmade_drafts" / case_id


def mfa_textgrid_path(case_id: str) -> pathlib.Path:
    return mfa_align_root() / f"{case_id}.TextGrid"


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def strip_phone_stress(phone: str) -> str:
    return re.sub(r"\d", "", (phone or "").strip().upper())


def parse_long_textgrid(path: pathlib.Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        raise SystemExit(f"MFA TextGrid missing: {path}")

    tier_name: str | None = None
    current: dict[str, object] | None = None
    tiers: dict[str, list[dict[str, object]]] = {"words": [], "phones": []}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith('name = "'):
            tier_name = line[len('name = "') : -1]
            continue
        if line.startswith("intervals ["):
            current = {"xmin": None, "xmax": None, "text": ""}
            continue
        if current is None or tier_name not in tiers:
            continue
        if line.startswith("xmin = "):
            current["xmin"] = float(line[len("xmin = ") :].strip())
        elif line.startswith("xmax = "):
            current["xmax"] = float(line[len("xmax = ") :].strip())
        elif line.startswith('text = "'):
            current["text"] = line[len('text = "') : -1]
            tiers[tier_name].append(
                {
                    "start": float(current["xmin"] or 0.0),
                    "end": float(current["xmax"] or 0.0),
                    "text": str(current["text"]),
                }
            )
            current = None

    return tiers
