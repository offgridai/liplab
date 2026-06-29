import csv
import json
import pathlib
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
