#!/usr/bin/env python3
"""Sweep LipLab preroll/buffer length and summarize each run incrementally.

The runner side should accept either:
  liplab_runner.exe <root> --preroll-ms 350
or, with the companion harness patch in this zip:
  liplab_runner.exe --root <root> --preroll-ms 350

Partial summaries are written after every completed preroll, so long sweeps can be
interrupted without losing completed results.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

DEFAULT_PREROLLS = [150, 350, 750, 1500, 3000, 10000]

PATTERNS = {
    "cases_total": re.compile(r"\bCASES\b.*?\btotal=(\d+)"),
    "graded": re.compile(r"\bCASES\b.*?\bgraded=(\d+)"),
    "phoneme_center_ms": re.compile(r"\bGRADE\b.*?\bcenter_ms=([-0-9.]+)"),
    "phoneme_start_ms": re.compile(r"\bGRADE\b.*?\bstart_ms=([-0-9.]+)"),
    "phoneme_end_ms": re.compile(r"\bGRADE\b.*?\bend_ms=([-0-9.]+)"),
    "word_start_ms": re.compile(r"\bWORDS\b.*?\bstart_ms=([-0-9.]+)"),
    "intra_word_coverage": re.compile(r"\bWORDS\b.*?\bintra_word_coverage=([-0-9.]+)"),
    "direct_center_ms": re.compile(r"\bDIRECT_ALIGNER\b.*?\bcenter_ms=([-0-9.]+)"),
    "audio_rows": re.compile(r"\bAUDIO_PROGRESS\b rows=(\d+)"),
    "audio_mfa_rows": re.compile(r"\bAUDIO_PROGRESS_MFA\b.*?\brows=(\d+)"),
    "audio_mae01": re.compile(r"\bAUDIO_PROGRESS_MFA\b.*?\bmae01=([-0-9.]+)"),
    "audio_mae_ms": re.compile(r"\bAUDIO_PROGRESS_MFA\b.*?\bmae_ms=([-0-9.]+)"),
    "audio_bias01": re.compile(r"\bAUDIO_PROGRESS_MFA\b.*?\bbias01=([-0-9.]+)"),
    "audio_corr": re.compile(r"\bAUDIO_PROGRESS_MFA\b.*?\bcorr=([-0-9.]+)"),
    "boundary_p": re.compile(r"\bBOUNDARY_FILTER\b.*?\bfiltered_p=([-0-9.]+)"),
    "boundary_r": re.compile(r"\bBOUNDARY_FILTER\b.*?\bfiltered_r=([-0-9.]+)"),
    "drift_abs_ms": re.compile(r"\bTIMING_PIVOT\b.*?\bdrift_abs_ms=([-0-9.]+)"),
    "drift_rate": re.compile(r"\bTIMING_PIVOT\b.*?\bdrift_rate=([-0-9.]+)"),
}

FIELDS = [
    "preroll_ms", "status", "seconds", "cases_total", "graded",
    "phoneme_center_ms", "phoneme_start_ms", "phoneme_end_ms", "word_start_ms",
    "intra_word_coverage", "direct_center_ms", "direct_gap_ms",
    "audio_rows", "audio_mfa_rows", "audio_mae01", "audio_mae_ms", "audio_bias01", "audio_corr",
    "drift_abs_ms", "drift_rate",
    "boundary_p", "boundary_r", "log",
]


def parse_metrics(text: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        row[key] = match.group(1) if match else ""
    try:
        row["direct_gap_ms"] = f"{float(row['phoneme_center_ms']) - float(row['direct_center_ms']):.3f}"
    except Exception:
        row["direct_gap_ms"] = ""
    return row


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")


def write_outputs(rows: list[dict[str, str]], out_dir: Path, partial: bool) -> None:
    suffix = ".partial" if partial else ""
    csv_path = out_dir / f"sweep_summary{suffix}.csv"
    md_path = out_dir / f"sweep_summary{suffix}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| preroll_ms | status | seconds | graded | phoneme_center_ms | direct_gap_ms | drift_abs_ms | drift_rate | log |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r.get('preroll_ms','')} | {r.get('status','')} | {r.get('seconds','')} | "
                f"{r.get('graded','')} | {r.get('phoneme_center_ms','')} | {r.get('direct_gap_ms','')} | "
                f"{r.get('drift_abs_ms','')} | {r.get('drift_rate','')} | {r.get('log','')} |\n"
            )


def load_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def should_echo(line: str, stream: bool) -> bool:
    if stream:
        return True
    return (
        line.startswith("[phase]")
        or line.startswith("CASES ")
        or line.startswith("GRADE ")
        or line.startswith("DIRECT_ALIGNER ")
        or line.startswith("AUDIO_PROGRESS_MFA ")
        or line.startswith("TIMING_PIVOT ")
        or line.startswith("BOUNDARY_FILTER ")
        or line.startswith("PHONE_CLASS_ERRORS ")
    )


def run_one(
    runner: Path,
    root: Path,
    preroll_ms: int,
    out_dir: Path,
    runner_args: list[str],
    stream: bool,
    flag_name: str,
) -> dict[str, str]:
    log_path = out_dir / f"preroll_{preroll_ms}.log"
    cmd = [str(runner), str(root), flag_name, str(preroll_ms), *runner_args]

    print(f"\n[SWEEP] START preroll={preroll_ms}ms", flush=True)
    print(f"[SWEEP] CMD {' '.join(cmd)}", flush=True)

    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        last_case_print = 0
        for line in proc.stdout:
            log.write(line)
            if should_echo(line, stream):
                print(line.rstrip(), flush=True)
            elif line.startswith("CASE_DIAG"):
                # Do not spam by default; emit an occasional heartbeat.
                last_case_print += 1
                if last_case_print == 1 or last_case_print % 25 == 0:
                    print(f"[SWEEP] preroll={preroll_ms}ms processed case_diag={last_case_print}", flush=True)
        rc = proc.wait()

    seconds = time.time() - start
    text = log_path.read_text(encoding="utf-8", errors="replace")
    row = {
        "preroll_ms": str(preroll_ms),
        "status": "ok" if rc == 0 else f"failed:{rc}",
        "seconds": f"{seconds:.1f}",
        "log": str(log_path),
        **parse_metrics(text),
    }

    if rc != 0:
        failed_path = unique_path(out_dir / f"preroll_{preroll_ms}_failed.log")
        log_path.replace(failed_path)
        row["log"] = str(failed_path)

    print(
        f"[SWEEP] DONE preroll={preroll_ms}ms status={row['status']} seconds={row['seconds']} "
        f"graded={row.get('graded','')} center={row.get('phoneme_center_ms','')} "
        f"drift_abs_ms={row.get('drift_abs_ms','')} drift_rate={row.get('drift_rate','')}",
        flush=True,
    )
    return row


def parse_prerolls(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--out-dir", default="runs/preroll_sweep", type=Path)
    parser.add_argument("--prerolls", default=",".join(str(x) for x in DEFAULT_PREROLLS))
    parser.add_argument("--flag", default="--preroll-ms", help="runner flag to set preroll; v11 fixed runner accepts --preroll-ms and --buffer-ms")
    parser.add_argument("--stream", action="store_true", help="stream all runner output")
    parser.add_argument("--resume", action="store_true", help="skip completed prerolls already in sweep_summary.csv")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prerolls = parse_prerolls(args.prerolls)
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]

    rows = load_existing(out_dir / "sweep_summary.csv") if args.resume else []
    completed = {
        int(r["preroll_ms"])
        for r in rows
        if r.get("status") == "ok" and r.get("preroll_ms", "").isdigit()
    }

    print(f"[SWEEP] runner={args.runner}", flush=True)
    print(f"[SWEEP] root={args.root}", flush=True)
    print(f"[SWEEP] out_dir={out_dir}", flush=True)
    print(f"[SWEEP] prerolls={prerolls}", flush=True)
    print(f"[SWEEP] flag={args.flag}", flush=True)
    print(f"[SWEEP] resume={args.resume} completed={sorted(completed)}", flush=True)

    for index, preroll_ms in enumerate(prerolls, start=1):
        if args.resume and preroll_ms in completed:
            print(f"[SWEEP] SKIP {index}/{len(prerolls)} preroll={preroll_ms}ms already completed", flush=True)
            continue

        print(f"[SWEEP] RUN {index}/{len(prerolls)} preroll={preroll_ms}ms", flush=True)
        row = run_one(args.runner, args.root, preroll_ms, out_dir, runner_args, args.stream, args.flag)
        rows = [r for r in rows if r.get("preroll_ms") != str(preroll_ms)]
        rows.append(row)
        rows.sort(key=lambda r: int(r.get("preroll_ms") or 0))
        write_outputs(rows, out_dir, partial=True)

    write_outputs(rows, out_dir, partial=False)
    print("\n[SWEEP] COMPLETE", flush=True)
    print(f"[SWEEP] wrote {out_dir / 'sweep_summary.csv'}", flush=True)
    print(f"[SWEEP] wrote {out_dir / 'sweep_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
