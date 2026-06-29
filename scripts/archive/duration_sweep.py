#!/usr/bin/env python3
"""Sweep island-duration estimator ideas against a liplab corpus run.

This is intentionally diagnostic-only. It does not change the lipsync runtime.
It consumes the normal liplab_runner per-case CSVs, reconstructs per text-island
features, compares variant duration estimates against observed audio island spans,
and writes ranked CSV/Markdown reports.

Typical use from the repo root:

  python scripts/duration_sweep.py \
      --inputs . \
      --runner build\\Release\\liplab_runner.exe \
      --baseline-run experiments/runs/duration_sweep_v1/baseline \
      --out experiments/runs/duration_sweep_v1

If --baseline-run already contains per_case/*/committed_events.csv, --runner is not
needed and the existing run is reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "i", "if", "in", "is", "it", "its", "let", "me", "my", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "would", "you", "your",
}
VOWELS = "aeiouy"


@dataclass
class IslandSample:
    case_id: str
    text_island_index: int
    transcript: str
    first_word_index: int
    last_word_index: int
    event_count: int
    dominant_count: int
    landmark_count: int
    function_word_count: int
    word_count: int
    syllable_count: int
    prosody_group_count: int
    comma_count: int
    semicolon_count: int
    colon_count: int
    dash_count: int
    ellipsis_count: int
    soft_punct_count: int
    baseline_duration_sec: float
    speech_material_sec: float
    punctuation_sec: float
    short_floor_sec: float
    observed_duration_sec: float

    @property
    def punct_weight(self) -> float:
        return (
            self.comma_count * 0.10
            + self.semicolon_count * 0.16
            + self.colon_count * 0.16
            + self.dash_count * 0.18
            + self.ellipsis_count * 0.24
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        return default if value == "" else int(float(value))
    except (TypeError, ValueError):
        return default


def words_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel
    if w.endswith("e") and groups > 1 and not w.endswith(("le", "ye")):
        groups -= 1
    return max(1, groups)


def punctuation_counts_between_words(text: str, words: list[tuple[str, int, int]], first_word: int, last_word: int) -> dict[str, int]:
    if not words or first_word < 0 or last_word < first_word:
        return {"comma": 0, "semicolon": 0, "colon": 0, "dash": 0, "ellipsis": 0}
    first_word = max(0, min(first_word, len(words) - 1))
    last_word = max(0, min(last_word, len(words) - 1))
    start = words[first_word][2]
    end = words[last_word][1]
    region = text[start:end] if end > start else ""
    ellipsis = region.count("...")
    region_no_ellipsis = region.replace("...", "")
    return {
        "comma": region_no_ellipsis.count(","),
        "semicolon": region_no_ellipsis.count(";"),
        "colon": region_no_ellipsis.count(":"),
        "dash": region_no_ellipsis.count("—") + region_no_ellipsis.count("–") + region_no_ellipsis.count("--"),
        "ellipsis": ellipsis,
    }


def group_rows_by_available_island(case_dir: Path, rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by_island: dict[int, list[dict[str, str]]] = {}

    # Newer diagnostic CSVs may write the text island directly on each event.
    for row in rows:
        text_island = i(row, "TextIslandIndex", -1)
        if text_island >= 0:
            by_island.setdefault(text_island, []).append(row)
    if by_island:
        return by_island

    # Current main-branch liplab CSVs do not include TextIslandIndex or
    # PlannerIsland* columns in committed_events.csv. Reconstruct samples from
    # speech_islands.csv by assigning committed playback centers to the measured
    # audio island windows. This keeps the sweep compatible with the branch as it
    # exists today and still tests estimator shapes against observed durations.
    speech_path = case_dir / "speech_islands.csv"
    if speech_path.exists():
        islands = read_csv(speech_path)
        for island_row in islands:
            island_index = i(island_row, "IslandIndex", -1)
            start = f(island_row, "AudioBufferStartSec", math.nan)
            end = f(island_row, "AudioBufferLastSpeechSec", math.nan)
            if not math.isfinite(end) or end <= start:
                end = f(island_row, "AudioBufferEndSec", math.nan)
            if island_index < 0 or not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            matched: list[dict[str, str]] = []
            for row in rows:
                center = f(row, "CommittedPlaybackCenterSec", math.nan)
                if math.isfinite(center) and start - 0.080 <= center <= end + 0.120:
                    augmented = dict(row)
                    augmented["_ObservedIslandStartSec"] = str(start)
                    augmented["_ObservedIslandEndSec"] = str(end)
                    matched.append(augmented)
            if matched:
                by_island[island_index] = matched
        if by_island:
            return by_island

    # Last-resort fallback: use PhraseIndex. This is less exact than hard
    # punctuation islands but avoids producing an empty report on older logs.
    for row in rows:
        phrase = i(row, "PhraseIndex", -1)
        if phrase >= 0:
            by_island.setdefault(phrase, []).append(row)
    return by_island


def infer_baseline_duration_sec(island_rows: list[dict[str, str]], observed: float) -> float:
    first = island_rows[0]
    baseline = f(first, "PlannerIslandPredictedDurationSec", 0.0)
    if baseline > 0.0:
        return baseline

    offsets = [f(r, "PlannerEventPredictedOffsetSec", math.nan) for r in island_rows]
    offsets = [x for x in offsets if math.isfinite(x)]
    if len(offsets) >= 2:
        return max(0.050, max(offsets) - min(offsets))

    text_centers = [f(r, "TextCenterSec", math.nan) for r in island_rows]
    text_centers = [x for x in text_centers if math.isfinite(x)]
    if len(text_centers) >= 2:
        # committed_events.csv only stores event centers in current main. Add a
        # small edge allowance so the inferred text-plan duration approximates
        # the invisible start/end envelope rather than only center-to-center span.
        return max(0.050, max(text_centers) - min(text_centers) + 0.110)

    return max(0.050, observed)


def load_case_sample(case_dir: Path, transcript_path: Path) -> list[IslandSample]:
    committed_path = case_dir / "committed_events.csv"
    if not committed_path.exists():
        return []
    rows = read_csv(committed_path)
    if not rows:
        return []
    transcript = transcript_path.read_text(encoding="utf-8-sig").strip()
    words = words_with_spans(transcript)
    by_island = group_rows_by_available_island(case_dir, rows)

    out: list[IslandSample] = []
    for text_island, island_rows in sorted(by_island.items()):
        first_word = min(i(r, "WordIndex", 10**9) for r in island_rows)
        last_word = max(i(r, "WordIndex", -1) for r in island_rows)
        word_slice = words[first_word:last_word + 1] if 0 <= first_word <= last_word < len(words) else []
        word_count = len(word_slice)
        syllables = sum(estimate_syllables(w[0]) for w in word_slice)
        function_count = sum(1 for w, _, _ in word_slice if w.lower() in FUNCTION_WORDS)
        punct = punctuation_counts_between_words(transcript, words, first_word, last_word)
        soft_punct = sum(punct.values())

        starts = [f(r, "IslandAudioStartSec", math.nan) for r in island_rows]
        ends = [f(r, "IslandAudioEndSec", math.nan) for r in island_rows]
        starts += [f(r, "_ObservedIslandStartSec", math.nan) for r in island_rows]
        ends += [f(r, "_ObservedIslandEndSec", math.nan) for r in island_rows]
        centers = [f(r, "CommittedPlaybackCenterSec", math.nan) for r in island_rows]
        render_starts = [f(r, "RenderStartSec", math.nan) for r in island_rows]
        render_ends = [f(r, "RenderEndSec", math.nan) for r in island_rows]
        starts = [x for x in starts if math.isfinite(x) and x >= 0.0]
        ends = [x for x in ends if math.isfinite(x) and x >= 0.0]
        centers = [x for x in centers if math.isfinite(x) and x >= 0.0]
        render_starts = [x for x in render_starts if math.isfinite(x) and x >= 0.0]
        render_ends = [x for x in render_ends if math.isfinite(x) and x >= 0.0]
        if starts and ends and max(ends) > min(starts):
            observed = max(ends) - min(starts)
        elif render_starts and render_ends and max(render_ends) > min(render_starts):
            observed = max(render_ends) - min(render_starts)
        elif len(centers) >= 2:
            observed = max(centers) - min(centers)
        else:
            observed = 0.0
        if observed <= 0.050:
            continue

        first = island_rows[0]
        groups = {
            i(r, "PlannerProsodyGroupIndex", -1)
            for r in island_rows
            if i(r, "PlannerProsodyGroupIndex", -1) >= 0
        }
        if not groups:
            groups = {
                i(r, "PhraseIndex", -1)
                for r in island_rows
                if i(r, "PhraseIndex", -1) >= 0
            }
        event_count = len(island_rows)
        dominant_count = sum(1 for r in island_rows if f(r, "Strength", 0.0) >= 0.78)
        landmark_count = sum(1 for r in island_rows if f(r, "AlignmentConfidence", 0.0) > 0.0 or f(r, "AppliedShiftMs", 0.0) != 0.0)
        baseline = infer_baseline_duration_sec(island_rows, observed)

        out.append(IslandSample(
            case_id=case_dir.name,
            text_island_index=text_island,
            transcript=transcript,
            first_word_index=first_word,
            last_word_index=last_word,
            event_count=event_count,
            dominant_count=dominant_count,
            landmark_count=landmark_count,
            function_word_count=function_count,
            word_count=word_count,
            syllable_count=syllables,
            prosody_group_count=max(1, len(groups)),
            comma_count=punct["comma"],
            semicolon_count=punct["semicolon"],
            colon_count=punct["colon"],
            dash_count=punct["dash"],
            ellipsis_count=punct["ellipsis"],
            soft_punct_count=soft_punct,
            baseline_duration_sec=baseline,
            speech_material_sec=f(first, "PlannerIslandSpeechMaterialSec", 0.0),
            punctuation_sec=f(first, "PlannerIslandPunctuationSec", 0.0),
            short_floor_sec=f(first, "PlannerIslandShortUtteranceFloorSec", 0.0),
            observed_duration_sec=observed,
        ))
    return out


def load_samples(inputs: Path, baseline_run: Path) -> list[IslandSample]:
    per_case = baseline_run / "per_case"
    samples: list[IslandSample] = []
    for case_dir in sorted(per_case.glob("case_*")):
        transcript_path = inputs / "transcripts" / f"{case_dir.name}.txt"
        if not transcript_path.exists():
            continue
        samples.extend(load_case_sample(case_dir, transcript_path))
    return samples



def resolve_inputs_root(repo: Path, raw_inputs: Path) -> Path:
    """Accept either the repo root or the actual inputs root.

    Supported layouts:
      <root>/transcripts + <root>/wav
      <repo>/inputs/transcripts + <repo>/inputs/wav
    """
    candidates: list[Path] = []
    if raw_inputs.is_absolute():
        candidates.append(raw_inputs)
    else:
        candidates.append((Path.cwd() / raw_inputs))
        candidates.append(repo / raw_inputs)

    expanded: list[Path] = []
    for c in candidates:
        c = c.resolve()
        expanded.append(c)
        expanded.append(c / "inputs")

    seen: set[Path] = set()
    for c in expanded:
        if c in seen:
            continue
        seen.add(c)
        if (c / "transcripts").is_dir() and (c / "wav").is_dir():
            return c

    tried = "\n".join(f"  {c}" for c in seen)
    raise SystemExit(
        "Could not find liplab corpus input folders. Expected either:\n"
        "  <inputs>/transcripts and <inputs>/wav\n"
        "  <repo>/inputs/transcripts and <repo>/inputs/wav\n"
        f"Checked:\n{tried}"
    )


def resolve_runner(repo: Path, explicit_runner: str | None) -> Path | None:
    """Resolve liplab_runner from an explicit path or common local build locations."""
    if explicit_runner:
        raw = Path(explicit_runner)
        candidates = [raw]
        if not raw.is_absolute():
            candidates.append(repo / raw)
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate
        raise SystemExit(
            "Runner not found: "
            f"{explicit_runner}\n"
            "Pass the real liplab_runner.exe path, or omit --runner after building in a common location."
        )

    candidates = [
        repo / "build_ninja" / "liplab_runner.exe",
        repo / "build_ninja" / "liplab_runner",
        repo / "build" / "tools" / "liplab_runner" / "Release" / "liplab_runner.exe",
        repo / "build" / "tools" / "liplab_runner" / "Debug" / "liplab_runner.exe",
        repo / "build" / "Release" / "liplab_runner.exe",
        repo / "build" / "Debug" / "liplab_runner.exe",
        repo / "build" / "liplab_runner.exe",
        repo / "build" / "liplab_runner",
        repo / "out" / "build" / "x64-release" / "tools" / "liplab_runner" / "liplab_runner.exe",
        repo / "out" / "build" / "x64-debug" / "tools" / "liplab_runner" / "liplab_runner.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Last-resort bounded search; useful when Visual Studio/CMake chooses a nested layout.
    for pattern in ("liplab_runner.exe", "liplab_runner"):
        for candidate in repo.glob(f"**/{pattern}"):
            if any(part in {".git", "experiments", "runs"} for part in candidate.parts):
                continue
            if candidate.is_file():
                return candidate.resolve()

    return None

def run_baseline_if_needed(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    per_case = Path(args.baseline_run) / "per_case"
    if any(per_case.glob("case_*/committed_events.csv")):
        return

    runner = resolve_runner(repo, args.runner)
    if runner is None:
        raise SystemExit(
            f"No baseline per-case CSVs found under {per_case}.\n"
            "Could not auto-detect liplab_runner.exe. Build liplab first or pass --runner with the real path.\n"
            "Useful check from repo root: where /r . liplab_runner.exe"
        )

    cmd = [
        str(runner),
        "--inputs", str(args.resolved_inputs),
        "--config", str(args.config),
        "--out", str(args.baseline_run),
    ]
    if args.chunk_ms is not None:
        cmd += ["--chunk_ms", str(args.chunk_ms)]
    if args.preroll_ms is not None:
        cmd += ["--preroll_ms", str(args.preroll_ms)]
    if args.fps is not None:
        cmd += ["--fps", str(args.fps)]
    print("Running baseline:", " ".join(cmd))
    subprocess.run(cmd, check=True)


@dataclass
class VariantResult:
    variant: str
    parameter: str
    sample_count: int
    mean_predicted_sec: float
    mean_observed_sec: float
    mean_ratio: float
    median_ratio: float
    mean_abs_error_ms: float
    median_abs_error_ms: float
    p90_abs_error_ms: float
    mean_signed_error_ms: float
    under_rate: float
    severe_under_rate: float
    over_rate: float
    severe_over_rate: float
    mean_under_error_ms: float
    objective: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil(len(values) * p) - 1))
    return values[idx]


def evaluate(samples: list[IslandSample], name: str, parameter: str, fn: Callable[[IslandSample], float]) -> VariantResult:
    preds = [max(0.050, fn(s)) for s in samples]
    obs = [s.observed_duration_sec for s in samples]
    errors = [p - o for p, o in zip(preds, obs)]
    abs_ms = [abs(e) * 1000.0 for e in errors]
    ratios = [p / max(0.001, o) for p, o in zip(preds, obs)]
    under_errors_ms = [max(0.0, -e) * 1000.0 for e in errors]
    mean_abs = statistics.fmean(abs_ms)
    under_rate = statistics.fmean(1.0 if r < 0.95 else 0.0 for r in ratios)
    severe_under = statistics.fmean(1.0 if r < 0.85 else 0.0 for r in ratios)
    severe_over = statistics.fmean(1.0 if r > 1.25 else 0.0 for r in ratios)
    # Bias the objective toward our perceptual preference: underprediction is worse than mild lateness.
    objective = (
        mean_abs
        + 0.90 * statistics.fmean(under_errors_ms)
        + 220.0 * severe_under
        + 80.0 * severe_over
    )
    return VariantResult(
        variant=name,
        parameter=parameter,
        sample_count=len(samples),
        mean_predicted_sec=statistics.fmean(preds),
        mean_observed_sec=statistics.fmean(obs),
        mean_ratio=statistics.fmean(ratios),
        median_ratio=statistics.median(ratios),
        mean_abs_error_ms=mean_abs,
        median_abs_error_ms=statistics.median(abs_ms),
        p90_abs_error_ms=percentile(abs_ms, 0.90),
        mean_signed_error_ms=statistics.fmean(errors) * 1000.0,
        under_rate=under_rate,
        severe_under_rate=severe_under,
        over_rate=statistics.fmean(1.0 if r > 1.05 else 0.0 for r in ratios),
        severe_over_rate=severe_over,
        mean_under_error_ms=statistics.fmean(under_errors_ms),
        objective=objective,
    )


def gaussian_solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-9:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_linear(samples: list[IslandSample]) -> list[float]:
    def x(s: IslandSample) -> list[float]:
        return [1.0, s.syllable_count, s.event_count, s.prosody_group_count, s.punct_weight, s.function_word_count]
    dim = 6
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    for s in samples:
        xs = x(s)
        y = s.observed_duration_sec
        for r in range(dim):
            xty[r] += xs[r] * y
            for c in range(dim):
                xtx[r][c] += xs[r] * xs[c]
    for d in range(dim):
        xtx[d][d] += 1e-4
    beta = gaussian_solve(xtx, xty)
    if beta is None:
        return [0.0, 0.16, 0.0, 0.0, 1.0, 0.0]
    return beta


def build_variants(samples: list[IslandSample]) -> list[tuple[str, str, Callable[[IslandSample], float]]]:
    variants: list[tuple[str, str, Callable[[IslandSample], float]]] = []
    variants.append(("baseline", "current", lambda s: s.baseline_duration_sec))

    for scalar in [1.10, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60]:
        variants.append(("global_scalar", f"x{scalar:.2f}", lambda s, scalar=scalar: s.baseline_duration_sec * scalar))

    for floor in [0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120]:
        variants.append(("visible_event_floor", f"{floor:.3f}s_per_event", lambda s, floor=floor: max(s.baseline_duration_sec, s.event_count * floor)))

    for floor in [0.110, 0.130, 0.145, 0.160, 0.175, 0.190, 0.205, 0.220]:
        variants.append(("syllable_floor", f"{floor:.3f}s_per_syllable", lambda s, floor=floor: max(s.baseline_duration_sec, s.syllable_count * floor)))

    for comma in [0.06, 0.10, 0.14, 0.18]:
        variants.append((
            "punctuation_mass",
            f"comma{comma:.2f}_semi{comma*1.6:.2f}_ellipsis{comma*2.4:.2f}",
            lambda s, comma=comma: s.baseline_duration_sec
            + s.comma_count * comma
            + s.semicolon_count * comma * 1.6
            + s.colon_count * comma * 1.6
            + s.dash_count * comma * 1.8
            + s.ellipsis_count * comma * 2.4,
        ))

    for short_scalar, mid_scalar, long_scalar in [(1.10, 1.30, 1.45), (1.15, 1.35, 1.50), (1.20, 1.40, 1.55)]:
        def length_variant(s: IslandSample, a=short_scalar, b=mid_scalar, c=long_scalar) -> float:
            if s.baseline_duration_sec < 1.0:
                return s.baseline_duration_sec * a
            if s.baseline_duration_sec < 3.0:
                return s.baseline_duration_sec * b
            return s.baseline_duration_sec * c
        variants.append(("length_sensitive_scalar", f"short{short_scalar:.2f}_mid{mid_scalar:.2f}_long{long_scalar:.2f}", length_variant))

    for floor in [0.25, 0.35, 0.45, 0.55, 0.65]:
        variants.append(("prosody_group_floor", f"{floor:.2f}s_per_group", lambda s, floor=floor: max(s.baseline_duration_sec, s.prosody_group_count * floor)))

    for bonus in [0.015, 0.025, 0.035, 0.050]:
        variants.append(("function_word_limit", f"+{bonus:.3f}s_per_function_word", lambda s, bonus=bonus: s.baseline_duration_sec + s.function_word_count * bonus))

    for scalar in [1.20, 1.30, 1.35, 1.40]:
        for event_floor in [0.070, 0.085, 0.100]:
            for syll_floor in [0.145, 0.165, 0.185]:
                variants.append((
                    "hybrid_scalar_floors",
                    f"x{scalar:.2f}_event{event_floor:.3f}_syll{syll_floor:.3f}",
                    lambda s, scalar=scalar, event_floor=event_floor, syll_floor=syll_floor:
                        max(s.baseline_duration_sec * scalar, s.event_count * event_floor, s.syllable_count * syll_floor),
                ))

    beta = fit_linear(samples)
    variants.append((
        "linear_model_fit_all",
        "intercept,syll,event,group,punct,function=" + ";".join(f"{v:.6f}" for v in beta),
        lambda s, beta=beta: max(0.050, beta[0] + beta[1] * s.syllable_count + beta[2] * s.event_count + beta[3] * s.prosody_group_count + beta[4] * s.punct_weight + beta[5] * s.function_word_count),
    ))
    return variants


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, results: list[VariantResult], samples: list[IslandSample]) -> None:
    top = results[:20]
    baseline = next((r for r in results if r.variant == "baseline"), None)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Island Duration Sweep v1\n\n")
        f.write(f"Samples: {len(samples)} text islands\n\n")
        if baseline:
            f.write("## Baseline\n\n")
            f.write(f"- mean ratio: {baseline.mean_ratio:.3f}\n")
            f.write(f"- mean abs error: {baseline.mean_abs_error_ms:.1f} ms\n")
            f.write(f"- severe underprediction rate: {baseline.severe_under_rate:.1%}\n")
            f.write(f"- objective: {baseline.objective:.1f}\n\n")
        f.write("## Top Variants\n\n")
        f.write("| rank | variant | parameter | objective | mean ratio | MAE ms | p90 ms | severe under | severe over |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|\n")
        for rank, r in enumerate(top, 1):
            f.write(
                f"| {rank} | {r.variant} | `{r.parameter}` | {r.objective:.1f} | {r.mean_ratio:.3f} | "
                f"{r.mean_abs_error_ms:.1f} | {r.p90_abs_error_ms:.1f} | {r.severe_under_rate:.1%} | {r.severe_over_rate:.1%} |\n"
            )
        f.write("\n## Notes\n\n")
        f.write("The objective intentionally penalizes underprediction harder than overprediction. ")
        f.write("Treat this as a duration-estimator ranking, not a final perceptual ranking; the next step is to patch the top one or two into the runtime and run the full liplab scoring pass.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate island duration estimator variants from a liplab corpus run.")
    parser.add_argument("--inputs", type=Path, default=Path("."), help="Repo root or input root. Supports ./transcripts+./wav or ./inputs/transcripts+./inputs/wav.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="Repo root used for runner auto-detection.")
    parser.add_argument("--baseline-run", type=Path, default=Path("experiments/runs/duration_sweep_v1/baseline"), help="Existing or generated liplab_runner output directory.")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/duration_sweep_v1"), help="Sweep report output directory.")
    parser.add_argument("--runner", type=Path, default=None, help="Optional path to liplab_runner.exe; used only if --baseline-run is missing per-case CSVs.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v26.json"))
    parser.add_argument("--chunk_ms", type=int, default=None)
    parser.add_argument("--preroll_ms", type=float, default=None)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    args.repo = Path(args.repo).resolve()
    args.resolved_inputs = resolve_inputs_root(args.repo, Path(args.inputs))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Using inputs: {args.resolved_inputs}")
    run_baseline_if_needed(args)
    samples = load_samples(args.resolved_inputs, args.baseline_run)
    if not samples:
        raise SystemExit("No island samples found. Ensure baseline run has per_case/*/committed_events.csv and inputs has transcripts/*.txt.")

    variants = build_variants(samples)
    results = [evaluate(samples, name, param, fn) for name, param, fn in variants]
    results.sort(key=lambda r: r.objective)

    write_csv(args.out / "duration_sweep_ranked.csv", [asdict(r) for r in results])
    write_markdown(args.out / "duration_sweep_ranked.md", results, samples)
    write_csv(args.out / "duration_sweep_island_samples.csv", [asdict(s) for s in samples])
    with (args.out / "duration_sweep_best_variant.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(results[0]), f, indent=2)

    print(f"Wrote {args.out / 'duration_sweep_ranked.md'}")
    print(f"Best: {results[0].variant} {results[0].parameter} objective={results[0].objective:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
