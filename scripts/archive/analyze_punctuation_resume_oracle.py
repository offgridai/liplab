#!/usr/bin/env python3
"""
E02 offline audio-pause / punctuation-correlation oracle.

This is a research tool only. It does not change runtime behavior.

Given an existing LipLab run plus the matching transcript/WAV corpus, it asks a
narrow question for every punctuation boundary in the transcript:

    Near the text-predicted boundary, is there an acoustic valley followed by a
    plausible speech resume, and how far is the current viseme phrase start from
    that resume?

The goal is to stop guessing about comma/period timing.  This script sweeps a
small family of relative energy detectors over real WAVs and writes CSVs that
show whether intra-line pause/resume is detectable, which punctuation classes are
reliable, and where the current timeline starts the next phrase too early/late.

E0/E1 originally searched from text punctuation outward.  E02 inverts the
question: first build an audio-only pause inventory, then correlate those pauses
with transcript punctuation, and finally run the same offline downstream-shift
oracle using matched audio resumes.  It is still diagnostic-only; no runtime
files or behavior are changed.

Expected inputs:
  <run>/per_case/case_XXXXXX/runtime_commit_events.csv
  <run>/per_case/case_XXXXXX/timing_diagnostics.csv      (optional but useful)
  <inputs>/transcripts/case_XXXXXX.txt
  <inputs>/wav/case_XXXXXX.wav

Outputs:
  <out>/audio_pause_inventory.csv                       (E02 audio-only pause/resume candidates)
  <out>/punctuation_resume_oracle_boundaries.csv       (E02 punctuation/pause correlation evidence)
  <out>/punctuation_resume_oracle_policy_summary.csv   (legacy text-centered detector reliability/error)
  <out>/punctuation_resume_oracle_by_case.csv
  <out>/punctuation_resume_oracle_event_errors.csv     (E1 per-event before/after oracle shift)
  <out>/punctuation_resume_oracle_e1_summary.csv       (E1 best-case timing impact)
  <out>/punctuation_resume_oracle_e1_by_case.csv
  <out>/punctuation_clock_pause_events.csv              (E04 boundary-level clock pause simulation)
  <out>/punctuation_clock_pause_event_errors.csv        (E04 per-event hold/bidirectional timing impact)
  <out>/punctuation_clock_pause_oracle_summary.csv      (E04 corpus-level timing impact)
  <out>/punctuation_clock_pause_oracle_by_case.csv

Usage:
  python scripts/analyze_punctuation_resume_oracle.py \
      --run experiments/runs/b09 \
      --inputs inputs
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - script dependency guard
    raise SystemExit("This script requires numpy, which is already used by the LipLab tooling environment.") from exc


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
PUNCT_RE = re.compile(r"[,:;.!?…]+|--+|—|-")
HARD_PUNCT = {".", "?", "!"}


@dataclass(frozen=True)
class WordToken:
    text: str
    start: int
    end: int
    word_index: int


@dataclass(frozen=True)
class Boundary:
    boundary_index: int
    prev_word_index: int
    next_word_index: int
    prev_word: str
    next_word: str
    punctuation: str
    punctuation_kind: str
    char_start: int
    char_end: int
    hard_break: bool


@dataclass(frozen=True)
class WordTiming:
    word_index: int
    word: str
    first_center_sec: float
    last_center_sec: float
    first_raw_center_sec: float
    last_raw_center_sec: float
    phrase_index: int
    event_count: int


@dataclass(frozen=True)
class Policy:
    name: str
    valley_frac_of_high: float
    min_valley_ms: float
    resume_frac_of_range: float
    consecutive_resume_frames: int = 2


POLICIES = [
    Policy("loose_30pct_40ms", 0.30, 40.0, 0.28, 2),
    Policy("loose_35pct_50ms", 0.35, 50.0, 0.32, 2),
    Policy("balanced_40pct_60ms", 0.40, 60.0, 0.38, 2),
    Policy("balanced_45pct_80ms", 0.45, 80.0, 0.42, 2),
    Policy("strict_50pct_100ms", 0.50, 100.0, 0.48, 3),
]


@dataclass
class Detection:
    detected: bool = False
    pause_start_sec: float = 0.0
    valley_sec: float = 0.0
    resume_sec: float = 0.0
    valley_duration_ms: float = 0.0
    local_high: float = 0.0
    local_low: float = 0.0
    valley_value: float = 0.0
    resume_value: float = 0.0
    drop_db: float = 0.0
    rise_db: float = 0.0
    confidence: float = 0.0
    reject_reason: str = "not_run"

@dataclass(frozen=True)
class PauseCandidate:
    case_pause_index: int
    pause_start_sec: float
    valley_sec: float
    pause_end_sec: float
    resume_sec: float
    duration_ms: float
    local_high: float
    local_low: float
    valley_value: float
    resume_value: float
    drop_db: float
    rise_db: float
    confidence: float


@dataclass(frozen=True)
class Correlation:
    matched: bool
    classification: str
    pause: PauseCandidate | None
    distance_ms: float
    abs_distance_ms: float
    wide_candidate_count: int
    near_candidate_count: int
    ambiguous_candidate_count: int



def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        s = str(row.get(key, "")).strip()
        return float(s) if s else default
    except Exception:
        return default


def as_int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        s = str(row.get(key, "")).strip()
        return int(float(s)) if s else default
    except Exception:
        return default


def percentile(values: np.ndarray, p: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.percentile(values, p))


def db_ratio(a: float, b: float) -> float:
    return 20.0 * math.log10(max(a, 1e-7) / max(b, 1e-7))


def parse_words(text: str) -> list[WordToken]:
    words: list[WordToken] = []
    for idx, m in enumerate(WORD_RE.finditer(text)):
        words.append(WordToken(m.group(0), m.start(), m.end(), idx))
    return words


def punctuation_kind(punct: str) -> str:
    if any(ch in ".?!" for ch in punct):
        return "sentence"
    if "," in punct:
        return "comma"
    if ";" in punct:
        return "semicolon"
    if ":" in punct:
        return "colon"
    if "…" in punct or "..." in punct:
        return "ellipsis"
    if "-" in punct or "—" in punct:
        return "dash"
    return "other"


def parse_punctuation_boundaries(text: str) -> list[Boundary]:
    words = parse_words(text)
    if len(words) < 2:
        return []
    out: list[Boundary] = []
    boundary_index = 0
    for i in range(len(words) - 1):
        prev_w = words[i]
        next_w = words[i + 1]
        between = text[prev_w.end:next_w.start]
        punct = "".join(m.group(0) for m in PUNCT_RE.finditer(between)).strip()
        if not punct:
            continue
        # Ignore apostrophe/word punctuation already absorbed into tokens; this is boundary punctuation only.
        kind = punctuation_kind(punct)
        hard = kind == "sentence"
        rel_start = between.find(punct[0]) if punct else -1
        char_start = prev_w.end + max(0, rel_start)
        out.append(Boundary(
            boundary_index=boundary_index,
            prev_word_index=prev_w.word_index,
            next_word_index=next_w.word_index,
            prev_word=prev_w.text,
            next_word=next_w.text,
            punctuation=punct,
            punctuation_kind=kind,
            char_start=char_start,
            char_end=char_start + len(punct),
            hard_break=hard,
        ))
        boundary_index += 1
    return out


def collect_word_timings(case_dir: Path) -> dict[int, WordTiming]:
    commit_rows = read_csv_dicts(case_dir / "runtime_commit_events.csv")
    timing_rows = read_csv_dicts(case_dir / "timing_diagnostics.csv")

    raw_by_word: dict[int, tuple[float, float]] = {}
    for row in timing_rows:
        wi = as_int(row, "WordIndex", -1)
        if wi < 0:
            continue
        c = as_float(row, "RawTextCenterSec", 0.0)
        if c <= 0.0:
            continue
        if wi not in raw_by_word:
            raw_by_word[wi] = (c, c)
        else:
            raw_by_word[wi] = (min(raw_by_word[wi][0], c), max(raw_by_word[wi][1], c))

    accum: dict[int, dict[str, object]] = {}
    for row in commit_rows:
        word = row.get("SourceWord", "")
        # runtime_commit_events.csv historically did not include WordIndex.  If it
        # is missing, fall back to timing_diagnostics below.
        wi = as_int(row, "WordIndex", -1)
        if wi < 0:
            wi = as_int(row, "SourceWordIndex", -1)
        if wi < 0:
            continue
        center = as_float(row, "FinalRenderCenterSec", as_float(row, "FinalRenderCenterSeconds", 0.0))
        if center <= 0.0:
            continue
        phrase = as_int(row, "PhraseIndex", -1)
        a = accum.setdefault(wi, {
            "word": word,
            "first": center,
            "last": center,
            "phrase": phrase,
            "count": 0,
        })
        a["word"] = a["word"] or word
        a["first"] = min(float(a["first"]), center)
        a["last"] = max(float(a["last"]), center)
        if phrase >= 0:
            a["phrase"] = phrase
        a["count"] = int(a["count"]) + 1

    # If commit rows lack word indices, timing_diagnostics has the necessary word mapping.
    if not accum:
        for row in timing_rows:
            wi = as_int(row, "WordIndex", -1)
            if wi < 0:
                continue
            center = as_float(row, "StreamingCenterSec", as_float(row, "RawTextCenterSec", 0.0))
            if center <= 0.0:
                continue
            phrase = as_int(row, "PhraseIndex", -1)
            word = row.get("SourceWord", "")
            a = accum.setdefault(wi, {
                "word": word,
                "first": center,
                "last": center,
                "phrase": phrase,
                "count": 0,
            })
            a["word"] = a["word"] or word
            a["first"] = min(float(a["first"]), center)
            a["last"] = max(float(a["last"]), center)
            if phrase >= 0:
                a["phrase"] = phrase
            a["count"] = int(a["count"]) + 1

    out: dict[int, WordTiming] = {}
    for wi, a in accum.items():
        raw = raw_by_word.get(wi, (float(a["first"]), float(a["last"])))
        out[wi] = WordTiming(
            word_index=wi,
            word=str(a["word"]),
            first_center_sec=float(a["first"]),
            last_center_sec=float(a["last"]),
            first_raw_center_sec=raw[0],
            last_raw_center_sec=raw[1],
            phrase_index=int(a["phrase"]),
            event_count=int(a["count"]),
        )
    return out



@dataclass(frozen=True)
class OracleAnchor:
    boundary_index: int
    prev_word_index: int
    next_word_index: int
    punctuation_kind: str
    planned_resume_sec: float
    detected_resume_sec: float
    raw_shift_ms: float
    applied_shift_ms: float
    confidence: float
    policy: str


@dataclass(frozen=True)
class ClockPauseAnchor:
    boundary_index: int
    prev_word_index: int
    next_word_index: int
    punctuation_kind: str
    planned_pause_start_sec: float
    planned_resume_sec: float
    audio_pause_start_sec: float
    audio_resume_sec: float
    raw_clock_delta_ms: float
    applied_clock_delta_ms: float
    confidence: float
    mode: str
    accept_reason: str


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def collect_event_timings(case_dir: Path) -> list[dict[str, object]]:
    rows = read_csv_dicts(case_dir / "timing_diagnostics.csv")
    out: list[dict[str, object]] = []
    for row in rows:
        event_index = as_int(row, "EventIndex", -1)
        word_index = as_int(row, "WordIndex", -1)
        streaming = as_float(row, "StreamingCenterSec", 0.0)
        evidence = as_float(row, "EvidenceCenterSec", 0.0)
        confidence = as_float(row, "EvidenceConfidence", 0.0)
        fallback = as_int(row, "EvidenceFallback", 0)
        if event_index < 0 or word_index < 0 or streaming <= 0.0 or evidence <= 0.0:
            continue
        out.append({
            "event_index": event_index,
            "word_index": word_index,
            "pose_id": row.get("PoseID", ""),
            "source_word": row.get("SourceWord", ""),
            "streaming_center_sec": streaming,
            "evidence_center_sec": evidence,
            "evidence_confidence": confidence,
            "evidence_fallback": fallback,
            "streaming_abs_error_ms": abs(evidence - streaming) * 1000.0,
        })
    out.sort(key=lambda r: int(r["event_index"]))
    return out


def apply_e1_oracle_to_events(
    events: list[dict[str, object]],
    anchors: list[OracleAnchor],
    min_event_gap_ms: float,
) -> list[dict[str, object]]:
    """Apply accepted punctuation resumes as downstream-only oracle shifts.

    This deliberately mirrors the proposed runtime shape without becoming a
    runtime policy: punctuation anchors are processed in text order; each anchor
    shifts the next word and everything after it by the delta needed to put that
    next word on the detected acoustic resume.  Earlier events are never moved.
    A final monotonic repair prevents the oracle from inventing impossible event
    orderings.
    """
    if not events:
        return []
    anchors = sorted(anchors, key=lambda a: (a.next_word_index, a.boundary_index))
    cumulative_shift_sec = 0.0
    anchor_cursor = 0
    min_gap_sec = max(0.0, min_event_gap_ms / 1000.0)
    out: list[dict[str, object]] = []
    previous_center_sec = -1.0

    for event in events:
        word_index = int(event["word_index"])
        while anchor_cursor < len(anchors) and anchors[anchor_cursor].next_word_index <= word_index:
            cumulative_shift_sec += anchors[anchor_cursor].applied_shift_ms / 1000.0
            anchor_cursor += 1

        original_center_sec = float(event["streaming_center_sec"])
        adjusted_center_sec = original_center_sec + cumulative_shift_sec
        order_repair_ms = 0.0
        if previous_center_sec >= 0.0 and adjusted_center_sec < previous_center_sec + min_gap_sec:
            repaired = previous_center_sec + min_gap_sec
            order_repair_ms = (repaired - adjusted_center_sec) * 1000.0
            adjusted_center_sec = repaired
        previous_center_sec = adjusted_center_sec

        evidence_center_sec = float(event["evidence_center_sec"])
        before_abs_ms = float(event["streaming_abs_error_ms"])
        after_abs_ms = abs(evidence_center_sec - adjusted_center_sec) * 1000.0
        row = dict(event)
        row.update({
            "oracle_center_sec": adjusted_center_sec,
            "oracle_cumulative_shift_ms": cumulative_shift_sec * 1000.0,
            "oracle_order_repair_ms": order_repair_ms,
            "oracle_abs_error_ms": after_abs_ms,
            "oracle_delta_error_ms": after_abs_ms - before_abs_ms,
            "oracle_improved": 1 if after_abs_ms + 1e-6 < before_abs_ms else 0,
            "oracle_worsened": 1 if after_abs_ms > before_abs_ms + 1e-6 else 0,
        })
        out.append(row)
    return out


def apply_clock_pause_to_events(
    events: list[dict[str, object]],
    anchors: list[ClockPauseAnchor],
    min_event_gap_ms: float,
) -> list[dict[str, object]]:
    """Simulate a punctuation-gated playback clock pause/resume.

    This is E04's offline model of the proposed runtime behavior.  The text
    plan remains monotonic and unchanged.  When playback reaches a punctuation
    resume point whose audio pause/resume has been matched, the virtual playback
    clock inserts the accepted delta before consuming the next post-punctuation
    word.  Earlier events are never moved.

    hold_only anchors only add positive wait time.  bidirectional anchors may
    also remove overestimated planned pause time, which is useful as an oracle
    but riskier for runtime.
    """
    if not events:
        return []
    anchors = sorted(anchors, key=lambda a: (a.next_word_index, a.boundary_index))
    cumulative_clock_delta_sec = 0.0
    anchor_cursor = 0
    min_gap_sec = max(0.0, min_event_gap_ms / 1000.0)
    out: list[dict[str, object]] = []
    previous_center_sec = -1.0

    for event in events:
        word_index = int(event["word_index"])
        while anchor_cursor < len(anchors) and anchors[anchor_cursor].next_word_index <= word_index:
            cumulative_clock_delta_sec += anchors[anchor_cursor].applied_clock_delta_ms / 1000.0
            anchor_cursor += 1

        original_center_sec = float(event["streaming_center_sec"])
        adjusted_center_sec = original_center_sec + cumulative_clock_delta_sec
        order_repair_ms = 0.0
        if previous_center_sec >= 0.0 and adjusted_center_sec < previous_center_sec + min_gap_sec:
            repaired = previous_center_sec + min_gap_sec
            order_repair_ms = (repaired - adjusted_center_sec) * 1000.0
            adjusted_center_sec = repaired
        previous_center_sec = adjusted_center_sec

        evidence_center_sec = float(event["evidence_center_sec"])
        before_abs_ms = float(event["streaming_abs_error_ms"])
        after_abs_ms = abs(evidence_center_sec - adjusted_center_sec) * 1000.0
        row = dict(event)
        row.update({
            "clock_center_sec": adjusted_center_sec,
            "clock_cumulative_delta_ms": cumulative_clock_delta_sec * 1000.0,
            "clock_order_repair_ms": order_repair_ms,
            "clock_abs_error_ms": after_abs_ms,
            "clock_delta_error_ms": after_abs_ms - before_abs_ms,
            "clock_improved": 1 if after_abs_ms + 1e-6 < before_abs_ms else 0,
            "clock_worsened": 1 if after_abs_ms > before_abs_ms + 1e-6 else 0,
        })
        out.append(row)
    return out


def summarize_clock_event_error_rows(rows: list[dict[str, object]], label: str, mode: str) -> dict[str, object]:
    if not rows:
        return {
            "Mode": mode,
            "Slice": label,
            "EventCount": 0,
            "MeanBeforeAbsErrorMs": "0.000",
            "MeanAfterAbsErrorMs": "0.000",
            "MeanImprovementMs": "0.000",
            "MedianBeforeAbsErrorMs": "0.000",
            "MedianAfterAbsErrorMs": "0.000",
            "P90BeforeAbsErrorMs": "0.000",
            "P90AfterAbsErrorMs": "0.000",
            "ImprovedCount": 0,
            "WorsenedCount": 0,
            "ImprovedRate": "0.000000",
            "WorsenedRate": "0.000000",
        }
    before = [float(r["streaming_abs_error_ms"]) for r in rows]
    after = [float(r["clock_abs_error_ms"]) for r in rows]
    before_mean, before_med, before_p90 = summarize_numeric(before)
    after_mean, after_med, after_p90 = summarize_numeric(after)
    improved = sum(1 for r in rows if int(r.get("clock_improved", 0)) == 1)
    worsened = sum(1 for r in rows if int(r.get("clock_worsened", 0)) == 1)
    return {
        "Mode": mode,
        "Slice": label,
        "EventCount": len(rows),
        "MeanBeforeAbsErrorMs": f"{before_mean:.3f}",
        "MeanAfterAbsErrorMs": f"{after_mean:.3f}",
        "MeanImprovementMs": f"{before_mean - after_mean:.3f}",
        "MedianBeforeAbsErrorMs": f"{before_med:.3f}",
        "MedianAfterAbsErrorMs": f"{after_med:.3f}",
        "P90BeforeAbsErrorMs": f"{before_p90:.3f}",
        "P90AfterAbsErrorMs": f"{after_p90:.3f}",
        "ImprovedCount": improved,
        "WorsenedCount": worsened,
        "ImprovedRate": f"{improved / max(1, len(rows)):.6f}",
        "WorsenedRate": f"{worsened / max(1, len(rows)):.6f}",
    }


def summarize_event_error_rows(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    if not rows:
        return {
            "Slice": label,
            "EventCount": 0,
            "MeanBeforeAbsErrorMs": "0.000",
            "MeanAfterAbsErrorMs": "0.000",
            "MeanImprovementMs": "0.000",
            "MedianBeforeAbsErrorMs": "0.000",
            "MedianAfterAbsErrorMs": "0.000",
            "P90BeforeAbsErrorMs": "0.000",
            "P90AfterAbsErrorMs": "0.000",
            "ImprovedCount": 0,
            "WorsenedCount": 0,
            "ImprovedRate": "0.000000",
            "WorsenedRate": "0.000000",
        }
    before = [float(r["streaming_abs_error_ms"]) for r in rows]
    after = [float(r["oracle_abs_error_ms"]) for r in rows]
    before_mean, before_med, before_p90 = summarize_numeric(before)
    after_mean, after_med, after_p90 = summarize_numeric(after)
    improved = sum(1 for r in rows if int(r.get("oracle_improved", 0)) == 1)
    worsened = sum(1 for r in rows if int(r.get("oracle_worsened", 0)) == 1)
    return {
        "Slice": label,
        "EventCount": len(rows),
        "MeanBeforeAbsErrorMs": f"{before_mean:.3f}",
        "MeanAfterAbsErrorMs": f"{after_mean:.3f}",
        "MeanImprovementMs": f"{before_mean - after_mean:.3f}",
        "MedianBeforeAbsErrorMs": f"{before_med:.3f}",
        "MedianAfterAbsErrorMs": f"{after_med:.3f}",
        "P90BeforeAbsErrorMs": f"{before_p90:.3f}",
        "P90AfterAbsErrorMs": f"{after_p90:.3f}",
        "ImprovedCount": improved,
        "WorsenedCount": worsened,
        "ImprovedRate": f"{improved / max(1, len(rows)):.6f}",
        "WorsenedRate": f"{worsened / max(1, len(rows)):.6f}",
    }

def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported for now: {path}")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sample_rate, data


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x
    kernel = np.ones(n, dtype=np.float32) / float(n)
    return np.convolve(x, kernel, mode="same")


def audio_features(samples: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float, smooth_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if samples.size < frame_len:
        padded = np.pad(samples, (0, frame_len - samples.size))
    else:
        padded = samples
    starts = np.arange(0, max(1, padded.size - frame_len + 1), hop_len, dtype=np.int64)
    rms = np.empty(starts.size, dtype=np.float32)
    for idx, start in enumerate(starts):
        frame = padded[start:start + frame_len]
        rms[idx] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    smooth_n = max(1, int(round(smooth_ms / hop_ms)))
    rms = moving_average(rms, smooth_n)
    times = (starts.astype(np.float32) + frame_len * 0.5) / float(sample_rate)
    return times, rms


def find_consecutive_true(mask: np.ndarray, start_idx: int, count: int) -> int | None:
    if count <= 1:
        idxs = np.flatnonzero(mask[start_idx:])
        return int(start_idx + idxs[0]) if idxs.size else None
    run = 0
    for idx in range(start_idx, mask.size):
        if bool(mask[idx]):
            run += 1
            if run >= count:
                return idx - count + 1
        else:
            run = 0
    return None


def detect_resume(
    times: np.ndarray,
    rms: np.ndarray,
    expected_sec: float,
    policy: Policy,
    search_before_sec: float,
    search_after_sec: float,
) -> Detection:
    lo_t = max(0.0, expected_sec - search_before_sec)
    hi_t = min(float(times[-1]) if times.size else 0.0, expected_sec + search_after_sec)
    idx = np.flatnonzero((times >= lo_t) & (times <= hi_t))
    if idx.size < 6:
        return Detection(reject_reason="too_few_frames")

    window = rms[idx]
    local_high = percentile(window, 82.0)
    local_low = percentile(window, 12.0)
    local_range = max(local_high - local_low, 1e-6)
    if local_high <= 1e-5:
        return Detection(local_high=local_high, local_low=local_low, reject_reason="silent_window")

    # A valley must be low relative to local speech in this punctuation-specific window.
    valley_threshold = max(local_low + 0.12 * local_range, local_high * policy.valley_frac_of_high)
    valley_mask = window <= valley_threshold
    valley_candidates = np.flatnonzero(valley_mask)
    if valley_candidates.size == 0:
        return Detection(local_high=local_high, local_low=local_low, reject_reason="no_valley")

    # Prefer valleys after the expected boundary, but allow slightly early valleys.
    expected_local = int(np.argmin(np.abs(times[idx] - expected_sec)))
    candidate_after = valley_candidates[valley_candidates >= max(0, expected_local - 10)]
    candidates = candidate_after if candidate_after.size else valley_candidates
    valley_local = int(candidates[np.argmin(window[candidates])])
    valley_global = int(idx[valley_local])

    # Measure the contiguous low-energy pocket around the valley.
    left = valley_local
    while left > 0 and valley_mask[left - 1]:
        left -= 1
    right = valley_local
    while right + 1 < valley_mask.size and valley_mask[right + 1]:
        right += 1
    valley_duration_ms = float((times[idx[right]] - times[idx[left]]) * 1000.0)
    if valley_duration_ms < policy.min_valley_ms:
        return Detection(
            pause_start_sec=float(times[idx[left]]),
            valley_sec=float(times[valley_global]),
            valley_duration_ms=valley_duration_ms,
            local_high=local_high,
            local_low=local_low,
            valley_value=float(rms[valley_global]),
            drop_db=db_ratio(float(rms[valley_global]), local_high),
            reject_reason="valley_too_short",
        )

    resume_threshold = local_low + policy.resume_frac_of_range * local_range
    # Require a rise after the low pocket.  This catches resumes and ignores the
    # decaying tail that often precedes a comma.
    post = window.copy()
    resume_mask = post >= resume_threshold
    resume_local = find_consecutive_true(resume_mask, right + 1, policy.consecutive_resume_frames)
    if resume_local is None:
        return Detection(
            pause_start_sec=float(times[idx[left]]),
            valley_sec=float(times[valley_global]),
            valley_duration_ms=valley_duration_ms,
            local_high=local_high,
            local_low=local_low,
            valley_value=float(rms[valley_global]),
            drop_db=db_ratio(float(rms[valley_global]), local_high),
            reject_reason="no_resume_after_valley",
        )

    resume_global = int(idx[resume_local])
    valley_value = float(rms[valley_global])
    resume_value = float(rms[resume_global])
    drop_db = db_ratio(valley_value, local_high)
    rise_db = db_ratio(resume_value, max(valley_value, 1e-7))
    confidence = max(0.0, min(1.0, ((-drop_db) / 24.0) * 0.55 + (rise_db / 24.0) * 0.45))
    return Detection(
        detected=True,
        pause_start_sec=float(times[idx[left]]),
        valley_sec=float(times[valley_global]),
        resume_sec=float(times[resume_global]),
        valley_duration_ms=valley_duration_ms,
        local_high=local_high,
        local_low=local_low,
        valley_value=valley_value,
        resume_value=resume_value,
        drop_db=drop_db,
        rise_db=rise_db,
        confidence=confidence,
        reject_reason="accepted",
    )


def inventory_audio_pauses(
    times: np.ndarray,
    rms: np.ndarray,
    min_pause_ms: float,
    min_resume_frames: int,
) -> list[PauseCandidate]:
    """Find pause/resume candidates without using transcript or planner timing.

    This is deliberately broad.  E02 wants to know what the audio thinks the
    pauses are before text punctuation is allowed to bias the answer.
    """
    if times.size < 8 or rms.size < 8:
        return []
    local_high = percentile(rms, 82.0)
    local_low = percentile(rms, 12.0)
    local_range = max(local_high - local_low, 1e-6)
    if local_high <= 1e-5:
        return []

    valley_threshold = max(local_low + 0.10 * local_range, local_high * 0.34)
    resume_threshold = local_low + 0.38 * local_range
    valley_mask = rms <= valley_threshold
    resume_mask = rms >= resume_threshold
    min_pause_sec = max(0.0, min_pause_ms / 1000.0)

    pauses: list[PauseCandidate] = []
    idx = 0
    while idx < valley_mask.size:
        if not bool(valley_mask[idx]):
            idx += 1
            continue
        left = idx
        while idx + 1 < valley_mask.size and bool(valley_mask[idx + 1]):
            idx += 1
        right = idx
        idx += 1

        duration_sec = float(times[right] - times[left])
        if duration_sec < min_pause_sec:
            continue
        resume_idx = find_consecutive_true(resume_mask, right + 1, min_resume_frames)
        if resume_idx is None:
            continue

        valley_slice = rms[left:right + 1]
        valley_offset = int(np.argmin(valley_slice))
        valley_idx = left + valley_offset
        valley_value = float(rms[valley_idx])
        resume_value = float(rms[resume_idx])
        drop = db_ratio(valley_value, local_high)
        rise = db_ratio(resume_value, max(valley_value, 1e-7))
        confidence = max(0.0, min(1.0, ((-drop) / 24.0) * 0.55 + (rise / 24.0) * 0.45))
        pauses.append(PauseCandidate(
            case_pause_index=len(pauses),
            pause_start_sec=float(times[left]),
            valley_sec=float(times[valley_idx]),
            pause_end_sec=float(times[right]),
            resume_sec=float(times[resume_idx]),
            duration_ms=duration_sec * 1000.0,
            local_high=local_high,
            local_low=local_low,
            valley_value=valley_value,
            resume_value=resume_value,
            drop_db=drop,
            rise_db=rise,
            confidence=confidence,
        ))
    return pauses


def correlate_pause_to_punctuation(
    pauses: list[PauseCandidate],
    planned_resume_sec: float,
    near_ms: float,
    wide_ms: float,
    min_confidence: float,
) -> Correlation:
    confident = [p for p in pauses if p.confidence >= min_confidence]
    if not confident:
        return Correlation(False, "NO_AUDIO_PAUSES_IN_CASE", None, 0.0, 0.0, 0, 0, 0)

    wide_sec = max(0.0, wide_ms / 1000.0)
    near_sec = max(0.0, near_ms / 1000.0)
    wide = [p for p in confident if abs(p.resume_sec - planned_resume_sec) <= wide_sec]
    near = [p for p in wide if abs(p.resume_sec - planned_resume_sec) <= near_sec]
    ambiguous = [p for p in wide if abs(p.resume_sec - planned_resume_sec) <= max(near_sec * 2.0, near_sec + 0.15)]

    if not wide:
        nearest_any = min(confident, key=lambda p: abs(p.resume_sec - planned_resume_sec))
        distance_ms = (nearest_any.resume_sec - planned_resume_sec) * 1000.0
        return Correlation(
            False,
            "NO_AUDIO_PAUSE_WITHIN_WIDE_WINDOW",
            nearest_any,
            distance_ms,
            abs(distance_ms),
            0,
            0,
            0,
        )

    best = min(wide, key=lambda p: (abs(p.resume_sec - planned_resume_sec), -p.confidence))
    distance_ms = (best.resume_sec - planned_resume_sec) * 1000.0
    if len(near) > 1:
        cls = "MULTIPLE_NEAR_CANDIDATES"
        matched = True
    elif abs(distance_ms) <= near_ms:
        cls = "MATCHED_NEAR_PLANNED"
        matched = True
    elif distance_ms < 0.0:
        cls = "PAUSE_FOUND_TOO_EARLY_FOR_PLANNED_TIME"
        matched = False
    else:
        cls = "PAUSE_FOUND_TOO_LATE_FOR_PLANNED_TIME"
        matched = False
    return Correlation(
        matched,
        cls,
        best,
        distance_ms,
        abs(distance_ms),
        len(wide),
        len(near),
        len(ambiguous),
    )


def discover_case_dirs(run: Path) -> list[Path]:
    return sorted({p.parent for p in run.rglob("runtime_commit_events.csv")})


def resolve_inputs_path(inputs: Path, case_id: str, kind: str) -> Path:
    if kind == "transcript":
        candidates = [
            inputs / "transcripts" / f"{case_id}.txt",
            inputs / "input" / "transcripts" / f"{case_id}.txt",
            inputs / "inputs" / "transcripts" / f"{case_id}.txt",
        ]
    else:
        candidates = [
            inputs / "wav" / f"{case_id}.wav",
            inputs / "input" / "wav" / f"{case_id}.wav",
            inputs / "inputs" / "wav" / f"{case_id}.wav",
        ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def case_id_from_dir(case_dir: Path) -> str:
    if case_dir.name.startswith("case_"):
        return case_dir.name
    rows = read_csv_dicts(case_dir / "runtime_commit_events.csv")
    if rows:
        line = rows[0].get("LineID", "").strip()
        if line:
            return line
    return case_dir.name


def best_detection(detections: dict[str, Detection]) -> tuple[str, Detection]:
    accepted = [(name, det) for name, det in detections.items() if det.detected]
    if not accepted:
        # Return the detector that got furthest for easier debugging.
        for reason in ["valley_too_short", "no_resume_after_valley", "no_valley", "too_few_frames"]:
            for name, det in detections.items():
                if det.reject_reason == reason:
                    return name, det
        name = next(iter(detections))
        return name, detections[name]
    accepted.sort(key=lambda item: (-item[1].confidence, abs(item[1].resume_sec)))
    return accepted[0]


def summarize_numeric(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    p90_idx = min(len(s) - 1, int(round(0.90 * (len(s) - 1))))
    return mean(values), median(values), s[p90_idx]


def build_oracle(run: Path, inputs: Path, out_dir: Path, args: argparse.Namespace) -> None:
    pause_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    by_case_rows: list[dict[str, object]] = []
    e1_event_rows: list[dict[str, object]] = []
    e1_case_rows: list[dict[str, object]] = []
    e04_boundary_rows: list[dict[str, object]] = []
    e04_event_rows: list[dict[str, object]] = []
    e04_case_rows: list[dict[str, object]] = []
    policy_accum: dict[str, list[dict[str, object]]] = {p.name: [] for p in POLICIES}

    case_dirs = discover_case_dirs(run)
    for case_dir in case_dirs:
        case_id = case_id_from_dir(case_dir)
        transcript_path = resolve_inputs_path(inputs, case_id, "transcript")
        wav_path = resolve_inputs_path(inputs, case_id, "wav")
        if not transcript_path.exists() or not wav_path.exists():
            continue
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        boundaries = parse_punctuation_boundaries(text)
        if not boundaries:
            by_case_rows.append({
                "CaseID": case_id,
                "BoundaryCount": 0,
                "DetectedCount": 0,
                "AcceptedRate": 0.0,
                "MeanAbsCurrentNextStartErrorMs": 0.0,
                "MeanSignedCurrentNextStartErrorMs": 0.0,
            })
            continue
        word_timings = collect_word_timings(case_dir)
        if not word_timings:
            continue
        try:
            sample_rate, samples = read_wav_mono(wav_path)
        except Exception as exc:
            by_case_rows.append({
                "CaseID": case_id,
                "BoundaryCount": len(boundaries),
                "DetectedCount": 0,
                "AcceptedRate": 0.0,
                "Error": str(exc),
            })
            continue
        times, rms = audio_features(samples, sample_rate, args.frame_ms, args.hop_ms, args.smooth_ms)
        pause_candidates = inventory_audio_pauses(
            times,
            rms,
            args.pause_inventory_min_ms,
            args.pause_inventory_resume_frames,
        )
        for pause in pause_candidates:
            pause_rows.append({
                "CaseID": case_id,
                "PauseIndex": pause.case_pause_index,
                "PauseStartSec": f"{pause.pause_start_sec:.6f}",
                "ValleySec": f"{pause.valley_sec:.6f}",
                "PauseEndSec": f"{pause.pause_end_sec:.6f}",
                "ResumeSec": f"{pause.resume_sec:.6f}",
                "DurationMs": f"{pause.duration_ms:.3f}",
                "LocalHighRms": f"{pause.local_high:.8f}",
                "LocalLowRms": f"{pause.local_low:.8f}",
                "ValleyRms": f"{pause.valley_value:.8f}",
                "ResumeRms": f"{pause.resume_value:.8f}",
                "DropDb": f"{pause.drop_db:.3f}",
                "RiseDb": f"{pause.rise_db:.3f}",
                "Confidence": f"{pause.confidence:.4f}",
            })

        case_detected = 0
        case_errors: list[float] = []
        case_signed: list[float] = []
        case_anchors: list[OracleAnchor] = []
        case_clock_anchors_by_mode: dict[str, list[ClockPauseAnchor]] = {"hold_only": [], "bidirectional": []}
        cumulative_oracle_shift_sec = 0.0
        cumulative_clock_shift_by_mode: dict[str, float] = {"hold_only": 0.0, "bidirectional": 0.0}
        for boundary in boundaries:
            prev_t = word_timings.get(boundary.prev_word_index)
            next_t = word_timings.get(boundary.next_word_index)
            if prev_t is None or next_t is None:
                continue
            expected_sec = next_t.first_center_sec
            raw_expected_sec = next_t.first_raw_center_sec
            detections = {
                p.name: detect_resume(
                    times,
                    rms,
                    expected_sec,
                    p,
                    args.search_before_ms / 1000.0,
                    args.search_after_ms / 1000.0,
                )
                for p in POLICIES
            }
            best_name, best = best_detection(detections)
            correlation = correlate_pause_to_punctuation(
                pause_candidates,
                expected_sec,
                args.pause_correlation_near_ms,
                args.pause_correlation_wide_ms,
                args.pause_correlation_min_confidence,
            )
            matched_pause = correlation.pause
            e02_resume_sec = matched_pause.resume_sec if matched_pause is not None else 0.0
            oracle_expected_sec = expected_sec + cumulative_oracle_shift_sec
            raw_oracle_shift_ms = (e02_resume_sec - oracle_expected_sec) * 1000.0 if correlation.matched and matched_pause is not None else 0.0
            applied_oracle_shift_ms = 0.0
            if matched_pause is not None:
                e02_err = (expected_sec - matched_pause.resume_sec) * 1000.0
            else:
                e02_err = 0.0
            e04_boundary_mode_rows: list[dict[str, object]] = []
            if correlation.matched and matched_pause is not None:
                case_detected += 1
                case_errors.append(abs(e02_err))
                case_signed.append(e02_err)
                applied_oracle_shift_ms = clamp_float(
                    raw_oracle_shift_ms,
                    -args.oracle_max_shift_ms,
                    args.oracle_max_shift_ms,
                )
                if abs(applied_oracle_shift_ms) >= args.oracle_min_shift_ms:
                    cumulative_oracle_shift_sec += applied_oracle_shift_ms / 1000.0
                    case_anchors.append(OracleAnchor(
                        boundary_index=boundary.boundary_index,
                        prev_word_index=boundary.prev_word_index,
                        next_word_index=boundary.next_word_index,
                        punctuation_kind=boundary.punctuation_kind,
                        planned_resume_sec=expected_sec,
                        detected_resume_sec=matched_pause.resume_sec,
                        raw_shift_ms=raw_oracle_shift_ms,
                        applied_shift_ms=applied_oracle_shift_ms,
                        confidence=matched_pause.confidence,
                        policy="audio_pause_inventory",
                    ))
                else:
                    applied_oracle_shift_ms = 0.0

                for mode in ["hold_only", "bidirectional"]:
                    planned_resume_after_clock_sec = expected_sec + cumulative_clock_shift_by_mode[mode]
                    raw_clock_delta_ms = (matched_pause.resume_sec - planned_resume_after_clock_sec) * 1000.0
                    if mode == "hold_only":
                        applied_clock_delta_ms = clamp_float(
                            max(0.0, raw_clock_delta_ms),
                            0.0,
                            args.clock_pause_max_hold_ms,
                        )
                    else:
                        applied_clock_delta_ms = clamp_float(
                            raw_clock_delta_ms,
                            -args.clock_pause_max_advance_ms,
                            args.clock_pause_max_hold_ms,
                        )

                    accept_reason = "accepted"
                    if abs(applied_clock_delta_ms) < args.clock_pause_min_delta_ms:
                        accept_reason = "delta_below_min"
                        applied_clock_delta_ms = 0.0

                    e04_boundary_mode_rows.append({
                        "CaseID": case_id,
                        "Mode": mode,
                        "BoundaryIndex": boundary.boundary_index,
                        "Punctuation": boundary.punctuation,
                        "PunctuationKind": boundary.punctuation_kind,
                        "PrevWordIndex": boundary.prev_word_index,
                        "NextWordIndex": boundary.next_word_index,
                        "PrevWord": boundary.prev_word,
                        "NextWord": boundary.next_word,
                        "PlanPauseStartSec": f"{prev_t.last_center_sec + cumulative_clock_shift_by_mode[mode]:.6f}",
                        "PlanResumeSec": f"{planned_resume_after_clock_sec:.6f}",
                        "AudioPauseStartSec": f"{matched_pause.pause_start_sec:.6f}",
                        "AudioPauseEndSec": f"{matched_pause.pause_end_sec:.6f}",
                        "AudioResumeSec": f"{matched_pause.resume_sec:.6f}",
                        "AudioPauseDurationMs": f"{matched_pause.duration_ms:.3f}",
                        "RawClockDeltaMs": f"{raw_clock_delta_ms:.3f}",
                        "AppliedClockDeltaMs": f"{applied_clock_delta_ms:.3f}",
                        "CumulativeClockDeltaBeforeMs": f"{cumulative_clock_shift_by_mode[mode] * 1000.0:.3f}",
                        "CumulativeClockDeltaAfterMs": f"{(cumulative_clock_shift_by_mode[mode] + applied_clock_delta_ms / 1000.0) * 1000.0:.3f}",
                        "Accepted": 1 if accept_reason == "accepted" else 0,
                        "Reason": accept_reason,
                        "Confidence": f"{matched_pause.confidence:.4f}",
                    })

                    if accept_reason == "accepted":
                        case_clock_anchors_by_mode[mode].append(ClockPauseAnchor(
                            boundary_index=boundary.boundary_index,
                            prev_word_index=boundary.prev_word_index,
                            next_word_index=boundary.next_word_index,
                            punctuation_kind=boundary.punctuation_kind,
                            planned_pause_start_sec=prev_t.last_center_sec,
                            planned_resume_sec=expected_sec,
                            audio_pause_start_sec=matched_pause.pause_start_sec,
                            audio_resume_sec=matched_pause.resume_sec,
                            raw_clock_delta_ms=raw_clock_delta_ms,
                            applied_clock_delta_ms=applied_clock_delta_ms,
                            confidence=matched_pause.confidence,
                            mode=mode,
                            accept_reason=accept_reason,
                        ))
                        cumulative_clock_shift_by_mode[mode] += applied_clock_delta_ms / 1000.0
            else:
                for mode in ["hold_only", "bidirectional"]:
                    e04_boundary_mode_rows.append({
                        "CaseID": case_id,
                        "Mode": mode,
                        "BoundaryIndex": boundary.boundary_index,
                        "Punctuation": boundary.punctuation,
                        "PunctuationKind": boundary.punctuation_kind,
                        "PrevWordIndex": boundary.prev_word_index,
                        "NextWordIndex": boundary.next_word_index,
                        "PrevWord": boundary.prev_word,
                        "NextWord": boundary.next_word,
                        "PlanPauseStartSec": f"{prev_t.last_center_sec + cumulative_clock_shift_by_mode[mode]:.6f}",
                        "PlanResumeSec": f"{expected_sec + cumulative_clock_shift_by_mode[mode]:.6f}",
                        "AudioPauseStartSec": "",
                        "AudioPauseEndSec": "",
                        "AudioResumeSec": "",
                        "AudioPauseDurationMs": "",
                        "RawClockDeltaMs": "",
                        "AppliedClockDeltaMs": "0.000",
                        "CumulativeClockDeltaBeforeMs": f"{cumulative_clock_shift_by_mode[mode] * 1000.0:.3f}",
                        "CumulativeClockDeltaAfterMs": f"{cumulative_clock_shift_by_mode[mode] * 1000.0:.3f}",
                        "Accepted": 0,
                        "Reason": correlation.classification,
                        "Confidence": "",
                    })
            e04_boundary_rows.extend(e04_boundary_mode_rows)
            err = (expected_sec - best.resume_sec) * 1000.0 if best.detected else 0.0

            phrase_changed = 1 if prev_t.phrase_index >= 0 and next_t.phrase_index >= 0 and prev_t.phrase_index != next_t.phrase_index else 0
            center_gap_ms = (next_t.first_center_sec - prev_t.last_center_sec) * 1000.0
            raw_gap_ms = (next_t.first_raw_center_sec - prev_t.last_raw_center_sec) * 1000.0
            row: dict[str, object] = {
                "CaseID": case_id,
                "BoundaryIndex": boundary.boundary_index,
                "Punctuation": boundary.punctuation,
                "PunctuationKind": boundary.punctuation_kind,
                "HardSentenceBreak": 1 if boundary.hard_break else 0,
                "PrevWordIndex": boundary.prev_word_index,
                "NextWordIndex": boundary.next_word_index,
                "PrevWord": boundary.prev_word,
                "NextWord": boundary.next_word,
                "PrevPhraseIndex": prev_t.phrase_index,
                "NextPhraseIndex": next_t.phrase_index,
                "PhraseChanged": phrase_changed,
                "PrevLastCenterSec": f"{prev_t.last_center_sec:.6f}",
                "NextFirstCenterSec": f"{next_t.first_center_sec:.6f}",
                "RawNextFirstCenterSec": f"{raw_expected_sec:.6f}",
                "CommittedCenterGapMs": f"{center_gap_ms:.3f}",
                "RawCenterGapMs": f"{raw_gap_ms:.3f}",
                "AudioPauseCandidateCount": len(pause_candidates),
                "E02CorrelationClass": correlation.classification,
                "E02Matched": 1 if correlation.matched else 0,
                "E02WideCandidateCount": correlation.wide_candidate_count,
                "E02NearCandidateCount": correlation.near_candidate_count,
                "E02AmbiguousCandidateCount": correlation.ambiguous_candidate_count,
                "E02BestPauseIndex": matched_pause.case_pause_index if matched_pause is not None else "",
                "E02BestPauseStartSec": f"{matched_pause.pause_start_sec:.6f}" if matched_pause is not None else "",
                "E02BestPauseValleySec": f"{matched_pause.valley_sec:.6f}" if matched_pause is not None else "",
                "E02BestPauseEndSec": f"{matched_pause.pause_end_sec:.6f}" if matched_pause is not None else "",
                "E02BestPauseResumeSec": f"{matched_pause.resume_sec:.6f}" if matched_pause is not None else "",
                "E02BestPauseDistanceMs": f"{correlation.distance_ms:.3f}" if matched_pause is not None else "",
                "E02BestPauseAbsDistanceMs": f"{correlation.abs_distance_ms:.3f}" if matched_pause is not None else "",
                "E02BestPauseDurationMs": f"{matched_pause.duration_ms:.3f}" if matched_pause is not None else "",
                "E02BestPauseConfidence": f"{matched_pause.confidence:.4f}" if matched_pause is not None else "",
                "BestPolicy": best_name,
                "Detected": 1 if best.detected else 0,
                "RejectReason": best.reject_reason,
                "PauseStartSec": f"{best.pause_start_sec:.6f}",
                "ValleySec": f"{best.valley_sec:.6f}",
                "ResumeSec": f"{best.resume_sec:.6f}",
                "ValleyDurationMs": f"{best.valley_duration_ms:.3f}",
                "LocalHighRms": f"{best.local_high:.8f}",
                "LocalLowRms": f"{best.local_low:.8f}",
                "ValleyRms": f"{best.valley_value:.8f}",
                "ResumeRms": f"{best.resume_value:.8f}",
                "DropDb": f"{best.drop_db:.3f}",
                "RiseDb": f"{best.rise_db:.3f}",
                "Confidence": f"{best.confidence:.4f}",
                "NextStartMinusResumeMs": f"{err:.3f}" if best.detected else "",
                "NextStartAbsErrorMs": f"{abs(err):.3f}" if best.detected else "",
                "E1ExpectedResumeAfterPriorShiftsSec": f"{oracle_expected_sec:.6f}",
                "E1RawDownstreamShiftMs": f"{raw_oracle_shift_ms:.3f}" if correlation.matched else "",
                "E1AppliedDownstreamShiftMs": f"{applied_oracle_shift_ms:.3f}" if correlation.matched else "",
                "E1CumulativeShiftAfterBoundaryMs": f"{cumulative_oracle_shift_sec * 1000.0:.3f}",
                "E1AnchorAccepted": 1 if abs(applied_oracle_shift_ms) >= args.oracle_min_shift_ms else 0,
                "StartsBeforeResume": 1 if best.detected and err < -1.0 else 0,
                "StartsAfterResume": 1 if best.detected and err > 1.0 else 0,
                "SearchStartSec": f"{max(0.0, expected_sec - args.search_before_ms / 1000.0):.6f}",
                "SearchEndSec": f"{min(float(times[-1]) if times.size else 0.0, expected_sec + args.search_after_ms / 1000.0):.6f}",
            }
            # Include every policy's compact outcome to make threshold failure obvious.
            for name, det in detections.items():
                row[f"{name}_Detected"] = 1 if det.detected else 0
                row[f"{name}_Reason"] = det.reject_reason
                row[f"{name}_ResumeSec"] = f"{det.resume_sec:.6f}"
                if det.detected:
                    row[f"{name}_NextStartMinusResumeMs"] = f"{(expected_sec - det.resume_sec) * 1000.0:.3f}"
                policy_accum[name].append({
                    "detected": det.detected,
                    "punctuation_kind": boundary.punctuation_kind,
                    "hard": boundary.hard_break,
                    "error_ms": (expected_sec - det.resume_sec) * 1000.0 if det.detected else None,
                    "confidence": det.confidence,
                    "reason": det.reject_reason,
                })
            boundary_rows.append(row)

        case_mean, case_med, case_p90 = summarize_numeric(case_errors)
        signed_mean = mean(case_signed) if case_signed else 0.0
        by_case_rows.append({
            "CaseID": case_id,
            "BoundaryCount": len(boundaries),
            "DetectedCount": case_detected,
            "AcceptedRate": f"{case_detected / max(1, len(boundaries)):.6f}",
            "MeanAbsCurrentNextStartErrorMs": f"{case_mean:.3f}",
            "MedianAbsCurrentNextStartErrorMs": f"{case_med:.3f}",
            "P90AbsCurrentNextStartErrorMs": f"{case_p90:.3f}",
            "MeanSignedCurrentNextStartErrorMs": f"{signed_mean:.3f}",
        })

        event_rows = collect_event_timings(case_dir)
        adjusted_rows = apply_e1_oracle_to_events(event_rows, case_anchors, args.oracle_min_event_gap_ms)
        for adjusted in adjusted_rows:
            adjusted["CaseID"] = case_id
            adjusted["AcceptedAnchorCount"] = len(case_anchors)
            e1_event_rows.append(adjusted)
        e1_case = summarize_event_error_rows(adjusted_rows, case_id)
        e1_case["CaseID"] = case_id
        e1_case["BoundaryCount"] = len(boundaries)
        e1_case["DetectedCount"] = case_detected
        e1_case["AcceptedAnchorCount"] = len(case_anchors)
        e1_case_rows.append(e1_case)

        for mode, anchors in case_clock_anchors_by_mode.items():
            clock_rows = apply_clock_pause_to_events(event_rows, anchors, args.oracle_min_event_gap_ms)
            for adjusted in clock_rows:
                adjusted["CaseID"] = case_id
                adjusted["Mode"] = mode
                adjusted["AcceptedClockPauseCount"] = len(anchors)
                e04_event_rows.append(adjusted)
            e04_case = summarize_clock_event_error_rows(clock_rows, case_id, mode)
            e04_case["CaseID"] = case_id
            e04_case["BoundaryCount"] = len(boundaries)
            e04_case["DetectedCount"] = case_detected
            e04_case["AcceptedClockPauseCount"] = len(anchors)
            e04_case_rows.append(e04_case)

    # Build policy summaries overall and by punctuation kind.
    summary_rows: list[dict[str, object]] = []
    for name, rows in policy_accum.items():
        for kind in ["ALL", "comma", "sentence", "semicolon", "colon", "dash", "ellipsis", "other"]:
            subset = rows if kind == "ALL" else [r for r in rows if r["punctuation_kind"] == kind]
            if not subset:
                continue
            detected = [r for r in subset if r["detected"]]
            errors = [abs(float(r["error_ms"])) for r in detected if r["error_ms"] is not None]
            signed = [float(r["error_ms"]) for r in detected if r["error_ms"] is not None]
            conf = [float(r["confidence"]) for r in detected]
            m, med, p90 = summarize_numeric(errors)
            summary_rows.append({
                "Policy": name,
                "PunctuationKind": kind,
                "BoundaryCount": len(subset),
                "DetectedCount": len(detected),
                "DetectedRate": f"{len(detected) / max(1, len(subset)):.6f}",
                "MeanAbsNextStartErrorMs": f"{m:.3f}",
                "MedianAbsNextStartErrorMs": f"{med:.3f}",
                "P90AbsNextStartErrorMs": f"{p90:.3f}",
                "MeanSignedNextStartErrorMs": f"{(mean(signed) if signed else 0.0):.3f}",
                "MeanConfidence": f"{(mean(conf) if conf else 0.0):.4f}",
            })

    correlation_summary_rows: list[dict[str, object]] = []
    for kind in ["ALL", "comma", "sentence", "semicolon", "colon", "dash", "ellipsis", "other"]:
        subset = boundary_rows if kind == "ALL" else [r for r in boundary_rows if r.get("PunctuationKind") == kind]
        if not subset:
            continue
        matched = [r for r in subset if int(r.get("E02Matched", 0)) == 1]
        with_pause = [r for r in subset if str(r.get("E02BestPauseAbsDistanceMs", "")) != ""]
        abs_dist = [float(r["E02BestPauseAbsDistanceMs"]) for r in with_pause]
        signed_dist = [float(r["E02BestPauseDistanceMs"]) for r in with_pause]
        m, med, p90 = summarize_numeric(abs_dist)
        classes: dict[str, int] = {}
        for r in subset:
            cls = str(r.get("E02CorrelationClass", ""))
            classes[cls] = classes.get(cls, 0) + 1
        correlation_summary_rows.append({
            "PunctuationKind": kind,
            "BoundaryCount": len(subset),
            "MatchedCount": len(matched),
            "MatchedRate": f"{len(matched) / max(1, len(subset)):.6f}",
            "HasWideOrNearestPauseCount": len(with_pause),
            "MeanAbsPauseDistanceMs": f"{m:.3f}",
            "MedianAbsPauseDistanceMs": f"{med:.3f}",
            "P90AbsPauseDistanceMs": f"{p90:.3f}",
            "MeanSignedPauseDistanceMs": f"{(mean(signed_dist) if signed_dist else 0.0):.3f}",
            "ClassCounts": ";".join(f"{k}:{v}" for k, v in sorted(classes.items())),
        })

    # Stable field order. Policy columns are appended alphabetically.
    base_fields = [
        "CaseID", "BoundaryIndex", "Punctuation", "PunctuationKind", "HardSentenceBreak",
        "PrevWordIndex", "NextWordIndex", "PrevWord", "NextWord",
        "PrevPhraseIndex", "NextPhraseIndex", "PhraseChanged",
        "PrevLastCenterSec", "NextFirstCenterSec", "RawNextFirstCenterSec",
        "CommittedCenterGapMs", "RawCenterGapMs",
        "AudioPauseCandidateCount", "E02CorrelationClass", "E02Matched",
        "E02WideCandidateCount", "E02NearCandidateCount", "E02AmbiguousCandidateCount",
        "E02BestPauseIndex", "E02BestPauseStartSec", "E02BestPauseValleySec",
        "E02BestPauseEndSec", "E02BestPauseResumeSec", "E02BestPauseDistanceMs",
        "E02BestPauseAbsDistanceMs", "E02BestPauseDurationMs", "E02BestPauseConfidence",
        "BestPolicy", "Detected", "RejectReason", "PauseStartSec", "ValleySec", "ResumeSec",
        "ValleyDurationMs", "LocalHighRms", "LocalLowRms", "ValleyRms", "ResumeRms",
        "DropDb", "RiseDb", "Confidence", "NextStartMinusResumeMs", "NextStartAbsErrorMs",
        "E1ExpectedResumeAfterPriorShiftsSec", "E1RawDownstreamShiftMs",
        "E1AppliedDownstreamShiftMs", "E1CumulativeShiftAfterBoundaryMs", "E1AnchorAccepted",
        "StartsBeforeResume", "StartsAfterResume", "SearchStartSec", "SearchEndSec",
    ]
    write_csv(out_dir / "audio_pause_inventory.csv", pause_rows, [
        "CaseID", "PauseIndex", "PauseStartSec", "ValleySec", "PauseEndSec", "ResumeSec",
        "DurationMs", "LocalHighRms", "LocalLowRms", "ValleyRms", "ResumeRms",
        "DropDb", "RiseDb", "Confidence",
    ])
    extra_fields = sorted({k for r in boundary_rows for k in r.keys() if k not in base_fields})
    write_csv(out_dir / "punctuation_resume_oracle_boundaries.csv", boundary_rows, base_fields + extra_fields)
    write_csv(out_dir / "punctuation_resume_oracle_by_case.csv", by_case_rows, [
        "CaseID", "BoundaryCount", "DetectedCount", "AcceptedRate",
        "MeanAbsCurrentNextStartErrorMs", "MedianAbsCurrentNextStartErrorMs",
        "P90AbsCurrentNextStartErrorMs", "MeanSignedCurrentNextStartErrorMs", "Error",
    ])
    write_csv(out_dir / "punctuation_resume_oracle_policy_summary.csv", summary_rows, [
        "Policy", "PunctuationKind", "BoundaryCount", "DetectedCount", "DetectedRate",
        "MeanAbsNextStartErrorMs", "MedianAbsNextStartErrorMs", "P90AbsNextStartErrorMs",
        "MeanSignedNextStartErrorMs", "MeanConfidence",
    ])
    write_csv(out_dir / "punctuation_audio_pause_correlation_summary.csv", correlation_summary_rows, [
        "PunctuationKind", "BoundaryCount", "MatchedCount", "MatchedRate",
        "HasWideOrNearestPauseCount", "MeanAbsPauseDistanceMs", "MedianAbsPauseDistanceMs",
        "P90AbsPauseDistanceMs", "MeanSignedPauseDistanceMs", "ClassCounts",
    ])

    e1_summary_rows = [summarize_event_error_rows(e1_event_rows, "ALL")]
    for kind in ["closure", "fv", "round", "front_vowel", "other"]:
        subset = [r for r in e1_event_rows if str(r.get("pose_id", "")).lower().find(kind) >= 0]
        if subset:
            e1_summary_rows.append(summarize_event_error_rows(subset, f"pose_name_contains_{kind}"))
    write_csv(out_dir / "punctuation_resume_oracle_e1_summary.csv", e1_summary_rows, [
        "Slice", "EventCount", "MeanBeforeAbsErrorMs", "MeanAfterAbsErrorMs", "MeanImprovementMs",
        "MedianBeforeAbsErrorMs", "MedianAfterAbsErrorMs", "P90BeforeAbsErrorMs", "P90AfterAbsErrorMs",
        "ImprovedCount", "WorsenedCount", "ImprovedRate", "WorsenedRate",
    ])
    write_csv(out_dir / "punctuation_resume_oracle_e1_by_case.csv", e1_case_rows, [
        "CaseID", "BoundaryCount", "DetectedCount", "AcceptedAnchorCount",
        "EventCount", "MeanBeforeAbsErrorMs", "MeanAfterAbsErrorMs", "MeanImprovementMs",
        "MedianBeforeAbsErrorMs", "MedianAfterAbsErrorMs", "P90BeforeAbsErrorMs", "P90AfterAbsErrorMs",
        "ImprovedCount", "WorsenedCount", "ImprovedRate", "WorsenedRate",
    ])
    write_csv(out_dir / "punctuation_resume_oracle_event_errors.csv", e1_event_rows, [
        "CaseID", "AcceptedAnchorCount", "event_index", "word_index", "pose_id", "source_word",
        "streaming_center_sec", "evidence_center_sec", "evidence_confidence", "evidence_fallback",
        "streaming_abs_error_ms", "oracle_center_sec", "oracle_cumulative_shift_ms",
        "oracle_order_repair_ms", "oracle_abs_error_ms", "oracle_delta_error_ms",
        "oracle_improved", "oracle_worsened",
    ])

    write_csv(out_dir / "punctuation_clock_pause_events.csv", e04_boundary_rows, [
        "CaseID", "Mode", "BoundaryIndex", "Punctuation", "PunctuationKind",
        "PrevWordIndex", "NextWordIndex", "PrevWord", "NextWord",
        "PlanPauseStartSec", "PlanResumeSec", "AudioPauseStartSec", "AudioPauseEndSec",
        "AudioResumeSec", "AudioPauseDurationMs", "RawClockDeltaMs", "AppliedClockDeltaMs",
        "CumulativeClockDeltaBeforeMs", "CumulativeClockDeltaAfterMs",
        "Accepted", "Reason", "Confidence",
    ])

    e04_summary_rows: list[dict[str, object]] = []
    for mode in ["hold_only", "bidirectional"]:
        mode_rows = [r for r in e04_event_rows if r.get("Mode") == mode]
        e04_summary_rows.append(summarize_clock_event_error_rows(mode_rows, "ALL", mode))
        for kind in ["closure", "fv", "round", "front_vowel", "other"]:
            subset = [r for r in mode_rows if str(r.get("pose_id", "")).lower().find(kind) >= 0]
            if subset:
                e04_summary_rows.append(summarize_clock_event_error_rows(subset, f"pose_name_contains_{kind}", mode))
    write_csv(out_dir / "punctuation_clock_pause_oracle_summary.csv", e04_summary_rows, [
        "Mode", "Slice", "EventCount", "MeanBeforeAbsErrorMs", "MeanAfterAbsErrorMs",
        "MeanImprovementMs", "MedianBeforeAbsErrorMs", "MedianAfterAbsErrorMs",
        "P90BeforeAbsErrorMs", "P90AfterAbsErrorMs", "ImprovedCount", "WorsenedCount",
        "ImprovedRate", "WorsenedRate",
    ])
    write_csv(out_dir / "punctuation_clock_pause_oracle_by_case.csv", e04_case_rows, [
        "Mode", "CaseID", "BoundaryCount", "DetectedCount", "AcceptedClockPauseCount",
        "EventCount", "MeanBeforeAbsErrorMs", "MeanAfterAbsErrorMs", "MeanImprovementMs",
        "MedianBeforeAbsErrorMs", "MedianAfterAbsErrorMs", "P90BeforeAbsErrorMs", "P90AfterAbsErrorMs",
        "ImprovedCount", "WorsenedCount", "ImprovedRate", "WorsenedRate",
    ])
    write_csv(out_dir / "punctuation_clock_pause_event_errors.csv", e04_event_rows, [
        "CaseID", "Mode", "AcceptedClockPauseCount", "event_index", "word_index", "pose_id", "source_word",
        "streaming_center_sec", "evidence_center_sec", "evidence_confidence", "evidence_fallback",
        "streaming_abs_error_ms", "clock_center_sec", "clock_cumulative_delta_ms",
        "clock_order_repair_ms", "clock_abs_error_ms", "clock_delta_error_ms",
        "clock_improved", "clock_worsened",
    ])

    print(f"Cases scanned: {len(case_dirs)}")
    print(f"Boundary rows: {len(boundary_rows)}")
    print(f"Output: {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="LipLab run directory containing per_case outputs")
    ap.add_argument("--inputs", default="inputs", type=Path, help="Corpus root containing transcripts/ and wav/")
    ap.add_argument("--out", default=None, type=Path, help="Output directory; default <run>/punctuation_oracle")
    ap.add_argument("--frame-ms", default=25.0, type=float)
    ap.add_argument("--hop-ms", default=10.0, type=float)
    ap.add_argument("--smooth-ms", default=45.0, type=float)
    ap.add_argument("--search-before-ms", default=550.0, type=float)
    ap.add_argument("--search-after-ms", default=750.0, type=float)
    ap.add_argument("--pause-inventory-min-ms", default=50.0, type=float, help="E02 minimum audio-only low-energy run to inventory as a pause")
    ap.add_argument("--pause-inventory-resume-frames", default=2, type=int, help="E02 consecutive speech frames needed after an audio-only pause")
    ap.add_argument("--pause-correlation-near-ms", default=350.0, type=float, help="E02 punctuation is a near match when audio resume is within this window")
    ap.add_argument("--pause-correlation-wide-ms", default=1500.0, type=float, help="E02 broad search window used to diagnose wrong timeframe vs no pause")
    ap.add_argument("--pause-correlation-min-confidence", default=0.18, type=float, help="E02 minimum audio pause confidence eligible for punctuation correlation")
    ap.add_argument("--oracle-max-shift-ms", default=450.0, type=float, help="E1 clamp for each downstream punctuation-resume shift")
    ap.add_argument("--oracle-min-shift-ms", default=15.0, type=float, help="E1 ignores accepted resumes whose correction is below this size")
    ap.add_argument("--oracle-min-event-gap-ms", default=18.0, type=float, help="E1/E04 monotonic repair gap between adjusted event centers")
    ap.add_argument("--clock-pause-min-delta-ms", default=15.0, type=float, help="E04 ignores punctuation clock corrections below this size")
    ap.add_argument("--clock-pause-max-hold-ms", default=900.0, type=float, help="E04 maximum extra wait inserted at a punctuation pause")
    ap.add_argument("--clock-pause-max-advance-ms", default=350.0, type=float, help="E04 bidirectional oracle maximum planned-pause removal")
    args = ap.parse_args()

    run = args.run.resolve()
    inputs = args.inputs.resolve()
    out_dir = (args.out.resolve() if args.out is not None else run / "punctuation_oracle")
    if not run.exists():
        raise SystemExit(f"Missing run directory: {run}")
    if not inputs.exists():
        raise SystemExit(f"Missing inputs directory: {inputs}")
    build_oracle(run, inputs, out_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
