#!/usr/bin/env python3
"""
Summarize a liplab experiment run.

Reads per-case scorecard.csv files in:
  <run>/per_case/case_XXXXXX/scorecard.csv

Supports scorecards shaped as either:
  Metric,Value
  StreamingTimingMAEMs,123.4
or already-flat one-row CSVs.

Writes:
  <run>/summary.csv
  <run>/metric_stats.csv
  <run>/worst_cases.csv
  <run>/corpus_summary.csv
  <run>/corpus_phase_summary.txt
and per-case:
  metrics_flat.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, pstdev


WORST_CASE_PRIORITY = [
    ("HardInvariantFailureCount", "desc"),
    ("MissingCommittedEventCount", "desc"),
    ("DuplicateCommittedEventCount", "desc"),
    ("InvalidCommittedEventIndexCount", "desc"),
    ("OutOfOrderCommittedEventCount", "desc"),
    ("NonMonotonicCommittedCenterCount", "desc"),
    ("EventIndexCoverageRate", "asc"),
    ("MaxMissingEventIndexRun", "desc"),
    ("EventsCenteredOutsideSpeechIslandCount", "desc"),
    ("EventsCenteredInsideSpeechIslandRate", "asc"),
    ("MaxDistanceOutsideSpeechIslandMs", "desc"),
    ("SoftIslandEndOverrunCount", "desc"),
    ("MaxSoftIslandEndOverrunMs", "desc"),
    ("CompressedEventGapCountUnder16ms", "desc"),
    ("CompressedEventGapCountUnder33ms", "desc"),
    ("MaxCompressedRunLengthUnder33ms", "desc"),
    ("MinCommittedCenterGapMs", "asc"),
    ("EarlyPhraseStartCount", "desc"),
    ("MaxEarlyPhraseStartLeadMs", "desc"),
    ("SeverelyCompressedPhraseCount", "desc"),
    ("MinPhraseDurationRatio", "asc"),
    ("MissingPeakCount", "desc"),
    ("MonotonicityViolations", "desc"),
    ("RawTextTimingMAEMs", "desc"),
    ("P90AbsRawTextTimingErrorMs", "desc"),
    ("P95AbsRawTextTimingErrorMs", "desc"),
    ("RawTextLeadBiasMsAbs", "desc"),
    ("VisiblePoseGapCountOver250ms", "desc"),
    ("MaxVisiblePoseGapSec", "desc"),
    ("ScaleCapHitCount", "desc"),
    ("EffectiveSegmentScaleMax", "desc"),
    ("StreamingTimingMAEMs", "desc"),
    ("P90AbsStreamingTimingErrorMs", "desc"),
    ("P95AbsStreamingTimingErrorMs", "desc"),
    ("StreamingLeadBiasMsAbs", "desc"),
    ("AnchoredStreamingTimingMAEMs", "desc"),
    ("P90AbsAnchoredStreamingTimingErrorMs", "desc"),
    ("P95AbsAnchoredStreamingTimingErrorMs", "desc"),
    ("WeakPeakRate", "desc"),
    ("LandmarkWeakPeakRate", "desc"),
    ("DominantWeakPeakRate", "desc"),
    ("DominantPeakRatioMin", "asc"),
    ("PeakTimingMAEMs", "desc"),
    ("PeakWindowCenterMAEMs", "desc"),
    ("ExplosivePeakLeadErrorMAEMs", "desc"),
    ("ExplosiveWeakPeakRate", "desc"),
    ("VowelPeakCenterMAEMs", "desc"),
    ("VowelWeakPeakRate", "desc"),
    ("CommitLeadMeanSec", "desc"),
]


ALIASES = {
    "CaseId": "case_id",
    "caseId": "case_id",
    "CaseID": "case_id",
}


def parse_float(value: str):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_key(key: str) -> str:
    return ALIASES.get(key, key)


def normalize_row(row: dict[str, str], case_id: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        normalized[normalize_key(key)] = value
    normalized["case_id"] = normalized.get("case_id", case_id) or case_id
    return normalized


def read_scorecard(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        return {}

    def looks_like_key_value_scorecard(all_rows: list[list[str]]) -> bool:
        if not all_rows:
            return False
        usable = [r for r in all_rows if len(r) >= 2 and r[0].strip()]
        if not usable:
            return False
        first_key = usable[0][0].strip().lower()
        # Headered two-column scorecards are handled separately below.  This
        # branch catches writer output of the form:
        #   SomeMetric,123
        #   AnotherMetric,456
        # with no Metric,Value header.  Keep this for compatibility with older
        # experimental runs that emitted bare key/value rows.
        if first_key in {"metric", "key", "field", "case_id", "caseid", "caseid"}:
            return False
        metric_like = 0
        for row in usable:
            key = row[0].strip()
            if key and not parse_float(key):
                metric_like += 1
        return metric_like >= max(1, int(0.8 * len(usable)))

    header = [c.strip() for c in rows[0]]

    if len(header) >= 2 and header[0].lower() == "metric" and header[1].lower() == "value":
        out: dict[str, str] = {}
        for row in rows[1:]:
            if len(row) < 2:
                continue
            key = normalize_key(row[0].strip())
            val = row[1].strip()
            if key:
                out[key] = val
        return out

    if looks_like_key_value_scorecard(rows):
        out: dict[str, str] = {}
        for row in rows:
            if len(row) < 2:
                continue
            key = normalize_key(row[0].strip())
            val = row[1].strip()
            if key:
                out[key] = val
        return out

    if len(rows) >= 2:
        values = rows[1]
        out: dict[str, str] = {}
        for i, raw_key in enumerate(header):
            key = normalize_key(raw_key)
            if not key:
                continue
            out[key] = values[i].strip() if i < len(values) else ""
        return out

    return {}



def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_int(value: str) -> int | None:
    v = parse_float(value)
    if v is None:
        return None
    return int(v)


def truthy_csv(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def unique_count(rows: list[dict[str, str]], key: str) -> int:
    values = {row.get(key, "").strip() for row in rows if row.get(key, "").strip() != ""}
    return len(values)


def sum_float(rows: list[dict[str, str]], key: str) -> float:
    total = 0.0
    for row in rows:
        v = parse_float(row.get(key, ""))
        if v is not None:
            total += v
    return total


def positive_sum_float(rows: list[dict[str, str]], key: str) -> float:
    total = 0.0
    for row in rows:
        v = parse_float(row.get(key, ""))
        if v is not None and math.isfinite(v) and v > 0.0:
            total += v
    return total


def speech_island_end(row: dict[str, str]) -> float | None:
    end = parse_float(row.get("AudioBufferEndSec", ""))
    last = parse_float(row.get("AudioBufferLastSpeechSec", ""))
    start = parse_float(row.get("AudioBufferStartSec", ""))
    if end is not None and math.isfinite(end):
        return end
    if last is not None and math.isfinite(last):
        return last
    if start is not None and math.isfinite(start):
        return start
    return None


def _float_values(rows: list[dict[str, str]], key: str, *, abs_value: bool = False) -> list[float]:
    values: list[float] = []
    for row in rows:
        v = parse_float(row.get(key, ""))
        if v is None or not math.isfinite(v):
            continue
        values.append(abs(v) if abs_value else v)
    return values


def _track_centers_by_event(rows: list[dict[str, str]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        idx = parse_int(row.get("EventIndex", ""))
        if idx is None:
            continue
        center = parse_float(row.get("FinalRenderCenterSec", ""))
        if center is None:
            center = parse_float(row.get("CenterSeconds", ""))
        if center is None or not math.isfinite(center):
            continue
        out[idx] = center
    return out


def _track_difference_stats_ms(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]]) -> tuple[int, float, float]:
    a = _track_centers_by_event(a_rows)
    b = _track_centers_by_event(b_rows)
    diffs = [abs((a[idx] - b[idx]) * 1000.0) for idx in sorted(set(a.keys()) & set(b.keys()))]
    changed = [d for d in diffs if d > 0.001]
    if not diffs:
        return 0, 0.0, 0.0
    return len(changed), mean(diffs), max(diffs)




def _simple_percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(len(sv) - 1, max(0, int(math.ceil(len(sv) * (pct / 100.0))) - 1))
    return sv[idx]


def _landmark_class_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    vowels = {"VowelOpen", "VowelFront", "WOO"}
    fricatives = {"FV", "SHCH"}
    return (a in vowels and b in vowels) or (a in fricatives and b in fricatives)


def _read_landmark_sequence(path: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    out: list[dict[str, object]] = []
    for idx, r in enumerate(rows):
        cls = (r.get("Class") or "").strip()
        t = parse_float(r.get("TimeSec", ""))
        avail = parse_float(r.get("AvailableAtSec", ""))
        conf = parse_float(r.get("Confidence", ""))
        if not cls or t is None or not math.isfinite(t):
            continue
        out.append({
            "index": idx,
            "class": cls,
            "time": t,
            "available": avail if avail is not None and math.isfinite(avail) else 0.0,
            "confidence": conf if conf is not None and math.isfinite(conf) else 0.0,
        })
    out.sort(key=lambda r: (float(r["time"]), int(r["index"])))
    return out


def _align_landmark_sequences(
    oracle: list[dict[str, object]],
    streaming: list[dict[str, object]],
    preroll_sec: float,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Order-preserving alignment between full-WAV oracle landmarks and causal streaming candidates.

    This intentionally scores the *series* of detected evidence, not planned text timing.
    A loose timing penalty is included only to avoid absurd cross-line matches; sequence
    order and class agreement are the primary signal.
    """
    n = len(oracle)
    m = len(streaming)
    gap_penalty = -1.0
    substitution_penalty = -1.25
    loose_window_sec = max(0.50, preroll_sec * 4.0)

    def pair_score(o: dict[str, object], st: dict[str, object]) -> float:
        oc = str(o["class"])
        sc = str(st["class"])
        dt = abs(float(st["time"]) - float(o["time"]))
        time_penalty = min(dt / loose_window_sec, 2.0) * 0.40
        if oc == sc:
            return 2.00 - time_penalty
        if _landmark_class_compatible(oc, sc):
            return 0.45 - time_penalty
        return substitution_penalty - time_penalty

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [["M"] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        bt[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + pair_score(oracle[i - 1], streaming[j - 1])
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            best = match
            op = "M"
            if delete > best:
                best = delete
                op = "D"
            if insert > best:
                best = insert
                op = "I"
            dp[i][j] = best
            bt[i][j] = op

    align_rows: list[dict[str, str]] = []
    same = compatible = substitutions = insertions = deletions = 0
    timing_abs_ms: list[float] = []
    timing_signed_ms: list[float] = []
    acoustic_lead_sec_values: list[float] = []
    buffered_useful_lead_sec_values: list[float] = []
    acoustic_late = 0
    useful_late = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j] if i >= 0 and j >= 0 else ""
        if i > 0 and j > 0 and op == "M":
            o = oracle[i - 1]
            st = streaming[j - 1]
            oc = str(o["class"])
            sc = str(st["class"])
            dt_ms = (float(st["time"]) - float(o["time"])) * 1000.0
            acoustic_lead_sec = float(o["time"]) - float(st.get("available", 0.0))
            buffered_useful_lead_sec = float(o["time"]) + preroll_sec - float(st.get("available", 0.0))
            relation = "same" if oc == sc else ("compatible" if _landmark_class_compatible(oc, sc) else "substitution")
            if relation == "same":
                same += 1
                timing_abs_ms.append(abs(dt_ms))
                timing_signed_ms.append(dt_ms)
                acoustic_lead_sec_values.append(acoustic_lead_sec)
                buffered_useful_lead_sec_values.append(buffered_useful_lead_sec)
                if acoustic_lead_sec < -max(0.005, preroll_sec * 0.10):
                    acoustic_late += 1
                if buffered_useful_lead_sec < 0.0:
                    useful_late += 1
            elif relation == "compatible":
                compatible += 1
            else:
                substitutions += 1
            align_rows.append({
                "AlignmentOp": relation,
                "OracleIndex": str(o["index"]),
                "StreamingIndex": str(st["index"]),
                "OracleClass": oc,
                "StreamingClass": sc,
                "OracleTimeSec": f"{float(o['time']):.9g}",
                "StreamingTimeSec": f"{float(st['time']):.9g}",
                "StreamingAvailableAtSec": f"{float(st.get('available', 0.0)):.9g}",
                "TimingDeltaMs": f"{dt_ms:.9g}",
                "AcousticLeadSec": f"{acoustic_lead_sec:.9g}",
                "BufferedUsefulLeadSec": f"{buffered_useful_lead_sec:.9g}",
            })
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or op == "D"):
            o = oracle[i - 1]
            deletions += 1
            align_rows.append({
                "AlignmentOp": "deletion",
                "OracleIndex": str(o["index"]),
                "StreamingIndex": "",
                "OracleClass": str(o["class"]),
                "StreamingClass": "",
                "OracleTimeSec": f"{float(o['time']):.9g}",
                "StreamingTimeSec": "",
                "StreamingAvailableAtSec": "",
                "TimingDeltaMs": "",
                "AcousticLeadSec": "",
                "BufferedUsefulLeadSec": "",
            })
            i -= 1
        else:
            st = streaming[j - 1]
            insertions += 1
            align_rows.append({
                "AlignmentOp": "insertion",
                "OracleIndex": "",
                "StreamingIndex": str(st["index"]),
                "OracleClass": "",
                "StreamingClass": str(st["class"]),
                "OracleTimeSec": "",
                "StreamingTimeSec": f"{float(st['time']):.9g}",
                "StreamingAvailableAtSec": f"{float(st.get('available', 0.0)):.9g}",
                "TimingDeltaMs": "",
                "AcousticLeadSec": "",
                "BufferedUsefulLeadSec": "",
            })
            j -= 1
    align_rows.reverse()

    aligned_same = same
    aligned_useful = same + compatible
    edit_distance = substitutions + insertions + deletions
    summary = {
        "AU34OracleSequenceCount": str(n),
        "AU34StreamingSequenceCount": str(m),
        "AU34SameClassAlignedCount": str(same),
        "AU34CompatibleAlignedCount": str(compatible),
        "AU34SubstitutionCount": str(substitutions),
        "AU34InsertionCount": str(insertions),
        "AU34DeletionCount": str(deletions),
        "AU34EditDistance": str(edit_distance),
        "AU34SequenceRecall": f"{aligned_same / n:.9g}" if n else "",
        "AU34SequencePrecision": f"{aligned_same / m:.9g}" if m else "",
        "AU34CompatibleSequenceRecall": f"{aligned_useful / n:.9g}" if n else "",
        "AU34CompatibleSequencePrecision": f"{aligned_useful / m:.9g}" if m else "",
        "AU34TimingMAEMs": f"{mean(timing_abs_ms):.9g}" if timing_abs_ms else "",
        "AU34TimingBiasMs": f"{mean(timing_signed_ms):.9g}" if timing_signed_ms else "",
        "AU34TimingP95Ms": f"{_simple_percentile(timing_abs_ms, 95):.9g}" if timing_abs_ms else "",
        "AU34MeanAcousticLeadSec": f"{mean(acoustic_lead_sec_values):.9g}" if acoustic_lead_sec_values else "",
        "AU34MeanBufferedUsefulLeadSec": f"{mean(buffered_useful_lead_sec_values):.9g}" if buffered_useful_lead_sec_values else "",
        "AU34AcousticLateCount": str(acoustic_late),
        "AU34UsefulLateCount": str(useful_late),
        "AU34NormalizedEditDistance": f"{edit_distance / max(n, m):.9g}" if max(n, m) else "",
        "AU34AlignmentScore": f"{dp[n][m]:.9g}",
    }
    return summary, align_rows

def augment_with_au32_landmark_detector_diagnostics(case_dir: Path, row: dict[str, str]) -> None:
    """Expose AU32 speech/phoneme split diagnostics.

    AU32 deliberately removes the previous causal-retime diagnostic layer. The
    runtime/scored track stays streaming-causal; full WAV evidence is used only
    as a landmark oracle for detector scoring.
    """
    effective_path = case_dir / "streaming_effective_commit_events.csv"
    runtime_path = case_dir / "runtime_commit_events.csv"
    streaming_path = case_dir / "streaming_landmark_candidates.csv"
    oracle_path = case_dir / "offline_landmark_oracle.csv"
    score_path = case_dir / "landmark_detector_score.csv"
    matches_path = case_dir / "planned_landmark_candidate_matches.csv"

    effective = read_csv_rows(effective_path)
    runtime = read_csv_rows(runtime_path)
    streaming = read_csv_rows(streaming_path)
    oracle = read_csv_rows(oracle_path)
    score_rows = read_csv_rows(score_path)
    matches = read_csv_rows(matches_path)

    row["AU32StreamingEffectiveTrackPresent"] = "1" if effective_path.exists() else "0"
    row["AU32StreamingLandmarkCandidatesPresent"] = "1" if streaming_path.exists() else "0"
    row["AU32OfflineLandmarkOracleAvailable"] = "1" if oracle_path.exists() else "0"
    row["AU32LandmarkDetectorScorePresent"] = "1" if score_path.exists() else "0"
    row["AU32PlannedLandmarkMatchesPresent"] = "1" if matches_path.exists() else "0"

    if not effective and runtime:
        effective = runtime
        row["AU32StreamingEffectiveTrackFallbackRuntime"] = "1"
    else:
        row["AU32StreamingEffectiveTrackFallbackRuntime"] = "0"

    row["AU32StreamingCandidateCount"] = str(len(streaming))
    row["AU32OracleLandmarkCount"] = str(len(oracle))

    preroll = parse_float(row.get("PrerollSec", ""))
    if preroll is None or not math.isfinite(preroll) or preroll <= 0.0:
        preroll = 0.150
    oracle_seq = _read_landmark_sequence(oracle_path)
    streaming_seq = _read_landmark_sequence(streaming_path)
    sequence_summary, _ = _align_landmark_sequences(oracle_seq, streaming_seq, preroll)
    row.update(sequence_summary)
    if score_rows:
        score = score_rows[0]
        for src, dst in [
            ("MatchedCount", "AU32MatchedCount"),
            ("Precision", "AU32Precision"),
            ("Recall", "AU32Recall"),
            ("TimingMAEMs", "AU32TimingMAEMs"),
            ("TimingP95Ms", "AU32TimingP95Ms"),
            ("AcousticLateCount", "AU32AcousticLateCount"),
            ("UsefulLateCount", "AU32UsefulLateCount"),
            ("WrongClassCount", "AU32WrongClassCount"),
            ("MissedStrongCount", "AU32MissedStrongCount"),
            ("MeanAcousticLeadSec", "AU32MeanAcousticLeadSec"),
            ("MeanBufferedUsefulLeadSec", "AU32MeanBufferedUsefulLeadSec"),
            ("UsefulLeadGE25msCount", "AU32UsefulLeadGE25msCount"),
            ("UsefulLeadGE50msCount", "AU32UsefulLeadGE50msCount"),
            ("UsefulLeadGE75msCount", "AU32UsefulLeadGE75msCount"),
            ("UsefulLeadGE100msCount", "AU32UsefulLeadGE100msCount"),
        ]:
            row[dst] = score.get(src, "")

    if matches:
        matched = [r for r in matches if truthy_csv(r.get("Matched", ""))]
        committed_matched = [r for r in matches if truthy_csv(r.get("MatchedByCommittedCenter", ""))]
        row["AU32PlannedLandmarkRows"] = str(len(matches))
        row["AU32PlannedLandmarkMatchedCount"] = str(len(matched))
        row["AU32PlannedLandmarkMatchRate"] = f"{len(matched) / len(matches):.9g}" if matches else ""
        row["AU33CommittedTimelineMatchedCount"] = str(len(committed_matched))
        row["AU33CommittedTimelineMatchRate"] = f"{len(committed_matched) / len(matches):.9g}" if matches else ""
        deltas = _float_values(matched, "DeltaMs", abs_value=True)
        if deltas:
            row["AU32PlannedLandmarkDeltaMAEMs"] = f"{mean(deltas):.9g}"
            row["AU32PlannedLandmarkDeltaMaxMs"] = f"{max(deltas):.9g}"
        committed_deltas = _float_values(committed_matched, "CommittedDeltaMs", abs_value=True)
        if committed_deltas:
            row["AU33CommittedTimelineDeltaMAEMs"] = f"{mean(committed_deltas):.9g}"
            row["AU33CommittedTimelineDeltaMaxMs"] = f"{max(committed_deltas):.9g}"

    if effective:
        row["AU32StreamingEffectiveEventCount"] = str(len(effective))
    if runtime and effective:
        changed, mae, maxdiff = _track_difference_stats_ms(runtime, effective)
        row["AU32RuntimeVsStreamingEffectiveChangedEventCount"] = str(changed)
        row["AU32RuntimeVsStreamingEffectiveMAEMs"] = f"{mae:.9g}"
        row["AU32RuntimeVsStreamingEffectiveMaxMs"] = f"{maxdiff:.9g}"

    validation_errors: list[str] = []
    if not effective_path.exists():
        validation_errors.append("missing_streaming_effective_commit_events")
    if not streaming_path.exists():
        validation_errors.append("missing_streaming_landmark_candidates")
    if not oracle_path.exists():
        validation_errors.append("missing_offline_landmark_oracle")
    if not score_path.exists():
        validation_errors.append("missing_landmark_detector_score")
    if not matches_path.exists():
        validation_errors.append("missing_planned_landmark_candidate_matches")
    row["AU32LandmarkValidationErrorCount"] = str(len(validation_errors))
    row["AU32LandmarkValidationErrors"] = ";".join(validation_errors)


def derive_speech_duration_attribution(case_dir: Path, row: dict[str, str]) -> dict[str, str]:
    """Attribute each case's duration budget into speech, punctuation, detector gaps, and outer silence.

    This is intentionally a diagnostic aggregation over already-emitted runner CSVs. It does
    not change runtime behavior and it does not treat the self-referential mapped-audio
    duration fields as ground truth.
    """
    audio = parse_float(row.get("AudioDurationSec", ""))
    speech_islands = read_csv_rows(case_dir / "speech_islands.csv")
    island_alignment = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    soft_punctuation = read_csv_rows(case_dir / "soft_punctuation_boundary_diagnostics.csv")

    out: dict[str, str] = {}
    if audio is None or not math.isfinite(audio) or audio <= 0.0:
        return out

    regions: list[tuple[float, float, float]] = []
    for speech_row in speech_islands:
        start = parse_float(speech_row.get("AudioBufferStartSec", ""))
        last = parse_float(speech_row.get("AudioBufferLastSpeechSec", ""))
        end = speech_island_end(speech_row)
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
            continue
        if last is None or not math.isfinite(last):
            last = end
        last = max(start, min(last, end))
        end = max(start, end)
        regions.append((start, last, end))
    regions.sort(key=lambda item: item[0])

    detector_region_count = len(regions)
    detector_speech_material_sec = sum(max(0.0, last - start) for start, last, _end in regions)
    detector_island_span_sec = sum(max(0.0, end - start) for start, _last, end in regions)
    detector_tail_silence_sec = sum(max(0.0, end - last) for _start, last, end in regions)
    detector_internal_gap_sec = 0.0
    for prev, curr in zip(regions, regions[1:]):
        detector_internal_gap_sec += max(0.0, curr[0] - prev[2])
    leading_silence_sec = max(0.0, regions[0][0]) if regions else audio
    trailing_silence_sec = max(0.0, audio - regions[-1][2]) if regions else 0.0
    active_detector_span_sec = max(0.0, regions[-1][2] - regions[0][0]) if regions else 0.0
    unassigned_audio_sec = max(0.0, audio - (detector_island_span_sec + detector_internal_gap_sec + leading_silence_sec + trailing_silence_sec))

    planner_duration_sec = sum_float(island_alignment, "PlannerDurationSec")
    planner_speech_sec = sum_float(island_alignment, "PlannerSpeechMaterialSec")
    planner_punctuation_sec = sum_float(island_alignment, "PlannerPunctuationSec")
    planned_soft_pocket_sec = sum_float(soft_punctuation, "PlannerPocketGapSec")
    detected_soft_render_gap_sec = positive_sum_float(soft_punctuation, "EvidenceRenderGapSec")
    detected_soft_center_gap_sec = positive_sum_float(soft_punctuation, "EvidenceCenterGapSec")

    speech_error_sec = planner_speech_sec - detector_speech_material_sec
    speech_abs_error_sec = abs(speech_error_sec)
    duration_error_sec = planner_duration_sec - audio

    def fmt(v: float) -> str:
        return f"{v:.9g}"

    out.update({
        "SpeechAttributionRegionCount": str(detector_region_count),
        "SpeechAttributionAudioDurationSec": fmt(audio),
        "SpeechAttributionDetectedSpeechMaterialSec": fmt(detector_speech_material_sec),
        "SpeechAttributionDetectedIslandSpanSec": fmt(detector_island_span_sec),
        "SpeechAttributionDetectorTailSilenceSec": fmt(detector_tail_silence_sec),
        "SpeechAttributionDetectorInternalGapSec": fmt(detector_internal_gap_sec),
        "SpeechAttributionLeadingSilenceSec": fmt(leading_silence_sec),
        "SpeechAttributionTrailingSilenceSec": fmt(trailing_silence_sec),
        "SpeechAttributionActiveDetectorSpanSec": fmt(active_detector_span_sec),
        "SpeechAttributionUnassignedAudioSec": fmt(unassigned_audio_sec),
        "SpeechAttributionPlannerDurationSec": fmt(planner_duration_sec),
        "SpeechAttributionPlannerSpeechMaterialSec": fmt(planner_speech_sec),
        "SpeechAttributionPlannerPunctuationSec": fmt(planner_punctuation_sec),
        "SpeechAttributionPlannedSoftPocketSec": fmt(planned_soft_pocket_sec),
        "SpeechAttributionDetectedSoftRenderGapSec": fmt(detected_soft_render_gap_sec),
        "SpeechAttributionDetectedSoftCenterGapSec": fmt(detected_soft_center_gap_sec),
        "SpeechAttributionPlannerSpeechMinusDetectedSpeechSec": fmt(speech_error_sec),
        "SpeechAttributionPlannerSpeechAbsErrorSec": fmt(speech_abs_error_sec),
        "SpeechAttributionPlannerDurationMinusAudioSec": fmt(duration_error_sec),
        "SpeechAttributionPlannerSpeechToDetectedSpeechRatio": fmt(planner_speech_sec / detector_speech_material_sec) if detector_speech_material_sec > 1e-9 else "",
        "SpeechAttributionPlannerDurationToAudioRatio": fmt(planner_duration_sec / audio),
        "SpeechAttributionDetectedSpeechToAudioRatio": fmt(detector_speech_material_sec / audio),
        "SpeechAttributionDetectedIslandSpanToAudioRatio": fmt(detector_island_span_sec / audio),
        "SpeechAttributionDetectedGapsAndSilenceToAudioRatio": fmt(max(0.0, audio - detector_speech_material_sec) / audio),
        "SpeechAttributionPlannedPunctuationToPlannerDurationRatio": fmt(planner_punctuation_sec / planner_duration_sec) if planner_duration_sec > 1e-9 else "",
    })
    return out


def augment_with_per_case_csv_metrics(case_dir: Path, row: dict[str, str]) -> None:
    """Add metrics that are easier to derive from the detailed per-case CSVs.

    The runner intentionally keeps the raw CSVs verbose. These derived fields make
    run-to-run branch comparisons cheap without reparsing the full corpus by hand.
    Missing debug CSVs are tolerated so summary generation still works when
    write_debug_csv=false; unavailable fields are left blank.
    """
    planned = read_csv_rows(case_dir / "planned_events.csv")
    submissions = read_csv_rows(case_dir / "event_submission_summary.csv")
    launches = read_csv_rows(case_dir / "island_launch_diagnostics.csv")
    speech_islands = read_csv_rows(case_dir / "speech_islands.csv")
    nudges = read_csv_rows(case_dir / "nudge_diagnostics.csv")
    alpha32_effectiveness = read_csv_rows(case_dir / "alpha32_nudge_effectiveness.csv")
    alpha31_opportunities = read_csv_rows(case_dir / "alpha31_nudge_opportunities.csv")
    au23_frontier = read_csv_rows(case_dir / "au23_viseme_anchor_frontier.csv")
    au26_warp = read_csv_rows(case_dir / "au26_piecewise_timeline_warp.csv")
    island_alignment = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    soft_punctuation = read_csv_rows(case_dir / "soft_punctuation_boundary_diagnostics.csv")
    au40_metrics = read_scorecard(case_dir / "au40_fragment_prosody_metrics.csv") if (case_dir / "au40_fragment_prosody_metrics.csv").exists() else {}
    if au40_metrics:
        row.update(au40_metrics)

    au39_metrics = read_scorecard(case_dir / "au39_runtime_topology_metrics.csv") if (case_dir / "au39_runtime_topology_metrics.csv").exists() else {}
    if au39_metrics:
        row.update(au39_metrics)

    au41_metrics = read_scorecard(case_dir / "au41_no_fragment_runtime_metrics.csv") if (case_dir / "au41_no_fragment_runtime_metrics.csv").exists() else {}
    if au41_metrics:
        row.update(au41_metrics)

    if planned:
        row["PlannedEventCountFromCsv"] = str(len(planned))
        row["TextPhraseCount"] = str(unique_count(planned, "PhraseIndex"))

    if submissions:
        submitted = sum(1 for r in submissions if not truthy_csv(r.get("NeverSubmitted", "")))
        never = sum(1 for r in submissions if truthy_csv(r.get("NeverSubmitted", "")))
        weak = sum(1 for r in submissions if truthy_csv(r.get("WeakPeak", "")))
        row["SubmittedEventCount"] = str(submitted)
        row["NeverSubmittedEventCount"] = str(never)
        row["WeakPoseCount"] = str(weak)

    if launches:
        fallback_rows = [r for r in launches if r.get("LaunchReason", "") == "fallback_or_unmapped"]
        row["TextIslandCount"] = str(unique_count(launches, "TextIslandIndex"))
        row["FallbackIslandCount"] = str(len(fallback_rows))
        row["UnmappedEventCount"] = f"{sum_float(fallback_rows, 'EventCount'):.0f}"
        total_launch_events = sum_float(launches, "EventCount")
        fallback_launch_events = sum_float(fallback_rows, "EventCount")
        row["FallbackIslandRate"] = f"{len(fallback_rows) / len(launches):.9g}" if launches else ""
        row["UnmappedEventRate"] = f"{fallback_launch_events / total_launch_events:.9g}" if total_launch_events > 0 else ""

    if speech_islands:
        row["AudioIslandCount"] = str(len(speech_islands))

    if nudges:
        row["NudgeDiagnosticEventCount"] = str(len(nudges))
        proposed = sum(1 for r in nudges if truthy_csv(r.get("NudgeProposed", "")))
        accepted = sum(1 for r in nudges if truthy_csv(r.get("NudgeAccepted", "")))
        row["ProposedNudgeCount"] = str(proposed)
        row["AcceptedNudgeCount"] = str(accepted)
        row["NudgeProposalRate"] = f"{proposed / len(nudges):.9g}" if nudges else ""
        row["NudgeAcceptanceRate"] = f"{accepted / len(nudges):.9g}" if nudges else ""
        row["AcceptedAmongProposedNudgeRate"] = f"{accepted / proposed:.9g}" if proposed > 0 else ""

        landmark_groups = {
            "MBP": lambda r: r.get("LandmarkClass", "") == "MBP" or r.get("PoseID", "") == "22_MBP",
            "FV": lambda r: r.get("LandmarkClass", "") == "FV" or r.get("PoseID", "") in {"20_FV", "19_FV-Or-", "21_FV-Ee-"},
            "WOO": lambda r: r.get("LandmarkClass", "") == "WOO_GLIDE" or r.get("PoseID", "") in {"12_Ww-Oo-", "16_Ww-Ew-"},
            "SHCH": lambda r: r.get("LandmarkClass", "") == "SH_CH_J" or r.get("PoseID", "") == "14_ChJjSh",
        }
        for name, pred in landmark_groups.items():
            rows_for_group = [r for r in nudges if pred(r)]
            accepted_for_group = sum(1 for r in rows_for_group if truthy_csv(r.get("NudgeAccepted", "")))
            row[f"{name}Count"] = str(len(rows_for_group))
            row[f"{name}AcceptedNudgeCount"] = str(accepted_for_group)
            row[f"{name}AcceptedNudgeRate"] = f"{accepted_for_group / len(rows_for_group):.9g}" if rows_for_group else ""

        def write_alpha32_effectiveness_metrics(prefix: str, rows_for_metric):
            target_rows = []
            for r in rows_for_metric:
                target = parse_float(r.get("TargetVisualCenterSec", ""))
                conf = parse_float(r.get("OfflineEvidenceConfidence", ""))
                required = parse_float(r.get("NudgeRequiredConfidence", ""))
                if target is None or conf is None or required is None:
                    continue
                if not all(math.isfinite(v) for v in [target, conf, required]):
                    continue
                if target <= 0.0 or required <= 0.0 or conf < required:
                    continue
                target_rows.append(r)

            row[f"{prefix}TargetLandmarkCount"] = str(len(target_rows))
            if not target_rows:
                return

            accepted_rows = [r for r in target_rows if truthy_csv(r.get("NudgeAccepted", ""))]
            partial_rows = []
            clip_ratios = []
            accepted_abs_shift = []

            def finite_values(key: str, abs_value: bool = False):
                vals = []
                for r in target_rows:
                    v = parse_float(r.get(key, ""))
                    if v is None or not math.isfinite(v):
                        continue
                    vals.append(abs(v) if abs_value else v)
                return vals

            pre_center_abs = finite_values("PreCommittedToTargetErrorMs", True)
            post_center_abs = finite_values("PostCommittedToTargetErrorMs", True)
            pre_face_abs = []
            post_face_abs = []
            face_improvements = []
            for r in target_rows:
                pre_face = parse_float(r.get("PreFaceToTargetErrorMs", ""))
                post_face = parse_float(r.get("PostFaceToTargetErrorMs", ""))
                pre_peak = parse_float(r.get("PreFacePeakSec", ""))
                post_peak = parse_float(r.get("PostFacePeakSec", ""))
                if pre_face is None or post_face is None or pre_peak is None or post_peak is None:
                    continue
                if not all(math.isfinite(v) for v in [pre_face, post_face, pre_peak, post_peak]):
                    continue
                if pre_peak < 0.0 or post_peak < 0.0:
                    continue
                pre_face_abs.append(abs(pre_face))
                post_face_abs.append(abs(post_face))
                face_improvements.append(abs(pre_face) - abs(post_face))

            for r in accepted_rows:
                applied_shift = parse_float(r.get("AppliedShiftMs", ""))
                raw_shift = parse_float(r.get("RawShiftMs", ""))
                if applied_shift is not None and math.isfinite(applied_shift):
                    accepted_abs_shift.append(abs(applied_shift))
                if raw_shift is not None and applied_shift is not None and math.isfinite(raw_shift) and math.isfinite(applied_shift) and abs(raw_shift) > 0.5:
                    clip_ratios.append(min(abs(applied_shift) / abs(raw_shift), 1.0))
                    if abs(applied_shift) + 0.5 < abs(raw_shift):
                        partial_rows.append(r)

            row[f"{prefix}AcceptedTargetNudgeCount"] = str(len(accepted_rows))
            row[f"{prefix}PartialClipCount"] = str(len(partial_rows))
            if pre_center_abs:
                row[f"{prefix}ScheduledTargetMAEMs"] = f"{mean(pre_center_abs):.9g}"
            if post_center_abs:
                row[f"{prefix}FinalTargetMAEMs"] = f"{mean(post_center_abs):.9g}"
            if pre_center_abs and post_center_abs:
                row[f"{prefix}TargetMAEImprovementMs"] = f"{mean(pre_center_abs) - mean(post_center_abs):.9g}"
            if pre_face_abs:
                row[f"{prefix}PreFaceTargetMAEMs"] = f"{mean(pre_face_abs):.9g}"
            if post_face_abs:
                row[f"{prefix}FaceTargetMAEMs"] = f"{mean(post_face_abs):.9g}"
                row[f"{prefix}PostFaceTargetMAEMs"] = f"{mean(post_face_abs):.9g}"
            if face_improvements:
                row[f"{prefix}FaceTargetMAEImprovementMs"] = f"{mean(face_improvements):.9g}"
            if accepted_abs_shift:
                row[f"{prefix}AcceptedShiftAbsMeanMs"] = f"{mean(accepted_abs_shift):.9g}"
                row[f"{prefix}AcceptedShiftAbsMaxMs"] = f"{max(accepted_abs_shift):.9g}"
            if clip_ratios:
                row[f"{prefix}ClipUtilizationMean"] = f"{mean(clip_ratios):.9g}"

        def alpha32_group_predicate(name: str):
            if name == "MBP":
                return lambda r: r.get("LandmarkClass", "") == "MBP" or r.get("PoseID", "") == "22_MBP"
            if name == "FV":
                return lambda r: r.get("LandmarkClass", "") == "FV" or r.get("PoseID", "") in {"20_FV", "19_FV-Or-", "21_FV-Ee-"}
            if name == "WOO":
                return lambda r: r.get("LandmarkClass", "") == "WOO_GLIDE" or r.get("PoseID", "") in {"12_Ww-Oo-", "16_Ww-Ew-"}
            return lambda r: r.get("LandmarkClass", "") == "ROUNDED_VOWEL" or r.get("PoseID", "") in {"11_Oo", "10_Or"}

        if alpha32_effectiveness:
            write_alpha32_effectiveness_metrics("Alpha32", alpha32_effectiveness)
            for prefix, name in [("Alpha32MBP", "MBP"), ("Alpha32FV", "FV"), ("Alpha32WOO", "WOO"), ("Alpha32Rounded", "Rounded")]:
                pred = alpha32_group_predicate(name)
                write_alpha32_effectiveness_metrics(prefix, [r for r in alpha32_effectiveness if pred(r)])
        else:
            submissions_by_event = {r.get("EventIndex", "").strip(): r for r in submissions}

            def alpha32_target_rows(src_rows):
                out = []
                for r in src_rows:
                    if not truthy_csv(r.get("NudgeEligible", "")):
                        continue
                    raw_target = parse_float(r.get("NudgeCandidateRawCenterSec", ""))
                    scheduled = parse_float(r.get("NudgeScheduledCenterSec", ""))
                    final = parse_float(r.get("FinalCenterSec", ""))
                    conf = parse_float(r.get("NudgeCandidateConfidence", ""))
                    required = parse_float(r.get("NudgeRequiredConfidence", ""))
                    if raw_target is None or scheduled is None or final is None:
                        continue
                    if not all(math.isfinite(v) for v in [raw_target, scheduled, final]):
                        continue
                    if raw_target <= 0.0:
                        continue
                    if conf is None or required is None or conf < required:
                        continue
                    event_key = r.get("EventIndex", "").strip()
                    sub = submissions_by_event.get(event_key, {})
                    face_peak = parse_float(sub.get("PeakPlaybackSec", ""))
                    out.append((r, scheduled, final, raw_target, face_peak))
                return out

            def write_alpha32_metrics(prefix: str, rows_for_metric):
                target_rows = alpha32_target_rows(rows_for_metric)
                row[f"{prefix}TargetLandmarkCount"] = str(len(target_rows))
                if not target_rows:
                    return
                accepted_rows = [t for t in target_rows if truthy_csv(t[0].get("NudgeAccepted", ""))]
                partial_rows = []
                clip_ratios = []
                scheduled_abs = []
                final_abs = []
                face_abs = []
                accepted_abs_shift = []
                for r, scheduled, final, raw_target, face_peak in target_rows:
                    scheduled_abs.append(abs((scheduled - raw_target) * 1000.0))
                    final_abs.append(abs((final - raw_target) * 1000.0))
                    if face_peak is not None and math.isfinite(face_peak) and face_peak >= 0.0:
                        face_abs.append(abs((face_peak - raw_target) * 1000.0))
                    raw_shift = parse_float(r.get("NudgeCandidateRawShiftMs", ""))
                    applied_shift = parse_float(r.get("NudgeCandidateAppliedShiftMs", ""))
                    if truthy_csv(r.get("NudgeAccepted", "")) and applied_shift is not None and math.isfinite(applied_shift):
                        accepted_abs_shift.append(abs(applied_shift))
                        if raw_shift is not None and math.isfinite(raw_shift) and abs(raw_shift) > 0.5:
                            clip_ratios.append(min(abs(applied_shift) / abs(raw_shift), 1.0))
                            if abs(applied_shift) + 0.5 < abs(raw_shift):
                                partial_rows.append(r)
                row[f"{prefix}AcceptedTargetNudgeCount"] = str(len(accepted_rows))
                row[f"{prefix}PartialClipCount"] = str(len(partial_rows))
                row[f"{prefix}ScheduledTargetMAEMs"] = f"{mean(scheduled_abs):.9g}"
                row[f"{prefix}FinalTargetMAEMs"] = f"{mean(final_abs):.9g}"
                row[f"{prefix}TargetMAEImprovementMs"] = f"{mean(scheduled_abs) - mean(final_abs):.9g}"
                if face_abs:
                    row[f"{prefix}FaceTargetMAEMs"] = f"{mean(face_abs):.9g}"
                    row[f"{prefix}PostFaceTargetMAEMs"] = f"{mean(face_abs):.9g}"
                if accepted_abs_shift:
                    row[f"{prefix}AcceptedShiftAbsMeanMs"] = f"{mean(accepted_abs_shift):.9g}"
                    row[f"{prefix}AcceptedShiftAbsMaxMs"] = f"{max(accepted_abs_shift):.9g}"
                if clip_ratios:
                    row[f"{prefix}ClipUtilizationMean"] = f"{mean(clip_ratios):.9g}"

            write_alpha32_metrics("Alpha32", nudges)
            alpha32_groups = {
                "Alpha32MBP": lambda r: r.get("LandmarkClass", "") == "MBP" or r.get("PoseID", "") == "22_MBP",
                "Alpha32FV": lambda r: r.get("LandmarkClass", "") == "FV" or r.get("PoseID", "") in {"20_FV", "19_FV-Or-", "21_FV-Ee-"},
                "Alpha32WOO": lambda r: r.get("LandmarkClass", "") == "WOO_GLIDE" or r.get("PoseID", "") in {"12_Ww-Oo-", "16_Ww-Ew-"},
                "Alpha32Rounded": lambda r: r.get("LandmarkClass", "") == "ROUNDED_VOWEL" or r.get("PoseID", "") in {"11_Oo", "10_Or"},
            }
            for prefix, pred in alpha32_groups.items():
                write_alpha32_metrics(prefix, [r for r in nudges if pred(r)])


    if au23_frontier:
        row["AU23AnchorFrontierEventCount"] = str(len(au23_frontier))
        strong_frontier = [r for r in au23_frontier if truthy_csv(r.get("StrongForClass", ""))]
        proposed_new = [r for r in au23_frontier if truthy_csv(r.get("ProposedNewHardAnchorClass", ""))]
        row["AU23AnchorFrontierStrongCount"] = str(len(strong_frontier))
        row["AU23CandidateNewHardAnchorCount"] = str(len(proposed_new))
        for cls in ["MBP", "FV", "WOO_GLIDE", "ROUNDED_VOWEL", "SH_CH_J", "FRONT_VOWEL", "OPEN_VOWEL", "TH_WEAK"]:
            cls_rows = [r for r in au23_frontier if r.get("AnchorClass", "") == cls]
            cls_strong = [r for r in cls_rows if truthy_csv(r.get("StrongForClass", ""))]
            prefix = "AU23" + cls.replace("_", "")
            row[f"{prefix}Count"] = str(len(cls_rows))
            row[f"{prefix}StrongCount"] = str(len(cls_strong))
            row[f"{prefix}StrongRate"] = f"{len(cls_strong) / len(cls_rows):.9g}" if cls_rows else ""
            err_vals = [parse_float(r.get("AbsCommittedToTargetErrorMs", "")) for r in cls_strong]
            err_vals = [v for v in err_vals if v is not None and math.isfinite(v)]
            if err_vals:
                row[f"{prefix}StrongAbsErrorMeanMs"] = f"{mean(err_vals):.9g}"
                row[f"{prefix}StrongAbsErrorMedianMs"] = f"{median(err_vals):.9g}"


    if au26_warp:
        row["AU26SharedRetimingInvoked"] = "1"
        row["AU26WarpRowCount"] = str(len(au26_warp))

        def au26_is_anchor(r: dict[str, str]) -> bool:
            # AU26 shared/runtime CSVs write IsAnchor. Older intermediate
            # diagnostic drafts used bAnchor/Anchor. Accept all three so summary
            # metrics remain stable across branch checkpoints.
            return truthy_csv(r.get("IsAnchor", r.get("bAnchor", r.get("Anchor", ""))))

        anchor_rows = [r for r in au26_warp if au26_is_anchor(r)]
        non_start_anchor_rows = [r for r in anchor_rows if (r.get("AnchorClass", "") or "") != "ISLAND_START"]
        hard_anchor_rows = [r for r in anchor_rows if (r.get("AnchorKind", "") or "").startswith("hard")]
        soft_anchor_rows = [r for r in anchor_rows if (r.get("AnchorKind", "") or "").startswith("soft")]
        moved_rows = []
        shifts = []
        anchor_shifts = []
        non_anchor_shifts = []
        for r in au26_warp:
            v = parse_float(r.get("WarpShiftMs", ""))
            if v is not None and math.isfinite(v):
                shifts.append(v)
                if au26_is_anchor(r):
                    anchor_shifts.append(v)
                else:
                    non_anchor_shifts.append(v)
                if abs(v) >= 0.5:
                    moved_rows.append(r)
        row["AU26AnchorCount"] = str(len(anchor_rows))
        row["AU26NonStartAnchorCount"] = str(len(non_start_anchor_rows))
        row["AU26HardAnchorCount"] = str(len(hard_anchor_rows))
        row["AU26SoftAnchorCount"] = str(len(soft_anchor_rows))
        row["AU26MovedEventCount"] = str(len(moved_rows))
        row["AU26MovedEventRate"] = f"{len(moved_rows) / len(au26_warp):.9g}" if au26_warp else ""
        if shifts:
            row["AU26WarpShiftMeanMs"] = f"{mean(shifts):.9g}"
            row["AU26WarpShiftAbsMeanMs"] = f"{mean(abs(v) for v in shifts):.9g}"
            row["AU26WarpShiftAbsMaxMs"] = f"{max(abs(v) for v in shifts):.9g}"
        if anchor_shifts:
            row["AU26AnchorShiftAbsMeanMs"] = f"{mean(abs(v) for v in anchor_shifts):.9g}"
            row["AU26AnchorShiftAbsMaxMs"] = f"{max(abs(v) for v in anchor_shifts):.9g}"
        if non_anchor_shifts:
            row["AU26NonAnchorShiftAbsMeanMs"] = f"{mean(abs(v) for v in non_anchor_shifts):.9g}"
            row["AU26NonAnchorShiftAbsMaxMs"] = f"{max(abs(v) for v in non_anchor_shifts):.9g}"
        by_class: dict[str, int] = {}
        for r in anchor_rows:
            cls = r.get("AnchorClass", "") or "unknown"
            by_class[cls] = by_class.get(cls, 0) + 1
        for cls in ["ISLAND_START", "MBP", "FV", "WOO_GLIDE", "ROUNDED_VOWEL", "FRONT_VOWEL"]:
            row["AU26Anchor" + cls.replace("_", "") + "Count"] = str(by_class.get(cls, 0))
    else:
        row["AU26SharedRetimingInvoked"] = "0"


    if alpha31_opportunities:
        row["Alpha31OpportunityCount"] = str(len(alpha31_opportunities))
        safe = [r for r in alpha31_opportunities if truthy_csv(r.get("HypotheticallySafe", ""))]
        strong = [r for r in alpha31_opportunities if truthy_csv(r.get("StrongOfflineEvidence", ""))]
        changed = [r for r in safe if abs(parse_float(r.get("ClampedOpportunityShiftMs", "")) or 0.0) >= 0.5]
        row["Alpha31StrongEvidenceCount"] = str(len(strong))
        row["Alpha31HypotheticallySafeCount"] = str(len(safe))
        row["Alpha31HypotheticallySafeRate"] = f"{len(safe) / len(alpha31_opportunities):.9g}" if alpha31_opportunities else ""
        row["Alpha31SafeChangingOpportunityCount"] = str(len(changed))
        row["Alpha31SafeChangingOpportunityRate"] = f"{len(changed) / len(alpha31_opportunities):.9g}" if alpha31_opportunities else ""
        abs_safe_shifts = [abs(parse_float(r.get("ClampedOpportunityShiftMs", "")) or 0.0) for r in changed]
        if abs_safe_shifts:
            row["Alpha31SafeShiftAbsMeanMs"] = f"{mean(abs_safe_shifts):.9g}"
            row["Alpha31SafeShiftAbsMaxMs"] = f"{max(abs_safe_shifts):.9g}"

        def alpha31_face_landmark_rows(src_rows):
            out = []
            for r in src_rows:
                if not truthy_csv(r.get("StrongOfflineEvidence", "")):
                    continue
                if parse_float(r.get("FacePeakWeight", "")) is None or (parse_float(r.get("FacePeakWeight", "")) or -1.0) < 0.0:
                    continue
                face_err = parse_float(r.get("FacePeakToTargetErrorMs", ""))
                commit_err = parse_float(r.get("CommittedToTargetErrorMs", ""))
                if face_err is None or commit_err is None or not math.isfinite(face_err) or not math.isfinite(commit_err):
                    continue
                out.append((r, face_err, commit_err))
            return out

        landmark_rows = alpha31_face_landmark_rows(alpha31_opportunities)
        row["Alpha31FaceLandmarkCount"] = str(len(landmark_rows))
        if landmark_rows:
            face_abs = [abs(face_err) for _, face_err, _ in landmark_rows]
            commit_abs = [abs(commit_err) for _, _, commit_err in landmark_rows]
            peak_lags = [parse_float(r.get("FacePeakMinusCommittedMs", "")) for r, _, _ in landmark_rows]
            peak_lags = [v for v in peak_lags if v is not None and math.isfinite(v)]
            row["Alpha31FaceTargetMAEMs"] = f"{mean(face_abs):.9g}"
            row["Alpha31CommittedTargetMAEMs"] = f"{mean(commit_abs):.9g}"
            row["Alpha31FaceMinusCommittedTargetMAEMs"] = f"{mean(face_abs) - mean(commit_abs):.9g}"
            if peak_lags:
                row["Alpha31FacePeakMinusCommittedMeanMs"] = f"{mean(peak_lags):.9g}"

        for name in ["MBP", "FV", "WOO_GLIDE", "ROUNDED_VOWEL"]:
            cls_rows = [r for r in alpha31_opportunities if r.get("LandmarkClass", "") == name]
            cls_safe = [r for r in cls_rows if truthy_csv(r.get("HypotheticallySafe", ""))]
            cls_changed = [r for r in cls_safe if abs(parse_float(r.get("ClampedOpportunityShiftMs", "")) or 0.0) >= 0.5]
            prefix = "Alpha31" + ("WOO" if name == "WOO_GLIDE" else "Rounded" if name == "ROUNDED_VOWEL" else name)
            row[f"{prefix}OpportunityCount"] = str(len(cls_rows))
            row[f"{prefix}SafeCount"] = str(len(cls_safe))
            row[f"{prefix}SafeRate"] = f"{len(cls_safe) / len(cls_rows):.9g}" if cls_rows else ""
            row[f"{prefix}SafeChangingCount"] = str(len(cls_changed))
            shifts = [parse_float(r.get("ClampedOpportunityShiftMs", "")) for r in cls_changed]
            shifts = [v for v in shifts if v is not None and math.isfinite(v)]
            if shifts:
                row[f"{prefix}SafeShiftMeanMs"] = f"{mean(shifts):.9g}"
                row[f"{prefix}SafeShiftAbsMeanMs"] = f"{mean(abs(v) for v in shifts):.9g}"

            cls_landmark_rows = alpha31_face_landmark_rows(cls_rows)
            row[f"{prefix}FaceLandmarkCount"] = str(len(cls_landmark_rows))
            if cls_landmark_rows:
                face_abs = [abs(face_err) for _, face_err, _ in cls_landmark_rows]
                commit_abs = [abs(commit_err) for _, _, commit_err in cls_landmark_rows]
                row[f"{prefix}FaceTargetMAEMs"] = f"{mean(face_abs):.9g}"
                row[f"{prefix}CommittedTargetMAEMs"] = f"{mean(commit_abs):.9g}"
                row[f"{prefix}FaceMinusCommittedTargetMAEMs"] = f"{mean(face_abs) - mean(commit_abs):.9g}"

    if island_alignment:
        row["RuntimeIslandAlignmentCount"] = str(len(island_alignment))
        multi = [r for r in island_alignment if (parse_int(r.get("TextIslandIndex", "")) or 0) > 0]
        row["RuntimeNonInitialIslandCount"] = str(len(multi))
        mean_errs = [parse_float(r.get("MeanStreamingAbsErrorMs", "")) for r in island_alignment]
        mean_errs = [v for v in mean_errs if v is not None and math.isfinite(v)]
        if mean_errs:
            row["RuntimeIslandMeanStreamingAbsErrorMs"] = f"{mean(mean_errs):.9g}"
            row["RuntimeIslandMaxStreamingAbsErrorMs"] = f"{max(mean_errs):.9g}"
        last_errs = [abs(v) for v in (parse_float(r.get("LastCenterErrorMs", "")) for r in island_alignment) if v is not None and math.isfinite(v)]
        if last_errs:
            row["RuntimeIslandLastCenterAbsErrorMeanMs"] = f"{mean(last_errs):.9g}"
            row["RuntimeIslandLastCenterAbsErrorMaxMs"] = f"{max(last_errs):.9g}"
        gap_deltas = [abs(v) for v in (parse_float(r.get("PrevGapDeltaCommittedVsEvidenceMs", "")) for r in multi) if v is not None and math.isfinite(v)]
        if gap_deltas:
            row["RuntimeIslandPrevGapDeltaAbsMeanMs"] = f"{mean(gap_deltas):.9g}"
            row["RuntimeIslandPrevGapDeltaAbsMaxMs"] = f"{max(gap_deltas):.9g}"


    if soft_punctuation:
        row["SoftPunctuationBoundaryCount"] = str(len(soft_punctuation))
        detector_splits = sum(1 for r in soft_punctuation if truthy_csv(r.get("DetectorSplitPresent", "")))
        fixed50 = sum(1 for r in soft_punctuation if truthy_csv(r.get("FixedPocketWithin50Ms", "")))
        fixed100 = sum(1 for r in soft_punctuation if truthy_csv(r.get("FixedPocketWithin100Ms", "")))
        committed50 = sum(1 for r in soft_punctuation if truthy_csv(r.get("CommittedPocketWithin50Ms", "")))
        committed100 = sum(1 for r in soft_punctuation if truthy_csv(r.get("CommittedPocketWithin100Ms", "")))
        row["SoftPunctuationDetectorSplitCount"] = str(detector_splits)
        row["SoftPunctuationDetectorSplitRate"] = f"{detector_splits / len(soft_punctuation):.9g}"
        row["SoftPunctuationFixedWithin50Rate"] = f"{fixed50 / len(soft_punctuation):.9g}"
        row["SoftPunctuationFixedWithin100Rate"] = f"{fixed100 / len(soft_punctuation):.9g}"
        row["SoftPunctuationCommittedWithin50Rate"] = f"{committed50 / len(soft_punctuation):.9g}"
        row["SoftPunctuationCommittedWithin100Rate"] = f"{committed100 / len(soft_punctuation):.9g}"
        for source, target in [
            ("EvidenceCenterGapSec", "SoftPunctuationEvidenceCenterGapMeanSec"),
            ("EvidenceRenderGapSec", "SoftPunctuationEvidenceRenderGapMeanSec"),
            ("PlannerPocketGapAbsErrorMs", "SoftPunctuationPlannerPocketAbsErrorMeanMs"),
            ("CommittedPocketGapAbsErrorMs", "SoftPunctuationCommittedPocketAbsErrorMeanMs"),
            ("CurrentFirstCenterAbsErrorMs", "SoftPunctuationCurrentFirstCenterAbsErrorMeanMs"),
        ]:
            values = [parse_float(r.get(source, "")) for r in soft_punctuation]
            values = [v for v in values if v is not None and math.isfinite(v)]
            if values:
                row[target] = f"{mean(values):.9g}"

    row.update(derive_speech_duration_attribution(case_dir, row))

    est = parse_float(row.get("EstimatedTextDurationSec", ""))
    audio = parse_float(row.get("AudioDurationSec", ""))
    if est is not None and audio is not None and audio > 0:
        row["DurationRatioTextToAudio"] = f"{est / audio:.9g}"

    runtime = parse_float(row.get("RuntimeCommittedDurationSec", ""))
    if runtime is not None and audio is not None and audio > 0:
        row["RuntimeCommittedDurationRatioToAudio"] = f"{runtime / audio:.9g}"

    planner_total = parse_float(row.get("RuntimePlannerIslandDurationTotalSec", ""))
    if planner_total is not None and audio is not None and audio > 0:
        row["RuntimePlannerIslandDurationRatioToAudio"] = f"{planner_total / audio:.9g}"

    raw = parse_float(row.get("RawTextTimingMAEMs", ""))
    streaming = parse_float(row.get("StreamingTimingMAEMs", ""))
    if raw is not None and streaming is not None:
        row["StreamingMinusRawMAEMs"] = f"{streaming - raw:.9g}"



def interval_overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _fmt(v: float) -> str:
    return f"{v:.9g}"


def load_detector_regions(case_dir: Path) -> list[dict[str, float | int]]:
    regions: list[dict[str, float | int]] = []
    for speech_row in read_csv_rows(case_dir / "speech_islands.csv"):
        start = parse_float(speech_row.get("AudioBufferStartSec", ""))
        last = parse_float(speech_row.get("AudioBufferLastSpeechSec", ""))
        end = speech_island_end(speech_row)
        island_index = parse_int(speech_row.get("IslandIndex", ""))
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
            continue
        if last is None or not math.isfinite(last):
            last = end
        start = max(0.0, start)
        last = max(start, min(last, end))
        end = max(start, end)
        regions.append({
            "index": island_index if island_index is not None else len(regions),
            "start": start,
            "last": last,
            "end": end,
        })
    regions.sort(key=lambda item: (float(item["start"]), int(item["index"])))
    return regions


def _runtime_island_by_audio_index(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        idx = parse_int(row.get("AudioIslandIndex", ""))
        if idx is None or idx < 0:
            continue
        out.setdefault(idx, []).append(row)
    for values in out.values():
        values.sort(key=lambda r: parse_int(r.get("TextIslandIndex", "")) if parse_int(r.get("TextIslandIndex", "")) is not None else 10**9)
    return out


def _text_boundary_by_island(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        idx = parse_int(row.get("RuntimeTextIslandIndex", ""))
        if idx is not None:
            out[idx] = row
    return out


def _matching_text_boundary_for_detector_gap(
    prev_audio_index: int,
    curr_audio_index: int,
    runtime_rows: list[dict[str, str]],
    text_boundary_rows: list[dict[str, str]],
) -> dict[str, str] | None:
    """Find the text-island boundary that most directly explains a detector gap.

    Runtime island alignment rows tell us which text island was launched on which
    detector island. For a gap between detector islands N and N+1, the most useful
    text explanation is usually the first text island mapped to the current detector
    island whose previous text island mapped to the previous detector island.
    """
    text_by_index = _text_boundary_by_island(text_boundary_rows)
    sorted_runtime = sorted(
        runtime_rows,
        key=lambda r: parse_int(r.get("TextIslandIndex", "")) if parse_int(r.get("TextIslandIndex", "")) is not None else 10**9,
    )
    prev_runtime: dict[str, str] | None = None
    for row in sorted_runtime:
        text_idx = parse_int(row.get("TextIslandIndex", ""))
        audio_idx = parse_int(row.get("AudioIslandIndex", ""))
        if text_idx is None or audio_idx is None:
            continue
        if audio_idx == curr_audio_index:
            if prev_runtime is not None:
                prev_audio = parse_int(prev_runtime.get("AudioIslandIndex", ""))
                if prev_audio == prev_audio_index:
                    boundary = text_by_index.get(text_idx)
                    if boundary:
                        return boundary
        prev_runtime = row

    # Fallback: use the first text island on the current detector island. This is
    # less exact, but still exposes whether the detector transition corresponds to
    # a visible hard text boundary or an unexplained split.
    candidates = [r for r in sorted_runtime if parse_int(r.get("AudioIslandIndex", "")) == curr_audio_index]
    if candidates:
        idx = parse_int(candidates[0].get("TextIslandIndex", ""))
        if idx is not None:
            return text_by_index.get(idx)
    return None


def _matching_soft_boundary_for_detector_gap(
    prev_audio_index: int,
    curr_audio_index: int,
    soft_rows: list[dict[str, str]],
) -> dict[str, str] | None:
    for row in soft_rows:
        if not truthy_csv(row.get("DetectorSplitPresent", "")):
            continue
        prev_idx = parse_int(row.get("PrevDetectorIslandIndex", ""))
        curr_idx = parse_int(row.get("CurrDetectorIslandIndex", ""))
        if prev_idx == prev_audio_index and curr_idx == curr_audio_index:
            return row
    return None


def classify_detector_gap(
    prev_audio_index: int,
    curr_audio_index: int,
    runtime_rows: list[dict[str, str]],
    text_boundary_rows: list[dict[str, str]],
    soft_rows: list[dict[str, str]],
) -> tuple[str, dict[str, str] | None, dict[str, str] | None]:
    soft = _matching_soft_boundary_for_detector_gap(prev_audio_index, curr_audio_index, soft_rows)
    if soft is not None:
        return "SoftPunctuationGap", None, soft

    boundary = _matching_text_boundary_for_detector_gap(prev_audio_index, curr_audio_index, runtime_rows, text_boundary_rows)
    if boundary is not None:
        if truthy_csv(boundary.get("BoundaryIsHardSentenceBreak", "")) or truthy_csv(boundary.get("BoundaryPlannerSentenceChanged", "")):
            return "HardPunctuationGap", boundary, None
        if truthy_csv(boundary.get("BoundaryPhraseChanged", "")):
            return "InterPhraseGap", boundary, None
        reasons = boundary.get("AllBoundaryReasons", "")
        if "runtime_split_without_visible_planner_boundary" in reasons:
            return "MidPhraseGap", boundary, None
        return "UnknownGap", boundary, None

    return "UnknownGap", None, None


def _gap_boundary_kind(boundary: dict[str, str] | None) -> tuple[bool, bool]:
    if boundary is None:
        return False, False
    is_hard = truthy_csv(boundary.get("BoundaryIsHardSentenceBreak", "")) or truthy_csv(boundary.get("BoundaryPlannerSentenceChanged", ""))
    is_phrase = truthy_csv(boundary.get("BoundaryPhraseChanged", ""))
    return is_hard, is_phrase


def classify_unknown_gap_forensics(
    duration_sec: float,
    boundary: dict[str, str] | None,
    prev_region: dict[str, float | int] | None,
    curr_region: dict[str, float | int] | None,
) -> dict[str, str]:
    """Provide a second-pass diagnosis for gaps not explained by known text boundaries.

    These labels are deliberately conservative. They are for corpus archaeology only;
    they should not be interpreted as a runtime policy. Short same-utterance gaps are
    most likely detector fragmentation, while longer gaps with no visible planner
    boundary are likely breath/hesitation/speaker pacing.
    """
    is_hard, is_phrase = _gap_boundary_kind(boundary)
    has_boundary = boundary is not None
    same_detector_transition = prev_region is not None and curr_region is not None
    inside_same_island = "false"
    inside_same_phrase = "false"
    nearest_boundary = "none"
    if has_boundary:
        if is_hard:
            nearest_boundary = "hard"
        elif is_phrase:
            nearest_boundary = "phrase"
        else:
            nearest_boundary = "same_phrase_or_unmodeled"
            inside_same_island = "true"
            inside_same_phrase = "true"
    elif same_detector_transition:
        nearest_boundary = "none"

    if has_boundary and is_hard:
        category = "IslandBoundaryCandidate"
    elif has_boundary and is_phrase:
        category = "PhraseBoundaryCandidate"
    elif duration_sec <= 0.120:
        category = "DetectorFragmentationLikely"
    elif duration_sec >= 0.180:
        category = "BreathingOrHesitationLikely"
    else:
        category = "OtherUnknown"

    return {
        "UnknownForensicCategory": category,
        "InsideSameIsland": inside_same_island,
        "InsideSamePhrase": inside_same_phrase,
        "NearestTextBoundaryKind": nearest_boundary,
        "UnknownGapThresholdRule": "short<=120ms;long>=180ms",
    }


def soft_punctuation_observed_pause_summary(run_dir: Path) -> dict[str, float | int]:
    """Summarize soft punctuation from boundary diagnostics, not detector splits.

    Alpha11's exclusive detector-gap taxonomy only counted soft punctuation when the
    speech detector split at the comma/soft boundary. Alpha07 showed that split rate
    is intentionally very low, so detector-gap seconds are not a valid estimate of
    observed soft-punctuation pause contribution. This summary keeps soft punctuation
    as a separate, non-exclusive observed/predicted budget.
    """
    rows = collect_soft_punctuation_boundary_rows(run_dir)
    evidence_render = positive_sum_float(rows, "EvidenceRenderGapSec")
    evidence_center = positive_sum_float(rows, "EvidenceCenterGapSec")
    planned_pocket = positive_sum_float(rows, "PlannerPocketGapSec")
    committed_gap = positive_sum_float(rows, "CommittedCenterGapSec")
    detector_split_count = sum(1 for row in rows if truthy_csv(row.get("DetectorSplitPresent", "")))
    return {
        "SoftBoundaryCount": len(rows),
        "SoftDetectorSplitCount": detector_split_count,
        "SoftObservedRenderGapSec": evidence_render,
        "SoftObservedCenterGapSec": evidence_center,
        "SoftPlannedPocketSec": planned_pocket,
        "SoftCommittedCenterGapSec": committed_gap,
    }


def derive_gap_attribution_rows(case_dir: Path, case_summary: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Classify detected pauses into text-explainable and unexplained buckets.

    This consumes existing liplab debug CSVs only. It does not rescan audio and it
    does not change runtime behavior. Each detector gap is assigned one category so
    corpus totals can answer how much non-speech time is hard punctuation, soft
    punctuation, boundary-like structure, detector fragmentation, or outer silence.
    """
    if case_summary is None:
        scorecard = case_dir / "scorecard.csv"
        case_summary = normalize_row(read_scorecard(scorecard), case_dir.name) if scorecard.exists() else {"case_id": case_dir.name}
    case_id = case_summary.get("case_id", case_dir.name)
    transcript = case_summary.get("Transcript", "")
    audio = parse_float(case_summary.get("AudioDurationSec", ""))
    if audio is None or not math.isfinite(audio) or audio < 0.0:
        audio = 0.0

    regions = load_detector_regions(case_dir)
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    text_boundary_rows = read_csv_rows(case_dir / "text_island_diagnostics.csv")
    soft_rows = read_csv_rows(case_dir / "soft_punctuation_boundary_diagnostics.csv")

    rows: list[dict[str, str]] = []

    def add_row(
        category: str,
        start: float,
        end: float,
        prev_region: dict[str, float | int] | None = None,
        curr_region: dict[str, float | int] | None = None,
        boundary: dict[str, str] | None = None,
        soft: dict[str, str] | None = None,
    ) -> None:
        duration = max(0.0, end - start)
        if duration <= 1e-9:
            return
        punctuation_kind = ""
        punctuation_char_index = ""
        primary_reason = ""
        all_reasons = ""
        prev_word = ""
        curr_word = ""
        prev_event = ""
        curr_event = ""
        text_island_index = ""
        phrase_index = ""
        if soft is not None:
            punctuation_kind = soft.get("BoundaryPunctuationKind", "")
            punctuation_char_index = soft.get("BoundaryPunctuationCharIndex", "")
            prev_word = soft.get("PrevLastWord", "")
            curr_word = soft.get("CurrFirstWord", "")
            prev_event = soft.get("PrevLastEventIndex", "")
            curr_event = soft.get("CurrFirstEventIndex", "")
            text_island_index = soft.get("TextIslandIndex", "")
            phrase_index = soft.get("PhraseIndex", "")
            primary_reason = "soft_punctuation_before_word"
            all_reasons = "soft_punctuation_before_word;detector_split_present"
        elif boundary is not None:
            punctuation_kind = boundary.get("BoundaryPunctuationKind", "")
            punctuation_char_index = boundary.get("BoundaryPunctuationCharIndex", "")
            prev_word = boundary.get("BoundaryPrevWord", "")
            curr_word = boundary.get("FirstWord", "")
            prev_event = boundary.get("BoundaryPrevEventIndex", "")
            curr_event = boundary.get("FirstEventIndex", "")
            text_island_index = boundary.get("RuntimeTextIslandIndex", "")
            phrase_index = boundary.get("FirstPhraseIndex", "")
            primary_reason = boundary.get("PrimaryBoundaryReason", "")
            all_reasons = boundary.get("AllBoundaryReasons", "")

        forensic = classify_unknown_gap_forensics(duration, boundary, prev_region, curr_region) if category == "UnknownGap" else {
            "UnknownForensicCategory": "",
            "InsideSameIsland": "",
            "InsideSamePhrase": "",
            "NearestTextBoundaryKind": "",
            "UnknownGapThresholdRule": "",
        }

        row_out = {
            "case_id": case_id,
            "GapCategory": category,
            "GapStartSec": _fmt(start),
            "GapEndSec": _fmt(end),
            "GapDurationSec": _fmt(duration),
            "GapDurationMs": _fmt(duration * 1000.0),
            "PrevDetectorIslandIndex": str(int(prev_region["index"])) if prev_region is not None else "",
            "CurrDetectorIslandIndex": str(int(curr_region["index"])) if curr_region is not None else "",
            "PrevDetectorEndSec": _fmt(float(prev_region["end"])) if prev_region is not None else "",
            "CurrDetectorStartSec": _fmt(float(curr_region["start"])) if curr_region is not None else "",
            "RuntimeTextIslandIndex": text_island_index,
            "PhraseIndex": phrase_index,
            "BoundaryPunctuationKind": punctuation_kind,
            "BoundaryPunctuationCharIndex": punctuation_char_index,
            "BoundaryPrimaryReason": primary_reason,
            "BoundaryAllReasons": all_reasons,
            "PrevWord": prev_word,
            "CurrWord": curr_word,
            "PrevEventIndex": prev_event,
            "CurrEventIndex": curr_event,
            "Transcript": transcript,
        }
        row_out.update(forensic)
        rows.append(row_out)

    if regions:
        add_row("LeadingSilence", 0.0, float(regions[0]["start"]), curr_region=regions[0])
        for prev_region, curr_region in zip(regions, regions[1:]):
            prev_index = int(prev_region["index"])
            curr_index = int(curr_region["index"])
            category, boundary, soft = classify_detector_gap(prev_index, curr_index, runtime_rows, text_boundary_rows, soft_rows)
            add_row(category, float(prev_region["end"]), float(curr_region["start"]), prev_region, curr_region, boundary, soft)
        add_row("TrailingSilence", float(regions[-1]["end"]), audio, prev_region=regions[-1])
    elif audio > 0.0:
        add_row("UnknownGap", 0.0, audio)

    return rows


def write_gap_attribution_csv(case_dir: Path, case_summary: dict[str, str]) -> None:
    rows = derive_gap_attribution_rows(case_dir, case_summary)
    if not rows:
        return
    fields = unique_in_order([key for row in rows for key in row.keys()])
    with (case_dir / "gap_attribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def collect_gap_attribution_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        path = case_dir / "gap_attribution.csv"
        if path.exists():
            case_rows = read_csv_rows(path)
        else:
            scorecard = case_dir / "scorecard.csv"
            summary = normalize_row(read_scorecard(scorecard), case_dir.name) if scorecard.exists() else {"case_id": case_dir.name}
            case_rows = derive_gap_attribution_rows(case_dir, summary)
        for row in case_rows:
            rows.append(dict(row))
    return rows


def write_corpus_gap_attribution_summaries(run_dir: Path) -> None:
    rows = collect_gap_attribution_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    detail_path = run_dir / "corpus_gap_attribution.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    total_sec = sum_float(rows, "GapDurationSec")
    category_rows: list[dict[str, str]] = []
    for category in sorted({row.get("GapCategory", "UnknownGap") or "UnknownGap" for row in rows}):
        subset = [row for row in rows if (row.get("GapCategory", "UnknownGap") or "UnknownGap") == category]
        durations = [v for v in (parse_float(row.get("GapDurationSec", "")) for row in subset) if v is not None and math.isfinite(v)]
        rec = {
            "GapCategory": category,
            "GapCount": str(len(subset)),
            "TotalGapSec": _fmt(sum(durations)),
            "TotalGapMs": _fmt(sum(durations) * 1000.0),
            "PercentOfAllGapSec": _fmt(sum(durations) / total_sec) if total_sec > 1e-9 else "",
        }
        rec.update({f"DurationSec_{k}": v for k, v in stats_for(durations).items()})
        category_rows.append(rec)

    category_fields = unique_in_order([key for row in category_rows for key in row.keys()])
    totals_path = run_dir / "corpus_gap_attribution_totals.csv"
    with totals_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=category_fields)
        w.writeheader()
        for row in sorted(category_rows, key=lambda r: -parse_float(r.get("TotalGapSec", "0"))):
            w.writerow({k: row.get(k, "") for k in category_fields})

    by_punctuation: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row.get("GapCategory", "") or "UnknownGap", row.get("BoundaryPunctuationKind", "") or "none")
        duration = parse_float(row.get("GapDurationSec", ""))
        if duration is not None and math.isfinite(duration):
            by_punctuation.setdefault(key, []).append(duration)
    punc_rows: list[dict[str, str]] = []
    for (category, punctuation), durations in by_punctuation.items():
        rec = {
            "GapCategory": category,
            "BoundaryPunctuationKind": punctuation,
            "GapCount": str(len(durations)),
            "TotalGapSec": _fmt(sum(durations)),
            "PercentOfAllGapSec": _fmt(sum(durations) / total_sec) if total_sec > 1e-9 else "",
        }
        rec.update({f"DurationSec_{k}": v for k, v in stats_for(durations).items()})
        punc_rows.append(rec)
    punc_fields = unique_in_order([key for row in punc_rows for key in row.keys()])
    punc_path = run_dir / "corpus_gap_attribution_by_punctuation.csv"
    with punc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=punc_fields)
        w.writeheader()
        for row in sorted(punc_rows, key=lambda r: -parse_float(r.get("TotalGapSec", "0"))):
            w.writerow({k: row.get(k, "") for k in punc_fields})

    text_predictable = {"HardPunctuationGap", "SoftPunctuationGap", "InterPhraseGap"}
    outer = {"LeadingSilence", "TrailingSilence"}
    text_predictable_sec = sum_float([row for row in rows if row.get("GapCategory", "") in text_predictable], "GapDurationSec")
    outer_sec = sum_float([row for row in rows if row.get("GapCategory", "") in outer], "GapDurationSec")
    unexplained_sec = sum_float([row for row in rows if row.get("GapCategory", "") not in text_predictable and row.get("GapCategory", "") not in outer], "GapDurationSec")
    hard_sec = sum_float([row for row in rows if row.get("GapCategory", "") == "HardPunctuationGap"], "GapDurationSec")
    detector_soft_sec = sum_float([row for row in rows if row.get("GapCategory", "") == "SoftPunctuationGap"], "GapDurationSec")
    inter_phrase_sec = sum_float([row for row in rows if row.get("GapCategory", "") == "InterPhraseGap"], "GapDurationSec")
    mid_phrase_sec = sum_float([row for row in rows if row.get("GapCategory", "") == "MidPhraseGap"], "GapDurationSec")
    unknown_sec = sum_float([row for row in rows if row.get("GapCategory", "") == "UnknownGap"], "GapDurationSec")
    soft_observed = soft_punctuation_observed_pause_summary(run_dir)
    soft_observed_render_sec = float(soft_observed.get("SoftObservedRenderGapSec", 0.0))
    soft_planned_sec = float(soft_observed.get("SoftPlannedPocketSec", 0.0))

    explanation_rows = [
        {"Metric": "TotalDetectorGapSec", "Value": _fmt(total_sec)},
        {"Metric": "ExplainedByHardPunctuationDetectorGapSec", "Value": _fmt(hard_sec)},
        {"Metric": "ExplainedBySoftPunctuationDetectorSplitSec", "Value": _fmt(detector_soft_sec)},
        {"Metric": "SoftPunctuationObservedRenderGapSec", "Value": _fmt(soft_observed_render_sec)},
        {"Metric": "SoftPunctuationPlannedPocketSec", "Value": _fmt(soft_planned_sec)},
        {"Metric": "SoftPunctuationObservedBoundaryCount", "Value": str(int(soft_observed.get("SoftBoundaryCount", 0)))},
        {"Metric": "SoftPunctuationDetectorSplitCount", "Value": str(int(soft_observed.get("SoftDetectorSplitCount", 0)))},
        {"Metric": "ExplainedByInterPhraseSec", "Value": _fmt(inter_phrase_sec)},
        {"Metric": "TextPredictableDetectorGapSec", "Value": _fmt(text_predictable_sec)},
        {"Metric": "TextPredictableDetectorGapRate", "Value": _fmt(text_predictable_sec / total_sec) if total_sec > 1e-9 else ""},
        {"Metric": "TextPredictableIncludingObservedSoftRenderSec", "Value": _fmt(hard_sec + inter_phrase_sec + soft_observed_render_sec)},
        {"Metric": "OuterSilenceSec", "Value": _fmt(outer_sec)},
        {"Metric": "OuterSilenceRate", "Value": _fmt(outer_sec / total_sec) if total_sec > 1e-9 else ""},
        {"Metric": "MidPhraseGapSec", "Value": _fmt(mid_phrase_sec)},
        {"Metric": "UnknownGapSec", "Value": _fmt(unknown_sec)},
        {"Metric": "UnexplainedInternalGapSec", "Value": _fmt(unexplained_sec)},
        {"Metric": "UnexplainedInternalGapRate", "Value": _fmt(unexplained_sec / total_sec) if total_sec > 1e-9 else ""},
    ]
    explanation_path = run_dir / "corpus_gap_explanation.csv"
    with explanation_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Metric", "Value"])
        w.writeheader()
        for row in explanation_rows:
            w.writerow(row)


    unknown_rows = [row for row in rows if row.get("GapCategory", "") == "UnknownGap"]
    unknown_total_sec = sum_float(unknown_rows, "GapDurationSec")
    unknown_breakdown_rows: list[dict[str, str]] = []
    for forensic_category in sorted({row.get("UnknownForensicCategory", "OtherUnknown") or "OtherUnknown" for row in unknown_rows}):
        subset = [row for row in unknown_rows if (row.get("UnknownForensicCategory", "OtherUnknown") or "OtherUnknown") == forensic_category]
        durations = [v for v in (parse_float(row.get("GapDurationSec", "")) for row in subset) if v is not None and math.isfinite(v)]
        rec = {
            "UnknownForensicCategory": forensic_category,
            "GapCount": str(len(subset)),
            "TotalGapSec": _fmt(sum(durations)),
            "PercentOfUnknownGapSec": _fmt(sum(durations) / unknown_total_sec) if unknown_total_sec > 1e-9 else "",
            "PercentOfAllDetectorGapSec": _fmt(sum(durations) / total_sec) if total_sec > 1e-9 else "",
        }
        rec.update({f"DurationSec_{k}": v for k, v in stats_for(durations).items()})
        unknown_breakdown_rows.append(rec)
    unknown_fields = unique_in_order([key for row in unknown_breakdown_rows for key in row.keys()])
    unknown_path = run_dir / "corpus_unknown_gap_breakdown.csv"
    with unknown_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=unknown_fields)
        w.writeheader()
        for row in sorted(unknown_breakdown_rows, key=lambda r: -parse_float(r.get("TotalGapSec", "0"))):
            w.writerow({k: row.get(k, "") for k in unknown_fields})

    unknown_detail_fields = unique_in_order([
        "case_id",
        "UnknownForensicCategory",
        "GapDurationSec",
        "GapStartSec",
        "GapEndSec",
        "InsideSameIsland",
        "InsideSamePhrase",
        "NearestTextBoundaryKind",
        "BoundaryPunctuationKind",
        "BoundaryPrimaryReason",
        "BoundaryAllReasons",
        "PrevWord",
        "CurrWord",
        "Transcript",
    ])
    unknown_detail_path = run_dir / "corpus_unknown_gap_forensics.csv"
    with unknown_detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=unknown_detail_fields)
        w.writeheader()
        for row in sorted(unknown_rows, key=lambda r: -(parse_float(r.get("GapDurationSec", "")) or 0.0)):
            w.writerow({k: row.get(k, "") for k in unknown_detail_fields})

    worst_fields = unique_in_order([
        "case_id",
        "GapCategory",
        "GapDurationSec",
        "GapStartSec",
        "GapEndSec",
        "BoundaryPunctuationKind",
        "BoundaryPrimaryReason",
        "UnknownForensicCategory",
        "InsideSamePhrase",
        "InsideSameIsland",
        "NearestTextBoundaryKind",
        "PrevWord",
        "CurrWord",
        "Transcript",
    ])
    worst_path = run_dir / "corpus_gap_attribution_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in sorted(rows, key=lambda r: -(parse_float(r.get("GapDurationSec", "")) or 0.0))[:100]:
            w.writerow({k: row.get(k, "") for k in worst_fields})

def write_corpus_summary(run_dir: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "case_id",
        "Transcript",
        "AudioDurationSec",
        "EstimatedTextDurationSec",
        "DurationRatioTextToAudio",
        "RuntimeCommittedDurationSec",
        "RuntimeCommittedDurationRatioToAudio",
        "RuntimePlannerIslandDurationTotalSec",
        "RuntimePlannerIslandDurationRatioToAudio",
        "RuntimePlannerIslandDurationMeanSec",
        "RuntimePlannerIslandDurationCount",
        "SpeechAttributionDetectedSpeechMaterialSec",
        "SpeechAttributionPlannerSpeechMaterialSec",
        "SpeechAttributionPlannerSpeechToDetectedSpeechRatio",
        "SpeechAttributionPlannerSpeechMinusDetectedSpeechSec",
        "SpeechAttributionDetectedSpeechToAudioRatio",
        "SpeechAttributionDetectedGapsAndSilenceToAudioRatio",
        "SpeechAttributionDetectorInternalGapSec",
        "SpeechAttributionLeadingSilenceSec",
        "SpeechAttributionTrailingSilenceSec",
        "PlannedEventCount",
        "CommittedEventCount",
        "SubmittedEventCount",
        "MissingCommittedEventCount",
        "TextIslandCount",
        "AudioIslandCount",
        "FallbackIslandCount",
        "FallbackIslandRate",
        "SoftPunctuationBoundaryCount",
        "SoftPunctuationDetectorSplitCount",
        "SoftPunctuationDetectorSplitRate",
        "SoftPunctuationFixedWithin50Rate",
        "SoftPunctuationFixedWithin100Rate",
        "SoftPunctuationCommittedWithin50Rate",
        "SoftPunctuationCommittedWithin100Rate",
        "SoftPunctuationEvidenceCenterGapMeanSec",
        "SoftPunctuationEvidenceRenderGapMeanSec",
        "SoftPunctuationPlannerPocketAbsErrorMeanMs",
        "SoftPunctuationCommittedPocketAbsErrorMeanMs",
        "SoftPunctuationCurrentFirstCenterAbsErrorMeanMs",
        "UnmappedEventCount",
        "UnmappedEventRate",
        "RawTextTimingMAEMs",
        "StreamingTimingMAEMs",
        "StreamingMinusRawMAEMs",
        "ProposedNudgeCount",
        "AcceptedNudgeCount",
        "NudgeProposalRate",
        "NudgeAcceptanceRate",
        "AcceptedAmongProposedNudgeRate",
        "AU26SharedRetimingInvoked",
        "AU26WarpRowCount",
        "AU26AnchorCount",
        "AU26NonStartAnchorCount",
        "AU26HardAnchorCount",
        "AU26SoftAnchorCount",
        "AU26MovedEventCount",
        "AU26MovedEventRate",
        "AU26WarpShiftMeanMs",
        "AU26WarpShiftAbsMeanMs",
        "AU26WarpShiftAbsMaxMs",
        "AU26AnchorShiftAbsMeanMs",
        "AU26AnchorShiftAbsMaxMs",
        "AU26NonAnchorShiftAbsMeanMs",
        "AU26NonAnchorShiftAbsMaxMs",
        "AU26AnchorISLANDSTARTCount",
        "AU26AnchorMBPCount",
        "AU26AnchorFVCount",
        "AU26AnchorWOOGLIDECount",
        "AU26AnchorROUNDEDVOWELCount",
        "AU26AnchorFRONTVOWELCount",
        "AcceptedNudgeAbsMeanMs",
        "AcceptedNudgeAbsMaxMs",
        "Alpha32TargetLandmarkCount",
        "Alpha32AcceptedTargetNudgeCount",
        "Alpha32PartialClipCount",
        "Alpha32ScheduledTargetMAEMs",
        "Alpha32FinalTargetMAEMs",
        "Alpha32TargetMAEImprovementMs",
        "Alpha32FaceTargetMAEMs",
        "Alpha32PreFaceTargetMAEMs",
        "Alpha32PostFaceTargetMAEMs",
        "Alpha32FaceTargetMAEImprovementMs",
        "Alpha32AcceptedShiftAbsMeanMs",
        "Alpha32AcceptedShiftAbsMaxMs",
        "Alpha32ClipUtilizationMean",
        "Alpha32MBPTargetLandmarkCount",
        "Alpha32MBPAcceptedTargetNudgeCount",
        "Alpha32MBPPartialClipCount",
        "Alpha32MBPScheduledTargetMAEMs",
        "Alpha32MBPFinalTargetMAEMs",
        "Alpha32MBPTargetMAEImprovementMs",
        "Alpha32MBPFaceTargetMAEMs",
        "Alpha32MBPPreFaceTargetMAEMs",
        "Alpha32MBPPostFaceTargetMAEMs",
        "Alpha32MBPFaceTargetMAEImprovementMs",
        "Alpha32MBPAcceptedShiftAbsMeanMs",
        "Alpha32MBPAcceptedShiftAbsMaxMs",
        "Alpha32MBPClipUtilizationMean",
        "Alpha32FVTargetLandmarkCount",
        "Alpha32FVAcceptedTargetNudgeCount",
        "Alpha32FVPartialClipCount",
        "Alpha32FVScheduledTargetMAEMs",
        "Alpha32FVFinalTargetMAEMs",
        "Alpha32FVTargetMAEImprovementMs",
        "Alpha32FVFaceTargetMAEMs",
        "Alpha32FVPreFaceTargetMAEMs",
        "Alpha32FVPostFaceTargetMAEMs",
        "Alpha32FVFaceTargetMAEImprovementMs",
        "Alpha32FVAcceptedShiftAbsMeanMs",
        "Alpha32FVAcceptedShiftAbsMaxMs",
        "Alpha32FVClipUtilizationMean",
        "Alpha32WOOTargetLandmarkCount",
        "Alpha32WOOAcceptedTargetNudgeCount",
        "Alpha32WOOPartialClipCount",
        "Alpha32WOOScheduledTargetMAEMs",
        "Alpha32WOOFinalTargetMAEMs",
        "Alpha32WOOTargetMAEImprovementMs",
        "Alpha32WOOFaceTargetMAEMs",
        "Alpha32WOOPreFaceTargetMAEMs",
        "Alpha32WOOPostFaceTargetMAEMs",
        "Alpha32WOOFaceTargetMAEImprovementMs",
        "Alpha32WOOAcceptedShiftAbsMeanMs",
        "Alpha32WOOAcceptedShiftAbsMaxMs",
        "Alpha32WOOClipUtilizationMean",
        "Alpha32RoundedTargetLandmarkCount",
        "Alpha32RoundedAcceptedTargetNudgeCount",
        "Alpha32RoundedPartialClipCount",
        "Alpha32RoundedScheduledTargetMAEMs",
        "Alpha32RoundedFinalTargetMAEMs",
        "Alpha32RoundedTargetMAEImprovementMs",
        "Alpha32RoundedFaceTargetMAEMs",
        "Alpha32RoundedPreFaceTargetMAEMs",
        "Alpha32RoundedPostFaceTargetMAEMs",
        "Alpha32RoundedFaceTargetMAEImprovementMs",
        "Alpha32RoundedAcceptedShiftAbsMeanMs",
        "Alpha32RoundedAcceptedShiftAbsMaxMs",
        "Alpha32RoundedClipUtilizationMean",
        "MBPCount",
        "MBPAcceptedNudgeCount",
        "MBPAcceptedNudgeRate",
        "FVCount",
        "FVAcceptedNudgeCount",
        "FVAcceptedNudgeRate",
        "WOOCount",
        "WOOAcceptedNudgeCount",
        "WOOAcceptedNudgeRate",
        "SHCHCount",
        "SHCHAcceptedNudgeCount",
        "SHCHAcceptedNudgeRate",
        "Alpha31OpportunityCount",
        "Alpha31StrongEvidenceCount",
        "Alpha31HypotheticallySafeCount",
        "Alpha31HypotheticallySafeRate",
        "Alpha31SafeChangingOpportunityCount",
        "Alpha31SafeChangingOpportunityRate",
        "Alpha31SafeShiftAbsMeanMs",
        "Alpha31SafeShiftAbsMaxMs",
        "Alpha31FaceLandmarkCount",
        "Alpha31FaceTargetMAEMs",
        "Alpha31CommittedTargetMAEMs",
        "Alpha31FaceMinusCommittedTargetMAEMs",
        "Alpha31FacePeakMinusCommittedMeanMs",
        "Alpha31MBPOpportunityCount",
        "Alpha31MBPSafeCount",
        "Alpha31MBPSafeRate",
        "Alpha31MBPSafeChangingCount",
        "Alpha31MBPSafeShiftMeanMs",
        "Alpha31MBPSafeShiftAbsMeanMs",
        "Alpha31FVOpportunityCount",
        "Alpha31FVSafeCount",
        "Alpha31FVSafeRate",
        "Alpha31FVSafeChangingCount",
        "Alpha31FVSafeShiftMeanMs",
        "Alpha31FVSafeShiftAbsMeanMs",
        "Alpha31WOOOpportunityCount",
        "Alpha31WOOSafeCount",
        "Alpha31WOOSafeRate",
        "Alpha31WOOSafeChangingCount",
        "Alpha31WOOSafeShiftMeanMs",
        "Alpha31WOOSafeShiftAbsMeanMs",
        "Alpha31RoundedOpportunityCount",
        "Alpha31RoundedSafeCount",
        "Alpha31RoundedSafeRate",
        "Alpha31RoundedSafeChangingCount",
        "Alpha31RoundedSafeShiftMeanMs",
        "Alpha31RoundedSafeShiftAbsMeanMs",
        "Alpha31MBPFaceLandmarkCount",
        "Alpha31MBPFaceTargetMAEMs",
        "Alpha31MBPCommittedTargetMAEMs",
        "Alpha31MBPFaceMinusCommittedTargetMAEMs",
        "Alpha31FVFaceLandmarkCount",
        "Alpha31FVFaceTargetMAEMs",
        "Alpha31FVCommittedTargetMAEMs",
        "Alpha31FVFaceMinusCommittedTargetMAEMs",
        "Alpha31WOOFaceLandmarkCount",
        "Alpha31WOOFaceTargetMAEMs",
        "Alpha31WOOCommittedTargetMAEMs",
        "Alpha31WOOFaceMinusCommittedTargetMAEMs",
        "Alpha31RoundedFaceLandmarkCount",
        "Alpha31RoundedFaceTargetMAEMs",
        "Alpha31RoundedCommittedTargetMAEMs",
        "Alpha31RoundedFaceMinusCommittedTargetMAEMs",
        "MaxVisiblePoseGapSec",
        "VisiblePoseGapCountOver250ms",
        "WeakPoseCount",
        "WeakPeakRate",
        "HardInvariantFailureCount",
    ]
    path = run_dir / "corpus_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def total_metric(rows: list[dict[str, str]], key: str) -> float:
    return sum(v for v in (parse_float(row.get(key, "")) for row in rows) if v is not None and math.isfinite(v))


def weighted_rate(numerator: float, denominator: float) -> str:
    return f"{numerator / denominator:.6g}" if denominator > 0 else ""


def mean_metric(rows: list[dict[str, str]], key: str) -> str:
    values = numeric_values(rows, key)
    return f"{mean(values):.6g}" if values else ""


def median_metric(rows: list[dict[str, str]], key: str) -> str:
    values = numeric_values(rows, key)
    return f"{median(values):.6g}" if values else ""


def collect_island_boundary_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        path = case_dir / "text_island_diagnostics.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            row = dict(row)
            row["case_id"] = case_dir.name
            rows.append(row)
    return rows


def write_corpus_island_boundary_summaries(run_dir: Path) -> None:
    rows = collect_island_boundary_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order(
        ["case_id"]
        + [key for row in rows for key in row.keys() if key != "case_id"]
    )
    detail_path = run_dir / "corpus_text_island_boundaries.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    counts: dict[tuple[str, str, str], dict[str, float | str]] = {}
    for row in rows:
        primary = row.get("PrimaryBoundaryReason", "") or "unknown"
        punct = row.get("BoundaryPunctuationKind", "") or "none"
        hard = row.get("BoundaryIsHardSentenceBreak", "") or "0"
        key = (primary, punct, hard)
        rec = counts.setdefault(
            key,
            {
                "PrimaryBoundaryReason": primary,
                "BoundaryPunctuationKind": punct,
                "BoundaryIsHardSentenceBreak": hard,
                "IslandCount": 0.0,
                "EventCount": 0.0,
                "OneEventIslandCount": 0.0,
            },
        )
        rec["IslandCount"] = float(rec["IslandCount"]) + 1.0
        event_count = parse_float(row.get("EventCount", "")) or 0.0
        rec["EventCount"] = float(rec["EventCount"]) + event_count
        if event_count == 1.0:
            rec["OneEventIslandCount"] = float(rec["OneEventIslandCount"]) + 1.0

    summary_path = run_dir / "corpus_text_island_boundary_summary.csv"
    fields = [
        "PrimaryBoundaryReason",
        "BoundaryPunctuationKind",
        "BoundaryIsHardSentenceBreak",
        "IslandCount",
        "EventCount",
        "OneEventIslandCount",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in sorted(counts.values(), key=lambda r: (-float(r["IslandCount"]), str(r["PrimaryBoundaryReason"]), str(r["BoundaryPunctuationKind"]))):
            out = dict(rec)
            for k in ["IslandCount", "EventCount", "OneEventIslandCount"]:
                out[k] = f"{float(out[k]):.0f}"
            w.writerow(out)




def _row_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    v = parse_float(row.get(key, ""))
    return v if v is not None and math.isfinite(v) else default


def _row_int(row: dict[str, str], key: str, default: int = -1) -> int:
    v = parse_int(row.get(key, ""))
    return v if v is not None else default


def derive_text_audio_island_assignment_rows(case_dir: Path) -> list[dict[str, str]]:
    """Diagnose whether text islands are being compared to the right detector island.

    UE alpha17b exposed a class of bugs where text island N was interpreted against
    detector fragment N even when the detector fragment was only a tiny partial breath
    or an intra-sentence fragment. This diagnostic is deliberately post-hoc: it uses
    the committed/evidence spans already written by liplab plus speech_islands.csv to
    find the detector island that best overlaps each text island's evidence span.

    It does not change runtime behavior. Its job is to flag mappings that are partial,
    fractured, missing, or likely assigned by stale index rather than temporal overlap.
    """
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    speech_rows = read_csv_rows(case_dir / "speech_islands.csv")
    if not runtime_rows:
        return []

    detectors: list[dict[str, float | int]] = []
    for ordinal, speech in enumerate(speech_rows):
        start = parse_float(speech.get("AudioBufferStartSec", ""))
        end = speech_island_end(speech)
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        idx = parse_int(speech.get("IslandIndex", ""))
        detectors.append({"index": idx if idx is not None else ordinal, "start": start, "end": end})

    def detector_by_index(index: int) -> dict[str, float | int] | None:
        for det in detectors:
            if int(det["index"]) == index:
                return det
        return None

    assigned_text_count_by_detector: dict[int, int] = {}
    for row in runtime_rows:
        idx = _row_int(row, "AudioIslandIndex")
        if idx >= 0:
            assigned_text_count_by_detector[idx] = assigned_text_count_by_detector.get(idx, 0) + 1

    rows: list[dict[str, str]] = []
    for row in runtime_rows:
        text_idx = _row_int(row, "TextIslandIndex")
        assigned_idx = _row_int(row, "AudioIslandIndex")
        evidence_start = _row_float(row, "EvidenceFirstStartSec")
        evidence_end = _row_float(row, "EvidenceLastEndSec")
        if evidence_end <= evidence_start:
            # Fall back to center span when render evidence is unavailable.
            evidence_start = _row_float(row, "EvidenceFirstCenterSec")
            evidence_end = _row_float(row, "EvidenceLastCenterSec")
        evidence_span = max(0.0, evidence_end - evidence_start)
        evidence_mid = (evidence_start + evidence_end) * 0.5 if evidence_span > 0.0 else 0.0

        assigned_det = detector_by_index(assigned_idx) if assigned_idx >= 0 else None
        assigned_start = float(assigned_det["start"]) if assigned_det is not None else _row_float(row, "DetectorAudioStartSec")
        assigned_end = float(assigned_det["end"]) if assigned_det is not None else _row_float(row, "DetectorAudioEndSec")
        assigned_span = max(0.0, assigned_end - assigned_start)
        assigned_overlap = interval_overlap_sec(evidence_start, evidence_end, assigned_start, assigned_end) if evidence_span > 0.0 and assigned_span > 0.0 else 0.0
        assigned_overlap_ratio = assigned_overlap / evidence_span if evidence_span > 1e-6 else 0.0

        best_det: dict[str, float | int] | None = None
        best_overlap = 0.0
        best_center_distance = math.inf
        overlapping_detector_count = 0
        overlapping_detector_seconds = 0.0
        for det in detectors:
            det_start = float(det["start"])
            det_end = float(det["end"])
            overlap = interval_overlap_sec(evidence_start, evidence_end, det_start, det_end) if evidence_span > 0.0 else 0.0
            if overlap > 0.020:
                overlapping_detector_count += 1
                overlapping_detector_seconds += overlap
            det_mid = (det_start + det_end) * 0.5
            center_distance = abs(det_mid - evidence_mid) if evidence_span > 0.0 else math.inf
            if overlap > best_overlap + 1e-9 or (abs(overlap - best_overlap) <= 1e-9 and center_distance < best_center_distance):
                best_det = det
                best_overlap = overlap
                best_center_distance = center_distance

        best_idx = int(best_det["index"]) if best_det is not None else -1
        best_start = float(best_det["start"]) if best_det is not None else 0.0
        best_end = float(best_det["end"]) if best_det is not None else 0.0
        best_span = max(0.0, best_end - best_start)
        best_overlap_ratio = best_overlap / evidence_span if evidence_span > 1e-6 else 0.0
        assigned_matches_best = assigned_idx == best_idx and assigned_idx >= 0

        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        mapped_span = max(0.0, mapped_end - mapped_start)
        committed_span = _row_float(row, "CommittedCenterSpanSec")
        evidence_center_span = _row_float(row, "EvidenceCenterSpanSec")
        mean_abs_error = _row_float(row, "MeanStreamingAbsErrorMs")

        notes: list[str] = []
        category = "Healthy"
        if evidence_span <= 1e-6:
            category = "NoEvidenceSpan"
            notes.append("no evidence render/center span")
        elif assigned_idx < 0:
            category = "NoAssignedDetectorIsland"
            notes.append("runtime row has no audio island index")
        elif assigned_det is None:
            category = "AssignedDetectorMissing"
            notes.append("assigned audio island index not found in speech_islands.csv")
        elif best_idx >= 0 and not assigned_matches_best and best_overlap > assigned_overlap + 0.050:
            category = "WrongDetectorIsland"
            notes.append("different detector island overlaps evidence span better")
        elif overlapping_detector_count >= 2:
            category = "DetectorFragmentedTextIsland"
            notes.append("evidence span overlaps multiple detector islands")
        elif assigned_overlap_ratio < 0.50 and best_overlap_ratio >= 0.50:
            category = "AssignedLowOverlap"
            notes.append("assigned detector has low overlap but another detector is plausible")
        elif assigned_overlap_ratio < 0.35:
            category = "LowDetectorOverlap"
            notes.append("text evidence span barely overlaps assigned detector island")
        elif assigned_span > 1e-6 and evidence_span > 1e-6:
            ratio = assigned_span / evidence_span
            if ratio < 0.55:
                category = "AssignedDetectorTooShort"
                notes.append("assigned detector island is much shorter than text evidence span")
            elif ratio > 1.90:
                category = "AssignedDetectorTooLong"
                notes.append("assigned detector island is much longer than text evidence span")

        if mean_abs_error >= 250.0:
            notes.append("high island MAE")
        if mapped_span > 1e-6 and evidence_span > 1e-6:
            mapped_ratio = mapped_span / evidence_span
            if mapped_ratio < 0.75:
                notes.append("mapped span shorter than evidence")
            elif mapped_ratio > 1.35:
                notes.append("mapped span longer than evidence")

        rows.append({
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "AssignedAudioIslandIndex": str(assigned_idx),
            "BestOverlapAudioIslandIndex": str(best_idx),
            "AssignmentCategory": category,
            "AssignedMatchesBestOverlap": "1" if assigned_matches_best else "0",
            "EventCount": row.get("EventCount", ""),
            "EvidenceEventCount": row.get("EvidenceEventCount", ""),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EvidenceStartSec": _fmt(evidence_start),
            "EvidenceEndSec": _fmt(evidence_end),
            "EvidenceSpanSec": _fmt(evidence_span),
            "AssignedDetectorStartSec": _fmt(assigned_start),
            "AssignedDetectorEndSec": _fmt(assigned_end),
            "AssignedDetectorSpanSec": _fmt(assigned_span),
            "AssignedOverlapSec": _fmt(assigned_overlap),
            "AssignedOverlapRatio": _fmt(assigned_overlap_ratio),
            "BestDetectorStartSec": _fmt(best_start),
            "BestDetectorEndSec": _fmt(best_end),
            "BestDetectorSpanSec": _fmt(best_span),
            "BestOverlapSec": _fmt(best_overlap),
            "BestOverlapRatio": _fmt(best_overlap_ratio),
            "OverlappingDetectorIslandCount": str(overlapping_detector_count),
            "OverlappingDetectorSeconds": _fmt(overlapping_detector_seconds),
            "AssignedTextIslandCountForDetector": str(assigned_text_count_by_detector.get(assigned_idx, 0)) if assigned_idx >= 0 else "0",
            "MappedAudioStartSec": _fmt(mapped_start),
            "MappedAudioEndSec": _fmt(mapped_end),
            "MappedAudioSpanSec": _fmt(mapped_span),
            "CommittedCenterSpanSec": _fmt(committed_span),
            "EvidenceCenterSpanSec": _fmt(evidence_center_span),
            "MeanStreamingAbsErrorMs": _fmt(mean_abs_error),
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
            "Notes": ";".join(notes),
        })

    assigned_detector_indices = { _row_int(row, "AudioIslandIndex") for row in runtime_rows if _row_int(row, "AudioIslandIndex") >= 0 }
    for det in detectors:
        det_idx = int(det["index"])
        if det_idx in assigned_detector_indices:
            continue
        rows.append({
            "case_id": case_dir.name,
            "TextIslandIndex": "",
            "AssignedAudioIslandIndex": "",
            "BestOverlapAudioIslandIndex": str(det_idx),
            "AssignmentCategory": "UnmatchedDetectorIsland",
            "AssignedMatchesBestOverlap": "0",
            "EventCount": "0",
            "EvidenceEventCount": "0",
            "FirstWord": "",
            "LastWord": "",
            "EvidenceStartSec": "",
            "EvidenceEndSec": "",
            "EvidenceSpanSec": "",
            "AssignedDetectorStartSec": "",
            "AssignedDetectorEndSec": "",
            "AssignedDetectorSpanSec": "",
            "AssignedOverlapSec": "",
            "AssignedOverlapRatio": "",
            "BestDetectorStartSec": _fmt(float(det["start"])),
            "BestDetectorEndSec": _fmt(float(det["end"])),
            "BestDetectorSpanSec": _fmt(max(0.0, float(det["end"]) - float(det["start"]))),
            "BestOverlapSec": "",
            "BestOverlapRatio": "",
            "OverlappingDetectorIslandCount": "",
            "OverlappingDetectorSeconds": "",
            "AssignedTextIslandCountForDetector": "0",
            "MappedAudioStartSec": "",
            "MappedAudioEndSec": "",
            "MappedAudioSpanSec": "",
            "CommittedCenterSpanSec": "",
            "EvidenceCenterSpanSec": "",
            "MeanStreamingAbsErrorMs": "",
            "FirstCenterErrorMs": "",
            "LastCenterErrorMs": "",
            "Notes": "detector island had no runtime text island assignment",
        })

    return rows


def collect_text_audio_island_assignment_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        rows.extend(derive_text_audio_island_assignment_rows(case_dir))
    return rows


def write_corpus_text_audio_island_assignment_summaries(run_dir: Path) -> None:
    rows = collect_text_audio_island_assignment_rows(run_dir)
    if not rows:
        return

    fields = unique_in_order([key for row in rows for key in row.keys()])
    detail_path = run_dir / "corpus_text_audio_island_assignment.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    categories: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get("AssignmentCategory", "") or "Unknown"
        rec = categories.setdefault(cat, {"Count": 0.0, "TotalEvidenceSpanSec": 0.0, "TotalAssignedOverlapSec": 0.0, "TotalBestOverlapSec": 0.0, "TotalMeanStreamingAbsErrorMs": 0.0, "ErrorCount": 0.0})
        rec["Count"] += 1.0
        rec["TotalEvidenceSpanSec"] += _row_float(row, "EvidenceSpanSec")
        rec["TotalAssignedOverlapSec"] += _row_float(row, "AssignedOverlapSec")
        rec["TotalBestOverlapSec"] += _row_float(row, "BestOverlapSec")
        err = parse_float(row.get("MeanStreamingAbsErrorMs", ""))
        if err is not None and math.isfinite(err):
            rec["TotalMeanStreamingAbsErrorMs"] += err
            rec["ErrorCount"] += 1.0

    summary_path = run_dir / "corpus_text_audio_island_assignment_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        summary_fields = [
            "AssignmentCategory",
            "Count",
            "TotalEvidenceSpanSec",
            "TotalAssignedOverlapSec",
            "TotalBestOverlapSec",
            "AssignedOverlapRate",
            "BestOverlapRate",
            "MeanStreamingAbsErrorMs",
        ]
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for cat, rec in sorted(categories.items(), key=lambda kv: (-kv[1]["Count"], kv[0])):
            evidence = rec["TotalEvidenceSpanSec"]
            err_count = rec["ErrorCount"]
            w.writerow({
                "AssignmentCategory": cat,
                "Count": f"{rec['Count']:.0f}",
                "TotalEvidenceSpanSec": _fmt(evidence),
                "TotalAssignedOverlapSec": _fmt(rec["TotalAssignedOverlapSec"]),
                "TotalBestOverlapSec": _fmt(rec["TotalBestOverlapSec"]),
                "AssignedOverlapRate": _fmt(rec["TotalAssignedOverlapSec"] / evidence) if evidence > 1e-6 else "",
                "BestOverlapRate": _fmt(rec["TotalBestOverlapSec"] / evidence) if evidence > 1e-6 else "",
                "MeanStreamingAbsErrorMs": _fmt(rec["TotalMeanStreamingAbsErrorMs"] / err_count) if err_count > 0.0 else "",
            })

    metric_rows = [r for r in rows if r.get("AssignmentCategory") != "UnmatchedDetectorIsland"]
    stat_metrics = [
        "AssignedOverlapRatio",
        "BestOverlapRatio",
        "AssignedDetectorSpanSec",
        "EvidenceSpanSec",
        "MappedAudioSpanSec",
        "CommittedCenterSpanSec",
        "MeanStreamingAbsErrorMs",
    ]
    stats_path = run_dir / "corpus_text_audio_island_assignment_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        stat_fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=stat_fields)
        w.writeheader()
        slices = {
            "all_text_islands": metric_rows,
            "healthy": [r for r in metric_rows if r.get("AssignmentCategory") == "Healthy"],
            "problematic": [r for r in metric_rows if r.get("AssignmentCategory") != "Healthy"],
            "wrong_or_low_overlap": [r for r in metric_rows if r.get("AssignmentCategory") in {"WrongDetectorIsland", "AssignedLowOverlap", "LowDetectorOverlap", "AssignedDetectorTooShort", "AssignedDetectorTooLong", "DetectorFragmentedTextIsland"}],
        }
        for slice_name, slice_rows in slices.items():
            for metric in stat_metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_rows = sorted(
        metric_rows,
        key=lambda r: (
            0 if r.get("AssignmentCategory") != "Healthy" else 1,
            -(_row_float(r, "MeanStreamingAbsErrorMs")),
            _row_float(r, "AssignedOverlapRatio"),
        ),
    )[:120]
    worst_path = run_dir / "corpus_text_audio_island_assignment_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        worst_fields = [
            "case_id",
            "TextIslandIndex",
            "AssignedAudioIslandIndex",
            "BestOverlapAudioIslandIndex",
            "AssignmentCategory",
            "AssignedMatchesBestOverlap",
            "EventCount",
            "FirstWord",
            "LastWord",
            "EvidenceSpanSec",
            "AssignedDetectorSpanSec",
            "AssignedOverlapRatio",
            "BestOverlapRatio",
            "OverlappingDetectorIslandCount",
            "AssignedTextIslandCountForDetector",
            "MappedAudioSpanSec",
            "CommittedCenterSpanSec",
            "MeanStreamingAbsErrorMs",
            "FirstCenterErrorMs",
            "LastCenterErrorMs",
            "Notes",
        ]
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in worst_fields})



def derive_streaming_island_assignment_sim_rows(case_dir: Path) -> list[dict[str, str]]:
    """Causal text-island -> detector-fragment assignment simulator.

    This intentionally does not compute an offline optimal assignment. It simulates
    the runtime shape we actually want: one active text island, confirmed onset,
    detector fragments merged into the active island across plausible internal gaps,
    and hard-boundary advancement only after the current island has enough evidence.

    The output is diagnostic-only. It tells us when the current index/order mapping
    differs from a causal active-island interpretation, and whether the causal span
    better explains the oracle/evidence span already emitted by liplab.
    """
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    speech_rows = read_csv_rows(case_dir / "speech_islands.csv")
    if not runtime_rows:
        return []

    text_rows = sorted(runtime_rows, key=lambda r: _row_int(r, "TextIslandIndex", 999999))

    detectors: list[dict[str, float | int]] = []
    for ordinal, speech in enumerate(speech_rows):
        start = parse_float(speech.get("AudioBufferStartSec", ""))
        end = speech_island_end(speech)
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        idx = parse_int(speech.get("IslandIndex", ""))
        detectors.append({"index": idx if idx is not None else ordinal, "start": start, "end": end})
    detectors.sort(key=lambda d: (float(d["start"]), float(d["end"])))

    # Tuned as diagnostic policy constants, not final runtime constants.
    onset_early_slack_sec = 0.250
    onset_late_slack_sec = 0.750
    merge_gap_sec = 0.450
    min_ready_ratio_before_handoff = 0.550
    max_span_ratio_before_handoff = 1.850
    next_island_guard_sec = 0.180

    next_det_pos = 0
    rows: list[dict[str, str]] = []

    for island_pos, row in enumerate(text_rows):
        text_idx = _row_int(row, "TextIslandIndex")
        assigned_idx = _row_int(row, "AudioIslandIndex")
        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        mapped_span = max(0.0, mapped_end - mapped_start)
        planner_duration = _row_float(row, "PlannerDurationSec", mapped_span)
        expected_span = max(0.050, mapped_span if mapped_span > 1e-6 else planner_duration)
        next_mapped_start = None
        if island_pos + 1 < len(text_rows):
            next_mapped_start = _row_float(text_rows[island_pos + 1], "MappedAudioStartSec")

        evidence_start = _row_float(row, "EvidenceFirstStartSec")
        evidence_end = _row_float(row, "EvidenceLastEndSec")
        if evidence_end <= evidence_start:
            evidence_start = _row_float(row, "EvidenceFirstCenterSec")
            evidence_end = _row_float(row, "EvidenceLastCenterSec")
        evidence_span = max(0.0, evidence_end - evidence_start)

        search_lo = max(0.0, mapped_start - onset_early_slack_sec)
        while next_det_pos < len(detectors) and float(detectors[next_det_pos]["end"]) < search_lo:
            next_det_pos += 1

        launch_pos = None
        search_hi = mapped_start + onset_late_slack_sec
        probe = next_det_pos
        while probe < len(detectors):
            det = detectors[probe]
            det_start = float(det["start"])
            det_end = float(det["end"])
            if det_start > search_hi:
                break
            if det_end >= search_lo:
                launch_pos = probe
                break
            probe += 1

        causal_indices: list[int] = []
        causal_start = 0.0
        causal_end = 0.0
        causal_detector_seconds = 0.0
        causal_gap_seconds = 0.0
        handoff_reason = "NoConfirmedOnset"
        if launch_pos is not None:
            pos = launch_pos
            causal_start = float(detectors[pos]["start"])
            causal_end = float(detectors[pos]["end"])
            causal_indices.append(int(detectors[pos]["index"]))
            causal_detector_seconds += max(0.0, causal_end - causal_start)
            pos += 1
            handoff_reason = "EndOfDetectorList"
            while pos < len(detectors):
                det = detectors[pos]
                det_start = float(det["start"])
                det_end = float(det["end"])
                gap = max(0.0, det_start - causal_end)
                current_span = max(0.0, causal_end - causal_start)
                ready_for_handoff = current_span >= expected_span * min_ready_ratio_before_handoff
                too_long = current_span >= expected_span * max_span_ratio_before_handoff
                near_next_island = next_mapped_start is not None and det_start >= next_mapped_start - next_island_guard_sec
                large_internal_gap = gap > merge_gap_sec

                if too_long:
                    handoff_reason = "CurrentIslandSpanTooLong"
                    break
                if ready_for_handoff and near_next_island:
                    handoff_reason = "NextIslandGuard"
                    break
                if ready_for_handoff and large_internal_gap:
                    handoff_reason = "LargeGapAfterReady"
                    break
                causal_gap_seconds += gap
                causal_indices.append(int(det["index"]))
                causal_detector_seconds += max(0.0, det_end - det_start)
                causal_end = max(causal_end, det_end)
                pos += 1
            next_det_pos = pos

        causal_span = max(0.0, causal_end - causal_start)
        causal_overlap = interval_overlap_sec(evidence_start, evidence_end, causal_start, causal_end) if causal_span > 0.0 and evidence_span > 0.0 else 0.0
        causal_overlap_ratio = causal_overlap / evidence_span if evidence_span > 1e-6 else 0.0
        current_start = _row_float(row, "DetectorAudioStartSec")
        current_end = _row_float(row, "DetectorAudioEndSec")
        current_span = max(0.0, current_end - current_start)
        current_overlap = interval_overlap_sec(evidence_start, evidence_end, current_start, current_end) if current_span > 0.0 and evidence_span > 0.0 else 0.0
        current_overlap_ratio = current_overlap / evidence_span if evidence_span > 1e-6 else 0.0

        causal_first_idx = causal_indices[0] if causal_indices else -1
        causal_last_idx = causal_indices[-1] if causal_indices else -1
        category = "SameAsCurrent"
        notes: list[str] = []
        if not causal_indices:
            category = "NoCausalDetectorSpan"
            notes.append("no confirmed onset in causal launch window")
        elif assigned_idx < 0:
            category = "CausalAssignsUnassignedTextIsland"
            notes.append("current runtime had no assigned detector island")
        elif assigned_idx not in causal_indices:
            category = "CausalUsesDifferentDetectorSpan"
            notes.append("current assigned detector is not part of causal active-island span")
        elif len(causal_indices) > 1 and current_span < causal_span * 0.70:
            category = "CausalMergesFragments"
            notes.append("causal policy merges detector fragments for active island")
        elif causal_overlap_ratio > current_overlap_ratio + 0.20:
            category = "CausalImprovesOverlap"
            notes.append("causal span overlaps oracle evidence materially better")

        if len(causal_indices) > 1:
            notes.append("fragment_count=" + str(len(causal_indices)))
        if causal_gap_seconds > 0.0:
            notes.append("merged_gap_sec=" + _fmt(causal_gap_seconds))
        if current_overlap_ratio < 0.35 and causal_overlap_ratio >= 0.35:
            notes.append("current low overlap rescued by causal span")
        if _row_float(row, "MeanStreamingAbsErrorMs") >= 250.0:
            notes.append("high current island MAE")

        rows.append({
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "CurrentAudioIslandIndex": str(assigned_idx) if assigned_idx >= 0 else "",
            "CausalFirstAudioIslandIndex": str(causal_first_idx) if causal_first_idx >= 0 else "",
            "CausalLastAudioIslandIndex": str(causal_last_idx) if causal_last_idx >= 0 else "",
            "CausalDetectorIndices": ";".join(str(i) for i in causal_indices),
            "CausalAssignmentCategory": category,
            "EventCount": row.get("EventCount", ""),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "MappedAudioStartSec": _fmt(mapped_start),
            "MappedAudioEndSec": _fmt(mapped_end),
            "MappedAudioSpanSec": _fmt(mapped_span),
            "ExpectedIslandSpanSec": _fmt(expected_span),
            "EvidenceStartSec": _fmt(evidence_start),
            "EvidenceEndSec": _fmt(evidence_end),
            "EvidenceSpanSec": _fmt(evidence_span),
            "CurrentDetectorStartSec": _fmt(current_start),
            "CurrentDetectorEndSec": _fmt(current_end),
            "CurrentDetectorSpanSec": _fmt(current_span),
            "CurrentOverlapSec": _fmt(current_overlap),
            "CurrentOverlapRatio": _fmt(current_overlap_ratio),
            "CausalDetectorStartSec": _fmt(causal_start),
            "CausalDetectorEndSec": _fmt(causal_end),
            "CausalDetectorSpanSec": _fmt(causal_span),
            "CausalDetectorSpeechSeconds": _fmt(causal_detector_seconds),
            "CausalMergedGapSeconds": _fmt(causal_gap_seconds),
            "CausalFragmentCount": str(len(causal_indices)),
            "CausalOverlapSec": _fmt(causal_overlap),
            "CausalOverlapRatio": _fmt(causal_overlap_ratio),
            "CausalMinusCurrentOverlapRatio": _fmt(causal_overlap_ratio - current_overlap_ratio),
            "CausalSpanToExpectedRatio": _fmt(causal_span / expected_span) if expected_span > 1e-6 and causal_span > 0.0 else "",
            "CurrentSpanToExpectedRatio": _fmt(current_span / expected_span) if expected_span > 1e-6 and current_span > 0.0 else "",
            "HandoffReason": handoff_reason,
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
            "Notes": ";".join(notes),
        })

    for pos in range(next_det_pos, len(detectors)):
        det = detectors[pos]
        rows.append({
            "case_id": case_dir.name,
            "TextIslandIndex": "",
            "CurrentAudioIslandIndex": "",
            "CausalFirstAudioIslandIndex": str(int(det["index"])),
            "CausalLastAudioIslandIndex": str(int(det["index"])),
            "CausalDetectorIndices": str(int(det["index"])),
            "CausalAssignmentCategory": "UnconsumedDetectorFragment",
            "EventCount": "0",
            "FirstWord": "",
            "LastWord": "",
            "MappedAudioStartSec": "",
            "MappedAudioEndSec": "",
            "MappedAudioSpanSec": "",
            "ExpectedIslandSpanSec": "",
            "EvidenceStartSec": "",
            "EvidenceEndSec": "",
            "EvidenceSpanSec": "",
            "CurrentDetectorStartSec": "",
            "CurrentDetectorEndSec": "",
            "CurrentDetectorSpanSec": "",
            "CurrentOverlapSec": "",
            "CurrentOverlapRatio": "",
            "CausalDetectorStartSec": _fmt(float(det["start"])),
            "CausalDetectorEndSec": _fmt(float(det["end"])),
            "CausalDetectorSpanSec": _fmt(max(0.0, float(det["end"]) - float(det["start"]))),
            "CausalDetectorSpeechSeconds": _fmt(max(0.0, float(det["end"]) - float(det["start"]))),
            "CausalMergedGapSeconds": "0",
            "CausalFragmentCount": "1",
            "CausalOverlapSec": "",
            "CausalOverlapRatio": "",
            "CausalMinusCurrentOverlapRatio": "",
            "CausalSpanToExpectedRatio": "",
            "CurrentSpanToExpectedRatio": "",
            "HandoffReason": "NoActiveTextIsland",
            "MeanStreamingAbsErrorMs": "",
            "FirstCenterErrorMs": "",
            "LastCenterErrorMs": "",
            "Notes": "detector fragment not consumed by causal active-island simulator",
        })
    return rows




def _frame_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    v = parse_float(row.get(key, ""))
    return v if v is not None and math.isfinite(v) else default


def _strict_voiced_frame(frame: dict[str, str]) -> bool:
    zcr = _frame_float(frame, "ZCR")
    flux = _frame_float(frame, "Flux")
    rms_norm = _frame_float(frame, "RMSNorm")
    return (zcr <= 0.180 and flux >= 0.010) or (rms_norm >= 0.420 and zcr <= 0.240)


def _feature_detector_regions(case_dir: Path, variant: str) -> list[dict[str, float | int]]:
    """Build candidate speech fragments from audio_feature_frames.csv for detector-policy ablations.

    These are diagnostic detector variants. They are intentionally implemented in Python
    so we can train/compare policies before moving a winner into the C++ streaming path.
    The important design split is first-onset qualification versus continuation evidence:
    first onset should reject breath/pop; continuation can be more permissive.
    """
    frames = read_csv_rows(case_dir / "audio_feature_frames.csv")
    if not frames:
        return []

    if variant == "current_reported":
        return [{"index": int(r["index"]), "start": float(r["start"]), "end": float(r["end"])} for r in load_detector_regions(case_dir)]

    configs = {
        "legacy_energy_20ms_backdated": {
            "onset_sustain": 0.020, "backdate": True, "strict_onset": False,
            "continue_mode": "legacy", "close_gap": 0.090, "min_region": 0.020,
        },
        "strict_55ms_voiced_no_backdate": {
            "onset_sustain": 0.055, "backdate": False, "strict_onset": True,
            "continue_mode": "strict", "close_gap": 0.120, "min_region": 0.045,
        },
        "dual_strict_onset_permissive_continue": {
            "onset_sustain": 0.055, "backdate": False, "strict_onset": True,
            "continue_mode": "permissive", "close_gap": 0.180, "min_region": 0.045,
        },
        "dual_strict_onset_low_continue": {
            "onset_sustain": 0.055, "backdate": False, "strict_onset": True,
            "continue_mode": "low", "close_gap": 0.240, "min_region": 0.045,
        },
        "balanced_45ms_voiced_permissive_continue": {
            "onset_sustain": 0.045, "backdate": False, "strict_onset": True,
            "continue_mode": "permissive", "close_gap": 0.180, "min_region": 0.040,
        },
    }
    cfg = configs.get(variant)
    if cfg is None:
        return []

    regions: list[dict[str, float | int]] = []
    noise_floor = 0.0001
    candidate_active = False
    candidate_start = 0.0
    candidate_accum = 0.0
    in_region = False
    region_start = 0.0
    region_last_speech_end = 0.0
    silence_accum = 0.0

    def frame_duration(f: dict[str, str]) -> float:
        return max(0.0, _frame_float(f, "AudioBufferEndSec") - _frame_float(f, "AudioBufferStartSec"))

    def thresholds(f: dict[str, str]) -> tuple[float, float]:
        legacy_open = max(0.004, noise_floor * 3.0 + 0.003)
        low_open = max(0.003, noise_floor * 2.0 + 0.002)
        return legacy_open, low_open

    def is_onset_frame(f: dict[str, str]) -> bool:
        legacy_open, _ = thresholds(f)
        if _frame_float(f, "RMS") < legacy_open:
            return False
        return (not cfg["strict_onset"]) or _strict_voiced_frame(f)

    def is_continue_frame(f: dict[str, str]) -> bool:
        legacy_open, low_open = thresholds(f)
        mode = str(cfg["continue_mode"])
        rms = _frame_float(f, "RMS")
        rms_norm = _frame_float(f, "RMSNorm")
        flux = _frame_float(f, "Flux")
        zcr = _frame_float(f, "ZCR")
        if mode == "strict":
            return rms >= legacy_open and _strict_voiced_frame(f)
        if mode == "legacy":
            return rms >= legacy_open
        if mode == "permissive":
            return (rms >= low_open and zcr <= 0.620) or rms_norm >= 0.180 or flux >= 0.020
        if mode == "low":
            return (rms >= low_open and zcr <= 0.720) or rms_norm >= 0.140 or flux >= 0.014
        return rms >= legacy_open

    for f in frames:
        rms = _frame_float(f, "RMS")
        noise_floor = min(0.050, max(0.000001, noise_floor * 0.980 + rms * 0.020))
        start = _frame_float(f, "AudioBufferStartSec")
        end = _frame_float(f, "AudioBufferEndSec")
        dur = frame_duration(f)
        if dur <= 0.0:
            continue

        if not in_region:
            if is_onset_frame(f):
                if not candidate_active:
                    candidate_active = True
                    candidate_start = start
                    candidate_accum = 0.0
                candidate_accum += dur
                if candidate_accum >= float(cfg["onset_sustain"]):
                    in_region = True
                    region_start = candidate_start if bool(cfg["backdate"]) else max(0.0, end - float(cfg["onset_sustain"]))
                    region_last_speech_end = end
                    silence_accum = 0.0
                    candidate_active = False
                    candidate_accum = 0.0
            else:
                candidate_active = False
                candidate_accum = 0.0
            continue

        if is_continue_frame(f):
            region_last_speech_end = end
            silence_accum = 0.0
        else:
            silence_accum += dur
            if silence_accum >= float(cfg["close_gap"]):
                if region_last_speech_end - region_start >= float(cfg["min_region"]):
                    regions.append({"index": len(regions), "start": region_start, "end": region_last_speech_end})
                in_region = False
                candidate_active = False
                candidate_accum = 0.0
                silence_accum = 0.0

    if in_region and region_last_speech_end - region_start >= float(cfg["min_region"]):
        regions.append({"index": len(regions), "start": region_start, "end": region_last_speech_end})
    return regions


def _simulate_causal_assignment_for_detectors(case_dir: Path, detectors: list[dict[str, float | int]], variant: str) -> list[dict[str, str]]:
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    if not runtime_rows:
        return []
    text_rows = sorted(runtime_rows, key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    detectors = sorted(detectors, key=lambda d: (float(d["start"]), float(d["end"])))

    onset_early_slack_sec = 0.250
    onset_late_slack_sec = 0.750
    merge_gap_sec = 0.450
    min_ready_ratio_before_handoff = 0.550
    max_span_ratio_before_handoff = 1.850
    next_island_guard_sec = 0.180

    next_det_pos = 0
    rows: list[dict[str, str]] = []
    for island_pos, row in enumerate(text_rows):
        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        mapped_span = max(0.0, mapped_end - mapped_start)
        planner_duration = _row_float(row, "PlannerDurationSec", mapped_span)
        expected_span = max(0.050, mapped_span if mapped_span > 1e-6 else planner_duration)
        next_mapped_start = _row_float(text_rows[island_pos + 1], "MappedAudioStartSec") if island_pos + 1 < len(text_rows) else None

        evidence_start = _row_float(row, "EvidenceFirstStartSec")
        evidence_end = _row_float(row, "EvidenceLastEndSec")
        if evidence_end <= evidence_start:
            evidence_start = _row_float(row, "EvidenceFirstCenterSec")
            evidence_end = _row_float(row, "EvidenceLastCenterSec")
        evidence_span = max(0.0, evidence_end - evidence_start)
        current_start = _row_float(row, "DetectorAudioStartSec")
        current_end = _row_float(row, "DetectorAudioEndSec")
        current_span = max(0.0, current_end - current_start)
        current_overlap = interval_overlap_sec(evidence_start, evidence_end, current_start, current_end) if current_span > 0.0 and evidence_span > 0.0 else 0.0
        current_overlap_ratio = current_overlap / evidence_span if evidence_span > 1e-6 else 0.0

        search_lo = max(0.0, mapped_start - onset_early_slack_sec)
        while next_det_pos < len(detectors) and float(detectors[next_det_pos]["end"]) < search_lo:
            next_det_pos += 1

        launch_pos = None
        probe = next_det_pos
        search_hi = mapped_start + onset_late_slack_sec
        while probe < len(detectors):
            det = detectors[probe]
            det_start = float(det["start"])
            det_end = float(det["end"])
            if det_start > search_hi:
                break
            if det_end >= search_lo:
                launch_pos = probe
                break
            probe += 1

        causal_indices: list[int] = []
        causal_start = 0.0
        causal_end = 0.0
        causal_detector_seconds = 0.0
        causal_gap_seconds = 0.0
        handoff_reason = "NoConfirmedOnset"
        if launch_pos is not None:
            pos = launch_pos
            causal_start = float(detectors[pos]["start"])
            causal_end = float(detectors[pos]["end"])
            causal_indices.append(int(detectors[pos]["index"]))
            causal_detector_seconds += max(0.0, causal_end - causal_start)
            pos += 1
            handoff_reason = "EndOfDetectorList"
            while pos < len(detectors):
                det = detectors[pos]
                det_start = float(det["start"])
                det_end = float(det["end"])
                gap = max(0.0, det_start - causal_end)
                current_causal_span = max(0.0, causal_end - causal_start)
                ready_for_handoff = current_causal_span >= expected_span * min_ready_ratio_before_handoff
                too_long = current_causal_span >= expected_span * max_span_ratio_before_handoff
                near_next_island = next_mapped_start is not None and det_start >= next_mapped_start - next_island_guard_sec
                large_internal_gap = gap > merge_gap_sec
                if too_long:
                    handoff_reason = "CurrentIslandSpanTooLong"
                    break
                if ready_for_handoff and near_next_island:
                    handoff_reason = "NextIslandGuard"
                    break
                if ready_for_handoff and large_internal_gap:
                    handoff_reason = "LargeGapAfterReady"
                    break
                causal_gap_seconds += gap
                causal_indices.append(int(det["index"]))
                causal_detector_seconds += max(0.0, det_end - det_start)
                causal_end = max(causal_end, det_end)
                pos += 1
            next_det_pos = pos

        causal_span = max(0.0, causal_end - causal_start)
        causal_overlap = interval_overlap_sec(evidence_start, evidence_end, causal_start, causal_end) if causal_span > 0.0 and evidence_span > 0.0 else 0.0
        causal_overlap_ratio = causal_overlap / evidence_span if evidence_span > 1e-6 else 0.0
        category = "Matched"
        if not causal_indices:
            category = "NoCausalDetectorSpan"
        elif len(causal_indices) > 1:
            category = "CausalMergesFragments"
        elif causal_overlap_ratio < 0.35:
            category = "LowOverlap"

        rows.append({
            "case_id": case_dir.name,
            "Variant": variant,
            "TextIslandIndex": row.get("TextIslandIndex", ""),
            "DetectorIndices": ";".join(str(i) for i in causal_indices),
            "AssignmentCategory": category,
            "FragmentCount": str(len(causal_indices)),
            "MappedAudioStartSec": _fmt(mapped_start),
            "MappedAudioEndSec": _fmt(mapped_end),
            "ExpectedIslandSpanSec": _fmt(expected_span),
            "EvidenceStartSec": _fmt(evidence_start),
            "EvidenceEndSec": _fmt(evidence_end),
            "EvidenceSpanSec": _fmt(evidence_span),
            "CurrentOverlapRatio": _fmt(current_overlap_ratio),
            "DetectorStartSec": _fmt(causal_start),
            "DetectorEndSec": _fmt(causal_end),
            "DetectorSpanSec": _fmt(causal_span),
            "DetectorSpeechSeconds": _fmt(causal_detector_seconds),
            "MergedGapSeconds": _fmt(causal_gap_seconds),
            "OverlapSec": _fmt(causal_overlap),
            "OverlapRatio": _fmt(causal_overlap_ratio),
            "OverlapImprovementVsCurrent": _fmt(causal_overlap_ratio - current_overlap_ratio),
            "SpanToExpectedRatio": _fmt(causal_span / expected_span) if expected_span > 1e-6 and causal_span > 0.0 else "",
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
            "HandoffReason": handoff_reason,
        })
    return rows


def collect_detector_variant_training_rows(run_dir: Path) -> list[dict[str, str]]:
    variants = [
        "current_reported",
        "legacy_energy_20ms_backdated",
        "strict_55ms_voiced_no_backdate",
        "balanced_45ms_voiced_permissive_continue",
        "dual_strict_onset_permissive_continue",
        "dual_strict_onset_low_continue",
    ]
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        for variant in variants:
            detectors = _feature_detector_regions(case_dir, variant)
            rows.extend(_simulate_causal_assignment_for_detectors(case_dir, detectors, variant))
    return rows


def write_corpus_detector_variant_training_summaries(run_dir: Path) -> None:
    rows = collect_detector_variant_training_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_detector_variant_training.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    variants = sorted({r.get("Variant", "") for r in rows})
    summary_fields = [
        "Variant", "TextIslandCount", "MatchedCount", "NoCausalDetectorSpanCount", "LowOverlapCount", "FragmentMergeCount",
        "MeanOverlapRatio", "MedianOverlapRatio", "MeanOverlapImprovementVsCurrent", "MeanSpanToExpectedRatio",
        "MeanFragmentCount", "TotalMergedGapSeconds", "Score", "RecommendationNotes",
    ]
    with (run_dir / "corpus_detector_variant_training_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        summaries: list[dict[str, str]] = []
        for variant in variants:
            vr = [r for r in rows if r.get("Variant") == variant]
            n = len(vr)
            no_span = sum(1 for r in vr if r.get("AssignmentCategory") == "NoCausalDetectorSpan")
            low_overlap = sum(1 for r in vr if r.get("AssignmentCategory") == "LowOverlap")
            merges = sum(1 for r in vr if r.get("AssignmentCategory") == "CausalMergesFragments")
            matched = n - no_span
            mean_overlap = mean(numeric_values(vr, "OverlapRatio")) if numeric_values(vr, "OverlapRatio") else 0.0
            med_overlap = median(numeric_values(vr, "OverlapRatio")) if numeric_values(vr, "OverlapRatio") else 0.0
            mean_improve = mean(numeric_values(vr, "OverlapImprovementVsCurrent")) if numeric_values(vr, "OverlapImprovementVsCurrent") else 0.0
            span_vals = numeric_values(vr, "SpanToExpectedRatio")
            mean_span = mean(span_vals) if span_vals else 0.0
            frag_vals = numeric_values(vr, "FragmentCount")
            mean_frag = mean(frag_vals) if frag_vals else 0.0
            merged_total = sum(numeric_values(vr, "MergedGapSeconds"))
            # Prefer high overlap and fewer misses. Penalize very long spans and low-overlap islands.
            span_penalty = abs(mean_span - 1.0) * 0.20 if mean_span > 0.0 else 0.20
            score = mean_overlap + 0.50 * mean_improve - (no_span / max(1, n)) * 0.60 - (low_overlap / max(1, n)) * 0.25 - span_penalty
            notes = []
            if no_span > 0:
                notes.append(f"no_span={no_span}")
            if merges > 0:
                notes.append(f"merges={merges}")
            if mean_span > 1.35:
                notes.append("may over-extend spans")
            if mean_span < 0.75:
                notes.append("may under-cover spans")
            rec = {
                "Variant": variant,
                "TextIslandCount": str(n),
                "MatchedCount": str(matched),
                "NoCausalDetectorSpanCount": str(no_span),
                "LowOverlapCount": str(low_overlap),
                "FragmentMergeCount": str(merges),
                "MeanOverlapRatio": _fmt(mean_overlap),
                "MedianOverlapRatio": _fmt(med_overlap),
                "MeanOverlapImprovementVsCurrent": _fmt(mean_improve),
                "MeanSpanToExpectedRatio": _fmt(mean_span),
                "MeanFragmentCount": _fmt(mean_frag),
                "TotalMergedGapSeconds": _fmt(merged_total),
                "Score": _fmt(score),
                "RecommendationNotes": ";".join(notes),
            }
            summaries.append(rec)
        for rec in sorted(summaries, key=lambda r: -_row_float(r, "Score")):
            w.writerow(rec)

    metrics = ["OverlapRatio", "OverlapImprovementVsCurrent", "SpanToExpectedRatio", "FragmentCount", "MergedGapSeconds", "MeanStreamingAbsErrorMs"]
    with (run_dir / "corpus_detector_variant_training_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Variant", "slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"])
        w.writeheader()
        for variant in variants:
            vr = [r for r in rows if r.get("Variant") == variant]
            slices = {
                "all_text_islands": vr,
                "no_span": [r for r in vr if r.get("AssignmentCategory") == "NoCausalDetectorSpan"],
                "low_overlap": [r for r in vr if r.get("AssignmentCategory") == "LowOverlap"],
                "merged_fragments": [r for r in vr if r.get("AssignmentCategory") == "CausalMergesFragments"],
            }
            for slice_name, sr in slices.items():
                for metric in metrics:
                    rec = {"Variant": variant, "slice": slice_name, "metric": metric}
                    rec.update(stats_for(numeric_values(sr, metric)))
                    w.writerow(rec)

    worst = sorted(rows, key=lambda r: (r.get("Variant", ""), _row_float(r, "OverlapRatio"), -_row_float(r, "MeanStreamingAbsErrorMs")))[:240]
    worst_fields = [
        "case_id", "Variant", "TextIslandIndex", "AssignmentCategory", "DetectorIndices", "FragmentCount",
        "ExpectedIslandSpanSec", "EvidenceSpanSec", "DetectorSpanSec", "OverlapRatio",
        "OverlapImprovementVsCurrent", "SpanToExpectedRatio", "MeanStreamingAbsErrorMs", "HandoffReason",
    ]
    with (run_dir / "corpus_detector_variant_training_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in worst:
            w.writerow({k: row.get(k, "") for k in worst_fields})


def _assignment_detector_fragments(case_dir: Path) -> list[dict[str, float | int]]:
    speech_rows = read_csv_rows(case_dir / "speech_islands.csv")
    detectors: list[dict[str, float | int]] = []
    for ordinal, speech in enumerate(speech_rows):
        start = parse_float(speech.get("AudioBufferStartSec", ""))
        end = speech_island_end(speech)
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        idx = parse_int(speech.get("IslandIndex", ""))
        detectors.append({"index": idx if idx is not None else ordinal, "start": start, "end": end})
    detectors.sort(key=lambda d: (float(d["start"]), float(d["end"])))
    return detectors


def _assignment_candidate_groups(detectors: list[dict[str, float | int]], max_group_size: int = 4) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for start_pos in range(len(detectors)):
        gap_sum = 0.0
        end_prev = float(detectors[start_pos]["end"])
        indices: list[int] = []
        speech_seconds = 0.0
        for end_pos in range(start_pos, min(len(detectors), start_pos + max_group_size)):
            det = detectors[end_pos]
            det_start = float(det["start"])
            det_end = float(det["end"])
            if end_pos > start_pos:
                gap_sum += max(0.0, det_start - end_prev)
            indices.append(int(det["index"]))
            speech_seconds += max(0.0, det_end - det_start)
            end_prev = max(end_prev, det_end)
            groups.append({
                "first_pos": start_pos,
                "last_pos": end_pos,
                "indices": list(indices),
                "start": float(detectors[start_pos]["start"]),
                "end": max(float(detectors[k]["end"]) for k in range(start_pos, end_pos + 1)),
                "speech_seconds": speech_seconds,
                "gap_seconds": gap_sum,
            })
    return groups


def _assignment_candidate_score(
    row: dict[str, str],
    group: dict[str, object] | None,
    text_pos: int,
    text_count: int,
    detector_count: int,
) -> dict[str, str | float]:
    mapped_start = _row_float(row, "MappedAudioStartSec")
    mapped_end = _row_float(row, "MappedAudioEndSec")
    mapped_span = max(0.0, mapped_end - mapped_start)
    planner_duration = _row_float(row, "PlannerDurationSec", mapped_span)
    expected_span = max(0.050, mapped_span if mapped_span > 1e-6 else planner_duration)
    evidence_start = _row_float(row, "EvidenceFirstStartSec")
    evidence_end = _row_float(row, "EvidenceLastEndSec")
    if evidence_end <= evidence_start:
        evidence_start = _row_float(row, "EvidenceFirstCenterSec")
        evidence_end = _row_float(row, "EvidenceLastCenterSec")
    evidence_span = max(0.0, evidence_end - evidence_start)
    remaining_text = max(0, text_count - text_pos - 1)

    if group is None:
        fallback_penalty = 1.5
        return {
            "CandidateFragments": "fallback",
            "CandidateFirstAudioIslandIndex": "",
            "CandidateLastAudioIslandIndex": "",
            "CandidateFragmentCount": "0",
            "CandidateStartSec": "",
            "CandidateEndSec": "",
            "CandidateSpanSec": "0.000000",
            "CandidateSpeechSeconds": "0.000000",
            "CandidateGapSeconds": "0.000000",
            "StartErrorSec": "",
            "SpanErrorSec": _fmt(expected_span),
            "GapPenaltySec": "0.000000",
            "StarvationPenalty": "0.000000",
            "FallbackPenalty": _fmt(fallback_penalty),
            "OverlapRatio": "0.000000",
            "TotalScore": _fmt(fallback_penalty + 0.50 * expected_span),
        }

    cand_start = float(group["start"])
    cand_end = float(group["end"])
    cand_span = max(0.0, cand_end - cand_start)
    gap_seconds = float(group["gap_seconds"])
    first_pos = int(group["first_pos"])
    last_pos = int(group["last_pos"])
    remaining_fragments = max(0, detector_count - last_pos - 1)
    starvation = 1.0 if remaining_fragments < remaining_text else 0.0
    start_error = abs(cand_start - mapped_start)
    span_error = abs(cand_span - expected_span)
    overlap = interval_overlap_sec(evidence_start, evidence_end, cand_start, cand_end) if evidence_span > 1e-6 else 0.0
    overlap_ratio = overlap / evidence_span if evidence_span > 1e-6 else 0.0
    # Diagnostic score only. Lower is better. Keep all components visible so future agents can change weights safely.
    total = start_error + 0.50 * span_error + 0.35 * gap_seconds + 2.0 * starvation - 0.35 * overlap_ratio
    indices = [int(i) for i in group["indices"]]  # type: ignore[index]
    return {
        "CandidateFragments": ";".join(str(i) for i in indices),
        "CandidateFirstAudioIslandIndex": str(indices[0]) if indices else "",
        "CandidateLastAudioIslandIndex": str(indices[-1]) if indices else "",
        "CandidateFragmentCount": str(len(indices)),
        "CandidateStartSec": _fmt(cand_start),
        "CandidateEndSec": _fmt(cand_end),
        "CandidateSpanSec": _fmt(cand_span),
        "CandidateSpeechSeconds": _fmt(float(group["speech_seconds"])),
        "CandidateGapSeconds": _fmt(gap_seconds),
        "StartErrorSec": _fmt(start_error),
        "SpanErrorSec": _fmt(span_error),
        "GapPenaltySec": _fmt(gap_seconds * 0.35),
        "StarvationPenalty": _fmt(starvation),
        "FallbackPenalty": "0.000000",
        "OverlapRatio": _fmt(overlap_ratio),
        "TotalScore": _fmt(total),
    }


def derive_assignment_decision_and_candidate_rows(case_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    if not runtime_rows:
        return [], []
    text_rows = sorted(runtime_rows, key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    detectors = _assignment_detector_fragments(case_dir)
    groups = _assignment_candidate_groups(detectors)
    decision_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []

    for text_pos, row in enumerate(text_rows):
        text_idx = _row_int(row, "TextIslandIndex")
        current_idx = _row_int(row, "AudioIslandIndex", -1)
        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        remaining_text = max(0, len(text_rows) - text_pos - 1)
        remaining_audio = max(0, len(detectors) - max(0, current_idx + 1)) if current_idx >= 0 else len(detectors)
        scored: list[dict[str, str]] = []
        for group in groups:
            base = _assignment_candidate_score(row, group, text_pos, len(text_rows), len(detectors))
            indices = str(base.get("CandidateFragments", ""))
            cand_start = _row_float(base, "CandidateStartSec")
            cand_end = _row_float(base, "CandidateEndSec")
            # AU10c: RuntimeChosen must identify the exact runtime-assigned fragment group,
            # not every candidate that merely contains the first runtime fragment.  Earlier
            # AU10bprime diagnostics marked several groups as chosen for the same text island,
            # which made chosen-vs-best margins ambiguous.  The mapped runtime span is the
            # authoritative signature because runtime rows do not expose LastAudioIslandIndex.
            chosen = (
                current_idx >= 0
                and indices.split(";")[0:1] == [str(current_idx)]
                and abs(cand_start - mapped_start) <= 0.015
                and abs(cand_end - mapped_end) <= 0.015
            )
            rec = {
                "case_id": case_dir.name,
                "TextIslandIndex": str(text_idx),
                "FirstWord": row.get("FirstWord", ""),
                "LastWord": row.get("LastWord", ""),
                "RuntimeAudioIslandIndex": str(current_idx) if current_idx >= 0 else "",
                "RuntimeCategory": row.get("AssignmentCategory", ""),
                "MappedAudioStartSec": row.get("MappedAudioStartSec", ""),
                "MappedAudioEndSec": row.get("MappedAudioEndSec", ""),
                "PlannerDurationSec": row.get("PlannerDurationSec", ""),
                "EventCount": row.get("EventCount", ""),
                "RemainingTextIslands": str(remaining_text),
                "RemainingAudioFragments": str(remaining_audio),
                **{k: str(v) for k, v in base.items()},
                "RuntimeChosen": "1" if chosen else "0",
            }
            scored.append(rec)
            candidate_rows.append(rec)
        # Always include explicit fallback candidate.
        fb = _assignment_candidate_score(row, None, text_pos, len(text_rows), len(detectors))
        fb_rec = {
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "RuntimeAudioIslandIndex": str(current_idx) if current_idx >= 0 else "",
            "RuntimeCategory": row.get("AssignmentCategory", ""),
            "MappedAudioStartSec": row.get("MappedAudioStartSec", ""),
            "MappedAudioEndSec": row.get("MappedAudioEndSec", ""),
            "PlannerDurationSec": row.get("PlannerDurationSec", ""),
            "EventCount": row.get("EventCount", ""),
            "RemainingTextIslands": str(remaining_text),
            "RemainingAudioFragments": str(remaining_audio),
            **{k: str(v) for k, v in fb.items()},
            "RuntimeChosen": "1" if current_idx < 0 else "0",
        }
        scored.append(fb_rec)
        candidate_rows.append(fb_rec)

        best = min(scored, key=lambda r: _row_float(r, "TotalScore", 999999.0)) if scored else None
        chosen_rows = [r for r in scored if truthy_csv(r.get("RuntimeChosen", ""))]
        chosen = min(chosen_rows, key=lambda r: _row_float(r, "TotalScore", 999999.0)) if chosen_rows else None
        chosen_score = _row_float(chosen, "TotalScore", 999999.0) if chosen else 999999.0
        best_score = _row_float(best, "TotalScore", 999999.0) if best else 999999.0
        margin = chosen_score - best_score if chosen and best else 0.0
        decision = {
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "RuntimeAudioIslandIndex": str(current_idx) if current_idx >= 0 else "",
            "RuntimeAssignmentCategory": row.get("AssignmentCategory", ""),
            "EventCount": row.get("EventCount", ""),
            "PredictedDurationSec": row.get("PlannerDurationSec", ""),
            "MappedAudioStartSec": row.get("MappedAudioStartSec", ""),
            "MappedAudioEndSec": row.get("MappedAudioEndSec", ""),
            "DetectorFragmentCount": str(len(detectors)),
            "CandidateCount": str(len(scored)),
            "ChosenFragments": chosen.get("CandidateFragments", "") if chosen else "",
            "ChosenStartSec": chosen.get("CandidateStartSec", "") if chosen else "",
            "ChosenEndSec": chosen.get("CandidateEndSec", "") if chosen else "",
            "ChosenSpanSec": chosen.get("CandidateSpanSec", "") if chosen else "",
            "ScoreChosen": chosen.get("TotalScore", "") if chosen else "",
            "BestAlternativeFragments": best.get("CandidateFragments", "") if best else "",
            "BestAlternativeStartSec": best.get("CandidateStartSec", "") if best else "",
            "BestAlternativeEndSec": best.get("CandidateEndSec", "") if best else "",
            "ScoreBestAlternative": best.get("TotalScore", "") if best else "",
            "ScoreMarginChosenMinusBest": _fmt(margin),
            "BestDiffersFromRuntime": "1" if chosen and best and chosen.get("CandidateFragments") != best.get("CandidateFragments") else "0",
            "RemainingTextIslands": str(remaining_text),
            "RemainingAudioFragments": str(remaining_audio),
            "FallbackReason": "runtime had no assigned detector island" if current_idx < 0 else "",
        }
        decision_rows.append(decision)
    return decision_rows, candidate_rows


def collect_assignment_decision_and_candidate_rows(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    decision_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return decision_rows, candidate_rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        d, c = derive_assignment_decision_and_candidate_rows(case_dir)
        decision_rows.extend(d)
        candidate_rows.extend(c)
    return decision_rows, candidate_rows


def write_corpus_assignment_decision_summaries(run_dir: Path) -> None:
    decision_rows, candidate_rows = collect_assignment_decision_and_candidate_rows(run_dir)
    if not decision_rows and not candidate_rows:
        return
    if decision_rows:
        fields = unique_in_order([key for row in decision_rows for key in row.keys()])
        with (run_dir / "corpus_assignment_decisions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in decision_rows:
                w.writerow({k: row.get(k, "") for k in fields})
    if candidate_rows:
        fields = unique_in_order([key for row in candidate_rows for key in row.keys()])
        with (run_dir / "corpus_assignment_candidates.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in candidate_rows:
                w.writerow({k: row.get(k, "") for k in fields})
    if decision_rows:
        summary_fields = [
            "TextIslandCount", "RuntimeFallbackCount", "BestDiffersFromRuntimeCount", "MeanScoreMarginChosenMinusBest",
            "MedianScoreMarginChosenMinusBest", "MeanCandidateCount", "MeanDetectorFragmentCount",
        ]
        with (run_dir / "corpus_assignment_decisions_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            margins = numeric_values(decision_rows, "ScoreMarginChosenMinusBest")
            cand_counts = numeric_values(decision_rows, "CandidateCount")
            det_counts = numeric_values(decision_rows, "DetectorFragmentCount")
            w.writerow({
                "TextIslandCount": str(len(decision_rows)),
                "RuntimeFallbackCount": str(sum(1 for r in decision_rows if r.get("RuntimeAudioIslandIndex", "") == "")),
                "BestDiffersFromRuntimeCount": str(sum(1 for r in decision_rows if truthy_csv(r.get("BestDiffersFromRuntime", "")))),
                "MeanScoreMarginChosenMinusBest": _fmt(mean(margins)) if margins else "",
                "MedianScoreMarginChosenMinusBest": _fmt(median(margins)) if margins else "",
                "MeanCandidateCount": _fmt(mean(cand_counts)) if cand_counts else "",
                "MeanDetectorFragmentCount": _fmt(mean(det_counts)) if det_counts else "",
            })
    if decision_rows:
        worst = sorted(decision_rows, key=lambda r: (-_row_float(r, "ScoreMarginChosenMinusBest"), r.get("case_id", ""), _row_int(r, "TextIslandIndex")))[:240]
        fields = [
            "case_id", "TextIslandIndex", "RuntimeAssignmentCategory", "FirstWord", "LastWord", "RuntimeAudioIslandIndex",
            "ChosenFragments", "BestAlternativeFragments", "ScoreChosen", "ScoreBestAlternative", "ScoreMarginChosenMinusBest",
            "MappedAudioStartSec", "MappedAudioEndSec", "ChosenStartSec", "ChosenEndSec", "BestAlternativeStartSec", "BestAlternativeEndSec",
            "RemainingTextIslands", "RemainingAudioFragments", "FallbackReason",
        ]
        with (run_dir / "corpus_assignment_decisions_worst.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in worst:
                w.writerow({k: row.get(k, "") for k in fields})

def collect_streaming_island_assignment_sim_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        rows.extend(derive_streaming_island_assignment_sim_rows(case_dir))
    return rows




def _au14_policy_limit_hypothesis(row: dict[str, str]) -> str:
    current_category = row.get("CurrentAssignmentCategory", "")
    first_err = _row_float(row, "FirstCenterErrorMs", 0.0)
    last_err = _row_float(row, "LastCenterErrorMs", 0.0)
    mean_err = _row_float(row, "MeanStreamingAbsErrorMs", 0.0)
    assigned_span = _row_float(row, "AssignedDetectorSpanSec", 0.0)
    mapped_span = _row_float(row, "MappedAudioSpanSec", 0.0)
    evidence_span = _row_float(row, "EvidenceSpanSec", 0.0)
    causal_span_ratio = _row_float(row, "CausalSpanToExpectedRatio", 0.0)

    if current_category == "NoAssignedDetectorIsland":
        return "NoDetectorAssignmentEvenCausal"
    if current_category in {"AssignedDetectorTooLong", "PossiblyOverMergedOrTextEvidenceWide"}:
        return "AssignedSpanTooWideOrOverMerged"
    if current_category in {"AssignedDetectorTooShort", "LowDetectorOverlap", "AssignedLowOverlap"}:
        return "AssignedSpanUnderCoversEvidence"
    if evidence_span > 1e-6 and assigned_span > 1e-6:
        ratio = assigned_span / evidence_span
        if ratio < 0.55:
            return "AssignedSpanMuchShorterThanEvidence"
        if ratio > 1.45:
            return "AssignedSpanMuchLongerThanEvidence"
    if mapped_span > 1e-6 and assigned_span > 1e-6:
        ratio = assigned_span / mapped_span
        if ratio < 0.70:
            return "MappedSpanLongerThanDetectorEvidence"
        if ratio > 1.30:
            return "DetectorEvidenceLongerThanMappedSpan"
    if first_err >= 250.0 and last_err >= 250.0:
        return "WholeIslandLateOffset"
    if first_err <= -250.0 and last_err <= -250.0:
        return "WholeIslandEarlyOffset"
    if abs(last_err - first_err) >= 300.0:
        return "IslandScaleOrEndDrift"
    if causal_span_ratio > 0.0 and (causal_span_ratio < 0.70 or causal_span_ratio > 1.30):
        return "CausalSpanExpectedRatioMismatch"
    if mean_err >= 250.0:
        return "HighErrorCurrentEqualsCausal"
    return "LowErrorPolicyLimitation"


def write_corpus_current_policy_limitation_summaries(run_dir: Path) -> None:
    """AU14 diagnostics for cases where the causal simulator reproduces runtime.

    These rows are important because changing ownership/assignment alone is unlikely
    to help them: the causal simulator says the current detector span is already the
    span it would choose.  The diagnostic therefore joins assignment, causal, runtime,
    detector-evidence, and candidate-score context so future patches can decide
    whether the remaining error is detector coverage, text/evidence span mismatch,
    or island-scale/end drift.
    """
    confusion_rows = read_csv_rows(run_dir / "corpus_assignment_confusion.csv")
    if not confusion_rows:
        return

    def key_for(row: dict[str, str]) -> tuple[str, str]:
        return (row.get("case_id", ""), row.get("TextIslandIndex", ""))

    runtime_by_key = {key_for(r): r for r in read_csv_rows(run_dir / "corpus_runtime_island_alignment.csv")}
    decision_by_key = {key_for(r): r for r in read_csv_rows(run_dir / "corpus_assignment_decisions.csv")}
    evidence_by_key = {key_for(r): r for r in read_csv_rows(run_dir / "corpus_detector_evidence.csv")}
    text_assign_by_key = {key_for(r): r for r in read_csv_rows(run_dir / "corpus_text_audio_island_assignment.csv")}
    causal_by_key = {key_for(r): r for r in read_csv_rows(run_dir / "corpus_streaming_island_assignment_sim.csv")}

    rows: list[dict[str, str]] = []
    for conf in confusion_rows:
        if conf.get("AssignmentDiagnosis", "") != "CurrentPolicyLimitation":
            continue
        key = key_for(conf)
        runtime = runtime_by_key.get(key, {})
        decision = decision_by_key.get(key, {})
        evidence = evidence_by_key.get(key, {})
        text_assign = text_assign_by_key.get(key, {})
        causal = causal_by_key.get(key, {})
        assigned_span = _row_float(conf, "AssignedDetectorSpanSec", _row_float(text_assign, "AssignedDetectorSpanSec", 0.0))
        mapped_span = _row_float(conf, "MappedAudioSpanSec", _row_float(runtime, "MappedAudioDurationSec", 0.0))
        evidence_span = _row_float(conf, "EvidenceSpanSec", _row_float(runtime, "EvidenceRenderSpanSec", 0.0))
        first_err = _row_float(conf, "FirstCenterErrorMs", _row_float(runtime, "FirstCenterErrorMs", 0.0))
        last_err = _row_float(conf, "LastCenterErrorMs", _row_float(runtime, "LastCenterErrorMs", 0.0))
        rec = {
            "case_id": key[0],
            "TextIslandIndex": key[1],
            "FirstWord": conf.get("FirstWord", runtime.get("FirstWord", "")),
            "LastWord": conf.get("LastWord", runtime.get("LastWord", "")),
            "EventCount": conf.get("EventCount", runtime.get("EventCount", "")),
            "CurrentAssignmentCategory": conf.get("CurrentAssignmentCategory", ""),
            "CausalAssignmentCategory": conf.get("CausalAssignmentCategory", ""),
            "PolicyLimitHypothesis": "",  # filled below after numeric fields are present
            "HandoffReason": conf.get("HandoffReason", causal.get("HandoffReason", "")),
            "MeanStreamingAbsErrorMs": conf.get("MeanStreamingAbsErrorMs", runtime.get("MeanStreamingAbsErrorMs", "")),
            "FirstCenterErrorMs": _fmt(first_err),
            "LastCenterErrorMs": _fmt(last_err),
            "MappedAudioStartSec": conf.get("MappedAudioStartSec", runtime.get("MappedAudioStartSec", "")),
            "MappedAudioEndSec": conf.get("MappedAudioEndSec", runtime.get("MappedAudioEndSec", "")),
            "MappedAudioSpanSec": _fmt(mapped_span),
            "AssignedDetectorSpanSec": _fmt(assigned_span),
            "EvidenceSpanSec": _fmt(evidence_span),
            "AssignedSpanToEvidenceRatio": _fmt(assigned_span / evidence_span) if evidence_span > 1e-6 else "",
            "AssignedSpanToMappedRatio": _fmt(assigned_span / mapped_span) if mapped_span > 1e-6 else "",
            "CausalDetectorIndices": conf.get("CausalDetectorIndices", causal.get("CausalDetectorIndices", "")),
            "CausalFragmentCount": conf.get("CausalFragmentCount", causal.get("CausalFragmentCount", "")),
            "CausalMergedGapSeconds": conf.get("CausalMergedGapSeconds", causal.get("CausalMergedGapSeconds", "")),
            "CausalSpanToExpectedRatio": conf.get("CausalSpanToExpectedRatio", causal.get("CausalSpanToExpectedRatio", "")),
            "CurrentSpanToExpectedRatio": causal.get("CurrentSpanToExpectedRatio", ""),
            "DetectorAuditCategory": evidence.get("DetectorAuditCategory", ""),
            "AssignedDetectorFragments": evidence.get("AssignedDetectorFragments", conf.get("CausalDetectorIndices", "")),
            "AssignedDetectorFragmentCount": evidence.get("AssignedDetectorFragmentCount", conf.get("CausalFragmentCount", "")),
            "AssignedDetectorSpeechSec": evidence.get("AssignedDetectorSpeechSec", ""),
            "AssignedDetectorGapSec": evidence.get("AssignedDetectorGapSec", conf.get("CausalMergedGapSeconds", "")),
            "DetectorFragmentCountInCase": evidence.get("DetectorFragmentCountInCase", ""),
            "BestAlternativeFragments": decision.get("BestAlternativeFragments", ""),
            "ScoreChosen": decision.get("ScoreChosen", ""),
            "ScoreBestAlternative": decision.get("ScoreBestAlternative", ""),
            "ScoreMarginChosenMinusBest": decision.get("ScoreMarginChosenMinusBest", ""),
            "BestDiffersFromRuntime": decision.get("BestDiffersFromRuntime", ""),
            "PlannerDurationSec": runtime.get("PlannerDurationSec", decision.get("PredictedDurationSec", "")),
            "PlannerSpeechMaterialSec": runtime.get("PlannerSpeechMaterialSec", ""),
            "PlannerPunctuationSec": runtime.get("PlannerPunctuationSec", ""),
            "CommittedCenterSpanSec": runtime.get("CommittedCenterSpanSec", ""),
            "EvidenceCenterSpanSec": runtime.get("EvidenceCenterSpanSec", ""),
            "PrevMappedAudioGapSec": runtime.get("PrevMappedAudioGapSec", ""),
            "PrevDetectorAudioGapSec": runtime.get("PrevDetectorAudioGapSec", ""),
            "PrevGapDeltaCommittedVsEvidenceMs": runtime.get("PrevGapDeltaCommittedVsEvidenceMs", ""),
        }
        rec["PolicyLimitHypothesis"] = _au14_policy_limit_hypothesis(rec)
        rows.append(rec)

    if not rows:
        return

    fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_current_policy_limitation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    by_hypothesis: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_hypothesis.setdefault(row.get("PolicyLimitHypothesis", "Unknown"), []).append(row)
    summary_fields = [
        "PolicyLimitHypothesis", "Count", "MeanStreamingAbsErrorMs", "MeanAbsFirstCenterErrorMs", "MeanAbsLastCenterErrorMs",
        "MeanMappedAudioSpanSec", "MeanAssignedDetectorSpanSec", "MeanEvidenceSpanSec", "MeanAssignedSpanToEvidenceRatio",
        "MeanAssignedSpanToMappedRatio", "BestDiffersFromRuntimeCount",
    ]
    with (run_dir / "corpus_current_policy_limitation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for hyp, hyp_rows in sorted(by_hypothesis.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            first = [abs(v) for v in numeric_values(hyp_rows, "FirstCenterErrorMs")]
            last = [abs(v) for v in numeric_values(hyp_rows, "LastCenterErrorMs")]
            w.writerow({
                "PolicyLimitHypothesis": hyp,
                "Count": str(len(hyp_rows)),
                "MeanStreamingAbsErrorMs": _fmt(mean(numeric_values(hyp_rows, "MeanStreamingAbsErrorMs"))) if numeric_values(hyp_rows, "MeanStreamingAbsErrorMs") else "",
                "MeanAbsFirstCenterErrorMs": _fmt(mean(first)) if first else "",
                "MeanAbsLastCenterErrorMs": _fmt(mean(last)) if last else "",
                "MeanMappedAudioSpanSec": _fmt(mean(numeric_values(hyp_rows, "MappedAudioSpanSec"))) if numeric_values(hyp_rows, "MappedAudioSpanSec") else "",
                "MeanAssignedDetectorSpanSec": _fmt(mean(numeric_values(hyp_rows, "AssignedDetectorSpanSec"))) if numeric_values(hyp_rows, "AssignedDetectorSpanSec") else "",
                "MeanEvidenceSpanSec": _fmt(mean(numeric_values(hyp_rows, "EvidenceSpanSec"))) if numeric_values(hyp_rows, "EvidenceSpanSec") else "",
                "MeanAssignedSpanToEvidenceRatio": _fmt(mean(numeric_values(hyp_rows, "AssignedSpanToEvidenceRatio"))) if numeric_values(hyp_rows, "AssignedSpanToEvidenceRatio") else "",
                "MeanAssignedSpanToMappedRatio": _fmt(mean(numeric_values(hyp_rows, "AssignedSpanToMappedRatio"))) if numeric_values(hyp_rows, "AssignedSpanToMappedRatio") else "",
                "BestDiffersFromRuntimeCount": str(sum(1 for r in hyp_rows if truthy_csv(r.get("BestDiffersFromRuntime", "")))),
            })

    worst_fields = [
        "case_id", "TextIslandIndex", "PolicyLimitHypothesis", "CurrentAssignmentCategory", "CausalAssignmentCategory",
        "FirstWord", "LastWord", "EventCount", "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs",
        "MappedAudioSpanSec", "AssignedDetectorSpanSec", "EvidenceSpanSec", "AssignedSpanToEvidenceRatio",
        "AssignedSpanToMappedRatio", "CausalDetectorIndices", "HandoffReason", "DetectorAuditCategory",
        "BestAlternativeFragments", "ScoreMarginChosenMinusBest",
    ]
    worst = sorted(rows, key=lambda r: _row_float(r, "MeanStreamingAbsErrorMs"), reverse=True)[:240]
    with (run_dir / "corpus_current_policy_limitation_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in worst:
            w.writerow({k: row.get(k, "") for k in worst_fields})

def _detector_overlap_fraction(det_start: float, det_end: float, span_start: float, span_end: float) -> float:
    det_span = max(0.0, det_end - det_start)
    if det_span <= 1e-6:
        return 0.0
    return interval_overlap_sec(det_start, det_end, span_start, span_end) / det_span


def _assigned_detector_indices_for_runtime_row(row: dict[str, str], detectors: list[dict[str, float | int]]) -> list[int]:
    mapped_start = _row_float(row, "MappedAudioStartSec")
    mapped_end = _row_float(row, "MappedAudioEndSec")
    audio_idx = _row_int(row, "AudioIslandIndex", -1)
    if mapped_end <= mapped_start and audio_idx < 0:
        return []
    out: list[int] = []
    for det in detectors:
        idx = int(det["index"])
        det_start = float(det["start"])
        det_end = float(det["end"])
        if idx == audio_idx or _detector_overlap_fraction(det_start, det_end, mapped_start, mapped_end) >= 0.50:
            out.append(idx)
    return sorted(set(out))


def _detector_audit_category(row: dict[str, str], assigned_indices: list[int], detectors: list[dict[str, float | int]]) -> str:
    audio_idx = _row_int(row, "AudioIslandIndex", -1)
    assignment_category = row.get("AssignmentCategory", "")
    if not detectors:
        return "NoDetectorFragments"
    if audio_idx < 0 and not assigned_indices:
        return "NoAssignedDetectorIsland"
    if len(assigned_indices) <= 0:
        return "MappedSpanWithoutDetectorOverlap"
    if len(assigned_indices) == 1:
        return "AssignedSingleFragment"
    if "Fragmented" in assignment_category or len(assigned_indices) > 1:
        return "AssignedMultipleFragments"
    return "AssignedDetectorEvidence"


def derive_detector_evidence_rows(case_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    runtime_rows = sorted(read_csv_rows(case_dir / "runtime_island_alignment.csv"), key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    detectors = _assignment_detector_fragments(case_dir)
    if not runtime_rows and not detectors:
        return [], [], []

    # Per-text-island detector evidence/adoption audit.
    evidence_rows: list[dict[str, str]] = []
    detector_to_text: dict[int, list[int]] = {int(d["index"]): [] for d in detectors}
    detector_to_overlap: dict[int, float] = {int(d["index"]): 0.0 for d in detectors}

    for text_pos, row in enumerate(runtime_rows):
        text_idx = _row_int(row, "TextIslandIndex", text_pos)
        assigned_indices = _assigned_detector_indices_for_runtime_row(row, detectors)
        for idx in assigned_indices:
            detector_to_text.setdefault(idx, []).append(text_idx)
            det = next((d for d in detectors if int(d["index"]) == idx), None)
            if det:
                detector_to_overlap[idx] = max(detector_to_overlap.get(idx, 0.0), _detector_overlap_fraction(float(det["start"]), float(det["end"]), _row_float(row, "MappedAudioStartSec"), _row_float(row, "MappedAudioEndSec")))

        assigned_starts = [float(d["start"]) for d in detectors if int(d["index"]) in assigned_indices]
        assigned_ends = [float(d["end"]) for d in detectors if int(d["index"]) in assigned_indices]
        assigned_start = min(assigned_starts) if assigned_starts else None
        assigned_end = max(assigned_ends) if assigned_ends else None
        assigned_speech = sum(max(0.0, float(d["end"]) - float(d["start"])) for d in detectors if int(d["index"]) in assigned_indices)
        assigned_gap = 0.0
        if len(assigned_indices) > 1:
            ordered = [d for d in detectors if int(d["index"]) in assigned_indices]
            ordered.sort(key=lambda d: float(d["start"]))
            for a, b in zip(ordered, ordered[1:]):
                assigned_gap += max(0.0, float(b["start"]) - float(a["end"]))
        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        mapped_span = max(0.0, mapped_end - mapped_start)
        assigned_span = max(0.0, (assigned_end or 0.0) - (assigned_start or 0.0)) if assigned_start is not None and assigned_end is not None else 0.0
        evidence_rows.append({
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EventCount": row.get("EventCount", ""),
            "PlannerDurationSec": row.get("PlannerDurationSec", ""),
            "MappedAudioStartSec": row.get("MappedAudioStartSec", ""),
            "MappedAudioEndSec": row.get("MappedAudioEndSec", ""),
            "MappedAudioSpanSec": _fmt(mapped_span),
            "RuntimeAudioIslandIndex": row.get("AudioIslandIndex", ""),
            "RuntimeAssignmentCategory": row.get("AssignmentCategory", ""),
            "DetectorFragmentCountInCase": str(len(detectors)),
            "AssignedDetectorFragments": ";".join(str(i) for i in assigned_indices),
            "AssignedDetectorFragmentCount": str(len(assigned_indices)),
            "AssignedDetectorStartSec": _fmt(assigned_start) if assigned_start is not None else "",
            "AssignedDetectorEndSec": _fmt(assigned_end) if assigned_end is not None else "",
            "AssignedDetectorSpanSec": _fmt(assigned_span),
            "AssignedDetectorSpeechSec": _fmt(assigned_speech),
            "AssignedDetectorGapSec": _fmt(assigned_gap),
            "AssignedSpeechToMappedSpanRatio": _fmt(assigned_speech / mapped_span) if mapped_span > 1e-6 else "",
            "AssignedSpanToMappedSpanRatio": _fmt(assigned_span / mapped_span) if mapped_span > 1e-6 else "",
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            "DetectorAuditCategory": _detector_audit_category(row, assigned_indices, detectors),
        })

    # Per-detector-fragment adoption/orphan audit.
    adoption_rows: list[dict[str, str]] = []
    for det in detectors:
        idx = int(det["index"])
        adopted_by = sorted(set(detector_to_text.get(idx, [])))
        det_start = float(det["start"])
        det_end = float(det["end"])
        # Basic orphan location context.
        if adopted_by:
            reason = "adopted_by_text_island"
        elif not runtime_rows:
            reason = "orphaned_no_text_islands"
        else:
            before = [r for r in runtime_rows if _row_float(r, "MappedAudioEndSec") <= det_start]
            after = [r for r in runtime_rows if _row_float(r, "MappedAudioStartSec") >= det_end]
            if before and after:
                reason = "orphaned_between_text_islands"
            elif after:
                reason = "orphaned_before_first_text_island"
            elif before:
                reason = "orphaned_after_last_text_island"
            else:
                reason = "orphaned_overlapping_text_span_but_unclaimed"
        adoption_rows.append({
            "case_id": case_dir.name,
            "DetectorFragmentIndex": str(idx),
            "FragmentStartSec": _fmt(det_start),
            "FragmentEndSec": _fmt(det_end),
            "FragmentSpanSec": _fmt(max(0.0, det_end - det_start)),
            "Adopted": "1" if adopted_by else "0",
            "AdoptedByTextIslands": ";".join(str(i) for i in adopted_by),
            "AdoptionOverlapRatio": _fmt(detector_to_overlap.get(idx, 0.0)),
            "Reason": reason,
            "TextIslandCount": str(len(runtime_rows)),
        })

    # Per-case fragmentation rollup.
    spans = [max(0.0, float(d["end"]) - float(d["start"])) for d in detectors]
    gaps: list[float] = []
    ordered = sorted(detectors, key=lambda d: float(d["start"]))
    for a, b in zip(ordered, ordered[1:]):
        gaps.append(max(0.0, float(b["start"]) - float(a["end"])))
    speech_duration = sum(spans)
    case_start = float(ordered[0]["start"]) if ordered else 0.0
    case_end = float(ordered[-1]["end"]) if ordered else 0.0
    outer_span = max(0.0, case_end - case_start)
    frag_rows = [{
        "case_id": case_dir.name,
        "TextIslandCount": str(len(runtime_rows)),
        "SpeechFragmentCount": str(len(detectors)),
        "EndedSpeechFragmentCount": str(sum(1 for r in read_csv_rows(case_dir / "speech_islands.csv") if truthy_csv(r.get("Ended", "")))),
        "SpeechDurationSec": _fmt(speech_duration),
        "OuterSpeechSpanSec": _fmt(outer_span),
        "PauseDurationBetweenFragmentsSec": _fmt(sum(gaps)),
        "MeanFragmentSpanSec": _fmt(mean(spans)) if spans else "",
        "MedianFragmentSpanSec": _fmt(median(spans)) if spans else "",
        "LargestFragmentGapSec": _fmt(max(gaps)) if gaps else "0.000000",
        "FragmentsPerTextIsland": _fmt(len(detectors) / len(runtime_rows)) if runtime_rows else "",
        "AdoptedFragmentCount": str(sum(1 for r in adoption_rows if r.get("Adopted") == "1")),
        "OrphanedFragmentCount": str(sum(1 for r in adoption_rows if r.get("Adopted") != "1")),
    }]
    return evidence_rows, adoption_rows, frag_rows


def collect_detector_evidence_audit_rows(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    evidence: list[dict[str, str]] = []
    adoption: list[dict[str, str]] = []
    fragmentation: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return evidence, adoption, fragmentation
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        e, a, f = derive_detector_evidence_rows(case_dir)
        evidence.extend(e)
        adoption.extend(a)
        fragmentation.extend(f)
    return evidence, adoption, fragmentation




def _au15_join_indices(indices: list[int]) -> str:
    return ";".join(str(i) for i in sorted(set(indices)))


def _au15_detector_positions(detectors: list[dict[str, float | int]]) -> dict[int, int]:
    return {int(det["index"]): pos for pos, det in enumerate(detectors)}


def _au15_span_for_detector_indices(detectors: list[dict[str, float | int]], indices: list[int]) -> tuple[float | None, float | None, float, float]:
    selected = [det for det in detectors if int(det["index"]) in set(indices)]
    if not selected:
        return None, None, 0.0, 0.0
    selected.sort(key=lambda d: float(d["start"]))
    start = min(float(d["start"]) for d in selected)
    end = max(float(d["end"]) for d in selected)
    speech = sum(max(0.0, float(d["end"]) - float(d["start"])) for d in selected)
    gap = 0.0
    for a, b in zip(selected, selected[1:]):
        gap += max(0.0, float(b["start"]) - float(a["end"]))
    return start, end, speech, gap


def _au15_detector_indices_overlapping_interval(
    detectors: list[dict[str, float | int]],
    start: float,
    end: float,
) -> list[int]:
    if end <= start:
        return []
    out: list[int] = []
    for det in detectors:
        det_start = float(det["start"])
        det_end = float(det["end"])
        if interval_overlap_sec(det_start, det_end, start, end) > 0.001:
            out.append(int(det["index"]))
    return sorted(set(out))


def _au15_contiguous_indices_between(detectors: list[dict[str, float | int]], indices: list[int]) -> list[int]:
    if not indices:
        return []
    pos_by_idx = _au15_detector_positions(detectors)
    positions = [pos_by_idx[i] for i in indices if i in pos_by_idx]
    if not positions:
        return []
    lo = min(positions)
    hi = max(positions)
    return [int(detectors[pos]["index"]) for pos in range(lo, hi + 1)]


def _au15_region_category(assigned_cov: float, merged_cov: float, full_count: int, full_gap: float) -> str:
    if full_count <= 0:
        return "NoDetectorEvidenceRegion"
    if assigned_cov >= 0.90:
        return "AssignedCoversRegion"
    if merged_cov >= 0.90 and assigned_cov < 0.90:
        return "AssignedStopsInsideRecoverableRegion"
    if full_count == 1:
        return "SingleFragmentRegionUnderCovered"
    if full_gap >= 0.500:
        return "MultiFragmentLargeGapRegion"
    return "MultiFragmentRegionUnderCovered"


def derive_speech_evidence_region_rows(case_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    runtime_rows = sorted(read_csv_rows(case_dir / "runtime_island_alignment.csv"), key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    detectors = _assignment_detector_fragments(case_dir)
    if not runtime_rows:
        return [], []

    region_rows: list[dict[str, str]] = []
    breakdown_rows: list[dict[str, str]] = []

    for text_pos, row in enumerate(runtime_rows):
        text_idx = _row_int(row, "TextIslandIndex", text_pos)
        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        mapped_span = max(0.0, mapped_end - mapped_start)
        evidence_start = _row_float(row, "EvidenceFirstStartSec")
        evidence_end = _row_float(row, "EvidenceLastEndSec")
        if evidence_end <= evidence_start:
            evidence_start = _row_float(row, "EvidenceFirstCenterSec")
            evidence_end = _row_float(row, "EvidenceLastCenterSec")
        if evidence_end <= evidence_start:
            evidence_start = mapped_start
            evidence_end = mapped_end

        assigned_indices = _assigned_detector_indices_for_runtime_row(row, detectors)
        assigned_start, assigned_end, assigned_speech, assigned_gap = _au15_span_for_detector_indices(detectors, assigned_indices)
        assigned_span = max(0.0, (assigned_end or 0.0) - (assigned_start or 0.0)) if assigned_start is not None and assigned_end is not None else 0.0

        merged_assigned_indices = _au15_contiguous_indices_between(detectors, assigned_indices)
        merged_start, merged_end, merged_speech, merged_gap = _au15_span_for_detector_indices(detectors, merged_assigned_indices)
        merged_span = max(0.0, (merged_end or 0.0) - (merged_start or 0.0)) if merged_start is not None and merged_end is not None else 0.0

        evidence_overlap_indices = _au15_detector_indices_overlapping_interval(detectors, evidence_start, evidence_end)
        full_region_seed = sorted(set(evidence_overlap_indices + assigned_indices))
        full_region_indices = _au15_contiguous_indices_between(detectors, full_region_seed)
        full_start, full_end, full_speech, full_gap = _au15_span_for_detector_indices(detectors, full_region_indices)
        full_span = max(0.0, (full_end or 0.0) - (full_start or 0.0)) if full_start is not None and full_end is not None else 0.0

        assigned_cov = assigned_span / full_span if full_span > 1e-6 else 0.0
        merged_cov = merged_span / full_span if full_span > 1e-6 else 0.0
        speech_cov = assigned_speech / full_speech if full_speech > 1e-6 else 0.0
        region_category = _au15_region_category(assigned_cov, merged_cov, len(full_region_indices), full_gap)

        region_rows.append({
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EventCount": row.get("EventCount", ""),
            "AssignmentCategory": row.get("AssignmentCategory", ""),
            "AudioIslandIndex": row.get("AudioIslandIndex", ""),
            "MappedAudioStartSec": _fmt(mapped_start),
            "MappedAudioEndSec": _fmt(mapped_end),
            "MappedAudioSpanSec": _fmt(mapped_span),
            "EvidenceFirstStartSec": _fmt(evidence_start),
            "EvidenceLastEndSec": _fmt(evidence_end),
            "EvidenceRegionRawSpanSec": _fmt(max(0.0, evidence_end - evidence_start)),
            "AssignedDetectorFragments": _au15_join_indices(assigned_indices),
            "AssignedDetectorFragmentCount": str(len(assigned_indices)),
            "AssignedDetectorStartSec": _fmt(assigned_start) if assigned_start is not None else "",
            "AssignedDetectorEndSec": _fmt(assigned_end) if assigned_end is not None else "",
            "AssignedDetectorSpanSec": _fmt(assigned_span),
            "AssignedDetectorSpeechSec": _fmt(assigned_speech),
            "AssignedDetectorGapSec": _fmt(assigned_gap),
            "MergedAssignedDetectorFragments": _au15_join_indices(merged_assigned_indices),
            "MergedAssignedDetectorFragmentCount": str(len(merged_assigned_indices)),
            "MergedAssignedDetectorStartSec": _fmt(merged_start) if merged_start is not None else "",
            "MergedAssignedDetectorEndSec": _fmt(merged_end) if merged_end is not None else "",
            "MergedAssignedDetectorSpanSec": _fmt(merged_span),
            "MergedAssignedDetectorSpeechSec": _fmt(merged_speech),
            "MergedAssignedDetectorGapSec": _fmt(merged_gap),
            "FullEvidenceRegionFragments": _au15_join_indices(full_region_indices),
            "FullEvidenceRegionFragmentCount": str(len(full_region_indices)),
            "FullEvidenceRegionStartSec": _fmt(full_start) if full_start is not None else "",
            "FullEvidenceRegionEndSec": _fmt(full_end) if full_end is not None else "",
            "FullEvidenceRegionSpanSec": _fmt(full_span),
            "FullEvidenceRegionSpeechSec": _fmt(full_speech),
            "FullEvidenceRegionGapSec": _fmt(full_gap),
            "AssignedCoverageRatio": _fmt(assigned_cov) if full_span > 1e-6 else "",
            "MergedCoverageRatio": _fmt(merged_cov) if full_span > 1e-6 else "",
            "AssignedSpeechCoverageRatio": _fmt(speech_cov) if full_speech > 1e-6 else "",
            "ExtraEvidenceSpanSec": _fmt(max(0.0, full_span - assigned_span)),
            "ExtraEvidenceSpeechSec": _fmt(max(0.0, full_speech - assigned_speech)),
            "EvidenceRegionCategory": region_category,
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
        })

        # Per-fragment breakdown for the region.  Include assigned-only fragments even if no full region was found.
        breakdown_indices = full_region_indices or assigned_indices
        sorted_region = [det for det in detectors if int(det["index"]) in set(breakdown_indices)]
        sorted_region.sort(key=lambda d: float(d["start"]))
        for local_i, det in enumerate(sorted_region):
            idx = int(det["index"])
            det_start = float(det["start"])
            det_end = float(det["end"])
            prev_gap = ""
            next_gap = ""
            if local_i > 0:
                prev_gap = _fmt(max(0.0, det_start - float(sorted_region[local_i - 1]["end"])))
            if local_i + 1 < len(sorted_region):
                next_gap = _fmt(max(0.0, float(sorted_region[local_i + 1]["start"]) - det_end))
            breakdown_rows.append({
                "case_id": case_dir.name,
                "TextIslandIndex": str(text_idx),
                "FragmentIndex": str(idx),
                "FragmentStartSec": _fmt(det_start),
                "FragmentEndSec": _fmt(det_end),
                "FragmentSpanSec": _fmt(max(0.0, det_end - det_start)),
                "Assigned": "1" if idx in assigned_indices else "0",
                "MergedAssigned": "1" if idx in merged_assigned_indices else "0",
                "InFullEvidenceRegion": "1" if idx in full_region_indices else "0",
                "OverlapsRawEvidenceInterval": "1" if idx in evidence_overlap_indices else "0",
                "GapFromPreviousInRegionSec": prev_gap,
                "GapToNextInRegionSec": next_gap,
                "EvidenceRegionCategory": region_category,
                "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            })

    return region_rows, breakdown_rows


def collect_speech_evidence_region_rows(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    region_rows: list[dict[str, str]] = []
    breakdown_rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return region_rows, breakdown_rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        regions, breakdown = derive_speech_evidence_region_rows(case_dir)
        region_rows.extend(regions)
        breakdown_rows.extend(breakdown)
    return region_rows, breakdown_rows


def _au15_coverage_bucket(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NoRegion"
    if value >= 0.90:
        return "Coverage_90_100"
    if value >= 0.75:
        return "Coverage_75_90"
    if value >= 0.50:
        return "Coverage_50_75"
    if value >= 0.25:
        return "Coverage_25_50"
    return "Coverage_0_25"


def write_corpus_speech_evidence_region_summaries(run_dir: Path) -> None:
    region_rows, breakdown_rows = collect_speech_evidence_region_rows(run_dir)
    if not region_rows and not breakdown_rows:
        return

    if region_rows:
        fields = unique_in_order([key for row in region_rows for key in row.keys()])
        with (run_dir / "corpus_speech_evidence_regions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in region_rows:
                w.writerow({k: row.get(k, "") for k in fields})

        summary_groups: dict[str, list[dict[str, str]]] = {}
        for row in region_rows:
            cat = row.get("EvidenceRegionCategory", "Unknown")
            summary_groups.setdefault(cat, []).append(row)
            cov = parse_float(row.get("AssignedCoverageRatio", ""))
            summary_groups.setdefault(_au15_coverage_bucket(cov), []).append(row)

        summary_fields = [
            "Group", "Count", "MeanStreamingAbsErrorMs", "MeanAssignedCoverageRatio", "MeanMergedCoverageRatio",
            "MeanAssignedDetectorSpanSec", "MeanFullEvidenceRegionSpanSec", "MeanExtraEvidenceSpanSec",
            "MeanFullEvidenceRegionFragmentCount", "MeanFullEvidenceRegionGapSec",
        ]
        with (run_dir / "corpus_speech_evidence_region_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for group, rows in sorted(summary_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                w.writerow({
                    "Group": group,
                    "Count": str(len(rows)),
                    "MeanStreamingAbsErrorMs": _fmt(mean(numeric_values(rows, "MeanStreamingAbsErrorMs"))) if numeric_values(rows, "MeanStreamingAbsErrorMs") else "",
                    "MeanAssignedCoverageRatio": _fmt(mean(numeric_values(rows, "AssignedCoverageRatio"))) if numeric_values(rows, "AssignedCoverageRatio") else "",
                    "MeanMergedCoverageRatio": _fmt(mean(numeric_values(rows, "MergedCoverageRatio"))) if numeric_values(rows, "MergedCoverageRatio") else "",
                    "MeanAssignedDetectorSpanSec": _fmt(mean(numeric_values(rows, "AssignedDetectorSpanSec"))) if numeric_values(rows, "AssignedDetectorSpanSec") else "",
                    "MeanFullEvidenceRegionSpanSec": _fmt(mean(numeric_values(rows, "FullEvidenceRegionSpanSec"))) if numeric_values(rows, "FullEvidenceRegionSpanSec") else "",
                    "MeanExtraEvidenceSpanSec": _fmt(mean(numeric_values(rows, "ExtraEvidenceSpanSec"))) if numeric_values(rows, "ExtraEvidenceSpanSec") else "",
                    "MeanFullEvidenceRegionFragmentCount": _fmt(mean(numeric_values(rows, "FullEvidenceRegionFragmentCount"))) if numeric_values(rows, "FullEvidenceRegionFragmentCount") else "",
                    "MeanFullEvidenceRegionGapSec": _fmt(mean(numeric_values(rows, "FullEvidenceRegionGapSec"))) if numeric_values(rows, "FullEvidenceRegionGapSec") else "",
                })

        worst_fields = [
            "case_id", "TextIslandIndex", "FirstWord", "LastWord", "EvidenceRegionCategory",
            "AssignedDetectorFragments", "FullEvidenceRegionFragments", "AssignedCoverageRatio", "MergedCoverageRatio",
            "AssignedDetectorSpanSec", "FullEvidenceRegionSpanSec", "ExtraEvidenceSpanSec",
            "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs",
        ]
        worst = sorted(region_rows, key=lambda r: (_row_float(r, "AssignedCoverageRatio", 1.0), -_row_float(r, "MeanStreamingAbsErrorMs", 0.0)))[:200]
        with (run_dir / "corpus_speech_evidence_region_worst.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=worst_fields)
            w.writeheader()
            for row in worst:
                w.writerow({k: row.get(k, "") for k in worst_fields})

    if breakdown_rows:
        fields = unique_in_order([key for row in breakdown_rows for key in row.keys()])
        with (run_dir / "corpus_evidence_region_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in breakdown_rows:
                w.writerow({k: row.get(k, "") for k in fields})



def _au15_nearest_detector_to_time(detectors: list[dict[str, float | int]], target_sec: float) -> dict[str, float | int] | None:
    best = None
    best_dist = float("inf")
    for det in detectors:
        start = float(det["start"])
        end = float(det["end"])
        if target_sec < start:
            dist = start - target_sec
        elif target_sec > end:
            dist = target_sec - end
        else:
            dist = 0.0
        if dist < best_dist:
            best = det
            best_dist = dist
    return best


def _au15_start_detection_category(row: dict[str, str]) -> str:
    assigned_count = _row_int(row, "AssignedDetectorFragmentCount", 0)
    nearest_distance_ms = abs(_row_float(row, "NearestDetectorDistanceToEvidenceStartMs", 999999.0))
    assigned_start_error = parse_float(row.get("AssignedStartErrorToEvidenceStartMs", ""))
    mapped_start_error = parse_float(row.get("MappedStartErrorToEvidenceStartMs", ""))
    best_start_error = parse_float(row.get("BestStartErrorToEvidenceStartMs", ""))
    assignment_category = row.get("AssignmentCategory", "")

    if assigned_count <= 0:
        if nearest_distance_ms <= 250.0:
            return "DetectorEvidenceExistsButNoAssignedStart"
        return "NoNearbyDetectorStartEvidence"

    if assignment_category in {"WrongDetectorIsland", "LowDetectorOverlap", "AssignedLowOverlap"} and best_start_error is not None:
        if assigned_start_error is None or abs(best_start_error) + 120.0 < abs(assigned_start_error):
            return "AssignedStartLikelyWrongFragment"

    if assigned_start_error is None:
        return "MissingAssignedStartMetric"
    if abs(assigned_start_error) <= 80.0:
        return "StartAligned_0_80ms"
    if abs(assigned_start_error) <= 160.0:
        return "StartNear_80_160ms"
    if assigned_start_error < -160.0:
        return "AssignedStartEarly"
    return "AssignedStartLate"


def derive_start_detection_rows(case_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    runtime_rows = sorted(read_csv_rows(case_dir / "runtime_island_alignment.csv"), key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    text_assignment_rows = {(_row_int(r, "TextIslandIndex", -1)): r for r in read_csv_rows(case_dir / "corpus_text_audio_island_assignment.csv")}
    # Per-case files do not contain corpus_text_audio_island_assignment.csv; derive the same rows directly when needed.
    if not text_assignment_rows:
        text_assignment_rows = {(_row_int(r, "TextIslandIndex", -1)): r for r in derive_text_audio_island_assignment_rows(case_dir) if r.get("TextIslandIndex", "") != ""}
    detectors = _assignment_detector_fragments(case_dir)
    if not runtime_rows:
        return [], []

    detail_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []

    for text_pos, row in enumerate(runtime_rows):
        text_idx = _row_int(row, "TextIslandIndex", text_pos)
        assignment_row = text_assignment_rows.get(text_idx, {})
        evidence_start = _row_float(row, "EvidenceFirstStartSec")
        evidence_first_center = _row_float(row, "EvidenceFirstCenterSec")
        if evidence_start <= 0.0 and evidence_first_center > 0.0:
            evidence_start = evidence_first_center
        mapped_start = _row_float(row, "MappedAudioStartSec")
        detector_start = _row_float(row, "DetectorAudioStartSec")
        best_start = _row_float(assignment_row, "BestDetectorStartSec")
        assigned_indices = _assigned_detector_indices_for_runtime_row(row, detectors)
        assigned_start, assigned_end, assigned_speech, assigned_gap = _au15_span_for_detector_indices(detectors, assigned_indices)
        nearest = _au15_nearest_detector_to_time(detectors, evidence_start)

        nearest_idx = ""
        nearest_start = None
        nearest_end = None
        nearest_dist_ms = None
        if nearest is not None:
            nearest_idx = str(int(nearest["index"]))
            nearest_start = float(nearest["start"])
            nearest_end = float(nearest["end"])
            if evidence_start < nearest_start:
                nearest_dist_ms = (nearest_start - evidence_start) * 1000.0
            elif evidence_start > nearest_end:
                nearest_dist_ms = (evidence_start - nearest_end) * 1000.0
            else:
                nearest_dist_ms = 0.0

        assigned_start_error_ms = (assigned_start - evidence_start) * 1000.0 if assigned_start is not None else None
        detector_start_error_ms = (detector_start - evidence_start) * 1000.0 if detector_start > 0.0 else None
        mapped_start_error_ms = (mapped_start - evidence_start) * 1000.0
        best_start_error_ms = (best_start - evidence_start) * 1000.0 if best_start > 0.0 else None
        nearest_start_error_ms = (nearest_start - evidence_start) * 1000.0 if nearest_start is not None else None

        # Nearby detector candidates for start diagnosis.  This is deliberately wider
        # than the runtime launch window so it can show missed/rejected candidates.
        nearby_indices: list[int] = []
        for det in detectors:
            det_idx = int(det["index"])
            det_start = float(det["start"])
            det_end = float(det["end"])
            distance_to_evidence = 0.0
            if evidence_start < det_start:
                distance_to_evidence = det_start - evidence_start
            elif evidence_start > det_end:
                distance_to_evidence = evidence_start - det_end
            overlaps_early_evidence = interval_overlap_sec(det_start, det_end, evidence_start, evidence_start + 0.350) > 0.0
            if distance_to_evidence <= 0.750 or overlaps_early_evidence:
                nearby_indices.append(det_idx)
                candidate_rows.append({
                    "case_id": case_dir.name,
                    "TextIslandIndex": str(text_idx),
                    "FragmentIndex": str(det_idx),
                    "FragmentStartSec": _fmt(det_start),
                    "FragmentEndSec": _fmt(det_end),
                    "FragmentSpanSec": _fmt(max(0.0, det_end - det_start)),
                    "EvidenceFirstStartSec": _fmt(evidence_start),
                    "StartDistanceToEvidenceMs": _fmt(distance_to_evidence * 1000.0),
                    "StartErrorToEvidenceMs": _fmt((det_start - evidence_start) * 1000.0),
                    "AssignedStartFragment": "1" if det_idx in assigned_indices else "0",
                    "BestOverlapFragment": "1" if str(det_idx) == assignment_row.get("BestOverlapAudioIslandIndex", "") else "0",
                    "NearestStartFragment": "1" if nearest_idx == str(det_idx) else "0",
                    "FirstWord": row.get("FirstWord", ""),
                    "LastWord": row.get("LastWord", ""),
                    "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
                    "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
                })

        out = {
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EventCount": row.get("EventCount", ""),
            "AssignmentCategory": assignment_row.get("AssignmentCategory", row.get("AssignmentCategory", "")),
            "AudioIslandIndex": row.get("AudioIslandIndex", ""),
            "BestOverlapAudioIslandIndex": assignment_row.get("BestOverlapAudioIslandIndex", ""),
            "MappedAudioStartSec": _fmt(mapped_start),
            "DetectorAudioStartSec": _fmt(detector_start) if detector_start > 0.0 else "",
            "EvidenceFirstStartSec": _fmt(evidence_start),
            "EvidenceFirstCenterSec": row.get("EvidenceFirstCenterSec", ""),
            "CommittedFirstCenterSec": row.get("CommittedFirstCenterSec", ""),
            "AssignedDetectorFragments": _au15_join_indices(assigned_indices),
            "AssignedDetectorFragmentCount": str(len(assigned_indices)),
            "AssignedDetectorStartSec": _fmt(assigned_start) if assigned_start is not None else "",
            "AssignedDetectorEndSec": _fmt(assigned_end) if assigned_end is not None else "",
            "BestDetectorStartSec": _fmt(best_start) if best_start > 0.0 else "",
            "NearestDetectorIndex": nearest_idx,
            "NearestDetectorStartSec": _fmt(nearest_start) if nearest_start is not None else "",
            "NearestDetectorEndSec": _fmt(nearest_end) if nearest_end is not None else "",
            "NearbyDetectorStartCandidateCount": str(len(nearby_indices)),
            "NearbyDetectorStartCandidates": ";".join(str(i) for i in nearby_indices),
            "MappedStartErrorToEvidenceStartMs": _fmt(mapped_start_error_ms),
            "DetectorStartErrorToEvidenceStartMs": _fmt(detector_start_error_ms) if detector_start_error_ms is not None else "",
            "AssignedStartErrorToEvidenceStartMs": _fmt(assigned_start_error_ms) if assigned_start_error_ms is not None else "",
            "BestStartErrorToEvidenceStartMs": _fmt(best_start_error_ms) if best_start_error_ms is not None else "",
            "NearestDetectorStartErrorToEvidenceStartMs": _fmt(nearest_start_error_ms) if nearest_start_error_ms is not None else "",
            "NearestDetectorDistanceToEvidenceStartMs": _fmt(nearest_dist_ms) if nearest_dist_ms is not None else "",
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
        }
        out["StartDetectionCategory"] = _au15_start_detection_category(out)
        detail_rows.append(out)

    return detail_rows, candidate_rows


def collect_start_detection_rows(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    detail_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return detail_rows, candidate_rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        d, c = derive_start_detection_rows(case_dir)
        detail_rows.extend(d)
        candidate_rows.extend(c)
    return detail_rows, candidate_rows


def write_corpus_start_detection_summaries(run_dir: Path) -> None:
    detail_rows, candidate_rows = collect_start_detection_rows(run_dir)
    if not detail_rows and not candidate_rows:
        return

    if detail_rows:
        fields = unique_in_order([key for row in detail_rows for key in row.keys()])
        with (run_dir / "corpus_start_detection.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in detail_rows:
                w.writerow({k: row.get(k, "") for k in fields})

        by_cat: dict[str, list[dict[str, str]]] = {}
        for row in detail_rows:
            by_cat.setdefault(row.get("StartDetectionCategory", "Unknown"), []).append(row)

        summary_fields = [
            "StartDetectionCategory", "Count", "MeanAbsAssignedStartErrorMs", "MeanAbsMappedStartErrorMs",
            "MeanAbsNearestDetectorStartErrorMs", "MeanFirstCenterAbsErrorMs", "MeanStreamingAbsErrorMs",
            "MeanNearbyDetectorStartCandidateCount",
        ]
        with (run_dir / "corpus_start_detection_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for cat, rows in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                w.writerow({
                    "StartDetectionCategory": cat,
                    "Count": str(len(rows)),
                    "MeanAbsAssignedStartErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "AssignedStartErrorToEvidenceStartMs")])) if numeric_values(rows, "AssignedStartErrorToEvidenceStartMs") else "",
                    "MeanAbsMappedStartErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "MappedStartErrorToEvidenceStartMs")])) if numeric_values(rows, "MappedStartErrorToEvidenceStartMs") else "",
                    "MeanAbsNearestDetectorStartErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "NearestDetectorStartErrorToEvidenceStartMs")])) if numeric_values(rows, "NearestDetectorStartErrorToEvidenceStartMs") else "",
                    "MeanFirstCenterAbsErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "FirstCenterErrorMs")])) if numeric_values(rows, "FirstCenterErrorMs") else "",
                    "MeanStreamingAbsErrorMs": _fmt(mean(numeric_values(rows, "MeanStreamingAbsErrorMs"))) if numeric_values(rows, "MeanStreamingAbsErrorMs") else "",
                    "MeanNearbyDetectorStartCandidateCount": _fmt(mean(numeric_values(rows, "NearbyDetectorStartCandidateCount"))) if numeric_values(rows, "NearbyDetectorStartCandidateCount") else "",
                })

        stat_fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        with (run_dir / "corpus_start_detection_stats.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=stat_fields)
            w.writeheader()
            slices = {
                "all_text_islands": detail_rows,
                "start_aligned_or_near": [r for r in detail_rows if r.get("StartDetectionCategory") in {"StartAligned_0_80ms", "StartNear_80_160ms"}],
                "start_problematic": [r for r in detail_rows if r.get("StartDetectionCategory") not in {"StartAligned_0_80ms", "StartNear_80_160ms"}],
                "detector_evidence_exists_but_unassigned": [r for r in detail_rows if r.get("StartDetectionCategory") == "DetectorEvidenceExistsButNoAssignedStart"],
            }
            for slice_name, rows in slices.items():
                for metric in [
                    "AssignedStartErrorToEvidenceStartMs",
                    "MappedStartErrorToEvidenceStartMs",
                    "NearestDetectorStartErrorToEvidenceStartMs",
                    "NearestDetectorDistanceToEvidenceStartMs",
                    "FirstCenterErrorMs",
                    "MeanStreamingAbsErrorMs",
                    "NearbyDetectorStartCandidateCount",
                ]:
                    rec = {"slice": slice_name, "metric": metric}
                    rec.update(stats_for(numeric_values(rows, metric)))
                    w.writerow(rec)

        worst_fields = [
            "case_id", "TextIslandIndex", "FirstWord", "LastWord", "StartDetectionCategory",
            "AssignmentCategory", "AudioIslandIndex", "BestOverlapAudioIslandIndex",
            "AssignedDetectorFragments", "NearbyDetectorStartCandidates",
            "EvidenceFirstStartSec", "AssignedDetectorStartSec", "BestDetectorStartSec", "NearestDetectorStartSec",
            "AssignedStartErrorToEvidenceStartMs", "BestStartErrorToEvidenceStartMs", "NearestDetectorDistanceToEvidenceStartMs",
            "FirstCenterErrorMs", "MeanStreamingAbsErrorMs",
        ]
        worst = sorted(detail_rows, key=lambda r: (abs(_row_float(r, "AssignedStartErrorToEvidenceStartMs", _row_float(r, "MappedStartErrorToEvidenceStartMs", 0.0))), _row_float(r, "MeanStreamingAbsErrorMs", 0.0)), reverse=True)[:200]
        with (run_dir / "corpus_start_detection_worst.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=worst_fields)
            w.writeheader()
            for row in worst:
                w.writerow({k: row.get(k, "") for k in worst_fields})

    if candidate_rows:
        fields = unique_in_order([key for row in candidate_rows for key in row.keys()])
        with (run_dir / "corpus_start_detection_candidates.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in candidate_rows:
                w.writerow({k: row.get(k, "") for k in fields})

def _au19_late_start_forensics_category(row: dict[str, str]) -> str:
    assigned_err = parse_float(row.get("AssignedMinusEvidenceMs", ""))
    nearest_dist = _row_float(row, "NearestDetectorDistanceToEvidenceStartMs", 999999.0)
    predicted_minus_evidence = _row_float(row, "PredictedMinusEvidenceMs", 0.0)
    prev_block_ms = _row_float(row, "PreviousIslandEndMinusEvidenceMs", -999999.0)
    evidence_inside_window = row.get("EvidenceInsideLaunchWindow", "0") == "1"
    assigned_minus_nearest = _row_float(row, "AssignedMinusNearestStartMs", 0.0)
    start_cat = row.get("StartDetectionCategory", "")

    if start_cat == "DetectorEvidenceExistsButNoAssignedStart":
        return "EvidenceExistsButNoAssignedStart"
    if nearest_dist > 250.0:
        return "NoDetectorNearEvidence"
    if assigned_err is not None and assigned_err > 160.0 and assigned_minus_nearest > 160.0:
        return "AssignedToLaterFragment"
    if evidence_inside_window and assigned_err is not None and assigned_err > 160.0:
        return "EvidenceInsideWindowButSkipped"
    if prev_block_ms > 120.0:
        return "PreviousIslandBlocked"
    if predicted_minus_evidence > 160.0:
        return "LaunchWindowOpenedLate"
    return "AssignedStartLateUnclassified"


def write_corpus_start_late_forensics_summaries(run_dir: Path) -> None:
    detail_rows, candidate_rows = collect_start_detection_rows(run_dir)
    if not detail_rows:
        return

    by_case: dict[str, list[dict[str, str]]] = {}
    for row in detail_rows:
        by_case.setdefault(row.get("case_id", ""), []).append(row)
    for rows in by_case.values():
        rows.sort(key=lambda r: _row_int(r, "TextIslandIndex", 999999))

    late_rows: list[dict[str, str]] = []
    late_key_set: set[tuple[str, str]] = set()
    for case_id, rows in by_case.items():
        prev_mapped_end = 0.0
        prev_assigned_end = None
        for pos, row in enumerate(rows):
            text_idx = row.get("TextIslandIndex", "")
            start_cat = row.get("StartDetectionCategory", "")
            assigned_err = parse_float(row.get("AssignedStartErrorToEvidenceStartMs", ""))
            is_late = start_cat in {"AssignedStartLate", "AssignedStartLikelyWrongFragment", "DetectorEvidenceExistsButNoAssignedStart"}
            if assigned_err is not None and assigned_err > 160.0:
                is_late = True
            if not is_late:
                mapped_end = _row_float(row, "MappedAudioStartSec", 0.0) + max(0.0, _row_float(row, "FirstCenterErrorMs", 0.0) * 0.0)
                # corpus_start_detection does not carry mapped end; use current mapped start as a conservative cursor only.
                prev_mapped_end = max(prev_mapped_end, _row_float(row, "MappedAudioStartSec", prev_mapped_end))
                assigned_end_tmp = parse_float(row.get("AssignedDetectorEndSec", ""))
                if assigned_end_tmp is not None:
                    prev_assigned_end = assigned_end_tmp
                continue

            evidence_start = _row_float(row, "EvidenceFirstStartSec", 0.0)
            assigned_start = parse_float(row.get("AssignedDetectorStartSec", ""))
            nearest_start = parse_float(row.get("NearestDetectorStartSec", ""))
            nearest_end = parse_float(row.get("NearestDetectorEndSec", ""))
            mapped_start = _row_float(row, "MappedAudioStartSec", 0.0)
            predicted_launch = 0.0 if pos == 0 else prev_mapped_end + 0.060
            launch_window_start = max(0.0, predicted_launch - 0.250)
            launch_window_end = predicted_launch + 0.750
            evidence_inside_window = launch_window_start <= evidence_start <= launch_window_end
            nearest_inside_window = nearest_start is not None and launch_window_start <= nearest_start <= launch_window_end
            previous_end = prev_assigned_end if prev_assigned_end is not None else prev_mapped_end

            out = {
                "case_id": case_id,
                "TextIslandIndex": text_idx,
                "FirstWord": row.get("FirstWord", ""),
                "LastWord": row.get("LastWord", ""),
                "StartDetectionCategory": start_cat,
                "AssignmentCategory": row.get("AssignmentCategory", ""),
                "AudioIslandIndex": row.get("AudioIslandIndex", ""),
                "BestOverlapAudioIslandIndex": row.get("BestOverlapAudioIslandIndex", ""),
                "PredictedLaunchSec": _fmt(predicted_launch),
                "MappedAudioStartSec": row.get("MappedAudioStartSec", ""),
                "EvidenceStartSec": row.get("EvidenceFirstStartSec", ""),
                "AssignedStartSec": row.get("AssignedDetectorStartSec", ""),
                "NearestDetectorIndex": row.get("NearestDetectorIndex", ""),
                "NearestDetectorStartSec": row.get("NearestDetectorStartSec", ""),
                "NearestDetectorEndSec": row.get("NearestDetectorEndSec", ""),
                "LaunchWindowStartSec": _fmt(launch_window_start),
                "LaunchWindowEndSec": _fmt(launch_window_end),
                "EvidenceInsideLaunchWindow": "1" if evidence_inside_window else "0",
                "NearestStartInsideLaunchWindow": "1" if nearest_inside_window else "0",
                "PreviousIslandEndSec": _fmt(previous_end),
                "PreviousIslandEndMinusEvidenceMs": _fmt((previous_end - evidence_start) * 1000.0),
                "AssignedMinusEvidenceMs": row.get("AssignedStartErrorToEvidenceStartMs", ""),
                "PredictedMinusEvidenceMs": _fmt((predicted_launch - evidence_start) * 1000.0),
                "AssignedMinusPredictedMs": _fmt((assigned_start - predicted_launch) * 1000.0) if assigned_start is not None else "",
                "AssignedMinusNearestStartMs": _fmt((assigned_start - nearest_start) * 1000.0) if assigned_start is not None and nearest_start is not None else "",
                "NearestDetectorDistanceToEvidenceStartMs": row.get("NearestDetectorDistanceToEvidenceStartMs", ""),
                "NearbyDetectorStartCandidateCount": row.get("NearbyDetectorStartCandidateCount", ""),
                "NearbyDetectorStartCandidates": row.get("NearbyDetectorStartCandidates", ""),
                "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
                "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            }
            out["StartLateForensicsCategory"] = _au19_late_start_forensics_category(out)
            late_rows.append(out)
            late_key_set.add((case_id, text_idx))

            assigned_end_tmp = parse_float(row.get("AssignedDetectorEndSec", ""))
            if assigned_end_tmp is not None:
                prev_assigned_end = assigned_end_tmp
            prev_mapped_end = max(prev_mapped_end, mapped_start)

    if late_rows:
        fields = unique_in_order([key for row in late_rows for key in row.keys()])
        with (run_dir / "corpus_start_late_forensics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in late_rows:
                w.writerow({k: row.get(k, "") for k in fields})

        by_cat: dict[str, list[dict[str, str]]] = {}
        for row in late_rows:
            by_cat.setdefault(row.get("StartLateForensicsCategory", "Unknown"), []).append(row)
        summary_fields = [
            "StartLateForensicsCategory", "Count", "MeanAssignedMinusEvidenceMs", "MeanPredictedMinusEvidenceMs",
            "MeanAssignedMinusPredictedMs", "MeanPreviousIslandEndMinusEvidenceMs", "MeanNearestDetectorDistanceMs",
            "MeanFirstCenterAbsErrorMs", "MeanStreamingAbsErrorMs",
        ]
        with (run_dir / "corpus_start_late_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for cat, rows in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                w.writerow({
                    "StartLateForensicsCategory": cat,
                    "Count": str(len(rows)),
                    "MeanAssignedMinusEvidenceMs": _fmt(mean(numeric_values(rows, "AssignedMinusEvidenceMs"))) if numeric_values(rows, "AssignedMinusEvidenceMs") else "",
                    "MeanPredictedMinusEvidenceMs": _fmt(mean(numeric_values(rows, "PredictedMinusEvidenceMs"))) if numeric_values(rows, "PredictedMinusEvidenceMs") else "",
                    "MeanAssignedMinusPredictedMs": _fmt(mean(numeric_values(rows, "AssignedMinusPredictedMs"))) if numeric_values(rows, "AssignedMinusPredictedMs") else "",
                    "MeanPreviousIslandEndMinusEvidenceMs": _fmt(mean(numeric_values(rows, "PreviousIslandEndMinusEvidenceMs"))) if numeric_values(rows, "PreviousIslandEndMinusEvidenceMs") else "",
                    "MeanNearestDetectorDistanceMs": _fmt(mean(numeric_values(rows, "NearestDetectorDistanceToEvidenceStartMs"))) if numeric_values(rows, "NearestDetectorDistanceToEvidenceStartMs") else "",
                    "MeanFirstCenterAbsErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "FirstCenterErrorMs")])) if numeric_values(rows, "FirstCenterErrorMs") else "",
                    "MeanStreamingAbsErrorMs": _fmt(mean(numeric_values(rows, "MeanStreamingAbsErrorMs"))) if numeric_values(rows, "MeanStreamingAbsErrorMs") else "",
                })

        worst_fields = [
            "case_id", "TextIslandIndex", "FirstWord", "LastWord", "StartLateForensicsCategory", "StartDetectionCategory",
            "AssignmentCategory", "PredictedLaunchSec", "EvidenceStartSec", "AssignedStartSec", "NearestDetectorStartSec",
            "LaunchWindowStartSec", "LaunchWindowEndSec", "EvidenceInsideLaunchWindow", "NearestStartInsideLaunchWindow",
            "PreviousIslandEndSec", "PreviousIslandEndMinusEvidenceMs", "AssignedMinusEvidenceMs", "PredictedMinusEvidenceMs",
            "AssignedMinusPredictedMs", "AssignedMinusNearestStartMs", "NearestDetectorDistanceToEvidenceStartMs",
            "NearbyDetectorStartCandidates", "FirstCenterErrorMs", "MeanStreamingAbsErrorMs",
        ]
        worst = sorted(late_rows, key=lambda r: (_row_float(r, "AssignedMinusEvidenceMs", 0.0), _row_float(r, "MeanStreamingAbsErrorMs", 0.0)), reverse=True)[:200]
        with (run_dir / "corpus_start_late_worst.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=worst_fields)
            w.writeheader()
            for row in worst:
                w.writerow({k: row.get(k, "") for k in worst_fields})

    if candidate_rows and late_key_set:
        enriched: list[dict[str, str]] = []
        late_by_key = {(r.get("case_id", ""), r.get("TextIslandIndex", "")): r for r in late_rows}
        for cand in candidate_rows:
            key = (cand.get("case_id", ""), cand.get("TextIslandIndex", ""))
            if key not in late_key_set:
                continue
            late = late_by_key.get(key, {})
            start = _row_float(cand, "FragmentStartSec", 0.0)
            win_lo = _row_float(late, "LaunchWindowStartSec", 0.0)
            win_hi = _row_float(late, "LaunchWindowEndSec", 0.0)
            accepted = cand.get("AssignedStartFragment", "") == "1"
            if accepted:
                reason = "AcceptedAssignedStart"
            elif start < win_lo:
                reason = "BeforeLaunchWindow"
            elif start > win_hi:
                reason = "AfterLaunchWindow"
            elif cand.get("NearestStartFragment", "") == "1":
                reason = "NearestStartSkipped"
            elif cand.get("BestOverlapFragment", "") == "1":
                reason = "BestOverlapSkipped"
            else:
                reason = "CandidateSkipped"
            out = dict(cand)
            out["StartLateForensicsCategory"] = late.get("StartLateForensicsCategory", "")
            out["LaunchWindowStartSec"] = late.get("LaunchWindowStartSec", "")
            out["LaunchWindowEndSec"] = late.get("LaunchWindowEndSec", "")
            out["CandidateInLaunchWindow"] = "1" if win_lo <= start <= win_hi else "0"
            out["CandidateRejectReason"] = reason
            enriched.append(out)
        if enriched:
            fields = unique_in_order([key for row in enriched for key in row.keys()])
            with (run_dir / "corpus_start_late_candidates.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in enriched:
                    w.writerow({k: row.get(k, "") for k in fields})

def write_corpus_detector_evidence_audit_summaries(run_dir: Path) -> None:
    evidence_rows, adoption_rows, fragmentation_rows = collect_detector_evidence_audit_rows(run_dir)
    if not evidence_rows and not adoption_rows and not fragmentation_rows:
        return

    if evidence_rows:
        fields = unique_in_order([key for row in evidence_rows for key in row.keys()])
        with (run_dir / "corpus_detector_evidence.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in evidence_rows:
                w.writerow({k: row.get(k, "") for k in fields})

        by_cat: dict[str, list[dict[str, str]]] = {}
        for row in evidence_rows:
            by_cat.setdefault(row.get("DetectorAuditCategory", "Unknown"), []).append(row)
        summary_fields = [
            "DetectorAuditCategory", "Count", "MeanStreamingAbsErrorMs", "MeanFirstCenterAbsErrorMs",
            "MeanLastCenterAbsErrorMs", "MeanAssignedDetectorFragmentCount", "MeanAssignedDetectorGapSec",
            "MeanAssignedSpeechToMappedSpanRatio", "MeanAssignedSpanToMappedSpanRatio",
        ]
        with (run_dir / "corpus_detector_evidence_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for cat, rows in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                w.writerow({
                    "DetectorAuditCategory": cat,
                    "Count": str(len(rows)),
                    "MeanStreamingAbsErrorMs": _fmt(mean(numeric_values(rows, "MeanStreamingAbsErrorMs"))) if numeric_values(rows, "MeanStreamingAbsErrorMs") else "",
                    "MeanFirstCenterAbsErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "FirstCenterErrorMs")])) if numeric_values(rows, "FirstCenterErrorMs") else "",
                    "MeanLastCenterAbsErrorMs": _fmt(mean([abs(v) for v in numeric_values(rows, "LastCenterErrorMs")])) if numeric_values(rows, "LastCenterErrorMs") else "",
                    "MeanAssignedDetectorFragmentCount": _fmt(mean(numeric_values(rows, "AssignedDetectorFragmentCount"))) if numeric_values(rows, "AssignedDetectorFragmentCount") else "",
                    "MeanAssignedDetectorGapSec": _fmt(mean(numeric_values(rows, "AssignedDetectorGapSec"))) if numeric_values(rows, "AssignedDetectorGapSec") else "",
                    "MeanAssignedSpeechToMappedSpanRatio": _fmt(mean(numeric_values(rows, "AssignedSpeechToMappedSpanRatio"))) if numeric_values(rows, "AssignedSpeechToMappedSpanRatio") else "",
                    "MeanAssignedSpanToMappedSpanRatio": _fmt(mean(numeric_values(rows, "AssignedSpanToMappedSpanRatio"))) if numeric_values(rows, "AssignedSpanToMappedSpanRatio") else "",
                })
        worst_fields = [
            "case_id", "TextIslandIndex", "DetectorAuditCategory", "RuntimeAssignmentCategory", "AssignedDetectorFragments",
            "FirstWord", "LastWord", "MappedAudioSpanSec", "AssignedDetectorSpanSec", "AssignedDetectorSpeechSec",
            "AssignedDetectorGapSec", "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs",
        ]
        worst = sorted(evidence_rows, key=lambda r: _row_float(r, "MeanStreamingAbsErrorMs"), reverse=True)[:200]
        with (run_dir / "corpus_detector_evidence_worst.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=worst_fields)
            w.writeheader()
            for row in worst:
                w.writerow({k: row.get(k, "") for k in worst_fields})

    if adoption_rows:
        fields = unique_in_order([key for row in adoption_rows for key in row.keys()])
        with (run_dir / "corpus_detector_adoption.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in adoption_rows:
                w.writerow({k: row.get(k, "") for k in fields})
        by_reason: dict[str, list[dict[str, str]]] = {}
        for row in adoption_rows:
            by_reason.setdefault(row.get("Reason", "Unknown"), []).append(row)
        with (run_dir / "corpus_detector_adoption_summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields2 = ["Reason", "Count", "MeanFragmentSpanSec", "MeanAdoptionOverlapRatio"]
            w = csv.DictWriter(f, fieldnames=fields2)
            w.writeheader()
            for reason, rows in sorted(by_reason.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                w.writerow({
                    "Reason": reason,
                    "Count": str(len(rows)),
                    "MeanFragmentSpanSec": _fmt(mean(numeric_values(rows, "FragmentSpanSec"))) if numeric_values(rows, "FragmentSpanSec") else "",
                    "MeanAdoptionOverlapRatio": _fmt(mean(numeric_values(rows, "AdoptionOverlapRatio"))) if numeric_values(rows, "AdoptionOverlapRatio") else "",
                })

    if fragmentation_rows:
        fields = unique_in_order([key for row in fragmentation_rows for key in row.keys()])
        with (run_dir / "corpus_detector_fragmentation.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in fragmentation_rows:
                w.writerow({k: row.get(k, "") for k in fields})
        stat_fields = ["metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        metrics = [
            "TextIslandCount", "SpeechFragmentCount", "EndedSpeechFragmentCount", "SpeechDurationSec",
            "OuterSpeechSpanSec", "PauseDurationBetweenFragmentsSec", "MeanFragmentSpanSec", "MedianFragmentSpanSec",
            "LargestFragmentGapSec", "FragmentsPerTextIsland", "AdoptedFragmentCount", "OrphanedFragmentCount",
        ]
        with (run_dir / "corpus_detector_fragmentation_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=stat_fields)
            w.writeheader()
            for metric in metrics:
                rec = {"metric": metric}
                rec.update(stats_for(numeric_values(fragmentation_rows, metric)))
                w.writerow(rec)


def _assignment_confusion_diagnosis(current_category: str, causal_category: str) -> tuple[str, str]:
    current = current_category or "UnknownCurrent"
    causal = causal_category or "UnknownCausal"
    if current == "Healthy" and causal == "SameAsCurrent":
        return "HealthyStable", "no assignment change indicated"
    if current == "NoAssignedDetectorIsland" and causal == "CausalAssignsUnassignedTextIsland":
        return "LateOrUnclaimedDetectorAdoption", "runtime missed a detector span the causal simulator can assign"
    if current == "NoAssignedDetectorIsland" and causal == "NoCausalDetectorSpan":
        return "DetectorMissOrNoLaunch", "neither runtime nor causal simulator found a usable detector span"
    if current == "WrongDetectorIsland" and causal in {"CausalUsesDifferentDetectorSpan", "CausalImprovesOverlap"}:
        return "WrongOwnership", "runtime chose a detector span that differs from the causal/better-overlap span"
    if current == "DetectorFragmentedTextIsland" and causal == "CausalMergesFragments":
        return "UnderMergedFragments", "text island likely needs adjacent detector fragments merged"
    if current in {"AssignedDetectorTooShort", "LowDetectorOverlap", "AssignedLowOverlap"} and causal == "CausalMergesFragments":
        return "UnderCoveredIsland", "current detector coverage is too short; causal merge may cover the island"
    if current in {"AssignedDetectorTooLong", "DetectorFragmentedTextIsland"} and causal == "SameAsCurrent":
        return "PossiblyOverMergedOrTextEvidenceWide", "current mapping is stable but span/evidence relationship is suspicious"
    if current == "Healthy" and causal != "SameAsCurrent":
        return "CausalWouldChangeHealthy", "causal simulator disagrees with an apparently healthy runtime assignment; inspect before changing policy"
    if current != "Healthy" and causal == "SameAsCurrent":
        return "CurrentPolicyLimitation", "causal simulator reproduces the current assignment, so detector/assignment policy may need different evidence"
    return "MixedAssignmentIssue", "current and causal diagnostics disagree or indicate multiple weak signals"


def write_corpus_assignment_confusion_summaries(run_dir: Path) -> None:
    """Join current runtime assignment diagnostics with causal assignment simulation.

    AU10b is diagnostic-only. It does not propose a new runtime policy directly;
    instead it builds a confusion matrix that separates wrong ownership,
    unassigned/late detector adoption, fragmented-under-merged islands, detector
    misses, and cases where the simulator would change an otherwise healthy row.
    """
    current_rows = read_csv_rows(run_dir / "corpus_text_audio_island_assignment.csv")
    causal_rows = read_csv_rows(run_dir / "corpus_streaming_island_assignment_sim.csv")
    if not current_rows or not causal_rows:
        return

    def key_for(row: dict[str, str]) -> tuple[str, str]:
        return (row.get("case_id", ""), row.get("TextIslandIndex", ""))

    causal_by_key = {key_for(row): row for row in causal_rows}
    rows: list[dict[str, str]] = []
    for cur in current_rows:
        key = key_for(cur)
        cau = causal_by_key.get(key, {})
        current_category = cur.get("AssignmentCategory", "") or "UnknownCurrent"
        causal_category = cau.get("CausalAssignmentCategory", "") or "MissingCausalRow"
        diagnosis, recommendation = _assignment_confusion_diagnosis(current_category, causal_category)
        cur_overlap = _row_float(cur, "AssignedOverlapRatio", 0.0)
        best_overlap = _row_float(cur, "BestOverlapRatio", 0.0)
        causal_overlap = _row_float(cau, "CausalOverlapRatio", 0.0)
        current_causal_delta = _row_float(cau, "CausalMinusCurrentOverlapRatio", causal_overlap - cur_overlap)
        best_current_delta = best_overlap - cur_overlap
        mean_err = _row_float(cur, "MeanStreamingAbsErrorMs", _row_float(cau, "MeanStreamingAbsErrorMs", 0.0))
        first_err = _row_float(cur, "FirstCenterErrorMs", 0.0)
        last_err = _row_float(cur, "LastCenterErrorMs", 0.0)
        high_error = mean_err >= 250.0 or abs(first_err) >= 250.0 or abs(last_err) >= 250.0
        rows.append({
            "case_id": key[0],
            "TextIslandIndex": key[1],
            "CurrentAssignmentCategory": current_category,
            "CausalAssignmentCategory": causal_category,
            "AssignmentDiagnosis": diagnosis,
            "Recommendation": recommendation,
            "CurrentAudioIslandIndex": cur.get("AssignedAudioIslandIndex", ""),
            "BestOverlapAudioIslandIndex": cur.get("BestOverlapAudioIslandIndex", ""),
            "CausalDetectorIndices": cau.get("CausalDetectorIndices", ""),
            "CausalFirstAudioIslandIndex": cau.get("CausalFirstAudioIslandIndex", ""),
            "CausalLastAudioIslandIndex": cau.get("CausalLastAudioIslandIndex", ""),
            "EventCount": cur.get("EventCount", cau.get("EventCount", "")),
            "FirstWord": cur.get("FirstWord", cau.get("FirstWord", "")),
            "LastWord": cur.get("LastWord", cau.get("LastWord", "")),
            "AssignedOverlapRatio": _fmt(cur_overlap),
            "BestOverlapRatio": _fmt(best_overlap),
            "CausalOverlapRatio": _fmt(causal_overlap),
            "BestMinusAssignedOverlapRatio": _fmt(best_current_delta),
            "CausalMinusCurrentOverlapRatio": _fmt(current_causal_delta),
            "AssignedDetectorSpanSec": cur.get("AssignedDetectorSpanSec", ""),
            "BestDetectorSpanSec": cur.get("BestDetectorSpanSec", ""),
            "CausalDetectorSpanSec": cau.get("CausalDetectorSpanSec", ""),
            "CausalFragmentCount": cau.get("CausalFragmentCount", ""),
            "CausalMergedGapSeconds": cau.get("CausalMergedGapSeconds", ""),
            "CausalSpanToExpectedRatio": cau.get("CausalSpanToExpectedRatio", ""),
            "MappedAudioSpanSec": cur.get("MappedAudioSpanSec", cau.get("MappedAudioSpanSec", "")),
            "EvidenceSpanSec": cur.get("EvidenceSpanSec", cau.get("EvidenceSpanSec", "")),
            "MeanStreamingAbsErrorMs": _fmt(mean_err),
            "FirstCenterErrorMs": cur.get("FirstCenterErrorMs", cau.get("FirstCenterErrorMs", "")),
            "LastCenterErrorMs": cur.get("LastCenterErrorMs", cau.get("LastCenterErrorMs", "")),
            "HighError": "1" if high_error else "0",
            "CurrentNotes": cur.get("Notes", ""),
            "CausalNotes": cau.get("Notes", ""),
            "HandoffReason": cau.get("HandoffReason", ""),
        })

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_assignment_confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    # Cross-tab: current category x causal category.
    matrix: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        k = (row.get("CurrentAssignmentCategory", "UnknownCurrent"), row.get("CausalAssignmentCategory", "UnknownCausal"))
        rec = matrix.setdefault(k, {
            "Count": 0.0,
            "HighErrorCount": 0.0,
            "MeanStreamingAbsErrorMsSum": 0.0,
            "FirstCenterAbsErrorMsSum": 0.0,
            "LastCenterAbsErrorMsSum": 0.0,
            "AssignedOverlapRatioSum": 0.0,
            "CausalOverlapRatioSum": 0.0,
            "CausalMinusCurrentOverlapRatioSum": 0.0,
        })
        rec["Count"] += 1.0
        rec["HighErrorCount"] += 1.0 if row.get("HighError") == "1" else 0.0
        rec["MeanStreamingAbsErrorMsSum"] += _row_float(row, "MeanStreamingAbsErrorMs")
        rec["FirstCenterAbsErrorMsSum"] += abs(_row_float(row, "FirstCenterErrorMs"))
        rec["LastCenterAbsErrorMsSum"] += abs(_row_float(row, "LastCenterErrorMs"))
        rec["AssignedOverlapRatioSum"] += _row_float(row, "AssignedOverlapRatio")
        rec["CausalOverlapRatioSum"] += _row_float(row, "CausalOverlapRatio")
        rec["CausalMinusCurrentOverlapRatioSum"] += _row_float(row, "CausalMinusCurrentOverlapRatio")

    matrix_fields = [
        "CurrentAssignmentCategory", "CausalAssignmentCategory", "Count", "HighErrorCount", "HighErrorRate",
        "MeanStreamingAbsErrorMs", "MeanFirstCenterAbsErrorMs", "MeanLastCenterAbsErrorMs",
        "MeanAssignedOverlapRatio", "MeanCausalOverlapRatio", "MeanCausalMinusCurrentOverlapRatio",
    ]
    with (run_dir / "corpus_assignment_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=matrix_fields)
        w.writeheader()
        for (cur_cat, cau_cat), rec in sorted(matrix.items(), key=lambda kv: (-kv[1]["Count"], kv[0][0], kv[0][1])):
            n = max(1.0, rec["Count"])
            w.writerow({
                "CurrentAssignmentCategory": cur_cat,
                "CausalAssignmentCategory": cau_cat,
                "Count": f"{rec['Count']:.0f}",
                "HighErrorCount": f"{rec['HighErrorCount']:.0f}",
                "HighErrorRate": _fmt(rec["HighErrorCount"] / n),
                "MeanStreamingAbsErrorMs": _fmt(rec["MeanStreamingAbsErrorMsSum"] / n),
                "MeanFirstCenterAbsErrorMs": _fmt(rec["FirstCenterAbsErrorMsSum"] / n),
                "MeanLastCenterAbsErrorMs": _fmt(rec["LastCenterAbsErrorMsSum"] / n),
                "MeanAssignedOverlapRatio": _fmt(rec["AssignedOverlapRatioSum"] / n),
                "MeanCausalOverlapRatio": _fmt(rec["CausalOverlapRatioSum"] / n),
                "MeanCausalMinusCurrentOverlapRatio": _fmt(rec["CausalMinusCurrentOverlapRatioSum"] / n),
            })

    # Diagnosis-level rollup: this is the primary AU10b output for deciding the next runtime experiment.
    by_diag: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_diag.setdefault(row.get("AssignmentDiagnosis", "UnknownDiagnosis"), []).append(row)
    summary_fields = [
        "AssignmentDiagnosis", "Count", "HighErrorCount", "HighErrorRate", "MeanStreamingAbsErrorMs",
        "MeanFirstCenterAbsErrorMs", "MeanLastCenterAbsErrorMs", "MeanAssignedOverlapRatio",
        "MeanCausalOverlapRatio", "MeanCausalMinusCurrentOverlapRatio", "Recommendation",
    ]
    with (run_dir / "corpus_assignment_confusion_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for diag, diag_rows in sorted(by_diag.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            n = len(diag_rows)
            high = sum(1 for r in diag_rows if r.get("HighError") == "1")
            w.writerow({
                "AssignmentDiagnosis": diag,
                "Count": str(n),
                "HighErrorCount": str(high),
                "HighErrorRate": _fmt(high / n) if n else "",
                "MeanStreamingAbsErrorMs": _fmt(mean(numeric_values(diag_rows, "MeanStreamingAbsErrorMs"))) if numeric_values(diag_rows, "MeanStreamingAbsErrorMs") else "",
                "MeanFirstCenterAbsErrorMs": _fmt(mean([abs(v) for v in numeric_values(diag_rows, "FirstCenterErrorMs")])) if numeric_values(diag_rows, "FirstCenterErrorMs") else "",
                "MeanLastCenterAbsErrorMs": _fmt(mean([abs(v) for v in numeric_values(diag_rows, "LastCenterErrorMs")])) if numeric_values(diag_rows, "LastCenterErrorMs") else "",
                "MeanAssignedOverlapRatio": _fmt(mean(numeric_values(diag_rows, "AssignedOverlapRatio"))) if numeric_values(diag_rows, "AssignedOverlapRatio") else "",
                "MeanCausalOverlapRatio": _fmt(mean(numeric_values(diag_rows, "CausalOverlapRatio"))) if numeric_values(diag_rows, "CausalOverlapRatio") else "",
                "MeanCausalMinusCurrentOverlapRatio": _fmt(mean(numeric_values(diag_rows, "CausalMinusCurrentOverlapRatio"))) if numeric_values(diag_rows, "CausalMinusCurrentOverlapRatio") else "",
                "Recommendation": diag_rows[0].get("Recommendation", "") if diag_rows else "",
            })

    stats_fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
    metrics = [
        "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs", "AssignedOverlapRatio",
        "BestMinusAssignedOverlapRatio", "CausalOverlapRatio", "CausalMinusCurrentOverlapRatio",
        "CausalFragmentCount", "CausalMergedGapSeconds", "CausalSpanToExpectedRatio",
    ]
    with (run_dir / "corpus_assignment_confusion_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=stats_fields)
        w.writeheader()
        slices = {"all_text_islands": rows, "high_error": [r for r in rows if r.get("HighError") == "1"]}
        slices.update({diag: dr for diag, dr in by_diag.items()})
        for slice_name, slice_rows in slices.items():
            for metric in metrics:
                vals = numeric_values(slice_rows, metric)
                if metric in {"FirstCenterErrorMs", "LastCenterErrorMs"}:
                    vals = [abs(v) for v in vals]
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(vals))
                w.writerow(rec)

    worst = sorted(rows, key=lambda r: (0 if r.get("HighError") == "1" else 1, -_row_float(r, "MeanStreamingAbsErrorMs"), -abs(_row_float(r, "LastCenterErrorMs"))))[:200]
    worst_fields = [
        "case_id", "TextIslandIndex", "AssignmentDiagnosis", "CurrentAssignmentCategory", "CausalAssignmentCategory",
        "CurrentAudioIslandIndex", "BestOverlapAudioIslandIndex", "CausalDetectorIndices", "EventCount", "FirstWord", "LastWord",
        "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs", "AssignedOverlapRatio",
        "CausalOverlapRatio", "CausalMinusCurrentOverlapRatio", "Recommendation", "CurrentNotes", "CausalNotes", "HandoffReason",
    ]
    with (run_dir / "corpus_assignment_confusion_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in worst:
            w.writerow({k: row.get(k, "") for k in worst_fields})


def write_corpus_streaming_island_assignment_sim_summaries(run_dir: Path) -> None:
    rows = collect_streaming_island_assignment_sim_rows(run_dir)
    if not rows:
        return

    fields = unique_in_order([key for row in rows for key in row.keys()])
    detail_path = run_dir / "corpus_streaming_island_assignment_sim.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    categories: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get("CausalAssignmentCategory", "") or "Unknown"
        rec = categories.setdefault(cat, {
            "Count": 0.0,
            "CurrentOverlapRatioSum": 0.0,
            "CausalOverlapRatioSum": 0.0,
            "OverlapCount": 0.0,
            "CausalFragmentCountSum": 0.0,
            "CausalMergedGapSeconds": 0.0,
            "MeanStreamingAbsErrorMsSum": 0.0,
            "ErrorCount": 0.0,
        })
        rec["Count"] += 1.0
        cur = parse_float(row.get("CurrentOverlapRatio", ""))
        cau = parse_float(row.get("CausalOverlapRatio", ""))
        if cur is not None and cau is not None and math.isfinite(cur) and math.isfinite(cau):
            rec["CurrentOverlapRatioSum"] += cur
            rec["CausalOverlapRatioSum"] += cau
            rec["OverlapCount"] += 1.0
        rec["CausalFragmentCountSum"] += _row_float(row, "CausalFragmentCount")
        rec["CausalMergedGapSeconds"] += _row_float(row, "CausalMergedGapSeconds")
        err = parse_float(row.get("MeanStreamingAbsErrorMs", ""))
        if err is not None and math.isfinite(err):
            rec["MeanStreamingAbsErrorMsSum"] += err
            rec["ErrorCount"] += 1.0

    summary_path = run_dir / "corpus_streaming_island_assignment_sim_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        summary_fields = [
            "CausalAssignmentCategory",
            "Count",
            "MeanCurrentOverlapRatio",
            "MeanCausalOverlapRatio",
            "MeanOverlapRatioImprovement",
            "MeanCausalFragmentCount",
            "TotalCausalMergedGapSeconds",
            "MeanStreamingAbsErrorMs",
        ]
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for cat, rec in sorted(categories.items(), key=lambda kv: (-kv[1]["Count"], kv[0])):
            n = rec["Count"]
            overlap_n = rec["OverlapCount"]
            err_n = rec["ErrorCount"]
            cur_mean = rec["CurrentOverlapRatioSum"] / overlap_n if overlap_n > 0.0 else None
            cau_mean = rec["CausalOverlapRatioSum"] / overlap_n if overlap_n > 0.0 else None
            w.writerow({
                "CausalAssignmentCategory": cat,
                "Count": f"{n:.0f}",
                "MeanCurrentOverlapRatio": _fmt(cur_mean) if cur_mean is not None else "",
                "MeanCausalOverlapRatio": _fmt(cau_mean) if cau_mean is not None else "",
                "MeanOverlapRatioImprovement": _fmt(cau_mean - cur_mean) if cur_mean is not None and cau_mean is not None else "",
                "MeanCausalFragmentCount": _fmt(rec["CausalFragmentCountSum"] / n) if n > 0.0 else "",
                "TotalCausalMergedGapSeconds": _fmt(rec["CausalMergedGapSeconds"]),
                "MeanStreamingAbsErrorMs": _fmt(rec["MeanStreamingAbsErrorMsSum"] / err_n) if err_n > 0.0 else "",
            })

    text_rows = [r for r in rows if r.get("TextIslandIndex", "") != ""]
    metrics = [
        "CurrentOverlapRatio",
        "CausalOverlapRatio",
        "CausalMinusCurrentOverlapRatio",
        "CausalFragmentCount",
        "CausalMergedGapSeconds",
        "CausalSpanToExpectedRatio",
        "CurrentSpanToExpectedRatio",
        "MeanStreamingAbsErrorMs",
    ]
    stats_path = run_dir / "corpus_streaming_island_assignment_sim_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        stat_fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=stat_fields)
        w.writeheader()
        slices = {
            "all_text_islands": text_rows,
            "causal_differs": [r for r in text_rows if r.get("CausalAssignmentCategory") not in {"SameAsCurrent"}],
            "causal_merges": [r for r in text_rows if r.get("CausalAssignmentCategory") == "CausalMergesFragments"],
            "causal_improves_overlap": [r for r in text_rows if _row_float(r, "CausalMinusCurrentOverlapRatio") > 0.20],
            "no_causal_span": [r for r in text_rows if r.get("CausalAssignmentCategory") == "NoCausalDetectorSpan"],
        }
        for slice_name, slice_rows in slices.items():
            for metric in metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_rows = sorted(
        text_rows,
        key=lambda r: (
            0 if r.get("CausalAssignmentCategory") != "SameAsCurrent" else 1,
            -_row_float(r, "CausalMinusCurrentOverlapRatio"),
            -_row_float(r, "MeanStreamingAbsErrorMs"),
        ),
    )[:160]
    worst_path = run_dir / "corpus_streaming_island_assignment_sim_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        worst_fields = [
            "case_id",
            "TextIslandIndex",
            "CurrentAudioIslandIndex",
            "CausalDetectorIndices",
            "CausalAssignmentCategory",
            "EventCount",
            "FirstWord",
            "LastWord",
            "MappedAudioSpanSec",
            "ExpectedIslandSpanSec",
            "EvidenceSpanSec",
            "CurrentDetectorSpanSec",
            "CausalDetectorSpanSec",
            "CurrentOverlapRatio",
            "CausalOverlapRatio",
            "CausalMinusCurrentOverlapRatio",
            "CausalFragmentCount",
            "CausalMergedGapSeconds",
            "HandoffReason",
            "MeanStreamingAbsErrorMs",
            "FirstCenterErrorMs",
            "LastCenterErrorMs",
            "Notes",
        ]
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in worst_fields})



def _simulate_handoff_policy_assignments(case_dir: Path, policy_name: str) -> list[dict[str, str]]:
    """Diagnostic-only causal handoff simulator.

    Alpha21 improved active-island ownership. Alpha22 asks the next question:
    when should an active island release detector ownership so the next hard-punctuation
    island can launch? These policies are measurements only; runtime behavior is not changed.
    """
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    if not runtime_rows:
        return []
    text_rows = sorted(runtime_rows, key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    detectors = load_detector_regions(case_dir)

    onset_early_slack_sec = 0.250
    onset_late_slack_sec = 0.750
    max_span_ratio_before_handoff = 1.900

    policy_cfg = {
        "conservative_alpha21": {"ready_ratio": 0.550, "gap_sec": 0.450, "next_guard_sec": 0.180, "mode": "alpha21"},
        "duration_ready": {"ready_ratio": 0.850, "gap_sec": 0.140, "next_guard_sec": 0.300, "mode": "duration"},
        "event_ready": {"ready_ratio": 0.650, "gap_sec": 0.100, "next_guard_sec": 0.220, "mode": "event"},
        "gap_ready": {"ready_ratio": 0.400, "gap_sec": 0.250, "next_guard_sec": 0.350, "mode": "gap"},
        "hybrid_ready_gap_next_onset": {"ready_ratio": 0.780, "gap_sec": 0.180, "next_guard_sec": 0.250, "mode": "hybrid"},
    }
    cfg = policy_cfg.get(policy_name, policy_cfg["conservative_alpha21"])

    next_det_pos = 0
    out: list[dict[str, str]] = []
    for island_pos, row in enumerate(text_rows):
        text_idx = _row_int(row, "TextIslandIndex")
        mapped_start = _row_float(row, "MappedAudioStartSec")
        mapped_end = _row_float(row, "MappedAudioEndSec")
        mapped_span = max(0.0, mapped_end - mapped_start)
        planner_duration = _row_float(row, "PlannerDurationSec", mapped_span)
        expected_span = max(0.050, mapped_span if mapped_span > 1e-6 else planner_duration)
        next_mapped_start = None
        if island_pos + 1 < len(text_rows):
            next_mapped_start = _row_float(text_rows[island_pos + 1], "MappedAudioStartSec")

        evidence_start = _row_float(row, "EvidenceFirstStartSec")
        evidence_end = _row_float(row, "EvidenceLastEndSec")
        if evidence_end <= evidence_start:
            evidence_start = _row_float(row, "EvidenceFirstCenterSec")
            evidence_end = _row_float(row, "EvidenceLastCenterSec")
        evidence_span = max(0.0, evidence_end - evidence_start)

        search_lo = max(0.0, mapped_start - onset_early_slack_sec)
        while next_det_pos < len(detectors) and float(detectors[next_det_pos]["end"]) < search_lo:
            next_det_pos += 1

        launch_pos = None
        search_hi = mapped_start + onset_late_slack_sec
        probe = next_det_pos
        while probe < len(detectors):
            det = detectors[probe]
            det_start = float(det["start"])
            det_end = float(det["end"])
            if det_start > search_hi:
                break
            if det_end >= search_lo:
                launch_pos = probe
                break
            probe += 1

        causal_indices: list[int] = []
        causal_start = 0.0
        causal_end = 0.0
        causal_detector_seconds = 0.0
        causal_gap_seconds = 0.0
        release_reason = "NoConfirmedOnset"
        release_candidate_start = ""
        release_candidate_index = ""
        if launch_pos is not None:
            pos = launch_pos
            causal_start = float(detectors[pos]["start"])
            causal_end = float(detectors[pos]["end"])
            causal_indices.append(int(detectors[pos]["index"]))
            causal_detector_seconds += max(0.0, causal_end - causal_start)
            pos += 1
            release_reason = "EndOfDetectorList"
            while pos < len(detectors):
                det = detectors[pos]
                det_start = float(det["start"])
                det_end = float(det["end"])
                gap = max(0.0, det_start - causal_end)
                current_span = max(0.0, causal_end - causal_start)
                ready = current_span >= expected_span * float(cfg["ready_ratio"])
                too_long = current_span >= expected_span * max_span_ratio_before_handoff
                near_next = next_mapped_start is not None and det_start >= next_mapped_start - float(cfg["next_guard_sec"])
                after_event_end = det_start >= mapped_end - 0.060
                large_gap = gap >= float(cfg["gap_sec"])

                should_release = False
                reason = ""
                mode = str(cfg["mode"])
                if too_long:
                    should_release = True
                    reason = "CurrentIslandSpanTooLong"
                elif mode == "alpha21":
                    if ready and near_next:
                        should_release = True
                        reason = "NextIslandGuard"
                    elif ready and large_gap:
                        should_release = True
                        reason = "LargeGapAfterReady"
                elif mode == "duration":
                    if ready and (large_gap or near_next):
                        should_release = True
                        reason = "DurationReadyThenGapOrNextGuard"
                elif mode == "event":
                    if ready and after_event_end and (large_gap or near_next):
                        should_release = True
                        reason = "EventEndReadyThenGapOrNextGuard"
                elif mode == "gap":
                    if ready and large_gap:
                        should_release = True
                        reason = "GapReady"
                elif mode == "hybrid":
                    if ready and large_gap and (near_next or after_event_end):
                        should_release = True
                        reason = "HybridReadyGapNextOnset"

                if should_release:
                    release_reason = reason
                    release_candidate_start = _fmt(det_start)
                    release_candidate_index = str(int(det["index"]))
                    break

                causal_gap_seconds += gap
                causal_indices.append(int(det["index"]))
                causal_detector_seconds += max(0.0, det_end - det_start)
                causal_end = max(causal_end, det_end)
                pos += 1
            next_det_pos = pos

        causal_span = max(0.0, causal_end - causal_start)
        causal_overlap = interval_overlap_sec(evidence_start, evidence_end, causal_start, causal_end) if causal_span > 0.0 and evidence_span > 0.0 else 0.0
        causal_overlap_ratio = causal_overlap / evidence_span if evidence_span > 1e-6 else 0.0
        out.append({
            "case_id": case_dir.name,
            "Policy": policy_name,
            "TextIslandIndex": str(text_idx),
            "EventCount": row.get("EventCount", ""),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "MappedAudioStartSec": _fmt(mapped_start),
            "MappedAudioEndSec": _fmt(mapped_end),
            "ExpectedIslandSpanSec": _fmt(expected_span),
            "EvidenceStartSec": _fmt(evidence_start),
            "EvidenceEndSec": _fmt(evidence_end),
            "EvidenceSpanSec": _fmt(evidence_span),
            "AssignedDetectorIndices": ";".join(str(i) for i in causal_indices),
            "AssignedDetectorStartSec": _fmt(causal_start) if causal_indices else "",
            "AssignedDetectorEndSec": _fmt(causal_end) if causal_indices else "",
            "AssignedDetectorSpanSec": _fmt(causal_span) if causal_indices else "",
            "AssignedDetectorSpeechSeconds": _fmt(causal_detector_seconds),
            "AssignedMergedGapSeconds": _fmt(causal_gap_seconds),
            "AssignedFragmentCount": str(len(causal_indices)),
            "AssignedOverlapRatio": _fmt(causal_overlap_ratio) if causal_indices else "",
            "AssignedSpanToExpectedRatio": _fmt(causal_span / expected_span) if expected_span > 1e-6 and causal_span > 0.0 else "",
            "ReleaseReason": release_reason,
            "ReleaseCandidateDetectorIndex": release_candidate_index,
            "ReleaseCandidateStartSec": release_candidate_start,
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", ""),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", ""),
        })
    return out


def derive_streaming_handoff_policy_rows(case_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    policies = [
        "conservative_alpha21",
        "duration_ready",
        "event_ready",
        "gap_ready",
        "hybrid_ready_gap_next_onset",
    ]
    assignment_rows: list[dict[str, str]] = []
    transition_rows: list[dict[str, str]] = []
    for policy in policies:
        rows = _simulate_handoff_policy_assignments(case_dir, policy)
        assignment_rows.extend(rows)
        rows_by_idx = { _row_int(r, "TextIslandIndex", -999999): r for r in rows if r.get("TextIslandIndex", "") != "" }
        indices = sorted(i for i in rows_by_idx.keys() if i != -999999)
        for cur_idx, next_idx in zip(indices, indices[1:]):
            cur = rows_by_idx[cur_idx]
            nxt = rows_by_idx[next_idx]
            cur_assigned_end = _row_float(cur, "AssignedDetectorEndSec")
            next_assigned_start = _row_float(nxt, "AssignedDetectorStartSec")
            cur_evidence_end = _row_float(cur, "EvidenceEndSec")
            next_evidence_start = _row_float(nxt, "EvidenceStartSec")
            mapped_next_start = _row_float(nxt, "MappedAudioStartSec")
            has_next = nxt.get("AssignedDetectorStartSec", "") != ""
            next_start_error_ms = (next_assigned_start - next_evidence_start) * 1000.0 if has_next and next_evidence_start > 0.0 else math.nan
            handoff_gap = next_assigned_start - cur_assigned_end if has_next and cur.get("AssignedDetectorEndSec", "") != "" else math.nan
            evidence_gap = next_evidence_start - cur_evidence_end if next_evidence_start > 0.0 and cur_evidence_end > 0.0 else math.nan
            premature_ms = max(0.0, (cur_evidence_end - next_assigned_start) * 1000.0) if has_next and cur_evidence_end > 0.0 else 0.0
            late_ms = max(0.0, (next_assigned_start - next_evidence_start) * 1000.0) if has_next and next_evidence_start > 0.0 else math.nan
            abs_err = abs(next_start_error_ms) if math.isfinite(next_start_error_ms) else math.nan
            category = "Healthy"
            notes: list[str] = []
            if not has_next:
                category = "NoNextIslandLaunch"
                notes.append("next island has no assigned detector span")
            elif premature_ms > 100.0:
                category = "PrematureNextLaunchRisk"
                notes.append("next assigned start occurs before current evidence end")
            elif math.isfinite(next_start_error_ms) and next_start_error_ms > 250.0:
                category = "LateNextLaunchRisk"
                notes.append("next assigned start lags next evidence start")
            elif math.isfinite(next_start_error_ms) and next_start_error_ms < -180.0:
                category = "EarlyNextLaunchRisk"
                notes.append("next assigned start precedes next evidence start")
            if cur.get("ReleaseReason", "") in {"EndOfDetectorList", "NoConfirmedOnset"}:
                notes.append("current release reason=" + cur.get("ReleaseReason", ""))
            transition_rows.append({
                "case_id": case_dir.name,
                "Policy": policy,
                "CurrentTextIslandIndex": str(cur_idx),
                "NextTextIslandIndex": str(next_idx),
                "CurrentLastWord": cur.get("LastWord", ""),
                "NextFirstWord": nxt.get("FirstWord", ""),
                "CurrentExpectedSpanSec": cur.get("ExpectedIslandSpanSec", ""),
                "CurrentAssignedEndSec": cur.get("AssignedDetectorEndSec", ""),
                "CurrentEvidenceEndSec": cur.get("EvidenceEndSec", ""),
                "NextMappedStartSec": _fmt(mapped_next_start),
                "NextEvidenceStartSec": _fmt(next_evidence_start),
                "NextAssignedStartSec": _fmt(next_assigned_start) if has_next else "",
                "EvidenceInterIslandGapSec": _fmt(evidence_gap) if math.isfinite(evidence_gap) else "",
                "AssignedInterIslandGapSec": _fmt(handoff_gap) if math.isfinite(handoff_gap) else "",
                "NextStartErrorMs": _fmt(next_start_error_ms) if math.isfinite(next_start_error_ms) else "",
                "AbsNextStartErrorMs": _fmt(abs_err) if math.isfinite(abs_err) else "",
                "PrematureOverlapMs": _fmt(premature_ms),
                "LateHandoffMs": _fmt(late_ms) if math.isfinite(late_ms) else "",
                "CurrentReleaseReason": cur.get("ReleaseReason", ""),
                "CurrentReleaseCandidateStartSec": cur.get("ReleaseCandidateStartSec", ""),
                "CurrentAssignedSpanToExpectedRatio": cur.get("AssignedSpanToExpectedRatio", ""),
                "NextAssignedSpanToExpectedRatio": nxt.get("AssignedSpanToExpectedRatio", ""),
                "NextAssignedFragmentCount": nxt.get("AssignedFragmentCount", ""),
                "HandoffCategory": category,
                "Notes": ";".join(notes),
            })
    return assignment_rows, transition_rows


def collect_streaming_handoff_policy_rows(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    assignment_rows: list[dict[str, str]] = []
    transition_rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return assignment_rows, transition_rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        a, t = derive_streaming_handoff_policy_rows(case_dir)
        assignment_rows.extend(a)
        transition_rows.extend(t)
    return assignment_rows, transition_rows


def write_corpus_streaming_handoff_policy_summaries(run_dir: Path) -> None:
    assignment_rows, transition_rows = collect_streaming_handoff_policy_rows(run_dir)
    if not assignment_rows and not transition_rows:
        return

    if assignment_rows:
        fields = unique_in_order([key for row in assignment_rows for key in row.keys()])
        with (run_dir / "corpus_streaming_handoff_policy_assignments.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in assignment_rows:
                w.writerow({k: row.get(k, "") for k in fields})

    if transition_rows:
        fields = unique_in_order([key for row in transition_rows for key in row.keys()])
        with (run_dir / "corpus_streaming_handoff_policy_transitions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in transition_rows:
                w.writerow({k: row.get(k, "") for k in fields})

        summary: dict[str, dict[str, float]] = {}
        for row in transition_rows:
            policy = row.get("Policy", "") or "Unknown"
            rec = summary.setdefault(policy, {
                "Count": 0.0,
                "NoNextIslandLaunch": 0.0,
                "PrematureNextLaunchRisk": 0.0,
                "EarlyNextLaunchRisk": 0.0,
                "LateNextLaunchRisk": 0.0,
                "Healthy": 0.0,
                "AbsNextStartErrorMsSum": 0.0,
                "AbsNextStartErrorCount": 0.0,
                "SignedNextStartErrorMsSum": 0.0,
                "AssignedInterIslandGapSecSum": 0.0,
                "AssignedInterIslandGapCount": 0.0,
            })
            rec["Count"] += 1.0
            cat = row.get("HandoffCategory", "") or "Unknown"
            if cat in rec:
                rec[cat] += 1.0
            abs_err = parse_float(row.get("AbsNextStartErrorMs", ""))
            signed_err = parse_float(row.get("NextStartErrorMs", ""))
            if abs_err is not None and math.isfinite(abs_err):
                rec["AbsNextStartErrorMsSum"] += abs_err
                rec["AbsNextStartErrorCount"] += 1.0
            if signed_err is not None and math.isfinite(signed_err):
                rec["SignedNextStartErrorMsSum"] += signed_err
            gap = parse_float(row.get("AssignedInterIslandGapSec", ""))
            if gap is not None and math.isfinite(gap):
                rec["AssignedInterIslandGapSecSum"] += gap
                rec["AssignedInterIslandGapCount"] += 1.0

        with (run_dir / "corpus_streaming_handoff_policy_summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "Policy",
                "TransitionCount",
                "MeanAbsNextStartErrorMs",
                "MeanSignedNextStartErrorMs",
                "MeanAssignedInterIslandGapSec",
                "NoNextIslandLaunchRate",
                "PrematureNextLaunchRiskRate",
                "EarlyNextLaunchRiskRate",
                "LateNextLaunchRiskRate",
                "HealthyRate",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for policy, rec in sorted(summary.items()):
                n = rec["Count"]
                err_n = rec["AbsNextStartErrorCount"]
                gap_n = rec["AssignedInterIslandGapCount"]
                w.writerow({
                    "Policy": policy,
                    "TransitionCount": f"{n:.0f}",
                    "MeanAbsNextStartErrorMs": _fmt(rec["AbsNextStartErrorMsSum"] / err_n) if err_n > 0.0 else "",
                    "MeanSignedNextStartErrorMs": _fmt(rec["SignedNextStartErrorMsSum"] / err_n) if err_n > 0.0 else "",
                    "MeanAssignedInterIslandGapSec": _fmt(rec["AssignedInterIslandGapSecSum"] / gap_n) if gap_n > 0.0 else "",
                    "NoNextIslandLaunchRate": _fmt(rec["NoNextIslandLaunch"] / n) if n > 0.0 else "",
                    "PrematureNextLaunchRiskRate": _fmt(rec["PrematureNextLaunchRisk"] / n) if n > 0.0 else "",
                    "EarlyNextLaunchRiskRate": _fmt(rec["EarlyNextLaunchRisk"] / n) if n > 0.0 else "",
                    "LateNextLaunchRiskRate": _fmt(rec["LateNextLaunchRisk"] / n) if n > 0.0 else "",
                    "HealthyRate": _fmt(rec["Healthy"] / n) if n > 0.0 else "",
                })

        with (run_dir / "corpus_streaming_handoff_policy_stats.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            metrics = [
                "NextStartErrorMs",
                "AbsNextStartErrorMs",
                "PrematureOverlapMs",
                "LateHandoffMs",
                "EvidenceInterIslandGapSec",
                "AssignedInterIslandGapSec",
                "CurrentAssignedSpanToExpectedRatio",
                "NextAssignedSpanToExpectedRatio",
            ]
            for policy in sorted({r.get("Policy", "") for r in transition_rows}):
                policy_rows = [r for r in transition_rows if r.get("Policy", "") == policy]
                for metric in metrics:
                    rec = {"slice": policy, "metric": metric}
                    rec.update(stats_for(numeric_values(policy_rows, metric)))
                    w.writerow(rec)

        worst_rows = sorted(
            transition_rows,
            key=lambda r: (
                0 if r.get("HandoffCategory") != "Healthy" else 1,
                -_row_float(r, "AbsNextStartErrorMs"),
                -_row_float(r, "PrematureOverlapMs"),
            ),
        )[:200]
        with (run_dir / "corpus_streaming_handoff_policy_worst.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "case_id", "Policy", "CurrentTextIslandIndex", "NextTextIslandIndex",
                "CurrentLastWord", "NextFirstWord", "NextStartErrorMs", "AbsNextStartErrorMs",
                "PrematureOverlapMs", "LateHandoffMs", "EvidenceInterIslandGapSec",
                "AssignedInterIslandGapSec", "CurrentReleaseReason", "CurrentAssignedSpanToExpectedRatio",
                "NextAssignedSpanToExpectedRatio", "NextAssignedFragmentCount", "HandoffCategory", "Notes",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in worst_rows:
                w.writerow({k: row.get(k, "") for k in fields})



def derive_streaming_restart_evidence_rows(case_dir: Path) -> list[dict[str, str]]:
    """Alpha23 diagnostic-only hard-boundary restart evidence.

    Alpha22 showed handoff policies did not materially beat Alpha21 and that next-island
    starts remain directionally late. This pass asks a narrower question: for hard-punctuation
    transitions, did detector evidence for the next island already exist earlier than Alpha21's
    assigned launch, and would using it look safe or steal speech from the current island?
    """
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    if not runtime_rows:
        return []
    text_rows = sorted(runtime_rows, key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    if len(text_rows) < 2:
        return []
    detectors = load_detector_regions(case_dir)
    assignments = _simulate_handoff_policy_assignments(case_dir, "conservative_alpha21")
    by_idx = {_row_int(r, "TextIslandIndex", 999999): r for r in assignments}

    rows: list[dict[str, str]] = []
    for pos in range(len(text_rows) - 1):
        cur = text_rows[pos]
        nxt = text_rows[pos + 1]
        cur_idx = _row_int(cur, "TextIslandIndex")
        next_idx = _row_int(nxt, "TextIslandIndex")
        cur_assign = by_idx.get(cur_idx, {})
        next_assign = by_idx.get(next_idx, {})

        cur_evidence_start = _row_float(cur, "EvidenceFirstStartSec")
        cur_evidence_end = _row_float(cur, "EvidenceLastEndSec")
        if cur_evidence_end <= cur_evidence_start:
            cur_evidence_start = _row_float(cur, "EvidenceFirstCenterSec")
            cur_evidence_end = _row_float(cur, "EvidenceLastCenterSec")
        next_evidence_start = _row_float(nxt, "EvidenceFirstStartSec")
        next_evidence_end = _row_float(nxt, "EvidenceLastEndSec")
        if next_evidence_end <= next_evidence_start:
            next_evidence_start = _row_float(nxt, "EvidenceFirstCenterSec")
            next_evidence_end = _row_float(nxt, "EvidenceLastCenterSec")

        cur_mapped_end = _row_float(cur, "MappedAudioEndSec")
        next_mapped_start = _row_float(nxt, "MappedAudioStartSec")
        cur_assigned_end = _row_float(cur_assign, "AssignedDetectorEndSec") if cur_assign else 0.0
        next_assigned_start = _row_float(next_assign, "AssignedDetectorStartSec") if next_assign else 0.0
        has_next_launch = bool(next_assign.get("AssignedDetectorStartSec", ""))

        evidence_gap = next_evidence_start - cur_evidence_end if next_evidence_start > 0.0 and cur_evidence_end > 0.0 else math.nan
        alpha21_error_ms = (next_assigned_start - next_evidence_start) * 1000.0 if has_next_launch and next_evidence_start > 0.0 else math.nan

        # Search for the earliest detector fragment that plausibly carries the restart.
        # We intentionally allow candidates shortly before the next evidence center/span, because
        # the question is whether Alpha21 merged a true next-island onset into the previous island.
        search_lo = max(0.0, min(
            x for x in [cur_evidence_end - 0.080, cur_assigned_end - 0.080, next_mapped_start - 0.600]
            if math.isfinite(x)
        ))
        search_hi = max(next_evidence_start + 0.800, next_mapped_start + 0.800, search_lo + 0.100)
        candidates = []
        for det in detectors:
            det_idx = int(det["index"])
            det_start = float(det["start"])
            det_end = float(det["end"])
            if det_end < search_lo:
                continue
            if det_start > search_hi:
                break
            if det_end <= cur_evidence_start:
                continue
            # Prefer starts, but keep overlap cases because a detector region may straddle the restart.
            overlap_next = interval_overlap_sec(det_start, det_end, next_evidence_start, next_evidence_end)
            distance_to_next = abs(det_start - next_evidence_start) if next_evidence_start > 0.0 else 999.0
            plausible = overlap_next > 0.0 or distance_to_next <= 0.650 or (det_start >= next_mapped_start - 0.350 and det_start <= next_mapped_start + 0.650)
            if plausible:
                candidates.append((det_idx, det_start, det_end, overlap_next, distance_to_next))
        candidate = candidates[0] if candidates else None

        cand_idx = ""
        cand_start = math.nan
        cand_end = math.nan
        cand_error_ms = math.nan
        cand_gap_after_current_ms = math.nan
        cand_vs_mapped_ms = math.nan
        improvement_ms = math.nan
        steal_risk_ms = 0.0
        category = "NoRestartCandidate"
        notes: list[str] = []
        if candidate is not None:
            cand_idx, cand_start, cand_end, _ov, _dist = candidate
            cand_error_ms = (cand_start - next_evidence_start) * 1000.0 if next_evidence_start > 0.0 else math.nan
            cand_gap_after_current_ms = (cand_start - cur_evidence_end) * 1000.0 if cur_evidence_end > 0.0 else math.nan
            cand_vs_mapped_ms = (cand_start - next_mapped_start) * 1000.0 if next_mapped_start > 0.0 else math.nan
            if has_next_launch and math.isfinite(alpha21_error_ms) and math.isfinite(cand_error_ms):
                improvement_ms = abs(alpha21_error_ms) - abs(cand_error_ms)
            if math.isfinite(cand_gap_after_current_ms) and cand_gap_after_current_ms < -40.0:
                steal_risk_ms = -cand_gap_after_current_ms

            if steal_risk_ms > 120.0:
                category = "CandidateLikelyStealsCurrentSpeech"
                notes.append("candidate starts well before current evidence end")
            elif math.isfinite(cand_error_ms) and cand_error_ms < -220.0:
                category = "CandidateTooEarlyVsNextEvidence"
                notes.append("candidate precedes next evidence start")
            elif not has_next_launch:
                category = "CandidateCouldRecoverNoLaunch"
                notes.append("alpha21 had no next-island launch")
            elif math.isfinite(alpha21_error_ms) and alpha21_error_ms > 250.0 and math.isfinite(improvement_ms) and improvement_ms > 120.0:
                category = "EarlyRestartOpportunity"
                notes.append("candidate materially improves late alpha21 next launch")
            elif math.isfinite(alpha21_error_ms) and abs(alpha21_error_ms) <= 180.0:
                category = "Alpha21AlreadyGood"
            elif math.isfinite(improvement_ms) and improvement_ms > 60.0:
                category = "MinorEarlyRestartOpportunity"
            else:
                category = "CandidateNotBetter"
        elif not has_next_launch:
            category = "NoLaunchNoCandidate"
        elif math.isfinite(alpha21_error_ms) and abs(alpha21_error_ms) <= 180.0:
            category = "Alpha21AlreadyGood"

        rows.append({
            "case_id": case_dir.name,
            "CurrentTextIslandIndex": str(cur_idx),
            "NextTextIslandIndex": str(next_idx),
            "CurrentLastWord": cur.get("LastWord", ""),
            "NextFirstWord": nxt.get("FirstWord", ""),
            "CurrentEvidenceEndSec": _fmt(cur_evidence_end) if cur_evidence_end > 0.0 else "",
            "NextEvidenceStartSec": _fmt(next_evidence_start) if next_evidence_start > 0.0 else "",
            "EvidenceInterIslandGapSec": _fmt(evidence_gap) if math.isfinite(evidence_gap) else "",
            "CurrentMappedEndSec": _fmt(cur_mapped_end),
            "NextMappedStartSec": _fmt(next_mapped_start),
            "CurrentAssignedEndSec": cur_assign.get("AssignedDetectorEndSec", ""),
            "NextAlpha21AssignedStartSec": next_assign.get("AssignedDetectorStartSec", ""),
            "Alpha21NextStartErrorMs": _fmt(alpha21_error_ms) if math.isfinite(alpha21_error_ms) else "",
            "RestartCandidateDetectorIndex": str(cand_idx) if cand_idx != "" else "",
            "RestartCandidateStartSec": _fmt(cand_start) if math.isfinite(cand_start) else "",
            "RestartCandidateEndSec": _fmt(cand_end) if math.isfinite(cand_end) else "",
            "CandidateStartErrorMs": _fmt(cand_error_ms) if math.isfinite(cand_error_ms) else "",
            "CandidateGapAfterCurrentEvidenceMs": _fmt(cand_gap_after_current_ms) if math.isfinite(cand_gap_after_current_ms) else "",
            "CandidateStartVsNextMappedMs": _fmt(cand_vs_mapped_ms) if math.isfinite(cand_vs_mapped_ms) else "",
            "CandidateAbsImprovementVsAlpha21Ms": _fmt(improvement_ms) if math.isfinite(improvement_ms) else "",
            "StealRiskMs": _fmt(steal_risk_ms),
            "CandidateCountInWindow": str(len(candidates)),
            "CurrentReleaseReason": cur_assign.get("ReleaseReason", ""),
            "NextAssignedFragmentCount": next_assign.get("AssignedFragmentCount", ""),
            "RestartEvidenceCategory": category,
            "Notes": ";".join(notes),
        })
    return rows


def collect_streaming_restart_evidence_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        rows.extend(derive_streaming_restart_evidence_rows(case_dir))
    return rows


def write_corpus_streaming_restart_evidence_summaries(run_dir: Path) -> None:
    rows = collect_streaming_restart_evidence_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_streaming_restart_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get("RestartEvidenceCategory", "") or "Unknown"
        rec = summary.setdefault(cat, {
            "Count": 0.0,
            "Alpha21AbsErrorSum": 0.0,
            "Alpha21AbsErrorCount": 0.0,
            "CandidateAbsErrorSum": 0.0,
            "CandidateAbsErrorCount": 0.0,
            "ImprovementSum": 0.0,
            "ImprovementCount": 0.0,
            "StealRiskSum": 0.0,
            "GapSum": 0.0,
            "GapCount": 0.0,
        })
        rec["Count"] += 1.0
        a = parse_float(row.get("Alpha21NextStartErrorMs", ""))
        c = parse_float(row.get("CandidateStartErrorMs", ""))
        imp = parse_float(row.get("CandidateAbsImprovementVsAlpha21Ms", ""))
        steal = parse_float(row.get("StealRiskMs", ""))
        gap = parse_float(row.get("EvidenceInterIslandGapSec", ""))
        if a is not None and math.isfinite(a):
            rec["Alpha21AbsErrorSum"] += abs(a)
            rec["Alpha21AbsErrorCount"] += 1.0
        if c is not None and math.isfinite(c):
            rec["CandidateAbsErrorSum"] += abs(c)
            rec["CandidateAbsErrorCount"] += 1.0
        if imp is not None and math.isfinite(imp):
            rec["ImprovementSum"] += imp
            rec["ImprovementCount"] += 1.0
        if steal is not None and math.isfinite(steal):
            rec["StealRiskSum"] += steal
        if gap is not None and math.isfinite(gap):
            rec["GapSum"] += gap
            rec["GapCount"] += 1.0

    with (run_dir / "corpus_streaming_restart_evidence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "RestartEvidenceCategory", "Count", "Rate", "MeanAlpha21AbsNextStartErrorMs",
            "MeanCandidateAbsStartErrorMs", "MeanCandidateAbsImprovementMs", "MeanStealRiskMs",
            "MeanEvidenceInterIslandGapSec",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total = float(len(rows))
        for cat, rec in sorted(summary.items(), key=lambda kv: (-kv[1]["Count"], kv[0])):
            n = rec["Count"]
            w.writerow({
                "RestartEvidenceCategory": cat,
                "Count": f"{n:.0f}",
                "Rate": _fmt(n / total) if total > 0.0 else "",
                "MeanAlpha21AbsNextStartErrorMs": _fmt(rec["Alpha21AbsErrorSum"] / rec["Alpha21AbsErrorCount"]) if rec["Alpha21AbsErrorCount"] > 0.0 else "",
                "MeanCandidateAbsStartErrorMs": _fmt(rec["CandidateAbsErrorSum"] / rec["CandidateAbsErrorCount"]) if rec["CandidateAbsErrorCount"] > 0.0 else "",
                "MeanCandidateAbsImprovementMs": _fmt(rec["ImprovementSum"] / rec["ImprovementCount"]) if rec["ImprovementCount"] > 0.0 else "",
                "MeanStealRiskMs": _fmt(rec["StealRiskSum"] / n) if n > 0.0 else "",
                "MeanEvidenceInterIslandGapSec": _fmt(rec["GapSum"] / rec["GapCount"]) if rec["GapCount"] > 0.0 else "",
            })

    with (run_dir / "corpus_streaming_restart_evidence_stats.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        slices = {"all_transitions": rows}
        for cat in sorted({r.get("RestartEvidenceCategory", "") for r in rows}):
            slices[cat or "Unknown"] = [r for r in rows if (r.get("RestartEvidenceCategory", "") or "Unknown") == (cat or "Unknown")]
        metrics = [
            "Alpha21NextStartErrorMs",
            "CandidateStartErrorMs",
            "CandidateAbsImprovementVsAlpha21Ms",
            "CandidateGapAfterCurrentEvidenceMs",
            "CandidateStartVsNextMappedMs",
            "StealRiskMs",
            "EvidenceInterIslandGapSec",
            "CandidateCountInWindow",
        ]
        for slice_name, slice_rows in slices.items():
            for metric in metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_rows = sorted(
        rows,
        key=lambda r: (
            0 if r.get("RestartEvidenceCategory") in {"EarlyRestartOpportunity", "CandidateCouldRecoverNoLaunch"} else 1,
            -_row_float(r, "CandidateAbsImprovementVsAlpha21Ms"),
            -abs(_row_float(r, "Alpha21NextStartErrorMs")),
        ),
    )[:200]
    with (run_dir / "corpus_streaming_restart_evidence_worst.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case_id", "CurrentTextIslandIndex", "NextTextIslandIndex", "CurrentLastWord", "NextFirstWord",
            "EvidenceInterIslandGapSec", "NextAlpha21AssignedStartSec", "Alpha21NextStartErrorMs",
            "RestartCandidateDetectorIndex", "RestartCandidateStartSec", "CandidateStartErrorMs",
            "CandidateGapAfterCurrentEvidenceMs", "CandidateAbsImprovementVsAlpha21Ms", "StealRiskMs",
            "CandidateCountInWindow", "CurrentReleaseReason", "NextAssignedFragmentCount",
            "RestartEvidenceCategory", "Notes",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in fields})



def _detector_owner_map_from_assignment_rows(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    owners: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        text_idx = _row_int(row, "TextIslandIndex", -999999)
        if text_idx < 0:
            continue
        raw = row.get("CausalDetectorIndices", "") or row.get("AssignedDetectorIndices", "")
        for part in str(raw).replace(";", " ").replace(",", " ").split():
            try:
                idx = int(part)
            except ValueError:
                continue
            owners.setdefault(idx, []).append(row)
    return owners


def derive_no_causal_detector_span_forensics_rows(case_dir: Path) -> list[dict[str, str]]:
    """Alpha25 forensic pass for Alpha21/24's biggest remaining bucket.

    Alpha24 showed that targeted safe restart is too low-leverage and risky.  This
    pass rolls that behavior back and instead explains every NoCausalDetectorSpan
    island from the causal streaming simulator: is speech evidence absent, present
    but absorbed by a neighbor, present but outside the causal launch window, or
    simply a tiny/detail island with no useful detector evidence?
    """
    runtime_rows = read_csv_rows(case_dir / "runtime_island_alignment.csv")
    if not runtime_rows:
        return []
    text_rows = sorted(runtime_rows, key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    text_by_idx = {_row_int(r, "TextIslandIndex", 999999): r for r in text_rows}
    causal_rows = derive_streaming_island_assignment_sim_rows(case_dir)
    no_span_rows = [r for r in causal_rows if r.get("CausalAssignmentCategory") == "NoCausalDetectorSpan"]
    if not no_span_rows:
        return []

    detectors = load_detector_regions(case_dir)
    owner_map = _detector_owner_map_from_assignment_rows(causal_rows)

    out: list[dict[str, str]] = []
    for sim in no_span_rows:
        text_idx = _row_int(sim, "TextIslandIndex", -1)
        runtime = text_by_idx.get(text_idx, {})
        mapped_start = _row_float(sim, "MappedAudioStartSec", _row_float(runtime, "MappedAudioStartSec"))
        mapped_end = _row_float(sim, "MappedAudioEndSec", _row_float(runtime, "MappedAudioEndSec"))
        expected_span = _row_float(sim, "ExpectedSpanSec", max(0.050, mapped_end - mapped_start))
        if expected_span <= 0.0:
            expected_span = max(0.050, mapped_end - mapped_start)

        evidence_start = _row_float(sim, "EvidenceStartSec", _row_float(runtime, "EvidenceFirstStartSec"))
        evidence_end = _row_float(sim, "EvidenceEndSec", _row_float(runtime, "EvidenceLastEndSec"))
        if evidence_end <= evidence_start:
            evidence_start = _row_float(runtime, "EvidenceFirstCenterSec")
            evidence_end = _row_float(runtime, "EvidenceLastCenterSec")
        evidence_span = max(0.0, evidence_end - evidence_start)
        has_evidence = evidence_span > 1e-6 and evidence_start > 0.0 and evidence_end > evidence_start

        # Reconstruct the causal launch window used by Alpha19/21 diagnostics.
        search_lo = max(0.0, mapped_start - 0.250)
        search_hi = mapped_start + 0.750
        expected_lo = evidence_start if has_evidence else mapped_start
        expected_hi = evidence_end if has_evidence else mapped_end
        if expected_hi <= expected_lo:
            expected_hi = expected_lo + max(0.050, expected_span)

        overlap_candidates = []
        nearby_candidates = []
        launch_window_candidates = []
        nearest_before = None
        nearest_after = None
        for det in detectors:
            det_idx = int(det["index"])
            det_start = float(det["start"])
            det_end = float(det["end"])
            if det_end <= expected_lo:
                gap = expected_lo - det_end
                if nearest_before is None or gap < nearest_before[0]:
                    nearest_before = (gap, det_idx, det_start, det_end)
            if det_start >= expected_hi:
                gap = det_start - expected_hi
                if nearest_after is None or gap < nearest_after[0]:
                    nearest_after = (gap, det_idx, det_start, det_end)
            ov = interval_overlap_sec(det_start, det_end, expected_lo, expected_hi)
            if ov > 0.0:
                overlap_candidates.append((det_idx, det_start, det_end, ov))
            # A broader evidence/mapped search window, useful for missed launches.
            distance = 0.0
            if det_end < expected_lo:
                distance = expected_lo - det_end
            elif det_start > expected_hi:
                distance = det_start - expected_hi
            if ov > 0.0 or distance <= 0.650:
                nearby_candidates.append((det_idx, det_start, det_end, ov, distance))
            if det_start <= search_hi and det_end >= search_lo:
                launch_window_candidates.append((det_idx, det_start, det_end))

        best_overlap = max(overlap_candidates, key=lambda x: x[3], default=None)
        best_nearby = min(nearby_candidates, key=lambda x: (0 if x[3] > 0.0 else 1, x[4], abs(x[1] - expected_lo)), default=None)
        best_idx = best_overlap[0] if best_overlap is not None else (best_nearby[0] if best_nearby is not None else -1)
        best_start = best_overlap[1] if best_overlap is not None else (best_nearby[1] if best_nearby is not None else math.nan)
        best_end = best_overlap[2] if best_overlap is not None else (best_nearby[2] if best_nearby is not None else math.nan)
        best_overlap_sec = best_overlap[3] if best_overlap is not None else 0.0
        best_owner_rows = owner_map.get(best_idx, []) if best_idx >= 0 else []
        owner_text_indices = sorted({_row_int(r, "TextIslandIndex", -1) for r in best_owner_rows if _row_int(r, "TextIslandIndex", -1) >= 0})
        owner_desc = ";".join(str(i) for i in owner_text_indices)
        assigned_to_prev = any(i < text_idx for i in owner_text_indices)
        assigned_to_next = any(i > text_idx for i in owner_text_indices)

        event_count = _row_int(sim, "EventCount", _row_int(runtime, "EventCount", 0))
        mean_mae = _row_float(sim, "MeanStreamingAbsErrorMs", _row_float(runtime, "MeanStreamingAbsErrorMs"))
        first_err = _row_float(sim, "FirstCenterErrorMs", _row_float(runtime, "FirstCenterErrorMs"))
        last_err = _row_float(sim, "LastCenterErrorMs", _row_float(runtime, "LastCenterErrorMs"))

        category = "UnknownNoSpan"
        notes: list[str] = []
        if not has_evidence:
            category = "NoOracleEvidenceSpan"
            notes.append("oracle/evidence span unavailable for this text island")
        elif best_overlap is not None and assigned_to_prev:
            category = "EvidenceAbsorbedByPreviousIsland"
            notes.append("detector region overlapping evidence was owned by previous island")
        elif best_overlap is not None and assigned_to_next:
            category = "EvidenceAbsorbedByNextIsland"
            notes.append("detector region overlapping evidence was owned by later island")
        elif best_overlap is not None and not owner_text_indices:
            category = "EvidenceDetectorUnconsumed"
            notes.append("detector evidence exists but causal simulator left it unconsumed")
        elif launch_window_candidates:
            category = "DetectorInLaunchWindowButRejected"
            notes.append("detector crossed launch window but did not become causal span")
        elif best_nearby is not None:
            det_start = best_nearby[1]
            det_end = best_nearby[2]
            if det_end < search_lo:
                category = "DetectorBeforeLaunchWindow"
                notes.append("nearest detector evidence ended before causal launch window")
            elif det_start > search_hi:
                category = "DetectorAfterLaunchWindow"
                notes.append("nearest detector evidence starts after causal launch window")
            else:
                category = "NearbyDetectorNoOverlap"
                notes.append("nearby detector evidence does not overlap expected/evidence span")
        elif event_count <= 2 or expected_span <= 0.180:
            category = "TinyIslandNoDetectorEvidence"
            notes.append("tiny/detail island with no nearby detector evidence")
        else:
            category = "NoNearbyDetectorEvidence"
            notes.append("no detector region within 650ms of expected/evidence span")

        if mean_mae >= 300.0:
            notes.append("high_mae")
        if best_overlap_sec > 0.0 and evidence_span > 1e-6:
            notes.append("best_overlap_ratio=" + _fmt(best_overlap_sec / evidence_span))
        if owner_desc:
            notes.append("candidate_owner_text_island=" + owner_desc)

        out.append({
            "case_id": case_dir.name,
            "TextIslandIndex": str(text_idx),
            "EventCount": str(event_count),
            "FirstWord": sim.get("FirstWord", runtime.get("FirstWord", "")),
            "LastWord": sim.get("LastWord", runtime.get("LastWord", "")),
            "MappedAudioStartSec": _fmt(mapped_start),
            "MappedAudioEndSec": _fmt(mapped_end),
            "ExpectedSpanSec": _fmt(expected_span),
            "EvidenceStartSec": _fmt(evidence_start) if has_evidence else "",
            "EvidenceEndSec": _fmt(evidence_end) if has_evidence else "",
            "EvidenceSpanSec": _fmt(evidence_span) if has_evidence else "",
            "LaunchWindowStartSec": _fmt(search_lo),
            "LaunchWindowEndSec": _fmt(search_hi),
            "OverlappingDetectorCount": str(len(overlap_candidates)),
            "NearbyDetectorCount": str(len(nearby_candidates)),
            "LaunchWindowDetectorCount": str(len(launch_window_candidates)),
            "BestDetectorIndex": str(best_idx) if best_idx >= 0 else "",
            "BestDetectorStartSec": _fmt(best_start) if math.isfinite(best_start) else "",
            "BestDetectorEndSec": _fmt(best_end) if math.isfinite(best_end) else "",
            "BestDetectorOverlapSec": _fmt(best_overlap_sec) if best_overlap_sec > 0.0 else "",
            "BestDetectorOverlapRatio": _fmt(best_overlap_sec / evidence_span) if has_evidence and evidence_span > 1e-6 and best_overlap_sec > 0.0 else "",
            "BestDetectorOwnerTextIslandIndices": owner_desc,
            "NearestDetectorBeforeGapSec": _fmt(nearest_before[0]) if nearest_before else "",
            "NearestDetectorBeforeIndex": str(nearest_before[1]) if nearest_before else "",
            "NearestDetectorAfterGapSec": _fmt(nearest_after[0]) if nearest_after else "",
            "NearestDetectorAfterIndex": str(nearest_after[1]) if nearest_after else "",
            "MeanStreamingAbsErrorMs": _fmt(mean_mae),
            "FirstCenterErrorMs": _fmt(first_err),
            "LastCenterErrorMs": _fmt(last_err),
            "NoSpanForensicCategory": category,
            "Notes": ";".join(notes),
        })
    return out



def _detector_by_index(detectors: list[dict[str, float | int]]) -> dict[int, dict[str, float | int]]:
    return {int(d["index"]): d for d in detectors}


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").replace(";", " ").replace(",", " ").split():
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _derive_ownership_retention_reason(
    missing_row: dict[str, str],
    owner_row: dict[str, str] | None,
    detectors_by_index: dict[int, dict[str, float | int]],
) -> dict[str, str]:
    """Explain why Alpha21 ownership kept evidence with a previous island.

    This is intentionally diagnostic-only.  It mirrors the causal assignment
    loop's three release gates: current-island readiness, near-next-island guard,
    and large-gap break.  For the detector fragment that should have served the
    missing island, it reconstructs what the previous island knew immediately
    before it consumed that fragment.
    """
    best_idx = _row_int(missing_row, "BestDetectorIndex", -1)
    if owner_row is None or best_idx < 0 or best_idx not in detectors_by_index:
        return {
            "OwnershipRetentionReason": "NoPreviousOwnerRow",
            "OwnershipRetentionNotes": "could not reconstruct previous owner for candidate detector",
        }

    owner_indices = _parse_int_list(owner_row.get("CausalDetectorIndices", "") or owner_row.get("AssignedDetectorIndices", ""))
    if best_idx not in owner_indices:
        return {
            "OwnershipRetentionReason": "CandidateNotInOwnerDetectorList",
            "OwnershipRetentionNotes": "best detector owner metadata did not include candidate detector index",
            "PreviousOwnerTextIslandIndex": owner_row.get("TextIslandIndex", ""),
            "PreviousOwnerHandoffReason": owner_row.get("HandoffReason", ""),
        }

    pos = owner_indices.index(best_idx)
    owner_start = _row_float(owner_row, "CausalDetectorStartSec")
    owner_expected = _row_float(owner_row, "ExpectedIslandSpanSec", _row_float(owner_row, "ExpectedSpanSec", 0.0))
    owner_frag_count = _row_int(owner_row, "CausalFragmentCount", len(owner_indices))
    owner_handoff = owner_row.get("HandoffReason", "")
    target = detectors_by_index[best_idx]
    target_start = float(target["start"])
    target_end = float(target["end"])

    if pos == 0:
        prior_end = owner_start
        span_before = 0.0
        gap_before = 0.0
    else:
        prev_idx = owner_indices[pos - 1]
        prev = detectors_by_index.get(prev_idx)
        prior_end = float(prev["end"]) if prev is not None else _row_float(owner_row, "CausalDetectorStartSec")
        span_before = max(0.0, prior_end - owner_start)
        gap_before = max(0.0, target_start - prior_end)

    mapped_start = _row_float(missing_row, "MappedAudioStartSec")
    near_next_guard_delta = target_start - (mapped_start - 0.180)
    ready_ratio = span_before / owner_expected if owner_expected > 1e-6 else 0.0
    total_owner_span_after = max(0.0, target_end - owner_start)
    total_ratio_after = total_owner_span_after / owner_expected if owner_expected > 1e-6 else 0.0

    reason = "UnknownRetentionGate"
    notes: list[str] = []
    if pos == 0:
        reason = "PreviousIslandLaunchedOnNextIslandEvidence"
        notes.append("candidate was first detector fragment owned by previous island")
    elif ready_ratio < 0.550:
        reason = "PreviousIslandBudgetNotReady"
        notes.append("previous island had not reached 55% of expected span before candidate")
    elif target_start < mapped_start - 0.180:
        reason = "NextIslandGuardNotReached"
        notes.append("candidate began before next-island guard window")
    elif gap_before <= 0.450:
        reason = "FragmentMergeGapTooSmall"
        notes.append("candidate gap did not exceed 450ms merge threshold")
    elif total_ratio_after < 1.850 and owner_handoff == "EndOfDetectorList":
        reason = "NoReleaseBeforeDetectorListEnd"
        notes.append("previous island consumed to end of detector list")
    else:
        notes.append("candidate was consumed even though reconstructed release gates looked plausible")

    if total_ratio_after >= 1.850:
        notes.append("owner_span_after_exceeds_too_long_threshold")
    if owner_frag_count > 1:
        notes.append(f"owner_fragment_count={owner_frag_count}")
    if owner_handoff:
        notes.append(f"owner_handoff={owner_handoff}")

    return {
        "PreviousOwnerTextIslandIndex": owner_row.get("TextIslandIndex", ""),
        "PreviousOwnerFirstWord": owner_row.get("FirstWord", ""),
        "PreviousOwnerLastWord": owner_row.get("LastWord", ""),
        "PreviousOwnerDetectorIndices": owner_row.get("CausalDetectorIndices", ""),
        "PreviousOwnerHandoffReason": owner_handoff,
        "PreviousOwnerExpectedSpanSec": _fmt(owner_expected),
        "PreviousOwnerSpanBeforeCandidateSec": _fmt(span_before),
        "PreviousOwnerReadyRatioBeforeCandidate": _fmt(ready_ratio),
        "GapBeforeCandidateDetectorSec": _fmt(gap_before),
        "CandidateStartMinusNextGuardSec": _fmt(near_next_guard_delta),
        "PreviousOwnerSpanAfterCandidateSec": _fmt(total_owner_span_after),
        "PreviousOwnerSpanAfterCandidateRatio": _fmt(total_ratio_after),
        "OwnershipRetentionReason": reason,
        "OwnershipRetentionNotes": ";".join(notes),
    }


def derive_ownership_retention_forensics_rows(case_dir: Path) -> list[dict[str, str]]:
    causal_rows = derive_streaming_island_assignment_sim_rows(case_dir)
    if not causal_rows:
        return []
    causal_by_idx = {_row_int(r, "TextIslandIndex", -999999): r for r in causal_rows}
    detectors = load_detector_regions(case_dir)
    detectors_by_idx = _detector_by_index(detectors)
    no_span_rows = derive_no_causal_detector_span_forensics_rows(case_dir)
    absorbed = [r for r in no_span_rows if r.get("NoSpanForensicCategory") == "EvidenceAbsorbedByPreviousIsland"]
    out: list[dict[str, str]] = []
    for row in absorbed:
        owners = _parse_int_list(row.get("BestDetectorOwnerTextIslandIndices", ""))
        prev_owners = [i for i in owners if i < _row_int(row, "TextIslandIndex", -1)]
        owner_idx = max(prev_owners) if prev_owners else -1
        owner_row = causal_by_idx.get(owner_idx)
        extra = _derive_ownership_retention_reason(row, owner_row, detectors_by_idx)
        combined = dict(row)
        combined.update(extra)
        out.append(combined)
    return out



def _text_island_reference_span(row: dict[str, str]) -> tuple[float, float, str]:
    """Return the best available reference span for launch attribution.

    Evidence spans are preferred because they are derived from oracle/runtime
    evaluation. Mapped spans are a fallback so we can still reason about islands
    without usable evidence rows.
    """
    es = _row_float(row, "EvidenceFirstStartSec")
    ee = _row_float(row, "EvidenceLastEndSec")
    if ee > es:
        return es, ee, "evidence"
    cs = _row_float(row, "EvidenceFirstCenterSec")
    ce = _row_float(row, "EvidenceLastCenterSec")
    if ce > cs:
        return cs, ce, "evidence_center"
    ms = _row_float(row, "MappedAudioStartSec")
    me = _row_float(row, "MappedAudioEndSec")
    if me > ms:
        return ms, me, "mapped"
    return 0.0, 0.0, "none"


def _best_text_island_for_detector(
    detector: dict[str, float | int],
    text_rows: list[dict[str, str]],
) -> dict[str, str]:
    det_start = float(detector["start"])
    det_end = float(detector["end"])
    det_span = max(0.0, det_end - det_start)
    candidates: list[dict[str, str | float | int]] = []
    for row in text_rows:
        text_idx = _row_int(row, "TextIslandIndex", -1)
        ref_start, ref_end, ref_source = _text_island_reference_span(row)
        ref_span = max(0.0, ref_end - ref_start)
        overlap = interval_overlap_sec(det_start, det_end, ref_start, ref_end) if ref_span > 0.0 and det_span > 0.0 else 0.0
        overlap_ratio_to_text = overlap / ref_span if ref_span > 1e-6 else 0.0
        overlap_ratio_to_detector = overlap / det_span if det_span > 1e-6 else 0.0
        distance = 0.0
        if det_end < ref_start:
            distance = ref_start - det_end
        elif det_start > ref_end:
            distance = det_start - ref_end
        score = overlap * 4.0 + overlap_ratio_to_text * 1.5 + overlap_ratio_to_detector * 0.75 - distance * 0.35
        candidates.append({
            "TextIslandIndex": text_idx,
            "RefStartSec": ref_start,
            "RefEndSec": ref_end,
            "RefSource": ref_source,
            "OverlapSec": overlap,
            "OverlapRatioToText": overlap_ratio_to_text,
            "OverlapRatioToDetector": overlap_ratio_to_detector,
            "DistanceSec": distance,
            "Score": score,
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", ""),
        })
    candidates.sort(key=lambda c: (-float(c["Score"]), -float(c["OverlapRatioToText"]), float(c["DistanceSec"])))
    best = candidates[0] if candidates else {}
    second = candidates[1] if len(candidates) > 1 else {}
    return {
        "BestMatchingTextIslandIndex": str(best.get("TextIslandIndex", "")),
        "BestMatchingFirstWord": str(best.get("FirstWord", "")),
        "BestMatchingLastWord": str(best.get("LastWord", "")),
        "BestMatchingRefSource": str(best.get("RefSource", "")),
        "BestMatchingRefStartSec": _fmt(float(best.get("RefStartSec", 0.0))) if best else "",
        "BestMatchingRefEndSec": _fmt(float(best.get("RefEndSec", 0.0))) if best else "",
        "BestMatchingOverlapSec": _fmt(float(best.get("OverlapSec", 0.0))) if best else "",
        "BestMatchingOverlapRatioToText": _fmt(float(best.get("OverlapRatioToText", 0.0))) if best else "",
        "BestMatchingOverlapRatioToDetector": _fmt(float(best.get("OverlapRatioToDetector", 0.0))) if best else "",
        "BestMatchingDistanceSec": _fmt(float(best.get("DistanceSec", 0.0))) if best else "",
        "SecondBestTextIslandIndex": str(second.get("TextIslandIndex", "")) if second else "",
        "SecondBestOverlapRatioToText": _fmt(float(second.get("OverlapRatioToText", 0.0))) if second else "",
        "SecondBestDistanceSec": _fmt(float(second.get("DistanceSec", 0.0))) if second else "",
        "BestScoreMinusSecondScore": _fmt(float(best.get("Score", 0.0)) - float(second.get("Score", 0.0))) if best and second else "",
        "BestMatchingMeanStreamingAbsErrorMs": str(best.get("MeanStreamingAbsErrorMs", "")) if best else "",
    }


def derive_launch_attribution_forensics_rows(case_dir: Path) -> list[dict[str, str]]:
    """Diagnose whether a detector fragment launched the correct text island.

    Alpha26 showed a major absorbed-evidence class where the previous island appears
    to launch on evidence that best belongs to the next island. This pass looks at
    every causal island launch and asks: if the launch detector fragment were matched
    against all text islands by evidence overlap, which text island would it choose?
    """
    causal_rows = derive_streaming_island_assignment_sim_rows(case_dir)
    runtime_rows = sorted(read_csv_rows(case_dir / "runtime_island_alignment.csv"), key=lambda r: _row_int(r, "TextIslandIndex", 999999))
    detectors_by_idx = _detector_by_index(load_detector_regions(case_dir))
    if not causal_rows or not runtime_rows:
        return []
    runtime_by_idx = {_row_int(r, "TextIslandIndex", -999999): r for r in runtime_rows}

    out: list[dict[str, str]] = []
    for row in causal_rows:
        text_idx = _row_int(row, "TextIslandIndex", -1)
        if text_idx < 0:
            continue
        first_det_idx = _row_int(row, "CausalFirstAudioIslandIndex", -1)
        if first_det_idx < 0:
            continue
        detector = detectors_by_idx.get(first_det_idx)
        if detector is None:
            continue
        runtime = runtime_by_idx.get(text_idx, {})
        match = _best_text_island_for_detector(detector, runtime_rows)
        best_idx = parse_int(match.get("BestMatchingTextIslandIndex", ""))
        best_overlap = parse_float(match.get("BestMatchingOverlapRatioToText", "")) or 0.0
        best_det_overlap = parse_float(match.get("BestMatchingOverlapRatioToDetector", "")) or 0.0
        second_overlap = parse_float(match.get("SecondBestOverlapRatioToText", "")) or 0.0
        best_distance = parse_float(match.get("BestMatchingDistanceSec", "")) or 0.0
        score_margin = parse_float(match.get("BestScoreMinusSecondScore", "")) or 0.0
        ambiguous = (best_overlap < 0.250 and best_det_overlap < 0.250 and best_distance > 0.250) or (abs(best_overlap - second_overlap) < 0.100 and score_margin < 0.200)
        category = "AmbiguousLaunchAttribution"
        notes: list[str] = []
        if best_idx is None or best_idx < 0:
            category = "NoBestMatchingTextIsland"
        elif ambiguous:
            category = "AmbiguousLaunchAttribution"
            notes.append("weak_or_tied_best_match")
        elif best_idx == text_idx:
            category = "CorrectLaunchAttribution"
        elif best_idx == text_idx + 1:
            category = "LaunchedPreviousOnNextIslandEvidence"
            notes.append("launch fragment best matches next text island")
        elif best_idx > text_idx:
            category = "LaunchedEarlierIslandOnLaterEvidence"
            notes.append("launch fragment best matches later text island")
        elif best_idx == text_idx - 1:
            category = "LaunchedNextOnPreviousIslandEvidence"
            notes.append("launch fragment best matches previous text island")
        else:
            category = "LaunchedLaterIslandOnEarlierEvidence"
            notes.append("launch fragment best matches earlier text island")

        # Connect this launch attribution back to the Alpha25 absorbed evidence class.
        next_row = runtime_by_idx.get(text_idx + 1, {})
        next_ref_start, next_ref_end, next_ref_source = _text_island_reference_span(next_row) if next_row else (0.0, 0.0, "")
        det_start = float(detector["start"])
        det_end = float(detector["end"])
        next_overlap = interval_overlap_sec(det_start, det_end, next_ref_start, next_ref_end) if next_ref_end > next_ref_start else 0.0
        next_overlap_ratio = next_overlap / max(1e-6, next_ref_end - next_ref_start) if next_ref_end > next_ref_start else 0.0
        if next_overlap_ratio >= 0.500 and best_idx == text_idx + 1:
            notes.append("strong_next_island_overlap")
        if row.get("HandoffReason"):
            notes.append("handoff=" + row.get("HandoffReason", ""))

        out.append({
            "case_id": case_dir.name,
            "LaunchTextIslandIndex": str(text_idx),
            "LaunchFirstWord": row.get("FirstWord", runtime.get("FirstWord", "")),
            "LaunchLastWord": row.get("LastWord", runtime.get("LastWord", "")),
            "LaunchDetectorIndex": str(first_det_idx),
            "LaunchDetectorStartSec": _fmt(det_start),
            "LaunchDetectorEndSec": _fmt(det_end),
            "LaunchDetectorSpanSec": _fmt(max(0.0, det_end - det_start)),
            "CausalDetectorIndices": row.get("CausalDetectorIndices", ""),
            "CausalFragmentCount": row.get("CausalFragmentCount", ""),
            "CausalAssignmentCategory": row.get("CausalAssignmentCategory", ""),
            "HandoffReason": row.get("HandoffReason", ""),
            "ExpectedIslandSpanSec": row.get("ExpectedIslandSpanSec", ""),
            "CausalDetectorSpanSec": row.get("CausalDetectorSpanSec", ""),
            "CausalSpanToExpectedRatio": row.get("CausalSpanToExpectedRatio", ""),
            **match,
            "NextIslandRefSource": next_ref_source,
            "NextIslandRefStartSec": _fmt(next_ref_start) if next_ref_end > next_ref_start else "",
            "NextIslandRefEndSec": _fmt(next_ref_end) if next_ref_end > next_ref_start else "",
            "LaunchDetectorOverlapWithNextIslandRatio": _fmt(next_overlap_ratio) if next_ref_end > next_ref_start else "",
            "MeanStreamingAbsErrorMs": row.get("MeanStreamingAbsErrorMs", runtime.get("MeanStreamingAbsErrorMs", "")),
            "FirstCenterErrorMs": row.get("FirstCenterErrorMs", runtime.get("FirstCenterErrorMs", "")),
            "LastCenterErrorMs": row.get("LastCenterErrorMs", runtime.get("LastCenterErrorMs", "")),
            "LaunchAttributionCategory": category,
            "LaunchAttributionNotes": ";".join(notes),
        })
    return out


def collect_launch_attribution_forensics_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        rows.extend(derive_launch_attribution_forensics_rows(case_dir))
    return rows


def write_corpus_launch_attribution_forensics(run_dir: Path) -> None:
    rows = collect_launch_attribution_forensics_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_launch_attribution_forensics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get("LaunchAttributionCategory", "") or "Unknown"
        rec = summary.setdefault(cat, {
            "Count": 0.0,
            "MaeSum": 0.0,
            "MaeCount": 0.0,
            "BestOverlapSum": 0.0,
            "BestOverlapCount": 0.0,
            "NextOverlapSum": 0.0,
            "NextOverlapCount": 0.0,
            "DetectorSpanSum": 0.0,
            "DetectorSpanCount": 0.0,
        })
        rec["Count"] += 1.0
        for field, sum_key, count_key in [
            ("MeanStreamingAbsErrorMs", "MaeSum", "MaeCount"),
            ("BestMatchingOverlapRatioToText", "BestOverlapSum", "BestOverlapCount"),
            ("LaunchDetectorOverlapWithNextIslandRatio", "NextOverlapSum", "NextOverlapCount"),
            ("LaunchDetectorSpanSec", "DetectorSpanSum", "DetectorSpanCount"),
        ]:
            val = parse_float(row.get(field, ""))
            if val is not None and math.isfinite(val):
                rec[sum_key] += val
                rec[count_key] += 1.0

    with (run_dir / "corpus_launch_attribution_forensics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "LaunchAttributionCategory", "Count", "Rate", "MeanStreamingAbsErrorMs",
            "MeanBestMatchingOverlapRatioToText", "MeanLaunchDetectorOverlapWithNextIslandRatio",
            "MeanLaunchDetectorSpanSec",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total = float(len(rows))
        for cat, rec in sorted(summary.items(), key=lambda kv: (-kv[1]["Count"], kv[0])):
            n = rec["Count"]
            w.writerow({
                "LaunchAttributionCategory": cat,
                "Count": f"{n:.0f}",
                "Rate": _fmt(n / total) if total > 0.0 else "",
                "MeanStreamingAbsErrorMs": _fmt(rec["MaeSum"] / rec["MaeCount"]) if rec["MaeCount"] > 0.0 else "",
                "MeanBestMatchingOverlapRatioToText": _fmt(rec["BestOverlapSum"] / rec["BestOverlapCount"]) if rec["BestOverlapCount"] > 0.0 else "",
                "MeanLaunchDetectorOverlapWithNextIslandRatio": _fmt(rec["NextOverlapSum"] / rec["NextOverlapCount"]) if rec["NextOverlapCount"] > 0.0 else "",
                "MeanLaunchDetectorSpanSec": _fmt(rec["DetectorSpanSum"] / rec["DetectorSpanCount"]) if rec["DetectorSpanCount"] > 0.0 else "",
            })

    with (run_dir / "corpus_launch_attribution_forensics_stats.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        slices = {"all_launches": rows}
        for cat in sorted({r.get("LaunchAttributionCategory", "") or "Unknown" for r in rows}):
            slices[cat] = [r for r in rows if (r.get("LaunchAttributionCategory", "") or "Unknown") == cat]
        for slice_name, slice_rows in slices.items():
            for metric in [
                "MeanStreamingAbsErrorMs", "BestMatchingOverlapRatioToText", "BestMatchingOverlapRatioToDetector",
                "BestMatchingDistanceSec", "BestScoreMinusSecondScore", "LaunchDetectorOverlapWithNextIslandRatio",
                "LaunchDetectorSpanSec", "CausalSpanToExpectedRatio",
            ]:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_rows = sorted(
        rows,
        key=lambda r: (
            0 if r.get("LaunchAttributionCategory") != "CorrectLaunchAttribution" else 1,
            -_row_float(r, "MeanStreamingAbsErrorMs"),
            -_row_float(r, "BestMatchingOverlapRatioToText"),
        ),
    )[:200]
    fields = [
        "case_id", "LaunchTextIslandIndex", "LaunchFirstWord", "LaunchLastWord", "LaunchDetectorIndex",
        "LaunchDetectorStartSec", "LaunchDetectorEndSec", "BestMatchingTextIslandIndex", "BestMatchingFirstWord",
        "BestMatchingLastWord", "BestMatchingOverlapRatioToText", "BestMatchingOverlapRatioToDetector",
        "BestMatchingDistanceSec", "SecondBestTextIslandIndex", "SecondBestOverlapRatioToText",
        "LaunchDetectorOverlapWithNextIslandRatio", "CausalAssignmentCategory", "HandoffReason",
        "CausalDetectorIndices", "CausalSpanToExpectedRatio", "MeanStreamingAbsErrorMs", "FirstCenterErrorMs",
        "LastCenterErrorMs", "LaunchAttributionCategory", "LaunchAttributionNotes",
    ]
    with (run_dir / "corpus_launch_attribution_forensics_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in fields})

def collect_ownership_retention_forensics_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        rows.extend(derive_ownership_retention_forensics_rows(case_dir))
    return rows


def write_corpus_ownership_retention_forensics(run_dir: Path) -> None:
    rows = collect_ownership_retention_forensics_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_ownership_retention_forensics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        reason = row.get("OwnershipRetentionReason", "") or "Unknown"
        rec = summary.setdefault(reason, {
            "Count": 0.0,
            "MeanMAESum": 0.0,
            "MeanMAECount": 0.0,
            "ReadyRatioSum": 0.0,
            "ReadyRatioCount": 0.0,
            "GapSum": 0.0,
            "GapCount": 0.0,
            "OverlapRatioSum": 0.0,
            "OverlapRatioCount": 0.0,
        })
        rec["Count"] += 1.0
        for field, sum_key, count_key in [
            ("MeanStreamingAbsErrorMs", "MeanMAESum", "MeanMAECount"),
            ("PreviousOwnerReadyRatioBeforeCandidate", "ReadyRatioSum", "ReadyRatioCount"),
            ("GapBeforeCandidateDetectorSec", "GapSum", "GapCount"),
            ("BestDetectorOverlapRatio", "OverlapRatioSum", "OverlapRatioCount"),
        ]:
            val = parse_float(row.get(field, ""))
            if val is not None and math.isfinite(val):
                rec[sum_key] += val
                rec[count_key] += 1.0

    with (run_dir / "corpus_ownership_retention_forensics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "OwnershipRetentionReason", "Count", "Rate", "MeanStreamingAbsErrorMs",
            "MeanPreviousReadyRatio", "MeanGapBeforeCandidateSec", "MeanBestDetectorOverlapRatio",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total = float(len(rows))
        for reason, rec in sorted(summary.items(), key=lambda kv: (-kv[1]["Count"], kv[0])):
            n = rec["Count"]
            w.writerow({
                "OwnershipRetentionReason": reason,
                "Count": f"{n:.0f}",
                "Rate": _fmt(n / total) if total > 0.0 else "",
                "MeanStreamingAbsErrorMs": _fmt(rec["MeanMAESum"] / rec["MeanMAECount"]) if rec["MeanMAECount"] > 0.0 else "",
                "MeanPreviousReadyRatio": _fmt(rec["ReadyRatioSum"] / rec["ReadyRatioCount"]) if rec["ReadyRatioCount"] > 0.0 else "",
                "MeanGapBeforeCandidateSec": _fmt(rec["GapSum"] / rec["GapCount"]) if rec["GapCount"] > 0.0 else "",
                "MeanBestDetectorOverlapRatio": _fmt(rec["OverlapRatioSum"] / rec["OverlapRatioCount"]) if rec["OverlapRatioCount"] > 0.0 else "",
            })

    with (run_dir / "corpus_ownership_retention_forensics_stats.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        slices = {"all_absorbed_by_previous": rows}
        for reason in sorted({r.get("OwnershipRetentionReason", "") or "Unknown" for r in rows}):
            slices[reason] = [r for r in rows if (r.get("OwnershipRetentionReason", "") or "Unknown") == reason]
        metrics = [
            "MeanStreamingAbsErrorMs", "BestDetectorOverlapRatio", "PreviousOwnerReadyRatioBeforeCandidate",
            "GapBeforeCandidateDetectorSec", "CandidateStartMinusNextGuardSec", "PreviousOwnerSpanAfterCandidateRatio",
        ]
        for slice_name, slice_rows in slices.items():
            for metric in metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_rows = sorted(rows, key=lambda r: (-_row_float(r, "MeanStreamingAbsErrorMs"), -_row_float(r, "BestDetectorOverlapRatio")))[:200]
    fields = [
        "case_id", "TextIslandIndex", "FirstWord", "LastWord", "BestDetectorIndex", "BestDetectorStartSec",
        "BestDetectorEndSec", "BestDetectorOverlapRatio", "PreviousOwnerTextIslandIndex", "PreviousOwnerFirstWord",
        "PreviousOwnerLastWord", "PreviousOwnerDetectorIndices", "PreviousOwnerHandoffReason",
        "PreviousOwnerReadyRatioBeforeCandidate", "GapBeforeCandidateDetectorSec", "CandidateStartMinusNextGuardSec",
        "PreviousOwnerSpanAfterCandidateRatio", "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs",
        "OwnershipRetentionReason", "OwnershipRetentionNotes", "Notes",
    ]
    with (run_dir / "corpus_ownership_retention_forensics_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in fields})

def collect_no_causal_detector_span_forensics_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        rows.extend(derive_no_causal_detector_span_forensics_rows(case_dir))
    return rows


def write_corpus_no_causal_detector_span_forensics(run_dir: Path) -> None:
    rows = collect_no_causal_detector_span_forensics_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_no_causal_detector_span_forensics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get("NoSpanForensicCategory", "") or "Unknown"
        rec = summary.setdefault(cat, {
            "Count": 0.0,
            "MeanMAESum": 0.0,
            "MeanMAECount": 0.0,
            "EvidenceSpanSum": 0.0,
            "EvidenceSpanCount": 0.0,
            "BestOverlapRatioSum": 0.0,
            "BestOverlapRatioCount": 0.0,
            "LaunchWindowDetectorSum": 0.0,
        })
        rec["Count"] += 1.0
        mae = parse_float(row.get("MeanStreamingAbsErrorMs", ""))
        ev = parse_float(row.get("EvidenceSpanSec", ""))
        bor = parse_float(row.get("BestDetectorOverlapRatio", ""))
        lw = parse_float(row.get("LaunchWindowDetectorCount", ""))
        if mae is not None and math.isfinite(mae):
            rec["MeanMAESum"] += mae
            rec["MeanMAECount"] += 1.0
        if ev is not None and math.isfinite(ev):
            rec["EvidenceSpanSum"] += ev
            rec["EvidenceSpanCount"] += 1.0
        if bor is not None and math.isfinite(bor):
            rec["BestOverlapRatioSum"] += bor
            rec["BestOverlapRatioCount"] += 1.0
        if lw is not None and math.isfinite(lw):
            rec["LaunchWindowDetectorSum"] += lw

    with (run_dir / "corpus_no_causal_detector_span_forensics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "NoSpanForensicCategory", "Count", "Rate", "MeanStreamingAbsErrorMs",
            "MeanEvidenceSpanSec", "MeanBestDetectorOverlapRatio", "MeanLaunchWindowDetectorCount",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total = float(len(rows))
        for cat, rec in sorted(summary.items(), key=lambda kv: (-kv[1]["Count"], kv[0])):
            n = rec["Count"]
            w.writerow({
                "NoSpanForensicCategory": cat,
                "Count": f"{n:.0f}",
                "Rate": _fmt(n / total) if total > 0.0 else "",
                "MeanStreamingAbsErrorMs": _fmt(rec["MeanMAESum"] / rec["MeanMAECount"]) if rec["MeanMAECount"] > 0.0 else "",
                "MeanEvidenceSpanSec": _fmt(rec["EvidenceSpanSum"] / rec["EvidenceSpanCount"]) if rec["EvidenceSpanCount"] > 0.0 else "",
                "MeanBestDetectorOverlapRatio": _fmt(rec["BestOverlapRatioSum"] / rec["BestOverlapRatioCount"]) if rec["BestOverlapRatioCount"] > 0.0 else "",
                "MeanLaunchWindowDetectorCount": _fmt(rec["LaunchWindowDetectorSum"] / n) if n > 0.0 else "",
            })

    with (run_dir / "corpus_no_causal_detector_span_forensics_stats.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        slices = {"all_no_causal_span": rows}
        for cat in sorted({r.get("NoSpanForensicCategory", "") or "Unknown" for r in rows}):
            slices[cat] = [r for r in rows if (r.get("NoSpanForensicCategory", "") or "Unknown") == cat]
        metrics = [
            "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs", "EvidenceSpanSec",
            "ExpectedSpanSec", "BestDetectorOverlapRatio", "NearbyDetectorCount", "LaunchWindowDetectorCount",
            "NearestDetectorBeforeGapSec", "NearestDetectorAfterGapSec",
        ]
        for slice_name, slice_rows in slices.items():
            for metric in metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_rows = sorted(
        rows,
        key=lambda r: (
            0 if r.get("NoSpanForensicCategory") in {"EvidenceAbsorbedByPreviousIsland", "EvidenceDetectorUnconsumed", "DetectorAfterLaunchWindow", "DetectorInLaunchWindowButRejected"} else 1,
            -_row_float(r, "MeanStreamingAbsErrorMs"),
            -_row_float(r, "EvidenceSpanSec"),
        ),
    )[:200]
    with (run_dir / "corpus_no_causal_detector_span_forensics_worst.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case_id", "TextIslandIndex", "EventCount", "FirstWord", "LastWord", "MappedAudioStartSec",
            "MappedAudioEndSec", "EvidenceStartSec", "EvidenceEndSec", "EvidenceSpanSec",
            "LaunchWindowStartSec", "LaunchWindowEndSec", "OverlappingDetectorCount", "NearbyDetectorCount",
            "LaunchWindowDetectorCount", "BestDetectorIndex", "BestDetectorStartSec", "BestDetectorEndSec",
            "BestDetectorOverlapRatio", "BestDetectorOwnerTextIslandIndices", "NearestDetectorBeforeGapSec",
            "NearestDetectorAfterGapSec", "MeanStreamingAbsErrorMs", "FirstCenterErrorMs", "LastCenterErrorMs",
            "NoSpanForensicCategory", "Notes",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in fields})



def collect_runtime_island_alignment_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        path = case_dir / "runtime_island_alignment.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            row = dict(row)
            row["case_id"] = case_dir.name
            rows.append(row)
    return rows


def write_corpus_runtime_island_alignment_summaries(run_dir: Path) -> None:
    rows = collect_runtime_island_alignment_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order(
        ["case_id"] + [key for row in rows for key in row.keys() if key != "case_id"]
    )
    detail_path = run_dir / "corpus_runtime_island_alignment.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    derived_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        text_island = parse_int(row.get("TextIslandIndex", ""))
        out["IsInitialTextIsland"] = "1" if text_island == 0 else "0"
        for source, target in [
            ("FirstCenterErrorMs", "FirstCenterAbsErrorMs"),
            ("LastCenterErrorMs", "LastCenterAbsErrorMs"),
            ("PrevGapDeltaCommittedVsEvidenceMs", "PrevGapDeltaCommittedVsEvidenceAbsMs"),
            ("PrevGapDeltaCommittedVsDetectorMs", "PrevGapDeltaCommittedVsDetectorAbsMs"),
        ]:
            v = parse_float(row.get(source, ""))
            out[target] = f"{abs(v):.9g}" if v is not None and math.isfinite(v) else ""
        derived_rows.append(out)

    stat_metrics = [
        # PlannerDurationToMappedAudioRatio is intentionally omitted: MappedAudioDurationSec
        # is derived from the planner's own mapped island endpoints, so that ratio is
        # self-referential and should not be used as an external duration-quality metric.
        "PlannerDurationToDetectorAudioRatio",
        "CommittedSpanToEvidenceCenterSpanRatio",
        "CommittedRenderSpanToEvidenceRenderSpanRatio",
        "FirstCenterAbsErrorMs",
        "LastCenterAbsErrorMs",
        "MeanStreamingAbsErrorMs",
        "PrevMappedAudioGapSec",
        "PrevDetectorAudioGapSec",
        "PrevCommittedCenterGapSec",
        "PrevEvidenceCenterGapSec",
        "PrevGapDeltaCommittedVsEvidenceAbsMs",
        "PrevGapDeltaCommittedVsDetectorAbsMs",
    ]

    stat_path = run_dir / "corpus_runtime_island_alignment_stats.csv"
    with stat_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        slices = {
            "all": derived_rows,
            "initial_text_island": [r for r in derived_rows if r.get("IsInitialTextIsland") == "1"],
            "non_initial_text_island": [r for r in derived_rows if r.get("IsInitialTextIsland") != "1"],
        }
        for slice_name, slice_rows in slices.items():
            for metric in stat_metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_fields = [
        "case_id",
        "TextIslandIndex",
        "AudioIslandIndex",
        "FirstEventIndex",
        "LastEventIndex",
        "EventCount",
        "FirstWord",
        "LastWord",
        "PlannerDurationSec",
        "PlannerSyllableCount",
        "PlannerSyllableFloorSec",
        "MappedAudioDurationSec",
        "DetectorAudioDurationSec",
        "CommittedCenterSpanSec",
        "EvidenceCenterSpanSec",
        "PlannerDurationToDetectorAudioRatio",
        "CommittedSpanToEvidenceCenterSpanRatio",
        "FirstCenterErrorMs",
        "LastCenterErrorMs",
        "LastCenterAbsErrorMs",
        "MeanStreamingAbsErrorMs",
        "PrevDetectorAudioGapSec",
        "PrevCommittedCenterGapSec",
        "PrevEvidenceCenterGapSec",
        "PrevGapDeltaCommittedVsEvidenceMs",
        "PrevGapDeltaCommittedVsEvidenceAbsMs",
        "PrevGapDeltaCommittedVsDetectorMs",
        "PrevGapDeltaCommittedVsDetectorAbsMs",
    ]
    worst_rows = sorted(
        derived_rows,
        key=lambda r: max(
            parse_float(r.get("MeanStreamingAbsErrorMs", "")) or 0.0,
            parse_float(r.get("LastCenterAbsErrorMs", "")) or 0.0,
            parse_float(r.get("PrevGapDeltaCommittedVsEvidenceAbsMs", "")) or 0.0,
        ),
        reverse=True,
    )[:50]
    worst_path = run_dir / "corpus_runtime_island_alignment_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields, extrasaction="ignore")
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in worst_fields})


def collect_phrase_duration_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        path = case_dir / "phrase_duration_diagnostics.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            row = dict(row)
            row["case_id"] = case_dir.name
            rows.append(row)
    return rows


def write_corpus_phrase_duration_summaries(run_dir: Path) -> None:
    rows = collect_phrase_duration_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order(
        ["case_id"] + [key for row in rows for key in row.keys() if key != "case_id"]
    )
    detail_path = run_dir / "corpus_phrase_duration_diagnostics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    derived_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        punct = row.get("BoundaryPunctuationKind", "") or "none"
        out["BoundaryPunctuationKindNorm"] = punct
        text_island = parse_int(row.get("TextIslandIndex", ""))
        phrase_index = parse_int(row.get("PhraseIndex", ""))
        out["IsInitialPhraseInIsland"] = "1" if phrase_index == 0 else "0"
        out["HasBoundaryPunctuation"] = "0" if punct in {"", "none"} else "1"
        for source, target in [
            ("PrevPhraseGapDeltaCommittedVsEvidenceMs", "PrevPhraseGapDeltaCommittedVsEvidenceAbsMs"),
        ]:
            v = parse_float(row.get(source, ""))
            out[target] = f"{abs(v):.9g}" if v is not None and math.isfinite(v) else ""
        derived_rows.append(out)

    stat_metrics = [
        "PlannerPhraseSpeechDurationSec",
        "PlannerPhraseEventSpanSec",
        "PlannerIslandDurationSec",
        "PlannerIslandPunctuationSec",
        "CommittedPhraseCenterSpanSec",
        "CommittedPhraseRenderSpanSec",
        "EvidencePhraseCenterSpanSec",
        "EvidencePhraseRenderSpanSec",
        "PlannerPhraseSpeechToEvidenceCenterRatio",
        "PlannerPhraseSpeechToEvidenceRenderRatio",
        "CommittedCenterToEvidenceCenterRatio",
        "PrevPhrasePlannerGapSec",
        "PrevPhraseCommittedCenterGapSec",
        "PrevPhraseEvidenceCenterGapSec",
        "PrevPhraseGapDeltaCommittedVsEvidenceAbsMs",
    ]

    slices: dict[str, list[dict[str, str]]] = {
        "all": derived_rows,
        "initial_phrase_in_island": [r for r in derived_rows if r.get("IsInitialPhraseInIsland") == "1"],
        "non_initial_phrase_in_island": [r for r in derived_rows if r.get("IsInitialPhraseInIsland") != "1"],
        "has_boundary_punctuation": [r for r in derived_rows if r.get("HasBoundaryPunctuation") == "1"],
    }
    for punct in sorted({r.get("BoundaryPunctuationKindNorm", "none") or "none" for r in derived_rows}):
        slices[f"punct_{punct}"] = [r for r in derived_rows if (r.get("BoundaryPunctuationKindNorm", "none") or "none") == punct]

    stat_path = run_dir / "corpus_phrase_duration_stats.csv"
    with stat_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for slice_name, slice_rows in slices.items():
            for metric in stat_metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_fields = [
        "case_id",
        "TextIslandIndex",
        "PhraseIndex",
        "FirstEventIndex",
        "LastEventIndex",
        "EventCount",
        "EvidenceEventCount",
        "FirstWord",
        "LastWord",
        "BoundaryPunctuationKind",
        "PlannerPhraseSpeechDurationSec",
        "EvidencePhraseCenterSpanSec",
        "EvidencePhraseRenderSpanSec",
        "PlannerPhraseSpeechToEvidenceCenterRatio",
        "PrevPhrasePlannerGapSec",
        "PrevPhraseEvidenceCenterGapSec",
        "PrevPhraseGapDeltaCommittedVsEvidenceMs",
        "PrevPhraseGapDeltaCommittedVsEvidenceAbsMs",
    ]
    worst_rows = sorted(
        derived_rows,
        key=lambda r: max(
            abs((parse_float(r.get("PlannerPhraseSpeechToEvidenceCenterRatio", "")) or 1.0) - 1.0),
            (parse_float(r.get("PrevPhraseGapDeltaCommittedVsEvidenceAbsMs", "")) or 0.0) / 1000.0,
        ),
        reverse=True,
    )[:50]
    worst_path = run_dir / "corpus_phrase_duration_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields, extrasaction="ignore")
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in worst_fields})


def collect_soft_punctuation_boundary_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        path = case_dir / "soft_punctuation_boundary_diagnostics.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            row = dict(row)
            row["case_id"] = case_dir.name
            rows.append(row)
    return rows


def write_corpus_soft_punctuation_boundary_summaries(run_dir: Path) -> None:
    rows = collect_soft_punctuation_boundary_rows(run_dir)
    if not rows:
        return

    def word_count_bucket(value: str) -> str:
        v = parse_float(value)
        if v is None or not math.isfinite(v):
            return "unknown"
        if v <= 1:
            return "1"
        if v <= 3:
            return "2_3"
        if v <= 6:
            return "4_6"
        return "7_plus"

    derived_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        punct = row.get("BoundaryPunctuationKind", "") or "none"
        out["BoundaryPunctuationKindNorm"] = punct
        out["PrevWordCountBucket"] = word_count_bucket(row.get("PrevWordCount", ""))
        out["CurrWordCountBucket"] = word_count_bucket(row.get("CurrWordCount", ""))
        for source, target in [
            ("PlannerPocketGapErrorMs", "PlannerPocketGapSignedErrorMs"),
            ("CommittedPocketGapErrorMs", "CommittedPocketGapSignedErrorMs"),
            ("DetectorResumeToEvidenceFirstStartErrorMs", "DetectorResumeToEvidenceFirstStartSignedErrorMs"),
        ]:
            v = parse_float(row.get(source, ""))
            out[target] = f"{v:.9g}" if v is not None and math.isfinite(v) else ""
        for source, target in [
            ("PlannerPocketGapErrorMs", "PlannerPocketGapAbsErrorMs"),
            ("CommittedPocketGapErrorMs", "CommittedPocketGapAbsErrorMs"),
            ("CurrentFirstCenterErrorMs", "CurrentFirstCenterAbsErrorMs"),
            ("CurrentLastCenterErrorMs", "CurrentLastCenterAbsErrorMs"),
            ("DetectorResumeToEvidenceFirstStartErrorMs", "DetectorResumeToEvidenceFirstStartAbsErrorMs"),
        ]:
            v = parse_float(row.get(source, ""))
            out[target] = f"{abs(v):.9g}" if v is not None and math.isfinite(v) else ""
        derived_rows.append(out)

    detail_fields = unique_in_order(
        ["case_id"] + [key for row in derived_rows for key in row.keys() if key != "case_id"]
    )
    detail_path = run_dir / "corpus_soft_punctuation_boundaries.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in derived_rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    stat_metrics = [
        "EvidenceCenterGapSec",
        "EvidenceRenderGapSec",
        "PlannerPocketGapSec",
        "CommittedCenterGapSec",
        "PlannerPocketGapSignedErrorMs",
        "PlannerPocketGapAbsErrorMs",
        "CommittedPocketGapSignedErrorMs",
        "CommittedPocketGapAbsErrorMs",
        "CurrentFirstCenterAbsErrorMs",
        "CurrentLastCenterAbsErrorMs",
        "DetectorResumeGapSec",
        "DetectorResumeToEvidenceFirstStartAbsErrorMs",
    ]
    slices: dict[str, list[dict[str, str]]] = {
        "all_soft_punctuation": derived_rows,
        "detector_split_present": [r for r in derived_rows if truthy_csv(r.get("DetectorSplitPresent", ""))],
        "detector_split_absent": [r for r in derived_rows if not truthy_csv(r.get("DetectorSplitPresent", ""))],
    }
    for punct in sorted({r.get("BoundaryPunctuationKindNorm", "none") or "none" for r in derived_rows}):
        slices[f"punct_{punct}"] = [r for r in derived_rows if (r.get("BoundaryPunctuationKindNorm", "none") or "none") == punct]

    stat_path = run_dir / "corpus_soft_punctuation_boundary_stats.csv"
    with stat_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for slice_name, slice_rows in slices.items():
            for metric in stat_metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    classifier_rows: list[dict[str, str]] = []
    for slice_name, slice_rows in slices.items():
        if not slice_rows:
            continue
        count = len(slice_rows)
        detector_split_count = sum(1 for r in slice_rows if truthy_csv(r.get("DetectorSplitPresent", "")))
        fixed50 = sum(1 for r in slice_rows if truthy_csv(r.get("FixedPocketWithin50Ms", "")))
        fixed100 = sum(1 for r in slice_rows if truthy_csv(r.get("FixedPocketWithin100Ms", "")))
        committed50 = sum(1 for r in slice_rows if truthy_csv(r.get("CommittedPocketWithin50Ms", "")))
        committed100 = sum(1 for r in slice_rows if truthy_csv(r.get("CommittedPocketWithin100Ms", "")))
        classifier_rows.append({
            "slice": slice_name,
            "BoundaryCount": str(count),
            "DetectorSplitCount": str(detector_split_count),
            "DetectorSplitRate": f"{detector_split_count / count:.6g}",
            "FixedPocketWithin50Rate": f"{fixed50 / count:.6g}",
            "FixedPocketWithin100Rate": f"{fixed100 / count:.6g}",
            "CommittedPocketWithin50Rate": f"{committed50 / count:.6g}",
            "CommittedPocketWithin100Rate": f"{committed100 / count:.6g}",
            "EvidenceCenterGapMedianSec": median_metric(slice_rows, "EvidenceCenterGapSec"),
            "EvidenceCenterGapStdDevSec": stats_for(numeric_values(slice_rows, "EvidenceCenterGapSec"))["stddev"],
            "PlannerPocketAbsErrorMedianMs": median_metric(slice_rows, "PlannerPocketGapAbsErrorMs"),
            "CommittedPocketAbsErrorMedianMs": median_metric(slice_rows, "CommittedPocketGapAbsErrorMs"),
        })

    classifier_path = run_dir / "corpus_soft_punctuation_boundary_classifier.csv"
    with classifier_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "slice",
            "BoundaryCount",
            "DetectorSplitCount",
            "DetectorSplitRate",
            "FixedPocketWithin50Rate",
            "FixedPocketWithin100Rate",
            "CommittedPocketWithin50Rate",
            "CommittedPocketWithin100Rate",
            "EvidenceCenterGapMedianSec",
            "EvidenceCenterGapStdDevSec",
            "PlannerPocketAbsErrorMedianMs",
            "CommittedPocketAbsErrorMedianMs",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in classifier_rows:
            w.writerow(row)

    worst_fields = [
        "case_id",
        "TextIslandIndex",
        "PhraseIndex",
        "BoundaryPunctuationKind",
        "PrevLastWord",
        "CurrFirstWord",
        "EvidenceCenterGapSec",
        "EvidenceRenderGapSec",
        "PlannerPocketGapSec",
        "CommittedCenterGapSec",
        "PlannerPocketGapErrorMs",
        "CommittedPocketGapErrorMs",
        "PlannerPocketGapAbsErrorMs",
        "CommittedPocketGapAbsErrorMs",
        "DetectorSplitPresent",
        "DetectorResumeGapSec",
        "DetectorResumeToEvidenceFirstStartAbsErrorMs",
    ]
    worst_rows = sorted(
        derived_rows,
        key=lambda r: max(
            parse_float(r.get("PlannerPocketGapAbsErrorMs", "")) or 0.0,
            parse_float(r.get("CommittedPocketGapAbsErrorMs", "")) or 0.0,
        ),
        reverse=True,
    )[:100]
    worst_path = run_dir / "corpus_soft_punctuation_boundary_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields, extrasaction="ignore")
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in worst_fields})



def _median_or_none(values: list[float]):
    return median(values) if values else None


def _loo_group_prediction(row: dict[str, str], rows: list[dict[str, str]], key_fields: tuple[str, ...], target: str, fallback: float) -> float:
    row_key = tuple(row.get(k, "") for k in key_fields)
    vals: list[float] = []
    for other in rows:
        if other is row:
            continue
        if tuple(other.get(k, "") for k in key_fields) != row_key:
            continue
        v = parse_float(other.get(target, ""))
        if v is not None and math.isfinite(v):
            vals.append(v)
    m = _median_or_none(vals)
    return m if m is not None else fallback


def _eval_gap_model(rows: list[dict[str, str]], name: str, predictions: list[float]) -> dict[str, str]:
    errors: list[float] = []
    abs_errors: list[float] = []
    within50 = 0
    within100 = 0
    used = 0
    for row, pred in zip(rows, predictions):
        actual = parse_float(row.get("EvidenceCenterGapSec", ""))
        if actual is None or not math.isfinite(actual) or pred is None or not math.isfinite(pred):
            continue
        err_ms = (pred - actual) * 1000.0
        errors.append(err_ms)
        abs_errors.append(abs(err_ms))
        within50 += 1 if abs(err_ms) <= 50.0 else 0
        within100 += 1 if abs(err_ms) <= 100.0 else 0
        used += 1
    stat_abs = stats_for(abs_errors)
    stat_signed = stats_for(errors)
    return {
        "model": name,
        "count": str(used),
        "mae_ms": stat_abs["mean"],
        "median_abs_error_ms": stat_abs["median"],
        "p90_abs_error_ms": stat_abs["p90"],
        "signed_bias_mean_ms": stat_signed["mean"],
        "signed_bias_median_ms": stat_signed["median"],
        "within50_rate": f"{within50 / used:.6g}" if used else "",
        "within100_rate": f"{within100 / used:.6g}" if used else "",
    }


def write_soft_punctuation_gap_model_summaries(run_dir: Path, rows: list[dict[str, str]]) -> None:
    valid_rows = []
    for row in rows:
        v = parse_float(row.get("EvidenceCenterGapSec", ""))
        if v is not None and math.isfinite(v):
            valid_rows.append(row)
    if not valid_rows:
        return

    actual_values = numeric_values(valid_rows, "EvidenceCenterGapSec")
    global_median = median(actual_values)
    global_mean = mean(actual_values)

    candidate_predictions: dict[str, list[float]] = {
        "current_planner_pocket": [parse_float(r.get("PlannerPocketGapSec", "")) or 0.0 for r in valid_rows],
        "fixed_100ms": [0.100 for _ in valid_rows],
        "fixed_175ms": [0.175 for _ in valid_rows],
        "fixed_200ms": [0.200 for _ in valid_rows],
        "global_median_in_sample": [global_median for _ in valid_rows],
        "global_mean_in_sample": [global_mean for _ in valid_rows],
        "loo_punctuation_median": [
            _loo_group_prediction(r, valid_rows, ("BoundaryPunctuationKind",), "EvidenceCenterGapSec", global_median)
            for r in valid_rows
        ],
        "loo_structure_class_median": [
            _loo_group_prediction(r, valid_rows, ("SoftBoundaryStructureClass",), "EvidenceCenterGapSec", global_median)
            for r in valid_rows
        ],
        "loo_punctuation_x_structure_median": [
            _loo_group_prediction(r, valid_rows, ("BoundaryPunctuationKind", "SoftBoundaryStructureClass"), "EvidenceCenterGapSec", global_median)
            for r in valid_rows
        ],
        "loo_prev_curr_word_bucket_median": [
            _loo_group_prediction(r, valid_rows, ("PrevWordCountBucket", "CurrWordCountBucket"), "EvidenceCenterGapSec", global_median)
            for r in valid_rows
        ],
        "loo_structure_x_word_bucket_median": [
            _loo_group_prediction(r, valid_rows, ("SoftBoundaryStructureClass", "PrevWordCountBucket", "CurrWordCountBucket"), "EvidenceCenterGapSec", global_median)
            for r in valid_rows
        ],
    }

    model_rows = [_eval_gap_model(valid_rows, name, preds) for name, preds in candidate_predictions.items()]
    model_path = run_dir / "corpus_soft_punctuation_gap_model_candidates.csv"
    fields = ["model", "count", "mae_ms", "median_abs_error_ms", "p90_abs_error_ms", "signed_bias_mean_ms", "signed_bias_median_ms", "within50_rate", "within100_rate"]
    with model_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in model_rows:
            w.writerow(row)

    # Feature-table view for deciding whether the added classes are actually separable.
    feature_rows: list[dict[str, str]] = []
    feature_specs = [
        ("punctuation", ("BoundaryPunctuationKind",)),
        ("structure_class", ("SoftBoundaryStructureClass",)),
        ("punctuation_x_structure", ("BoundaryPunctuationKind", "SoftBoundaryStructureClass")),
        ("prev_curr_word_bucket", ("PrevWordCountBucket", "CurrWordCountBucket")),
        ("structure_x_word_bucket", ("SoftBoundaryStructureClass", "PrevWordCountBucket", "CurrWordCountBucket")),
    ]
    for feature_name, key_fields in feature_specs:
        groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
        for row in valid_rows:
            groups.setdefault(tuple(row.get(k, "") for k in key_fields), []).append(row)
        for key, group_rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            vals = numeric_values(group_rows, "EvidenceCenterGapSec")
            if not vals:
                continue
            rec = {
                "feature": feature_name,
                "key": "|".join(key),
            }
            rec.update(stats_for(vals))
            rec["median_ms"] = f"{median(vals) * 1000.0:.6g}"
            rec["mean_ms"] = f"{mean(vals) * 1000.0:.6g}"
            feature_rows.append(rec)

    feature_path = run_dir / "corpus_soft_punctuation_gap_feature_stats.csv"
    feature_fields = ["feature", "key", "count", "mean", "median", "min", "max", "p90", "stddev", "mean_ms", "median_ms"]
    with feature_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=feature_fields)
        w.writeheader()
        for row in feature_rows:
            w.writerow({k: row.get(k, "") for k in feature_fields})

def collect_speech_attribution_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        metrics_path = case_dir / "metrics_flat.csv"
        scorecard_path = case_dir / "scorecard.csv"
        if metrics_path.exists():
            metrics_rows = read_csv_rows(metrics_path)
            row = dict(metrics_rows[0]) if metrics_rows else {}
        elif scorecard_path.exists():
            row = normalize_row(read_scorecard(scorecard_path), case_dir.name)
            add_derived_metrics(row)
            augment_with_per_case_csv_metrics(case_dir, row)
        else:
            row = {}
        if not row:
            continue
        row["case_id"] = row.get("case_id", case_dir.name) or case_dir.name
        if row.get("SpeechAttributionAudioDurationSec", ""):
            rows.append(row)
    return rows


def write_corpus_speech_attribution_summaries(run_dir: Path, rows: list[dict[str, str]] | None = None) -> None:
    if rows is None:
        rows = collect_speech_attribution_rows(run_dir)
    if not rows:
        return

    fields = unique_in_order([
        "case_id",
        "Transcript",
        "SpeechAttributionAudioDurationSec",
        "SpeechAttributionRegionCount",
        "SpeechAttributionDetectedSpeechMaterialSec",
        "SpeechAttributionDetectedIslandSpanSec",
        "SpeechAttributionDetectorTailSilenceSec",
        "SpeechAttributionDetectorInternalGapSec",
        "SpeechAttributionLeadingSilenceSec",
        "SpeechAttributionTrailingSilenceSec",
        "SpeechAttributionActiveDetectorSpanSec",
        "SpeechAttributionUnassignedAudioSec",
        "SpeechAttributionPlannerDurationSec",
        "SpeechAttributionPlannerSpeechMaterialSec",
        "SpeechAttributionPlannerPunctuationSec",
        "SpeechAttributionPlannedSoftPocketSec",
        "SpeechAttributionDetectedSoftRenderGapSec",
        "SpeechAttributionDetectedSoftCenterGapSec",
        "SpeechAttributionPlannerSpeechMinusDetectedSpeechSec",
        "SpeechAttributionPlannerSpeechAbsErrorSec",
        "SpeechAttributionPlannerDurationMinusAudioSec",
        "SpeechAttributionPlannerSpeechToDetectedSpeechRatio",
        "SpeechAttributionPlannerDurationToAudioRatio",
        "SpeechAttributionDetectedSpeechToAudioRatio",
        "SpeechAttributionDetectedIslandSpanToAudioRatio",
        "SpeechAttributionDetectedGapsAndSilenceToAudioRatio",
        "SpeechAttributionPlannedPunctuationToPlannerDurationRatio",
    ])

    detail_path = run_dir / "corpus_speech_duration_attribution.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    metric_names = [k for k in fields if k not in {"case_id", "Transcript"}]
    stats_path = run_dir / "corpus_speech_duration_attribution_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "median", "min", "max", "p90", "stddev"])
        w.writeheader()
        for metric in metric_names:
            rec = {"metric": metric}
            rec.update(stats_for(numeric_values(rows, metric)))
            w.writerow(rec)

    total_audio = total_metric(rows, "SpeechAttributionAudioDurationSec")
    total_detected_speech = total_metric(rows, "SpeechAttributionDetectedSpeechMaterialSec")
    total_detector_span = total_metric(rows, "SpeechAttributionDetectedIslandSpanSec")
    total_internal_gap = total_metric(rows, "SpeechAttributionDetectorInternalGapSec")
    total_leading = total_metric(rows, "SpeechAttributionLeadingSilenceSec")
    total_trailing = total_metric(rows, "SpeechAttributionTrailingSilenceSec")
    total_tail = total_metric(rows, "SpeechAttributionDetectorTailSilenceSec")
    total_unassigned = total_metric(rows, "SpeechAttributionUnassignedAudioSec")
    total_planner_speech = total_metric(rows, "SpeechAttributionPlannerSpeechMaterialSec")
    total_planner_punctuation = total_metric(rows, "SpeechAttributionPlannerPunctuationSec")
    total_planner_duration = total_metric(rows, "SpeechAttributionPlannerDurationSec")
    total_soft_planned = total_metric(rows, "SpeechAttributionPlannedSoftPocketSec")
    total_soft_detected = total_metric(rows, "SpeechAttributionDetectedSoftRenderGapSec")

    total_path = run_dir / "corpus_speech_duration_attribution_totals.csv"
    with total_path.open("w", newline="", encoding="utf-8") as f:
        fields_totals = ["component", "seconds", "audio_ratio", "notes"]
        w = csv.DictWriter(f, fieldnames=fields_totals)
        w.writeheader()
        def emit(component: str, seconds: float, notes: str = "") -> None:
            w.writerow({
                "component": component,
                "seconds": f"{seconds:.9g}",
                "audio_ratio": f"{seconds / total_audio:.9g}" if total_audio > 1e-9 else "",
                "notes": notes,
            })
        emit("audio_duration", total_audio, "denominator")
        emit("detected_speech_material", total_detected_speech, "sum(AudioBufferLastSpeechSec - AudioBufferStartSec)")
        emit("detected_detector_island_span", total_detector_span, "sum(AudioBufferEndSec - AudioBufferStartSec)")
        emit("detector_tail_silence", total_tail, "detector closure tail inside speech islands")
        emit("detector_internal_gap", total_internal_gap, "gaps between detected speech islands")
        emit("leading_silence", total_leading, "audio before first detected speech island")
        emit("trailing_silence", total_trailing, "audio after final detected speech island")
        emit("unassigned_audio", total_unassigned, "residual after detector spans/gaps/outer silence")
        emit("planner_speech_material", total_planner_speech, "sum PlannerSpeechMaterialSec")
        emit("planner_punctuation", total_planner_punctuation, "sum PlannerPunctuationSec")
        emit("planner_duration", total_planner_duration, "sum PlannerDurationSec")
        emit("planned_soft_pockets", total_soft_planned, "sum PlannerPocketGapSec")
        emit("detected_soft_render_gaps", total_soft_detected, "sum positive EvidenceRenderGapSec")

    ratio_path = run_dir / "corpus_speech_duration_attribution_ratios.csv"
    with ratio_path.open("w", newline="", encoding="utf-8") as f:
        fields_ratios = ["ratio", "value", "interpretation"]
        w = csv.DictWriter(f, fieldnames=fields_ratios)
        w.writeheader()
        def ratio(name: str, numerator: float, denominator: float, interpretation: str) -> None:
            w.writerow({
                "ratio": name,
                "value": f"{numerator / denominator:.9g}" if denominator > 1e-9 else "",
                "interpretation": interpretation,
            })
        ratio("planner_speech_to_detected_speech", total_planner_speech, total_detected_speech, "primary speech-material attribution test")
        ratio("planner_duration_to_audio", total_planner_duration, total_audio, "whole-line budget ratio")
        ratio("detected_speech_to_audio", total_detected_speech, total_audio, "how much wav time is detected speech material")
        ratio("detected_non_speech_to_audio", max(0.0, total_audio - total_detected_speech), total_audio, "outer silence + detector gaps + detector closure tails")
        ratio("planner_punctuation_to_planner_duration", total_planner_punctuation, total_planner_duration, "punctuation share of planner budget")
        ratio("planned_soft_to_detected_soft_render", total_soft_planned, total_soft_detected, "soft punctuation pocket calibration")

    worst_rows = sorted(
        rows,
        key=lambda row: abs(parse_float(row.get("SpeechAttributionPlannerSpeechMinusDetectedSpeechSec", "")) or 0.0),
        reverse=True,
    )[:25]
    worst_path = run_dir / "corpus_speech_duration_attribution_worst.csv"
    with worst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in fields})



def _duration_event_bucket(event_count: int) -> str:
    if event_count <= 4:
        return "events_01_04"
    if event_count <= 8:
        return "events_05_08"
    if event_count <= 12:
        return "events_09_12"
    if event_count <= 20:
        return "events_13_20"
    return "events_21_plus"


def _duration_syllable_bucket(syllable_count: int) -> str:
    if syllable_count <= 2:
        return "syllables_00_02"
    if syllable_count <= 5:
        return "syllables_03_05"
    if syllable_count <= 8:
        return "syllables_06_08"
    if syllable_count <= 12:
        return "syllables_09_12"
    return "syllables_13_plus"


def _duration_span_bucket(span_sec: float) -> str:
    if span_sec < 0.75:
        return "target_under_0750ms"
    if span_sec < 1.25:
        return "target_0750_1250ms"
    if span_sec < 2.00:
        return "target_1250_2000ms"
    if span_sec < 3.00:
        return "target_2000_3000ms"
    return "target_3000ms_plus"


def _duration_soft_bucket(punctuation_sec: float) -> str:
    if punctuation_sec <= 0.001:
        return "no_soft_pocket"
    if punctuation_sec <= 0.200:
        return "soft_pocket_small"
    return "soft_pocket_large"


def _duration_bucket_stats(rows: list[dict[str, str]], bucket_key: str) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for row in rows:
        bucket = row.get(bucket_key, "") or "unknown"
        ratio = parse_float(row.get("TargetToPlannerRatio", ""))
        if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
            continue
        by.setdefault(bucket, []).append(ratio)
    return by


def _duration_predict(row: dict[str, str], model: str, global_mean: float, global_median: float, multipliers: dict[str, dict[str, float]]) -> float:
    planner = _row_float(row, "PlannerDurationSec", 0.0)
    if planner <= 0.0:
        return 0.0
    if model == "current":
        mult = 1.0
    elif model == "global_mean_ratio":
        mult = global_mean
    elif model == "global_median_ratio":
        mult = global_median
    elif model == "event_count_bucket_median":
        mult = multipliers.get("EventBucket", {}).get(row.get("EventBucket", ""), global_median)
    elif model == "syllable_bucket_median":
        mult = multipliers.get("SyllableBucket", {}).get(row.get("SyllableBucket", ""), global_median)
    elif model == "soft_punctuation_bucket_median":
        mult = multipliers.get("SoftPunctuationBucket", {}).get(row.get("SoftPunctuationBucket", ""), global_median)
    elif model == "target_span_bucket_median_oracle":
        # Diagnostic upper-bound only. This uses the target span bucket and must not be
        # treated as deployable runtime logic; it tells us whether the duration error is
        # concentrated in short/medium/long islands.
        mult = multipliers.get("TargetSpanBucket", {}).get(row.get("TargetSpanBucket", ""), global_median)
    elif model == "island_ordinal_bucket_median":
        mult = multipliers.get("IslandOrdinalBucket", {}).get(row.get("IslandOrdinalBucket", ""), global_median)
    else:
        mult = 1.0
    return planner * mult


def write_corpus_duration_model_fit_summaries(run_dir: Path, summary_rows: list[dict[str, str]]) -> None:
    """AU18a duration-prior audit.

    AU16 deliberately removed detector-end authority from runtime timing. That makes the
    text/prosody island duration prior the main determinant of island end timing. This
    diagnostic evaluates that prior against the per-island evidence spans already emitted
    by liplab; it does not change runtime behavior and does not require WAV DSP.
    """
    runtime_rows = read_csv_rows(run_dir / "corpus_runtime_island_alignment.csv")
    if not runtime_rows:
        return

    transcript_by_case = {r.get("case_id", ""): r.get("Transcript", "") for r in summary_rows}
    fit_rows: list[dict[str, str]] = []
    for row in runtime_rows:
        case_id = row.get("case_id", "")
        planner = _row_float(row, "PlannerDurationSec", 0.0)
        target_render = _row_float(row, "EvidenceRenderSpanSec", 0.0)
        target_center = _row_float(row, "EvidenceCenterSpanSec", 0.0)
        detector = _row_float(row, "DetectorAudioDurationSec", 0.0)
        committed = _row_float(row, "CommittedCenterSpanSec", 0.0)
        if planner <= 0.0 or target_render <= 0.0:
            continue
        event_count = _row_int(row, "EventCount", 0)
        syllables = _row_int(row, "PlannerSyllableCount", 0)
        punctuation = _row_float(row, "PlannerPunctuationSec", 0.0)
        island_index = _row_int(row, "TextIslandIndex", 0)
        target_ratio = target_render / planner if planner > 1e-9 else 0.0
        center_ratio = target_center / planner if planner > 1e-9 else 0.0
        detector_ratio = detector / planner if planner > 1e-9 else 0.0
        committed_ratio = committed / target_render if target_render > 1e-9 else 0.0
        fit_rows.append({
            "case_id": case_id,
            "Transcript": transcript_by_case.get(case_id, ""),
            "TextIslandIndex": row.get("TextIslandIndex", ""),
            "AudioIslandIndex": row.get("AudioIslandIndex", ""),
            "FirstWord": row.get("FirstWord", ""),
            "LastWord": row.get("LastWord", ""),
            "EventCount": str(event_count),
            "PlannerSyllableCount": str(syllables),
            "PlannerDurationSec": _fmt(planner),
            "PlannerSpeechMaterialSec": row.get("PlannerSpeechMaterialSec", ""),
            "PlannerPunctuationSec": _fmt(punctuation),
            "EvidenceRenderSpanSec": _fmt(target_render),
            "EvidenceCenterSpanSec": _fmt(target_center),
            "DetectorAudioDurationSec": _fmt(detector),
            "CommittedCenterSpanSec": _fmt(committed),
            "TargetToPlannerRatio": _fmt(target_ratio),
            "EvidenceCenterToPlannerRatio": _fmt(center_ratio),
            "DetectorToPlannerRatio": _fmt(detector_ratio),
            "CommittedToTargetRatio": _fmt(committed_ratio),
            "PlannerMinusTargetSec": _fmt(planner - target_render),
            "PlannerAbsErrorMs": _fmt(abs(planner - target_render) * 1000.0),
            "CommittedMinusTargetSec": _fmt(committed - target_render),
            "CommittedAbsErrorMs": _fmt(abs(committed - target_render) * 1000.0),
            "EventBucket": _duration_event_bucket(event_count),
            "SyllableBucket": _duration_syllable_bucket(syllables),
            "SoftPunctuationBucket": _duration_soft_bucket(punctuation),
            "TargetSpanBucket": _duration_span_bucket(target_render),
            "IslandOrdinalBucket": "first_island" if island_index == 0 else "later_island",
        })

    if not fit_rows:
        return

    detail_fields = [
        "case_id", "Transcript", "TextIslandIndex", "AudioIslandIndex", "FirstWord", "LastWord",
        "EventCount", "PlannerSyllableCount", "PlannerDurationSec", "PlannerSpeechMaterialSec",
        "PlannerPunctuationSec", "EvidenceRenderSpanSec", "EvidenceCenterSpanSec", "DetectorAudioDurationSec",
        "CommittedCenterSpanSec", "TargetToPlannerRatio", "EvidenceCenterToPlannerRatio", "DetectorToPlannerRatio",
        "CommittedToTargetRatio", "PlannerMinusTargetSec", "PlannerAbsErrorMs", "CommittedMinusTargetSec",
        "CommittedAbsErrorMs", "EventBucket", "SyllableBucket", "SoftPunctuationBucket", "TargetSpanBucket",
        "IslandOrdinalBucket",
    ]
    with (run_dir / "corpus_duration_model_fit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        w.writeheader()
        for row in fit_rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    ratios = numeric_values(fit_rows, "TargetToPlannerRatio")
    global_mean = mean(ratios) if ratios else 1.0
    global_median = median(ratios) if ratios else 1.0

    bucket_keys = ["EventBucket", "SyllableBucket", "SoftPunctuationBucket", "TargetSpanBucket", "IslandOrdinalBucket"]
    bucket_multipliers: dict[str, dict[str, float]] = {}
    bucket_rows: list[dict[str, str]] = []
    for bucket_key in bucket_keys:
        by = _duration_bucket_stats(fit_rows, bucket_key)
        bucket_multipliers[bucket_key] = {bucket: median(vals) for bucket, vals in by.items() if vals}
        for bucket, vals in sorted(by.items()):
            members = [r for r in fit_rows if (r.get(bucket_key, "") or "unknown") == bucket]
            current_errors = [abs(_row_float(r, "PlannerDurationSec", 0.0) - _row_float(r, "EvidenceRenderSpanSec", 0.0)) * 1000.0 for r in members]
            mult = bucket_multipliers[bucket_key].get(bucket, global_median)
            fitted_errors = [abs((_row_float(r, "PlannerDurationSec", 0.0) * mult) - _row_float(r, "EvidenceRenderSpanSec", 0.0)) * 1000.0 for r in members]
            rec = {
                "BucketType": bucket_key,
                "Bucket": bucket,
                "IslandCount": str(len(vals)),
                "TargetToPlannerRatioMean": _fmt(mean(vals)),
                "TargetToPlannerRatioMedian": _fmt(median(vals)),
                "CurrentMAEMs": _fmt(mean(current_errors)) if current_errors else "",
                "BucketMedianFitMAEMs": _fmt(mean(fitted_errors)) if fitted_errors else "",
            }
            bucket_rows.append(rec)
    bucket_fields = ["BucketType", "Bucket", "IslandCount", "TargetToPlannerRatioMean", "TargetToPlannerRatioMedian", "CurrentMAEMs", "BucketMedianFitMAEMs"]
    with (run_dir / "corpus_duration_model_by_bucket.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bucket_fields)
        w.writeheader()
        for row in bucket_rows:
            w.writerow(row)

    candidate_models = [
        "current",
        "global_mean_ratio",
        "global_median_ratio",
        "event_count_bucket_median",
        "syllable_bucket_median",
        "soft_punctuation_bucket_median",
        "island_ordinal_bucket_median",
        "target_span_bucket_median_oracle",
    ]
    candidate_rows: list[dict[str, str]] = []
    for model in candidate_models:
        errors_ms: list[float] = []
        signed_ms: list[float] = []
        ratios_pred: list[float] = []
        for row in fit_rows:
            target = _row_float(row, "EvidenceRenderSpanSec", 0.0)
            if target <= 0.0:
                continue
            pred = _duration_predict(row, model, global_mean, global_median, bucket_multipliers)
            err = pred - target
            signed_ms.append(err * 1000.0)
            errors_ms.append(abs(err) * 1000.0)
            ratios_pred.append(pred / target if target > 1e-9 else 0.0)
        rec = {
            "Model": model,
            "IslandCount": str(len(errors_ms)),
            "MeanAbsErrorMs": _fmt(mean(errors_ms)) if errors_ms else "",
            "MedianAbsErrorMs": _fmt(median(errors_ms)) if errors_ms else "",
            "P90AbsErrorMs": _fmt(sorted(errors_ms)[int(min(len(errors_ms) - 1, math.ceil(0.90 * len(errors_ms)) - 1))]) if errors_ms else "",
            "MeanSignedErrorMs": _fmt(mean(signed_ms)) if signed_ms else "",
            "MeanPredictedToTargetRatio": _fmt(mean(ratios_pred)) if ratios_pred else "",
            "Notes": "diagnostic_oracle_upper_bound" if model == "target_span_bucket_median_oracle" else "diagnostic_candidate",
        }
        candidate_rows.append(rec)
    candidate_fields = ["Model", "IslandCount", "MeanAbsErrorMs", "MedianAbsErrorMs", "P90AbsErrorMs", "MeanSignedErrorMs", "MeanPredictedToTargetRatio", "Notes"]
    with (run_dir / "corpus_duration_model_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=candidate_fields)
        w.writeheader()
        for row in candidate_rows:
            w.writerow(row)

    worst_rows = sorted(fit_rows, key=lambda r: _row_float(r, "PlannerAbsErrorMs", 0.0), reverse=True)[:100]
    with (run_dir / "corpus_duration_model_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})


def write_corpus_phase_summary(run_dir: Path, rows: list[dict[str, str]]) -> None:
    total_events = total_metric(rows, "PlannedEventCount")
    total_text_islands = total_metric(rows, "TextIslandCount")
    total_audio_islands = total_metric(rows, "AudioIslandCount")
    total_fallback_islands = total_metric(rows, "FallbackIslandCount")
    total_unmapped_events = total_metric(rows, "UnmappedEventCount")
    total_proposed_nudges = total_metric(rows, "ProposedNudgeCount")
    total_accepted_nudges = total_metric(rows, "AcceptedNudgeCount")
    total_weak_pose = total_metric(rows, "WeakPoseCount")
    total_mbp = total_metric(rows, "MBPCount")
    total_mbp_accepted = total_metric(rows, "MBPAcceptedNudgeCount")
    total_fv = total_metric(rows, "FVCount")
    total_fv_accepted = total_metric(rows, "FVAcceptedNudgeCount")
    total_woo = total_metric(rows, "WOOCount")
    total_woo_accepted = total_metric(rows, "WOOAcceptedNudgeCount")
    total_shch = total_metric(rows, "SHCHCount")
    total_shch_accepted = total_metric(rows, "SHCHAcceptedNudgeCount")

    lines = [
        "LipLab corpus phase summary",
        "============================",
        "",
        f"cases processed: {len(rows)}",
        f"total events: {total_events:.0f}",
        f"total text islands: {total_text_islands:.0f}",
        f"total audio islands: {total_audio_islands:.0f}",
        f"fallback island rate: {weighted_rate(total_fallback_islands, total_text_islands)}",
        f"unmapped event rate: {weighted_rate(total_unmapped_events, total_events)}",
        "",
        "Timing",
        "------",
        f"raw MAE mean ms: {mean_metric(rows, 'RawTextTimingMAEMs')}",
        f"raw MAE median ms: {median_metric(rows, 'RawTextTimingMAEMs')}",
        f"streaming MAE mean ms: {mean_metric(rows, 'StreamingTimingMAEMs')}",
        f"streaming MAE median ms: {median_metric(rows, 'StreamingTimingMAEMs')}",
        f"streaming-minus-raw MAE mean ms: {mean_metric(rows, 'StreamingMinusRawMAEMs')}",
        f"duration ratio text/audio mean: {mean_metric(rows, 'DurationRatioTextToAudio')}",
        f"duration ratio text/audio median: {median_metric(rows, 'DurationRatioTextToAudio')}",
        f"runtime committed duration ratio/audio mean: {mean_metric(rows, 'RuntimeCommittedDurationRatioToAudio')}",
        f"runtime committed duration ratio/audio median: {median_metric(rows, 'RuntimeCommittedDurationRatioToAudio')}",
        f"runtime planner island duration ratio/audio mean: {mean_metric(rows, 'RuntimePlannerIslandDurationRatioToAudio')}",
        f"runtime planner island duration ratio/audio median: {median_metric(rows, 'RuntimePlannerIslandDurationRatioToAudio')}",
        f"speech attribution planner speech/detected speech mean: {mean_metric(rows, 'SpeechAttributionPlannerSpeechToDetectedSpeechRatio')}",
        f"speech attribution planner speech/detected speech median: {median_metric(rows, 'SpeechAttributionPlannerSpeechToDetectedSpeechRatio')}",
        f"speech attribution detected speech/audio mean: {mean_metric(rows, 'SpeechAttributionDetectedSpeechToAudioRatio')}",
        f"speech attribution detected speech/audio median: {median_metric(rows, 'SpeechAttributionDetectedSpeechToAudioRatio')}",
        "",
        "Nudging",
        "-------",
        f"nudge proposal rate: {weighted_rate(total_proposed_nudges, total_events)}",
        f"nudge acceptance rate: {weighted_rate(total_accepted_nudges, total_events)}",
        f"accepted among proposed nudge rate: {weighted_rate(total_accepted_nudges, total_proposed_nudges)}",
        f"alpha32 scheduled→target MAE mean ms: {mean_metric(rows, 'Alpha32ScheduledTargetMAEMs')}",
        f"alpha32 final→target MAE mean ms: {mean_metric(rows, 'Alpha32FinalTargetMAEMs')}",
        f"alpha32 target MAE improvement mean ms: {mean_metric(rows, 'Alpha32TargetMAEImprovementMs')}",
        f"alpha32 face→target MAE mean ms: {mean_metric(rows, 'Alpha32FaceTargetMAEMs')}",
        f"alpha32 accepted shift abs mean ms: {mean_metric(rows, 'Alpha32AcceptedShiftAbsMeanMs')}",
        f"MBP acceptance rate: {weighted_rate(total_mbp_accepted, total_mbp)}",
        f"MBP target MAE improvement mean ms: {mean_metric(rows, 'Alpha32MBPTargetMAEImprovementMs')}",
        f"FV target MAE improvement mean ms: {mean_metric(rows, 'Alpha32FVTargetMAEImprovementMs')}",
        f"WOO target MAE improvement mean ms: {mean_metric(rows, 'Alpha32WOOTargetMAEImprovementMs')}",
        f"rounded target MAE improvement mean ms: {mean_metric(rows, 'Alpha32RoundedTargetMAEImprovementMs')}",
        f"FV acceptance rate: {weighted_rate(total_fv_accepted, total_fv)}",
        f"WOO acceptance rate: {weighted_rate(total_woo_accepted, total_woo)}",
        f"SHCH acceptance rate: {weighted_rate(total_shch_accepted, total_shch)}",
        "",
        "FaceDriver",
        "----------",
        f"weak pose rate: {weighted_rate(total_weak_pose, total_events)}",
        f"max visible gap mean sec: {mean_metric(rows, 'MaxVisiblePoseGapSec')}",
        f"max visible gap median sec: {median_metric(rows, 'MaxVisiblePoseGapSec')}",
        "",
        "Invariants",
        "----------",
        f"hard invariant failures: {total_metric(rows, 'HardInvariantFailureCount'):.0f}",
    ]
    (run_dir / "corpus_phase_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_flat_metrics(case_dir: Path, row: dict[str, str]) -> None:
    out = case_dir / "metrics_flat.csv"
    keys = ["case_id"] + sorted(k for k in row.keys() if k != "case_id")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in keys})


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    vals = []
    for row in rows:
        v = parse_float(row.get(key, ""))
        if v is not None and math.isfinite(v):
            vals.append(v)
    return vals


def stats_for(values: list[float]) -> dict[str, str]:
    if not values:
        return {
            "count": "0",
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "p90": "",
            "stddev": "",
        }
    sv = sorted(values)
    p90_idx = min(len(sv) - 1, int(math.ceil(len(sv) * 0.90)) - 1)
    return {
        "count": str(len(values)),
        "mean": f"{mean(values):.6g}",
        "median": f"{median(values):.6g}",
        "min": f"{min(values):.6g}",
        "max": f"{max(values):.6g}",
        "p90": f"{sv[p90_idx]:.6g}",
        "stddev": f"{pstdev(values):.6g}" if len(values) > 1 else "0",
    }




def collect_first_onset_comparison_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        path = case_dir / "first_onset_detector_comparison.csv"
        if not path.exists():
            continue
        case_rows = read_csv_rows(path)
        for row in case_rows:
            row = dict(row)
            row["case_id"] = case_dir.name
            rows.append(row)
    return rows


def write_corpus_first_onset_comparison_summaries(run_dir: Path) -> None:
    rows = collect_first_onset_comparison_rows(run_dir)
    if not rows:
        return

    detail_fields = unique_in_order(["case_id"] + [key for row in rows for key in row.keys() if key != "case_id"])
    with (run_dir / "corpus_first_onset_detector_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    stat_metrics = [
        "CandidateStartSec",
        "AcceptedStartSec",
        "ActualFirstSpeechStartSec",
        "ActualMinusModeAcceptedMs",
        "StrictMinusLegacyAcceptedMs",
        "StrictMinusLegacyCandidateMs",
        "OpenThreshold",
        "NoiseFloorRMS",
        "RMS",
        "RMSNorm",
        "Flux",
        "ZCR",
    ]
    slices: dict[str, list[dict[str, str]]] = {"all": rows}
    for mode in sorted({row.get("Mode", "") for row in rows}):
        if mode:
            slices[f"mode_{mode}"] = [row for row in rows if row.get("Mode", "") == mode]

    with (run_dir / "corpus_first_onset_detector_comparison_stats.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for slice_name, slice_rows in slices.items():
            for metric in stat_metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    # One row per case, useful for quickly sorting cases where strict onset moved far from legacy.
    pair_rows: list[dict[str, str]] = []
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row.get("case_id", ""), []).append(row)
    for case_id, case_rows in sorted(by_case.items()):
        legacy = next((r for r in case_rows if r.get("Mode") == "legacy_20ms_energy_backdated"), None)
        strict = next((r for r in case_rows if r.get("Mode") == "strict_55ms_voiced_no_backdate"), None)
        if not legacy or not strict:
            continue
        pair_rows.append({
            "case_id": case_id,
            "LegacyFound": legacy.get("Found", ""),
            "StrictFound": strict.get("Found", ""),
            "LegacyAcceptedStartSec": legacy.get("AcceptedStartSec", ""),
            "StrictAcceptedStartSec": strict.get("AcceptedStartSec", ""),
            "StrictMinusLegacyAcceptedMs": strict.get("StrictMinusLegacyAcceptedMs", ""),
            "StrictMinusLegacyCandidateMs": strict.get("StrictMinusLegacyCandidateMs", ""),
            "StrictRMS": strict.get("RMS", ""),
            "StrictRMSNorm": strict.get("RMSNorm", ""),
            "StrictFlux": strict.get("Flux", ""),
            "StrictZCR": strict.get("ZCR", ""),
        })
    with (run_dir / "corpus_first_onset_detector_comparison_worst.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["case_id", "LegacyFound", "StrictFound", "LegacyAcceptedStartSec", "StrictAcceptedStartSec", "StrictMinusLegacyAcceptedMs", "StrictMinusLegacyCandidateMs", "StrictRMS", "StrictRMSNorm", "StrictFlux", "StrictZCR"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(pair_rows, key=lambda r: abs(parse_float(r.get("StrictMinusLegacyAcceptedMs", "")) or 0.0), reverse=True)[:100]:
            w.writerow({k: row.get(k, "") for k in fields})


def collect_prosody_group_timing_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return rows
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        committed = read_csv_rows(case_dir / "committed_events.csv")
        timing = read_csv_rows(case_dir / "timing_diagnostics.csv")
        if not committed or not timing:
            continue
        evidence_by_event: dict[int, dict[str, str]] = {}
        for row in timing:
            idx = parse_int(row.get("EventIndex", ""))
            if idx is not None:
                evidence_by_event[idx] = row

        groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}
        for row in committed:
            text_island = parse_int(row.get("TextIslandIndex", ""))
            phrase = parse_int(row.get("PhraseIndex", ""))
            group = parse_int(row.get("PlannerProsodyGroupIndex", ""))
            if text_island is None or phrase is None or group is None:
                continue
            groups.setdefault((text_island, phrase, group), []).append(row)

        for (text_island, phrase, group), group_rows in groups.items():
            group_rows.sort(key=lambda r: parse_int(r.get("EventIndex", "")) if parse_int(r.get("EventIndex", "")) is not None else 10**9)
            first = group_rows[0]
            last = group_rows[-1]
            first_event = parse_int(first.get("EventIndex", ""))
            last_event = parse_int(last.get("EventIndex", ""))
            ev_rows = [evidence_by_event[idx] for idx in [parse_int(r.get("EventIndex", "")) for r in group_rows] if idx is not None and idx in evidence_by_event]
            evidence_count = len(ev_rows)
            evidence_first_center = parse_float(ev_rows[0].get("EvidenceCenterSec", "")) if ev_rows else None
            evidence_last_center = parse_float(ev_rows[-1].get("EvidenceCenterSec", "")) if ev_rows else None
            evidence_first_start = parse_float(ev_rows[0].get("EvidenceStartSec", "")) if ev_rows else None
            evidence_last_end = parse_float(ev_rows[-1].get("EvidenceEndSec", "")) if ev_rows else None
            committed_first_center = parse_float(first.get("CommittedPlaybackCenterSec", "")) or 0.0
            committed_last_center = parse_float(last.get("CommittedPlaybackCenterSec", "")) or committed_first_center
            committed_first_render = parse_float(first.get("RenderStartSec", "")) or committed_first_center
            committed_last_render = parse_float(last.get("RenderEndSec", "")) or committed_last_center
            planner_alloc = parse_float(first.get("PlannerProsodyGroupAllocatedSec", "")) or 0.0
            group_event_count = parse_int(first.get("PlannerProsodyGroupEventCount", ""))
            group_event_count_model = parse_float(first.get("PlannerProsodyGroupEventCountModelSec", ""))
            island_event_count_model = parse_float(first.get("PlannerProsodyIslandEventCountModelSec", ""))
            island_speech_budget = parse_float(first.get("PlannerProsodyIslandSpeechBudgetSec", ""))
            allocation_scale = parse_float(first.get("PlannerProsodyAllocationScale", ""))
            planner_first_offset = parse_float(first.get("PlannerEventPredictedOffsetSec", "")) or 0.0
            planner_last_offset = parse_float(last.get("PlannerEventPredictedOffsetSec", "")) or planner_first_offset
            evidence_center_span = max(0.0, (evidence_last_center - evidence_first_center)) if evidence_first_center is not None and evidence_last_center is not None else 0.0
            evidence_render_span = max(0.0, (evidence_last_end - evidence_first_start)) if evidence_first_start is not None and evidence_last_end is not None else 0.0
            committed_center_span = max(0.0, committed_last_center - committed_first_center)
            committed_render_span = max(0.0, committed_last_render - committed_first_render)
            planner_event_span = max(0.0, planner_last_offset - planner_first_offset)
            role = first.get("PlannerProsodyRole", "") or "unknown"
            rec = {
                "case_id": case_dir.name,
                "TextIslandIndex": str(text_island),
                "PhraseIndex": str(phrase),
                "PlannerProsodyGroupIndex": str(group),
                "PlannerProsodyRole": role,
                "FirstEventIndex": str(first_event) if first_event is not None else "",
                "LastEventIndex": str(last_event) if last_event is not None else "",
                "EventCount": str(len(group_rows)),
                "PlannerProsodyGroupEventCount": str(group_event_count) if group_event_count is not None else "",
                "EvidenceEventCount": str(evidence_count),
                "FirstWord": first.get("SourceWord", ""),
                "LastWord": last.get("SourceWord", ""),
                "PlannerGroupAllocatedSec": _fmt(planner_alloc),
                "PlannerGroupEventCountModelSec": _fmt(group_event_count_model) if group_event_count_model is not None else "",
                "PlannerIslandEventCountModelSec": _fmt(island_event_count_model) if island_event_count_model is not None else "",
                "PlannerIslandSpeechBudgetSec": _fmt(island_speech_budget) if island_speech_budget is not None else "",
                "PlannerAllocationScale": _fmt(allocation_scale) if allocation_scale is not None else "",
                "PlannerGroupEventSpanSec": _fmt(planner_event_span),
                "CommittedGroupCenterSpanSec": _fmt(committed_center_span),
                "CommittedGroupRenderSpanSec": _fmt(committed_render_span),
                "EvidenceGroupCenterSpanSec": _fmt(evidence_center_span),
                "EvidenceGroupRenderSpanSec": _fmt(evidence_render_span),
                "PlannerGroupAllocatedToEvidenceCenterRatio": _fmt(planner_alloc / evidence_center_span) if evidence_center_span > 1e-9 else "",
                "PlannerGroupEventCountModelToEvidenceCenterRatio": _fmt(group_event_count_model / evidence_center_span) if group_event_count_model is not None and evidence_center_span > 1e-9 else "",
                "PlannerIslandSpeechBudgetToEventCountModelRatio": _fmt(island_speech_budget / island_event_count_model) if island_speech_budget is not None and island_event_count_model is not None and island_event_count_model > 1e-9 else "",
                "PlannerGroupAllocatedToEvidenceRenderRatio": _fmt(planner_alloc / evidence_render_span) if evidence_render_span > 1e-9 else "",
                "CommittedCenterToEvidenceCenterRatio": _fmt(committed_center_span / evidence_center_span) if evidence_center_span > 1e-9 else "",
                "PlannerMinusEvidenceCenterSec": _fmt(planner_alloc - evidence_center_span),
                "PlannerMinusEvidenceRenderSec": _fmt(planner_alloc - evidence_render_span),
            }
            rows.append(rec)
    return rows


def write_corpus_prosody_group_timing_summaries(run_dir: Path) -> None:
    rows = collect_prosody_group_timing_rows(run_dir)
    if not rows:
        return
    detail_fields = unique_in_order([key for row in rows for key in row.keys()])
    with (run_dir / "corpus_prosody_group_timing_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in detail_fields})

    trace_fields = [
        "case_id", "TextIslandIndex", "PhraseIndex", "PlannerProsodyGroupIndex",
        "EventCount", "PlannerProsodyGroupEventCount", "EvidenceEventCount",
        "PlannerGroupEventCountModelSec", "PlannerIslandEventCountModelSec",
        "PlannerIslandSpeechBudgetSec", "PlannerAllocationScale",
        "PlannerGroupAllocatedSec", "EvidenceGroupCenterSpanSec", "EvidenceGroupRenderSpanSec",
        "PlannerGroupEventCountModelToEvidenceCenterRatio",
        "PlannerGroupAllocatedToEvidenceCenterRatio",
        "PlannerIslandSpeechBudgetToEventCountModelRatio",
        "CommittedCenterToEvidenceCenterRatio",
    ]
    with (run_dir / "corpus_prosody_group_allocation_trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=trace_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in trace_fields})

    derived_rows: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        text_island = parse_int(row.get("TextIslandIndex", ""))
        phrase = parse_int(row.get("PhraseIndex", ""))
        event_count = parse_int(row.get("EventCount", "")) or 0
        out["IsInitialTextIsland"] = "1" if text_island == 0 else "0"
        out["IsInitialPhrase"] = "1" if phrase == 0 else "0"
        out["EventCountBucket"] = "1" if event_count <= 1 else ("2_3" if event_count <= 3 else "4_plus")
        derived_rows.append(out)

    stat_metrics = [
        "PlannerGroupAllocatedSec",
        "PlannerGroupEventSpanSec",
        "CommittedGroupCenterSpanSec",
        "CommittedGroupRenderSpanSec",
        "EvidenceGroupCenterSpanSec",
        "EvidenceGroupRenderSpanSec",
        "PlannerGroupAllocatedToEvidenceCenterRatio",
        "PlannerGroupEventCountModelToEvidenceCenterRatio",
        "PlannerIslandSpeechBudgetToEventCountModelRatio",
        "PlannerGroupAllocatedToEvidenceRenderRatio",
        "CommittedCenterToEvidenceCenterRatio",
        "PlannerMinusEvidenceCenterSec",
        "PlannerMinusEvidenceRenderSec",
    ]
    slices: dict[str, list[dict[str, str]]] = {
        "all": derived_rows,
        "initial_text_island": [r for r in derived_rows if r.get("IsInitialTextIsland") == "1"],
        "non_initial_text_island": [r for r in derived_rows if r.get("IsInitialTextIsland") != "1"],
        "initial_phrase": [r for r in derived_rows if r.get("IsInitialPhrase") == "1"],
        "non_initial_phrase": [r for r in derived_rows if r.get("IsInitialPhrase") != "1"],
    }
    for role in sorted({r.get("PlannerProsodyRole", "unknown") or "unknown" for r in derived_rows}):
        slices[f"role_{role}"] = [r for r in derived_rows if (r.get("PlannerProsodyRole", "unknown") or "unknown") == role]
    for bucket in sorted({r.get("EventCountBucket", "") for r in derived_rows}):
        if bucket:
            slices[f"event_count_{bucket}"] = [r for r in derived_rows if r.get("EventCountBucket") == bucket]

    with (run_dir / "corpus_prosody_group_timing_stats.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["slice", "metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for slice_name, slice_rows in slices.items():
            for metric in stat_metrics:
                rec = {"slice": slice_name, "metric": metric}
                rec.update(stats_for(numeric_values(slice_rows, metric)))
                w.writerow(rec)

    worst_fields = [
        "case_id", "TextIslandIndex", "PhraseIndex", "PlannerProsodyGroupIndex", "PlannerProsodyRole",
        "FirstEventIndex", "LastEventIndex", "EventCount", "EvidenceEventCount", "FirstWord", "LastWord",
        "PlannerGroupAllocatedSec", "EvidenceGroupCenterSpanSec", "EvidenceGroupRenderSpanSec",
        "PlannerGroupAllocatedToEvidenceCenterRatio", "PlannerGroupAllocatedToEvidenceRenderRatio",
        "PlannerMinusEvidenceCenterSec", "PlannerMinusEvidenceRenderSec",
    ]
    with (run_dir / "corpus_prosody_group_timing_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in sorted(derived_rows, key=lambda r: abs((parse_float(r.get("PlannerGroupAllocatedToEvidenceCenterRatio", "")) or 1.0) - 1.0), reverse=True)[:100]:
            w.writerow({k: row.get(k, "") for k in worst_fields})


def write_corpus_prosody_group_duration_model_summaries(run_dir: Path) -> None:
    """Run the alpha14 dependency-free retrainer on alpha13 group diagnostics."""
    diag = run_dir / "corpus_prosody_group_timing_diagnostics.csv"
    if not diag.exists():
        return
    trainer = Path(__file__).with_name("train_prosody_group_duration_model.py")
    if not trainer.exists():
        return
    try:
        subprocess.run([sys.executable, str(trainer), str(run_dir)], check=True)
    except Exception as exc:  # diagnostic helper must not break normal summaries
        print(f"Warning: failed to write prosody group duration model diagnostics: {exc}", file=sys.stderr)

def add_derived_metrics(row: dict[str, str]) -> None:
    for source, target in [
        ("RawTextLeadBiasMs", "RawTextLeadBiasMsAbs"),
        ("StreamingLeadBiasMs", "StreamingLeadBiasMsAbs"),
    ]:
        v = parse_float(row.get(source, ""))
        if v is not None:
            row[target] = f"{abs(v):.6g}"


def worst_sort_key(row: dict[str, str]):
    parts = []
    for metric, direction in WORST_CASE_PRIORITY:
        v = parse_float(row.get(metric, ""))
        if v is None:
            v = 0.0
        parts.append(-v if direction == "desc" else v)
    parts.append(row.get("case_id", ""))
    return tuple(parts)



def write_corpus_au23_anchor_frontier_summaries(run_dir: Path) -> None:
    per_case = run_dir / "per_case"
    rows: list[dict[str, str]] = []
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        for r in read_csv_rows(case_dir / "au23_viseme_anchor_frontier.csv"):
            rr = dict(r)
            rr["CaseId"] = case_dir.name
            rows.append(rr)
    if not rows:
        return

    fields = unique_in_order(["CaseId"] + [k for r in rows for k in r.keys() if k != "CaseId"])
    with (run_dir / "corpus_viseme_anchor_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    classes = sorted({r.get("AnchorClass", "") for r in rows if r.get("AnchorClass", "")})
    summary_rows: list[dict[str, str]] = []
    for cls in classes:
        cls_rows = [r for r in rows if r.get("AnchorClass", "") == cls]
        evidence = [r for r in cls_rows if truthy_csv(r.get("HasOfflineEvidence", ""))]
        strong = [r for r in cls_rows if truthy_csv(r.get("StrongForClass", ""))]
        proposed = [r for r in cls_rows if truthy_csv(r.get("ProposedNewHardAnchorClass", ""))]
        current = [r for r in cls_rows if truthy_csv(r.get("CurrentHardAnchorClass", ""))]
        abs_err = [parse_float(r.get("AbsCommittedToTargetErrorMs", "")) for r in strong]
        abs_err = [v for v in abs_err if v is not None and math.isfinite(v)]
        shifts = [abs(parse_float(r.get("RawShiftMs", "")) or 0.0) for r in strong]
        rec = {
            "AnchorClass": cls,
            "EventCount": str(len(cls_rows)),
            "EvidenceCount": str(len(evidence)),
            "EvidenceRate": f"{len(evidence) / len(cls_rows):.9g}" if cls_rows else "",
            "StrongForClassCount": str(len(strong)),
            "StrongForClassRate": f"{len(strong) / len(cls_rows):.9g}" if cls_rows else "",
            "CurrentHardAnchorClassCount": str(len(current)),
            "ProposedNewHardAnchorClassCount": str(len(proposed)),
        }
        if abs_err:
            sorted_abs_err = sorted(abs_err)
            p90_idx = min(len(sorted_abs_err) - 1, int(math.ceil(len(sorted_abs_err) * 0.90)) - 1)
            rec.update({
                "StrongAbsErrorMeanMs": f"{mean(abs_err):.9g}",
                "StrongAbsErrorMedianMs": f"{median(abs_err):.9g}",
                "StrongAbsErrorP90Ms": f"{sorted_abs_err[p90_idx]:.9g}",
            })
        if shifts:
            rec["StrongRawShiftAbsMeanMs"] = f"{mean(shifts):.9g}"
        summary_rows.append(rec)

    summary_fields = [
        "AnchorClass", "EventCount", "EvidenceCount", "EvidenceRate", "StrongForClassCount", "StrongForClassRate",
        "CurrentHardAnchorClassCount", "ProposedNewHardAnchorClassCount", "StrongAbsErrorMeanMs", "StrongAbsErrorMedianMs",
        "StrongAbsErrorP90Ms", "StrongRawShiftAbsMeanMs",
    ]
    with (run_dir / "corpus_viseme_anchor_frontier_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in summary_fields})

    candidate_rows = [r for r in rows if truthy_csv(r.get("ProposedNewHardAnchorClass", ""))]
    candidate_rows.sort(key=lambda r: abs(parse_float(r.get("RawShiftMs", "")) or 0.0), reverse=True)
    candidate_fields = [
        "CaseId", "EventIndex", "TextIslandIndex", "PoseID", "SourceWord", "AnchorClass", "CommittedCenterSec",
        "OfflineEvidenceCenterSec", "TargetVisualCenterSec", "CommittedToTargetErrorMs", "RawShiftMs",
        "OfflineEvidenceConfidence", "RequiredAnchorConfidence", "NearestPrevEventGapMs", "NearestNextEventGapMs",
    ]
    with (run_dir / "corpus_viseme_anchor_frontier_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=candidate_fields)
        w.writeheader()
        for r in candidate_rows[:500]:
            w.writerow({k: r.get(k, "") for k in candidate_fields})


def _au24_anchor_set_definitions() -> dict[str, set[str]]:
    return {
        "baseline_mbp_fv_woo": {"MBP", "FV", "WOO_GLIDE"},
        "plus_rounded": {"MBP", "FV", "WOO_GLIDE", "ROUNDED_VOWEL"},
        "plus_rounded_front": {"MBP", "FV", "WOO_GLIDE", "ROUNDED_VOWEL", "FRONT_VOWEL"},
    }


def _au24_warp_time(x: float, anchors: list[tuple[float, float]]) -> float:
    """Piecewise-linear diagnostic warp from committed time to audio-target time.

    This is a summary-only simulation. It mirrors the AU22 idea without changing
    runtime behavior: start from strong audio anchors, preserve order, interpolate
    between them, and extrapolate the tail conservatively from the last local scale.
    """
    if not anchors:
        return x
    anchors = sorted(anchors, key=lambda pair: pair[0])
    filtered: list[tuple[float, float]] = []
    last_x = -1.0e9
    last_y = -1.0e9
    for ax, ay in anchors:
        if ax <= last_x + 1.0e-5 or ay <= last_y - 0.050:
            continue
        filtered.append((ax, ay))
        last_x = ax
        last_y = ay
    anchors = filtered
    if not anchors:
        return x
    if x <= anchors[0][0]:
        return anchors[0][1] + (x - anchors[0][0])
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            denom = max(1.0e-5, x1 - x0)
            t = max(0.0, min(1.0, (x - x0) / denom))
            return y0 + t * (y1 - y0)
    if len(anchors) >= 2:
        x0, y0 = anchors[-2]
        x1, y1 = anchors[-1]
        scale = (y1 - y0) / max(1.0e-5, x1 - x0)
        scale = max(0.70, min(1.35, scale))
    else:
        scale = 1.0
    return anchors[-1][1] + (x - anchors[-1][0]) * scale


def write_corpus_au24_anchor_set_comparison_summaries(run_dir: Path) -> None:
    """Compare virtual audio-anchor sets without promoting new runtime anchors.

    AU23a answered which viseme classes often have detectable audio evidence.
    AU24a asks whether adding rounded/front vowels as *soft* anchors would improve
    anchor density and a diagnostic piecewise-warp simulation.  This function reads
    existing per-case AU23 frontier CSVs and runtime island alignment CSVs only; it
    does not change liplab runtime behavior.
    """
    per_case = run_dir / "per_case"
    if not per_case.exists():
        return

    set_defs = _au24_anchor_set_definitions()
    comparison_rows: list[dict[str, str]] = []
    simulation_rows: list[dict[str, str]] = []

    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        frontier = read_csv_rows(case_dir / "au23_viseme_anchor_frontier.csv")
        align = read_csv_rows(case_dir / "runtime_island_alignment.csv")
        if not frontier or not align:
            continue
        by_island: dict[int, list[dict[str, str]]] = {}
        for row in frontier:
            idx = parse_int(row.get("TextIslandIndex", ""))
            if idx is None:
                continue
            by_island.setdefault(idx, []).append(row)

        for island_row in align:
            island_idx = parse_int(island_row.get("TextIslandIndex", ""))
            if island_idx is None:
                continue
            island_events = by_island.get(island_idx, [])
            mapped_start = parse_float(island_row.get("MappedAudioStartSec", ""))
            committed_first = parse_float(island_row.get("CommittedFirstCenterSec", ""))
            committed_last = parse_float(island_row.get("CommittedLastCenterSec", ""))
            evidence_first = parse_float(island_row.get("EvidenceFirstCenterSec", ""))
            evidence_last = parse_float(island_row.get("EvidenceLastCenterSec", ""))
            evidence_span = parse_float(island_row.get("EvidenceCenterSpanSec", ""))
            committed_span = parse_float(island_row.get("CommittedCenterSpanSec", ""))
            if mapped_start is None or not math.isfinite(mapped_start):
                mapped_start = committed_first if committed_first is not None and math.isfinite(committed_first) else 0.0
            if committed_first is None or not math.isfinite(committed_first):
                committed_first = mapped_start
            if committed_last is None or not math.isfinite(committed_last):
                committed_last = committed_first
            if evidence_first is None or not math.isfinite(evidence_first):
                evidence_first = committed_first
            if evidence_last is None or not math.isfinite(evidence_last):
                evidence_last = committed_last
            if evidence_span is None or not math.isfinite(evidence_span):
                evidence_span = max(0.0, evidence_last - evidence_first)
            if committed_span is None or not math.isfinite(committed_span):
                committed_span = max(0.0, committed_last - committed_first)

            for set_name, classes in set_defs.items():
                strong_rows: list[dict[str, str]] = []
                for r in island_events:
                    if r.get("AnchorClass", "") not in classes:
                        continue
                    if not truthy_csv(r.get("StrongForClass", "")):
                        continue
                    cx = parse_float(r.get("CommittedCenterSec", ""))
                    ty = parse_float(r.get("TargetVisualCenterSec", ""))
                    if cx is None or ty is None or not math.isfinite(cx) or not math.isfinite(ty):
                        continue
                    strong_rows.append(r)
                strong_rows.sort(key=lambda r: parse_float(r.get("CommittedCenterSec", "")) or 0.0)

                anchors: list[tuple[float, float]] = [(mapped_start, mapped_start)]
                for r in strong_rows:
                    cx = parse_float(r.get("CommittedCenterSec", ""))
                    ty = parse_float(r.get("TargetVisualCenterSec", ""))
                    if cx is not None and ty is not None and math.isfinite(cx) and math.isfinite(ty):
                        anchors.append((cx, ty))

                anchor_committed = sorted([a[0] for a in anchors])
                # Include the committed island tail as a gap endpoint so anchor deserts near the end are visible.
                gap_points = sorted(anchor_committed + [committed_last])
                gaps = [max(0.0, b - a) * 1000.0 for a, b in zip(gap_points, gap_points[1:])]
                max_gap = max(gaps) if gaps else 0.0
                mean_gap = mean(gaps) if gaps else 0.0
                anchor_min = min(anchor_committed) if anchor_committed else committed_first
                anchor_max = max(anchor_committed) if anchor_committed else committed_first
                coverage_den = max(0.001, committed_span)
                coverage_ratio = max(0.0, min(1.0, (anchor_max - anchor_min) / coverage_den))

                eval_rows = []
                for r in island_events:
                    cx = parse_float(r.get("CommittedCenterSec", ""))
                    ty = parse_float(r.get("TargetVisualCenterSec", ""))
                    if cx is None or ty is None or not math.isfinite(cx) or not math.isfinite(ty):
                        continue
                    if not truthy_csv(r.get("HasOfflineEvidence", "")):
                        continue
                    eval_rows.append((r, cx, ty))

                baseline_errors = [abs((cx - ty) * 1000.0) for _r, cx, ty in eval_rows]
                warped_errors = []
                for r, cx, ty in eval_rows:
                    wx = _au24_warp_time(cx, anchors)
                    err = (wx - ty) * 1000.0
                    warped_errors.append(abs(err))
                    simulation_rows.append({
                        "CaseId": case_dir.name,
                        "TextIslandIndex": str(island_idx),
                        "AnchorSet": set_name,
                        "EventIndex": r.get("EventIndex", ""),
                        "PoseID": r.get("PoseID", ""),
                        "SourceWord": r.get("SourceWord", ""),
                        "AnchorClass": r.get("AnchorClass", ""),
                        "StrongForClass": r.get("StrongForClass", ""),
                        "CommittedCenterSec": _fmt(cx),
                        "TargetVisualCenterSec": _fmt(ty),
                        "SimulatedWarpedCenterSec": _fmt(wx),
                        "BaselineAbsErrorMs": _fmt(abs((cx - ty) * 1000.0)),
                        "SimulatedAbsErrorMs": _fmt(abs(err)),
                        "SimulatedSignedErrorMs": _fmt(err),
                    })

                row_out = {
                    "CaseId": case_dir.name,
                    "TextIslandIndex": str(island_idx),
                    "AnchorSet": set_name,
                    "FirstWord": island_row.get("FirstWord", ""),
                    "LastWord": island_row.get("LastWord", ""),
                    "EventCount": island_row.get("EventCount", ""),
                    "StrongAnchorCount": str(len(strong_rows)),
                    "AnchorCountIncludingStart": str(len(anchors)),
                    "CommittedCenterSpanSec": _fmt(committed_span),
                    "EvidenceCenterSpanSec": _fmt(evidence_span),
                    "AnchorCoverageRatio": _fmt(coverage_ratio),
                    "MaxAnchorGapMs": _fmt(max_gap),
                    "MeanAnchorGapMs": _fmt(mean_gap),
                    "EvaluatedEvidenceEventCount": str(len(eval_rows)),
                }
                if baseline_errors:
                    row_out["BaselineMeanAbsErrorMs"] = _fmt(mean(baseline_errors))
                    row_out["BaselineMedianAbsErrorMs"] = _fmt(median(baseline_errors))
                    row_out["BaselineMaxAbsErrorMs"] = _fmt(max(baseline_errors))
                if warped_errors:
                    row_out["SimulatedMeanAbsErrorMs"] = _fmt(mean(warped_errors))
                    row_out["SimulatedMedianAbsErrorMs"] = _fmt(median(warped_errors))
                    row_out["SimulatedMaxAbsErrorMs"] = _fmt(max(warped_errors))
                    row_out["SimulatedMeanImprovementMs"] = _fmt((mean(baseline_errors) if baseline_errors else 0.0) - mean(warped_errors))
                comparison_rows.append(row_out)

    if not comparison_rows:
        return

    comparison_fields = unique_in_order([key for row in comparison_rows for key in row.keys()])
    with (run_dir / "corpus_anchor_set_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=comparison_fields)
        w.writeheader()
        for row in comparison_rows:
            w.writerow({k: row.get(k, "") for k in comparison_fields})

    if simulation_rows:
        simulation_fields = unique_in_order([key for row in simulation_rows for key in row.keys()])
        with (run_dir / "corpus_anchor_warp_simulation.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=simulation_fields)
            w.writeheader()
            for row in simulation_rows:
                w.writerow({k: row.get(k, "") for k in simulation_fields})

    summary_rows: list[dict[str, str]] = []
    for set_name in set_defs.keys():
        subset = [r for r in comparison_rows if r.get("AnchorSet", "") == set_name]
        rec = {"AnchorSet": set_name, "IslandCount": str(len(subset))}
        for metric in [
            "StrongAnchorCount", "AnchorCoverageRatio", "MaxAnchorGapMs", "MeanAnchorGapMs",
            "BaselineMeanAbsErrorMs", "SimulatedMeanAbsErrorMs", "SimulatedMeanImprovementMs",
        ]:
            vals = [parse_float(r.get(metric, "")) for r in subset]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            for k, v in stats_for(vals).items():
                rec[f"{metric}_{k}"] = v
        summary_rows.append(rec)

    summary_fields = unique_in_order([key for row in summary_rows for key in row.keys()])
    with (run_dir / "corpus_anchor_density_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in summary_fields})

    worst_fields = [
        "CaseId", "TextIslandIndex", "AnchorSet", "FirstWord", "LastWord", "EventCount",
        "StrongAnchorCount", "AnchorCoverageRatio", "MaxAnchorGapMs", "MeanAnchorGapMs",
        "BaselineMeanAbsErrorMs", "SimulatedMeanAbsErrorMs", "SimulatedMeanImprovementMs",
    ]
    with (run_dir / "corpus_anchor_gap_worst.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in sorted(comparison_rows, key=lambda r: parse_float(r.get("MaxAnchorGapMs", "")) or 0.0, reverse=True)[:200]:
            w.writerow({k: row.get(k, "") for k in worst_fields})


def write_corpus_au32_landmark_detector_diagnostics(run_dir: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "case_id",
        "AU32StreamingEffectiveTrackPresent",
        "AU32StreamingLandmarkCandidatesPresent",
        "AU32OfflineLandmarkOracleAvailable",
        "AU32LandmarkDetectorScorePresent",
        "AU32PlannedLandmarkMatchesPresent",
        "AU32StreamingCandidateCount",
        "AU32OracleLandmarkCount",
        "AU34OracleSequenceCount",
        "AU34StreamingSequenceCount",
        "AU34SameClassAlignedCount",
        "AU34CompatibleAlignedCount",
        "AU34SubstitutionCount",
        "AU34InsertionCount",
        "AU34DeletionCount",
        "AU34EditDistance",
        "AU34NormalizedEditDistance",
        "AU34SequenceRecall",
        "AU34SequencePrecision",
        "AU34CompatibleSequenceRecall",
        "AU34CompatibleSequencePrecision",
        "AU34TimingMAEMs",
        "AU34TimingBiasMs",
        "AU34TimingP95Ms",
        "AU34MeanAcousticLeadSec",
        "AU34MeanBufferedUsefulLeadSec",
        "AU34AcousticLateCount",
        "AU34UsefulLateCount",
        "AU34AlignmentScore",
        "AU32MatchedCount",
        "AU32Precision",
        "AU32Recall",
        "AU32TimingMAEMs",
        "AU32TimingP95Ms",
        "AU32AcousticLateCount",
        "AU32UsefulLateCount",
        "AU32WrongClassCount",
        "AU32MissedStrongCount",
        "AU32MeanAcousticLeadSec",
        "AU32MeanBufferedUsefulLeadSec",
        "AU32UsefulLeadGE25msCount",
        "AU32UsefulLeadGE50msCount",
        "AU32UsefulLeadGE75msCount",
        "AU32UsefulLeadGE100msCount",
        "AU32PlannedLandmarkRows",
        "AU32PlannedLandmarkMatchedCount",
        "AU32PlannedLandmarkMatchRate",
        "AU32PlannedLandmarkDeltaMAEMs",
        "AU32PlannedLandmarkDeltaMaxMs",
        "AU32StreamingEffectiveEventCount",
        "AU32RuntimeVsStreamingEffectiveChangedEventCount",
        "AU32RuntimeVsStreamingEffectiveMAEMs",
        "AU32RuntimeVsStreamingEffectiveMaxMs",
        "AU32LandmarkValidationErrorCount",
        "AU32LandmarkValidationErrors",
    ]
    path = run_dir / "landmark_detector_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    aggregate_path = run_dir / "landmark_detector_aggregate.csv"
    numeric_fields = [f for f in fields if f != "case_id" and numeric_values(rows, f)]
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "median", "min", "max", "p90", "stddev"])
        w.writeheader()
        for field in numeric_fields:
            rec = {"metric": field}
            rec.update(stats_for(numeric_values(rows, field)))
            w.writerow(rec)

    missing = [row for row in rows if parse_int(row.get("AU32LandmarkValidationErrorCount", "0")) not in (None, 0)]
    validation_path = run_dir / "landmark_detector_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "AU32LandmarkValidationErrorCount", "AU32LandmarkValidationErrors"])
        w.writeheader()
        for row in missing:
            w.writerow({
                "case_id": row.get("case_id", ""),
                "AU32LandmarkValidationErrorCount": row.get("AU32LandmarkValidationErrorCount", ""),
                "AU32LandmarkValidationErrors": row.get("AU32LandmarkValidationErrors", ""),
            })

    au34_fields = [f for f in fields if f.startswith("AU34") or f == "case_id"]
    au34_summary_path = run_dir / "landmark_sequence_summary.csv"
    with au34_summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=au34_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in au34_fields})

    au34_aggregate_path = run_dir / "landmark_sequence_aggregate.csv"
    au34_numeric_fields = [f for f in au34_fields if f != "case_id" and numeric_values(rows, f)]
    with au34_aggregate_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "median", "min", "max", "p90", "stddev"])
        w.writeheader()
        for field in au34_numeric_fields:
            rec = {"metric": field}
            rec.update(stats_for(numeric_values(rows, field)))
            w.writerow(rec)

    detailed_fields = [
        "case_id",
        "AlignmentOp",
        "OracleIndex",
        "StreamingIndex",
        "OracleClass",
        "StreamingClass",
        "OracleTimeSec",
        "StreamingTimeSec",
        "StreamingAvailableAtSec",
        "TimingDeltaMs",
        "AcousticLeadSec",
        "BufferedUsefulLeadSec",
    ]
    detail_path = run_dir / "landmark_sequence_alignment.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=detailed_fields)
        w.writeheader()
        for row in rows:
            case_id = row.get("case_id", "")
            case_dir = run_dir / "per_case" / case_id
            preroll = parse_float(row.get("PrerollSec", ""))
            if preroll is None or not math.isfinite(preroll) or preroll <= 0.0:
                preroll = 0.150
            oracle_seq = _read_landmark_sequence(case_dir / "offline_landmark_oracle.csv")
            streaming_seq = _read_landmark_sequence(case_dir / "streaming_landmark_candidates.csv")
            _, alignment_rows = _align_landmark_sequences(oracle_seq, streaming_seq, preroll)
            for ar in alignment_rows:
                out = {k: ar.get(k, "") for k in detailed_fields}
                out["case_id"] = case_id
                w.writerow(out)



def write_corpus_landmark_preroll_sweep_summaries(run_dir: Path, rows: list[dict[str, str]]) -> None:
    """Aggregate AU35 preroll sweep diagnostics.

    The per-case sweep reruns the landmark detector at several configured preroll
    values. It is diagnostic only: it does not change runtime scoring or the
    committed track. This lets us find the latency/utility knee before exposing a
    new default to Unreal.
    """
    sweep_rows: list[dict[str, str]] = []
    for row in rows:
        case_id = row.get("case_id", "")
        case_dir = run_dir / "per_case" / case_id
        for sr in read_csv_rows(case_dir / "landmark_preroll_sweep.csv"):
            out = dict(sr)
            out["case_id"] = case_id
            sweep_rows.append(out)

    summary_path = run_dir / "landmark_preroll_sweep_summary.csv"
    fields = [
        "case_id",
        "CaseId",
        "PrerollSec",
        "PrerollMs",
        "IsRuntimePreroll",
        "OracleCount",
        "StreamingCandidateCount",
        "MatchedCount",
        "Precision",
        "Recall",
        "TimingMAEMs",
        "TimingP95Ms",
        "AcousticLateCount",
        "UsefulLateCount",
        "MeanAcousticLeadSec",
        "MeanBufferedUsefulLeadSec",
        "UsefulLeadGE25msCount",
        "UsefulLeadGE50msCount",
        "UsefulLeadGE75msCount",
        "UsefulLeadGE100msCount",
        "UsefulLeadGE25msRate",
        "UsefulLeadGE50msRate",
        "UsefulLeadGE75msRate",
        "UsefulLeadGE100msRate",
        "AcousticLateRate",
        "UsefulLateRate",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sweep_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    aggregate_path = run_dir / "landmark_preroll_sweep_aggregate.csv"
    aggregate_fields = [
        "PrerollSec",
        "PrerollMs",
        "CaseCount",
        "OracleCountMean",
        "StreamingCandidateCountMean",
        "MatchedCountMean",
        "PrecisionMean",
        "RecallMean",
        "TimingMAEMsMean",
        "TimingP95MsMean",
        "AcousticLateCountMean",
        "UsefulLateCountMean",
        "MeanAcousticLeadSecMean",
        "MeanBufferedUsefulLeadSecMean",
        "UsefulLeadGE25msRateMean",
        "UsefulLeadGE50msRateMean",
        "UsefulLeadGE75msRateMean",
        "UsefulLeadGE100msRateMean",
        "AcousticLateRateMean",
        "UsefulLateRateMean",
        "UsefulScore25",
        "UsefulScore50",
        "UsefulScore75",
        "UsefulScore100",
    ]
    by_preroll: dict[float, list[dict[str, str]]] = {}
    for row in sweep_rows:
        psec = parse_float(row.get("PrerollSec", ""))
        if psec is None or not math.isfinite(psec):
            continue
        by_preroll.setdefault(round(psec, 6), []).append(row)

    def mean_key(group: list[dict[str, str]], key: str) -> float:
        vals = _float_values(group, key)
        return mean(vals) if vals else 0.0

    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=aggregate_fields)
        w.writeheader()
        for psec in sorted(by_preroll.keys()):
            group = by_preroll[psec]
            recall = mean_key(group, "Recall")
            precision = mean_key(group, "Precision")
            lead25 = mean_key(group, "UsefulLeadGE25msRate")
            lead50 = mean_key(group, "UsefulLeadGE50msRate")
            lead75 = mean_key(group, "UsefulLeadGE75msRate")
            lead100 = mean_key(group, "UsefulLeadGE100msRate")
            acoustic_late_rate = mean_key(group, "AcousticLateRate")
            useful_late_rate = mean_key(group, "UsefulLateRate")
            base = {
                "PrerollSec": f"{psec:.6f}",
                "PrerollMs": f"{psec * 1000.0:.3f}",
                "CaseCount": str(len(group)),
                "OracleCountMean": f"{mean_key(group, 'OracleCount'):.9g}",
                "StreamingCandidateCountMean": f"{mean_key(group, 'StreamingCandidateCount'):.9g}",
                "MatchedCountMean": f"{mean_key(group, 'MatchedCount'):.9g}",
                "PrecisionMean": f"{precision:.9g}",
                "RecallMean": f"{recall:.9g}",
                "TimingMAEMsMean": f"{mean_key(group, 'TimingMAEMs'):.9g}",
                "TimingP95MsMean": f"{mean_key(group, 'TimingP95Ms'):.9g}",
                "AcousticLateCountMean": f"{mean_key(group, 'AcousticLateCount'):.9g}",
                "UsefulLateCountMean": f"{mean_key(group, 'UsefulLateCount'):.9g}",
                "MeanAcousticLeadSecMean": f"{mean_key(group, 'MeanAcousticLeadSec'):.9g}",
                "MeanBufferedUsefulLeadSecMean": f"{mean_key(group, 'MeanBufferedUsefulLeadSec'):.9g}",
                "UsefulLeadGE25msRateMean": f"{lead25:.9g}",
                "UsefulLeadGE50msRateMean": f"{lead50:.9g}",
                "UsefulLeadGE75msRateMean": f"{lead75:.9g}",
                "UsefulLeadGE100msRateMean": f"{lead100:.9g}",
                "AcousticLateRateMean": f"{acoustic_late_rate:.9g}",
                "UsefulLateRateMean": f"{useful_late_rate:.9g}",
                # A deliberately simple utility proxy: useful early recall, penalized
                # by false positives and lateness. The point is to locate the knee,
                # not to become the final objective function.
                "UsefulScore25": f"{recall * precision * lead25 * (1.0 - useful_late_rate):.9g}",
                "UsefulScore50": f"{recall * precision * lead50 * (1.0 - useful_late_rate):.9g}",
                "UsefulScore75": f"{recall * precision * lead75 * (1.0 - useful_late_rate):.9g}",
                "UsefulScore100": f"{recall * precision * lead100 * (1.0 - useful_late_rate):.9g}",
            }
            w.writerow(base)

    validation_path = run_dir / "landmark_preroll_sweep_validation.csv"
    with validation_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "Error"])
        w.writeheader()
        seen = {row.get("case_id", "") for row in sweep_rows}
        for row in rows:
            case_id = row.get("case_id", "")
            if case_id not in seen:
                w.writerow({"case_id": case_id, "Error": "missing_landmark_preroll_sweep_csv"})



def write_corpus_au37_prosody_reflow_feasibility(run_dir: Path, rows: list[dict[str, str]]) -> None:
    """Aggregate AU37 anchor-driven prosody-group reflow feasibility metrics.

    AU37 is diagnostic only. It asks whether the streaming landmark detector
    provides enough causal anchors inside prosody groups to support a future
    monotonic group-reflow nudger. It does not move the runtime/scored track.
    """
    anchor_rows: list[dict[str, str]] = []
    group_rows: list[dict[str, str]] = []
    chain_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []

    for row in rows:
        case_id = row.get("case_id", "")
        case_dir = run_dir / "per_case" / case_id
        expected = [
            ("prosody_group_anchor_matches.csv", anchor_rows),
            ("prosody_group_reflow_feasibility.csv", group_rows),
            ("prosody_group_chain_pressure.csv", chain_rows),
        ]
        for filename, target in expected:
            path = case_dir / filename
            if not path.exists():
                validation_rows.append({"case_id": case_id, "Error": f"missing_{filename}"})
                continue
            for rec in read_csv_rows(path):
                out = dict(rec)
                out["case_id"] = case_id
                target.append(out)

    def write_rows(path: Path, rows_in: list[dict[str, str]], preferred: list[str]) -> None:
        fields = unique_in_order(preferred + [key for r in rows_in for key in r.keys()])
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows_in:
                w.writerow({k: r.get(k, "") for k in fields})

    write_rows(
        run_dir / "prosody_group_anchor_matches.csv",
        anchor_rows,
        [
            "case_id", "IslandIndex", "GroupIndex", "EventIndex", "VisemeId", "SourceWord",
            "PlannedCenterSec", "CommittedCenterSec", "MatchedLandmarkClass", "MatchedAudioSec",
            "BufferedUsefulLeadSec", "DeltaSec", "MatchConfidence", "UsableLead25", "UsableLead50",
        ],
    )
    write_rows(
        run_dir / "prosody_group_reflow_feasibility.csv",
        group_rows,
        [
            "case_id", "IslandIndex", "GroupIndex", "GroupStartSec", "GroupEndSec", "GroupDurationSec",
            "PlannedEventCount", "StrongLandmarkCount", "MatchedAnchorCount", "UsableLead25AnchorCount",
            "UsableLead50AnchorCount", "AnchorCoverageRate", "FirstAnchorDeltaSec", "LastAnchorDeltaSec",
            "AnchorSpanPlannedSec", "AnchorSpanAudioSec", "ImpliedScale", "ImpliedShiftSec",
            "CorrectedStartSec", "CorrectedEndSec", "ReflowMode",
        ],
    )
    write_rows(
        run_dir / "prosody_group_chain_pressure.csv",
        chain_rows,
        [
            "case_id", "IslandIndex", "GroupIndex", "NextGroupIndex", "OriginalTailGapSec", "CorrectedTailGapSec",
            "RequiredNextShiftSec", "WouldOverlap", "WouldCreateDeadAir", "ForwardPressureSec",
        ],
    )

    summary_rows: list[dict[str, str]] = []
    for row in rows:
        case_id = row.get("case_id", "")
        groups = [g for g in group_rows if g.get("case_id", "") == case_id]
        chains = [c for c in chain_rows if c.get("case_id", "") == case_id]
        anchors = [a for a in anchor_rows if a.get("case_id", "") == case_id]

        def count_groups(pred) -> int:
            return sum(1 for g in groups if pred(g))

        total_groups = len(groups)
        strong_groups = count_groups(lambda g: (parse_int(g.get("StrongLandmarkCount", "0")) or 0) > 0)
        any_anchor_groups = count_groups(lambda g: (parse_int(g.get("MatchedAnchorCount", "0")) or 0) > 0)
        two_anchor_groups = count_groups(lambda g: (parse_int(g.get("MatchedAnchorCount", "0")) or 0) >= 2)
        usable25_groups = count_groups(lambda g: (parse_int(g.get("UsableLead25AnchorCount", "0")) or 0) > 0)
        usable50_groups = count_groups(lambda g: (parse_int(g.get("UsableLead50AnchorCount", "0")) or 0) > 0)
        matched_anchor_count = sum((parse_int(g.get("MatchedAnchorCount", "0")) or 0) for g in groups)
        strong_anchor_count = sum((parse_int(g.get("StrongLandmarkCount", "0")) or 0) for g in groups)

        shifts_ms = [abs(v) * 1000.0 for v in _float_values(groups, "ImpliedShiftSec")]
        scales = [v for v in _float_values(groups, "ImpliedScale") if math.isfinite(v) and v > 0.0]
        pressure_ms = [abs(v) * 1000.0 for v in _float_values(chains, "ForwardPressureSec")]
        required_shift_ms = [v * 1000.0 for v in _float_values(chains, "RequiredNextShiftSec")]
        overlap_count = sum(1 for c in chains if truthy_csv(c.get("WouldOverlap", "")))
        dead_air_count = sum(1 for c in chains if truthy_csv(c.get("WouldCreateDeadAir", "")))

        rec = {
            "case_id": case_id,
            "AU37ProsodyGroupsTotal": str(total_groups),
            "AU37GroupsWithStrongLandmarks": str(strong_groups),
            "AU37GroupsWithAnyAnchor": str(any_anchor_groups),
            "AU37GroupsWithTwoPlusAnchors": str(two_anchor_groups),
            "AU37GroupsWithUsableLead25Anchor": str(usable25_groups),
            "AU37GroupsWithUsableLead50Anchor": str(usable50_groups),
            "AU37StrongAnchorCount": str(strong_anchor_count),
            "AU37MatchedAnchorCount": str(matched_anchor_count),
            "AU37AnchorCoverageRate": f"{matched_anchor_count / strong_anchor_count:.9g}" if strong_anchor_count else "",
            "AU37GroupAnyAnchorRate": f"{any_anchor_groups / total_groups:.9g}" if total_groups else "",
            "AU37GroupTwoPlusAnchorRate": f"{two_anchor_groups / total_groups:.9g}" if total_groups else "",
            "AU37GroupUsableLead25Rate": f"{usable25_groups / total_groups:.9g}" if total_groups else "",
            "AU37GroupUsableLead50Rate": f"{usable50_groups / total_groups:.9g}" if total_groups else "",
            "AU37SingleAnchorShiftGroups": str(count_groups(lambda g: g.get("ReflowMode", "") == "single_anchor_shift")),
            "AU37TwoAnchorAffineGroups": str(count_groups(lambda g: g.get("ReflowMode", "") == "two_anchor_affine")),
            "AU37MultiAnchorPiecewiseGroups": str(count_groups(lambda g: g.get("ReflowMode", "") == "multi_anchor_piecewise")),
            "AU37MedianImpliedShiftMs": f"{_simple_percentile(shifts_ms, 50):.9g}" if shifts_ms else "",
            "AU37P95ImpliedShiftMs": f"{_simple_percentile(shifts_ms, 95):.9g}" if shifts_ms else "",
            "AU37MedianImpliedScale": f"{_simple_percentile(scales, 50):.9g}" if scales else "",
            "AU37P95ImpliedScale": f"{_simple_percentile(scales, 95):.9g}" if scales else "",
            "AU37ChainPairCount": str(len(chains)),
            "AU37MeanForwardPressureMs": f"{mean(pressure_ms):.9g}" if pressure_ms else "",
            "AU37P95ForwardPressureMs": f"{_simple_percentile(pressure_ms, 95):.9g}" if pressure_ms else "",
            "AU37MeanRequiredNextShiftMs": f"{mean(required_shift_ms):.9g}" if required_shift_ms else "",
            "AU37OverlapRiskCount": str(overlap_count),
            "AU37DeadAirRiskCount": str(dead_air_count),
            "AU37OverlapRiskRate": f"{overlap_count / len(chains):.9g}" if chains else "",
            "AU37DeadAirRiskRate": f"{dead_air_count / len(chains):.9g}" if chains else "",
        }
        summary_rows.append(rec)

    summary_fields = unique_in_order([key for r in summary_rows for key in r.keys()])
    with (run_dir / "prosody_group_reflow_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in summary_fields})

    numeric_fields = [f for f in summary_fields if f != "case_id" and numeric_values(summary_rows, f)]
    with (run_dir / "prosody_group_reflow_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "median", "min", "max", "p90", "stddev"])
        w.writeheader()
        for field in numeric_fields:
            rec = {"metric": field}
            rec.update(stats_for(numeric_values(summary_rows, field)))
            w.writerow(rec)

    with (run_dir / "prosody_group_reflow_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "Error"])
        w.writeheader()
        for r in validation_rows:
            w.writerow(r)


def write_corpus_au41_no_fragment_runtime(run_dir: Path, rows: list[dict[str, str]]) -> None:
    """Validate AU41's simplified runtime boundary.

    AU41 intentionally removes speech-fragment ownership from runtime timing.
    Fragments may exist inside the VAD, but the committed track should report
    onset-only audio timing and zero fragment-timed events.
    """
    summary_fields = [
        "case_id",
        "AU41DiagnosticsPresent",
        "AU41RuntimeNoFragments",
        "AU41EventCount",
        "AU41OnsetOnlyEventCount",
        "AU41FragmentTimedEventCount",
        "AU41MultiFragmentModeEventCount",
        "AU41TextIslandCount",
        "AU41ReferencedOnsetCount",
        "AU41DetectorSpeechIslandCount",
        "AU41PrerollSec",
    ]
    with (run_dir / "au41_no_fragment_runtime_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in rows:
            if any(k in row for k in summary_fields if k != "case_id"):
                w.writerow({k: row.get(k, "") for k in summary_fields})

    numeric = {field: numeric_values(rows, field) for field in summary_fields if field != "case_id"}
    with (run_dir / "au41_no_fragment_runtime_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Metric", "Count", "Mean", "Median", "Min", "Max", "P90", "StdDev"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for field, vals in numeric.items():
            if not vals:
                continue
            st = stats_for(vals)
            w.writerow({
                "Metric": field,
                "Count": st.get("count", ""),
                "Mean": st.get("mean", ""),
                "Median": st.get("median", ""),
                "Min": st.get("min", ""),
                "Max": st.get("max", ""),
                "P90": st.get("p90", ""),
                "StdDev": st.get("stddev", ""),
            })

    validation_rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    for row in rows:
        case_id = row.get("case_id", "")
        if not case_id:
            continue
        case_dir = per_case / case_id
        if not (case_dir / "au41_no_fragment_runtime_metrics.csv").exists():
            validation_rows.append({"case_id": case_id, "Error": "missing_au41_no_fragment_runtime_metrics_csv"})
        if (case_dir / "prosody_group_fragment_assignment.csv").exists():
            validation_rows.append({"case_id": case_id, "Error": "au41_should_not_write_fragment_assignment_csv"})
        if row.get("AU41DiagnosticsPresent", "") != "1":
            validation_rows.append({"case_id": case_id, "Error": "au41_diagnostics_not_present"})
        if row.get("AU41RuntimeNoFragments", "") != "1":
            validation_rows.append({"case_id": case_id, "Error": "au41_runtime_no_fragments_not_set"})
        if parse_float(row.get("AU41FragmentTimedEventCount", "")) not in (0.0,):
            validation_rows.append({"case_id": case_id, "Error": "au41_fragment_timed_events_nonzero"})
        if parse_float(row.get("AU41MultiFragmentModeEventCount", "")) not in (0.0,):
            validation_rows.append({"case_id": case_id, "Error": "au41_multi_fragment_mode_events_nonzero"})
    with (run_dir / "au41_no_fragment_runtime_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "Error"])
        w.writeheader()
        for vr in validation_rows:
            w.writerow(vr)

def write_corpus_au39_runtime_topology(run_dir: Path, rows: list[dict[str, str]]) -> None:
    """Aggregate AU39 runtime continuity/topology metrics for the actual scored track.

    AU39 deliberately disables AU38/AU38b runtime reflow. These metrics quantify
    whether the streaming-effective track has dense clusters, large empty gaps,
    or ordering problems that could explain perceived animation stutter.
    """
    summary_fields = [
        "case_id",
        "AU39EventCount",
        "AU39GapCount",
        "AU39MinEventGapMs",
        "AU39P50EventGapMs",
        "AU39P90EventGapMs",
        "AU39MaxEventGapMs",
        "AU39MeanAbsEventGapMs",
        "AU39DenseGapCount",
        "AU39LargeGapCount",
        "AU39NegativeGapCount",
        "AU39MinHealthyGapMs",
        "AU39LargeGapThresholdMs",
        "AU39RuntimeReflowDisabled",
    ]
    with (run_dir / "au39_runtime_topology_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in rows:
            if any(k in row for k in summary_fields if k != "case_id"):
                w.writerow({k: row.get(k, "") for k in summary_fields})

    numeric = {field: numeric_values(rows, field) for field in summary_fields if field != "case_id"}
    with (run_dir / "au39_runtime_topology_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["Metric", "Count", "Mean", "Median", "Min", "Max", "P90", "StdDev"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for field, vals in numeric.items():
            if not vals:
                continue
            st = stats_for(vals)
            w.writerow({
                "Metric": field,
                "Count": st.get("count", ""),
                "Mean": st.get("mean", ""),
                "Median": st.get("median", ""),
                "Min": st.get("min", ""),
                "Max": st.get("max", ""),
                "P90": st.get("p90", ""),
                "StdDev": st.get("stddev", ""),
            })

    validation_rows: list[dict[str, str]] = []
    per_case = run_dir / "per_case"
    for row in rows:
        case_id = row.get("case_id", "")
        if not case_id:
            continue
        case_dir = per_case / case_id
        if not (case_dir / "au39_runtime_topology_metrics.csv").exists():
            validation_rows.append({"case_id": case_id, "Error": "missing_au39_runtime_topology_metrics_csv"})
        if (case_dir / "au38_reflow_runtime_commit_events.csv").exists():
            validation_rows.append({"case_id": case_id, "Error": "unsafe_au38_reflow_runtime_track_present"})
    with (run_dir / "au39_runtime_topology_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "Error"])
        w.writeheader()
        for vr in validation_rows:
            w.writerow(vr)


def run_punctuation_resume_oracle_if_requested(run_dir: Path, inputs_dir: Path) -> bool:
    """Run the E-series punctuation/audio-pause oracle as part of summary generation.

    The oracle is intentionally offline-only.  It writes into
    <run>/punctuation_oracle so normal summary zips can include the E02/E03
    evidence without changing runtime behavior.
    """
    script = Path(__file__).resolve().parent / "analyze_punctuation_resume_oracle.py"
    if not script.exists():
        print(f"Warning: missing punctuation oracle script: {script}")
        return False
    if not inputs_dir.exists():
        print(f"Warning: skipping punctuation oracle; missing inputs directory: {inputs_dir}")
        return False
    subprocess.run([
        sys.executable,
        str(script),
        "--run",
        str(run_dir),
        "--inputs",
        str(inputs_dir),
    ], check=True)
    return True

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory, e.g. experiments/runs/baseline_v26")
    ap.add_argument("--out", default=None, help="Optional summary.csv output path")
    ap.add_argument("--worst-count", type=int, default=25)
    ap.add_argument("--legacy-reports", action="store_true", help="Also write deprecated historical corpus reports. By default summarize_run writes only the compact AU42 report set.")
    ap.add_argument("--current-drilldowns", action="store_true", help="Also write current but verbose AU39/AU41/landmark/island drill-down CSVs. By default only dashboard outputs are written.")
    ap.add_argument("--punctuation-oracle", action="store_true", help="Also run the offline E-series punctuation/audio-pause oracle and write <run>/punctuation_oracle outputs.")
    ap.add_argument("--inputs", default="input", help="Corpus root containing transcripts/ and wav/ for --punctuation-oracle. Defaults to ./input for c:\\git\\liplab.")
    args = ap.parse_args()

    run_dir = Path(args.run)
    per_case = run_dir / "per_case"
    if not per_case.exists():
        raise SystemExit(f"Missing per_case directory: {per_case}")

    rows: list[dict[str, str]] = []
    for case_dir in sorted(p for p in per_case.iterdir() if p.is_dir()):
        scorecard = case_dir / "scorecard.csv"
        if not scorecard.exists():
            continue
        row = read_scorecard(scorecard)
        if not row:
            continue
        row = normalize_row(row, case_dir.name)
        add_derived_metrics(row)
        augment_with_per_case_csv_metrics(case_dir, row)
        augment_with_au32_landmark_detector_diagnostics(case_dir, row)
        # Deprecated: per-case gap attribution belongs to older fragment/assignment investigations.
        # Keep the function available for --legacy-reports, but do not spam every summary run.
        if args.legacy_reports:
            write_gap_attribution_csv(case_dir, row)
        write_flat_metrics(case_dir, row)
        rows.append(row)

    if not rows:
        raise SystemExit(f"No scorecards found under {per_case}")

    rows.sort(key=lambda row: row["case_id"])

    summary_columns = ["case_id"] + sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key != "case_id"
        }
    )

    summary_path = Path(args.out) if args.out else run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in summary_columns})

    metric_stats_path = run_dir / "metric_stats.csv"
    numeric_keys = [
        key
        for key in summary_columns
        if key != "case_id" and numeric_values(rows, key)
    ]

    with metric_stats_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "count", "mean", "median", "min", "max", "p90", "stddev"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(unique_in_order(numeric_keys)):
            rec = {"metric": key}
            rec.update(stats_for(numeric_values(rows, key)))
            w.writerow(rec)

    worst_path = run_dir / "worst_cases.csv"
    worst_rows = sorted(rows, key=worst_sort_key)[: args.worst_count]
    worst_fields = unique_in_order(
        ["case_id"]
        + [metric for metric, _ in WORST_CASE_PRIORITY if metric in summary_columns]
        + [
            extra
            for extra in [
                "RawTextLeadBiasMs",
                "RawTextTimingMAEMs",
                "P90AbsRawTextTimingErrorMs",
                "StreamingLeadBiasMs",
                "StreamingTimingMAEMs",
                "PeakTimingMAEMs",
                "PeakWindowCenterMAEMs",
                "PeakWindowWidthMeanMs",
                "ExplosiveWeakPeakRate",
                "ExplosivePeakLeadMeanMs",
                "ExplosivePeakLeadErrorMAEMs",
                "VowelWeakPeakRate",
                "VowelPeakCenterMAEMs",
                "CommitLeadMeanSec",
                "SpeechOnlyScaleMax",
                "ScaleCapHitRate",
            ]
            if extra in summary_columns
        ]
    )

    with worst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worst_fields)
        w.writeheader()
        for row in worst_rows:
            w.writerow({k: row.get(k, "") for k in worst_fields})

    # Compact AU42 report set. Keep default output small and focused:
    #   - summary.csv / metric_stats.csv for sortable corpus-wide numbers
    #   - worst_cases.csv for triage
    #   - corpus_summary.csv / corpus_phase_summary.txt for the human dashboard
    # Verbose current drill-down CSVs are still available via --current-drilldowns.
    write_corpus_summary(run_dir, rows)
    write_corpus_phase_summary(run_dir, rows)

    punctuation_oracle_ran = False
    if args.punctuation_oracle:
        punctuation_oracle_ran = run_punctuation_resume_oracle_if_requested(run_dir, Path(args.inputs))

    if args.current_drilldowns:
        write_corpus_runtime_island_alignment_summaries(run_dir)
        write_corpus_start_detection_summaries(run_dir)
        write_corpus_au32_landmark_detector_diagnostics(run_dir, rows)
        write_corpus_landmark_preroll_sweep_summaries(run_dir, rows)
        write_corpus_au41_no_fragment_runtime(run_dir, rows)
        write_corpus_au39_runtime_topology(run_dir, rows)

    # Historical research reports. These are intentionally off by default because
    # most are relics of retired fragment assignment, offline retime, reflow, gap
    # attribution, and detector-policy investigations. Re-enable only when
    # intentionally revisiting one of those branches.
    if args.legacy_reports:
        write_corpus_island_boundary_summaries(run_dir)
        write_corpus_text_audio_island_assignment_summaries(run_dir)
        write_corpus_streaming_island_assignment_sim_summaries(run_dir)
        write_corpus_assignment_confusion_summaries(run_dir)
        write_corpus_assignment_decision_summaries(run_dir)
        write_corpus_detector_evidence_audit_summaries(run_dir)
        write_corpus_speech_evidence_region_summaries(run_dir)
        write_corpus_start_late_forensics_summaries(run_dir)
        write_corpus_current_policy_limitation_summaries(run_dir)
        write_corpus_no_causal_detector_span_forensics(run_dir)
        write_corpus_ownership_retention_forensics(run_dir)
        write_corpus_launch_attribution_forensics(run_dir)
        write_corpus_speech_attribution_summaries(run_dir, rows)
        write_corpus_duration_model_fit_summaries(run_dir, rows)
        write_corpus_au23_anchor_frontier_summaries(run_dir)
        write_corpus_au24_anchor_set_comparison_summaries(run_dir)
        write_corpus_gap_attribution_summaries(run_dir)
        write_corpus_au37_prosody_reflow_feasibility(run_dir, rows)

    print(f"Cases: {len(rows)}")
    current_outputs = [
        summary_path,
        metric_stats_path,
        worst_path,
        run_dir / "corpus_summary.csv",
        run_dir / "corpus_phase_summary.txt",
    ]
    if punctuation_oracle_ran:
        current_outputs.extend([
            run_dir / "punctuation_oracle" / "audio_pause_inventory.csv",
            run_dir / "punctuation_oracle" / "punctuation_audio_pause_correlation_summary.csv",
            run_dir / "punctuation_oracle" / "punctuation_resume_oracle_boundaries.csv",
            run_dir / "punctuation_oracle" / "punctuation_resume_oracle_e1_summary.csv",
            run_dir / "punctuation_oracle" / "punctuation_resume_oracle_e1_by_case.csv",
        ])
    if args.current_drilldowns:
        current_outputs.extend([
            run_dir / "corpus_runtime_island_alignment.csv",
            run_dir / "corpus_runtime_island_alignment_stats.csv",
            run_dir / "corpus_runtime_island_alignment_worst.csv",
            run_dir / "corpus_start_detection.csv",
            run_dir / "corpus_start_detection_summary.csv",
            run_dir / "corpus_start_detection_stats.csv",
            run_dir / "corpus_start_detection_worst.csv",
            run_dir / "corpus_start_detection_candidates.csv",
            run_dir / "landmark_detector_summary.csv",
            run_dir / "landmark_detector_aggregate.csv",
            run_dir / "landmark_detector_validation.csv",
            run_dir / "landmark_preroll_sweep_summary.csv",
            run_dir / "landmark_preroll_sweep_aggregate.csv",
            run_dir / "landmark_preroll_sweep_validation.csv",
            run_dir / "au41_no_fragment_runtime_summary.csv",
            run_dir / "au41_no_fragment_runtime_aggregate.csv",
            run_dir / "au41_no_fragment_runtime_validation.csv",
            run_dir / "au39_runtime_topology_summary.csv",
            run_dir / "au39_runtime_topology_aggregate.csv",
            run_dir / "au39_runtime_topology_validation.csv",
        ])
    for path in current_outputs:
        if path.exists():
            print(f"Wrote: {path}")
    if args.current_drilldowns:
        print("Current drill-down reports enabled; AU39/AU41/landmark/island detail CSVs were written.")
    else:
        print("Current drill-down reports disabled; use --current-drilldowns for detailed active-architecture CSVs.")
    if args.legacy_reports:
        print("Legacy reports enabled; additional historical CSVs may also have been written.")
    else:
        print("Legacy reports disabled; use --legacy-reports to regenerate retired research CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
