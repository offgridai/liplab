import csv
import json
import pathlib
import sys
from collections import Counter


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float_any(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row.get(key, "") != "":
            return _as_float(row, key, default)
    return default


def _as_int_any(row: dict, *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in row and row.get(key, "") != "":
            return _as_int(row, key, default)
    return default


def _read_csv_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_length_ms(rows: list[dict], predicate) -> float:
    best = 0.0
    current = 0.0
    for row in rows:
        duration = max(0.0, _as_float(row, "end") - _as_float(row, "start")) * 1000.0
        if predicate(row):
            current += duration
            best = max(best, current)
        else:
            current = 0.0
    return best


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_occupancy_summary(case_dir: pathlib.Path) -> dict:
    rows = _read_csv_rows(case_dir / "occupancy_frames.csv")
    if not rows:
        return {}

    evidence = [_as_float(row, "evidence") for row in rows]
    rms_norm = [_as_float(row, "rms_norm") for row in rows]
    decisions = Counter(row.get("decision", "") or "none" for row in rows)
    if "none" in decisions and decisions["none"] == 0:
        del decisions["none"]

    occupied = sum(1 for row in rows if _as_int(row, "in_speech_after") != 0)
    below_close = [
        row for row in rows
        if _as_float(row, "evidence") < _as_float(row, "close_threshold")
    ]

    return {
        "frames": len(rows),
        "occupied": occupied,
        "occupied_pct": (100.0 * occupied / len(rows)) if rows else 0.0,
        "started": sum(1 for row in rows if _as_int_any(row, "frame_started_speech_region", "frame_started") != 0),
        "closed": sum(1 for row in rows if _as_int_any(row, "frame_closed_speech_region", "frame_closed") != 0),
        "bridged": sum(1 for row in rows if _as_int_any(row, "frame_bridged_speech_region", "frame_bridged") != 0),
        "open_candidate": sum(1 for row in rows if _as_int(row, "open_candidate") != 0),
        "keep_open": sum(1 for row in rows if _as_int(row, "keep_open") != 0),
        "low_evidence": sum(1 for row in rows if _as_int(row, "low_evidence") != 0),
        "strong_quiet": sum(1 for row in rows if _as_int(row, "strong_quiet") != 0),
        "endpoint_active": sum(1 for row in rows if _as_int(row, "endpoint_active") != 0),
        "below_close": len(below_close),
        "evidence_min": min(evidence) if evidence else 0.0,
        "evidence_mean": _mean(evidence),
        "evidence_max": max(evidence) if evidence else 0.0,
        "rms_mean": _mean(rms_norm),
        "longest_unoccupied_ms": _run_length_ms(rows, lambda r: _as_int(r, "in_speech_after") == 0),
        "longest_low_evidence_ms": _run_length_ms(rows, lambda r: _as_int(r, "low_evidence") != 0),
        "longest_below_close_ms": _run_length_ms(rows, lambda r: _as_float(r, "evidence") < _as_float(r, "close_threshold")),
        "decisions": dict(sorted(decisions.items())),
    }


def load_streaming_detector_summary(case_dir: pathlib.Path) -> dict:
    path = case_dir / "streaming_detector_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}




def load_close_suppressions(case_dir: pathlib.Path) -> dict:
    rows = _read_csv_rows(case_dir / "occupancy_frames.csv")
    if not rows:
        return {}

    runs = []
    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]
        if _as_float(row, "evidence") >= _as_float(row, "close_threshold"):
            i += 1
            continue
        start_i = i
        while i < n and _as_float(rows[i], "evidence") < _as_float(rows[i], "close_threshold"):
            i += 1
        end_i = i - 1
        run_rows = rows[start_i:i]
        next_row = rows[i] if i < n else None
        prev_row = rows[start_i - 1] if start_i > 0 else None
        duration_ms = sum(max(0.0, _as_float(r, "end") - _as_float(r, "start")) * 1000.0 for r in run_rows)
        successful = any(_as_int_any(r, "frame_closed_speech_region", "frame_closed") != 0 or (r.get("decision", "") == "closed_frame") for r in run_rows)
        if not successful and next_row is not None:
            # A close is often recorded on the first frame after the strictly-below-close run.
            # Count that as successful if the detector closed immediately on exit.
            successful = (_as_int_any(next_row, "frame_closed_speech_region", "frame_closed") != 0 or next_row.get("decision", "") == "closed_frame")
        terminated_by = "end_of_file"
        if successful:
            terminated_by = "successful_close"
        elif next_row is not None:
            decision = next_row.get("decision", "") or "none"
            if _as_int_any(next_row, "frame_started_speech_region", "frame_started") != 0 or decision == "new_island_open":
                terminated_by = "reopen"
            elif decision == "candidate_decay":
                terminated_by = "candidate_decay"
            elif _as_int(next_row, "endpoint_active") != 0:
                terminated_by = "endpoint_hangover"
            elif _as_float(next_row, "evidence") >= _as_float(next_row, "close_threshold"):
                terminated_by = "high_evidence"
            elif _as_int(next_row, "keep_open") != 0:
                terminated_by = "keep_open"
            else:
                terminated_by = decision

        runs.append({
            "run_index": len(runs),
            "start_ms": _as_float(run_rows[0], "start") * 1000.0,
            "end_ms": _as_float(run_rows[-1], "end") * 1000.0,
            "duration_ms": duration_ms,
            "frames": len(run_rows),
            "successful": 1 if successful else 0,
            "terminated_by": terminated_by,
            "evidence_min": min(_as_float(r, "evidence") for r in run_rows),
            "evidence_mean": _mean([_as_float(r, "evidence") for r in run_rows]),
            "close_threshold_mean": _mean([_as_float(r, "close_threshold") for r in run_rows]),
            "endpoint_frames": sum(1 for r in run_rows if _as_int(r, "endpoint_active") != 0),
            "low_frames": sum(1 for r in run_rows if _as_int(r, "low_evidence") != 0),
            "keep_open_frames": sum(1 for r in run_rows if _as_int(r, "keep_open") != 0),
            "closed_frames": sum(1 for r in run_rows if _as_int_any(r, "frame_closed_speech_region", "frame_closed") != 0),
            "prev_evidence": _as_float(prev_row, "evidence") if prev_row is not None else 0.0,
            "next_evidence": _as_float(next_row, "evidence") if next_row is not None else 0.0,
            "next_decision": next_row.get("decision", "") if next_row is not None else "",
            "next_endpoint": _as_int(next_row, "endpoint_active") if next_row is not None else 0,
            "next_keep_open": _as_int(next_row, "keep_open") if next_row is not None else 0,
        })

    failed = [r for r in runs if not r["successful"]]
    successful = [r for r in runs if r["successful"]]
    broken = Counter(r["terminated_by"] for r in failed)
    return {
        "runs": runs,
        "below_close_runs": len(runs),
        "successful_closes": len(successful),
        "failed_runs": len(failed),
        "broken_by": dict(sorted(broken.items())),
        "longest_failed_ms": max((r["duration_ms"] for r in failed), default=0.0),
        "mean_failed_ms": _mean([r["duration_ms"] for r in failed]),
        "longest_successful_ms": max((r["duration_ms"] for r in successful), default=0.0),
        "mean_successful_ms": _mean([r["duration_ms"] for r in successful]),
    }


def _nearest_frame(rows: list[dict], time_seconds: float, prefer_key: str | None = None) -> dict | None:
    if not rows:
        return None
    best = None
    best_dist = float("inf")
    for row in rows:
        center = _as_float(row, "center", 0.5 * (_as_float(row, "start") + _as_float(row, "end")))
        dist = abs(center - time_seconds)
        # Prefer explicit transition frames when they are close enough to be relevant.
        if prefer_key and _as_int(row, prefer_key) != 0 and dist <= 0.120:
            dist -= 0.060
        if dist < best_dist:
            best_dist = dist
            best = row
    return best


def _derive_islands_from_occupancy(rows: list[dict]) -> list[dict]:
    islands = []
    current_start = None
    current_open_row = None
    prev_in = 0
    for row in rows:
        in_after = _as_int(row, "in_speech_after") != 0
        if in_after and not prev_in:
            current_start = _as_float(row, "start")
            current_open_row = row
        if prev_in and not in_after and current_start is not None:
            islands.append({
                "start": current_start,
                "end": _as_float(row, "end"),
                "open_row": current_open_row,
                "close_row": row,
                "source": "occupancy_derived",
            })
            current_start = None
            current_open_row = None
        prev_in = 1 if in_after else 0
    if current_start is not None and rows:
        islands.append({
            "start": current_start,
            "end": _as_float(rows[-1], "end"),
            "open_row": current_open_row,
            "close_row": rows[-1],
            "source": "occupancy_derived_open_at_eof",
        })
    return islands


def load_island_lifecycle(case_dir: pathlib.Path) -> dict:
    speech_rows = _read_csv_rows(case_dir / "speech_regions.csv")
    frames = _read_csv_rows(case_dir / "occupancy_frames.csv")

    islands = []
    source = "speech_regions"
    for idx, row in enumerate(speech_rows):
        start = _as_float(row, "start")
        end = _as_float(row, "end")
        if end <= start:
            continue
        open_row = _nearest_frame(frames, start, "frame_started_speech_region")
        close_row = _nearest_frame(frames, end, "frame_closed_speech_region")
        islands.append({
            "index": len(islands),
            "start": start,
            "end": end,
            "open_row": open_row,
            "close_row": close_row,
            "source": source,
        })

    if not islands and frames:
        source = "occupancy_derived"
        for item in _derive_islands_from_occupancy(frames):
            item["index"] = len(islands)
            islands.append(item)

    if not islands:
        return {}

    traced = []
    for i, island in enumerate(islands):
        start = float(island["start"])
        end = float(island["end"])
        open_row = island.get("open_row") or {}
        close_row = island.get("close_row") or {}
        next_start = float(islands[i + 1]["start"]) if i + 1 < len(islands) else 0.0
        gap_after = max(0.0, next_start - end) if i + 1 < len(islands) else 0.0
        traced.append({
            "index": i,
            "start_ms": start * 1000.0,
            "end_ms": end * 1000.0,
            "duration_ms": max(0.0, end - start) * 1000.0,
            "gap_after_ms": gap_after * 1000.0,
            "source": island.get("source", source),
            "open_decision": open_row.get("decision", "") if isinstance(open_row, dict) else "",
            "close_decision": close_row.get("decision", "") if isinstance(close_row, dict) else "",
            "open_frame_started": _as_int_any(open_row, "frame_started_speech_region", "frame_started") if isinstance(open_row, dict) else 0,
            "close_frame_closed": _as_int_any(close_row, "frame_closed_speech_region", "frame_closed") if isinstance(close_row, dict) else 0,
            "open_evidence": _as_float(open_row, "evidence") if isinstance(open_row, dict) else 0.0,
            "close_evidence": _as_float(close_row, "evidence") if isinstance(close_row, dict) else 0.0,
            "open_time_error_ms": (_as_float(open_row, "center") - start) * 1000.0 if isinstance(open_row, dict) and open_row else 0.0,
            "close_time_error_ms": (_as_float(close_row, "center") - end) * 1000.0 if isinstance(close_row, dict) and close_row else 0.0,
        })

    durations = [x["duration_ms"] for x in traced]
    gaps = [x["gap_after_ms"] for x in traced[:-1]]
    return {
        "source": source,
        "islands": traced,
        "count": len(traced),
        "total_ms": sum(durations),
        "mean_duration_ms": _mean(durations),
        "first_open_ms": traced[0]["start_ms"],
        "last_close_ms": traced[-1]["end_ms"],
        "mean_gap_ms": _mean(gaps),
        "max_gap_ms": max(gaps) if gaps else 0.0,
    }



def load_attach_traces(case_dir: pathlib.Path) -> list[dict]:
    """Summarize the single decision that matters for region count:
    when a post-close onset appears, did it attach to the previous island or split?

    The detector writes gap_candidates.csv for every candidate inter-island gap. A
    bridged candidate means the new onset was attached to the previous emitted
    speech region. A non-bridged candidate means it became a new emitted region.
    """
    gaps = _read_csv_rows(case_dir / "gap_candidates.csv")
    speech_rows = _read_csv_rows(case_dir / "speech_regions.csv")
    frames = _read_csv_rows(case_dir / "occupancy_frames.csv")
    if not gaps:
        return []

    regions_by_index = {}
    for idx, row in enumerate(speech_rows):
        start = _as_float(row, "start")
        end = _as_float(row, "end")
        if end > start:
            region_index = _as_int(row, "index", idx)
            regions_by_index[region_index] = {"index": region_index, "start": start, "end": end}

    traces = []
    for gap in gaps:
        gap_start = _as_float(gap, "gap_start")
        gap_end = _as_float(gap, "gap_end")
        gap_duration = _as_float(gap, "gap_duration", max(0.0, gap_end - gap_start))
        bridged = _as_int(gap, "bridged") != 0

        prev_region = regions_by_index.get(_as_int_any(gap, "prev_speech_region_index", "prev_island_index", default=-1))
        next_region = None if bridged else regions_by_index.get(_as_int_any(gap, "next_speech_region_index", "next_island_index", default=-1))

        onset_row = _nearest_frame(frames, gap_end, "frame_started_speech_region")
        quiet_row = _nearest_frame(frames, gap_start, "frame_closed_speech_region")
        emitted_gap = 0.0
        if bridged:
            emitted_gap = 0.0
        elif prev_region is not None and next_region is not None:
            emitted_gap = max(0.0, next_region["start"] - prev_region["end"])

        # What the region builder effectively did with this onset.
        attach_decision = "attach" if bridged else "split"
        decision_class = gap.get("decision_class", "")
        if not decision_class:
            decision_class = "attached" if bridged else "split"

        traces.append({
            "gap_index": _as_int(gap, "gap_index"),
            "attach": 1 if bridged else 0,
            "attach_decision": attach_decision,
            "decision_class": decision_class,
            "gap_start_ms": gap_start * 1000.0,
            "gap_end_ms": gap_end * 1000.0,
            "onset_ms": gap_end * 1000.0,
            "acoustic_gap_ms": gap_duration * 1000.0,
            "emitted_gap_ms": emitted_gap * 1000.0,
            "prev_region": prev_region["index"] if prev_region is not None else -1,
            "prev_region_end_ms": prev_region["end"] * 1000.0 if prev_region is not None else 0.0,
            "next_region": next_region["index"] if next_region is not None else -1,
            "next_region_start_ms": next_region["start"] * 1000.0 if next_region is not None else 0.0,
            "resolved_region": prev_region["index"] if bridged and prev_region is not None else -1,
            "close_reason": gap.get("close_reason", ""),
            "quiet_evidence": _as_float(gap, "quiet_evidence"),
            "quiet_rms_norm": _as_float(gap, "quiet_rms_norm"),
            "reopen_evidence": _as_float(gap, "reopen_evidence"),
            "reopen_flux": _as_float(gap, "reopen_flux"),
            "strong_quiet_close": _as_int(gap, "strong_quiet_close"),
            "strong_onset_reopen": _as_int(gap, "strong_onset_reopen"),
            "onset_decision": onset_row.get("decision", "") if isinstance(onset_row, dict) and onset_row else "",
            "onset_started": _as_int_any(onset_row, "frame_started_speech_region", "frame_started") if isinstance(onset_row, dict) else 0,
            "onset_evidence": _as_float(onset_row, "evidence") if isinstance(onset_row, dict) else 0.0,
            "quiet_decision": quiet_row.get("decision", "") if isinstance(quiet_row, dict) and quiet_row else "",
            "quiet_closed": _as_int_any(quiet_row, "frame_closed_speech_region", "frame_closed") if isinstance(quiet_row, dict) else 0,
        })
    return traces

def load_gap_traces(case_dir: pathlib.Path) -> list[dict]:
    gaps = _read_csv_rows(case_dir / "gap_candidates.csv")
    frames = _read_csv_rows(case_dir / "occupancy_frames.csv")
    traces = []
    for gap in gaps:
        start = _as_float(gap, "gap_start")
        end = _as_float(gap, "gap_end")
        in_gap = [
            row for row in frames
            if _as_float(row, "center") >= start and _as_float(row, "center") <= end
        ]
        evidence = [_as_float(row, "evidence") for row in in_gap]
        rms_norm = [_as_float(row, "rms_norm") for row in in_gap]
        traces.append({
            "gap_index": _as_int(gap, "gap_index"),
            "duration_ms": _as_float(gap, "gap_duration") * 1000.0,
            "bridged": _as_int(gap, "bridged"),
            "decision_class": gap.get("decision_class", ""),
            "close_reason": gap.get("close_reason", ""),
            "quiet_evidence": _as_float(gap, "quiet_evidence"),
            "quiet_rms_norm": _as_float(gap, "quiet_rms_norm"),
            "reopen_evidence": _as_float(gap, "reopen_evidence"),
            "reopen_flux": _as_float(gap, "reopen_flux"),
            "strong_quiet_close": _as_int(gap, "strong_quiet_close"),
            "strong_onset_reopen": _as_int(gap, "strong_onset_reopen"),
            "frames": len(in_gap),
            "evidence_min": min(evidence) if evidence else 0.0,
            "evidence_mean": _mean(evidence),
            "rms_mean": _mean(rms_norm),
            "low_frames": sum(1 for row in in_gap if _as_int(row, "low_evidence") != 0),
            "endpoint_frames": sum(1 for row in in_gap if _as_int(row, "endpoint_active") != 0),
            "closed_frames": sum(1 for row in in_gap if _as_int_any(row, "frame_closed_speech_region", "frame_closed") != 0),
            "bridged_frames": sum(1 for row in in_gap if _as_int_any(row, "frame_bridged_speech_region", "frame_bridged") != 0),
            "longest_below_close_ms": _run_length_ms(in_gap, lambda r: _as_float(r, "evidence") < _as_float(r, "close_threshold")),
        })
    return traces


def load_phrase_burst_traces(case_dir: pathlib.Path) -> list[dict]:
    rows = _read_csv_rows(case_dir / "committed.csv")
    if not rows:
        return []

    by_phrase: dict[int, list[dict]] = {}
    for row in rows:
        phrase_index = _as_int(row, "phrase_index", 0)
        if phrase_index <= 0:
            continue
        by_phrase.setdefault(phrase_index, []).append(row)

    traces = []
    for phrase_index, phrase_rows in sorted(by_phrase.items()):
        if len(phrase_rows) < 2:
            continue
        phrase_rows = sorted(phrase_rows, key=lambda r: _as_int(r, "index"))
        first_rows = phrase_rows[: min(4, len(phrase_rows))]
        first_commit = _as_float(first_rows[0], "commit_playback")
        same_commit_first4 = sum(
            1 for row in first_rows
            if abs(_as_float(row, "commit_playback") - first_commit) <= 1e-6
        )
        nonpositive_first4 = sum(1 for row in first_rows if _as_float(row, "commit_lead") <= 0.0)
        sub100_first4 = sum(1 for row in first_rows if _as_float(row, "commit_lead") < 0.100)
        mean_lead_first4 = _mean([_as_float(row, "commit_lead") for row in first_rows])
        center_span_first4_ms = (
            _as_float(first_rows[-1], "center") - _as_float(first_rows[0], "center")
        ) * 1000.0
        traces.append({
            "phrase_index": phrase_index,
            "event_count": len(phrase_rows),
            "same_commit_first4": same_commit_first4,
            "nonpositive_first4": nonpositive_first4,
            "sub100_first4": sub100_first4,
            "mean_lead_first4_ms": mean_lead_first4 * 1000.0,
            "center_span_first4_ms": center_span_first4_ms,
            "first_commit_playback_ms": first_commit * 1000.0,
            "first_center_ms": _as_float(first_rows[0], "center") * 1000.0,
            "first4_words": "/".join((row.get("word", "") or "?") for row in first_rows),
            "first4_poses": "/".join((row.get("pose", "") or "?") for row in first_rows),
            "flagged": 1 if (
                same_commit_first4 >= 3
                or nonpositive_first4 >= 1
                or sub100_first4 >= 2
            ) else 0,
        })
    return traces

from grade_summary import compute_summary, load_case_grades, write_summary


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "outputs" / "runs" / "latest"
    gold_root = pathlib.Path(__file__).resolve().parents[1] / "inputs" / "gold"
    rows, graded, ungraded = load_case_grades(root, gold_root)
    summary = compute_summary(rows, graded, ungraded)
    write_summary(root, summary)

    for row in graded:
        case = row.get("case", "")
        aps = row.get("audio_progress_summary", {})
        srd = row.get("speech_region_diagnostics", {})
        gcs = row.get("gap_candidate_summary", {})
        pcs = row.get("phone_class_summary", {})
        worst = "none"
        if pcs:
            worst_name, worst_stats = max(
                pcs.items(),
                key=lambda item: (float(item[1].get("center_ms", 0.0)), int(item[1].get("unmapped", 0))),
            )
            worst = (
                f"{worst_name}:n={int(worst_stats.get('n', 0))} "
                f"center_ms={float(worst_stats.get('center_ms', 0.0)):.1f} "
                f"unmapped={int(worst_stats.get('unmapped', 0))}"
            )
        print(
            f"CASE_REGION_DIAG {case} "
            f"gold_regions={int(srd.get('gold_count', 0))} "
            f"pred_regions={int(srd.get('predicted_count', 0))} "
            f"count_delta={int(srd.get('count_delta', 0))} "
            f"indexed_start_ms={float(srd.get('indexed', {}).get('mean_start_error_ms', 0.0)):.1f} "
            f"indexed_end_ms={float(srd.get('indexed', {}).get('mean_end_error_ms', 0.0)):.1f} "
            f"indexed_overlap={float(srd.get('indexed', {}).get('mean_overlap_ratio', 0.0)):.3f} "
            f"overlap_start_ms={float(srd.get('overlap', {}).get('mean_start_error_ms', 0.0)):.1f} "
            f"overlap_end_ms={float(srd.get('overlap', {}).get('mean_end_error_ms', 0.0)):.1f} "
            f"overlap_ratio={float(srd.get('overlap', {}).get('mean_overlap_ratio', 0.0)):.3f} "
            f"overlap_matched={int(srd.get('overlap', {}).get('overlap_matched_count', 0))} "
            f"nearest_matched={int(srd.get('overlap', {}).get('nearest_matched_count', 0))} "
            f"pred_only={int(srd.get('overlap', {}).get('predicted_only_count', 0))} "
            f"gold_only={int(srd.get('overlap', {}).get('gold_only_count', 0))}"
        )
        if row.get("audio_progress_rows"):
            print(
                f"CASE_DIAG {case} "
                f"progress_rows={int(aps.get('rows', 0))} "
                f"progress_mae01={float(aps.get('mae01', 0.0)):.4f} "
                f"progress_mae_ms={float(aps.get('mae_ms', 0.0)):.1f} "
                f"progress_bias01={float(aps.get('bias01', 0.0)):.4f} "
                f"progress_corr={float(aps.get('corr', 0.0)):.3f} "
                f"anchor_phone={float(aps.get('anchor_mean_phone_probability', 0.0)):.2f} "
                f"anchor_boundary={float(aps.get('anchor_mean_boundary_probability', 0.0)):.2f} "
                f"anchor_speech={float(aps.get('anchor_mean_speech_probability', 0.0)):.2f} "
                f"pause50={float(aps.get('micro_pause_50_count', 0.0)):.1f} "
                f"drift_abs_ms={float(aps.get('retrospective_drift_abs_ms', 0.0)):.1f} "
                f"drift_rate={float(aps.get('drift_mean_playrate', 1.0)):.3f} "
                f"would_advance={int(aps.get('would_advance_rows', 0))} "
                f"advance_reasons={json.dumps(aps.get('advance_reasons', {}), sort_keys=True, separators=(',', ':'))} "
                f"worst_phone_class={worst}"
            )
        else:
            print(
                f"CASE_DIAG {case} "
                f"worst_phone_class={worst}"
            )
        if gcs:
            print(
                f"CASE_GAPS {case} "
                f"count={int(gcs.get('count', 0))} "
                f"bridged={int(gcs.get('bridged', 0))} "
                f"split={int(gcs.get('split', 0))} "
                f"mean_gap_ms={float(gcs.get('mean_gap_ms', 0.0)):.1f} "
                f"ambiguous={int(gcs.get('ambiguous_count', 0))} "
                f"ambiguous_bridged={int(gcs.get('ambiguous_bridged', 0))} "
                f"ambiguous_split={int(gcs.get('ambiguous_split', 0))} "
                f"decisions={json.dumps(gcs.get('decision_counts', {}), sort_keys=True, separators=(',', ':'))}"
            )

        case_dir = root / case
        occ = load_occupancy_summary(case_dir)
        if occ:
            print(
                f"CASE_OCCUPANCY {case} "
                f"frames={int(occ.get('frames', 0))} "
                f"occupied_pct={float(occ.get('occupied_pct', 0.0)):.1f} "
                f"started={int(occ.get('started', 0))} "
                f"closed={int(occ.get('closed', 0))} "
                f"bridged={int(occ.get('bridged', 0))} "
                f"open_candidate={int(occ.get('open_candidate', 0))} "
                f"keep_open={int(occ.get('keep_open', 0))} "
                f"low_evidence={int(occ.get('low_evidence', 0))} "
                f"strong_quiet={int(occ.get('strong_quiet', 0))} "
                f"endpoint_active={int(occ.get('endpoint_active', 0))} "
                f"below_close={int(occ.get('below_close', 0))} "
                f"longest_unoccupied_ms={float(occ.get('longest_unoccupied_ms', 0.0)):.1f} "
                f"longest_low_ms={float(occ.get('longest_low_evidence_ms', 0.0)):.1f} "
                f"longest_below_close_ms={float(occ.get('longest_below_close_ms', 0.0)):.1f} "
                f"evidence_min={float(occ.get('evidence_min', 0.0)):.3f} "
                f"evidence_mean={float(occ.get('evidence_mean', 0.0)):.3f} "
                f"evidence_max={float(occ.get('evidence_max', 0.0)):.3f} "
                f"decisions={json.dumps(occ.get('decisions', {}), sort_keys=True, separators=(',', ':'))}"
            )

        close_supp = load_close_suppressions(case_dir)
        if close_supp:
            print(
                f"CASE_CLOSE_SUPPRESSIONS {case} "
                f"below_close_runs={int(close_supp.get('below_close_runs', 0))} "
                f"successful_closes={int(close_supp.get('successful_closes', 0))} "
                f"failed_runs={int(close_supp.get('failed_runs', 0))} "
                f"longest_failed_ms={float(close_supp.get('longest_failed_ms', 0.0)):.1f} "
                f"mean_failed_ms={float(close_supp.get('mean_failed_ms', 0.0)):.1f} "
                f"longest_successful_ms={float(close_supp.get('longest_successful_ms', 0.0)):.1f} "
                f"mean_successful_ms={float(close_supp.get('mean_successful_ms', 0.0)):.1f} "
                f"broken_by={json.dumps(close_supp.get('broken_by', {}), sort_keys=True, separators=(',', ':'))}"
            )
            failed_runs = [r for r in close_supp.get('runs', []) if not r.get('successful')]
            failed_runs = sorted(failed_runs, key=lambda r: float(r.get('duration_ms', 0.0)), reverse=True)[:8]
            for run in failed_runs:
                print(
                    f"CASE_CLOSE_TRACE {case} "
                    f"run={int(run.get('run_index', 0))} "
                    f"start_ms={float(run.get('start_ms', 0.0)):.1f} "
                    f"end_ms={float(run.get('end_ms', 0.0)):.1f} "
                    f"dur_ms={float(run.get('duration_ms', 0.0)):.1f} "
                    f"terminated_by={run.get('terminated_by', '')} "
                    f"frames={int(run.get('frames', 0))} "
                    f"evidence_min={float(run.get('evidence_min', 0.0)):.3f} "
                    f"evidence_mean={float(run.get('evidence_mean', 0.0)):.3f} "
                    f"close_threshold={float(run.get('close_threshold_mean', 0.0)):.3f} "
                    f"endpoint_frames={int(run.get('endpoint_frames', 0))} "
                    f"low_frames={int(run.get('low_frames', 0))} "
                    f"keep_open_frames={int(run.get('keep_open_frames', 0))} "
                    f"prev_evidence={float(run.get('prev_evidence', 0.0)):.3f} "
                    f"next_evidence={float(run.get('next_evidence', 0.0)):.3f} "
                    f"next_endpoint={int(run.get('next_endpoint', 0))} "
                    f"next_keep_open={int(run.get('next_keep_open', 0))} "
                    f"next_decision={run.get('next_decision', '')}"
                )


        speech_regions = load_island_lifecycle(case_dir)
        if speech_regions:
            print(
                f"CASE_SPEECH_REGIONS {case} "
                f"count={int(speech_regions.get('count', 0))} "
                f"source={speech_regions.get('source', '')} "
                f"first_open_ms={float(speech_regions.get('first_open_ms', 0.0)):.1f} "
                f"last_close_ms={float(speech_regions.get('last_close_ms', 0.0)):.1f} "
                f"total_ms={float(speech_regions.get('total_ms', 0.0)):.1f} "
                f"mean_duration_ms={float(speech_regions.get('mean_duration_ms', 0.0)):.1f} "
                f"mean_gap_ms={float(speech_regions.get('mean_gap_ms', 0.0)):.1f} "
                f"max_gap_ms={float(speech_regions.get('max_gap_ms', 0.0)):.1f}"
            )
            for speech_region in speech_regions.get('islands', [])[:12]:
                print(
                    f"CASE_SPEECH_REGION_TRACE {case} "
                    f"region={int(speech_region.get('index', 0))} "
                    f"start_ms={float(speech_region.get('start_ms', 0.0)):.1f} "
                    f"end_ms={float(speech_region.get('end_ms', 0.0)):.1f} "
                    f"dur_ms={float(speech_region.get('duration_ms', 0.0)):.1f} "
                    f"gap_after_ms={float(speech_region.get('gap_after_ms', 0.0)):.1f} "
                    f"source={speech_region.get('source', '')} "
                    f"open_started={int(speech_region.get('open_frame_started', 0))} "
                    f"close_closed={int(speech_region.get('close_frame_closed', 0))} "
                    f"open_decision={speech_region.get('open_decision', '')} "
                    f"close_decision={speech_region.get('close_decision', '')} "
                    f"open_evidence={float(speech_region.get('open_evidence', 0.0)):.3f} "
                    f"close_evidence={float(speech_region.get('close_evidence', 0.0)):.3f} "
                    f"open_time_error_ms={float(speech_region.get('open_time_error_ms', 0.0)):.1f} "
                    f"close_time_error_ms={float(speech_region.get('close_time_error_ms', 0.0)):.1f}"
                )

        attach_traces = load_attach_traces(case_dir)
        if attach_traces:
            attach_count = sum(1 for t in attach_traces if int(t.get('attach', 0)) != 0)
            split_count = len(attach_traces) - attach_count
            print(
                f"CASE_ATTACHMENTS {case} "
                f"candidates={len(attach_traces)} "
                f"attached={attach_count} "
                f"split={split_count} "
                f"decisions={json.dumps(dict(sorted(Counter(t.get('decision_class', '') for t in attach_traces).items())), sort_keys=True, separators=(',', ':'))}"
            )
            for trace in attach_traces[:12]:
                print(
                    f"CASE_ATTACH_TRACE {case} "
                    f"gap={int(trace.get('gap_index', 0))} "
                    f"attach={int(trace.get('attach', 0))} "
                    f"decision={trace.get('decision_class', '')} "
                    f"onset_ms={float(trace.get('onset_ms', 0.0)):.1f} "
                    f"acoustic_gap_ms={float(trace.get('acoustic_gap_ms', 0.0)):.1f} "
                    f"emitted_gap_ms={float(trace.get('emitted_gap_ms', 0.0)):.1f} "
                    f"prev_region={int(trace.get('prev_region', -1))} "
                    f"prev_end_ms={float(trace.get('prev_region_end_ms', 0.0)):.1f} "
                    f"next_region={int(trace.get('next_region', -1))} "
                    f"next_start_ms={float(trace.get('next_region_start_ms', 0.0)):.1f} "
                    f"resolved_region={int(trace.get('resolved_region', -1))} "
                    f"close_reason={trace.get('close_reason', '')} "
                    f"quiet={float(trace.get('quiet_evidence', 0.0)):.3f} "
                    f"reopen={float(trace.get('reopen_evidence', 0.0)):.3f} "
                    f"reopen_flux={float(trace.get('reopen_flux', 0.0)):.3f} "
                    f"strong_quiet={int(trace.get('strong_quiet_close', 0))} "
                    f"strong_onset={int(trace.get('strong_onset_reopen', 0))} "
                    f"onset_started={int(trace.get('onset_started', 0))} "
                    f"onset_decision={trace.get('onset_decision', '')} "
                    f"onset_evidence={float(trace.get('onset_evidence', 0.0)):.3f} "
                    f"quiet_closed={int(trace.get('quiet_closed', 0))} "
                    f"quiet_decision={trace.get('quiet_decision', '')}"
                )

        for gap in load_gap_traces(case_dir):
            print(
                f"CASE_GAP_TRACE {case} "
                f"gap={int(gap.get('gap_index', 0))} "
                f"dur_ms={float(gap.get('duration_ms', 0.0)):.1f} "
                f"bridged={int(gap.get('bridged', 0))} "
                f"decision={gap.get('decision_class', '')} "
                f"close_reason={gap.get('close_reason', '')} "
                f"quiet={float(gap.get('quiet_evidence', 0.0)):.3f} "
                f"quiet_rms={float(gap.get('quiet_rms_norm', 0.0)):.3f} "
                f"reopen={float(gap.get('reopen_evidence', 0.0)):.3f} "
                f"reopen_flux={float(gap.get('reopen_flux', 0.0)):.3f} "
                f"strong_quiet={int(gap.get('strong_quiet_close', 0))} "
                f"strong_onset={int(gap.get('strong_onset_reopen', 0))} "
                f"frames={int(gap.get('frames', 0))} "
                f"evidence_min={float(gap.get('evidence_min', 0.0)):.3f} "
                f"evidence_mean={float(gap.get('evidence_mean', 0.0)):.3f} "
                f"low_frames={int(gap.get('low_frames', 0))} "
                f"endpoint_frames={int(gap.get('endpoint_frames', 0))} "
                f"closed_frames={int(gap.get('closed_frames', 0))} "
                f"bridged_frames={int(gap.get('bridged_frames', 0))} "
                f"longest_below_close_ms={float(gap.get('longest_below_close_ms', 0.0)):.1f}"
            )


    print(
        f"CASES total={summary['total_cases']} graded={summary['graded_cases']} "
        f"qualified={summary.get('qualified_cases', 0)} "
        f"disqualified_region_mismatch={summary.get('disqualified_speech_region_mismatch_cases', 0)} "
        f"ungraded={summary['ungraded_cases']} degenerate={summary['degenerate_cases']}"
    )
    print(
        f"GRADE center_ms={summary['phoneme_center_ms']:.3f} start_ms={summary['phoneme_start_ms']:.3f} "
        f"end_ms={summary['phoneme_end_ms']:.3f} order_fail_cases={summary['order_fail_cases']}"
    )
    print(
        f"GRADE_STATS "
        f"center_mean_ms={summary.get('phoneme_center_mean_ms', 0.0):.3f} "
        f"center_median_ms={summary.get('phoneme_center_median_ms', 0.0):.3f} "
        f"center_p90_ms={summary.get('phoneme_center_p90_ms', 0.0):.3f} "
        f"center_max_ms={summary.get('phoneme_center_max_ms', 0.0):.3f} "
        f"center_stddev_ms={summary.get('phoneme_center_stddev_ms', 0.0):.3f} "
        f"start_mean_ms={summary.get('phoneme_start_mean_ms', 0.0):.3f} "
        f"start_median_ms={summary.get('phoneme_start_median_ms', 0.0):.3f} "
        f"start_p90_ms={summary.get('phoneme_start_p90_ms', 0.0):.3f} "
        f"end_mean_ms={summary.get('phoneme_end_mean_ms', 0.0):.3f} "
        f"end_median_ms={summary.get('phoneme_end_median_ms', 0.0):.3f} "
        f"end_p90_ms={summary.get('phoneme_end_p90_ms', 0.0):.3f}"
    )
    print(
        f"REGIONS speech_start_ms={summary['speech_boundary_start_ms']:.3f} "
        f"speech_end_ms={summary['speech_boundary_end_ms']:.3f} "
        f"count_mismatch={summary['speech_region_count_mismatch_cases']}"
    )
    print(
        f"REGIONS_INDEXED gold_count={summary.get('speech_region_gold_count', 0.0):.3f} "
        f"pred_count={summary.get('speech_region_predicted_count', 0.0):.3f} "
        f"count_delta={summary.get('speech_region_count_delta', 0.0):.3f} "
        f"start_ms={summary.get('speech_region_indexed_start_ms', 0.0):.3f} "
        f"end_ms={summary.get('speech_region_indexed_end_ms', 0.0):.3f} "
        f"overlap_ratio={summary.get('speech_region_indexed_overlap_ratio', 0.0):.3f}"
    )
    print(
        f"REGIONS_OVERLAP gold_count={summary.get('speech_region_gold_count', 0.0):.3f} "
        f"pred_count={summary.get('speech_region_predicted_count', 0.0):.3f} "
        f"overlap_matched={summary.get('speech_region_overlap_matched_count', 0.0):.3f} "
        f"nearest_matched={summary.get('speech_region_nearest_matched_count', 0.0):.3f} "
        f"pred_only={summary.get('speech_region_predicted_only_count', 0.0):.3f} "
        f"gold_only={summary.get('speech_region_gold_only_count', 0.0):.3f} "
        f"start_ms={summary.get('speech_region_overlap_start_ms', 0.0):.3f} "
        f"end_ms={summary.get('speech_region_overlap_end_ms', 0.0):.3f} "
        f"overlap_ratio={summary.get('speech_region_overlap_ratio', 0.0):.3f}"
    )
    print(
        f"REGION_FLOW leak_cases={summary.get('speech_region_containment_leak_cases', 0)} "
        f"invalid_cases={summary.get('speech_region_containment_invalid_cases', 0)} "
        f"leak_events={summary.get('speech_region_containment_leak_events', 0)} "
        f"early_entry_events={summary.get('speech_region_containment_early_entry_events', 0)} "
        f"late_tail_events={summary.get('speech_region_containment_late_tail_events', 0)} "
        f"invalid_events={summary.get('speech_region_containment_invalid_events', 0)} "
        f"early_leak_ms={summary.get('speech_region_containment_early_leak_ms', 0.0):.3f} "
        f"late_leak_ms={summary.get('speech_region_containment_late_leak_ms', 0.0):.3f}"
    )
    print(
        f"DROPPED_VISEMES cases={summary.get('dropped_viseme_cases', 0)} "
        f"dropped={summary.get('dropped_viseme_count', 0)} "
        f"region_closed={summary.get('dropped_region_closed_count', 0)} "
        f"missing_region={summary.get('dropped_missing_region_count', 0)} "
        f"strong_visible={summary.get('dropped_strong_visible_count', 0)} "
        f"rate={summary.get('dropped_viseme_rate', 0.0):.4f}"
    )
    print(
        f"DROP_CAUSES regions={summary.get('drop_region_count', 0)} "
        f"regions_with_drops={summary.get('drop_regions_with_drops', 0)} "
        f"without_observed_region={summary.get('drop_regions_without_observed_region', 0)} "
        f"with_observed_region={summary.get('drop_regions_with_observed_region', 0)} "
        f"regions_without_commits={summary.get('drop_regions_without_any_commits', 0)} "
        f"tail_pinned_aligner_commits={summary.get('drop_aligner_tail_pinned_committed_count', 0)} "
        f"first_commit_lag_ms={summary.get('drop_first_commit_lag_ms', 0.0):.1f} "
        f"last_commit_to_region_end_ms={summary.get('drop_last_commit_to_region_end_ms', 0.0):.1f} "
        f"required_deficit_mean_ms={summary.get('drop_required_deficit_mean_ms', 0.0):.1f} "
        f"required_deficit_max_ms={summary.get('drop_required_deficit_max_ms', 0.0):.1f}"
    )
    print(
        f"UNDERRUN regions={summary.get('underrun_region_count', 0)} "
        f"idle_tail_ms={summary.get('underrun_idle_tail_ms', 0.0):.1f} "
        f"idle_tail_max_ms={summary.get('underrun_idle_tail_max_ms', 0.0):.1f}"
    )
    print(
        f"GAPS count={summary.get('gap_candidate_count', 0)} "
        f"bridged={summary.get('gap_candidate_bridged', 0)} "
        f"split={summary.get('gap_candidate_split', 0)} "
        f"mean_gap_ms={summary.get('gap_candidate_mean_gap_ms', 0.0):.3f} "
        f"ambiguous={summary.get('gap_candidate_ambiguous_count', 0)} "
        f"ambiguous_bridged={summary.get('gap_candidate_ambiguous_bridged', 0)} "
        f"ambiguous_split={summary.get('gap_candidate_ambiguous_split', 0)} "
        f"ambiguous_mean_gap_ms={summary.get('gap_candidate_ambiguous_mean_gap_ms', 0.0):.3f}"
    )
    print(
        "GAP_DECISIONS "
        + json.dumps(summary.get("gap_decision_counts", {}), sort_keys=True, separators=(",", ":"))
    )

    occupancy_case_count = 0
    occupancy_frame_count = 0
    occupancy_occupied_pct = []
    occupancy_longest_unoccupied = []
    occupancy_decisions = Counter()
    for row in graded:
        case_dir = root / row.get("case", "")
        occ = load_occupancy_summary(case_dir)
        if not occ:
            continue
        occupancy_case_count += 1
        occupancy_frame_count += int(occ.get("frames", 0))
        occupancy_occupied_pct.append(float(occ.get("occupied_pct", 0.0)))
        occupancy_longest_unoccupied.append(float(occ.get("longest_unoccupied_ms", 0.0)))
        occupancy_decisions.update(occ.get("decisions", {}))
    if occupancy_case_count:
        print(
            f"OCCUPANCY_SUMMARY cases={occupancy_case_count} "
            f"frames={occupancy_frame_count} "
            f"occupied_pct_mean={_mean(occupancy_occupied_pct):.1f} "
            f"longest_unoccupied_max_ms={max(occupancy_longest_unoccupied):.1f} "
            f"decisions={json.dumps(dict(sorted(occupancy_decisions.items())), sort_keys=True, separators=(',', ':'))}"
        )

    close_case_count = 0
    close_runs_total = 0
    close_success_total = 0
    close_failed_total = 0
    close_longest_failed = []
    close_broken = Counter()
    for row in graded:
        case_dir = root / row.get("case", "")
        cs = load_close_suppressions(case_dir)
        if not cs:
            continue
        close_case_count += 1
        close_runs_total += int(cs.get("below_close_runs", 0))
        close_success_total += int(cs.get("successful_closes", 0))
        close_failed_total += int(cs.get("failed_runs", 0))
        close_longest_failed.append(float(cs.get("longest_failed_ms", 0.0)))
        close_broken.update(cs.get("broken_by", {}))
    if close_case_count:
        print(
            f"CLOSE_SUPPRESSION_SUMMARY cases={close_case_count} "
            f"below_close_runs={close_runs_total} "
            f"successful_closes={close_success_total} "
            f"failed_runs={close_failed_total} "
            f"longest_failed_max_ms={max(close_longest_failed) if close_longest_failed else 0.0:.1f} "
            f"broken_by={json.dumps(dict(sorted(close_broken.items())), sort_keys=True, separators=(',', ':'))}"
        )

    speech_region_case_count = 0
    speech_region_count_total = 0
    speech_region_duration_means = []
    speech_region_gap_maxes = []
    speech_region_sources = Counter()
    for row in graded:
        case_dir = root / row.get("case", "")
        speech_regions = load_island_lifecycle(case_dir)
        if not speech_regions:
            continue
        speech_region_case_count += 1
        speech_region_count_total += int(speech_regions.get("count", 0))
        speech_region_duration_means.append(float(speech_regions.get("mean_duration_ms", 0.0)))
        speech_region_gap_maxes.append(float(speech_regions.get("max_gap_ms", 0.0)))
        speech_region_sources[speech_regions.get("source", "unknown")] += 1
    if speech_region_case_count:
        print(
            f"SPEECH_REGION_SUMMARY cases={speech_region_case_count} "
            f"regions={speech_region_count_total} "
            f"mean_duration_ms={_mean(speech_region_duration_means):.1f} "
            f"max_gap_ms={max(speech_region_gap_maxes) if speech_region_gap_maxes else 0.0:.1f} "
            f"sources={json.dumps(dict(sorted(speech_region_sources.items())), sort_keys=True, separators=(',', ':'))}"
        )

    attach_case_count = 0
    attach_candidate_total = 0
    attach_attached_total = 0
    attach_split_total = 0
    attach_decisions = Counter()
    attach_gap_ms = []
    for row in graded:
        case_dir = root / row.get("case", "")
        traces = load_attach_traces(case_dir)
        if not traces:
            continue
        attach_case_count += 1
        attach_candidate_total += len(traces)
        for trace in traces:
            if int(trace.get("attach", 0)) != 0:
                attach_attached_total += 1
            else:
                attach_split_total += 1
            attach_decisions[trace.get("decision_class", "")] += 1
            attach_gap_ms.append(float(trace.get("acoustic_gap_ms", 0.0)))
    if attach_case_count:
        print(
            f"ATTACHMENT_SUMMARY cases={attach_case_count} "
            f"candidates={attach_candidate_total} "
            f"attached={attach_attached_total} "
            f"split={attach_split_total} "
            f"mean_gap_ms={_mean(attach_gap_ms):.1f} "
            f"decisions={json.dumps(dict(sorted(attach_decisions.items())), sort_keys=True, separators=(',', ':'))}"
        )

    print(
        f"WORDS assignment_rate={summary.get('word_assignment_rate', summary['word_f1']):.4f} detected_start_ms={summary['detected_word_onset_ms']:.3f} "
        f"intra_word_coverage={summary['intra_word_coverage_rate']:.4f} "
        f"intra_word_center_ms={summary['intra_word_center_ms']:.3f}"
    )
    print(
        f"WORD_STATS "
        f"detected_start_mean_ms={summary.get('detected_word_onset_mean_ms', 0.0):.3f} "
        f"detected_start_median_ms={summary.get('detected_word_onset_median_ms', 0.0):.3f} "
        f"detected_start_p90_ms={summary.get('detected_word_onset_p90_ms', 0.0):.3f} "
        f"detected_start_max_ms={summary.get('detected_word_onset_max_ms', 0.0):.3f} "
        f"detected_start_stddev_ms={summary.get('detected_word_onset_stddev_ms', 0.0):.3f} "
        f"case_median_median_ms={summary.get('detected_word_onset_event_median_median_ms', 0.0):.3f} "
        f"case_median_p90_ms={summary.get('detected_word_onset_event_median_p90_ms', 0.0):.3f} "
        f"duration_mean_ms={summary.get('word_duration_mean_ms', 0.0):.3f} "
        f"duration_median_ms={summary.get('word_duration_median_ms', 0.0):.3f} "
        f"duration_p90_ms={summary.get('word_duration_p90_ms', 0.0):.3f}"
    )
    print(
        f"INTRA_WORD_STATS "
        f"center_mean_ms={summary.get('intra_word_center_mean_ms', 0.0):.3f} "
        f"center_median_ms={summary.get('intra_word_center_median_ms', 0.0):.3f} "
        f"center_p90_ms={summary.get('intra_word_center_p90_ms', 0.0):.3f} "
        f"center_max_ms={summary.get('intra_word_center_max_ms', 0.0):.3f} "
        f"center_stddev_ms={summary.get('intra_word_center_stddev_ms', 0.0):.3f}"
    )
    if summary.get("direct_aligner_available", False):
        print(
            f"DIRECT_ALIGNER match_rate={summary['direct_aligner_match_rate']:.4f} "
            f"center_ms={summary['direct_aligner_center_ms']:.3f}"
        )
        print(
            f"DIRECT_ALIGNER_STATS "
            f"center_mean_ms={summary.get('direct_aligner_center_mean_ms', 0.0):.3f} "
            f"center_median_ms={summary.get('direct_aligner_center_median_ms', 0.0):.3f} "
            f"center_p90_ms={summary.get('direct_aligner_center_p90_ms', 0.0):.3f} "
            f"center_max_ms={summary.get('direct_aligner_center_max_ms', 0.0):.3f} "
            f"center_stddev_ms={summary.get('direct_aligner_center_stddev_ms', 0.0):.3f}"
        )
    if summary.get("audio_progress_available", False):
        print(f"AUDIO_PROGRESS rows={summary['audio_progress_rows']}")
        print(
            f"AUDIO_PROGRESS_MFA rows={summary['audio_progress_mfa_rows']} "
            f"would_advance={summary.get('audio_progress_would_advance_rows', 0)} "
            f"mae01={summary['audio_progress_mfa_mae01']:.4f} "
            f"median01={summary['audio_progress_mfa_median01']:.4f} "
            f"mae_ms={summary['audio_progress_mfa_mae_ms']:.1f} "
            f"p90_ms={summary['audio_progress_mfa_p90_ms']:.1f} "
            f"bias01={summary['audio_progress_mfa_bias01']:.4f} "
            f"corr={summary['audio_progress_mfa_corr']:.3f} "
            f"component_mae01=time:{summary['audio_progress_time_prior_mae01']:.4f},"
            f"density:{summary['audio_progress_density_mae01']:.4f},"
            f"boundary:{summary['audio_progress_boundary_mae01']:.4f},"
            f"phone:{summary['audio_progress_phone_mae01']:.4f}"
        )
        print(
            "ADVANCE_REASONS "
            + json.dumps(summary.get("advance_reason_counts", {}), sort_keys=True, separators=(",", ":"))
        )
        print(
            f"TIMING_PIVOT micro_pause_50={summary.get('audio_progress_micro_pause_50_count', 0.0):.3f} "
            f"micro_pause_75={summary.get('audio_progress_micro_pause_75_count', 0.0):.3f} "
            f"micro_pause_120={summary.get('audio_progress_micro_pause_120_count', 0.0):.3f} "
            f"drift_abs01={summary.get('audio_progress_retrospective_drift_abs01', 0.0):.4f} "
            f"drift_abs_ms={summary.get('audio_progress_retrospective_drift_abs_ms', 0.0):.1f} "
            f"drift_bias01={summary.get('audio_progress_retrospective_drift_bias01', 0.0):.4f} "
            f"drift_conf={summary.get('audio_progress_retrospective_drift_confidence', 0.0):.3f} "
            f"drift_rate={summary.get('audio_progress_drift_mean_playrate', 1.0):.3f} "
            f"pll_rate={summary.get('audio_progress_filtered_pll_playrate', 1.0):.3f} "
            f"pll_tempo={summary.get('audio_progress_pll_tempo_playrate', 1.0):.3f} "
            f"pll_phase={summary.get('audio_progress_pll_phase_playrate', 1.0):.3f} "
            f"pll_conf={summary.get('audio_progress_filtered_pll_confidence', 0.0):.3f} "
            f"warp_rate={summary.get('audio_progress_timing_warp_rate', 1.0):.3f} "
            f"warp_conf={summary.get('audio_progress_timing_warp_confidence', 0.0):.3f} "
            f"warp_err_ms={summary.get('audio_progress_timing_warp_error_ms', 0.0):.1f}"
        )
    phone_chunks = []
    for klass, stats in sorted(summary.get('phone_class_errors', {}).items()):
        phone_chunks.append(
            f"{klass}:n={int(stats.get('n', 0))},matched={int(stats.get('matched', 0))},"
            f"unmapped={int(stats.get('unmapped', 0))},center_ms={float(stats.get('center_ms', 0.0)):.1f},"
            f"start_ms={float(stats.get('start_ms', 0.0)):.1f},end_ms={float(stats.get('end_ms', 0.0)):.1f}"
        )
    print("PHONE_CLASS_ERRORS " + " | ".join(phone_chunks))

    detector_case_count = 0
    pause_frames = 0
    pause_matches = 0
    gold_word_boundaries = 0
    predicted_word_boundaries = 0
    matched_word_boundaries = 0
    boundary_error_weighted_sum = 0.0
    for row in graded:
        case_dir = root / row.get("case", "")
        detector = load_streaming_detector_summary(case_dir)
        if not detector:
            continue
        detector_case_count += 1
        pause_frames += int(detector.get("pause_frames", 0))
        pause_matches += int(detector.get("pause_matches", 0))
        gold_word_boundaries += int(detector.get("gold_word_boundaries", 0))
        predicted_word_boundaries += int(detector.get("predicted_word_boundaries", 0))
        matched_word_boundaries += int(detector.get("matched_word_boundaries", 0))
        boundary_error_weighted_sum += (
            float(detector.get("word_boundary_mean_abs_ms", 0.0))
            * int(detector.get("matched_word_boundaries", 0))
        )
    if detector_case_count:
        pause_acc = (pause_matches / pause_frames) if pause_frames else 0.0
        boundary_precision = (
            matched_word_boundaries / predicted_word_boundaries
            if predicted_word_boundaries else 0.0
        )
        boundary_recall = (
            matched_word_boundaries / gold_word_boundaries
            if gold_word_boundaries else 0.0
        )
        boundary_f1 = (
            2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
            if (boundary_precision + boundary_recall) else 0.0
        )
        boundary_mean_abs_ms = (
            boundary_error_weighted_sum / matched_word_boundaries
            if matched_word_boundaries else 0.0
        )
        print(
            f"STREAMING_DETECTORS cases={detector_case_count} "
            f"pause_frames={pause_frames} pause_top1={pause_acc:.4f} "
            f"gold_word_boundaries={gold_word_boundaries} "
            f"predicted_word_boundaries={predicted_word_boundaries} "
            f"matched_word_boundaries={matched_word_boundaries} "
            f"word_boundary_precision={boundary_precision:.4f} "
            f"word_boundary_recall={boundary_recall:.4f} "
            f"word_boundary_f1={boundary_f1:.4f} "
            f"word_boundary_mean_abs_ms={boundary_mean_abs_ms:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
