#!/usr/bin/env python3
"""
D08 diagnostic-only phase timing analysis.

This script reads either an Unreal LipsyncDebug directory or a liplab run directory
and writes explicit timing rows that separate:

  audio speech chunk start/end
  committed viseme/event center
  observed submitted/FaceDriver peak proxy
  phrase/prosody group starts and ends

It does not change runtime behavior. It is intended to answer whether perceived
"late after resume" is caused by planner center timing, submitted pose/FaceDriver
peak lag, or whole-phrase phase drift.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        f = float(v)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return default


def _int(v: Any, default: int = -1) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def find_case_dirs(root: Path) -> List[Path]:
    root = root.resolve()
    if (root / "runtime_commit_events.csv").exists() or (root / "submitted_poses.csv").exists():
        return [root]
    dirs: List[Path] = []
    for p in sorted(root.rglob("runtime_commit_events.csv")):
        dirs.append(p.parent)
    return sorted(set(dirs))


def read_metadata(case_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    p = case_dir / "line_metadata.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def speech_islands(case_dir: Path) -> List[Dict[str, Any]]:
    rows = read_csv(case_dir / "runtime_speech_islands.csv") or read_csv(case_dir / "speech_islands.csv")
    out: List[Dict[str, Any]] = []
    for r in rows:
        start = _float(r.get("AudioBufferStartSec", r.get("StartSec", r.get("StartSeconds"))))
        end = _float(r.get("AudioBufferEndSec", r.get("EndSec", r.get("EndSeconds", r.get("AudioBufferLastSpeechSec")))))
        last = _float(r.get("AudioBufferLastSpeechSec", end), end)
        idx = _int(r.get("IslandIndex", r.get("Index")), len(out))
        if end <= start:
            end = last if last > start else start
        out.append({
            "SpeechChunkIndex": idx,
            "SpeechStartSec": start,
            "SpeechEndSec": end,
            "LastSpeechSec": last,
            "SpeechSpanSec": max(0.0, end - start),
        })
    out.sort(key=lambda x: (x["SpeechStartSec"], x["SpeechEndSec"]))
    for i, row in enumerate(out):
        row["SpeechChunkIndex"] = i
    return out


def committed_events(case_dir: Path) -> List[Dict[str, Any]]:
    rows = read_csv(case_dir / "runtime_commit_events.csv")
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "LineID": r.get("LineID", ""),
            "EventIndex": _int(r.get("EventIndex")),
            "PoseID": r.get("PoseID", ""),
            "SourceWord": r.get("SourceWord", ""),
            "CenterSec": _float(r.get("FinalRenderCenterSec", r.get("FinalRenderCenterSeconds"))),
            "CommitPlaybackSec": _float(r.get("PlaybackSecAtCommit", r.get("CommitPlaybackSec"))),
            "CommitLeadSec": _float(r.get("CommitLeadSec")),
            "CommitReason": r.get("CommitReason", ""),
            "SpeechIslandStartSec": _float(r.get("SpeechIslandStartSec", r.get("IslandAudioStartSeconds"))),
            "SpeechIslandEndSec": _float(r.get("SpeechIslandEndSec", r.get("IslandAudioEndSeconds"))),
        })
    out.sort(key=lambda x: x["EventIndex"])
    return out


def submitted_pose_stats(case_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, int]]:
    rows = read_csv(case_dir / "submitted_poses.csv")
    by_event: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    phrase_by_event: Dict[int, int] = {}
    for r in rows:
        ei = _int(r.get("SourceEventIndex", r.get("EventIndex")))
        if ei < 0:
            continue
        by_event[ei].append(r)
        phrase = _int(r.get("PlannerProsodyGroupIndex", r.get("PhraseIndex")), -1)
        if phrase >= 0:
            phrase_by_event[ei] = phrase

    stats: Dict[int, Dict[str, Any]] = {}
    for ei, evrows in by_event.items():
        # Use playback-domain time: this is what the FaceDriver samples against.
        visible_rows = []
        for r in evrows:
            fdw = _float(r.get("FaceDriverWeight", r.get("SubmittedWeight")))
            subw = _float(r.get("SubmittedWeight"))
            t = _float(r.get("PlaybackSec"))
            visible_rows.append((t, fdw, subw, r))
        if not visible_rows:
            continue
        max_fd = max(w for _, w, _, _ in visible_rows)
        peak_rows = [(t, w) for t, w, _, _ in visible_rows if abs(w - max_fd) <= max(0.001, max_fd * 0.02)]
        peak_time = sum(t * w for t, w in peak_rows) / max(1e-6, sum(w for _, w in peak_rows)) if peak_rows else 0.0
        first_vis_10 = min((t for t, w, _, _ in visible_rows if w >= max_fd * 0.10 and w > 0.001), default=0.0)
        first_vis_25 = min((t for t, w, _, _ in visible_rows if w >= max_fd * 0.25 and w > 0.001), default=0.0)
        first_vis_50 = min((t for t, w, _, _ in visible_rows if w >= max_fd * 0.50 and w > 0.001), default=0.0)
        last_vis_10 = max((t for t, w, _, _ in visible_rows if w >= max_fd * 0.10 and w > 0.001), default=0.0)
        sample = visible_rows[0][3]
        stats[ei] = {
            "EventIndex": ei,
            "PoseID": sample.get("PoseID", ""),
            "SourceWord": sample.get("SourceWord", ""),
            "PhraseIndex": phrase_by_event.get(ei, -1),
            "CommittedStartSec": _float(sample.get("CommittedPlaybackStartSec")),
            "CommittedCenterSec": _float(sample.get("CommittedPlaybackCenterSec")),
            "CommittedEndSec": _float(sample.get("CommittedPlaybackEndSec")),
            "CommitPlaybackSec": _float(sample.get("CommitPlaybackSec")),
            "CommitLeadMs": _float(sample.get("CommitLeadMs")),
            "PeakPlaybackSec": peak_time,
            "PeakWeight": max_fd,
            "VisibleOnset10Sec": first_vis_10,
            "VisibleOnset25Sec": first_vis_25,
            "VisibleOnset50Sec": first_vis_50,
            "VisibleEnd10Sec": last_vis_10,
        }
    return stats, phrase_by_event


def nearest_chunk(chunks: List[Dict[str, Any]], t: float, prefer_start: bool = True) -> Optional[Dict[str, Any]]:
    if not chunks:
        return None
    key = "SpeechStartSec" if prefer_start else "SpeechEndSec"
    return min(chunks, key=lambda c: abs(c[key] - t))


def chunk_by_order(chunks: List[Dict[str, Any]], order: int) -> Optional[Dict[str, Any]]:
    if 0 <= order < len(chunks):
        return chunks[order]
    return nearest_chunk(chunks, 0.0) if chunks else None


def analyze_case(case_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    meta = read_metadata(case_dir)
    line_id = meta.get("LineID", case_dir.name)
    npc = meta.get("NPCID", "")
    chunks = speech_islands(case_dir)
    events = committed_events(case_dir)
    pose_stats, phrase_by_event = submitted_pose_stats(case_dir)

    # Fill phrase from submitted poses when commit CSV lacks it.
    event_rows: List[Dict[str, Any]] = []
    for e in events:
        ei = e["EventIndex"]
        ps = pose_stats.get(ei, {})
        phrase = ps.get("PhraseIndex", phrase_by_event.get(ei, -1))
        center = e["CenterSec"]
        chunk = nearest_chunk(chunks, center, prefer_start=True)
        chunk_idx = chunk["SpeechChunkIndex"] if chunk else -1
        chunk_start = chunk["SpeechStartSec"] if chunk else 0.0
        chunk_end = chunk["SpeechEndSec"] if chunk else 0.0
        row = {
            "CaseDir": case_dir.name,
            "NPCID": npc,
            "LineID": line_id or e.get("LineID", ""),
            "EventIndex": ei,
            "PhraseIndex": phrase,
            "PoseID": e["PoseID"],
            "SourceWord": e["SourceWord"],
            "CommittedCenterSec": center,
            "CommitPlaybackSec": e["CommitPlaybackSec"],
            "CommitLeadMs": e["CommitLeadSec"] * 1000.0,
            "FacePeakSec": ps.get("PeakPlaybackSec", 0.0),
            "FacePeakWeight": ps.get("PeakWeight", 0.0),
            "FacePeakMinusCenterMs": (ps.get("PeakPlaybackSec", center) - center) * 1000.0 if ps else 0.0,
            "VisibleOnset10Sec": ps.get("VisibleOnset10Sec", 0.0),
            "VisibleOnset25Sec": ps.get("VisibleOnset25Sec", 0.0),
            "VisibleOnset50Sec": ps.get("VisibleOnset50Sec", 0.0),
            "VisibleOnset10MinusCenterMs": (ps.get("VisibleOnset10Sec", center) - center) * 1000.0 if ps else 0.0,
            "VisibleOnset25MinusCenterMs": (ps.get("VisibleOnset25Sec", center) - center) * 1000.0 if ps else 0.0,
            "MatchedSpeechChunkIndex": chunk_idx,
            "MatchedSpeechStartSec": chunk_start,
            "MatchedSpeechEndSec": chunk_end,
            "CenterMinusMatchedSpeechStartMs": (center - chunk_start) * 1000.0 if chunk else 0.0,
            "CenterMinusMatchedSpeechEndMs": (center - chunk_end) * 1000.0 if chunk else 0.0,
        }
        event_rows.append(row)

    # Phrase rows from groups of event rows. Pair phrase N with speech chunk N when possible;
    # this makes resume-start diagnostics explicit for multi-chunk lines.
    phrase_rows: List[Dict[str, Any]] = []
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in event_rows:
        grouped[int(r.get("PhraseIndex", -1))].append(r)
    phrase_keys = [k for k in sorted(grouped) if k >= 0]
    for ordinal, phrase in enumerate(phrase_keys):
        rows = sorted(grouped[phrase], key=lambda r: r["CommittedCenterSec"])
        first = rows[0]
        last = rows[-1]
        # Prefer chunk order for phrase starts, but fall back to nearest to first center.
        chunk = chunk_by_order(chunks, ordinal)
        if chunk and abs(chunk["SpeechStartSec"] - first["CommittedCenterSec"]) > 0.75:
            near = nearest_chunk(chunks, first["CommittedCenterSec"], prefer_start=True)
            if near:
                chunk = near
        chunk_idx = chunk["SpeechChunkIndex"] if chunk else -1
        start = chunk["SpeechStartSec"] if chunk else 0.0
        end = chunk["SpeechEndSec"] if chunk else 0.0
        phrase_rows.append({
            "CaseDir": case_dir.name,
            "NPCID": npc,
            "LineID": line_id,
            "PhraseOrdinal": ordinal,
            "PhraseIndex": phrase,
            "FirstEventIndex": first["EventIndex"],
            "FirstWord": first["SourceWord"],
            "FirstPoseID": first["PoseID"],
            "LastEventIndex": last["EventIndex"],
            "LastWord": last["SourceWord"],
            "EventCount": len(rows),
            "FirstCenterSec": first["CommittedCenterSec"],
            "FirstFacePeakSec": first["FacePeakSec"],
            "FirstVisibleOnset10Sec": first["VisibleOnset10Sec"],
            "FirstVisibleOnset25Sec": first["VisibleOnset25Sec"],
            "LastCenterSec": last["CommittedCenterSec"],
            "LastFacePeakSec": last["FacePeakSec"],
            "MatchedSpeechChunkIndex": chunk_idx,
            "SpeechStartSec": start,
            "SpeechEndSec": end,
            "SpeechSpanSec": max(0.0, end - start),
            "FirstCenterMinusSpeechStartMs": (first["CommittedCenterSec"] - start) * 1000.0 if chunk else 0.0,
            "FirstPeakMinusSpeechStartMs": (first["FacePeakSec"] - start) * 1000.0 if chunk and first["FacePeakSec"] else 0.0,
            "FirstVisible10MinusSpeechStartMs": (first["VisibleOnset10Sec"] - start) * 1000.0 if chunk and first["VisibleOnset10Sec"] else 0.0,
            "LastCenterMinusSpeechEndMs": (last["CommittedCenterSec"] - end) * 1000.0 if chunk else 0.0,
            "LastPeakMinusSpeechEndMs": (last["FacePeakSec"] - end) * 1000.0 if chunk and last["FacePeakSec"] else 0.0,
            "PhraseCenterSpanSec": max(0.0, last["CommittedCenterSec"] - first["CommittedCenterSec"]),
            "PhraseCenterSpanToSpeechSpanRatio": (max(0.0, last["CommittedCenterSec"] - first["CommittedCenterSec"]) / max(0.001, end - start)) if chunk else 0.0,
        })

    summary_rows: List[Dict[str, Any]] = []
    if event_rows:
        centers = [r["CommittedCenterSec"] for r in event_rows]
        speech_start = chunks[0]["SpeechStartSec"] if chunks else 0.0
        speech_end = chunks[-1]["SpeechEndSec"] if chunks else 0.0
        first_center = min(centers)
        last_center = max(centers)
        summary_rows.append({
            "CaseDir": case_dir.name,
            "NPCID": npc,
            "LineID": line_id,
            "EventCount": len(event_rows),
            "PhraseCount": len(phrase_rows),
            "SpeechChunkCount": len(chunks),
            "FirstSpeechStartSec": speech_start,
            "LastSpeechEndSec": speech_end,
            "FirstCommittedCenterSec": first_center,
            "LastCommittedCenterSec": last_center,
            "FirstCenterMinusSpeechStartMs": (first_center - speech_start) * 1000.0 if chunks else 0.0,
            "LastCenterMinusSpeechEndMs": (last_center - speech_end) * 1000.0 if chunks else 0.0,
            "CommittedCenterSpanSec": max(0.0, last_center - first_center),
            "SpeechChunkEnvelopeSec": max(0.0, speech_end - speech_start),
            "CommittedSpanToSpeechEnvelopeRatio": max(0.0, last_center - first_center) / max(0.001, speech_end - speech_start) if chunks else 0.0,
        })
    return event_rows, phrase_rows, summary_rows


def aggregate_summary(phrase_rows: List[Dict[str, Any]], case_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def mean(vals: Iterable[float]) -> float:
        vals = [v for v in vals if math.isfinite(v)]
        return statistics.mean(vals) if vals else 0.0

    out: List[Dict[str, Any]] = []
    if phrase_rows:
        out.append({
            "Metric": "phrase_first_center_minus_speech_start_ms_mean",
            "Value": mean(float(r["FirstCenterMinusSpeechStartMs"]) for r in phrase_rows),
        })
        out.append({
            "Metric": "phrase_first_peak_minus_speech_start_ms_mean",
            "Value": mean(float(r["FirstPeakMinusSpeechStartMs"]) for r in phrase_rows if float(r.get("FirstFacePeakSec", 0.0)) > 0.0),
        })
        out.append({
            "Metric": "phrase_last_center_minus_speech_end_ms_mean",
            "Value": mean(float(r["LastCenterMinusSpeechEndMs"]) for r in phrase_rows),
        })
        out.append({
            "Metric": "phrase_span_to_speech_span_ratio_mean",
            "Value": mean(float(r["PhraseCenterSpanToSpeechSpanRatio"]) for r in phrase_rows if float(r.get("SpeechSpanSec", 0.0)) > 0.0),
        })
    if case_rows:
        out.append({
            "Metric": "case_last_center_minus_speech_end_ms_mean",
            "Value": mean(float(r["LastCenterMinusSpeechEndMs"]) for r in case_rows),
        })
        out.append({
            "Metric": "case_committed_span_to_speech_envelope_ratio_mean",
            "Value": mean(float(r["CommittedSpanToSpeechEnvelopeRatio"]) for r in case_rows),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="D08 phase timing diagnostics for LipLab/Unreal logs")
    ap.add_argument("--logs", "--run", dest="logs", required=True, help="LipsyncDebug root, liplab run root, or a single case directory")
    ap.add_argument("--out", default=None, help="Output directory. Default: <logs>/phase_timing_diagnostics")
    args = ap.parse_args()

    root = Path(args.logs)
    out_dir = Path(args.out) if args.out else root / "phase_timing_diagnostics"
    cases = find_case_dirs(root)
    if not cases:
        raise SystemExit(f"No case directories with runtime_commit_events.csv found under {root}")

    all_events: List[Dict[str, Any]] = []
    all_phrases: List[Dict[str, Any]] = []
    all_cases: List[Dict[str, Any]] = []
    for case in cases:
        ev, phr, summ = analyze_case(case)
        all_events.extend(ev)
        all_phrases.extend(phr)
        all_cases.extend(summ)

    event_fields = [
        "CaseDir","NPCID","LineID","EventIndex","PhraseIndex","PoseID","SourceWord",
        "CommittedCenterSec","CommitPlaybackSec","CommitLeadMs","FacePeakSec","FacePeakWeight",
        "FacePeakMinusCenterMs","VisibleOnset10Sec","VisibleOnset25Sec","VisibleOnset50Sec",
        "VisibleOnset10MinusCenterMs","VisibleOnset25MinusCenterMs","MatchedSpeechChunkIndex",
        "MatchedSpeechStartSec","MatchedSpeechEndSec","CenterMinusMatchedSpeechStartMs","CenterMinusMatchedSpeechEndMs",
    ]
    phrase_fields = [
        "CaseDir","NPCID","LineID","PhraseOrdinal","PhraseIndex","FirstEventIndex","FirstWord","FirstPoseID",
        "LastEventIndex","LastWord","EventCount","FirstCenterSec","FirstFacePeakSec","FirstVisibleOnset10Sec",
        "FirstVisibleOnset25Sec","LastCenterSec","LastFacePeakSec","MatchedSpeechChunkIndex","SpeechStartSec",
        "SpeechEndSec","SpeechSpanSec","FirstCenterMinusSpeechStartMs","FirstPeakMinusSpeechStartMs",
        "FirstVisible10MinusSpeechStartMs","LastCenterMinusSpeechEndMs","LastPeakMinusSpeechEndMs",
        "PhraseCenterSpanSec","PhraseCenterSpanToSpeechSpanRatio",
    ]
    case_fields = [
        "CaseDir","NPCID","LineID","EventCount","PhraseCount","SpeechChunkCount","FirstSpeechStartSec",
        "LastSpeechEndSec","FirstCommittedCenterSec","LastCommittedCenterSec","FirstCenterMinusSpeechStartMs",
        "LastCenterMinusSpeechEndMs","CommittedCenterSpanSec","SpeechChunkEnvelopeSec","CommittedSpanToSpeechEnvelopeRatio",
    ]
    write_csv(out_dir / "event_phase_timing.csv", all_events, event_fields)
    write_csv(out_dir / "phrase_phase_timing.csv", all_phrases, phrase_fields)
    write_csv(out_dir / "case_phase_timing.csv", all_cases, case_fields)
    write_csv(out_dir / "phase_timing_summary.csv", aggregate_summary(all_phrases, all_cases), ["Metric", "Value"])

    review = [
        "# D08 phase timing diagnostics",
        "",
        f"Cases analyzed: {len(cases)}",
        f"Events analyzed: {len(all_events)}",
        f"Phrases analyzed: {len(all_phrases)}",
        "",
        "Read `phrase_phase_timing.csv` first. The key columns are:",
        "",
        "- `FirstCenterMinusSpeechStartMs`: committed first viseme center vs detected speech/resume start.",
        "- `FirstPeakMinusSpeechStartMs`: submitted/FaceDriver peak proxy vs detected speech/resume start.",
        "- `LastCenterMinusSpeechEndMs`: phrase tail drift vs detected speech end.",
        "- `PhraseCenterSpanToSpeechSpanRatio`: whether the phrase schedule is slower/faster than detected speech.",
        "",
        "Interpretation:",
        "",
        "- first-center near 0 but first-peak positive => FaceDriver/envelope peak lag.",
        "- first-center positive and first-peak positive => planner/playhead phase is late.",
        "- span ratio > 1 with late tail => animation is slower than speech.",
        "- span ratio near 1 but positive first-start error => release/phase offset issue, not duration.",
    ]
    (out_dir / "phase_timing_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(f"Wrote D08 diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
