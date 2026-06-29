import argparse
import datetime as dt
from collections import Counter

from handmade_tools import (
    case_stems,
    handmade_draft_dir,
    offline_gold_case_dir,
    offline_gold_run_root,
    parse_float,
    parse_int,
    read_csv_rows,
    read_text,
    repo_root,
    run_runner_and_capture,
    wav_metadata,
    write_json,
    write_text,
)


def draft_flags(event: dict[str, str], previous_center: float | None) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    start = parse_float(event.get("start"))
    end = parse_float(event.get("end"))
    center = parse_float(event.get("center"))
    reason = event.get("reason", "")
    alignment_reason = event.get("alignment_reason", "")
    mapped = event.get("mapped_to_observed_speech", "0") == "1"
    strength = parse_float(event.get("strength"), 0.0)

    if not mapped or alignment_reason == "no_phone_evidence":
        flags.append(
            {
                "kind": "no_observed_phone_evidence",
                "start": start,
                "end": end,
                "note": "Draft timing was not anchored to observed phone evidence.",
            }
        )

    if "final_duration_drain" in reason or "final_duration_drain" in alignment_reason:
        flags.append(
            {
                "kind": "final_drain_fallback",
                "start": start,
                "end": end,
                "note": "Draft timing fell back to transcript-duration drain after stream close.",
            }
        )

    if "guard" in reason:
        flags.append(
            {
                "kind": "boundary_guard_adjustment",
                "start": start,
                "end": end,
                "note": f"Placement was clamped by runtime boundary guard: {reason}.",
            }
        )

    if strength < 0.6:
        flags.append(
            {
                "kind": "low_strength_viseme",
                "start": start,
                "end": end,
                "note": f"Viseme strength is low ({strength:.3f}).",
            }
        )

    if previous_center is not None and center - previous_center > 0.45:
        flags.append(
            {
                "kind": "large_inter_event_gap",
                "start": start,
                "end": end,
                "note": f"Large gap from previous viseme center ({center - previous_center:.3f}s).",
            }
        )

    return flags


def build_visemes(committed_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    visemes: list[dict[str, object]] = []
    all_flags: list[dict[str, object]] = []
    previous_center: float | None = None

    for row in committed_rows:
        start = parse_float(row.get("start"))
        center = parse_float(row.get("center"))
        end = parse_float(row.get("end"))
        viseme = {
            "index": parse_int(row.get("index")),
            "id": row.get("pose", ""),
            "pose": row.get("pose", ""),
            "start": start,
            "center": center,
            "end": end,
            "word": row.get("word", ""),
            "word_index": parse_int(row.get("word_index"), -1),
            "phrase_index": parse_int(row.get("phrase_index"), -1),
            "sentence_index": parse_int(row.get("sentence_index"), -1),
            "strength": parse_float(row.get("strength")),
            "commit_reason": row.get("reason", ""),
            "alignment_reason": row.get("alignment_reason", ""),
            "source_phone_class": row.get("source_phone_class", ""),
            "mapped_to_observed_speech": row.get("mapped_to_observed_speech", "0") == "1",
            "approval": {
                "status": "draft",
                "reviewed": False,
            },
            "notes": [],
        }
        flags = draft_flags(row, previous_center)
        viseme["flags"] = flags
        visemes.append(viseme)
        all_flags.extend(flags)
        previous_center = center

    return visemes, all_flags


def build_planned_visemes(planned_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    planned: list[dict[str, object]] = []
    for row in planned_rows:
        planned.append(
            {
                "index": parse_int(row.get("index")),
                "pose": row.get("pose", ""),
                "word": row.get("word", ""),
                "word_index": parse_int(row.get("word_index"), -1),
                "phrase_index": parse_int(row.get("phrase_index"), -1),
                "sentence_index": parse_int(row.get("sentence_index"), -1),
                "text_center_norm": parse_float(row.get("text_center_norm")),
                "strength": parse_float(row.get("strength")),
                "source_phone": row.get("source_phone", ""),
                "source_phone_index": parse_int(row.get("source_phone_index"), -1),
                "generator": row.get("generator", ""),
            }
        )
    return planned


def build_speech_regions(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for row in rows:
        regions.append(
            {
                "index": parse_int(row.get("index")),
                "start": parse_float(row.get("start")),
                "end": parse_float(row.get("end")),
                "last_speech": parse_float(row.get("last_speech")),
                "started": row.get("started", "0") == "1",
                "ended": row.get("ended", "0") == "1",
            }
        )
    return regions


def build_summary(visemes: list[dict[str, object]], flags: list[dict[str, object]]) -> dict[str, object]:
    reasons = Counter(v["commit_reason"] for v in visemes)
    alignments = Counter(v["alignment_reason"] for v in visemes)
    return {
        "viseme_count": len(visemes),
        "flag_count": len(flags),
        "flags_by_kind": dict(Counter(flag["kind"] for flag in flags)),
        "commit_reasons": dict(reasons),
        "alignment_reasons": dict(alignments),
        "mapped_to_observed_speech_count": sum(1 for v in visemes if v["mapped_to_observed_speech"]),
    }


def write_review_notes(case_id: str, payload: dict[str, object]) -> None:
    summary = payload["summary"]
    lines = [
        f"case: {case_id}",
        f"transcript: {payload['transcript']}",
        f"audio: {payload['audio']['duration_sec']:.3f}s @ {payload['audio']['sample_rate_hz']} Hz",
        f"draft visemes: {summary['viseme_count']}",
        f"draft flags: {summary['flag_count']}",
        "",
        "review guidance:",
        "- Confirm pose identity and timing against the WAV.",
        "- Focus first on visemes flagged for missing phone evidence or final-drain fallback.",
        "- Tighten phrase starts and ends around real pauses.",
        "- Change approval.status to approved before export if this case is ready.",
    ]
    write_text(handmade_draft_dir(case_id) / "review_notes.txt", "\n".join(lines) + "\n")


def build_case_annotation(case_id: str, buffer_ms: int, chunk_ms: int) -> dict[str, object]:
    root = repo_root()
    transcript = read_text(root / "inputs" / "transcripts" / f"{case_id}.txt")
    audio = wav_metadata(root / "inputs" / "wav" / f"{case_id}.wav")
    case_dir = offline_gold_case_dir(case_id)
    planned_rows = read_csv_rows(case_dir / "planned.csv")
    committed_rows = read_csv_rows(case_dir / "committed.csv")
    speech_rows = read_csv_rows(case_dir / "speech_regions.csv")

    visemes, flags = build_visemes(committed_rows)
    payload = {
        "case_id": case_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "kind": "liplab_offline_full_wav",
            "buffer_ms": buffer_ms,
            "chunk_ms": chunk_ms,
            "source_run_dir": case_dir.relative_to(root).as_posix(),
        },
        "approval": {
            "status": "draft",
            "reviewed": False,
            "reviewer": "",
        },
        "transcript": transcript,
        "audio": audio,
        "speech_regions": build_speech_regions(speech_rows),
        "planned_visemes": build_planned_visemes(planned_rows),
        "visemes": visemes,
        "flags": flags,
        "notes": [],
    }
    payload["summary"] = build_summary(visemes, flags)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate batch draft handmade annotations from full-WAV offline case passes.")
    parser.add_argument("--buffer-ms", type=int, default=600000)
    parser.add_argument("--chunk-ms", type=int, default=600000)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--skip-runner", action="store_true")
    args = parser.parse_args()

    if not args.skip_runner:
        rc = run_runner_and_capture(args.buffer_ms, args.chunk_ms, offline_gold_run_root())
        if rc != 0:
            return rc

    cases = args.cases or case_stems()
    for case_id in cases:
        payload = build_case_annotation(case_id, args.buffer_ms, args.chunk_ms)
        write_json(handmade_draft_dir(case_id) / "draft.annotation.json", payload)
        write_review_notes(case_id, payload)
        print(
            f"{case_id}: draft_visemes={payload['summary']['viseme_count']} "
            f"flags={payload['summary']['flag_count']} "
            f"mapped={payload['summary']['mapped_to_observed_speech_count']}"
        )

    print(f"Wrote draft annotation package(s) for {len(cases)} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
