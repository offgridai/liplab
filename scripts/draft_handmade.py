import argparse
import datetime as dt
from collections import Counter

from handmade_tools import (
    case_stems,
    handmade_draft_dir,
    mfa_align_root,
    mfa_textgrid_path,
    offline_gold_case_dir,
    offline_gold_run_root,
    parse_float,
    parse_int,
    parse_long_textgrid,
    read_csv_rows,
    read_text,
    repo_root,
    run_mfa_align,
    run_runner_and_capture,
    strip_phone_stress,
    wav_metadata,
    write_json,
    write_text,
)


def draft_flags(event: dict[str, object], previous_center: float | None) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    start = float(event.get("start", 0.0))
    end = float(event.get("end", 0.0))
    center = float(event.get("center", 0.0))
    reason = str(event.get("reason", ""))
    alignment_reason = str(event.get("alignment_reason", ""))
    mapped = bool(event.get("mapped_to_observed_speech", False))
    strength = float(event.get("strength", 0.0))

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


def non_empty_word_intervals(parsed: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    for interval in parsed.get("words", []):
        text = str(interval.get("text", "")).strip()
        if not text:
            continue
        words.append(
            {
                "word_index": len(words),
                "word": text,
                "start": float(interval["start"]),
                "end": float(interval["end"]),
            }
        )
    return words


def phones_by_word(parsed: dict[str, list[dict[str, object]]], words: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    result: dict[int, list[dict[str, object]]] = {int(word["word_index"]): [] for word in words}
    word_cursor = 0
    phone_index = 0
    phones = parsed.get("phones", [])

    while phone_index < len(phones) and word_cursor < len(words):
        phone = phones[phone_index]
        label = str(phone.get("text", "")).strip()
        if not label:
            phone_index += 1
            continue

        center = (float(phone["start"]) + float(phone["end"])) * 0.5
        while word_cursor + 1 < len(words) and center > float(words[word_cursor]["end"]) + 1e-6:
            word_cursor += 1

        word = words[word_cursor]
        if float(word["start"]) - 1e-6 <= center <= float(word["end"]) + 1e-6:
            result[int(word["word_index"])].append(
                {
                    "index_in_word": len(result[int(word["word_index"])]),
                    "phone": label,
                    "phone_base": strip_phone_stress(label),
                    "start": float(phone["start"]),
                    "end": float(phone["end"]),
                    "word_index": int(word["word_index"]),
                    "word": str(word["word"]),
                }
            )
        phone_index += 1

    return result


def locate_phone_interval(
    planned_row: dict[str, str],
    phones_for_word: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    source_phone = strip_phone_stress(planned_row.get("source_phone", ""))
    source_phone_index = parse_int(planned_row.get("source_phone_index"), -1)

    if 0 <= source_phone_index < len(phones_for_word):
        phone = phones_for_word[source_phone_index]
        if phone["phone_base"] == source_phone:
            return phone, "mfa_exact_source_phone_index"
        return phone, "mfa_source_phone_index_label_mismatch"

    matches = [phone for phone in phones_for_word if phone["phone_base"] == source_phone]
    if matches:
        return matches[0], "mfa_source_phone_label_fallback"

    return None, "no_phone_evidence"


def build_visemes(
    planned_rows: list[dict[str, str]],
    committed_rows: list[dict[str, str]],
    phones_for_words: dict[int, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    visemes: list[dict[str, object]] = []
    all_flags: list[dict[str, object]] = []
    previous_center: float | None = None
    previous_start: float | None = None
    committed_by_index = {parse_int(row.get("index")): row for row in committed_rows}

    for row in planned_rows:
        source, alignment_reason = locate_phone_interval(
            row,
            phones_for_words.get(parse_int(row.get("word_index"), -1), []),
        )
        used_fallback = False

        if source is not None:
            start = float(source["start"])
            end = float(source["end"])
            center = (start + end) * 0.5
        else:
            fallback = committed_by_index.get(parse_int(row.get("index")))
            if fallback is None:
                continue
            start = parse_float(fallback.get("start"))
            center = parse_float(fallback.get("center"))
            end = parse_float(fallback.get("end"))
            used_fallback = True

        fallback = committed_by_index.get(parse_int(row.get("index")))
        if previous_start is not None and start + 1e-6 < previous_start:
            if fallback is not None:
                fallback_start = parse_float(fallback.get("start"))
                fallback_center = parse_float(fallback.get("center"))
                fallback_end = parse_float(fallback.get("end"))
                if fallback_start + 1e-6 >= previous_start:
                    start = fallback_start
                    center = fallback_center
                    end = fallback_end
                    used_fallback = True
                    alignment_reason = f"{alignment_reason}_monotonic_fallback"
                else:
                    start = previous_start
                    end = max(end, start + 0.03)
                    center = (start + end) * 0.5
                    alignment_reason = f"{alignment_reason}_monotonic_clamp"
            else:
                start = previous_start
                end = max(end, start + 0.03)
                center = (start + end) * 0.5
                alignment_reason = f"{alignment_reason}_monotonic_clamp"

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
            "commit_reason": "mfa_phone_alignment" if not used_fallback else "offline_committed_fallback",
            "alignment_reason": alignment_reason,
            "source_phone": row.get("source_phone", ""),
            "source_phone_index": parse_int(row.get("source_phone_index"), -1),
            "source_phone_class": "",
            "mapped_to_observed_speech": source is not None,
            "approval": {
                "status": "draft",
                "reviewed": False,
            },
            "notes": [],
        }
        flags = draft_flags(viseme, previous_center)
        viseme["flags"] = flags
        visemes.append(viseme)
        all_flags.extend(flags)
        previous_center = center
        previous_start = start

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
        "- Focus first on visemes flagged for missing MFA phone evidence or offline fallback timing.",
        "- Tighten phrase starts and ends around real pauses.",
        "- Change approval.status to approved before export if this case is ready.",
    ]
    write_text(handmade_draft_dir(case_id) / "review_notes.txt", "\n".join(lines) + "\n")


def build_case_annotation(case_id: str, buffer_ms: int, chunk_ms: int, mfa_num_jobs: int) -> dict[str, object]:
    root = repo_root()
    transcript = read_text(root / "inputs" / "transcripts" / f"{case_id}.txt")
    audio = wav_metadata(root / "inputs" / "wav" / f"{case_id}.wav")
    case_dir = offline_gold_case_dir(case_id)
    planned_rows = read_csv_rows(case_dir / "planned.csv")
    committed_rows = read_csv_rows(case_dir / "committed.csv")
    speech_rows = read_csv_rows(case_dir / "speech_regions.csv")
    parsed = parse_long_textgrid(mfa_textgrid_path(case_id))
    words = non_empty_word_intervals(parsed)
    per_word_phones = phones_by_word(parsed, words)
    visemes, flags = build_visemes(planned_rows, committed_rows, per_word_phones)
    payload = {
        "case_id": case_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "kind": "liplab_mfa_phase1",
            "buffer_ms": buffer_ms,
            "chunk_ms": chunk_ms,
            "mfa_num_jobs": mfa_num_jobs,
            "source_run_dir": case_dir.relative_to(root).as_posix(),
            "mfa_align_dir": mfa_align_root().relative_to(root).as_posix(),
        },
        "approval": {
            "status": "draft",
            "reviewed": False,
            "reviewer": "",
        },
        "transcript": transcript,
        "audio": audio,
        "speech_regions": build_speech_regions(speech_rows),
        "mfa_words": words,
        "mfa_phones": [phone for group in per_word_phones.values() for phone in group],
        "planned_visemes": build_planned_visemes(planned_rows),
        "visemes": visemes,
        "flags": flags,
        "notes": [],
    }
    payload["summary"] = build_summary(visemes, flags)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate batch draft handmade annotations from MFA phone alignments plus transcript-owned viseme identity.")
    parser.add_argument("--buffer-ms", type=int, default=600000)
    parser.add_argument("--chunk-ms", type=int, default=600000)
    parser.add_argument("--mfa-num-jobs", type=int, default=4)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--skip-runner", action="store_true")
    parser.add_argument("--skip-mfa", action="store_true")
    args = parser.parse_args()
    cases = args.cases or case_stems()

    if not args.skip_runner:
        rc = run_runner_and_capture(args.buffer_ms, args.chunk_ms, offline_gold_run_root())
        if rc != 0:
            return rc

    if not args.skip_mfa:
        rc = run_mfa_align(cases, mfa_align_root(), args.mfa_num_jobs)
        if rc != 0:
            return rc

    for case_id in cases:
        payload = build_case_annotation(case_id, args.buffer_ms, args.chunk_ms, args.mfa_num_jobs)
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
