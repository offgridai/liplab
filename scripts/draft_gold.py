import argparse
import datetime as dt
from collections import Counter
import math
import re

from gold_tools import (
    case_stems,
    gold_draft_dir,
    mfa_align_root,
    mfa_textgrid_path,
    normalize_word,
    offline_gold_case_dir,
    offline_gold_run_root,
    parse_float,
    parse_int,
    parse_long_textgrid,
    read_wav_mono_samples,
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



SILENCE_PHONE_LABELS = {"", "sil", "sp", "<eps>", "silence", "pau"}
BOUNDARY_PUNCTUATION = {".", ",", ";", ":", "?", "!", "-"}
HARD_BOUNDARY_PUNCTUATION = {".", ";", ":", "?", "!", "-"}


def is_silence_phone_label(label: object) -> bool:
    return str(label or "").strip().lower() in SILENCE_PHONE_LABELS


def non_silence_phone_intervals(parsed: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    phones: list[dict[str, object]] = []
    for interval in parsed.get("phones", []):
        label = str(interval.get("text", interval.get("phone", ""))).strip()
        if is_silence_phone_label(label):
            continue
        start = float(interval.get("start", 0.0))
        end = float(interval.get("end", start))
        if end <= start:
            continue
        phones.append({"phone": label, "start": start, "end": end})
    phones.sort(key=lambda row: (float(row["start"]), float(row["end"])))
    return phones

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
                "note": "Draft timing was not anchored to MFA phone evidence.",
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


def word_metadata_by_index(planned_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {}
    for row in planned_rows:
        word_index = parse_int(row.get("word_index"), -1)
        if word_index < 0 or word_index in metadata:
            continue
        metadata[word_index] = {
            "word_index": word_index,
            "word": row.get("word", ""),
            "phrase_index": parse_int(row.get("phrase_index"), -1),
            "sentence_index": parse_int(row.get("sentence_index"), -1),
        }
    return [metadata[key] for key in sorted(metadata)]


def transcript_word_sequence_with_boundaries(transcript: str) -> list[dict[str, object]]:
    tokens: list[dict[str, object]] = []
    for token in re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*|[.,;:!?-]", transcript):
        if token in BOUNDARY_PUNCTUATION:
            if tokens:
                tokens[-1]["boundary_marks"].append(token)
            continue
        tokens.append(
            {
                "word": token,
                "boundary_marks": [],
            }
        )
    return tokens


def preferred_boundary_mark(marks: list[str]) -> str:
    if not marks:
        return ""
    for mark in marks:
        if mark in HARD_BOUNDARY_PUNCTUATION:
            return mark
    return marks[-1]


def non_empty_word_intervals(parsed: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    for interval in parsed.get("words", []):
        text = str(interval.get("text", "")).strip()
        if not text:
            continue
        words.append(
            {
                "word": text,
                "start": float(interval["start"]),
                "end": float(interval["end"]),
            }
        )
    return words


def attach_transcript_boundaries(
    words: list[dict[str, object]],
    transcript: str,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    transcript_words = transcript_word_sequence_with_boundaries(transcript)
    if len(transcript_words) != len(words):
        flags.append(
            {
                "kind": "transcript_boundary_word_count_mismatch",
                "count_transcript_words": len(transcript_words),
                "count_mfa_words": len(words),
                "note": "Transcript word count did not match MFA word count while attaching punctuation boundaries.",
            }
        )

    for index, word in enumerate(words):
        marks: list[str] = []
        if index < len(transcript_words):
            transcript_word = transcript_words[index]
            marks = list(transcript_word.get("boundary_marks", []))
            if normalize_word(str(transcript_word.get("word", ""))) != normalize_word(str(word.get("word", ""))):
                flags.append(
                    {
                        "kind": "transcript_boundary_word_text_mismatch",
                        "word_index": parse_int(word.get("word_index"), index),
                        "transcript_word": transcript_word.get("word", ""),
                        "mfa_word": word.get("word", ""),
                        "note": "Transcript and MFA word text differed while attaching punctuation boundaries.",
                    }
                )
        word["boundary_marks_after"] = marks
        word["boundary_mark_after"] = preferred_boundary_mark(marks)
    return flags


def build_gold_words(
    parsed_words: list[dict[str, object]],
    metadata_by_order: list[dict[str, object]],
    audio_duration_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    words: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []
    if len(parsed_words) != len(metadata_by_order):
        flags.append(
            {
                "kind": "word_count_mismatch",
                "count_mfa_words": len(parsed_words),
                "count_transcript_words": len(metadata_by_order),
                "note": "MFA word count does not match transcript-owned word count.",
            }
        )

    for index, interval in enumerate(parsed_words):
        extra = metadata_by_order[index] if index < len(metadata_by_order) else {}
        transcript_word = str(extra.get("word", interval["word"]))
        mfa_word = str(interval["word"])
        words.append(
            {
                "word_index": parse_int(extra.get("word_index"), index),
                "word": transcript_word or mfa_word,
                "mfa_word": mfa_word,
                "start": float(interval["start"]),
                "end": float(interval["end"]),
                "phrase_index": parse_int(extra.get("phrase_index"), -1),
                "sentence_index": parse_int(extra.get("sentence_index"), -1),
                "source": "mfa_word_interval",
                "approval": {
                    "status": "draft_auto",
                    "reviewed": False,
                },
            }
        )
        if normalize_word(transcript_word) != normalize_word(mfa_word):
            flags.append(
                {
                    "kind": "word_text_mismatch",
                    "word_index": parse_int(extra.get("word_index"), index),
                    "mfa_word": mfa_word,
                    "transcript_word": transcript_word,
                    "note": "Transcript and MFA word text differ at this position.",
                }
            )

    if words:
        final_word_end = float(words[-1]["end"])
        trailing_audio = max(0.0, audio_duration_seconds - final_word_end)
        if trailing_audio > 0.45:
            flags.append(
                {
                    "kind": "mfa_timeline_truncated_tail",
                    "seconds": round(trailing_audio, 3),
                    "note": "The final MFA word ends well before the WAV ends.",
                }
            )

    for word in words:
        duration = float(word["end"]) - float(word["start"])
        word_text = str(word.get("word", ""))
        letters = max(1, len(normalize_word(word_text)))
        if letters >= 6 and duration < 0.18:
            flags.append(
                {
                    "kind": "implausibly_short_word",
                    "word_index": parse_int(word.get("word_index"), -1),
                    "word": word_text,
                    "duration_seconds": round(duration, 3),
                    "note": "A relatively long word was aligned to an implausibly short duration.",
                }
            )

    return words, flags


def planned_rows_by_word(planned_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in planned_rows:
        word_index = parse_int(row.get("word_index"), -1)
        if word_index < 0:
            continue
        grouped.setdefault(word_index, []).append(row)
    return grouped


def phones_by_word(
    parsed: dict[str, list[dict[str, object]]],
    words: list[dict[str, object]],
) -> dict[int, list[dict[str, object]]]:
    result: dict[int, list[dict[str, object]]] = {parse_int(word["word_index"]): [] for word in words}
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
            word_index = parse_int(word["word_index"])
            result[word_index].append(
                {
                    "index_in_word": len(result[word_index]),
                    "phone": label,
                    "phone_base": strip_phone_stress(label),
                    "start": float(phone["start"]),
                    "end": float(phone["end"]),
                    "word_index": word_index,
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

    if phones_for_word and source_phone_index < 0 and not source_phone:
        return phones_for_word[0], "synthetic_word_fallback"

    indexed_matches = [
        phone
        for phone in phones_for_word
        if parse_int(phone.get("source_phone_index"), parse_int(phone.get("index_in_word"), -1)) == source_phone_index
    ]
    if indexed_matches:
        phone = indexed_matches[0]
        if phone["phone_base"] == source_phone or not source_phone:
            return phone, "mfa_exact_source_phone_index"
        return phone, "mfa_source_phone_index_label_mismatch"

    if 0 <= source_phone_index < len(phones_for_word):
        phone = phones_for_word[source_phone_index]
        if phone["phone_base"] == source_phone:
            return phone, "mfa_exact_source_phone_index"
        return phone, "mfa_source_phone_index_label_mismatch"

    matches = [phone for phone in phones_for_word if phone["phone_base"] == source_phone]
    if matches:
        return matches[0], "mfa_source_phone_label_fallback"

    return None, "no_phone_evidence"


def synthesize_pseudo_phones(
    words: list[dict[str, object]],
    planned_by_word: dict[int, list[dict[str, str]]],
    phones_for_words: dict[int, list[dict[str, object]]],
) -> dict[int, list[dict[str, object]]]:
    augmented = {word_index: [dict(phone) for phone in phones] for word_index, phones in phones_for_words.items()}

    for word in words:
        word_index = parse_int(word["word_index"], -1)
        actual = augmented.get(word_index, [])
        non_spn = [phone for phone in actual if str(phone.get("phone_base", "")).lower() != "spn"]
        planned = planned_by_word.get(word_index, [])
        max_planned_index = max((parse_int(row.get("source_phone_index"), -1) for row in planned), default=-1)
        force_synthetic = False
        if non_spn and max_planned_index >= len(non_spn):
            force_synthetic = True
        if any(not strip_phone_stress(row.get("source_phone", "")) for row in planned):
            force_synthetic = True
        if non_spn and not force_synthetic:
            continue

        if not planned:
            continue

        units: list[dict[str, object]] = []
        seen_keys: set[tuple[int, str]] = set()
        for order, row in enumerate(planned):
            source_index = parse_int(row.get("source_phone_index"), order)
            source_phone = strip_phone_stress(row.get("source_phone", "")) or f"pseudo_{order}"
            key = (source_index, source_phone)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            units.append(
                {
                    "index_in_word": len(units),
                    "phone": source_phone,
                    "phone_base": source_phone,
                    "source_phone_index": source_index,
                }
            )

        if not units:
            continue

        start = float(word["start"])
        end = float(word["end"])
        span = max(0.06, end - start)
        step = span / len(units)
        synthetic: list[dict[str, object]] = []
        for index, unit in enumerate(units):
            unit_start = start + step * index
            unit_end = end if index == len(units) - 1 else start + step * (index + 1)
            synthetic.append(
                {
                    "index_in_word": index,
                    "phone": unit["phone"],
                    "phone_base": unit["phone_base"],
                    "start": unit_start,
                    "end": unit_end,
                    "word_index": word_index,
                    "word": str(word["word"]),
                    "synthetic": True,
                }
            )
        augmented[word_index] = synthetic

    return augmented


def build_visemes(
    planned_rows: list[dict[str, str]],
    committed_rows: list[dict[str, str]],
    phones_for_words: dict[int, list[dict[str, object]]],
    words_by_index: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    visemes: list[dict[str, object]] = []
    all_flags: list[dict[str, object]] = []
    previous_center: float | None = None
    previous_start: float | None = None
    committed_by_index = {parse_int(row.get("index")): row for row in committed_rows}

    for row in planned_rows:
        word_index = parse_int(row.get("word_index"), -1)
        word = words_by_index.get(word_index)
        source, alignment_reason = locate_phone_interval(row, phones_for_words.get(word_index, []))
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

        if word is not None:
            word_start = float(word["start"])
            word_end = float(word["end"])
            start = max(start, word_start)
            end = min(end, word_end)
            if end <= start:
                slot = min(0.03, max(0.0, word_end - word_start))
                if slot <= 1e-6:
                    start = word_start
                    end = word_end
                else:
                    end = word_end
                    start = max(word_start, end - slot)
                center = (start + end) * 0.5
                alignment_reason = f"{alignment_reason}_word_clamp"
            else:
                center = (start + end) * 0.5

        viseme = {
            "index": parse_int(row.get("index")),
            "id": row.get("pose", ""),
            "pose": row.get("pose", ""),
            "start": start,
            "center": center,
            "end": end,
            "word": row.get("word", ""),
            "word_index": word_index,
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
                "status": "draft_auto",
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


def build_mfa_phone_events(
    phones_for_words: dict[int, list[dict[str, object]]],
    words_by_index: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create first-class gold phone rows directly from MFA TextGrid phone intervals.

    These rows intentionally use CMU/ARPAbet phone labels from MFA exactly as emitted
    by the aligner, including stress digits such as AH0/IY1 where present.  The
    reviewer/exporter may still call the file visemes.csv for compatibility, but
    each row is a phone interval, not a reduced Offgrid viseme.
    """
    events: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []

    for word_index in sorted(phones_for_words):
        word = words_by_index.get(word_index, {})
        phones = phones_for_words.get(word_index, [])
        if not phones:
            if word:
                flags.append(
                    {
                        "kind": "word_has_no_mfa_phones",
                        "word_index": word_index,
                        "word": word.get("word", ""),
                        "note": "MFA produced a word interval but no contained phone intervals.",
                    }
                )
            continue

        for phone in phones:
            label = str(phone.get("phone", phone.get("text", ""))).strip()
            if not label:
                continue
            start = float(phone.get("start", 0.0))
            end = float(phone.get("end", start))
            if end <= start:
                flags.append(
                    {
                        "kind": "non_positive_phone_duration",
                        "word_index": word_index,
                        "phone": label,
                        "start": start,
                        "end": end,
                        "note": "MFA phone interval had non-positive duration and was skipped.",
                    }
                )
                continue

            index_in_word = parse_int(phone.get("index_in_word"), 0)
            events.append(
                {
                    "index": len(events),
                    "id": label,
                    "pose": label,
                    "phone": label,
                    "start": start,
                    "center": (start + end) * 0.5,
                    "end": end,
                    "word": word.get("word", phone.get("word", "")),
                    "mfa_word": word.get("mfa_word", phone.get("word", "")),
                    "word_index": word_index,
                    "phrase_index": parse_int(word.get("phrase_index"), -1),
                    "sentence_index": parse_int(word.get("sentence_index"), -1),
                    "strength": 1.0,
                    "commit_reason": "mfa_phone_interval",
                    "alignment_reason": "mfa_textgrid_phone",
                    "source_phone": label,
                    "source_phone_index": index_in_word,
                    "source_phone_class": "cmu_arpa",
                    "mapped_to_observed_speech": True,
                    "approval": {
                        "status": "approved_gold",
                        "reviewed": True,
                    },
                    "notes": [],
                    "flags": [],
                }
            )

    return events, flags


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


def analyze_wave_activity(
    wav_path,
    leading_pad_seconds: float = 0.05,
    trailing_pad_seconds: float = 0.16,
) -> dict[str, object]:
    sample_rate, mono = read_wav_mono_samples(wav_path)
    if sample_rate <= 0 or not mono:
        return {
            "sample_rate": sample_rate,
            "duration_seconds": 0.0,
            "window_seconds": 0.02,
            "starts": [],
            "energies": [],
            "active_flags": [],
            "active_threshold": 0.0,
            "strong_threshold": 0.0,
            "leading_pad_seconds": leading_pad_seconds,
            "trailing_pad_seconds": trailing_pad_seconds,
        }

    duration_seconds = len(mono) / sample_rate
    window_seconds = 0.02
    window_samples = max(1, int(round(sample_rate * window_seconds)))
    energies: list[float] = []
    starts: list[float] = []
    for start_index in range(0, len(mono), window_samples):
        chunk = mono[start_index : start_index + window_samples]
        if not chunk:
            continue
        mean_square = sum(sample * sample for sample in chunk) / len(chunk)
        energies.append(math.sqrt(mean_square))
        starts.append(start_index / sample_rate)

    if not energies:
        return {
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "window_seconds": window_seconds,
            "starts": [],
            "energies": [],
            "active_flags": [],
            "active_threshold": 0.0,
            "strong_threshold": 0.0,
            "leading_pad_seconds": leading_pad_seconds,
            "trailing_pad_seconds": trailing_pad_seconds,
        }

    sorted_energies = sorted(energies)
    noise_floor = sorted_energies[max(0, int(len(sorted_energies) * 0.2) - 1)]
    peak_energy = sorted_energies[-1]
    active_threshold = max(noise_floor * 3.5, peak_energy * 0.12, 0.008)
    strong_threshold = max(active_threshold * 1.8, peak_energy * 0.22)
    active_flags = [energy >= active_threshold for energy in energies]

    return {
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "window_seconds": window_seconds,
        "starts": starts,
        "energies": energies,
        "active_flags": active_flags,
        "active_threshold": active_threshold,
        "strong_threshold": strong_threshold,
        "leading_pad_seconds": leading_pad_seconds,
        "trailing_pad_seconds": trailing_pad_seconds,
    }


def detect_wave_speech_regions(
    wav_path,
    max_gap_seconds: float,
    leading_pad_seconds: float = 0.05,
    trailing_pad_seconds: float = 0.16,
) -> list[tuple[float, float]]:
    activity = analyze_wave_activity(
        wav_path,
        leading_pad_seconds=leading_pad_seconds,
        trailing_pad_seconds=trailing_pad_seconds,
    )
    starts = list(activity["starts"])
    energies = list(activity["energies"])
    active_flags = list(activity["active_flags"])
    strong_threshold = float(activity["strong_threshold"])
    duration_seconds = float(activity["duration_seconds"])
    window_seconds = float(activity["window_seconds"])
    if not starts or not energies:
        return []

    min_region_seconds = 0.10
    end_hold_windows = max(1, int(round(0.20 / window_seconds)))
    min_region_windows = max(1, int(round(min_region_seconds / window_seconds)))

    raw_regions: list[tuple[float, float]] = []
    in_region = False
    region_start = 0.0
    silent_windows = 0
    active_windows = 0

    for index, (energy, start) in enumerate(zip(energies, starts)):
        is_active = bool(active_flags[index])
        is_strong = energy >= strong_threshold
        window_end = min(start + window_seconds, duration_seconds)

        if in_region:
            if is_active:
                active_windows += 1
                silent_windows = 0
            else:
                silent_windows += 1
                if silent_windows >= end_hold_windows:
                    region_end = max(region_start + min_region_seconds, start - (silent_windows - 1) * window_seconds)
                    raw_regions.append((region_start, region_end))
                    in_region = False
                    silent_windows = 0
                    active_windows = 0
        else:
            if is_strong or is_active:
                in_region = True
                region_start = start
                active_windows = 1
                silent_windows = 0

    if in_region:
        raw_regions.append((region_start, duration_seconds))

    merged: list[tuple[float, float]] = []
    for start, end in raw_regions:
        if end - start < min_region_windows * window_seconds:
            continue
        start = max(0.0, start - leading_pad_seconds)
        end = min(duration_seconds, end + trailing_pad_seconds)
        effective_merge_gap = max(max_gap_seconds, 0.24)
        if not merged or start > merged[-1][1] + effective_merge_gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def detect_wave_silence_spans(
    wav_path,
    min_silence_seconds: float = 0.10,
) -> list[tuple[float, float]]:
    activity = analyze_wave_activity(wav_path, leading_pad_seconds=0.0, trailing_pad_seconds=0.0)
    starts = list(activity["starts"])
    active_flags = list(activity["active_flags"])
    duration_seconds = float(activity["duration_seconds"])
    window_seconds = float(activity["window_seconds"])
    if not starts or not active_flags:
        return []

    spans: list[tuple[float, float]] = []
    silence_start: float | None = None
    min_windows = max(1, int(round(min_silence_seconds / window_seconds)))
    silence_windows = 0

    for index, start in enumerate(starts):
        is_active = bool(active_flags[index])
        if is_active:
            if silence_start is not None and silence_windows >= min_windows:
                spans.append((silence_start, start))
            silence_start = None
            silence_windows = 0
            continue
        if silence_start is None:
            silence_start = start
        silence_windows += 1

    if silence_start is not None:
        silence_end = duration_seconds
        if silence_windows >= min_windows:
            spans.append((silence_start, silence_end))

    return spans


def build_pause_boundaries(
    words: list[dict[str, object]],
    phone_intervals: list[dict[str, object]],
    wave_silence_spans: list[tuple[float, float]],
    max_gap_seconds: float,
) -> list[dict[str, object]]:
    boundaries: list[dict[str, object]] = []
    for index in range(len(words) - 1):
        current_word = words[index]
        next_word = words[index + 1]
        current_end = float(current_word.get("end", 0.0))
        next_start = float(next_word.get("start", current_end))
        mark = str(current_word.get("boundary_mark_after", ""))
        marks = list(current_word.get("boundary_marks_after", []))

        direct_word_gap = max(0.0, next_start - current_end)
        center = (current_end + next_start) * 0.5

        phone_gap = 0.0
        for left_phone, right_phone in zip(phone_intervals, phone_intervals[1:]):
            left_end = float(left_phone.get("end", 0.0))
            right_start = float(right_phone.get("start", left_end))
            if left_end - 1e-6 <= center <= right_start + 1e-6:
                phone_gap = max(phone_gap, right_start - left_end)

        wave_silence_overlap = 0.0
        for silence_start, silence_end in wave_silence_spans:
            wave_silence_overlap = max(
                wave_silence_overlap,
                overlap_seconds(current_end, next_start, silence_start, silence_end),
            )

        has_hard_mark = any(boundary_mark in HARD_BOUNDARY_PUNCTUATION for boundary_mark in marks)
        has_comma_mark = "," in marks
        acoustic_gap = max(direct_word_gap, phone_gap, wave_silence_overlap)

        split_applied = False
        split_reason = "none"
        if has_hard_mark and acoustic_gap >= 0.080:
            split_applied = True
            split_reason = "hard_punctuation_with_lull"
        elif has_comma_mark and acoustic_gap >= max_gap_seconds:
            split_applied = True
            split_reason = "comma_with_lull"
        elif acoustic_gap >= max(0.240, max_gap_seconds * 1.5):
            split_applied = True
            split_reason = "strong_acoustic_gap"

        pause_class = "none"
        if split_applied:
            pause_class = "region_break"
        elif has_comma_mark and acoustic_gap >= 0.020:
            pause_class = "soft_pause"
        elif has_hard_mark and acoustic_gap >= 0.020:
            pause_class = "hard_pause"

        boundaries.append(
            {
                "word_index": parse_int(current_word.get("word_index"), index),
                "word": current_word.get("word", ""),
                "next_word_index": parse_int(next_word.get("word_index"), index + 1),
                "next_word": next_word.get("word", ""),
                "mark": mark,
                "mark_sequence": "".join(marks),
                "pause_class": pause_class,
                "split_applied": split_applied,
                "split_reason": split_reason,
                "direct_word_gap_seconds": direct_word_gap,
                "phone_gap_seconds": phone_gap,
                "wave_silence_seconds": wave_silence_overlap,
                "acoustic_gap_seconds": acoustic_gap,
            }
        )
    return boundaries


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _best_region_index_for_span(start: float, end: float, regions: list[dict[str, object]]) -> int:
    best_index = -1
    best_overlap = 0.0
    center = (start + end) * 0.5
    for region in regions:
        ri = parse_int(region.get("index"), -1)
        rs = float(region.get("start", 0.0))
        re = float(region.get("end", 0.0))
        overlap = overlap_seconds(start, end, rs, re)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = ri
        elif best_overlap <= 1e-9 and rs - 1e-6 <= center <= re + 1e-6:
            best_index = ri
    return best_index


def build_speech_regions(
    words: list[dict[str, object]],
    pause_boundaries: list[dict[str, object]],
    max_gap_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build speech regions from MFA word timing plus acoustic pause evidence.

    Region ownership is word-aligned because pause/resume is the primary truth:
    starts align to MFA word starts, ends align to MFA word ends, and splits are
    created only where punctuation and/or acoustic lull evidence makes the pause
    perceptually meaningful.
    """
    regions: list[dict[str, object]] = []
    flags: list[dict[str, object]] = []
    if not words:
        return regions, flags

    if not pause_boundaries:
        if words:
            regions.append(
                {
                    "index": 0,
                    "start": float(words[0]["start"]),
                    "end": float(words[-1]["end"]),
                    "word_start_index": parse_int(words[0].get("word_index"), -1),
                    "word_end_index": parse_int(words[-1].get("word_index"), -1),
                    "sentence_start_index": parse_int(words[0].get("sentence_index"), -1),
                    "sentence_end_index": parse_int(words[-1].get("sentence_index"), -1),
                    "source": "mfa_word_span_fallback_no_pause_boundaries",
                    "approval": {"status": "approved_gold", "reviewed": True},
                }
            )
            flags.append(
                {
                    "kind": "speech_regions_from_word_fallback",
                    "note": "No pause boundaries were available, so one word-span speech region was exported.",
                }
            )
        return regions, flags

    split_indices = {
        parse_int(boundary.get("word_index"), -1)
        for boundary in pause_boundaries
        if bool(boundary.get("split_applied"))
    }
    split_gaps = [
        float(boundary.get("acoustic_gap_seconds", 0.0))
        for boundary in pause_boundaries
        if bool(boundary.get("split_applied"))
    ]

    region_start_word = 0
    for word_list_index, word in enumerate(words):
        boundary_after = parse_int(word.get("word_index"), -1)
        should_split_after = boundary_after in split_indices or word_list_index == len(words) - 1
        if not should_split_after:
            continue

        region_words = words[region_start_word : word_list_index + 1]
        start = float(region_words[0].get("start", 0.0))
        end = float(region_words[-1].get("end", start))
        word_start_index = parse_int(region_words[0].get("word_index"), -1)
        word_end_index = parse_int(region_words[-1].get("word_index"), -1)
        sentence_start_index = parse_int(region_words[0].get("sentence_index"), -1)
        sentence_end_index = parse_int(region_words[-1].get("sentence_index"), -1)

        regions.append(
            {
                "index": len(regions),
                "start": start,
                "end": end,
                "word_start_index": word_start_index,
                "word_end_index": word_end_index,
                "sentence_start_index": sentence_start_index,
                "sentence_end_index": sentence_end_index,
                "source": "mfa_word_aligned_pause_regions",
                "merge_gap_seconds": max_gap_seconds,
                "approval": {"status": "approved_gold", "reviewed": True},
            }
        )
        region_start_word = word_list_index + 1

    if split_gaps:
        flags.append(
            {
                "kind": "speech_regions_split_on_pause_boundaries",
                "region_count": len(regions),
                "gap_threshold_seconds": round(max_gap_seconds, 3),
                "largest_split_gap_seconds": round(max(split_gaps), 3),
                "note": "Speech regions were split using MFA word timing plus punctuation/acoustic pause evidence.",
            }
        )

    return regions, flags


def assign_speech_region_indices(
    words: list[dict[str, object]],
    phone_events: list[dict[str, object]],
    regions: list[dict[str, object]],
) -> None:
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        word["speech_region_index"] = _best_region_index_for_span(start, end, regions)
    for phone in phone_events:
        start = float(phone.get("start", 0.0))
        end = float(phone.get("end", start))
        region_index = _best_region_index_for_span(start, end, regions)
        phone["speech_region_index"] = region_index
        # Keep phrase_index backward-compatible for older review/grading paths that
        # used phrase_index as the only visible grouping column.
        phone["phrase_index"] = region_index


def annotate_pause_boundaries_with_regions(
    pause_boundaries: list[dict[str, object]],
    words: list[dict[str, object]],
) -> None:
    words_by_index = {parse_int(word.get("word_index"), -1): word for word in words}
    for boundary in pause_boundaries:
        word_index = parse_int(boundary.get("word_index"), -1)
        next_word_index = parse_int(boundary.get("next_word_index"), -1)
        boundary["speech_region_index_before"] = parse_int(words_by_index.get(word_index, {}).get("speech_region_index"), -1)
        boundary["speech_region_index_after"] = parse_int(words_by_index.get(next_word_index, {}).get("speech_region_index"), -1)


def build_word_heads(visemes: list[dict[str, object]]) -> list[dict[str, object]]:
    heads: list[dict[str, object]] = []
    seen: set[int] = set()
    for viseme in visemes:
        word_index = parse_int(viseme.get("word_index"), -1)
        if word_index < 0 or word_index in seen:
            continue
        seen.add(word_index)
        heads.append(
            {
                "word_index": word_index,
                "word": viseme.get("word", ""),
                "start": float(viseme.get("start", 0.0)),
                "end": float(viseme.get("end", 0.0)),
                "pose": viseme.get("pose", ""),
                "phrase_index": parse_int(viseme.get("phrase_index"), -1),
                "sentence_index": parse_int(viseme.get("sentence_index"), -1),
                "source_phone": viseme.get("source_phone", ""),
                "source_phone_index": parse_int(viseme.get("source_phone_index"), -1),
                "alignment_reason": viseme.get("alignment_reason", ""),
                "approval": {
                    "status": "draft_auto",
                    "reviewed": False,
                },
            }
        )
    return heads


def build_summary(
    words: list[dict[str, object]],
    speech_regions: list[dict[str, object]],
    pause_boundaries: list[dict[str, object]],
    word_heads: list[dict[str, object]],
    visemes: list[dict[str, object]],
    flags: list[dict[str, object]],
) -> dict[str, object]:
    reasons = Counter(v["commit_reason"] for v in visemes)
    alignments = Counter(v["alignment_reason"] for v in visemes)
    return {
        "word_count": len(words),
        "speech_region_count": len(speech_regions),
        "pause_boundary_count": len(pause_boundaries),
        "pause_region_break_count": sum(1 for boundary in pause_boundaries if bool(boundary.get("split_applied"))),
        "word_head_count": len(word_heads),
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
        f"speech regions: {summary['speech_region_count']}",
        f"pause boundaries: {summary['pause_boundary_count']}",
        f"pause-driven region breaks: {summary['pause_region_break_count']}",
        f"word heads: {summary['word_head_count']}",
        f"dense visemes: {summary['viseme_count']}",
        f"draft flags: {summary['flag_count']}",
        "",
        "review order:",
        "1. Confirm speech region starts and ends against the WAV.",
        "   Region starts should align to MFA word starts. Region ends should align to the last word before a meaningful lull.",
        "2. Confirm the first visible viseme for each word against the WAV.",
        "3. Review dense intra-word visemes, prioritizing flagged regions.",
        "",
        "approval workflow:",
        "- review_layers.speech_regions.status -> reviewed_boundary when speech/pause boundaries are trusted",
        "- review_layers.word_heads.status -> reviewed_boundary when word-entry timing is trusted",
        "- review_layers.dense_visemes.status -> reviewed_dense when dense timing is trusted",
        "- approval.status -> approved_gold only when the full case is ready for grading",
    ]
    write_text(gold_draft_dir(case_id) / "review_notes.txt", "\n".join(lines) + "\n")


def build_case_annotation(
    case_id: str,
    buffer_ms: int,
    chunk_ms: int,
    mfa_num_jobs: int,
    speech_merge_gap_ms: int,
) -> dict[str, object]:
    root = repo_root()
    transcript = read_text(root / "inputs" / "transcripts" / f"{case_id}.txt")
    wav_path = root / "inputs" / "wav" / f"{case_id}.wav"
    audio = wav_metadata(wav_path)
    case_dir = offline_gold_case_dir(case_id)
    planned_rows = read_csv_rows(case_dir / "planned.csv")
    committed_rows = read_csv_rows(case_dir / "committed.csv")
    parsed = parse_long_textgrid(mfa_textgrid_path(case_id))
    metadata = word_metadata_by_index(planned_rows)
    parsed_words = non_empty_word_intervals(parsed)
    words, word_flags = build_gold_words(parsed_words, metadata, float(audio["duration_sec"]))
    boundary_flags = attach_transcript_boundaries(words, transcript)
    per_word_phones = phones_by_word(parsed, words)
    words_by_index = {parse_int(word["word_index"]): word for word in words}
    visemes, viseme_flags = build_mfa_phone_events(per_word_phones, words_by_index)
    all_non_silence_phones = non_silence_phone_intervals(parsed)
    wave_silence_spans = detect_wave_silence_spans(wav_path, min_silence_seconds=max(0.08, speech_merge_gap_ms / 1000.0))
    pause_boundaries = build_pause_boundaries(words, all_non_silence_phones, wave_silence_spans, max(0.0, speech_merge_gap_ms / 1000.0))
    speech_regions, speech_flags = build_speech_regions(words, pause_boundaries, max(0.0, speech_merge_gap_ms / 1000.0))
    assign_speech_region_indices(words, visemes, speech_regions)
    annotate_pause_boundaries_with_regions(pause_boundaries, words)
    word_heads = build_word_heads(visemes)
    flags = word_flags + boundary_flags + speech_flags + viseme_flags

    payload = {
        "schema_version": 1,
        "case_id": case_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator": {
            "kind": "liplab_gold_mfa_phase1",
            "buffer_ms": buffer_ms,
            "chunk_ms": chunk_ms,
            "mfa_num_jobs": mfa_num_jobs,
            "speech_merge_gap_ms": speech_merge_gap_ms,
            "source_run_dir": case_dir.relative_to(root).as_posix(),
            "mfa_align_dir": mfa_align_root().relative_to(root).as_posix(),
        },
        "approval": {
            "status": "approved_gold",
            "reviewed": True,
            "reviewer": "auto_mfa_phone_export",
            "reviewed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "review_layers": {
            "speech_regions": {
                "status": "reviewed_boundary",
                "reviewed": True,
                "reviewer": "auto_mfa_phone_export",
            },
            "word_heads": {
                "status": "reviewed_boundary",
                "reviewed": True,
                "reviewer": "auto_mfa_phone_export",
            },
            "dense_visemes": {
                "status": "reviewed_dense",
                "reviewed": True,
                "reviewer": "auto_mfa_phone_export",
            },
        },
        "transcript": transcript,
        "audio": audio,
        "speech_regions": speech_regions,
        "pause_boundaries": pause_boundaries,
        "gold_words": words,
        "word_heads": word_heads,
        "mfa_words": words,
        "mfa_phones": [phone for group in per_word_phones.values() for phone in group],
        "mfa_non_silence_phone_intervals": all_non_silence_phones,
        "wave_silence_spans": [
            {"start": start, "end": end, "duration_seconds": end - start}
            for start, end in wave_silence_spans
        ],
        "planned_visemes": build_planned_visemes(planned_rows),
        "visemes": visemes,
        "flags": flags,
        "notes": [],
    }
    payload["summary"] = build_summary(words, speech_regions, pause_boundaries, word_heads, visemes, flags)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate layered draft gold annotations from MFA timing plus transcript-owned viseme identity."
    )
    parser.add_argument("--buffer-ms", type=int, default=600000)
    parser.add_argument("--chunk-ms", type=int, default=600000)
    parser.add_argument("--mfa-num-jobs", type=int, default=4)
    parser.add_argument("--speech-merge-gap-ms", type=int, default=150)
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
        payload = build_case_annotation(
            case_id,
            args.buffer_ms,
            args.chunk_ms,
            args.mfa_num_jobs,
            args.speech_merge_gap_ms,
        )
        write_json(gold_draft_dir(case_id) / "draft.annotation.json", payload)
        write_review_notes(case_id, payload)
        print(
            f"{case_id}: speech={payload['summary']['speech_region_count']} "
            f"word_heads={payload['summary']['word_head_count']} "
            f"visemes={payload['summary']['viseme_count']} "
            f"flags={payload['summary']['flag_count']}"
        )

    print(f"Wrote draft gold annotation package(s) for {len(cases)} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
