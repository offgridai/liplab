import csv
import json
import pathlib
import re
import statistics


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"
GOLD_ROOT = ROOT / "inputs" / "gold"
OUT_PATH = ROOT / "outputs" / "analysis" / "streaming_region_failures.md"
DETAIL_PATH = ROOT / "outputs" / "analysis" / "streaming_region_failures.csv"
MODEL_PATH = ROOT / "offgrid_dropin" / "Private" / "Lipsync" / "OffgridAIStreamingRegionModel.inl"
MODEL_REPORT_PATH = ROOT / "outputs" / "streaming_region_model_report.json"
TOLERANCE_SEC = 0.100


def rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def payload(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def model_config() -> tuple[float, int, int]:
    text = MODEL_PATH.read_text(encoding="utf-8")

    def constant(name: str, cast):
        match = re.search(rf"{name}\s*=\s*([0-9.]+)f?;", text)
        if not match:
            raise RuntimeError(f"Missing {name} in {MODEL_PATH}")
        return cast(match.group(1))

    return (
        constant("StreamingRegionSpeechThreshold", float),
        constant("StreamingRegionMinimumSpeechFrames", int),
        constant("StreamingRegionMinimumPauseFrames", int),
    )


def gaps(regions: list[dict[str, str]]) -> list[tuple[float, float]]:
    return [
        (value(regions[index], "end"), value(regions[index + 1], "start"))
        for index in range(len(regions) - 1)
    ]


def matched_pairs(
    predicted: list[tuple[float, float]], reference: list[tuple[float, float]]
) -> list[tuple[int, int]]:
    n, m = len(predicted), len(reference)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            compatible = all(
                abs(predicted[i - 1][edge] - reference[j - 1][edge]) <= TOLERANCE_SEC
                for edge in (0, 1)
            )
            dp[i][j] = max(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1] + int(compatible),
            )
    result: list[tuple[int, int]] = []
    i, j = n, m
    while i and j:
        compatible = all(
            abs(predicted[i - 1][edge] - reference[j - 1][edge]) <= TOLERANCE_SEC
            for edge in (0, 1)
        )
        if compatible and dp[i][j] == dp[i - 1][j - 1] + 1:
            result.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(result))


def nearby_unmatched_pairs(
    predicted: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    unmatched_predicted: set[int],
    unmatched_reference: set[int],
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for predicted_index in unmatched_predicted:
        predicted_start, predicted_end = predicted[predicted_index]
        predicted_center = (predicted_start + predicted_end) * 0.5
        for reference_index in unmatched_reference:
            reference_start, reference_end = reference[reference_index]
            reference_center = (reference_start + reference_end) * 0.5
            overlap = min(predicted_end, reference_end) - max(predicted_start, reference_start)
            center_distance = abs(predicted_center - reference_center)
            if overlap > 0.0 or center_distance <= 0.300:
                candidates.append((center_distance, predicted_index, reference_index))
    result: list[tuple[int, int]] = []
    used_predicted: set[int] = set()
    used_reference: set[int] = set()
    for _, predicted_index, reference_index in sorted(candidates):
        if predicted_index in used_predicted or reference_index in used_reference:
            continue
        result.append((predicted_index, reference_index))
        used_predicted.add(predicted_index)
        used_reference.add(reference_index)
    return sorted(result)


def occupancy_stats(
    frames: list[dict[str, str]], start: float, end: float, speech_threshold: float
) -> dict[str, float]:
    selected = [
        frame
        for frame in frames
        if value(frame, "center") >= start and value(frame, "center") <= end
    ]
    if not selected:
        return {}
    rms = [value(frame, "rms_norm") for frame in selected]
    evidence = [value(frame, "evidence") for frame in selected]
    learned = [value(frame, "learned_speech_probability") for frame in selected]
    longest_quiet_count = 0
    longest_quiet_start = 0.0
    longest_quiet_end = 0.0
    quiet_count = 0
    quiet_start = 0.0
    for frame in selected:
        if value(frame, "learned_speech_probability") < speech_threshold:
            if quiet_count == 0:
                quiet_start = value(frame, "start")
            quiet_count += 1
            if quiet_count > longest_quiet_count:
                longest_quiet_count = quiet_count
                longest_quiet_start = quiet_start
                longest_quiet_end = value(frame, "end")
        else:
            quiet_count = 0
    return {
        "rms_median": statistics.median(rms),
        "rms_min": min(rms),
        "evidence_median": statistics.median(evidence),
        "learned_median": statistics.median(learned),
        "learned_min": min(learned),
        "learned_max": max(learned),
        "speech_fraction": sum(value(frame, "learned_speech") >= 0.5 for frame in selected)
        / len(selected),
        "longest_quiet_frames": longest_quiet_count,
        "longest_quiet_start": longest_quiet_start,
        "longest_quiet_end": longest_quiet_end,
    }


def nearest_raw_candidate(
    candidates: list[dict[str, str]], start: float, end: float
) -> dict[str, str]:
    center = (start + end) * 0.5
    ranked: list[tuple[float, dict[str, str]]] = []
    for candidate in candidates:
        candidate_start = value(candidate, "gap_start")
        candidate_end = value(candidate, "gap_end")
        overlap = min(end, candidate_end) - max(start, candidate_start)
        center_distance = abs(center - (candidate_start + candidate_end) * 0.5)
        if overlap > 0.0 or center_distance <= 0.300:
            ranked.append((center_distance - max(overlap, 0.0), candidate))
    return min(ranked, default=(0.0, {}), key=lambda item: item[0])[1]


def raw_candidate_summary(candidate: dict[str, str]) -> str:
    if not candidate:
        return "none"
    disposition = "bridged" if int(value(candidate, "bridged")) else "split"
    return f"{disposition}:{candidate.get('decision_class', 'unknown')}"


def learned_failure_class(
    kind: str, stats: dict[str, float], minimum_pause_frames: int
) -> str:
    quiet_frames = int(stats.get("longest_quiet_frames", 0))
    if kind == "missed":
        if quiet_frames < minimum_pause_frames:
            return "classifier did not sustain quiet long enough"
        return "decoder did not expose a qualifying quiet run as a region gap"
    if kind == "invented":
        if quiet_frames >= minimum_pause_frames:
            return "classifier sustained a false quiet run long enough to split"
        return "predicted gap is not explained by frames inside its reported interval"
    return "classifier transition is present but one or both decoded edges are shifted"


def boundary_context(
    boundary_rows: list[dict[str, str]], region_index: int
) -> tuple[str, str, str, str]:
    for row in boundary_rows:
        if int(value(row, "speech_region_index_before", -1)) == region_index and int(
            value(row, "speech_region_index_after", -1)
        ) == region_index + 1:
            return (
                row.get("word", "?"),
                row.get("next_word", "?"),
                row.get("mark_sequence", ""),
                row.get("split_reason", ""),
            )
    return "?", "?", "", ""


def inferred_cause(
    kind: str,
    duration: float,
    stats: dict[str, float],
    punctuation: str,
    overlap_word: str,
    gold_acoustic_gap: float = 0.0,
) -> str:
    speech_fraction = stats.get("speech_fraction", 0.0)
    rms_median = stats.get("rms_median", 0.0)
    if kind == "missed":
        if speech_fraction >= 0.50:
            return "gap contains breath/noisy low-energy audio classified as speech"
        if duration <= 0.180:
            return "short gap is near endpoint/reopen hysteresis duration"
        if rms_median >= 0.20:
            return "relative energy does not collapse enough for the close gate"
        if punctuation == ",":
            return "comma/list lull lacks enough acoustic close confidence"
        return "low-energy gap exists but close evidence is fragmented"
    if overlap_word:
        return f"word-internal trough or stop closure inside '{overlap_word}'"
    if punctuation == "," and gold_acoustic_gap < 0.120:
        return "comma lull is below gold's 120 ms split threshold but runtime accepted it"
    if not punctuation and gold_acoustic_gap < 0.240:
        return "unpunctuated lull is below gold's 240 ms strong-gap policy but runtime accepted it"
    if duration <= 0.180:
        return "short inter-word trough accepted more aggressively than gold"
    if speech_fraction <= 0.10:
        return "clean acoustic lull retained by gold's punctuation-conditioned policy"
    return "transient low-energy/noisy interval caused a false close and reopen"


def shifted_cause(
    predicted: tuple[float, float], reference: tuple[float, float]
) -> str:
    close_delta = predicted[0] - reference[0]
    resume_delta = predicted[1] - reference[1]
    descriptions: list[str] = []
    if close_delta <= -TOLERANCE_SEC:
        descriptions.append("close is early: trailing weak speech was assigned to the pause")
    elif close_delta >= TOLERANCE_SEC:
        descriptions.append("close is late: hangover or breath/noise kept speech open")
    if resume_delta <= -TOLERANCE_SEC:
        descriptions.append("resume is early: breath/noise or weak pre-onset energy reopened speech")
    elif resume_delta >= TOLERANCE_SEC:
        descriptions.append("resume is late: weak onset did not clear the reopen gate promptly")
    return "; ".join(descriptions) or "both edges narrowly miss the paired-edge tolerance"


def endpoint_cause(kind: str, delta: float) -> str:
    if kind == "initial":
        return (
            "initial open is early: leading noise/breath was accepted as speech"
            if delta < 0.0
            else "initial open is late: weak initial speech did not clear the onset gate"
        )
    return (
        "final close is early: weak trailing speech was discarded"
        if delta < 0.0
        else "final close is late: endpoint hangover or trailing noise extended speech"
    )


def overlapping_gold_word(
    words: list[dict[str, str]], start: float, end: float
) -> str:
    midpoint = (start + end) * 0.5
    for word in words:
        if value(word, "start") < midpoint < value(word, "end"):
            return word.get("word", "")
    return ""


def nearest_boundary(
    boundaries: list[dict[str, str]],
    words: list[dict[str, str]],
    center: float,
) -> dict[str, str]:
    words_by_index = {int(value(word, "word_index", -1)): word for word in words}
    candidates: list[tuple[float, dict[str, str]]] = []
    for boundary in boundaries:
        left = words_by_index.get(int(value(boundary, "word_index", -1)))
        right = words_by_index.get(int(value(boundary, "next_word_index", -1)))
        if left is None or right is None:
            continue
        boundary_center = (value(left, "end") + value(right, "start")) * 0.5
        candidates.append((abs(boundary_center - center), boundary))
    return min(candidates, default=(0.0, {}), key=lambda item: item[0])[1]


def main() -> int:
    speech_threshold, minimum_speech_frames, minimum_pause_frames = model_config()
    model_report = payload(MODEL_REPORT_PATH)
    run_case_count = sum(path.is_dir() for path in RUN_ROOT.iterdir())
    trained_case_count = int(model_report.get("case_count", 0))
    cases: list[dict] = []
    cause_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    cross_path_counts: dict[str, int] = {}
    decoder_reason_counts: dict[str, dict[str, int]] = {}
    detail_rows: list[dict[str, str | int | float]] = []
    for case_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        gold_dir = GOLD_ROOT / case_dir.name
        gold_regions = rows(gold_dir / "speech.csv")
        predicted_regions = rows(case_dir / "refined_speech_regions.csv")
        grade = payload(case_dir / "streaming_region_boundary_grade.json")
        if not gold_regions or not predicted_regions or not grade.get("available"):
            continue

        reference_gaps = gaps(gold_regions)
        predicted_gaps = gaps(predicted_regions)
        pairs = matched_pairs(predicted_gaps, reference_gaps)
        matched_predicted = {pair[0] for pair in pairs}
        matched_reference = {pair[1] for pair in pairs}
        nearby_pairs = nearby_unmatched_pairs(
            predicted_gaps,
            reference_gaps,
            set(range(len(predicted_gaps))) - matched_predicted,
            set(range(len(reference_gaps))) - matched_reference,
        )
        nearby_predicted = {pair[0] for pair in nearby_pairs}
        nearby_reference = {pair[1] for pair in nearby_pairs}
        initial_failure = not grade.get("initial_open_matched", False)
        final_failure = not grade.get("final_close_matched", False)
        if (
            len(pairs) == len(reference_gaps) == len(predicted_gaps)
            and not initial_failure
            and not final_failure
        ):
            continue

        transcript_path = ROOT / "inputs" / "transcripts" / f"{case_dir.name}.txt"
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        frames = rows(case_dir / "occupancy_frames.csv")
        raw_candidates = rows(case_dir / "gap_candidates.csv")
        refined_candidates = rows(case_dir / "refined_gap_candidates.csv")
        boundaries = rows(gold_dir / "boundaries.csv")
        words = rows(gold_dir / "words.csv")
        failures: list[dict] = []
        for predicted_index in range(len(predicted_gaps)):
            reason = (
                refined_candidates[predicted_index].get("close_reason", "unknown")
                if predicted_index < len(refined_candidates)
                else "unknown"
            )
            status = (
                "matched" if predicted_index in matched_predicted
                else "shifted" if predicted_index in nearby_predicted
                else "invented"
            )
            counts = decoder_reason_counts.setdefault(reason, {})
            counts[status] = counts.get(status, 0) + 1
        initial_delta = value(predicted_regions[0], "start") - value(gold_regions[0], "start")
        final_delta = value(predicted_regions[-1], "end") - value(gold_regions[-1], "end")
        initial_cause = endpoint_cause("initial", initial_delta) if initial_failure else ""
        final_cause = endpoint_cause("final", final_delta) if final_failure else ""
        if initial_cause:
            cause_counts[initial_cause] = cause_counts.get(initial_cause, 0) + 1
        if final_cause:
            cause_counts[final_cause] = cause_counts.get(final_cause, 0) + 1

        for predicted_index, reference_index in nearby_pairs:
            predicted = predicted_gaps[predicted_index]
            reference = reference_gaps[reference_index]
            before, after, punctuation, reason = boundary_context(boundaries, reference_index)
            stats = occupancy_stats(
                frames,
                min(predicted[0], reference[0]),
                max(predicted[1], reference[1]),
                speech_threshold,
            )
            cause = shifted_cause(predicted, reference)
            mechanism = learned_failure_class("shifted", stats, minimum_pause_frames)
            raw_summary = raw_candidate_summary(
                nearest_raw_candidate(raw_candidates, reference[0], reference[1])
            )
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
            cross_path_counts[f"shifted / raw={raw_summary}"] = (
                cross_path_counts.get(f"shifted / raw={raw_summary}", 0) + 1
            )
            failures.append(
                {
                    "kind": "shifted",
                    "start": reference[0],
                    "end": reference[1],
                    "predicted_start": predicted[0],
                    "predicted_end": predicted[1],
                    "context": f"{before}{punctuation} | {after}",
                    "punctuation": punctuation,
                    "gold_reason": reason,
                    "stats": stats,
                    "cause": cause,
                    "mechanism": mechanism,
                    "raw_candidate": raw_summary,
                }
            )

        for index, (start, end) in enumerate(reference_gaps):
            if index in matched_reference or index in nearby_reference:
                continue
            before, after, punctuation, reason = boundary_context(boundaries, index)
            stats = occupancy_stats(frames, start, end, speech_threshold)
            cause = inferred_cause("missed", end - start, stats, punctuation, "")
            mechanism = learned_failure_class("missed", stats, minimum_pause_frames)
            raw_summary = raw_candidate_summary(
                nearest_raw_candidate(raw_candidates, start, end)
            )
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
            cross_path_counts[f"missed / raw={raw_summary}"] = (
                cross_path_counts.get(f"missed / raw={raw_summary}", 0) + 1
            )
            failures.append(
                {
                    "kind": "missed",
                    "start": start,
                    "end": end,
                    "context": f"{before}{punctuation} | {after}",
                    "punctuation": punctuation,
                    "gold_reason": reason,
                    "stats": stats,
                    "cause": cause,
                    "mechanism": mechanism,
                    "raw_candidate": raw_summary,
                }
            )

        for index, (start, end) in enumerate(predicted_gaps):
            if index in matched_predicted or index in nearby_predicted:
                continue
            stats = occupancy_stats(frames, start, end, speech_threshold)
            overlap_word = overlapping_gold_word(words, start, end)
            boundary = nearest_boundary(boundaries, words, (start + end) * 0.5)
            punctuation = boundary.get("mark_sequence", "")
            gold_acoustic_gap = value(boundary, "acoustic_gap_seconds")
            cause = inferred_cause(
                "invented",
                end - start,
                stats,
                punctuation,
                overlap_word,
                gold_acoustic_gap,
            )
            mechanism = learned_failure_class("invented", stats, minimum_pause_frames)
            raw_summary = raw_candidate_summary(
                nearest_raw_candidate(raw_candidates, start, end)
            )
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
            cross_path_counts[f"invented / raw={raw_summary}"] = (
                cross_path_counts.get(f"invented / raw={raw_summary}", 0) + 1
            )
            boundary_context_text = (
                f"{boundary.get('word', '?')}{punctuation} | "
                f"{boundary.get('next_word', '?')} gold_gap={gold_acoustic_gap * 1000:.0f}ms"
                if boundary
                else "between words"
            )
            failures.append(
                {
                    "kind": "invented",
                    "start": start,
                    "end": end,
                    "context": (
                        f"inside {overlap_word}"
                        if overlap_word
                        else boundary_context_text
                    ),
                    "punctuation": punctuation,
                    "gold_reason": "",
                    "stats": stats,
                    "cause": cause,
                    "mechanism": mechanism,
                    "raw_candidate": raw_summary,
                }
            )

        for failure in failures:
            stats = failure["stats"]
            detail_rows.append(
                {
                    "case": case_dir.name,
                    "kind": failure["kind"],
                    "gold_start": failure["start"],
                    "gold_end": failure["end"],
                    "runtime_start": failure.get("predicted_start", ""),
                    "runtime_end": failure.get("predicted_end", ""),
                    "context": failure["context"],
                    "punctuation": failure.get("punctuation", ""),
                    "learned_mechanism": failure["mechanism"],
                    "longest_quiet_frames": int(stats.get("longest_quiet_frames", 0)),
                    "longest_quiet_ms": int(stats.get("longest_quiet_frames", 0)) * 10,
                    "learned_probability_min": stats.get("learned_min", 0.0),
                    "learned_probability_median": stats.get("learned_median", 0.0),
                    "learned_speech_fraction": stats.get("speech_fraction", 0.0),
                    "raw_candidate": failure["raw_candidate"],
                }
            )

        cases.append(
            {
                "case": case_dir.name,
                "transcript": transcript,
                "gold_count": len(gold_regions),
                "predicted_count": len(predicted_regions),
                "matched_pairs": len(pairs),
                "initial_error_ms": grade.get("initial_open_error_ms", 0.0),
                "final_error_ms": grade.get("final_close_error_ms", 0.0),
                "initial_failure": initial_failure,
                "final_failure": final_failure,
                "initial_delta_ms": initial_delta * 1000.0,
                "final_delta_ms": final_delta * 1000.0,
                "initial_cause": initial_cause,
                "final_cause": final_cause,
                "failures": failures,
            }
        )

    lines = [
        "# Streaming Speech-Region Failure Audit",
        "",
        f"Tolerance: {TOLERANCE_SEC * 1000:.0f} ms per close/resume edge.",
        f"Active learned decoder: speech probability >= {speech_threshold:.2f}; "
        f"open after {minimum_speech_frames} speech frames; close after "
        f"{minimum_pause_frames} quiet frames ({minimum_pause_frames * 10} ms).",
        f"Cases with any region failure: {len(cases)}.",
        f"Model training cases: {trained_case_count}; current run cases: {run_case_count}.",
        "",
        "## Measured learned-model failure mechanisms",
        "",
    ]
    if trained_case_count and trained_case_count != run_case_count:
        lines.extend(
            [
                "**Warning:** the checked-in learned model is stale relative to the current corpus; "
                f"it has not seen {run_case_count - trained_case_count} current cases.",
                "",
            ]
        )
    for cause, count in sorted(mechanism_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {count}: {cause}")
    lines.extend(["", "## Learned model vs legacy raw-gap detector", ""])
    for cause, count in sorted(cross_path_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {count}: {cause}")
    lines.extend(["", "## Accepted split outcomes by decoder reason", ""])
    for reason, counts in sorted(decoder_reason_counts.items()):
        total = sum(counts.values())
        lines.append(
            f"- {reason}: total={total}, matched={counts.get('matched', 0)}, "
            f"shifted={counts.get('shifted', 0)}, invented={counts.get('invented', 0)}"
        )
    lines.extend(["", "## Acoustic interpretation", ""])
    for cause, count in sorted(cause_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {count}: {cause}")
    lines.extend(["", "## Case review", ""])

    for case in cases:
        lines.extend(
            [
                f"### {case['case']}",
                "",
                f"Transcript: {case['transcript']}",
                "",
                f"Regions: gold={case['gold_count']} runtime={case['predicted_count']} "
                f"matched_gaps={case['matched_pairs']}. Initial error={case['initial_error_ms']:.1f} ms; "
                f"final error={case['final_error_ms']:.1f} ms.",
                "",
            ]
        )
        if case["initial_failure"]:
            lines.append(
                f"- Initial speech onset is outside tolerance "
                f"(signed error={case['initial_delta_ms']:+.1f} ms). "
                f"Theory: {case['initial_cause']}."
            )
        if case["final_failure"]:
            lines.append(
                f"- Final speech close is outside tolerance "
                f"(signed error={case['final_delta_ms']:+.1f} ms). "
                f"Theory: {case['final_cause']}."
            )
        for failure in case["failures"]:
            stats = failure["stats"]
            observed = ""
            if failure["kind"] == "shifted":
                observed = (
                    f" runtime={failure['predicted_start']:.3f}-"
                    f"{failure['predicted_end']:.3f}s"
                )
            lines.append(
                f"- {failure['kind'].upper()} {failure['start']:.3f}-{failure['end']:.3f}s "
                f"({(failure['end'] - failure['start']) * 1000:.0f} ms),{observed} "
                f"{failure['context']}; "
                f"rms_med={stats.get('rms_median', 0.0):.3f}, "
                f"speech_fraction={stats.get('speech_fraction', 0.0):.2f}. "
                f"quiet_run={int(stats.get('longest_quiet_frames', 0)) * 10}ms, "
                f"raw={failure['raw_candidate']}. "
                f"Mechanism: {failure['mechanism']}. "
                f"Acoustic interpretation: {failure['cause']}."
            )
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    with DETAIL_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(detail_rows[0]) if detail_rows else ["case", "kind"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(
        f"Wrote {OUT_PATH} and {DETAIL_PATH} for {len(cases)} failing case(s), "
        f"{len(detail_rows)} failed gap(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
