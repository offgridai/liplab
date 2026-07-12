import math
import statistics
from bisect import bisect_right
from typing import Any


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, str], key: str, default: int = -1) -> int:
    try:
        value = row.get(key, "")
        return int(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _true_plan_time(
    targets: list[dict[str, Any]],
    audio_time: float,
    start_index: int = 0,
    end_index: int | None = None,
) -> float:
    """Map MFA audio time back onto the local text-plan axis for grading only."""
    segment = targets[start_index:end_index]
    gold = [target["gold_center"] for target in segment]
    index = bisect_right(gold, audio_time)
    if index <= 0:
        return segment[0]["prior_center"] + (audio_time - gold[0])
    if index >= len(segment):
        return segment[-1]["prior_center"] + (audio_time - gold[-1])
    left = segment[index - 1]
    right = segment[index]
    alpha = (audio_time - left["gold_center"]) / max(
        right["gold_center"] - left["gold_center"], 0.001
    )
    return left["prior_center"] + alpha * (
        right["prior_center"] - left["prior_center"]
    )


def _nearest_gold_index(
    targets: list[dict[str, Any]],
    audio_time: float,
    start_index: int = 0,
    end_index: int | None = None,
) -> int:
    segment_end = len(targets) if end_index is None else end_index
    return min(
        range(start_index, segment_end),
        key=lambda index: abs(targets[index]["gold_center"] - audio_time),
    )


def _parse_targets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    targets = []
    for row in rows:
        if _int(row, "has_gold", 0) == 0:
            continue
        targets.append(
            {
                "target_index": _int(row, "target_index"),
                "expected_phone_index": _int(row, "expected_phone_index"),
                "global_phone_index": _int(row, "global_phone_index"),
                "word_index": _int(row, "word_index"),
                "speech_region_index": _int(row, "speech_region_index"),
                "prior_center": _float(row, "prior_center"),
                "gold_center": _float(row, "gold_center"),
                "stress": _int(row, "stress", 0),
                "word": row.get("word", ""),
                "phone": row.get("phone", ""),
                "family": row.get("vowel_family", ""),
            }
        )
    return sorted(targets, key=lambda target: target["prior_center"])


def _parse_observations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    observations = []
    for row in rows:
        peak = _float(row, "peak_sec", -1.0)
        decision = _float(row, "decision_sec", -1.0)
        if peak < 0.0 or decision < peak:
            continue
        observations.append(
            {
                "peak": peak,
                "decision": decision,
                "prominence": _clamp(_float(row, "prominence"), 0.0, 1.05),
            }
        )
    return sorted(observations, key=lambda observation: observation["decision"])


def _parse_speech_regions(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    regions = []
    for row in rows:
        start = _float(row, "start", -1.0)
        end = _float(row, "end", -1.0)
        if start < 0.0 or end <= start:
            continue
        regions.append((start, end))
    return sorted(regions, key=lambda region: region[0])


def _derive_transcript_segments(
    targets: list[dict[str, Any]],
    transcript_landmark_rows: list[dict[str, str]],
) -> list[tuple[int, int]]:
    if not targets:
        return []

    segment_starts = {0}
    for index in range(1, len(targets)):
        previous = targets[index - 1]["speech_region_index"]
        current = targets[index]["speech_region_index"]
        if previous >= 0 and current >= 0 and previous != current:
            segment_starts.add(index)

    word_to_target_index: dict[int, int] = {}
    for index, target in enumerate(targets):
        word_to_target_index.setdefault(target["word_index"], index)

    for row in transcript_landmark_rows:
        if row.get("type", "") != "pause_lull":
            continue
        next_word_index = _int(row, "next_word_index", -1)
        if next_word_index < 0:
            continue
        next_target_index = word_to_target_index.get(next_word_index)
        if next_target_index is not None and 0 < next_target_index < len(targets):
            segment_starts.add(next_target_index)

    ordered = sorted(segment_starts)
    return list(zip(ordered, ordered[1:] + [len(targets)]))


def _segment_audio_window(
    targets: list[dict[str, Any]],
    start_index: int,
    end_index: int,
) -> tuple[float, float]:
    start = targets[start_index]["gold_center"]
    end = targets[end_index - 1]["gold_center"]
    if end_index - start_index >= 2:
        left_gap = targets[start_index + 1]["gold_center"] - targets[start_index]["gold_center"]
        right_gap = targets[end_index - 1]["gold_center"] - targets[end_index - 2]["gold_center"]
    else:
        left_gap = right_gap = 0.120
    lead = _clamp(left_gap * 0.50, 0.045, 0.160)
    tail = _clamp(right_gap * 0.55, 0.050, 0.180)
    return start - lead, end + tail


def _overlap_seconds(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _pair_cost(
    targets: list[dict[str, Any]],
    segment: tuple[int, int],
    region: tuple[float, float],
) -> float:
    segment_window = _segment_audio_window(targets, segment[0], segment[1])
    segment_center = (segment_window[0] + segment_window[1]) * 0.5
    region_center = (region[0] + region[1]) * 0.5
    overlap = _overlap_seconds(segment_window, region)
    segment_duration = max(segment_window[1] - segment_window[0], 0.050)
    region_duration = max(region[1] - region[0], 0.050)
    overlap_ratio = overlap / max(min(segment_duration, region_duration), 0.050)

    center_cost = abs(segment_center - region_center) / 0.220
    start_cost = abs(segment_window[0] - region[0]) / 0.180
    end_cost = abs(segment_window[1] - region[1]) / 0.220
    overlap_bonus = overlap_ratio * 1.35
    no_overlap_penalty = 1.25 if overlap <= 0.0 else 0.0
    duration_cost = abs(segment_duration - region_duration) / max(segment_duration, region_duration, 0.080)
    return center_cost * 0.55 + start_cost * 0.20 + end_cost * 0.25 + duration_cost * 0.28 + no_overlap_penalty - overlap_bonus


def _pair_segments(
    targets: list[dict[str, Any]],
    transcript_segments: list[tuple[int, int]],
    speech_regions: list[tuple[float, float]],
) -> tuple[list[tuple[int, int, tuple[float, float]]], bool, int, int]:
    if not transcript_segments or not speech_regions:
        return [], True, len(transcript_segments), len(speech_regions)

    transcript_count = len(transcript_segments)
    region_count = len(speech_regions)
    skip_segment_cost = 1.10
    skip_region_cost = 1.05

    dp = [[math.inf] * (region_count + 1) for _ in range(transcript_count + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [
        [None] * (region_count + 1) for _ in range(transcript_count + 1)
    ]
    dp[0][0] = 0.0

    for i in range(transcript_count + 1):
        for j in range(region_count + 1):
            base = dp[i][j]
            if not math.isfinite(base):
                continue
            if i < transcript_count and base + skip_segment_cost < dp[i + 1][j]:
                dp[i + 1][j] = base + skip_segment_cost
                parent[i + 1][j] = (i, j, "skip_segment")
            if j < region_count and base + skip_region_cost < dp[i][j + 1]:
                dp[i][j + 1] = base + skip_region_cost
                parent[i][j + 1] = (i, j, "skip_region")
            if i < transcript_count and j < region_count:
                pair = base + _pair_cost(targets, transcript_segments[i], speech_regions[j])
                if pair < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = pair
                    parent[i + 1][j + 1] = (i, j, "pair")

    pairs: list[tuple[int, int, tuple[float, float]]] = []
    i = transcript_count
    j = region_count
    skipped_segments = 0
    skipped_regions = 0
    while i > 0 or j > 0:
        step = parent[i][j]
        if step is None:
            break
        prev_i, prev_j, action = step
        if action == "pair":
            segment = transcript_segments[prev_i]
            region = speech_regions[prev_j]
            pair_cost = _pair_cost(targets, segment, region)
            if pair_cost <= 1.85:
                pairs.append((segment[0], segment[1], region))
            else:
                skipped_segments += 1
                skipped_regions += 1
        elif action == "skip_segment":
            skipped_segments += 1
        elif action == "skip_region":
            skipped_regions += 1
        i, j = prev_i, prev_j

    pairs.reverse()
    ambiguous = not pairs
    return pairs, ambiguous, skipped_segments, skipped_regions


def _parse_landmark_anchors(
    transcript_landmark_rows: list[dict[str, str]],
    landmark_observation_rows: list[dict[str, str]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transcript_rows_by_target = {
        _int(row, "target_index"): row for row in transcript_landmark_rows
    }
    anchors: list[dict[str, Any]] = []
    allowed_types = {"round", "w", "fv", "chjjsh", "mbp"}

    for row in landmark_observation_rows:
        if row.get("source", "") != "conditioned_phone":
            continue
        landmark_type = row.get("type", "")
        if landmark_type not in allowed_types:
            continue
        confidence = _float(row, "confidence", 0.0)
        if confidence < 0.72:
            continue
        transcript_target_index = _int(row, "target_index")
        transcript_row = transcript_rows_by_target.get(transcript_target_index)
        if transcript_row is None:
            continue
        word_index = _int(transcript_row, "word_index")
        phone_index = _int(transcript_row, "global_phone_index")
        speech_region_index = _int(transcript_row, "speech_region_index")
        decision_sec = _float(row, "decision_sec", -1.0)
        if decision_sec < 0.0:
            continue

        candidates = [
            (index, target)
            for index, target in enumerate(targets)
            if target["speech_region_index"] == speech_region_index
        ]
        if not candidates:
            continue
        candidate_index = min(
            [index for index, target in candidates],
            key=lambda index: (
                abs(targets[index]["word_index"] - word_index),
                abs(targets[index]["global_phone_index"] - phone_index),
                abs(targets[index]["prior_center"] - decision_sec),
            ),
        )
        anchors.append(
            {
                "time": decision_sec,
                "target_list_index": candidate_index,
                "strength": confidence,
                "type": landmark_type,
            }
        )
    return sorted(anchors, key=lambda anchor: anchor["time"])


def _anchor_penalty(
    observation_peak: float,
    candidate_index: int,
    anchors: list[dict[str, Any]],
) -> float:
    penalty = 0.0
    for anchor in anchors:
        time_delta = abs(observation_peak - anchor["time"])
        if time_delta > 0.180:
            continue
        temporal_weight = 1.0 - (time_delta / 0.180)
        index_delta = abs(candidate_index - anchor["target_list_index"])
        penalty += index_delta * temporal_weight * anchor["strength"] * 0.26
    return penalty


def _beam_plan_estimate(beam: dict[str, Any], observation_peak: float) -> float:
    return beam["anchor_plan"] + (
        observation_peak - beam["anchor_audio"]
    ) / max(beam["rate"], 0.001)


def _anchor_plan_blend(
    estimated_plan: float,
    observation_peak: float,
    anchors: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> float:
    best_anchor = None
    best_time_delta = 1e9
    for anchor in anchors:
        time_delta = abs(observation_peak - anchor["time"])
        if time_delta < best_time_delta:
            best_time_delta = time_delta
            best_anchor = anchor
    if best_anchor is None or best_time_delta > 0.080:
        return estimated_plan

    anchor_plan = targets[best_anchor["target_list_index"]]["prior_center"]
    blend = _clamp((1.0 - best_time_delta / 0.080) * best_anchor["strength"] * 0.38, 0.0, 0.38)
    return estimated_plan * (1.0 - blend) + anchor_plan * blend


def _recent_sequence_cost(
    beam: dict[str, Any],
    candidate_index: int,
    observation_peak: float,
    targets: list[dict[str, Any]],
) -> float:
    history = beam.get("history", [])
    if len(history) < 2:
        return 0.0

    recent = history[-3:]
    previous_plan = [entry["plan"] for entry in recent] + [targets[candidate_index]["prior_center"]]
    previous_audio = [entry["audio"] for entry in recent] + [observation_peak]
    if len(previous_plan) < 3:
        return 0.0

    plan_intervals = [
        previous_plan[index + 1] - previous_plan[index]
        for index in range(len(previous_plan) - 1)
    ]
    audio_intervals = [
        previous_audio[index + 1] - previous_audio[index]
        for index in range(len(previous_audio) - 1)
    ]
    if any(interval <= 0.0 for interval in plan_intervals) or any(interval <= 0.0 for interval in audio_intervals):
        return 0.0

    rate = _clamp(beam["rate"], 0.62, 1.55)
    interval_residuals = [
        abs(audio - (plan * rate)) / max(plan * rate, 0.050)
        for plan, audio in zip(plan_intervals, audio_intervals)
    ]
    mean_plan_interval = _mean(plan_intervals)
    mean_audio_interval = _mean(audio_intervals)
    normalized_plan = [interval / max(mean_plan_interval, 0.001) for interval in plan_intervals]
    normalized_audio = [interval / max(mean_audio_interval, 0.001) for interval in audio_intervals]
    rhythm_shape = [
        abs(audio - plan)
        for plan, audio in zip(normalized_plan, normalized_audio)
    ]

    sequence_cost = _mean(interval_residuals) * 0.18 + _mean(rhythm_shape) * 0.28

    if len(plan_intervals) >= 2:
        latest_plan = plan_intervals[-1]
        prior_plan = plan_intervals[-2]
        latest_audio = audio_intervals[-1]
        prior_audio = audio_intervals[-2]
        latest_plan_ratio = latest_plan / max(prior_plan, 0.001)
        latest_audio_ratio = latest_audio / max(prior_audio, 0.001)
        sequence_cost += abs(latest_audio_ratio - latest_plan_ratio) * 0.12

    return sequence_cost


def _neighbor_bonus(
    beam: dict[str, Any],
    candidate_index: int,
    anchors: list[dict[str, Any]],
    observation_peak: float,
) -> float:
    history = beam.get("history", [])
    if not history:
        return 0.0
    previous_target_index = history[-1]["target_index"]
    bonus = 0.0
    for anchor in anchors:
        time_delta = abs(observation_peak - anchor["time"])
        if time_delta > 0.120:
            continue
        if abs(anchor["target_list_index"] - candidate_index) <= 1 and abs(anchor["target_list_index"] - previous_target_index) <= 2:
            bonus += (1.0 - time_delta / 0.120) * anchor["strength"] * 0.06
    return bonus


def _anchor_candidate_window(
    anchors: list[dict[str, Any]],
    observation_peak: float,
    default_start: int,
    default_end: int,
) -> tuple[int, int]:
    best_anchor = None
    best_time_delta = 1e9
    for anchor in anchors:
        time_delta = abs(observation_peak - anchor["time"])
        if time_delta < best_time_delta:
            best_time_delta = time_delta
            best_anchor = anchor
    if best_anchor is None or best_time_delta > 0.115:
        return default_start, default_end

    anchor_index = best_anchor["target_list_index"]
    start = max(default_start, anchor_index - 1)
    end = min(default_end, anchor_index + 4)
    if start >= end:
        return default_start, default_end
    return start, end


def _region_progress_candidate_window(
    targets: list[dict[str, Any]],
    segment_start: int,
    segment_end: int,
    region: tuple[float, float],
    observation_peak: float,
    default_start: int,
    default_end: int,
) -> tuple[int, int]:
    region_duration = max(region[1] - region[0], 0.080)
    segment_start_prior = targets[segment_start]["prior_center"]
    segment_end_prior = targets[segment_end - 1]["prior_center"]
    segment_duration = max(segment_end_prior - segment_start_prior, 0.080)
    progress = _clamp((observation_peak - region[0]) / region_duration, 0.0, 1.0)
    expected_prior = segment_start_prior + progress * segment_duration
    expected_index = min(
        range(segment_start, segment_end),
        key=lambda index: abs(targets[index]["prior_center"] - expected_prior),
    )
    start = max(default_start, expected_index - 2)
    end = min(default_end, expected_index + 4)
    if start >= end:
        return default_start, default_end
    return start, end


def _region_progress_penalty(
    targets: list[dict[str, Any]],
    segment_start: int,
    segment_end: int,
    region: tuple[float, float],
    observation_peak: float,
    candidate_index: int,
) -> float:
    region_duration = max(region[1] - region[0], 0.080)
    segment_start_prior = targets[segment_start]["prior_center"]
    segment_end_prior = targets[segment_end - 1]["prior_center"]
    segment_duration = max(segment_end_prior - segment_start_prior, 0.080)
    progress = _clamp((observation_peak - region[0]) / region_duration, 0.0, 1.0)
    expected_prior = segment_start_prior + progress * segment_duration
    candidate_prior = targets[candidate_index]["prior_center"]
    return abs(candidate_prior - expected_prior) / 0.180


def _simulate_syllable_rebases(
    targets: list[dict[str, Any]],
    paired_segments: list[tuple[int, int, tuple[float, float]]],
    segment_offsets: list[float],
    segment_rates: list[float],
    accepted_anchors: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not paired_segments:
        return {"available": False}, []

    anchors_by_segment: dict[int, list[dict[str, Any]]] = {}
    for anchor in accepted_anchors:
        if anchor["confidence"] < 0.40:
            continue
        anchors_by_segment.setdefault(anchor["speech_region_index"], []).append(anchor)

    projections: list[dict[str, Any]] = []
    baseline_errors: list[float] = []
    rebased_errors: list[float] = []
    improved = 0
    degraded = 0
    unchanged = 0
    used_anchor_count = 0

    for segment_index, (start_index, end_index, region) in enumerate(paired_segments):
        segment_anchors = sorted(
            anchors_by_segment.get(segment_index, []),
            key=lambda row: (row["matched_list_index"], row["peak_sec"]),
        )
        filtered_anchors: list[dict[str, Any]] = []
        last_target = start_index - 99
        for anchor in segment_anchors:
            matched_index = int(anchor["matched_list_index"])
            if matched_index <= start_index:
                continue
            if matched_index - last_target < 2:
                continue
            filtered_anchors.append(anchor)
            last_target = matched_index
        used_anchor_count += len(filtered_anchors)

        active_anchor_plan = targets[start_index]["prior_center"]
        active_anchor_audio = active_anchor_plan + segment_offsets[segment_index]
        active_rate = segment_rates[segment_index]
        next_anchor_cursor = 0

        for target_index in range(start_index, end_index):
            while next_anchor_cursor < len(filtered_anchors):
                next_anchor = filtered_anchors[next_anchor_cursor]
                matched_index = int(next_anchor["matched_list_index"])
                if matched_index > target_index:
                    break
                active_anchor_plan = targets[matched_index]["prior_center"]
                active_anchor_audio = float(next_anchor["peak_sec"])
                active_rate = float(next_anchor["rate"])
                next_anchor_cursor += 1

            target = targets[target_index]
            baseline_audio = target["prior_center"] + segment_offsets[segment_index]
            rebased_audio = active_anchor_audio + (
                target["prior_center"] - active_anchor_plan
            ) * active_rate
            baseline_error_ms = abs(baseline_audio - target["gold_center"]) * 1000.0
            rebased_error_ms = abs(rebased_audio - target["gold_center"]) * 1000.0
            baseline_errors.append(baseline_error_ms)
            rebased_errors.append(rebased_error_ms)
            if rebased_error_ms + 1e-6 < baseline_error_ms:
                improved += 1
            elif baseline_error_ms + 1e-6 < rebased_error_ms:
                degraded += 1
            else:
                unchanged += 1
            projections.append(
                {
                    "speech_region_index": segment_index,
                    "target_index": target["target_index"],
                    "word": target["word"],
                    "phone": target["phone"],
                    "prior_center_sec": round(target["prior_center"], 6),
                    "gold_center_sec": round(target["gold_center"], 6),
                    "baseline_audio_sec": round(baseline_audio, 6),
                    "rebased_audio_sec": round(rebased_audio, 6),
                    "baseline_error_ms": round(baseline_error_ms, 3),
                    "rebased_error_ms": round(rebased_error_ms, 3),
                    "active_anchor_plan_sec": round(active_anchor_plan, 6),
                    "active_anchor_audio_sec": round(active_anchor_audio, 6),
                    "active_rate": round(active_rate, 6),
                }
            )

    report = {
        "available": True,
        "used_anchor_count": used_anchor_count,
        "projection_count": len(projections),
        "baseline_center_error_ms": _mean(baseline_errors),
        "rebased_center_error_ms": _mean(rebased_errors),
        "improvement_rate": improved / max(len(projections), 1),
        "degradation_rate": degraded / max(len(projections), 1),
        "unchanged_rate": unchanged / max(len(projections), 1),
    }
    return report, projections


def compute_prosodic_time_mapper(
    target_rows: list[dict[str, str]],
    observation_rows: list[dict[str, str]],
    speech_rows: list[dict[str, str]],
    transcript_landmark_rows: list[dict[str, str]] | None = None,
    landmark_observation_rows: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Causally align streamed energy pulses to the known nucleus sequence.

    The matcher sees a pulse only at decision_sec, 145 ms after its measured peak
    in the current detector. It may ignore a pulse or assign it to one of the next
    few transcript nuclei, but it can never move backward. Sparse high-confidence
    phone landmarks can bias the candidate set toward the right local neighborhood.
    """
    targets = _parse_targets(target_rows)
    observations = _parse_observations(observation_rows)
    if len(targets) < 2 or not observations:
        return {"available": False}, [], [], []

    transcript_landmark_rows = transcript_landmark_rows or []
    anchors = _parse_landmark_anchors(
        transcript_landmark_rows,
        landmark_observation_rows or [],
        targets,
    )

    speech_regions = _parse_speech_regions(speech_rows)
    transcript_segments = _derive_transcript_segments(targets, transcript_landmark_rows)
    paired_segments, ambiguous_pairing, skipped_segments, skipped_regions = _pair_segments(
        targets,
        transcript_segments,
        speech_regions,
    )
    if ambiguous_pairing or not paired_segments:
        return {
            "available": False,
            "runtime_region_count": len(speech_regions),
            "transcript_region_count": len(transcript_segments),
            "ambiguous_region_pairing": True,
            "skipped_transcript_regions": skipped_segments,
            "skipped_runtime_regions": skipped_regions,
        }, [], [], []

    target_segment = [-1] * len(targets)
    segment_offsets = []
    segment_starts = []
    segment_ends = []
    segment_rates = []
    for segment_index, (start_index, end_index, region) in enumerate(paired_segments):
        segment_starts.append(start_index)
        segment_ends.append(end_index)
        for target_index in range(start_index, end_index):
            target_segment[target_index] = segment_index
        plan_duration = max(
            targets[end_index - 1]["prior_center"] - targets[start_index]["prior_center"],
            0.080,
        )
        region_duration = max(region[1] - region[0], 0.080)
        segment_rates.append(_clamp(region_duration / plan_duration, 0.68, 1.45))
        segment_offsets.append(
            region[0] - max(0.0, targets[start_index]["prior_center"] - 0.060)
        )

    trace: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    accepted_anchors: list[dict[str, Any]] = []
    accepted = 0
    high_confidence = 0
    skipped_outside_regions = 0

    def initial_beam(segment_index: int) -> dict[str, Any]:
        start_index = segment_starts[segment_index]
        plan = targets[start_index]["prior_center"]
        audio = plan + segment_offsets[segment_index]
        return {
            "cost": 0.0,
            "next": start_index,
            "anchor_plan": plan,
            "anchor_audio": audio,
            "previous_plan": plan,
            "previous_audio": audio,
            "rate": segment_rates[segment_index],
            "matched_observation": -1,
            "matched_target": -1,
            "match_residual": 0.0,
            "history": [],
        }

    active_segment = 0
    beams = [initial_beam(active_segment)]

    for observation_index, observation in enumerate(observations):
        while (
            active_segment < len(paired_segments)
            and observation["peak"] > paired_segments[active_segment][2][1]
        ):
            active_segment += 1
            if active_segment < len(paired_segments):
                beams = [initial_beam(active_segment)]
        if active_segment >= len(paired_segments):
            break
        region_start, region_end = paired_segments[active_segment][2]
        if observation["peak"] < region_start:
            skipped_outside_regions += 1
            continue
        if observation["peak"] > region_end:
            skipped_outside_regions += 1
            continue
        while (
            active_segment + 1 < len(paired_segments)
            and observation["peak"] >= paired_segments[active_segment + 1][2][0]
        ):
            active_segment += 1
            beams = [initial_beam(active_segment)]
        active_segment_end = segment_ends[active_segment]
        expanded = []
        for beam in beams:
            ignored = dict(beam)
            ignored["cost"] += 0.72 + observation["prominence"] * 0.28
            ignored["matched_observation"] = -1
            ignored["matched_target"] = -1
            ignored["match_residual"] = 0.0
            expanded.append(ignored)

            candidate_start, candidate_end = _anchor_candidate_window(
                anchors,
                observation["peak"],
                beam["next"],
                min(active_segment_end, beam["next"] + 5),
            )
            candidate_start, candidate_end = _region_progress_candidate_window(
                targets,
                segment_starts[active_segment],
                active_segment_end,
                paired_segments[active_segment][2],
                observation["peak"],
                candidate_start,
                candidate_end,
            )
            for candidate_index in range(candidate_start, candidate_end):
                target = targets[candidate_index]
                expected_audio = beam["anchor_audio"] + (
                    target["prior_center"] - beam["anchor_plan"]
                ) * beam["rate"]
                residual = observation["peak"] - expected_audio
                if abs(residual) > 0.330:
                    continue
                expected_prominence = 0.78 if target["stress"] > 0 else 0.48
                prominence_cost = abs(observation["prominence"] - expected_prominence) * 0.25
                skip_count = candidate_index - beam["next"]
                skip_cost = 0.0
                for skipped_index in range(beam["next"], candidate_index):
                    skipped_target = targets[skipped_index]
                    skip_cost += 0.34 + (0.22 if skipped_target["stress"] > 0 else 0.0)
                matched = dict(beam)
                matched["cost"] += (
                    abs(residual) / 0.145
                    + skip_cost
                    + skip_count * 0.18
                    + prominence_cost
                    + _region_progress_penalty(
                        targets,
                        segment_starts[active_segment],
                        active_segment_end,
                        paired_segments[active_segment][2],
                        observation["peak"],
                        candidate_index,
                    ) * 0.16
                    + _anchor_penalty(observation["peak"], candidate_index, anchors)
                    + _recent_sequence_cost(beam, candidate_index, observation["peak"], targets)
                    - _neighbor_bonus(beam, candidate_index, anchors, observation["peak"])
                )
                plan_delta = target["prior_center"] - beam["previous_plan"]
                audio_delta = observation["peak"] - beam["previous_audio"]
                if beam["matched_target"] >= 0 and plan_delta >= 0.070 and audio_delta > 0.0:
                    measured_rate = _clamp(audio_delta / plan_delta, 0.62, 1.55)
                    matched["rate"] = beam["rate"] * 0.60 + measured_rate * 0.40
                matched["next"] = candidate_index + 1
                matched["anchor_plan"] = target["prior_center"]
                matched["anchor_audio"] = observation["peak"]
                matched["previous_plan"] = target["prior_center"]
                matched["previous_audio"] = observation["peak"]
                matched["matched_observation"] = observation_index
                matched["matched_target"] = candidate_index
                matched["match_residual"] = residual
                matched["history"] = (beam.get("history", []) + [
                    {
                        "target_index": candidate_index,
                        "plan": target["prior_center"],
                        "audio": observation["peak"],
                    }
                ])[-4:]
                expanded.append(matched)

        expanded.sort(key=lambda beam: (beam["cost"], -beam["next"]))
        beams = expanded[:20]
        best_beam = beams[0]
        alternatives = [beam for beam in beams[1:] if beam["next"] != best_beam["next"]]
        margin = (alternatives[0]["cost"] - best_beam["cost"]) if alternatives else 1.5
        accept = best_beam["matched_observation"] == observation_index
        margin_confidence = _clamp(1.0 - math.exp(-max(0.0, margin) * 1.25), 0.0, 1.0)
        residual_confidence = 0.0
        if accept:
            residual_confidence = _clamp(1.0 - abs(best_beam["match_residual"]) / 0.220, 0.0, 1.0)
        confidence = margin_confidence * 0.70 + residual_confidence * 0.30
        candidate_index = best_beam["matched_target"] if accept else max(0, best_beam["next"] - 1)
        if accept:
            target = targets[candidate_index]
            accepted += 1
            high_confidence += int(confidence >= 0.40)
            accepted_anchors.append(
                {
                    "observation_index": observation_index,
                    "speech_region_index": active_segment,
                    "matched_list_index": candidate_index,
                    "matched_target_index": target["target_index"],
                    "peak_sec": observation["peak"],
                    "decision_sec": observation["decision"],
                    "confidence": confidence,
                    "rate": best_beam["rate"],
                }
            )

            for future_index in range(
                candidate_index + 1, min(active_segment_end, candidate_index + 5)
            ):
                future = targets[future_index]
                projected_audio = best_beam["anchor_audio"] + (
                    future["prior_center"] - best_beam["anchor_plan"]
                ) * best_beam["rate"]
                baseline_audio = future["prior_center"] + segment_offsets[target_segment[future_index]]
                projections.append(
                    {
                        "observation_index": observation_index,
                        "anchor_target_index": target["target_index"],
                        "future_target_index": future["target_index"],
                        "future_word": future["word"],
                        "future_phone": future["phone"],
                        "confidence": round(confidence, 6),
                        "rate": round(best_beam["rate"], 6),
                        "baseline_audio": round(baseline_audio, 6),
                        "projected_audio": round(projected_audio, 6),
                        "gold_audio": round(future["gold_center"], 6),
                        "baseline_error_ms": round(abs(baseline_audio - future["gold_center"]) * 1000.0, 3),
                        "projected_error_ms": round(abs(projected_audio - future["gold_center"]) * 1000.0, 3),
                    }
                )

        plan_weights = []
        for beam in beams[:8]:
            relative_cost = max(0.0, beam["cost"] - best_beam["cost"])
            weight = math.exp(-relative_cost / 0.55)
            plan_weights.append((weight, _beam_plan_estimate(beam, observation["peak"])))
        weight_total = sum(weight for weight, _ in plan_weights)
        estimated_plan = (
            sum(weight * plan for weight, plan in plan_weights) / weight_total
            if weight_total > 0.0
            else _beam_plan_estimate(best_beam, observation["peak"])
        )
        estimated_plan = _anchor_plan_blend(estimated_plan, observation["peak"], anchors, targets)
        segment_start = segment_starts[active_segment]
        segment_end = segment_ends[active_segment]
        true_plan = _true_plan_time(targets, observation["peak"], segment_start, segment_end)
        predicted_index = min(
            range(segment_start, segment_end),
            key=lambda index: abs(targets[index]["prior_center"] - estimated_plan),
        )
        true_index = _nearest_gold_index(targets, observation["peak"], segment_start, segment_end)
        local_predicted_index = predicted_index - segment_start
        local_true_index = true_index - segment_start
        trace.append(
            {
                "observation_index": observation_index,
                "speech_region_index": active_segment,
                "peak_sec": round(observation["peak"], 6),
                "decision_sec": round(observation["decision"], 6),
                "prominence": round(observation["prominence"], 6),
                "accepted": int(accept),
                "matched_target_index": targets[candidate_index]["target_index"] if accept else -1,
                "matched_word": targets[candidate_index]["word"] if accept else "",
                "matched_phone": targets[candidate_index]["phone"] if accept else "",
                "confidence": round(confidence, 6),
                "rate": round(best_beam["rate"], 6),
                "estimated_plan_sec": round(estimated_plan, 6),
                "true_plan_sec": round(true_plan, 6),
                "progress_error_ms": round(abs(estimated_plan - true_plan) * 1000.0, 3),
                "predicted_syllable_index": local_predicted_index,
                "true_syllable_index": local_true_index,
                "syllable_index_error": abs(local_predicted_index - local_true_index),
            }
        )

    progress_errors = [float(row["progress_error_ms"]) for row in trace]
    high_rows = [row for row in trace if row["accepted"] and row["confidence"] >= 0.40]
    high_errors = [float(row["progress_error_ms"]) for row in high_rows]
    baseline_projection_errors = [float(row["baseline_error_ms"]) for row in projections]
    corrected_projection_errors = [float(row["projected_error_ms"]) for row in projections]
    improved = sum(
        corrected < baseline
        for corrected, baseline in zip(corrected_projection_errors, baseline_projection_errors)
    )
    rebase_report, rebase_projections = _simulate_syllable_rebases(
        targets,
        paired_segments,
        segment_offsets,
        segment_rates,
        accepted_anchors,
    )
    report = {
        "available": True,
        "target_count": len(targets),
        "runtime_region_count": len(speech_regions),
        "transcript_region_count": len(transcript_segments),
        "paired_region_count": len(paired_segments),
        "skipped_transcript_regions": skipped_segments,
        "skipped_runtime_regions": skipped_regions,
        "region_reset_count": len(paired_segments) - 1,
        "observation_count": len(trace),
        "raw_observation_count": len(observations),
        "skipped_outside_regions": skipped_outside_regions,
        "accepted_count": accepted,
        "acceptance_rate": accepted / max(len(trace), 1),
        "high_confidence_count": high_confidence,
        "high_confidence_coverage": high_confidence / max(len(targets), 1),
        "progress_error_mean_ms": _mean(progress_errors),
        "progress_error_median_ms": _median(progress_errors),
        "progress_error_p90_ms": _percentile(progress_errors, 0.90),
        "high_confidence_progress_error_mean_ms": _mean(high_errors),
        "high_confidence_progress_error_median_ms": _median(high_errors),
        "syllable_index_exact_rate": sum(row["syllable_index_error"] == 0 for row in trace) / max(len(trace), 1),
        "syllable_index_within_one_rate": sum(row["syllable_index_error"] <= 1 for row in trace) / max(len(trace), 1),
        "projection_count": len(projections),
        "baseline_future_nucleus_error_ms": _mean(baseline_projection_errors),
        "corrected_future_nucleus_error_ms": _mean(corrected_projection_errors),
        "projection_improvement_rate": improved / max(len(projections), 1),
        "advisory_rebase_anchor_count": int(rebase_report.get("used_anchor_count", 0)),
        "advisory_rebase_target_count": int(rebase_report.get("projection_count", 0)),
        "advisory_rebase_baseline_center_ms": float(rebase_report.get("baseline_center_error_ms", 0.0)),
        "advisory_rebase_center_ms": float(rebase_report.get("rebased_center_error_ms", 0.0)),
        "advisory_rebase_improvement_rate": float(rebase_report.get("improvement_rate", 0.0)),
        "advisory_rebase_degradation_rate": float(rebase_report.get("degradation_rate", 0.0)),
        "anchor_count": len(anchors),
    }
    return report, trace, projections, rebase_projections
