"""Measure non-silence prosodic restarts at comma-list word boundaries.

This is an offline diagnostic: MFA word times are used only to score acoustic
features already emitted by the streaming detector.  It does not provide
runtime timing or modify scheduling.
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"
GOLD_ROOT = ROOT / "inputs" / "gold"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze_boundary(
    frames: list[dict[str, str]], word_start: float
) -> dict[str, float] | None:
    # A restart is a local energy valley followed causally by renewed energy
    # and spectral change. The deliberately broad MFA-relative window is only
    # for evaluating whether such a cue exists near the reference boundary.
    window = [
        row
        for row in frames
        if word_start - 0.180 <= number(row, "center") <= word_start + 0.160
    ]
    if not window:
        return None

    valley_pool = [row for row in window if row.get("local_rms_valley") == "1"]
    if not valley_pool:
        valley_pool = window

    best: dict[str, float] | None = None
    for valley in valley_pool:
        valley_time = number(valley, "center")
        valley_rms = max(number(valley, "rms"), 1.0e-7)
        after = [
            row
            for row in window
            if valley_time + 0.010 <= number(row, "center") <= valley_time + 0.180
        ]
        if not after:
            continue
        peak = max(after, key=lambda row: number(row, "flux"))
        rebound = max(after, key=lambda row: number(row, "rms"))
        rebound_ratio = number(rebound, "rms") / valley_rms
        flux = number(peak, "flux")
        restart_frames = [
            row
            for row in after
            if number(row, "rms") >= valley_rms * 3.0
            and (
                number(row, "flux") >= 0.040
                or row.get("strong_onset") == "1"
                or number(row, "evidence") >= 0.230
            )
        ]
        restart = restart_frames[0] if restart_frames else peak
        # Prefer a deep/recovering valley and a spectral reattack, while
        # mildly penalizing landmarks far from the scored word start.
        score = math.log2(max(rebound_ratio, 1.0)) + 4.0 * flux
        anchor = number(restart, "center")
        score -= 2.0 * abs(anchor - word_start)
        candidate = {
            "valley_time": valley_time,
            "valley_rms": valley_rms,
            "valley_rms_norm": number(valley, "rms_norm"),
            "anchor_time": anchor,
            "anchor_error_ms": (anchor - word_start) * 1000.0,
            "restart_delay_ms": (anchor - valley_time) * 1000.0,
            "rebound_ratio": rebound_ratio,
            "flux": flux,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def main() -> int:
    rows: list[dict[str, object]] = []
    for case_dir in sorted(RUN_ROOT.glob("case_*")):
        boundaries = read_csv(case_dir / "planned_boundaries.csv")
        frames = read_csv(case_dir / "occupancy_frames.csv")
        words = {
            int(number(row, "word_index", -1)): row
            for row in read_csv(GOLD_ROOT / case_dir.name / "words.csv")
        }
        if not frames or not words:
            continue
        for boundary in boundaries:
            if boundary.get("pause_class") != "soft_list_pause":
                continue
            previous = words.get(int(number(boundary, "word_index", -1)))
            following = words.get(int(number(boundary, "next_word_index", -1)))
            if previous is None or following is None:
                continue
            start = number(following, "start")
            result = analyze_boundary(frames, start)
            if result is None:
                continue
            result.update(
                {
                    "case": case_dir.name.split("_")[1],
                    "previous_word": previous.get("word", ""),
                    "next_word": following.get("word", ""),
                    "gold_gap_ms": (start - number(previous, "end")) * 1000.0,
                }
            )
            rows.append(result)

    if not rows:
        raise SystemExit("No detailed focused-alignment output found")

    errors = [abs(float(row["anchor_error_ms"])) for row in rows]
    ratios = [float(row["rebound_ratio"]) for row in rows]
    fluxes = [float(row["flux"]) for row in rows]
    detectable = [
        row
        for row in rows
        if float(row["rebound_ratio"]) >= 3.0 and float(row["flux"]) >= 0.06
    ]
    detectable_errors = [abs(float(row["anchor_error_ms"])) for row in detectable]
    positive_gaps = [row for row in rows if float(row["gold_gap_ms"]) >= 30.0]
    positive_detected = [row for row in detectable if float(row["gold_gap_ms"]) >= 30.0]
    continuous = [row for row in rows if float(row["gold_gap_ms"]) < 30.0]

    def group_line(label: str, group: list[dict[str, object]]) -> str:
        return (
            f"{label}: n={len(group)}, "
            f"delay median={statistics.median(float(row['restart_delay_ms']) for row in group):.0f}ms, "
            f"valley_norm median={statistics.median(float(row['valley_rms_norm']) for row in group):.3f}, "
            f"rebound median={statistics.median(float(row['rebound_ratio']) for row in group):.1f}x"
        )

    print(f"soft-list boundaries: {len(rows)}")
    print(
        "restart cue (rebound >=3x, flux >=0.06): "
        f"{len(detectable)}/{len(rows)} ({len(detectable) / len(rows):.1%})"
    )
    print(
        "reference gaps >=30ms detected: "
        f"{len(positive_detected)}/{len(positive_gaps)} "
        f"({len(positive_detected) / max(len(positive_gaps), 1):.1%})"
    )
    print(
        "candidate anchor abs error: "
        f"median={statistics.median(errors):.1f}ms "
        f"p90={percentile(errors, 0.90):.1f}ms"
    )
    print(
        "qualified anchor abs error: "
        f"median={statistics.median(detectable_errors):.1f}ms "
        f"p90={percentile(detectable_errors, 0.90):.1f}ms"
        if detectable_errors
        else "qualified anchor abs error: unavailable"
    )
    print(
        "feature distribution: "
        f"rebound median={statistics.median(ratios):.1f}x, "
        f"flux median={statistics.median(fluxes):.3f}"
    )
    print(group_line("reference gaps >=30ms", positive_gaps))
    print(group_line("reference gaps <30ms", continuous))
    print("pause-worthy cue thresholds:")
    for delay_min in (20.0, 30.0):
        for norm_max in (0.010, 0.020, 0.040):
            predicted = [
                row
                for row in rows
                if float(row["restart_delay_ms"]) >= delay_min
                and float(row["valley_rms_norm"]) <= norm_max
                and float(row["rebound_ratio"]) >= 3.0
                and float(row["flux"]) >= 0.06
            ]
            true_positive = sum(float(row["gold_gap_ms"]) >= 30.0 for row in predicted)
            precision = true_positive / max(len(predicted), 1)
            recall = true_positive / max(len(positive_gaps), 1)
            print(
                f"  delay>={delay_min:.0f}ms valley_norm<={norm_max:.3f}: "
                f"predicted={len(predicted)}, precision={precision:.1%}, recall={recall:.1%}"
            )

    print("\nweakest qualified/missed boundaries:")
    for row in sorted(rows, key=lambda item: float(item["score"]))[:12]:
        print(
            f"  {row['case']} {row['previous_word']}->{row['next_word']}: "
            f"gap={float(row['gold_gap_ms']):.0f}ms "
            f"rebound={float(row['rebound_ratio']):.1f}x "
            f"flux={float(row['flux']):.3f} "
            f"anchor_error={float(row['anchor_error_ms']):+.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
