import csv
import pathlib
import statistics


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"
GOLD_ROOT = ROOT / "inputs" / "gold"
TOLERANCE = 0.100


def rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_matches(observations, references):
    n = len(observations)
    m = len(references)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            compatible = all(
                abs(observations[i - 1][edge] - references[j - 1][edge]) <= TOLERANCE
                for edge in (0, 1)
            )
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + int(compatible))
    return dp[n][m]


def matched_pairs(observations, references, tolerance):
    n = len(observations)
    m = len(references)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            compatible = all(abs(observations[i - 1][edge] - references[j - 1][edge]) <= tolerance for edge in (0, 1))
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + int(compatible))
    result = []
    i, j = n, m
    while i and j:
        compatible = all(abs(observations[i - 1][edge] - references[j - 1][edge]) <= tolerance for edge in (0, 1))
        if compatible and dp[i][j] == dp[i - 1][j - 1] + 1:
            result.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(result))


def refined_resume(candidate_end, frames, mode):
    if mode.startswith("shift:"):
        return candidate_end + float(mode.split(":", 1)[1])

    local = [
        frame for frame in frames
        if candidate_end - 0.160 <= float(frame["center"]) <= candidate_end + 0.100
    ]
    if not local:
        return candidate_end

    kind, value = mode.split(":", 1)
    threshold = float(value)
    if kind == "evidence":
        qualifying = [frame for frame in local if float(frame["evidence"]) >= threshold]
        return float(qualifying[0]["start"]) if qualifying else candidate_end
    if kind == "rms":
        qualifying = [frame for frame in local if float(frame["rms_norm"]) >= threshold]
        return float(qualifying[0]["start"]) if qualifying else candidate_end
    if kind == "delta":
        qualifying = [frame for frame in local if float(frame["delta_rms"]) >= threshold]
        return float(qualifying[0]["start"]) if qualifying else candidate_end
    if kind == "max_delta":
        return float(max(local, key=lambda frame: float(frame["delta_rms"]))["start"])
    if kind == "max_flux":
        return float(max(local, key=lambda frame: float(frame["flux"]))["start"])
    return candidate_end


def main():
    modes = ["shift:0.000"]
    modes += [f"shift:{shift:.3f}" for shift in (-0.080, -0.060, -0.040, -0.020, 0.020, 0.040, 0.060, 0.080)]
    modes += [f"evidence:{value:.3f}" for value in (0.08, 0.12, 0.16, 0.20, 0.24, 0.30)]
    modes += [f"rms:{value:.3f}" for value in (0.003, 0.005, 0.008, 0.012, 0.020)]
    modes += [f"delta:{value:.3f}" for value in (0.002, 0.004, 0.008, 0.012, 0.020)]
    modes += ["max_delta:0", "max_flux:0"]
    totals = {mode: 0 for mode in modes}
    fold_totals = {mode: [0] * 5 for mode in modes}
    reference_total = 0
    fold_references = [0] * 5
    recoverable_close_errors = []
    recoverable_resume_errors = []

    for case_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        gold = rows(GOLD_ROOT / case_dir.name / "speech.csv")
        if not gold:
            continue
        references = [
            (float(gold[index]["end"]), float(gold[index + 1]["start"]))
            for index in range(len(gold) - 1)
        ]
        candidates = rows(case_dir / "gap_candidates.csv")
        frames = rows(case_dir / "occupancy_frames.csv")
        raw_observations = [(float(candidate["gap_start"]), float(candidate["gap_end"])) for candidate in candidates]
        matched_100 = {reference for _, reference in matched_pairs(raw_observations, references, 0.100)}
        for observation, reference in matched_pairs(raw_observations, references, 0.200):
            if reference not in matched_100:
                recoverable_close_errors.append(raw_observations[observation][0] - references[reference][0])
                recoverable_resume_errors.append(raw_observations[observation][1] - references[reference][1])
        fold = int(case_dir.name.split("_", 2)[1]) % 5
        reference_total += len(references)
        fold_references[fold] += len(references)
        for mode in modes:
            observations = [
                (
                    float(candidate["gap_start"]),
                    refined_resume(float(candidate["gap_end"]), frames, mode),
                )
                for candidate in candidates
            ]
            matched = pair_matches(observations, references)
            totals[mode] += matched
            fold_totals[mode][fold] += matched

    ranked = sorted(modes, key=lambda mode: totals[mode], reverse=True)
    for mode in ranked[:15]:
        folds = " ".join(
            f"{fold_totals[mode][fold]}/{fold_references[fold]}"
            for fold in range(5)
        )
        print(f"{mode:18s} {totals[mode]}/{reference_total} {totals[mode] / reference_total:.3f} folds={folds}")
    if recoverable_close_errors:
        print(
            "recoverable_100_to_200 "
            f"count={len(recoverable_close_errors)} "
            f"close_median_ms={statistics.median(recoverable_close_errors) * 1000.0:.1f} "
            f"resume_median_ms={statistics.median(recoverable_resume_errors) * 1000.0:.1f} "
            f"close_abs_dominant={sum(abs(close) >= abs(resume) for close, resume in zip(recoverable_close_errors, recoverable_resume_errors))}"
        )
        print("close_errors_ms=" + ",".join(f"{value * 1000.0:.0f}" for value in sorted(recoverable_close_errors)))
        print("resume_errors_ms=" + ",".join(f"{value * 1000.0:.0f}" for value in sorted(recoverable_resume_errors)))


if __name__ == "__main__":
    main()
