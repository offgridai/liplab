import csv
import itertools
import pathlib


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
    n, m = len(observations), len(references)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            compatible = all(abs(observations[i - 1][edge] - references[j - 1][edge]) <= TOLERANCE for edge in (0, 1))
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + int(compatible))
    return dp[n][m]


def main():
    cases = []
    reference_total = 0
    for case_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        gold = rows(GOLD_ROOT / case_dir.name / "speech.csv")
        if not gold:
            continue
        references = [(float(gold[index]["end"]), float(gold[index + 1]["start"])) for index in range(len(gold) - 1)]
        candidates = rows(case_dir / "gap_candidates.csv")
        fold = int(case_dir.name.split("_", 2)[1]) % 5
        cases.append((candidates, references, fold))
        reference_total += len(references)

    windows = (0.120, 0.140, 0.160, 0.180, 0.200, 0.220, 0.240)
    results = []
    for normal_window, strong_window in itertools.product(windows, windows):
        predicted_total = matched_total = 0
        fold_predicted = [0] * 5
        fold_matched = [0] * 5
        fold_reference = [0] * 5
        for candidates, references, fold in cases:
            observations = []
            for candidate in candidates:
                duration = float(candidate["gap_duration"])
                threshold = strong_window if int(candidate["strong_quiet_close"]) else normal_window
                if duration > threshold:
                    observations.append((float(candidate["gap_start"]), float(candidate["gap_end"])))
            matched = pair_matches(observations, references)
            predicted_total += len(observations)
            matched_total += matched
            fold_predicted[fold] += len(observations)
            fold_matched[fold] += matched
            fold_reference[fold] += len(references)
        precision = matched_total / predicted_total if predicted_total else 0.0
        recall = matched_total / reference_total if reference_total else 0.0
        f1 = 2.0 * matched_total / (predicted_total + reference_total) if predicted_total + reference_total else 0.0
        fold_f1 = [
            2.0 * fold_matched[index] / (fold_predicted[index] + fold_reference[index])
            if fold_predicted[index] + fold_reference[index] else 0.0
            for index in range(5)
        ]
        results.append((f1, precision, recall, matched_total, predicted_total, normal_window, strong_window, fold_f1))

    for result in sorted(results, reverse=True)[:15]:
        f1, precision, recall, matched, predicted, normal, strong, folds = result
        print(
            f"f1={f1:.3f} p={precision:.3f} r={recall:.3f} matched={matched}/{reference_total} predicted={predicted} "
            f"normal={normal:.3f} strong={strong:.3f} folds=" + ",".join(f"{value:.3f}" for value in folds)
        )


if __name__ == "__main__":
    main()
