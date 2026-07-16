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


def matched_reference_indices(observations, references):
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
    matched = set()
    i, j = n, m
    while i and j:
        compatible = all(
            abs(observations[i - 1][edge] - references[j - 1][edge]) <= TOLERANCE
            for edge in (0, 1)
        )
        if compatible and dp[i][j] == dp[i - 1][j - 1] + 1:
            matched.add(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return matched


def valley_candidates(frames, evidence_max, rms_max, minimum_duration):
    low = [
        float(frame["evidence"]) <= evidence_max and float(frame["rms_norm"]) <= rms_max
        for frame in frames
    ]
    candidates = []
    index = 0
    while index < len(frames):
        if not low[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(frames) and low[index + 1]:
            index += 1
        end = index
        start_sec = float(frames[start]["start"])
        end_sec = float(frames[end]["end"])
        if end_sec - start_sec >= minimum_duration:
            before = frames[max(0, start - 12):start]
            after = frames[end + 1:min(len(frames), end + 13)]
            speech_before = any(float(frame["evidence"]) >= 0.16 for frame in before)
            speech_after = any(float(frame["evidence"]) >= 0.16 for frame in after)
            if speech_before and speech_after:
                candidates.append((start_sec, end_sec))
        index += 1
    return candidates


def main():
    cases = []
    reference_total = 0
    for case_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        gold = rows(GOLD_ROOT / case_dir.name / "speech.csv")
        frames = rows(case_dir / "occupancy_frames.csv")
        if not gold or not frames:
            continue
        references = [
            (float(gold[index]["end"]), float(gold[index + 1]["start"]))
            for index in range(len(gold) - 1)
        ]
        reference_total += len(references)
        existing = [
            (float(row["gap_start"]), float(row["gap_end"]))
            for row in rows(case_dir / "gap_candidates.csv")
        ]
        planned = {
            int(row["word_index"]): row["pause_class"]
            for row in rows(case_dir / "planned_boundaries.csv")
        }
        reference_classes = [planned.get(int(gold[index]["word_end_index"]), "none") for index in range(len(gold) - 1)]
        cases.append((frames, references, existing, reference_classes))

    results = []
    for evidence_max, rms_max, minimum_duration in itertools.product(
        (0.03, 0.05, 0.08, 0.12, 0.16, 0.20),
        (0.002, 0.004, 0.008, 0.012, 0.020),
        (0.03, 0.05, 0.08, 0.10, 0.12),
    ):
        candidate_total = 0
        matched_total = 0
        union_candidate_total = 0
        union_matched_total = 0
        for frames, references, existing, _ in cases:
            candidates = valley_candidates(frames, evidence_max, rms_max, minimum_duration)
            candidate_total += len(candidates)
            matched_total += pair_matches(candidates, references)
            combined = sorted(existing + candidates)
            union_candidate_total += len(combined)
            union_matched_total += pair_matches(combined, references)
        precision = matched_total / candidate_total if candidate_total else 0.0
        recall = matched_total / reference_total if reference_total else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append((f1, precision, recall, matched_total, candidate_total, union_matched_total, union_candidate_total, evidence_max, rms_max, minimum_duration))

    for result in sorted(results, reverse=True)[:20]:
        f1, precision, recall, matched, candidates, _, _, evidence, rms, duration = result
        print(
            f"f1={f1:.3f} p={precision:.3f} r={recall:.3f} "
            f"matched={matched}/{reference_total} candidates={candidates} "
            f"evidence<={evidence:.3f} rms<={rms:.3f} duration>={duration:.3f}"
        )

    print("\nHighest union recall:")
    for result in sorted(results, key=lambda item: (item[5], -item[6]), reverse=True)[:15]:
        _, _, _, _, _, union_matched, union_candidates, evidence, rms, duration = result
        print(
            f"union_r={union_matched / reference_total:.3f} matched={union_matched}/{reference_total} "
            f"candidates={union_candidates} evidence<={evidence:.3f} rms<={rms:.3f} duration>={duration:.3f}"
        )

    print("\nRecovery channel using best standalone valley rule:")
    existing_total = 0
    union_total = 0
    valley_total = 0
    missing_classes = {}
    recovered_classes = {}
    for frames, references, existing, reference_classes in cases:
        valleys = valley_candidates(frames, 0.200, 0.020, 0.120)
        existing_matches = matched_reference_indices(existing, references)
        valley_matches = matched_reference_indices(valleys, references)
        combined = sorted(existing + valleys)
        union_matches = matched_reference_indices(combined, references)
        existing_total += len(existing_matches)
        valley_total += len(valley_matches)
        union_total += len(union_matches)
        for index, boundary_class in enumerate(reference_classes):
            if index not in existing_matches:
                missing_classes[boundary_class] = missing_classes.get(boundary_class, 0) + 1
                if index in union_matches:
                    recovered_classes[boundary_class] = recovered_classes.get(boundary_class, 0) + 1
    print(f"existing={existing_total}/{reference_total} valley={valley_total}/{reference_total} union={union_total}/{reference_total}")
    print(f"missing_by_class={missing_classes}")
    print(f"recovered_by_class={recovered_classes}")


if __name__ == "__main__":
    main()
