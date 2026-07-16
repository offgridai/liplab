import csv
import itertools
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"
GOLD_ROOT = ROOT / "inputs" / "gold"
TOLERANCE_SEC = 0.100
HARD_MARKS = set(".!?:;-")


def rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def policy_regions(
    words: list[dict[str, str]],
    boundaries: list[dict[str, str]],
    hard_threshold: float,
    comma_threshold: float,
    strong_threshold: float,
) -> tuple[list[tuple[float, float]], dict[str, int]]:
    split_after: set[int] = set()
    reasons = {"hard": 0, "comma": 0, "strong": 0}
    for boundary in boundaries:
        marks = boundary.get("mark_sequence", "")
        acoustic_gap = number(boundary, "acoustic_gap_seconds")
        word_index = int(number(boundary, "word_index"))
        reason = ""
        if any(mark in HARD_MARKS for mark in marks) and acoustic_gap >= hard_threshold:
            reason = "hard"
        elif "," in marks and acoustic_gap >= comma_threshold:
            reason = "comma"
        elif acoustic_gap >= strong_threshold:
            reason = "strong"
        if reason:
            split_after.add(word_index)
            reasons[reason] += 1

    regions: list[tuple[float, float]] = []
    region_start = 0
    for list_index, word in enumerate(words):
        word_index = int(number(word, "word_index"))
        if word_index not in split_after and list_index != len(words) - 1:
            continue
        regions.append(
            (number(words[region_start], "start"), number(words[list_index], "end"))
        )
        region_start = list_index + 1
    return regions, reasons


def gaps(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(regions[index][1], regions[index + 1][0]) for index in range(len(regions) - 1)]


def match_count(
    predicted: list[tuple[float, float]], reference: list[tuple[float, float]]
) -> int:
    n, m = len(predicted), len(reference)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    tolerance = TOLERANCE_SEC + 1e-6
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            compatible = all(
                abs(predicted[i - 1][edge] - reference[j - 1][edge]) <= tolerance
                for edge in (0, 1)
            )
            dp[i][j] = max(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1] + int(compatible),
            )
    return dp[n][m]


def evaluate(hard_ms: int, comma_ms: int, strong_ms: int) -> dict[str, float]:
    matched = 0
    predicted_count = 0
    reference_count = 0
    perfect = 0
    under = 0
    over = 0
    equal_mismatch = 0
    split_reasons = {"hard": 0, "comma": 0, "strong": 0}
    case_count = 0

    for case_dir in sorted(path for path in RUN_ROOT.iterdir() if path.is_dir()):
        gold_dir = GOLD_ROOT / case_dir.name
        words = rows(gold_dir / "words.csv")
        boundaries = rows(gold_dir / "boundaries.csv")
        predicted_rows = rows(case_dir / "refined_speech_regions.csv")
        if not words or not predicted_rows:
            continue
        reference_regions, reasons = policy_regions(
            words,
            boundaries,
            hard_ms / 1000.0,
            comma_ms / 1000.0,
            strong_ms / 1000.0,
        )
        predicted_regions = [
            (number(region, "start"), number(region, "end")) for region in predicted_rows
        ]
        predicted_gaps = gaps(predicted_regions)
        reference_gaps = gaps(reference_regions)
        case_matches = match_count(predicted_gaps, reference_gaps)
        matched += case_matches
        predicted_count += len(predicted_gaps)
        reference_count += len(reference_gaps)
        case_count += 1
        for reason, count in reasons.items():
            split_reasons[reason] += count

        if len(predicted_regions) < len(reference_regions):
            under += 1
        elif len(predicted_regions) > len(reference_regions):
            over += 1
        elif case_matches == len(reference_gaps):
            perfect += 1
        else:
            equal_mismatch += 1

    precision = matched / predicted_count if predicted_count else 1.0
    recall = matched / reference_count if reference_count else 1.0
    return {
        "hard_ms": hard_ms,
        "comma_ms": comma_ms,
        "strong_ms": strong_ms,
        "cases": case_count,
        "matched": matched,
        "predicted": predicted_count,
        "reference": reference_count,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * matched / (predicted_count + reference_count),
        "perfect": perfect,
        "under": under,
        "over": over,
        "equal_mismatch": equal_mismatch,
        "hard_splits": split_reasons["hard"],
        "comma_splits": split_reasons["comma"],
        "strong_splits": split_reasons["strong"],
    }


def main() -> int:
    results = [
        evaluate(hard, comma, strong)
        for hard, comma, strong in itertools.product(
            (60, 80, 90, 100),
            (60, 80, 100, 110, 115, 120, 125, 130, 135, 140),
            (200, 220, 240, 260, 280),
        )
    ]
    results.sort(key=lambda row: (row["f1"], row["perfect"], -row["over"]), reverse=True)
    print(
        "hard comma strong  pair_p pair_r pair_f1 perfect under over equal "
        "refs hard_n comma_n strong_n"
    )
    for row in results[:24]:
        print(
            f"{row['hard_ms']:>4} {row['comma_ms']:>5} {row['strong_ms']:>6} "
            f"{row['precision']:.3f} {row['recall']:.3f} {row['f1']:.3f} "
            f"{row['perfect']:>7} {row['under']:>5} {row['over']:>4} "
            f"{row['equal_mismatch']:>5} {row['reference']:>4} "
            f"{row['hard_splits']:>6} {row['comma_splits']:>7} {row['strong_splits']:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
