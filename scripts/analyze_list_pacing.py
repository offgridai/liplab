import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("outputs/runs/latest")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


areas: list[dict[str, object]] = []
errors: list[float] = []
case_errors: dict[str, list[float]] = defaultdict(list)

for case_dir in ROOT.iterdir():
    words_path = case_dir / "planned_words.csv"
    onset_path = case_dir / "word_onset_diagnostics.csv"
    grade_path = case_dir / "grade.json"
    if not case_dir.is_dir() or not all(path.exists() for path in (words_path, onset_path, grade_path)):
        continue
    with grade_path.open(encoding="utf-8") as handle:
        grade = json.load(handle)
    region_qualified = not grade.get("pause_alignment", {}).get(
        "speech_region_count_mismatch", True
    )

    words = read_csv(words_path)
    onsets = {int(row["word_index"]): row for row in read_csv(onset_path)}
    by_sentence: dict[int, list[dict[str, str]]] = defaultdict(list)
    for word in words:
        by_sentence[int(word["text_sentence_index"])].append(word)

    for sentence_index, sentence_words in by_sentence.items():
        sentence_words.sort(key=lambda row: int(row["word_index"]))
        boundaries = [
            word for word in sentence_words if word.get("pause_class") == "soft_list_pause"
        ]
        if len(boundaries) < 2:
            continue
        first_index = int(boundaries[0]["word_index"])
        span = [word for word in sentence_words if int(word["word_index"]) >= first_index]
        span_errors: list[float] = []
        for word in span:
            if not region_qualified:
                continue
            onset = onsets.get(int(word["word_index"]))
            if not onset or onset.get("missing_event", "").lower() == "true":
                continue
            # The exported error is gold minus runtime: positive means animation was early.
            error = float(onset["error_ms"])
            errors.append(error)
            span_errors.append(error)
            case_errors[case_dir.name].append(error)
        areas.append(
            {
                "case": case_dir.name,
                "sentence_index": sentence_index,
                "boundary_count": len(boundaries),
                "word_count": len(span),
                "first_word": span[0]["word"] if span else "",
                "last_word": span[-1]["word"] if span else "",
                "region_qualified": region_qualified,
                "mean_signed_ms": statistics.fmean(span_errors) if span_errors else 0.0,
                "mean_abs_ms": statistics.fmean(abs(value) for value in span_errors)
                if span_errors
                else 0.0,
            }
        )

analysis_dir = Path("outputs/analysis")
analysis_dir.mkdir(parents=True, exist_ok=True)
with (analysis_dir / "comma_list_areas.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(areas[0]) if areas else ["case"])
    writer.writeheader()
    writer.writerows(areas)

abs_errors = [abs(value) for value in errors]
qualified_areas = sum(bool(area["region_qualified"]) for area in areas)
print(
    f"areas={len(areas)} qualified_areas={qualified_areas} "
    f"qualified_cases={len(case_errors)} scored_words={len(errors)}"
)
if errors:
    print(
        f"mean_signed_ms={statistics.fmean(errors):.1f} "
        f"mean_abs_ms={statistics.fmean(abs_errors):.1f} "
        f"median_abs_ms={statistics.median(abs_errors):.1f} "
        f"early={sum(value > 100.0 for value in errors)} "
        f"late={sum(value < -100.0 for value in errors)} "
        f"within_100ms={sum(abs(value) <= 100.0 for value in errors)}"
    )
print("worst cases:")
for case, values in sorted(
    case_errors.items(), key=lambda item: statistics.fmean(abs(value) for value in item[1]), reverse=True
)[:15]:
    print(
        f"  {case[:9]} signed={statistics.fmean(values):+.1f}ms "
        f"mae={statistics.fmean(abs(value) for value in values):.1f}ms words={len(values)}"
    )
