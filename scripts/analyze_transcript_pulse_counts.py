import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("outputs/runs/latest")
GOLD_ROOT = Path("inputs/gold")
REPORT = Path("outputs/analysis/transcript_pulse_counts.csv")
REGION_REPORT = Path("outputs/analysis/transcript_pulse_region_counts.csv")
INTER_WORD_SECONDS = 0.020


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def stress(phone: str) -> int:
    return int(phone[-1]) if phone and phone[-1].isdigit() else 0


def planned_pulses(case_dir: Path) -> list[dict[str, float | int | str]]:
    phones = read_csv(case_dir / "expected_phones.csv")
    syllables = read_csv(case_dir / "planned_syllables.csv")
    centers: dict[int, float] = {}
    cursor = 0.0
    previous_word = None
    for phone in phones:
        word_index = int(phone["word_index"])
        if previous_word is not None and word_index != previous_word:
            cursor += INTER_WORD_SECONDS
        duration = max(float(phone["weight_seconds"]), 0.025)
        centers[int(phone["phone_index"])] = cursor + duration * 0.5
        cursor += duration
        previous_word = word_index

    pulses = []
    for syllable in syllables:
        nucleus = syllable["nucleus_phone"]
        phone_index = int(syllable["nucleus_phone_index"])
        pulses.append(
            {
                "region": int(syllable["speech_region_index"]),
                "word": int(syllable["word_index"]),
                "center": centers.get(phone_index, 0.0),
                "stress": stress(nucleus),
                "phone": nucleus,
            }
        )
    return pulses


def merged_count(pulses: list[dict[str, float | int | str]], threshold: float) -> int:
    count = 0
    previous = None
    for pulse in pulses:
        merge = (
            previous is not None
            and pulse["region"] == previous["region"]
            and float(pulse["center"]) - float(previous["center"]) < threshold
            and (int(pulse["stress"]) == 0 or int(previous["stress"]) == 0)
        )
        if not merge:
            count += 1
        previous = pulse
    return count


rows: list[dict[str, object]] = []
region_rows: list[dict[str, object]] = []
for case_dir in sorted(ROOT.iterdir()):
    grade_path = case_dir / "grade.json"
    evidence_path = case_dir / "streaming_evidence_grade.json"
    if not case_dir.is_dir() or not all(
        path.exists()
        for path in (
            grade_path,
            evidence_path,
            case_dir / "expected_phones.csv",
            case_dir / "planned_syllables.csv",
        )
    ):
        continue
    grade = json.loads(grade_path.read_text(encoding="utf-8"))
    if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", True):
        continue
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    pulse_grade = evidence.get("types", {}).get("pulse", {})
    if not evidence.get("available") or not pulse_grade:
        continue
    pulses = planned_pulses(case_dir)
    words = read_csv(case_dir / "planned_words.csv")
    word_regions: dict[int, int] = {}
    for phone in read_csv(case_dir / "expected_phones.csv"):
        word_regions.setdefault(int(phone["word_index"]), int(phone["speech_region_index"]))
    list_boundaries_by_sentence: dict[int, int] = defaultdict(int)
    for word in words:
        if word.get("pause_class") == "soft_list_pause":
            list_boundaries_by_sentence[int(word["text_sentence_index"])] += 1
    list_sentences = {
        sentence for sentence, count in list_boundaries_by_sentence.items() if count >= 2
    }
    rows.append(
        {
            "case": case_dir.name,
            "mfa_nuclei": int(pulse_grade.get("target_count", 0)),
            "observed_pulses": int(pulse_grade.get("observation_count", 0)),
            "all_nuclei": len(pulses),
            "word_count": len({int(word["word_index"]) for word in words}),
            "list_case": any(count >= 2 for count in list_boundaries_by_sentence.values()),
            "stress_bearing": sum(int(pulse["stress"]) > 0 for pulse in pulses),
            "primary_stress": sum(int(pulse["stress"]) == 1 for pulse in pulses),
            "merge_weak_80ms": merged_count(pulses, 0.080),
            "merge_weak_100ms": merged_count(pulses, 0.100),
            "merge_weak_120ms": merged_count(pulses, 0.120),
        }
    )

    gold_dir = GOLD_ROOT / case_dir.name
    if not (gold_dir / "phones.csv").exists():
        continue
    gold_speech = read_csv(gold_dir / "speech.csv")
    gold_region_for_word: dict[int, int] = {}
    for region in gold_speech:
        for word_index in range(
            int(region["word_start_index"]), int(region["word_end_index"]) + 1
        ):
            gold_region_for_word[word_index] = int(region["index"])
    expected_by_region: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for pulse in pulses:
        region = gold_region_for_word.get(int(pulse["word"]))
        if region is not None:
            expected_by_region[region].append(pulse)
    words_by_region: dict[int, set[int]] = defaultdict(set)
    list_regions: set[int] = set()
    for word in words:
        word_index = int(word["word_index"])
        region = gold_region_for_word.get(word_index, word_regions.get(word_index, 0))
        words_by_region[region].add(word_index)
        if int(word["text_sentence_index"]) in list_sentences:
            list_regions.add(region)
    mfa_by_region: dict[int, int] = defaultdict(int)
    for phone in read_csv(gold_dir / "phones.csv"):
        if phone["phone"] and phone["phone"][-1].isdigit():
            mfa_by_region[int(phone["speech_region_index"])] += 1
    acoustic_regions = read_csv(case_dir / "speech_regions.csv")
    observed_by_region: dict[int, int] = defaultdict(int)
    for observation in read_csv(case_dir / "streaming_evidence_observations.csv"):
        if observation["type"] != "pulse":
            continue
        center = float(observation["center"])
        for region in acoustic_regions:
            if float(region["start"]) - 0.001 <= center <= float(region["end"]) + 0.001:
                observed_by_region[int(region["index"])] += 1
                break
    for region in sorted(mfa_by_region):
        region_pulses = [
            dict(pulse, region=region) for pulse in expected_by_region.get(region, [])
        ]
        region_rows.append(
            {
                "case": case_dir.name,
                "region": region,
                "list_region": region in list_regions,
                "mfa_nuclei": mfa_by_region[region],
                "observed_pulses": observed_by_region[region],
                "all_nuclei": len(region_pulses),
                "word_count": len(words_by_region[region]),
                "merge_weak_80ms": merged_count(region_pulses, 0.080),
                "merge_weak_100ms": merged_count(region_pulses, 0.100),
                "merge_weak_120ms": merged_count(region_pulses, 0.120),
            }
        )

REPORT.parent.mkdir(parents=True, exist_ok=True)
with REPORT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case"])
    writer.writeheader()
    writer.writerows(rows)
with REGION_REPORT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=list(region_rows[0]) if region_rows else ["case"]
    )
    writer.writeheader()
    writer.writerows(region_rows)


def summarize(name: str, target: str, selected_rows: list[dict[str, object]] = rows) -> None:
    differences = [int(row[name]) - int(row[target]) for row in selected_rows]
    absolute = [abs(value) for value in differences]
    predicted = sum(int(row[name]) for row in selected_rows)
    reference = sum(int(row[target]) for row in selected_rows)
    print(
        f"{name:18s} vs {target:15s} ratio={predicted / max(reference, 1):.3f} "
        f"case_mae={statistics.fmean(absolute):.3f} "
        f"exact={sum(value == 0 for value in differences) / max(len(selected_rows), 1):.3f} "
        f"within1={sum(value <= 1 for value in absolute) / max(len(selected_rows), 1):.3f} "
        f"signed={statistics.fmean(differences):+.3f}"
    )


print(f"qualified_cases={len(rows)}")
if not rows:
    raise SystemExit("No region-qualified evidence cases found")
for hypothesis in (
    "all_nuclei",
    "word_count",
    "stress_bearing",
    "primary_stress",
    "merge_weak_80ms",
    "merge_weak_100ms",
    "merge_weak_120ms",
):
    summarize(hypothesis, "mfa_nuclei")
print("-- expected transcript pulses versus emitted acoustic pulses --")
for hypothesis in (
    "all_nuclei",
    "merge_weak_80ms",
    "merge_weak_100ms",
    "merge_weak_120ms",
):
    summarize(hypothesis, "observed_pulses")

list_rows = [row for row in rows if row["list_case"]]
print(f"-- comma-list cases ({len(list_rows)}) --")
for hypothesis in (
    "all_nuclei",
    "word_count",
    "merge_weak_100ms",
):
    summarize(hypothesis, "mfa_nuclei", list_rows)
for hypothesis in ("all_nuclei", "merge_weak_100ms"):
    summarize(hypothesis, "observed_pulses", list_rows)

print(f"-- aligned speech regions ({len(region_rows)}) --")
for hypothesis in ("all_nuclei", "word_count", "merge_weak_100ms", "merge_weak_120ms"):
    summarize(hypothesis, "mfa_nuclei", region_rows)
for hypothesis in ("all_nuclei", "merge_weak_100ms", "merge_weak_120ms"):
    summarize(hypothesis, "observed_pulses", region_rows)

list_region_rows = [row for row in region_rows if row["list_region"]]
print(f"-- aligned comma-list speech regions ({len(list_region_rows)}) --")
for hypothesis in ("all_nuclei", "word_count", "merge_weak_100ms", "merge_weak_120ms"):
    summarize(hypothesis, "mfa_nuclei", list_region_rows)
for hypothesis in ("all_nuclei", "merge_weak_100ms", "merge_weak_120ms"):
    summarize(hypothesis, "observed_pulses", list_region_rows)
