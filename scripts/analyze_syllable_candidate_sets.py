import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "latest"
GOLD = ROOT / "inputs" / "gold"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    counts = {"hit": 0, "ahead": 0, "behind": 0, "hole": 0, "region_head_miss": 0}
    deltas = []
    for case_dir in sorted(path for path in RUN.iterdir() if path.is_dir()):
        grade_path = case_dir / "grade.json"
        candidate_path = case_dir / "streaming_syllable_candidate_sets.csv"
        gold_path = GOLD / case_dir.name / "phones.csv"
        if not grade_path.exists() or not candidate_path.exists() or not gold_path.exists():
            continue
        grade = json.loads(grade_path.read_text(encoding="utf-8"))
        if grade.get("pause_alignment", {}).get("speech_region_count_mismatch", False):
            continue
        gold_phones = rows(gold_path)
        targets = []
        for syllable in rows(case_dir / "planned_syllables.csv"):
            nucleus = int(syllable["nucleus_phone_index"])
            if nucleus < 0 or nucleus >= len(gold_phones):
                continue
            phone = gold_phones[nucleus]
            targets.append({
                "start": float(phone["start"]),
                "end": float(phone["end"]),
                "region": int(syllable["speech_region_index"]),
            })
        grouped = {}
        for candidate in rows(candidate_path):
            key = (float(candidate["audio_center"]), float(candidate["decision"]))
            grouped.setdefault(key, []).append(int(candidate["syllable_index"]))
        used = [False] * len(targets)
        seen_regions = set()
        for (center, _), candidates in sorted(grouped.items()):
            best = None
            best_error = 0.100001
            for index, target in enumerate(targets):
                if used[index]:
                    continue
                error = max(target["start"] - center, center - target["end"], 0.0)
                if error < best_error:
                    best = index
                    best_error = error
            if best is None:
                continue
            used[best] = True
            is_region_head = targets[best]["region"] not in seen_regions
            seen_regions.add(targets[best]["region"])
            if best in candidates:
                counts["hit"] += 1
                continue
            if is_region_head:
                counts["region_head_miss"] += 1
            if best < min(candidates):
                counts["ahead"] += 1
                deltas.append(min(candidates) - best)
            elif best > max(candidates):
                counts["behind"] += 1
                deltas.append(best - max(candidates))
            else:
                counts["hole"] += 1
    print(json.dumps({"counts": counts, "mean_outside_distance": sum(deltas) / len(deltas)}, indent=2))


if __name__ == "__main__":
    main()
