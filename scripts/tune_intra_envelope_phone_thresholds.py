import csv
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "latest"
GOLD = ROOT / "inputs" / "gold"
FAMILIES = {
    "mbp": {"M", "B", "P"},
    "fv": {"F", "V"},
    "glide": {"W", "Y"},
    "sibilant": {"S", "Z", "SH", "ZH", "CH", "JH"},
    "round": {"UW", "UH", "OW", "OY", "AO"},
}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_phone(phone):
    return phone.rstrip("012")


def qualified(case_dir):
    grade = json.loads((case_dir / "grade.json").read_text(encoding="utf-8"))
    return not grade.get("pause_alignment", {}).get("speech_region_count_mismatch", False)


def family_for(phone):
    base = base_phone(phone)
    for family, phones in FAMILIES.items():
        if base in phones:
            return family
    return None


def load_cases():
    cases = []
    for case_dir in sorted(path for path in RUN.iterdir() if path.is_dir()):
        gold_path = GOLD / case_dir.name / "phones.csv"
        if not gold_path.exists() or not qualified(case_dir):
            continue
        gold_phones = rows(gold_path)
        syllable_rows = rows(case_dir / "planned_syllables.csv")
        expected_phones = rows(case_dir / "expected_phones.csv")
        observations = rows(case_dir / "streaming_permissive_evidence_candidates.csv")
        syllables = []
        for syllable in syllable_rows:
            begin = int(syllable["phone_begin_index"])
            end = int(syllable["phone_end_index"])
            nucleus = int(syllable["nucleus_phone_index"])
            if nucleus < 0 or nucleus >= len(gold_phones) or begin < 0 or begin >= end:
                continue
            phones = gold_phones[begin:min(end, len(gold_phones))]
            planned_phones = expected_phones[begin:min(end, len(expected_phones))]
            if not phones:
                continue
            nucleus_phone = gold_phones[nucleus]
            targets = {family: [] for family in FAMILIES}
            for phone in phones:
                family = family_for(phone["phone"])
                if family:
                    targets[family].append((float(phone["start"]), float(phone["end"])))
            syllables.append({
                "start": min(float(phone["start"]) for phone in phones),
                "end": max(float(phone["end"]) for phone in phones),
                "nucleus_start": float(nucleus_phone["start"]),
                "nucleus_end": float(nucleus_phone["end"]),
                "targets": targets,
                "expected_families": {
                    family for phone in planned_phones
                    if (family := family_for(phone["base_phone"]))
                },
            })

        pulses = [item for item in observations if item["type"] == "pulse"]
        used = [False] * len(syllables)
        matched_syllables = []
        for pulse in pulses:
            center = float(pulse["center"])
            best = None
            best_error = 0.100001
            for index, syllable in enumerate(syllables):
                if used[index]:
                    continue
                error = max(syllable["nucleus_start"] - center, center - syllable["nucleus_end"], 0.0)
                if error < best_error:
                    best = index
                    best_error = error
            if best is not None:
                used[best] = True
                matched_syllables.append(syllables[best])

        pulse_centers = [float(item["center"]) for item in pulses]
        family_observations = {family: [] for family in FAMILIES}
        for item in observations:
            if item["type"] in family_observations:
                center = float(item["center"])
                nearest_pulse = min(pulse_centers, key=lambda pulse: abs(center - pulse), default=-999.0)
                nearest_pulse_delta = center - nearest_pulse if nearest_pulse >= 0.0 else 999.0
                family_observations[item["type"]].append(
                    (center, float(item["score"]), nearest_pulse_delta, nearest_pulse)
                )
        cases.append((matched_syllables, family_observations))
    return cases


def score(cases, family, threshold, max_pulse_distance, strongest_per_pulse):
    targets = observations = matched = 0
    for syllables, by_family in cases:
        candidates = by_family[family]
        for syllable in syllables:
            if family not in syllable["expected_families"]:
                continue
            syllable_targets = syllable["targets"][family]
            syllable_candidates = [
                (center, value, pulse_center) for center, value, pulse_delta, pulse_center in candidates
                if value >= threshold
                and abs(pulse_delta) <= max_pulse_distance
                and syllable["start"] <= center <= syllable["end"]
            ]
            if strongest_per_pulse:
                strongest = {}
                for center, value, pulse_center in syllable_candidates:
                    if pulse_center not in strongest or value > strongest[pulse_center][1]:
                        strongest[pulse_center] = (center, value)
                syllable_candidates = list(strongest.values())
            targets += len(syllable_targets)
            observations += len(syllable_candidates)
            matched += min(len(syllable_targets), len(syllable_candidates))
    precision = matched / observations if observations else 0.0
    recall = matched / targets if targets else 0.0
    f1 = 2 * matched / (observations + targets) if observations + targets else 0.0
    return precision, recall, f1, observations, targets, matched


def main():
    cases = load_cases()
    report = {"case_count": len(cases), "families": {}}
    for family in FAMILIES:
        candidates = []
        for step in range(0, 51):
            threshold = 0.48 + step * 0.01
            for max_pulse_distance in (0.05, 0.08, 0.10, 0.12, 0.16, 0.24, 999.0):
                for strongest_per_pulse in (False, True):
                    result = score(cases, family, threshold, max_pulse_distance, strongest_per_pulse)
                    candidates.append((result[2], min(result[0], result[1]), threshold, max_pulse_distance, strongest_per_pulse, result))
        best = max(candidates)
        precision, recall, f1, observation_count, target_count, matched_count = best[5]
        report["families"][family] = {
            "threshold": best[2],
            "max_pulse_distance": best[3],
            "strongest_per_pulse": best[4],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "observation_count": observation_count,
            "target_count": target_count,
            "matched_count": matched_count,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
