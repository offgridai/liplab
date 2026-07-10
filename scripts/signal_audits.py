import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import re
from collections import Counter, defaultdict


ROOT = pathlib.Path(__file__).resolve().parents[1]
LATEST_RUN = ROOT / "outputs" / "runs" / "latest"
GOLD_ROOT = ROOT / "inputs" / "gold"

TEST_BUCKET_MOD = 5
TEST_BUCKET_VALUE = 0

PHONE_CLASS_COLUMNS = [
    "silence",
    "vowel_open",
    "vowel_front",
    "vowel_round",
    "bilabial",
    "labiodental",
    "dental",
    "sibilant",
    "stop_burst",
    "liquid",
    "glide",
    "nasal",
    "unknown",
]


def read_csv_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone or "").upper()


def phone_class_for_base(base_phone: str) -> str:
    p = strip_stress(base_phone)
    if not p:
        return "unknown"
    if p in {"AA", "AE", "AH", "AW", "AY"}:
        return "vowel_open"
    if p in {"EH", "ER", "EY", "IH", "IY"}:
        return "vowel_front"
    if p in {"AO", "OW", "OY", "UH", "UW"}:
        return "vowel_round"
    if p in {"B", "M", "P"}:
        return "bilabial"
    if p in {"F", "V"}:
        return "labiodental"
    if p in {"TH", "DH"}:
        return "dental"
    if p in {"S", "Z", "SH", "ZH", "CH", "JH"}:
        return "sibilant"
    if p in {"T", "D", "K", "G", "HH"}:
        return "stop_burst"
    if p in {"L", "R"}:
        return "liquid"
    if p in {"W", "Y"}:
        return "glide"
    if p in {"N", "NG"}:
        return "nasal"
    return "unknown"


def map_active_to_clock(regions: list[tuple[float, float]], target_active_sec: float) -> float:
    remaining = max(target_active_sec, 0.0)
    for start, end in regions:
        dur = max(end - start, 0.0)
        if dur <= 0.0:
            continue
        if remaining <= dur:
            return start + remaining
        remaining -= dur
    return regions[-1][1] if regions else 0.0


def deterministic_bucket(case_id: str) -> int:
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % TEST_BUCKET_MOD


def is_test_case(case_id: str) -> bool:
    return deterministic_bucket(case_id) == TEST_BUCKET_VALUE


def load_case_bundle(case_dir: pathlib.Path) -> dict | None:
    case_id = case_dir.name
    occupancy_rows = read_csv_rows(case_dir / "occupancy_frames.csv")
    phone_rows = read_csv_rows(case_dir / "phone_class_frames.csv")
    expected_phone_rows = read_csv_rows(case_dir / "expected_phones.csv")
    gold_words = read_csv_rows(GOLD_ROOT / case_id / "words.csv")
    gold_speech = read_csv_rows(GOLD_ROOT / case_id / "speech.csv")
    gold_phone_rows = read_csv_rows(GOLD_ROOT / case_id / "phones.csv")
    mfa_phones = []
    for row in gold_phone_rows:
        phone = row.get("phone", "")
        if not phone:
            continue
        mfa_phones.append(
            {
                "start": as_float(row, "start"),
                "end": as_float(row, "end"),
                "text": phone,
                "base_phone": strip_stress(phone),
                "class": phone_class_for_base(phone),
            }
        )

    if not occupancy_rows or not phone_rows or not expected_phone_rows or not gold_words or not gold_speech or not mfa_phones:
        return None

    expected_phones = []
    for row in expected_phone_rows:
        expected_phones.append(
            {
                "phone_index": as_int(row, "phone_index", -1),
                "base_phone": strip_stress(row.get("base_phone", "")),
                "class": phone_class_for_base(row.get("base_phone", "")),
                "word_index": as_int(row, "word_index", -1),
                "word": row.get("word", ""),
                "weight_seconds": as_float(row, "weight_seconds", 0.0),
            }
        )

    speech_regions = [(as_float(row, "start"), as_float(row, "end")) for row in gold_speech]
    total_active_sec = sum(max(end - start, 0.0) for start, end in speech_regions)
    total_weight_sec = sum(max(phone["weight_seconds"], 0.020) for phone in expected_phones)
    cumulative = 0.0
    for phone in expected_phones:
        weight = max(phone["weight_seconds"], 0.020)
        center_active = cumulative + weight * 0.5
        mapped_active = center_active * total_active_sec / max(total_weight_sec, 0.001)
        phone["expected_time"] = map_active_to_clock(speech_regions, mapped_active)
        cumulative += weight

    matched = []
    exp_i = 0
    mfa_i = 0
    while exp_i < len(expected_phones) and mfa_i < len(mfa_phones):
        exp_base = expected_phones[exp_i]["base_phone"]
        mfa_base = mfa_phones[mfa_i]["base_phone"]
        if exp_base == mfa_base:
            matched.append(
                {
                    "expected_index": exp_i,
                    "class": expected_phones[exp_i]["class"],
                    "word_index": expected_phones[exp_i]["word_index"],
                    "expected_time": expected_phones[exp_i]["expected_time"],
                    "actual_start": mfa_phones[mfa_i]["start"],
                    "actual_end": mfa_phones[mfa_i]["end"],
                    "actual_time": 0.5 * (mfa_phones[mfa_i]["start"] + mfa_phones[mfa_i]["end"]),
                    "base_phone": exp_base,
                }
            )
            exp_i += 1
            mfa_i += 1
            continue
        if mfa_i + 1 < len(mfa_phones) and exp_base == mfa_phones[mfa_i + 1]["base_phone"]:
            mfa_i += 1
            continue
        if exp_i + 1 < len(expected_phones) and expected_phones[exp_i + 1]["base_phone"] == mfa_base:
            exp_i += 1
            continue
        exp_i += 1
        mfa_i += 1

    if len(matched) < 8:
        return None

    gold_word_starts = [as_float(row, "start") for row in gold_words if as_int(row, "word_index", -1) > 0]
    return {
        "case_id": case_id,
        "split": "test" if is_test_case(case_id) else "train",
        "occupancy_rows": occupancy_rows,
        "phone_rows": phone_rows,
        "expected_phones": expected_phones,
        "mfa_phones": mfa_phones,
        "matched_phones": matched,
        "speech_regions": speech_regions,
        "gold_word_starts": gold_word_starts,
    }


def nearest_index(times: list[float], t: float) -> int:
    if not times:
        return -1
    best_index = 0
    best_dist = abs(times[0] - t)
    for i in range(1, len(times)):
        dist = abs(times[i] - t)
        if dist < best_dist:
            best_index = i
            best_dist = dist
    return best_index


def build_direction_rows(bundle: dict, lookaround: int) -> list[dict]:
    matched = bundle["matched_phones"]
    expected_times = [row["expected_time"] for row in matched]
    actual_times = [row["actual_time"] for row in matched]
    rows = []
    for occ, phone in zip(bundle["occupancy_rows"], bundle["phone_rows"]):
        t = as_float(occ, "center")
        if not any(start <= t <= end for start, end in bundle["speech_regions"]):
            continue
        planned_idx = nearest_index(expected_times, t)
        actual_idx = nearest_index(actual_times, t)
        if planned_idx < 0 or actual_idx < 0:
            continue

        ahead_scores = []
        behind_scores = []
        for step in range(1, lookaround + 1):
            if planned_idx + step < len(matched):
                ahead_scores.append(as_float(phone, matched[planned_idx + step]["class"]))
            if planned_idx - step >= 0:
                behind_scores.append(as_float(phone, matched[planned_idx - step]["class"]))
        if not ahead_scores or not behind_scores:
            continue

        current_class = matched[planned_idx]["class"]
        rows.append(
            {
                "case_id": bundle["case_id"],
                "split": bundle["split"],
                "time": t,
                "planned_idx": planned_idx,
                "actual_idx": actual_idx,
                "offset": actual_idx - planned_idx,
                "current_class": current_class,
                "current_score": as_float(phone, current_class),
                "ahead_score": max(ahead_scores),
                "behind_score": max(behind_scores),
                "directional_score": max(ahead_scores) - max(behind_scores),
                "pause_confidence": as_float(occ, "pause_family_confidence"),
                "pause_family": occ.get("pause_family", ""),
            }
        )
    return rows


def evaluate_direction_accuracy(rows: list[dict], min_abs_offset: int) -> dict:
    labeled = [row for row in rows if abs(int(row["offset"])) >= min_abs_offset]
    if not labeled:
        return {"count": 0, "accuracy": 0.0, "majority_accuracy": 0.0}

    correct = 0
    labels = []
    for row in labeled:
        label = 1 if row["offset"] > 0 else -1
        pred = 1 if row["directional_score"] > 0.0 else -1
        labels.append(label)
        if pred == label:
            correct += 1
    majority = 1 if sum(1 for x in labels if x > 0) >= sum(1 for x in labels if x < 0) else -1
    majority_correct = sum(1 for x in labels if x == majority)
    return {
        "count": len(labeled),
        "accuracy": correct / len(labeled),
        "majority_accuracy": majority_correct / len(labeled),
    }


def average_directional_score(rows: list[dict], center_index: int, history_frames: int) -> float:
    start = max(0, center_index - history_frames + 1)
    window = rows[start:center_index + 1]
    if not window:
        return 0.0
    return sum(float(row["directional_score"]) for row in window) / len(window)


def evaluate_retrospective_accuracy(rows: list[dict], min_abs_offset: int, history_frames: int) -> dict:
    labeled_indices = [i for i, row in enumerate(rows) if abs(int(row["offset"])) >= min_abs_offset]
    if not labeled_indices:
        return {"count": 0, "accuracy": 0.0, "majority_accuracy": 0.0}
    labels = []
    correct = 0
    for i in labeled_indices:
        label = 1 if rows[i]["offset"] > 0 else -1
        pred = 1 if average_directional_score(rows, i, history_frames) > 0.0 else -1
        labels.append(label)
        if pred == label:
            correct += 1
    majority = 1 if sum(1 for x in labels if x > 0) >= sum(1 for x in labels if x < 0) else -1
    majority_correct = sum(1 for x in labels if x == majority)
    return {
        "count": len(labeled_indices),
        "accuracy": correct / len(labeled_indices),
        "majority_accuracy": majority_correct / len(labeled_indices),
    }


def shuffled_rows(rows: list[dict], seed: int, mode: str) -> list[dict]:
    scores = [float(row["directional_score"]) for row in rows]
    rng = random.Random(seed)
    if mode == "shuffle":
        rng.shuffle(scores)
    elif mode == "shift" and scores:
        shift = rng.randrange(len(scores))
        scores = scores[shift:] + scores[:shift]
    clone = [dict(row) for row in rows]
    for row, score in zip(clone, scores):
        row["directional_score"] = score
    return clone


def run_audit_1(bundles: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for bundle in bundles:
        occupancy = bundle["occupancy_rows"]
        gold_word_starts = bundle["gold_word_starts"]
        for phone in bundle["mfa_phones"]:
            if phone["class"] not in {"bilabial", "stop_burst"}:
                continue
            if phone["class"] == "stop_burst" and phone["base_phone"] == "HH":
                continue

            overlap = [
                row for row in occupancy
                if as_float(row, "end") > phone["start"] and as_float(row, "start") < phone["end"]
            ]
            if not overlap:
                continue

            max_pause_conf = max(as_float(row, "pause_family_confidence") for row in overlap)
            pause_rows = [row for row in overlap if (row.get("pause_family", "") and row.get("pause_family", "") != "continuous_speech")]
            fired = 1 if pause_rows else 0
            strong_fired = 1 if any(as_float(row, "pause_family_confidence") >= 0.50 for row in pause_rows) else 0
            dominant_pause_family = ""
            if pause_rows:
                dominant_pause_family = max(
                    pause_rows,
                    key=lambda row: as_float(row, "pause_family_confidence"),
                ).get("pause_family", "")

            true_boundary = any(abs(start - phone["end"]) <= 0.030 for start in gold_word_starts)
            near_boundary = any(abs(start - phone["end"]) <= 0.060 for start in gold_word_starts)
            resume_rows = [
                row for row in occupancy
                if as_float(row, "start") >= phone["end"] and as_float(row, "start") <= phone["end"] + 0.080
            ]
            strong_resume = 1 if any(as_int(row, "strong_onset") != 0 for row in resume_rows) else 0
            closure_class = "mbp_closure" if phone["class"] == "bilabial" else "other_stop_closure"

            rows.append(
                {
                    "case_id": bundle["case_id"],
                    "split": bundle["split"],
                    "closure_class": closure_class,
                    "phone": phone["base_phone"],
                    "start": f"{phone['start']:.6f}",
                    "end": f"{phone['end']:.6f}",
                    "duration_ms": f"{(phone['end'] - phone['start']) * 1000.0:.3f}",
                    "pause_fired": fired,
                    "strong_pause_fired": strong_fired,
                    "max_pause_confidence": f"{max_pause_conf:.6f}",
                    "dominant_pause_family": dominant_pause_family,
                    "true_word_boundary": 1 if true_boundary else 0,
                    "near_word_boundary": 1 if near_boundary else 0,
                    "false_boundary_promotion": 1 if fired and not true_boundary else 0,
                    "strong_resume": strong_resume,
                }
            )

    summary = {}
    for closure_class in sorted({row["closure_class"] for row in rows}):
        subset = [row for row in rows if row["closure_class"] == closure_class]
        total = len(subset)
        if total == 0:
            continue
        summary[closure_class] = {
            "count": total,
            "pause_fire_rate": sum(int(row["pause_fired"]) for row in subset) / total,
            "strong_pause_fire_rate": sum(int(row["strong_pause_fired"]) for row in subset) / total,
            "false_boundary_promotion_rate": sum(int(row["false_boundary_promotion"]) for row in subset) / total,
            "strong_resume_rate": sum(int(row["strong_resume"]) for row in subset) / total,
            "near_boundary_rate": sum(int(row["near_word_boundary"]) for row in subset) / total,
        }
    return rows, summary


def run_audit_2(bundles: list[dict]) -> tuple[list[dict], dict]:
    candidates = []
    for lookaround in (1, 2, 3, 4):
        all_rows = []
        for bundle in bundles:
            all_rows.extend(build_direction_rows(bundle, lookaround))
        train_rows = [row for row in all_rows if row["split"] == "train"]
        test_rows = [row for row in all_rows if row["split"] == "test"]
        for min_abs_offset in (1, 2):
            train_eval = evaluate_direction_accuracy(train_rows, min_abs_offset)
            candidates.append(
                {
                    "lookaround": lookaround,
                    "min_abs_offset": min_abs_offset,
                    "train_accuracy": train_eval["accuracy"],
                    "train_count": train_eval["count"],
                    "rows": all_rows,
                    "test_rows": test_rows,
                }
            )
    best = max(candidates, key=lambda row: (row["train_accuracy"], row["train_count"]))
    test_eval = evaluate_direction_accuracy(best["test_rows"], best["min_abs_offset"])
    return best["rows"], {
        "lookaround": best["lookaround"],
        "min_abs_offset": best["min_abs_offset"],
        "train_accuracy": best["train_accuracy"],
        "train_count": best["train_count"],
        "test_accuracy": test_eval["accuracy"],
        "test_count": test_eval["count"],
        "test_majority_accuracy": test_eval["majority_accuracy"],
    }


def run_audit_3(direction_rows: list[dict], direction_summary: dict) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in direction_rows:
        grouped[row["case_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["time"])

    candidates = []
    for history_ms in (300, 450, 600, 800):
        history_frames = max(1, round(history_ms / 10.0))
        train_evals = []
        for case_id, rows in grouped.items():
            if rows[0]["split"] != "train":
                continue
            train_evals.append(evaluate_retrospective_accuracy(rows, direction_summary["min_abs_offset"], history_frames))
        total_count = sum(x["count"] for x in train_evals)
        weighted_accuracy = (
            sum(x["accuracy"] * x["count"] for x in train_evals) / total_count
            if total_count else 0.0
        )
        candidates.append({"history_ms": history_ms, "history_frames": history_frames, "train_accuracy": weighted_accuracy, "train_count": total_count})
    best = max(candidates, key=lambda row: (row["train_accuracy"], row["train_count"]))

    def aggregate(mode: str | None) -> dict:
        evals = []
        for case_id, rows in grouped.items():
            if rows[0]["split"] != "test":
                continue
            working = rows
            if mode:
                working = shuffled_rows(rows, int(hashlib.md5(f"{case_id}:{mode}".encode("utf-8")).hexdigest()[:8], 16), mode)
            evals.append(evaluate_retrospective_accuracy(working, direction_summary["min_abs_offset"], best["history_frames"]))
        total_count = sum(x["count"] for x in evals)
        return {
            "count": total_count,
            "accuracy": (sum(x["accuracy"] * x["count"] for x in evals) / total_count) if total_count else 0.0,
            "majority_accuracy": (sum(x["majority_accuracy"] * x["count"] for x in evals) / total_count) if total_count else 0.0,
        }

    real_eval = aggregate(None)
    shuffle_eval = aggregate("shuffle")
    shift_eval = aggregate("shift")
    return candidates, {
        "history_ms": best["history_ms"],
        "history_frames": best["history_frames"],
        "train_accuracy": best["train_accuracy"],
        "train_count": best["train_count"],
        "test_accuracy": real_eval["accuracy"],
        "test_count": real_eval["count"],
        "test_majority_accuracy": real_eval["majority_accuracy"],
        "shuffle_accuracy": shuffle_eval["accuracy"],
        "shift_accuracy": shift_eval["accuracy"],
    }


def render_report(audit1_summary: dict, audit2_summary: dict, audit3_summary: dict) -> str:
    lines = []
    lines.append("# Streaming Signal Audits")
    lines.append("")
    lines.append("## Audit 1: Closure vs Pause Contamination")
    lines.append("")
    for closure_class, stats in sorted(audit1_summary.items()):
        lines.append(
            f"- `{closure_class}`: count={stats['count']} "
            f"pause_fire_rate={stats['pause_fire_rate']:.3f} "
            f"strong_pause_fire_rate={stats['strong_pause_fire_rate']:.3f} "
            f"false_boundary_promotion_rate={stats['false_boundary_promotion_rate']:.3f} "
            f"strong_resume_rate={stats['strong_resume_rate']:.3f} "
            f"near_boundary_rate={stats['near_boundary_rate']:.3f}"
        )
    lines.append("")
    lines.append("## Audit 2: Held-Out Directional Correlation")
    lines.append("")
    lines.append(
        f"- best_lookaround={audit2_summary['lookaround']} "
        f"min_abs_offset={audit2_summary['min_abs_offset']} "
        f"train_accuracy={audit2_summary['train_accuracy']:.3f} "
        f"test_accuracy={audit2_summary['test_accuracy']:.3f} "
        f"test_majority_accuracy={audit2_summary['test_majority_accuracy']:.3f} "
        f"test_count={audit2_summary['test_count']}"
    )
    lines.append("")
    lines.append("## Audit 3: Retrospective Rate Proxy")
    lines.append("")
    lines.append(
        f"- best_history_ms={audit3_summary['history_ms']} "
        f"train_accuracy={audit3_summary['train_accuracy']:.3f} "
        f"test_accuracy={audit3_summary['test_accuracy']:.3f} "
        f"shuffle_accuracy={audit3_summary['shuffle_accuracy']:.3f} "
        f"shift_accuracy={audit3_summary['shift_accuracy']:.3f} "
        f"test_majority_accuracy={audit3_summary['test_majority_accuracy']:.3f} "
        f"test_count={audit3_summary['test_count']}"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run held-out streaming signal audits over latest corpus outputs.")
    parser.add_argument("--root", type=pathlib.Path, default=LATEST_RUN)
    parser.add_argument("--out", type=pathlib.Path, default=LATEST_RUN)
    args = parser.parse_args()

    case_dirs = [path for path in sorted(args.root.iterdir()) if path.is_dir()]
    bundles = [bundle for path in case_dirs if (bundle := load_case_bundle(path)) is not None]
    if not bundles:
        print("No auditable cases found.")
        return 1

    audit1_rows, audit1_summary = run_audit_1(bundles)
    direction_rows, audit2_summary = run_audit_2(bundles)
    audit3_rows, audit3_summary = run_audit_3(direction_rows, audit2_summary)

    summary = {
        "cases": len(bundles),
        "train_cases": sum(1 for bundle in bundles if bundle["split"] == "train"),
        "test_cases": sum(1 for bundle in bundles if bundle["split"] == "test"),
        "audit1": audit1_summary,
        "audit2": audit2_summary,
        "audit3": audit3_summary,
    }

    write_csv(
        args.out / "signal_audit_closure_contamination.csv",
        audit1_rows,
        [
            "case_id",
            "split",
            "closure_class",
            "phone",
            "start",
            "end",
            "duration_ms",
            "pause_fired",
            "strong_pause_fired",
            "max_pause_confidence",
            "dominant_pause_family",
            "true_word_boundary",
            "near_word_boundary",
            "false_boundary_promotion",
            "strong_resume",
        ],
    )
    write_csv(
        args.out / "signal_audit_directional_rows.csv",
        direction_rows,
        [
            "case_id",
            "split",
            "time",
            "planned_idx",
            "actual_idx",
            "offset",
            "current_class",
            "current_score",
            "ahead_score",
            "behind_score",
            "directional_score",
            "pause_confidence",
            "pause_family",
        ],
    )
    write_csv(
        args.out / "signal_audit_retrospective_candidates.csv",
        audit3_rows,
        ["history_ms", "history_frames", "train_accuracy", "train_count"],
    )
    write_text(args.out / "signal_audit_summary.json", json.dumps(summary, indent=2))
    write_text(args.out / "signal_audit_report.md", render_report(audit1_summary, audit2_summary, audit3_summary))

    print(
        "SIGNAL_AUDITS "
        f"cases={summary['cases']} train_cases={summary['train_cases']} test_cases={summary['test_cases']} "
        f"audit2_test_accuracy={audit2_summary['test_accuracy']:.4f} "
        f"audit2_test_majority={audit2_summary['test_majority_accuracy']:.4f} "
        f"audit3_test_accuracy={audit3_summary['test_accuracy']:.4f} "
        f"audit3_shuffle_accuracy={audit3_summary['shuffle_accuracy']:.4f} "
        f"audit3_shift_accuracy={audit3_summary['shift_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
