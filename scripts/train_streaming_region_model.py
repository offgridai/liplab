import argparse
import csv
import json
import pathlib
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "runs" / "latest"
GOLD_ROOT = ROOT / "inputs" / "gold"
REPORT_PATH = ROOT / "outputs" / "streaming_region_model_report.json"
MODEL_PATH = ROOT / "offgrid_dropin" / "Private" / "Lipsync" / "OffgridAIStreamingRegionModel.inl"

BASE_FEATURE_NAMES = (
    "log_rms",
    "rms_norm",
    "delta_rms",
    "flux",
    "zcr",
    "low_band",
    "mid_band",
    "high_band",
    "centroid",
    "periodicity",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_group(case_name):
    match = re.search(r"_Line_([0-9A-F]+)_", case_name)
    return match.group(1) if match else case_name


def rolling_mean(values, before, after):
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    output = np.empty_like(values)
    for index in range(values.shape[0]):
        begin = max(0, index - before)
        end = min(values.shape[0], index + after + 1)
        output[index] = (prefix[end] - prefix[begin]) / max(end - begin, 1)
    return output


def case_features(frame_rows):
    rms = np.asarray([max(float(row["rms"]), 1.0e-6) for row in frame_rows])
    base = np.column_stack((
        np.log(rms),
        [float(row["rms_norm"]) for row in frame_rows],
        [float(row["delta_rms"]) for row in frame_rows],
        [float(row["flux"]) for row in frame_rows],
        [float(row["zcr"]) for row in frame_rows],
        [float(row["low_band"]) for row in frame_rows],
        [float(row["mid_band"]) for row in frame_rows],
        [float(row["high_band"]) for row in frame_rows],
        [float(row["centroid"]) for row in frame_rows],
        [float(row["periodicity"]) for row in frame_rows],
    ))
    # A frame is finalized 100 ms later, well inside the existing postroll.
    past_100 = rolling_mean(base, 10, 0)
    future_100 = rolling_mean(base, 0, 10)
    neighborhood_200 = rolling_mean(base, 10, 10)
    return np.hstack((base, past_100, future_100, neighborhood_200))


def feature_names():
    names = list(BASE_FEATURE_NAMES)
    for prefix in ("past100", "future100", "around200"):
        names.extend(f"{prefix}_{name}" for name in BASE_FEATURE_NAMES)
    return names


def load_dataset(run_root):
    all_x = []
    all_y = []
    all_groups = []
    cases = 0
    case_metadata = []
    offset = 0
    for case_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        frame_path = case_dir / "occupancy_frames.csv"
        gold_path = GOLD_ROOT / case_dir.name / "speech.csv"
        if not frame_path.exists() or not gold_path.exists():
            continue
        frames = read_rows(frame_path)
        gold = read_rows(gold_path)
        if not frames or not gold:
            continue
        intervals = [(float(row["start"]), float(row["end"])) for row in gold]
        centers = np.asarray([float(row["center"]) for row in frames])
        labels = np.asarray([
            int(any(start <= center < end for start, end in intervals))
            for center in centers
        ])
        features = case_features(frames)
        all_x.append(features)
        all_y.append(labels)
        all_groups.extend([source_group(case_dir.name)] * len(frames))
        case_metadata.append({
            "name": case_dir.name,
            "begin": offset,
            "end": offset + len(frames),
            "intervals": intervals,
        })
        offset += len(frames)
        cases += 1
    return np.vstack(all_x), np.concatenate(all_y), np.asarray(all_groups), cases, case_metadata


def normalize(train_x, test_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale, mean, scale


def best_threshold(labels, probabilities):
    best = None
    for threshold in np.arange(0.20, 0.81, 0.01):
        predictions = probabilities >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        score = min(precision, recall), f1
        if best is None or score > best[0]:
            best = score, float(threshold), float(precision), float(recall), float(f1)
    return best[1:]


def decoded_regions(
    probabilities,
    threshold,
    minimum_speech_frames,
    minimum_pause_frames,
):
    regions = []
    in_speech = False
    candidate_start = None
    quiet_start = None
    for index, probability in enumerate(probabilities):
        speech = probability >= threshold
        if not in_speech:
            if speech:
                candidate_start = index if candidate_start is None else candidate_start
                if index - candidate_start + 1 >= minimum_speech_frames:
                    regions.append([candidate_start * 0.01, (index + 1) * 0.01])
                    in_speech = True
                    quiet_start = None
            else:
                candidate_start = None
        elif speech:
            quiet_start = None
            regions[-1][1] = (index + 1) * 0.01
        else:
            quiet_start = index if quiet_start is None else quiet_start
            if index - quiet_start + 1 >= minimum_pause_frames:
                regions[-1][1] = quiet_start * 0.01
                in_speech = False
                candidate_start = None
                quiet_start = None
    if in_speech and quiet_start is not None:
        regions[-1][1] = quiet_start * 0.01
    return regions


def monotonic_pair_match_count(predicted, references, tolerance=0.100):
    predicted_pairs = [(predicted[index][1], predicted[index + 1][0]) for index in range(len(predicted) - 1)]
    reference_pairs = [(references[index][1], references[index + 1][0]) for index in range(len(references) - 1)]
    n, m = len(predicted_pairs), len(reference_pairs)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            compatible = all(abs(predicted_pairs[i - 1][edge] - reference_pairs[j - 1][edge]) <= tolerance for edge in (0, 1))
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + int(compatible))
    return len(predicted_pairs), len(reference_pairs), dp[n][m]


def tune_region_decoder(probabilities, case_metadata):
    best = None
    for threshold in np.arange(0.10, 0.61, 0.05):
        for minimum_speech_frames in (2, 3, 4, 5, 6):
            for minimum_pause_frames in (8, 10, 12, 14, 16, 18, 20):
                predicted_total = reference_total = matched_total = 0
                for case in case_metadata:
                    case_probabilities = probabilities[case["begin"]:case["end"]]
                    predicted = decoded_regions(case_probabilities, threshold, minimum_speech_frames, minimum_pause_frames)
                    predicted_count, reference_count, matched = monotonic_pair_match_count(predicted, case["intervals"])
                    predicted_total += predicted_count
                    reference_total += reference_count
                    matched_total += matched
                precision = matched_total / predicted_total if predicted_total else 0.0
                recall = matched_total / reference_total if reference_total else 0.0
                f1 = 2.0 * matched_total / (predicted_total + reference_total) if predicted_total + reference_total else 0.0
                score = min(precision, recall), f1
                if best is None or score > best[0]:
                    best = score, {
                        "threshold": float(threshold),
                        "minimum_speech_frames": minimum_speech_frames,
                        "minimum_pause_frames": minimum_pause_frames,
                        "predicted_pair_count": predicted_total,
                        "reference_pair_count": reference_total,
                        "matched_pair_count": matched_total,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
    return best[1]


def write_model(model, mean, scale, region_decoder):
    weights = model.coef_[0] / scale
    bias = float(model.intercept_[0] - np.dot(model.coef_[0], mean / scale))
    lines = [
        "// Generated by scripts/train_streaming_region_model.py.",
        f"static constexpr int32 StreamingRegionFeatureCount = {len(weights)};",
        f"static constexpr float StreamingRegionBias = {bias:.9g}f;",
        f"static constexpr float StreamingRegionSpeechThreshold = {region_decoder['threshold']:.9g}f;",
        f"static constexpr int32 StreamingRegionMinimumSpeechFrames = {region_decoder['minimum_speech_frames']};",
        f"static constexpr int32 StreamingRegionMinimumPauseFrames = {region_decoder['minimum_pause_frames']};",
        "static constexpr float StreamingRegionWeights[StreamingRegionFeatureCount] = {",
    ]
    lines.extend(f"    {value:.9g}f," for value in weights)
    lines.append("};")
    MODEL_PATH.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN)
    parser.add_argument("--write-model", action="store_true")
    args = parser.parse_args()

    x, y, groups, case_count, case_metadata = load_dataset(args.run_root)
    probabilities = np.zeros(len(y))
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        train_x, test_x, _, _ = normalize(x[train], x[test])
        model = LogisticRegression(C=0.25, class_weight="balanced", max_iter=400)
        model.fit(train_x, y[train])
        probabilities[test] = model.predict_proba(test_x)[:, 1]
        threshold, precision, recall, f1 = best_threshold(y[test], probabilities[test])
        folds.append({"fold": fold, "threshold": threshold, "precision": precision, "recall": recall, "f1": f1})

    threshold, precision, recall, f1 = best_threshold(y, probabilities)
    region_decoder = tune_region_decoder(probabilities, case_metadata)
    report = {
        "case_count": case_count,
        "frame_count": int(len(y)),
        "speech_frame_rate": float(y.mean()),
        "held_out_group_count": int(len(np.unique(groups))),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "folds": folds,
        "region_decoder": region_decoder,
        "feature_names": feature_names(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.write_model:
        normalized_x, _, mean, scale = normalize(x, x)
        model = LogisticRegression(C=0.25, class_weight="balanced", max_iter=400)
        model.fit(normalized_x, y)
        write_model(model, mean, scale, region_decoder)
        print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
