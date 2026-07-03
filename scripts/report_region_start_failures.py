import argparse
import csv
import pathlib
from collections import Counter

from grade_summary import load_case_grades


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _spans(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for row in rows:
        start = _safe_float(row.get("start"))
        end = _safe_float(row.get("end"))
        if end > start:
            spans.append((start, end))
    spans.sort()
    return spans


def _overlap_seconds(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _pair_by_overlap_or_nearest(
    gold: list[tuple[float, float]], predicted: list[tuple[float, float]]
) -> tuple[list[tuple[int, int, str]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(gold):
            overlap = _overlap_seconds(p, g)
            if overlap > 0.0:
                pairs.append((overlap, pi, gi))
    pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    matches: list[tuple[int, int, str]] = []
    for _, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gold:
            continue
        matched_pred.add(pi)
        matched_gold.add(gi)
        matches.append((pi, gi, "overlap"))

    unmatched_pred = [pi for pi in range(len(predicted)) if pi not in matched_pred]
    unmatched_gold = [gi for gi in range(len(gold)) if gi not in matched_gold]
    for pi in unmatched_pred:
        if not unmatched_gold:
            break
        pred_center = 0.5 * (predicted[pi][0] + predicted[pi][1])
        best_idx = min(
            range(len(unmatched_gold)),
            key=lambda idx: abs(pred_center - 0.5 * (gold[unmatched_gold[idx]][0] + gold[unmatched_gold[idx]][1])),
        )
        gi = unmatched_gold.pop(best_idx)
        matched_pred.add(pi)
        matched_gold.add(gi)
        matches.append((pi, gi, "nearest"))

    pred_only = [pi for pi in range(len(predicted)) if pi not in matched_pred]
    gold_only = [gi for gi in range(len(gold)) if gi not in matched_gold]
    return matches, pred_only, gold_only


def _classify_failure(
    gold: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
    matches: list[tuple[int, int, str]],
    pred_only: list[int],
    gold_only: list[int],
    start_error_ms: float,
) -> str:
    if gold_only and not pred_only:
        return "under-segmented_missing_region"
    if pred_only and not gold_only:
        return "over-segmented_extra_region"
    if gold_only and pred_only:
        return "mixed_segmentation_drift"
    if any(mode == "nearest" for _, _, mode in matches):
        return "mispaired_region_order"

    if not gold or not predicted:
        return "missing_region_data"

    first_gold_start = gold[0][0]
    first_pred_start = predicted[0][0]
    delta_ms = (first_pred_start - first_gold_start) * 1000.0
    if delta_ms >= 120.0:
        return "late_onset_confirmation"
    if delta_ms <= -120.0:
        return "early_false_attack"
    if start_error_ms >= 120.0:
        return "local_start_timing_error"
    return "moderate_start_noise"


def _render_report(rows: list[dict], limit: int) -> str:
    analyzed: list[dict] = []
    class_counts: Counter[str] = Counter()

    for row in rows:
        speech_diag = row.get("speech_region_diagnostics", {})
        overlap = speech_diag.get("overlap", {})
        start_error_ms = float(overlap.get("mean_start_error_ms", 0.0))
        gold = _spans(row.get("gold_speech_rows", []))
        predicted = _spans(row.get("predicted_speech_rows", []))
        matches, pred_only, gold_only = _pair_by_overlap_or_nearest(gold, predicted)
        classification = _classify_failure(gold, predicted, matches, pred_only, gold_only, start_error_ms)
        class_counts[classification] += 1
        analyzed.append(
            {
                "case": row.get("case", ""),
                "classification": classification,
                "start_error_ms": start_error_ms,
                "end_error_ms": float(overlap.get("mean_end_error_ms", 0.0)),
                "gold_regions": len(gold),
                "pred_regions": len(predicted),
                "overlap_matched": int(overlap.get("overlap_matched_count", 0)),
                "nearest_matched": int(overlap.get("nearest_matched_count", 0)),
                "pred_only": len(pred_only),
                "gold_only": len(gold_only),
                "first_gold_start": gold[0][0] if gold else 0.0,
                "first_pred_start": predicted[0][0] if predicted else 0.0,
                "gap_ambiguous": int(row.get("gap_candidate_summary", {}).get("ambiguous_count", 0)),
            }
        )

    analyzed.sort(key=lambda item: (-item["start_error_ms"], item["case"]))
    worst = analyzed[:limit]

    lines = []
    lines.append("# Speech Region Start Failure Report")
    lines.append("")
    lines.append(f"Cases analyzed: {len(analyzed)}")
    lines.append("")
    lines.append("## Failure Classes")
    lines.append("")
    for label, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append(f"## Worst {len(worst)} Cases By Overlap-Paired Start Error")
    lines.append("")
    lines.append("| Case | Class | Start ms | End ms | Gold | Pred | Pred-only | Gold-only | First gold start | First pred start | Ambiguous gaps |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in worst:
        lines.append(
            f"| `{item['case']}` | `{item['classification']}` | {item['start_error_ms']:.1f} | "
            f"{item['end_error_ms']:.1f} | {item['gold_regions']} | {item['pred_regions']} | "
            f"{item['pred_only']} | {item['gold_only']} | {item['first_gold_start']:.3f} | "
            f"{item['first_pred_start']:.3f} | {item['gap_ambiguous']} |"
        )
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `under-segmented_missing_region`: a gold region never gets its own predicted region, so starts drift because earlier speech stayed merged.")
    lines.append("- `over-segmented_extra_region`: the detector inserts an extra region, often from a medium pause that should have stayed inside a sentence.")
    lines.append("- `late_onset_confirmation`: the first predicted start is materially later than the first gold start even without count drift.")
    lines.append("- `early_false_attack`: the detector opens too early, usually from leading noise or an overly eager onset.")
    lines.append("- `mispaired_region_order`: overlap matching had to fall back to nearest pairing, meaning indexing no longer reflects the same regions.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report worst speech-region start failures from latest corpus outputs.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "outputs" / "runs" / "latest")
    parser.add_argument("--gold-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "inputs" / "gold")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    _, graded, _ = load_case_grades(args.root, args.gold_root)
    report = _render_report(graded, max(args.limit, 1))
    out_path = args.root / "speech_region_start_failures.md"
    out_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
