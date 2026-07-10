import json
import pathlib
import sys

from grade_summary import compute_summary, load_case_grades


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    rows, graded, ungraded = load_case_grades(latest, root / "inputs" / "gold")
    if not rows:
        print("No grade outputs found. Run the corpus first.", file=sys.stderr)
        return 2

    summary = compute_summary(rows, graded, ungraded)
    summary_path = latest / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"Cases: total={summary['cases']} graded={summary['graded_cases']} "
        f"qualified={summary['qualified_cases']} mismatched_regions={summary['speech_region_count_mismatch_cases']}"
    )
    print(
        f"Speech regions: f1={summary['speech_f1']:.3f} "
        f"start_ms={summary['speech_boundary_start_ms']:.1f} "
        f"end_ms={summary['speech_boundary_end_ms']:.1f} "
        f"tail_ms={summary['speech_tail_leakage_ms']:.1f}"
    )
    print(
        f"Words: f1={summary['word_f1']:.3f} "
        f"start_ms={summary['word_head_start_ms']:.1f} "
        f"duration_ms={summary['word_duration_ms']:.1f}"
    )
    print(
        f"Phonemes: coverage={summary['phoneme_coverage_rate']:.3f} "
        f"center_ms={summary['phoneme_center_ms']:.1f} "
        f"start_ms={summary['phoneme_start_ms']:.1f} "
        f"end_ms={summary['phoneme_end_ms']:.1f}"
    )
    print(
        f"Intra-word: coverage={summary['intra_word_coverage_rate']:.3f} "
        f"center_ms={summary['intra_word_center_ms']:.1f}"
    )
    if summary["streaming_pause_top1_accuracy"] > 0.0 or summary["streaming_word_boundary_f1"] > 0.0:
        print(
            f"Streaming detector: pause_top1={summary['streaming_pause_top1_accuracy']:.3f} "
            f"word_boundary_f1={summary['streaming_word_boundary_f1']:.3f}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
