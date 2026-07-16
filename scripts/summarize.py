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

    diagnostics = compute_summary(rows, graded, ungraded)
    region_categories = diagnostics.get("streaming_region_boundary_case_categories", {})

    # Keep the aggregate report small and outcome-oriented. Detailed research
    # metrics remain in each case directory for diagnosis, but do not define
    # whether the runtime is successful.
    summary = {
        "cases": diagnostics["cases"],
        "graded_cases": diagnostics["graded_cases"],
        "order_fail_cases": diagnostics["order_fail_cases"],
        "degenerate_cases": diagnostics["degenerate_cases"],
        "visible_articulation_coverage_rate": diagnostics["visible_articulation_coverage_rate"],
        "visible_articulation_center_ms": diagnostics["visible_articulation_center_ms"],
        "streaming_region_boundary_pair_recall": diagnostics.get(
            "streaming_region_boundary_pair_recall", 0.0
        ),
        "streaming_region_boundary_pair_f1": diagnostics.get(
            "streaming_region_boundary_pair_f1", 0.0
        ),
        "streaming_region_undersegmented_cases": region_categories.get("undersegmented", 0),
        "uncommitted_visible_event_count": diagnostics.get(
            "audio_health_uncommitted_visible_count", 0
        ),
    }

    focus_path = latest / "focus_alignment_summary.json"
    if focus_path.exists():
        summary["focused_alignment"] = json.loads(focus_path.read_text(encoding="utf-8"))

    summary_path = latest / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"Cases: total={summary['cases']} graded={summary['graded_cases']} "
        f"order_failures={summary['order_fail_cases']} degenerate={summary['degenerate_cases']}"
    )
    print(
        "Aggregate guardrails: "
        f"visible_coverage={summary['visible_articulation_coverage_rate']:.3f} "
        f"visible_center_ms={summary['visible_articulation_center_ms']:.1f} "
        f"region_recall={summary['streaming_region_boundary_pair_recall']:.3f} "
        f"region_f1={summary['streaming_region_boundary_pair_f1']:.3f} "
        f"undersegmented_cases={summary['streaming_region_undersegmented_cases']} "
        f"uncommitted_visible={summary['uncommitted_visible_event_count']}"
    )
    focus = summary.get("focused_alignment")
    if focus:
        print(
            "Focused outcomes: "
            f"region_mae={focus['region_head_mae_ms']:.1f}ms "
            f"region_p90={focus['region_head_p90_abs_ms']:.1f}ms "
            f"region_coverage={focus['region_head_coverage_rate']:.3f} "
            f"pause_clean={focus['pause_clean_rate']:.3f} "
            f"word_mae={focus['word_head_mae_ms']:.1f}ms "
            f"word_p90={focus['word_head_p90_abs_ms']:.1f}ms "
            f"word_coverage={focus['word_head_coverage_rate']:.3f} "
            f"event_completion={focus['event_completion_rate']:.3f}"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
