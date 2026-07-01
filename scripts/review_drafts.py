import argparse
import csv
import json
from collections import Counter

from gold_tools import case_stems, gold_draft_dir, read_json, repo_root, write_text


RISK_WEIGHTS = {
    "word_count_mismatch": 10,
    "word_text_mismatch": 6,
    "no_observed_phone_evidence": 5,
    "final_drain_fallback": 4,
    "boundary_guard_adjustment": 2,
    "large_inter_event_gap": 2,
    "low_strength_viseme": 1,
}


def phrase_count(words: list[dict]) -> int:
    indices = {int(word.get("phrase_index", -1)) for word in words if int(word.get("phrase_index", -1)) >= 0}
    return len(indices)


def sentence_count(words: list[dict]) -> int:
    indices = {int(word.get("sentence_index", -1)) for word in words if int(word.get("sentence_index", -1)) >= 0}
    return len(indices)


def risk_score(payload: dict) -> tuple[int, dict[str, int]]:
    flags = payload.get("flags", [])
    counts = Counter(str(flag.get("kind", "")).strip() for flag in flags)
    score = 0
    for kind, count in counts.items():
        score += RISK_WEIGHTS.get(kind, 1) * count

    visemes = payload.get("visemes", [])
    fallback_count = sum(1 for viseme in visemes if viseme.get("commit_reason") == "offline_committed_fallback")
    mismatch_count = sum(1 for viseme in visemes if "label_mismatch" in str(viseme.get("alignment_reason", "")))
    monotonic_clamp_count = sum(1 for viseme in visemes if "monotonic_" in str(viseme.get("alignment_reason", "")))
    unmapped_count = sum(1 for viseme in visemes if not bool(viseme.get("mapped_to_observed_speech", False)))

    score += fallback_count * 4
    score += mismatch_count * 2
    score += monotonic_clamp_count * 2
    score += unmapped_count * 3

    counts["offline_committed_fallback"] = fallback_count
    counts["alignment_label_mismatch"] = mismatch_count
    counts["monotonic_adjustment"] = monotonic_clamp_count
    counts["unmapped_viseme"] = unmapped_count
    return score, dict(counts)


def draft_status(payload: dict) -> str:
    approval = payload.get("approval", {})
    review_layers = payload.get("review_layers", {})
    if approval.get("status") == "approved_gold":
        return "approved"
    if any(layer.get("reviewed", False) for layer in review_layers.values()):
        return "in_review"
    return "draft"


def easy_candidate(row: dict) -> bool:
    return (
        row["risk_score"] <= 3
        and row["flag_count"] <= 2
        and row["fallback_count"] == 0
        and row["unmapped_viseme_count"] == 0
        and row["word_text_mismatch_count"] == 0
        and row["word_count_mismatch_count"] == 0
    )


def build_case_row(case_id: str, payload: dict) -> dict[str, object]:
    visemes = payload.get("visemes", [])
    words = payload.get("gold_words", [])
    speech_regions = payload.get("speech_regions", [])
    word_heads = payload.get("word_heads", [])
    score, counts = risk_score(payload)

    row = {
        "case_id": case_id,
        "status": draft_status(payload),
        "risk_score": score,
        "flag_count": int(payload.get("summary", {}).get("flag_count", len(payload.get("flags", [])))),
        "speech_region_count": len(speech_regions),
        "word_count": len(words),
        "word_head_count": len(word_heads),
        "viseme_count": len(visemes),
        "phrase_count": phrase_count(words),
        "sentence_count": sentence_count(words),
        "fallback_count": counts.get("offline_committed_fallback", 0),
        "unmapped_viseme_count": counts.get("unmapped_viseme", 0),
        "word_text_mismatch_count": counts.get("word_text_mismatch", 0),
        "word_count_mismatch_count": counts.get("word_count_mismatch", 0),
        "label_mismatch_count": counts.get("alignment_label_mismatch", 0),
        "monotonic_adjustment_count": counts.get("monotonic_adjustment", 0),
        "flags_by_kind": counts,
    }
    row["bucket"] = "likely_easy" if easy_candidate(row) else "needs_review"
    return row


def write_csv(rows: list[dict[str, object]], path) -> None:
    fieldnames = [
        "case_id",
        "status",
        "bucket",
        "risk_score",
        "flag_count",
        "fallback_count",
        "unmapped_viseme_count",
        "word_text_mismatch_count",
        "word_count_mismatch_count",
        "label_mismatch_count",
        "monotonic_adjustment_count",
        "speech_region_count",
        "word_count",
        "word_head_count",
        "viseme_count",
        "phrase_count",
        "sentence_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def markdown_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| case | bucket | risk | flags | fallback | unmapped | word_text | label_mismatch | speech | words | visemes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['bucket']} | {row['risk_score']} | {row['flag_count']} | "
            f"{row['fallback_count']} | {row['unmapped_viseme_count']} | {row['word_text_mismatch_count']} | "
            f"{row['label_mismatch_count']} | {row['speech_region_count']} | {row['word_count']} | {row['viseme_count']} |"
        )
    return lines


def write_markdown(rows: list[dict[str, object]], path, top_n: int) -> None:
    ranked = sorted(rows, key=lambda row: (-int(row["risk_score"]), str(row["case_id"])))
    hardest = ranked[:top_n]
    easiest = [row for row in sorted(rows, key=lambda row: (row["risk_score"], row["flag_count"], row["case_id"])) if row["bucket"] == "likely_easy"][:top_n]

    lines = [
        "# Draft Review Report",
        "",
        f"cases: {len(rows)}",
        f"likely_easy: {sum(1 for row in rows if row['bucket'] == 'likely_easy')}",
        f"needs_review: {sum(1 for row in rows if row['bucket'] == 'needs_review')}",
        "",
        "## Hardest Cases",
        "",
    ]
    lines.extend(markdown_table(hardest))
    lines.extend(["", "## Likely Easy Cases", ""])
    lines.extend(markdown_table(easiest))
    lines.extend(["", "## Review Guidance", ""])
    lines.extend(
        [
            "- Review all `needs_review` cases manually.",
            "- Spot-check a sample of `likely_easy` cases; if they hold up, batch-approve that bucket.",
            "- Prioritize speech boundaries first, then word heads, then dense visemes.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank draft gold cases by review risk and emit reviewer-assist reports.")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="print ranked rows as JSON to stdout")
    args = parser.parse_args()

    cases = args.cases or case_stems()
    rows: list[dict[str, object]] = []
    for case_id in cases:
        draft_path = gold_draft_dir(case_id) / "draft.annotation.json"
        if not draft_path.exists():
            continue
        payload = read_json(draft_path)
        rows.append(build_case_row(case_id, payload))

    rows.sort(key=lambda row: (-int(row["risk_score"]), str(row["case_id"])))

    report_root = repo_root() / "outputs" / "gold_review"
    write_csv(rows, report_root / "latest" / "draft_review_ranked.csv")
    write_markdown(rows, report_root / "latest" / "draft_review_report.md", max(1, args.top))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    easy = sum(1 for row in rows if row["bucket"] == "likely_easy")
    hard = len(rows) - easy
    print(f"cases={len(rows)} likely_easy={easy} needs_review={hard}")
    print("hardest:")
    for row in rows[: max(1, args.top)]:
        print(
            f"  {row['case_id']} risk={row['risk_score']} flags={row['flag_count']} "
            f"fallback={row['fallback_count']} unmapped={row['unmapped_viseme_count']} "
            f"word_text={row['word_text_mismatch_count']} label_mismatch={row['label_mismatch_count']}"
        )
    print(f"wrote {report_root / 'latest' / 'draft_review_ranked.csv'}")
    print(f"wrote {report_root / 'latest' / 'draft_review_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
