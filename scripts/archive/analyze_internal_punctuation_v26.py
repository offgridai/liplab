#!/usr/bin/env python3
"""V26 offline internal-punctuation pause estimator.

This diagnostic-only script measures non-sentence punctuation boundaries inside
strict sentence islands. It does not change runtime behavior.

For each transcript boundary like:

    word_before , word_after

it joins the boundary to per-event full-WAV evidence timing from
``timing_diagnostics.csv`` and committed/render timing from
``committed_events.csv``. The goal is to estimate how much local pause pocket
internal punctuation should reserve, independent of sentence-island launch.

Typical use:

  python scripts/analyze_internal_punctuation_v26.py \
      --run experiments/runs/local_run \
      --inputs C:\\git\\liplab\\inputs \
      --out experiments/runs/local_run/v26_internal_punctuation

Outputs:
  v26_internal_punctuation_notes.md
  v26_punctuation_boundary_rows.csv
  v26_pause_estimates_by_punctuation.csv
  v26_pause_estimates_by_context.csv
  v26_worst_bridged_boundaries.csv
  v26_stage_counts.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
SENTENCE_PUNCT = {".", "?", "!"}
INTERNAL_PUNCT_CHARS = {",", ":", ";", "-", "—", "–"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else 0.0


def median(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.median(vals) if vals else 0.0


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if abs(den) > 1e-9 else default


@dataclass(frozen=True)
class Boundary:
    punct: str
    word_before_index: int
    word_after_index: int
    word_before: str
    word_after: str
    sentence_like_nearby: bool
    raw_between: str


@dataclass
class WordTiming:
    word_index: int
    source_word: str = ""
    event_count: int = 0
    evidence_event_count: int = 0
    evidence_start: float = math.inf
    evidence_end: float = -math.inf
    evidence_first_center: float = math.inf
    evidence_last_center: float = -math.inf
    evidence_fallback_count: int = 0
    committed_start: float = math.inf
    committed_end: float = -math.inf
    committed_first_center: float = math.inf
    committed_last_center: float = -math.inf
    text_island_index: int = -1

    def valid_evidence(self) -> bool:
        return self.evidence_event_count > 0 and self.evidence_start < math.inf and self.evidence_end > -math.inf

    def valid_committed(self) -> bool:
        return self.event_count > 0 and self.committed_start < math.inf and self.committed_end > -math.inf


def normalize_punctuation(raw: str) -> str:
    if "..." in raw or "…" in raw:
        return "ellipsis"
    if ";" in raw:
        return "semicolon"
    if ":" in raw:
        return "colon"
    if "—" in raw or "–" in raw or "-" in raw:
        return "dash"
    if "," in raw:
        return "comma"
    # Preserve unknown internal punctuation as a stable token for diagnostics.
    return raw.strip() or "unknown"


def is_sentence_punct_only(raw: str) -> bool:
    stripped = raw.strip()
    if not stripped:
        return False
    # Ellipsis is intentionally not sentence punctuation in the current architecture.
    if "..." in stripped or "…" in stripped:
        return False
    chars = [ch for ch in stripped if not ch.isspace() and ch not in {'"', "'", "”", "“", "’", ")", "(", "]", "["}]
    return bool(chars) and all(ch in SENTENCE_PUNCT for ch in chars)


def contains_internal_punctuation(raw: str) -> bool:
    if "..." in raw or "…" in raw:
        return True
    return any(ch in raw for ch in INTERNAL_PUNCT_CHARS)


def parse_internal_boundaries(transcript: str) -> list[Boundary]:
    words = list(WORD_RE.finditer(transcript))
    out: list[Boundary] = []
    if len(words) < 2:
        return out
    for i in range(len(words) - 1):
        before = words[i]
        after = words[i + 1]
        between = transcript[before.end():after.start()]
        if not between.strip():
            continue
        if is_sentence_punct_only(between):
            continue
        if not contains_internal_punctuation(between):
            continue
        punct = normalize_punctuation(between)
        # Skip decimal/hyphenated word artifacts where the punctuation is part of
        # a tokenization ambiguity rather than a prosody boundary.
        if punct == "dash" and before.end() + 1 == after.start():
            continue
        out.append(Boundary(
            punct=punct,
            word_before_index=i,
            word_after_index=i + 1,
            word_before=before.group(0),
            word_after=after.group(0),
            sentence_like_nearby=any(ch in between for ch in SENTENCE_PUNCT),
            raw_between=between,
        ))
    return out


def collect_word_timings(case_dir: Path) -> dict[int, WordTiming]:
    timings: dict[int, WordTiming] = {}

    for row in read_csv(case_dir / "committed_events.csv"):
        wi = as_int(row, "WordIndex", -1)
        if wi < 0:
            continue
        wt = timings.setdefault(wi, WordTiming(word_index=wi))
        wt.event_count += 1
        wt.source_word = wt.source_word or row.get("SourceWord", "")
        wt.text_island_index = as_int(row, "TextIslandIndex", wt.text_island_index)
        center = as_float(row, "CommittedPlaybackCenterSec", as_float(row, "FinalRenderCenterSeconds", 0.0))
        start = as_float(row, "RenderStartSec", center)
        end = as_float(row, "RenderEndSec", center)
        wt.committed_start = min(wt.committed_start, start)
        wt.committed_end = max(wt.committed_end, end)
        wt.committed_first_center = min(wt.committed_first_center, center)
        wt.committed_last_center = max(wt.committed_last_center, center)

    for row in read_csv(case_dir / "timing_diagnostics.csv"):
        wi = as_int(row, "WordIndex", -1)
        if wi < 0:
            continue
        wt = timings.setdefault(wi, WordTiming(word_index=wi))
        wt.source_word = wt.source_word or row.get("SourceWord", "")
        start = as_float(row, "EvidenceStartSec", 0.0)
        center = as_float(row, "EvidenceCenterSec", 0.0)
        end = as_float(row, "EvidenceEndSec", 0.0)
        if end <= 0.0 and center <= 0.0:
            continue
        wt.evidence_event_count += 1
        wt.evidence_start = min(wt.evidence_start, start if start > 0.0 else center)
        wt.evidence_end = max(wt.evidence_end, end if end > 0.0 else center)
        wt.evidence_first_center = min(wt.evidence_first_center, center)
        wt.evidence_last_center = max(wt.evidence_last_center, center)
        wt.evidence_fallback_count += 1 if as_int(row, "EvidenceFallback", 0) != 0 else 0

    return timings


def case_id_from_path(path: Path) -> str:
    return path.stem


def iter_transcripts(inputs: Path) -> list[Path]:
    transcript_dir = inputs / "transcripts"
    if not transcript_dir.exists() and (inputs / "inputs" / "transcripts").exists():
        transcript_dir = inputs / "inputs" / "transcripts"
    return sorted(transcript_dir.glob("case_*.txt"))


def classify_context(boundary: Boundary, before: WordTiming, after: WordTiming) -> str:
    if before.text_island_index >= 0 and after.text_island_index >= 0 and before.text_island_index != after.text_island_index:
        return "crosses_text_island_unexpected"
    if boundary.punct == "comma":
        if boundary.word_before.lower() in {"yes", "no", "okay", "ok", "well"}:
            return "comma_discourse_marker"
        return "comma_inline"
    if boundary.punct in {"semicolon", "colon"}:
        return "strong_internal_boundary"
    if boundary.punct in {"dash", "ellipsis"}:
        return "hesitation_or_dash"
    return "other_internal_boundary"


def build_rows(run: Path, inputs: Path) -> tuple[list[dict[str, object]], Counter]:
    counts: Counter = Counter()
    rows: list[dict[str, object]] = []
    per_case = run / "per_case"
    transcripts = iter_transcripts(inputs)
    counts["transcripts_seen"] = len(transcripts)
    if not per_case.exists():
        counts["missing_per_case_dir"] += 1
        return rows, counts

    for transcript_path in transcripts:
        case_id = case_id_from_path(transcript_path)
        case_dir = per_case / case_id
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        boundaries = parse_internal_boundaries(text)
        counts["boundaries_from_transcripts"] += len(boundaries)
        if not case_dir.exists():
            counts["missing_case_dirs"] += 1
            continue
        timings = collect_word_timings(case_dir)
        if not timings:
            counts["cases_without_event_timing"] += 1
            continue
        for b in boundaries:
            before = timings.get(b.word_before_index)
            after = timings.get(b.word_after_index)
            if before is None or after is None:
                counts["boundaries_missing_word_timing"] += 1
                continue
            same_text_island = before.text_island_index == after.text_island_index and before.text_island_index >= 0
            evidence_ok = before.valid_evidence() and after.valid_evidence()
            committed_ok = before.valid_committed() and after.valid_committed()
            if not evidence_ok:
                counts["boundaries_missing_evidence"] += 1
            if not committed_ok:
                counts["boundaries_missing_committed"] += 1

            evidence_gap = after.evidence_start - before.evidence_end if evidence_ok else 0.0
            evidence_center_gap = after.evidence_first_center - before.evidence_last_center if evidence_ok else 0.0
            committed_render_gap = after.committed_start - before.committed_end if committed_ok else 0.0
            committed_center_gap = after.committed_first_center - before.committed_last_center if committed_ok else 0.0
            visible_gap = max(0.0, committed_render_gap)
            visible_overlap = max(0.0, -committed_render_gap)
            evidence_pause = max(0.0, evidence_gap)
            underrepresented = max(0.0, evidence_pause - visible_gap)
            bridge_ratio = safe_div(visible_gap, evidence_pause, 1.0 if evidence_pause <= 1e-6 else 0.0)
            rows.append({
                "case_id": case_id,
                "punctuation": b.punct,
                "context": classify_context(b, before, after),
                "word_before_index": b.word_before_index,
                "word_after_index": b.word_after_index,
                "word_before": b.word_before,
                "word_after": b.word_after,
                "raw_between": b.raw_between.replace("\n", "\\n"),
                "same_text_island": int(same_text_island),
                "text_island_index": before.text_island_index if same_text_island else -1,
                "evidence_ok": int(evidence_ok),
                "committed_ok": int(committed_ok),
                "evidence_gap_ms": evidence_gap * 1000.0,
                "evidence_pause_ms": evidence_pause * 1000.0,
                "evidence_center_gap_ms": evidence_center_gap * 1000.0,
                "committed_render_gap_ms": committed_render_gap * 1000.0,
                "committed_center_gap_ms": committed_center_gap * 1000.0,
                "visible_gap_ms": visible_gap * 1000.0,
                "visible_overlap_ms": visible_overlap * 1000.0,
                "underrepresented_pause_ms": underrepresented * 1000.0,
                "visible_gap_to_evidence_pause_ratio": bridge_ratio,
                "before_evidence_end_sec": before.evidence_end if evidence_ok else 0.0,
                "after_evidence_start_sec": after.evidence_start if evidence_ok else 0.0,
                "before_committed_render_end_sec": before.committed_end if committed_ok else 0.0,
                "after_committed_render_start_sec": after.committed_start if committed_ok else 0.0,
                "before_evidence_fallback_count": before.evidence_fallback_count,
                "after_evidence_fallback_count": after.evidence_fallback_count,
            })
            counts["boundary_rows"] += 1
            if evidence_pause > 0.050:
                counts["boundaries_with_50ms_evidence_pause"] += 1
            if underrepresented > 0.050:
                counts["boundaries_underrepresented_by_50ms"] += 1
    return rows, counts


def summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row.get("evidence_ok", 0)) == 0:
            continue
        bucket_key = tuple(str(row.get(k, "")) for k in keys)
        buckets[bucket_key].append(row)

    out: list[dict[str, object]] = []
    for bucket_key, bucket_rows in sorted(buckets.items()):
        evidence_pause = [float(r["evidence_pause_ms"]) for r in bucket_rows]
        raw_gap = [float(r["evidence_gap_ms"]) for r in bucket_rows]
        visible_gap = [float(r["visible_gap_ms"]) for r in bucket_rows if int(r.get("committed_ok", 0))]
        underrep = [float(r["underrepresented_pause_ms"]) for r in bucket_rows if int(r.get("committed_ok", 0))]
        overlap = [float(r["visible_overlap_ms"]) for r in bucket_rows if int(r.get("committed_ok", 0))]
        positive = [v for v in evidence_pause if v > 1e-6]
        rec = max(0.0, min(350.0, percentile(positive, 0.50) if positive else percentile(evidence_pause, 0.75)))
        row: dict[str, object] = {k: v for k, v in zip(keys, bucket_key)}
        row.update({
            "count": len(bucket_rows),
            "positive_pause_rate": safe_div(len(positive), len(bucket_rows)),
            "mean_evidence_gap_ms": mean(raw_gap),
            "median_evidence_gap_ms": median(raw_gap),
            "mean_positive_pause_ms": mean(positive),
            "median_positive_pause_ms": median(positive),
            "p75_positive_pause_ms": percentile(positive, 0.75),
            "p90_positive_pause_ms": percentile(positive, 0.90),
            "mean_visible_gap_ms": mean(visible_gap),
            "median_visible_gap_ms": median(visible_gap),
            "mean_underrepresented_pause_ms": mean(underrep),
            "median_underrepresented_pause_ms": median(underrep),
            "p90_underrepresented_pause_ms": percentile(underrep, 0.90),
            "mean_visible_overlap_ms": mean(overlap),
            "recommended_pause_pocket_ms": rec,
        })
        out.append(row)
    return out


def write_notes(path: Path, rows: list[dict[str, object]], counts: Counter, by_punct: list[dict[str, object]]) -> None:
    total = len(rows)
    valid = sum(1 for r in rows if int(r.get("evidence_ok", 0)))
    under_50 = sum(1 for r in rows if float(r.get("underrepresented_pause_ms", 0.0)) >= 50.0)
    lines = [
        "# V26 internal punctuation pause estimator",
        "",
        "This is diagnostic-only. It estimates soft pause pockets for internal punctuation; it does not change runtime behavior.",
        "",
        "## Stage counts",
        "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Summary",
        "",
        f"- Boundary rows: {total}",
        f"- Rows with full-WAV evidence: {valid}",
        f"- Rows where acoustic pause exceeds visible gap by at least 50 ms: {under_50}",
        "",
        "## Recommended starting pause pockets",
        "",
        "These are robust corpus estimates from positive full-WAV evidence gaps. Treat them as initial sweep centers, not final runtime constants.",
        "",
        "| punctuation | count | positive pause rate | median positive pause | p75 positive pause | recommended pocket | underrepresented mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_punct:
        lines.append(
            f"| {row.get('punctuation','')} | {row.get('count',0)} | "
            f"{float(row.get('positive_pause_rate',0.0)):.2f} | "
            f"{float(row.get('median_positive_pause_ms',0.0)):.1f} ms | "
            f"{float(row.get('p75_positive_pause_ms',0.0)):.1f} ms | "
            f"{float(row.get('recommended_pause_pocket_ms',0.0)):.1f} ms | "
            f"{float(row.get('mean_underrepresented_pause_ms',0.0)):.1f} ms |"
        )
    lines += [
        "",
        "## Interpretation guide",
        "",
        "- `evidence_gap_ms` is after-word evidence start minus before-word evidence end. Positive means the full-WAV evidence contains an acoustic pause pocket.",
        "- `committed_render_gap_ms` is the visible render gap between the committed envelopes. Negative means the face bridges/overlaps across punctuation.",
        "- `underrepresented_pause_ms` is the part of the acoustic pause not represented visually.",
        "- Internal punctuation should remain inside the sentence island; these numbers are for local distribution/envelope rules, not detector reacquisition.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate internal punctuation pause pockets from a liplab run.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows, counts = build_rows(args.run, args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)

    boundary_fields = [
        "case_id", "punctuation", "context", "word_before_index", "word_after_index",
        "word_before", "word_after", "raw_between", "same_text_island", "text_island_index",
        "evidence_ok", "committed_ok", "evidence_gap_ms", "evidence_pause_ms",
        "evidence_center_gap_ms", "committed_render_gap_ms", "committed_center_gap_ms",
        "visible_gap_ms", "visible_overlap_ms", "underrepresented_pause_ms",
        "visible_gap_to_evidence_pause_ratio", "before_evidence_end_sec", "after_evidence_start_sec",
        "before_committed_render_end_sec", "after_committed_render_start_sec",
        "before_evidence_fallback_count", "after_evidence_fallback_count",
    ]
    write_csv(args.out / "v26_punctuation_boundary_rows.csv", rows, boundary_fields)

    by_punct = summarize(rows, ["punctuation"])
    by_context = summarize(rows, ["punctuation", "context"])
    summary_fields = [
        "punctuation", "context", "count", "positive_pause_rate", "mean_evidence_gap_ms",
        "median_evidence_gap_ms", "mean_positive_pause_ms", "median_positive_pause_ms",
        "p75_positive_pause_ms", "p90_positive_pause_ms", "mean_visible_gap_ms",
        "median_visible_gap_ms", "mean_underrepresented_pause_ms",
        "median_underrepresented_pause_ms", "p90_underrepresented_pause_ms",
        "mean_visible_overlap_ms", "recommended_pause_pocket_ms",
    ]
    write_csv(args.out / "v26_pause_estimates_by_punctuation.csv", by_punct, [f for f in summary_fields if f != "context"])
    write_csv(args.out / "v26_pause_estimates_by_context.csv", by_context, summary_fields)

    worst = sorted(
        [r for r in rows if int(r.get("evidence_ok", 0)) and int(r.get("committed_ok", 0))],
        key=lambda r: float(r.get("underrepresented_pause_ms", 0.0)),
        reverse=True,
    )[:200]
    write_csv(args.out / "v26_worst_bridged_boundaries.csv", worst, boundary_fields)

    stage_rows = [{"stage": k, "count": v} for k, v in sorted(counts.items())]
    write_csv(args.out / "v26_stage_counts.csv", stage_rows, ["stage", "count"])
    write_notes(args.out / "v26_internal_punctuation_notes.md", rows, counts, by_punct)

    print(f"Wrote V26 internal punctuation diagnostics to {args.out}")
    print(f"Boundary rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
