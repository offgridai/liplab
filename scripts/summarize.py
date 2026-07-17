import json
import pathlib
import subprocess
import sys


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    for obsolete_name in (
        "focus_alignment_ranking.csv",
        "focus_alignment_summary.json",
        "region_ownership_cases.csv",
        "region_ownership_words.csv",
        "region_ownership_summary.json",
    ):
        (latest / obsolete_name).unlink(missing_ok=True)
    alignment_script = root / "scripts" / "summarize_alignment.py"
    if subprocess.call([sys.executable, str(alignment_script)]) != 0:
        return 2
    summary = json.loads((latest / "alignment_summary.json").read_text(encoding="utf-8"))

    summary_path = latest / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
