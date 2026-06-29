# Agent Iteration Task

Improve lipsync logic in `offgrid_dropin` only. Run `liplab_runner` before and after changes. Do not touch LineCoach. Do not add or consume TTS hint streams; the harness and core are transcript + PCM only.

Primary failure classes to reduce:

- early phrase starts
- late or truncated tails
- missing planned visemes
- phrase/list compression
- pause-boundary leakage
- monotonicity violations

Use `outputs/runs/latest/<case>/committed.csv` and `grade.json` as the review surface.
