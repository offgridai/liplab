# Handmade Authoring Pipeline

This document describes the offline batch system used to create `inputs/handmade` gold answers from full `transcript + WAV` case inputs.

The goal is to create reviewer-approved viseme timing answers for grading. This system is not part of the runtime lipsync path and must not become a second production scheduler.

## Scope

Inputs:

- `inputs/transcripts/<case>.txt`
- `inputs/wav/<case>.wav`

Outputs:

- `inputs/handmade/<case>.csv` as the grader-facing gold answer
- a richer draft annotation artifact used during human review
- `outputs/mfa_align/latest/<case>.TextGrid` as the offline timing evidence

## Non-goals

The authoring pipeline must not:

- change runtime lipsync behavior,
- add new scheduling heuristics to the harness,
- make audio choose viseme identity,
- become a hidden second implementation of LineCoach logic,
- require Unreal Engine code or editor tooling.

## Ownership model

The same ownership split used by the runtime core still applies:

- transcript owns viseme identity,
- audio helps place timings,
- human review approves the final answer,
- the grader compares runtime output against the approved gold answer.

This means offline tools may use full-file alignment to improve timing proposals, but they should not invent viseme identities from acoustics.

## Batch pipeline

### 1. Draft generation

For each case:

1. Read the transcript and WAV.
2. Run the authoritative `offgrid_dropin` lipsync core offline to generate a first-pass viseme plan and fallback committed schedule.
3. Run offline MFA over the full WAV plus transcript to estimate:
   - word timings,
   - phone timings,
   - silence and pause spans.
4. Keep viseme identity from the transcript-owned `planned.csv` output of `offgrid_dropin`.
5. Map each planned viseme to the corresponding MFA phone timing using:
   - `word_index`
   - `source_phone`
   - `source_phone_index`
6. If MFA cannot provide a usable monotonic phone timing, fall back to the offline committed timing for that viseme.
7. Mark low-confidence or fallback regions for review.

The draft generator is an assistant for annotation. It is allowed to use stronger full-file timing analysis than the streaming runtime, because its output is reviewed by a human before becoming gold data.

## Implemented Phase 1

The current implementation is intentionally narrow:

- MFA is the offline timing authority
- `offgrid_dropin` `planned.csv` is the viseme identity authority
- no acoustic model is allowed to invent viseme identity
- no streaming heuristics are used during gold creation
- the exporter preserves transcript order even when MFA phone evidence is imperfect

Current script entrypoints:

- `python scripts/draft_handmade.py --mfa-num-jobs 4`
- `python scripts/export_handmade.py --allow-draft`
- `python scripts/check_handmade.py --include-drafts`

Current generated directories:

```text
outputs/offline_gold/latest/<case>/
outputs/mfa_align/latest/<case>.TextGrid
outputs/handmade_drafts/<case>/draft.annotation.json
inputs/handmade/<case>.csv
```

Current `draft.annotation.json` carries:

- transcript and audio metadata
- MFA word intervals
- MFA phone intervals
- planned visemes from the transcript path
- exported viseme timings
- alignment provenance such as:
  - `mfa_exact_source_phone_index`
  - `mfa_source_phone_index_label_fallback`
  - `mfa_source_phone_index_label_mismatch`
  - `offline_committed_fallback`

### 2. Review package

Each case should produce one rich editable artifact, for example:

`outputs/handmade_drafts/<case>/draft.annotation.json`

Suggested contents:

- case metadata
- transcript text
- WAV metadata
- word intervals
- phone intervals
- draft viseme intervals
- confidence flags
- reviewer notes
- revision history

Suggested JSON shape:

```json
{
  "case_id": "case001",
  "transcript": "welcome back",
  "audio": {
    "path": "inputs/wav/case001.wav",
    "sample_rate_hz": 16000,
    "duration_sec": 1.84
  },
  "mfa_words": [
    { "text": "welcome", "start": 0.11, "end": 0.62 }
  ],
  "mfa_phones": [
    { "phone": "W", "start": 0.11, "end": 0.16, "source": "mfa" }
  ],
  "visemes": [
    {
      "id": "22_MBP",
      "start": 0.18,
      "end": 0.28,
      "center": 0.23,
      "source_phone": "M",
      "alignment_reason": "mfa_exact_source_phone_index"
    }
  ],
  "flags": [
    {
      "kind": "low_confidence_timing",
      "start": 0.18,
      "end": 0.28,
      "note": "weak plosive release"
    }
  ],
  "notes": []
}
```

This format is intentionally richer than the final grader input. It should preserve enough context to revisit why a gold answer was chosen.

### 3. Human review

Review should happen in a small standalone local tool, not in Unreal.

Minimum review surface:

- waveform
- audio playback
- transcript
- word and phone intervals
- viseme intervals
- confidence flags

Required edit actions:

- move viseme boundaries
- move viseme centers
- split a viseme
- merge adjacent visemes
- relabel viseme identity when the transcript pronunciation draft is wrong
- mark silence or hold regions
- leave reviewer notes

Required guardrails:

- preserve monotonic order
- prevent impossible overlaps
- restrict viseme labels to the allowed inventory
- keep transcript-linked provenance when a viseme comes from a specific phone

### 4. Gold export

After review, export a simplified gold file into `inputs/handmade`.

The export step should discard authoring-only details and keep only the fields required by grading.

Suggested final JSON shape:

```json
{
  "case_id": "case001",
  "visemes": [
    { "start": 0.18, "end": 0.28, "pose": "22_MBP" }
  ]
}
```

Or, if CSV remains the primary grader format:

```csv
start,end,pose,word,confidence
0.180,0.280,22_MBP,welcome,1.0
```

The grader-facing format should stay simple and stable. Rich draft data belongs in the draft artifact, not in the final gold answer.

## Repository-facing workflow

Repository scripts:

- `scripts/draft_handmade.py`
  Batch-generate draft authoring artifacts for every case using offline core outputs plus MFA TextGrids.
- `scripts/export_handmade.py`
  Convert approved draft annotations into final `inputs/handmade` gold files.
- `scripts/check_handmade.py`
  Validate schema and monotonicity.

Suggested directories:

```text
outputs/mfa_align/latest/<case>.TextGrid
outputs/handmade_drafts/<case>/draft.annotation.json
outputs/handmade_drafts/<case>/review_notes.txt
inputs/handmade/<case>.csv
```

## How this fits the current harness

The current standalone harness should remain responsible for:

- building the authoritative `offgrid_dropin` core,
- running the streaming runtime path,
- emitting `planned.csv`, `speech_regions.csv`, `committed.csv`, and `grade.json`,
- comparing committed visemes against `inputs/handmade`.

The offline authoring system is separate. Its purpose is to create and maintain the gold answers used by that existing grading flow.

## Why full-file offline analysis is acceptable here

The runtime harness must behave like LineCoach and operate on a stream. The handmade authoring system has a different job:

- it sees the whole WAV,
- it is allowed to run slower,
- it may use stronger alignment passes,
- a human reviewer approves the result before it becomes gold.

Because of that review boundary, offline analysis can be more powerful without contaminating runtime logic.

## Recommended first version

Start with a narrow and deterministic first version:

1. Generate a viseme plan from `offgrid_dropin`.
2. Run offline MFA to produce full-file word and phone timings.
3. Convert aligned phones to draft visemes while keeping transcript-owned identity.
4. Review flagged cases in a simple local timeline tool.
5. Export final gold CSV into `inputs/handmade`.

This is enough to build a useful corpus without redesigning the runtime core or the grader.

## Future extensions

Possible later improvements:

- import/export Praat TextGrid as an authoring convenience,
- confidence ranking so reviewers see the hardest cases first,
- side-by-side display of `planned.csv`, `committed.csv`, and gold answers,
- per-viseme provenance for auditability,
- a custom MFA dictionary generated from the larger Offgrid lexicon,
- a corpus migration tool if the grader format evolves.

Those should remain optional conveniences. The core contract is simple: full-file offline tools help create gold answers, while runtime lipsync stays in `offgrid_dropin` and is evaluated against those answers.
