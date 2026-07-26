# Gold Dataset Plan

This repo no longer treats auto-exported runtime artifacts as gold.

The gold corpus is rebuilt around one rule:

- transcript owns viseme identity
- offline alignment owns timing suggestions
- explicit review-layer approval owns export eligibility
- the harness grades only approved gold

Reviewer provenance is recorded in each package. The current checked-in corpus
was promoted by the deterministic `auto_mfa_phone_export` workflow; future
human review can replace that provenance without changing the runtime contract.

## Problem

The old `inputs/handmade*` corpus was not trustworthy:

- viseme timings were often copied from runtime fallback output
- speech regions were copied from older streaming occupancy logs
- drafts could be exported with `--allow-draft`
- the grader could not distinguish reviewed truth from provisional guesses

That made the corpus unsuitable as a research target.

## Target Repo Shape

Inputs:

- `inputs/transcripts/<case>.txt`
- `inputs/wav/<case>.wav`
- `inputs/gold/<case>/visemes.csv`
- `inputs/gold/<case>/phones.csv`
- `inputs/gold/<case>/words.csv`
- `inputs/gold/<case>/speech.csv`
- `inputs/gold/<case>/boundaries.csv`
- `inputs/gold/manifest.json`

Draft authoring outputs:

- `outputs/offline_gold/latest/<case>/...`
- `outputs/mfa_align/latest/<case>.TextGrid`
- `outputs/gold_drafts/<case>/draft.annotation.json`
- `outputs/gold_drafts/<case>/review_notes.txt`

## Review Layers

Each draft case contains four review layers:

1. `speech_regions`
   Derived from offline MFA word timing plus acoustic lull evidence. Region starts align to MFA word starts, region ends align to MFA word ends, and this is the primary pause/resume truth.
2. `word_heads`
   The first visible viseme per word. This is the primary word-onset truth.
3. `dense_visemes`
   The full viseme plan, still transcript-owned for identity and MFA-assisted for timing.
4. `pause_boundaries`
   Explicit punctuation and inter-word pause metadata used to explain why a speech region split was or was not created.

The draft carries explicit layer states:

- `draft_auto`
- `reviewed_boundary`
- `reviewed_dense`
- `approved_gold`

The export gate requires:

- top-level `approval.status == approved_gold`
- `speech_regions.status` reviewed at boundary level
- `word_heads.status` reviewed at boundary level
- `dense_visemes.status` reviewed at dense level

## Repo Workflow

1. Build offline evidence
   - run the standalone core offline against full WAVs
   - run MFA over the full corpus
2. Build draft gold packages
   - map transcript-owned planned visemes onto MFA phone timing
   - derive gold words from MFA word intervals plus transcript sentence ownership
   - derive speech regions from MFA word timing plus punctuation/acoustic lull evidence
   - derive explicit pause-boundary metadata for each inter-word punctuation boundary
3. Review drafts
   - first speech boundaries
   - then word heads
   - then dense visemes
4. Export approved gold
   - write `inputs/gold/<case>/...`
   - update `inputs/gold/manifest.json`
5. Grade runtime output
   - the harness reads only approved gold packages

## Phase 1 Implementation

Phase 1 is intentionally deterministic:

- MFA is the offline timing authority
- `offgrid_dropin` planned viseme identity remains authoritative
- no acoustic model invents viseme identity
- speech regions come from offline word timing plus acoustic lull evidence rather than runtime segmentation
- missing phone evidence falls back to offline committed timings and is flagged for review

## Expected Commands

Draft generation:

```bat
python scripts\draft_gold.py --mfa-num-jobs 4
```

Validate drafts and approved gold:

```bat
python scripts\check_gold.py --include-drafts
```

Export approved gold only:

```bat
python scripts\export_gold.py
```

Run streaming harness and grading:

```bat
scripts\verify.bat
```

## Acceptance Criteria

The repo is behaving correctly when:

- no draft can be exported without explicit approval
- the harness no longer depends on deleted `inputs/handmade*` files
- approved gold packages are validated structurally before grading
- speech-region truth comes from offline evidence tied to the WAV
- word and viseme timing truth remains transcript-owned for identity
