# Gold Authoring Workflow

The historical `handmade` name remains in this file for continuity, but the active workflow is now the gold-dataset workflow described in [gold_dataset_plan.md](/C:/git/liplab/docs/gold_dataset_plan.md).

## Current Contract

The repo uses three distinct artifact classes:

- source inputs
  - `inputs/transcripts/<case>.txt`
  - `inputs/wav/<case>.wav`
- draft review packages
  - `outputs/gold_drafts/<case>/draft.annotation.json`
  - `outputs/gold_drafts/<case>/review_notes.txt`
- approved grader inputs
  - `inputs/gold/<case>/visemes.csv`
  - `inputs/gold/<case>/words.csv`
  - `inputs/gold/<case>/speech.csv`

The harness grades only the approved gold package.

## Draft Generator

`python scripts/draft_gold.py --mfa-num-jobs 4`

Phase 1 behavior:

- runs the authoritative `offgrid_dropin` path offline for transcript-owned viseme identity
- runs MFA over the full WAV for word and phone timing
- derives speech regions from offline word runs, not runtime speech islands
- maps planned visemes to MFA phone timing when possible
- falls back to offline committed timing when phone evidence is missing
- emits layer statuses as `draft_auto`

## Review Layers

Each draft contains:

- `speech_regions`
- `gold_words`
- `word_heads`
- `visemes`
- `flags`

Review order:

1. speech boundaries
2. word-head timing
3. dense intra-word visemes

## Export Gate

`python scripts/export_gold.py`

Export now requires explicit approval. There is no draft export bypass.

Required states:

- `approval.status == approved_gold`
- `review_layers.speech_regions.status` reviewed at boundary level
- `review_layers.word_heads.status` reviewed at boundary level
- `review_layers.dense_visemes.status` reviewed at dense level

## Validation

Validate approved gold only:

```bat
python scripts\check_gold.py
```

Validate approved gold plus locally generated drafts:

```bat
python scripts\check_gold.py --include-drafts
```

Generate a ranked reviewer-assist report:

```bat
python scripts\review_drafts.py --top 20
```

Outputs:

- `outputs/gold_review/latest/draft_review_ranked.csv`
- `outputs/gold_review/latest/draft_review_report.md`

## Compatibility

Compatibility shims remain:

- `scripts/draft_handmade.py` calls `draft_gold.py`
- `scripts/export_handmade.py` calls `export_gold.py`
- `scripts/check_handmade.py` calls `check_gold.py`

They exist only to avoid breaking old command muscle memory. The gold workflow is authoritative.
