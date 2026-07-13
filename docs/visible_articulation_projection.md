# Visible articulation projection

## Goal

Produce believable visible speech from a known English transcript without
pretending that every CMU phone owns a distinct facial pose.

The authoritative sequence remains:

1. Transcript words are expanded to the complete CMU phone sequence.
2. MFA aligns that complete phone sequence to offline audio.
3. Runtime audio evidence estimates progress through the same complete phone
   sequence.
4. A separate articulation projection chooses the visible gestures driven by
   those phones.

Filtering the MFA input would be incorrect. Phones that are visually weak can
still carry duration, syllable, acoustic, and transcript-position information.

## Why the old mapping is unsafe

The previous planner mostly treated `phone == animation event`. That creates
independent poses for sounds such as `HH`, even though their visible realization
is absent or inherited from an adjacent vowel.
It also makes visual grading circular: the planner's own questionable mapping
is projected onto MFA timing and then treated as visible gold.

Visual speech research describes visemes as many-to-one and strongly affected
by coarticulation. A fixed one-phone/one-pose mapping is therefore a useful
indexing convenience, not a sufficient animation model.

## Phone roles

Every expected phone has one visual role:

- `timing_only`: no independent visible target. The phone remains in every
  phonetic and acoustic representation.
- `coarticulated`: no independent target; the adjacent visible articulation is
  expected to occupy its interval.
- `supporting_pose`: a subtle visible tongue/teeth articulation that may blend
  with the dominant mouth shape.
- `primary_pose`: a strong visible lip, jaw, or vowel target.

The initial conservative policy is:

| CMU phones | Role | Visible behavior |
| --- | --- | --- |
| `M B P` | primary | bilabial closure |
| `F V` | primary | lower-lip/upper-teeth contact |
| `W` | primary | rounded funnel |
| vowels | primary | differentiated jaw/lip target |
| `R` | primary | rhotic rounding target |
| `CH JH SH ZH` | primary | affricate/post-alveolar target |
| `S Z TH DH T D L N K G Y` | supporting | low-salience teeth/tongue target |
| `HH` | coarticulated | inherit the following vowel context |
| `NG` and unmatched phones | timing only | no independent pose |

This is intentionally more permissive than a pure lip-reading viseme map. The
MetaHuman asset contains tongue targets, so demonstrably visible tongue/teeth
articulation is retained as supporting motion rather than discarded.

An initial corpus experiment also removed `K/G/Y`. Together with `HH`, it
removed 827 visible events from the current corpus but exposed long dead zones in the state-based
performer and reduced speech-animation coverage from 0.826 to 0.788. Until a
coarticulation layer can explicitly carry adjacent visible states through those
phone intervals, these phones remain supporting targets rather than being
deleted prematurely.

## Vowel policy

The active Offgrid projection uses the library's distinct monophthong poses:

- `AA/AW -> 07_Aa`
- `AE -> 08_Ah`, `AH -> 18_Uh`
- `EH/EY -> 06_Eh`
- `IH -> 04_Ih`, `IY -> 03_Ee`
- `AO/OW/OY -> 09_Oh`
- `UH -> 18_Uh`, `UW -> 11_Oo`
- `ER -> 10_Or`
- `AY -> 05_Ay`

The visual grader now keys every target to its originating plan-phone index.
Within each word, a monotonic base-phone sequence alignment relates the CMU plan
to MFA, tolerating pronunciation insertions/deletions without assigning a
neighbor merely because its ordinal index fits. The summary reports both the
aggregate correspondence rate and per-phone correspondence counts.

MFA timing can test when these poses are rendered, but it cannot determine
whether `Ah` is a better facial shape than `Aa`; that decision requires a
mapping standard, visual ground truth, or perceptual review. Timing scores are
therefore deliberately not presented as proof of pose-shape quality.

Diphthongs ultimately need trajectories rather than a single static pose. That
is deferred because current runtime scheduling gives one acoustic center to a
source phone. Adding two events without adding phase-aware timing would merely
create a second arbitrary animation.

## Gold and grading contract

Gold remains layered:

- MFA words and phones are phonetic gold and remain complete.
- Speech regions and punctuation boundaries remain temporal gold.
- Visible gesture targets are derived by applying the versioned articulation
  projection to MFA phone intervals.

Timing-only and coarticulated phones must never count as missing visemes.
Phone-alignment metrics continue to cover the complete CMU/MFA sequence.
Visual metrics cover only projected primary/supporting gestures. Summary output
reports the projection counts and rate so mapping changes cannot silently make
scores easier by deleting targets.

The current corpus provides trustworthy MFA correspondence for roughly 93% of
renderable projected events. Unmatched projection events remain visible at
runtime but are excluded from timing error rather than being paired with the
wrong phone. Their count is reported explicitly.

The current scheduler still advances through `Events`, so a coarticulated `HH`
is represented internally by a non-renderable timing waypoint. The performer,
visual gold builder, and visual/playback graders all ignore that waypoint. A
future cleanup should make the scheduler advance directly through
`ExpectedPhones`; at that point the compatibility waypoint can disappear.

## Pronunciation selection

CMUdict entries with numbered suffixes are alternate pronunciations, not
duplicate records. The planner preserves every variant and chooses a portable
TTS-voice default from `OffgridAITtsPronunciationPreferences.inl`. The table
contains only existing CMUdict pronunciations observed in approved MFA
alignments; it never invents a phone sequence or consults audio at runtime.

Regenerate the table after an approved gold-corpus refresh with:

```bat
python scripts\build_tts_pronunciation_preferences.py
```

The generator uses a deterministic utterance-level held-out split and records
its evidence in `docs/tts_pronunciation_preferences.json`. Words without a
qualified preference continue to use CMUdict's first pronunciation.

## Next experiments

1. Perceptually compare the conservative projection against the old planner,
   especially `hello`, velar-heavy words, and initial `Y` words.
2. Decide whether supporting tongue poses improve Offgrid playback or introduce
   distracting pose churn. Test the entire supporting class as one controlled
   variable before tuning individual phones.
3. Add phase-aware diphthong trajectories only after the runtime can schedule
   start/end articulation within one MFA phone interval.
4. If visual motion capture or blendshape traces become available, learn
   context-dependent role/strength decisions from that visual evidence. MFA
   audio alignment alone cannot establish whether a gesture was visible.

## Research basis

- Bear et al., *Which phoneme-to-viseme maps best improve visual-only computer
  lip-reading?*: https://ueaeprints.uea.ac.uk/id/eprint/64350/
- Edwards et al., *JALI: An Animator-Centric Viseme Model for Expressive
  Lip-Synchronization*: https://www.dgp.toronto.edu/~elf/JALISIG16.pdf
- Microsoft Research, *Modeling Co-articulation in Text-to-Audio-Visual
  Speech*: https://www.microsoft.com/en-us/research/publication/modeling-co-articulation-text-audio-visual-speech/
