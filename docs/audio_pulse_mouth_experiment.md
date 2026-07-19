# Audio pulse mouth experiment

This branch contains an opt-in diagnostic renderer for visually auditing
streaming syllable-beat detection. It asks one narrow question: where did the
audio-only streaming detector believe it found a syllable nucleus?

Run one case with full diagnostics:

```bat
build-ninja\liplab_runner.exe . --case case_0001 --audio-pulse-mouth --full-diagnostics
```

The shared runtime switch is
`FOffgridAILipsyncRuntimeBeginInput::bEnableAudioPulseMouthExperiment`. It is
false by default. When enabled, the normal viseme scheduler is bypassed for
rendering. Each stable acoustic nucleus candidate commits a full-strength
`08_Ah` at the pulse center, followed by a full-strength `22_MBP` at the
lowest-energy still-renderable valley before the next candidate. Because the
following pulse is what proves that a gap is inter-nuclear, the closure is
constrained to remain at least 30 ms ahead of the live playhead and 25 ms before
the next opening. Transcript timing and identity are deliberately excluded so
the animation honestly displays only what the audio detector found. This is an
intentionally exaggerated debugging indicator, not a cosmetic lipsync mode.

Nucleus candidates wait 120 ms beyond the evidence decision before they become
immutable. This reduces peak refinement churn while preserving live lead under
the default 350 ms preroll. Every open event records its acoustic pulse center
and every close event records its measured inter-pulse valley. MFA-derived
nuclei independently score the displayed beats after the run.

The indicator uses experiment-only 20, 30, 50, and 80 ms half-width sonority
envelopes. A deterministic 12-tree temporal ranker selects vowel-like peaks,
then a 120 ms spacing rule preserves close syllables only when the 30 ms
envelope contains a strong intervening valley. The ranker sees local audio
features only. This profile is not used by the production transcript scheduler.
The native corpus pass retained at least 68 ms of live commit lead and produced
no difference between the final assigned-beat set and the rendered-beat set.

The MFA score uses MFA vowel-phone intervals directly; it does not project
transcript/CMUdict vowel ordinals onto the MFA phone sequence. This distinction
matters when the two pronunciations contain different phone counts. MFA `spn`
intervals remain unscorable because they contain speech without phone labels.

Each experimental case writes:

- `audio_pulse_mouth_grade.json`: detected-versus-rendered nucleus beats, center
  error, live commit lead, animation duty cycle, and lull leakage.
- `audio_nucleus_beat_mfa_grade.json`: independent precision and recall against
  vowel-nucleus intervals derived from the MFA phone alignment.
- `region_conditioned_nucleus_grade.json`: the primary joint objective. A beat
  counts only when it is within 100 ms of an MFA vowel interval and belongs to
  the runtime region that overlaps that MFA region. It also reports first-beat
  recall per region, wrong-region beats, and completely perfect lines.
- `audio_pulse_mouth_samples.csv`: generic open and full-close weights sampled
  every 10 ms, suitable for plotting or comparison against video playback.

Normal transcript alignment grades are intentionally not meaningful in this
mode. Run `scripts\verify.bat` without the experiment flag to verify that the
default production scheduler remains unchanged.

## Shared acoustic coordinator

Confirmed speech regions now define independent nucleus epochs. Peak spacing
and replacement state is cleared at every region transition, so a strong final
syllable cannot suppress the first syllable after a pause. Candidates that
remain outside all resolved speech regions are not rendered. Conversely, a
confirmed nucleus breaks a still-provisional lull on the presentation evidence
surface, preventing the close-between-beats indicator from treating a detected
syllable as silence. Committed speech regions are never retroactively changed.

After resolving 19 legacy MFA `spn` intervals and running with the intended
focused/list-sensitive region configuration, the full-corpus result is 4,785
matches from 5,000 scorable beats and 5,301 MFA nuclei: 95.70% precision,
90.27% recall, and 92.90% F1. Only 18 beats remain ungradeable inside five
explicitly quarantined `spn` spans.

A trial that allowed strong nuclei to retroactively merge short resolved
regions was rejected. It improved region precision but removed 17 genuine MFA
pauses, reducing internal-boundary recall by about three percentage points.
Nucleus evidence may veto a provisional presentation lull, but not a committed
speech boundary.

## Syllable-paced transcript-viseme experiment

`--syllable-paced-visemes` replaces the ordinary duration scheduler with one
monotonic experimental scheduler. The tuned acoustic-nucleus surface supplies
the pace, an online beam assigns each stable pulse to an ordered planned
syllable, and transcript-derived priors provide coherent timing inside each
word. Audio still cannot choose pose identity.

A committed word head is immutable: its first perceptually rendered viseme is
centered on its accepted acoustic nucleus. One pulse may advance at most one
whole word, and the runtime may not manufacture intervening word starts from a
projected transcript schedule. Region close no longer rewrites an accepted
pulse to the end of a text region or word; doing so was found to introduce
one-beat phase errors after weak or missed nuclei.

Region transitions constrain assignments and reset acoustic evidence. When an
acoustic region closes inside a multisyllabic word, its last pulse completes
the word locally rather than allowing the word to split. A complete planned
text-region tail is consumed only when at most two syllables remain. When
transcript punctuation and acoustic region ordinals disagree, the same
monotonic scheduler continues with the best forward candidate while the audio
region remains authoritative for ownership.

When a multisyllabic active word ends at transcript punctuation, an ambiguous
audio pulse that explicitly includes the word's next syllable candidate is
assigned to that continuation before the following word may start. This keeps
boundary-final syllables such as the second vowel of `turkey,` or `veggie.`
from advancing the scheduler one word early. Punctuation neither creates a
pause nor supplies its timing; audio-only speech regions still own pause and
resume, and the preference is inapplicable when the continuation is absent
from the acoustic candidate beam.

Full diagnostics write `runtime_word_start_nucleus_grade.json`. This is the
P2 experiment score: it maps each direct runtime word assignment to that
word's first transcript/MFA nucleus, then grades the center of the actually
rendered anchor viseme. It therefore fails wrong syllable identity, wrong word
identity, missing words, and any later scheduling shift of an otherwise correct
audio observation. `scripts\summarize.py` reports its corpus precision, recall,
and matched center error as `performed_word_start_nucleus`.

The separate `performed_word_head_nearby_nucleus_grade.json` measures the
weaker but essential acoustic invariant: each performed word head is matched
to the nearest MFA vowel nucleus without assuming that the runtime chose the
right transcript word. This exposes phase/identity errors instead of mixing
them with beat-detection errors. On the current 350-case corpus, 3,452 of 3,595
performed word heads land inside a real MFA nucleus (96.02% precision); misses
average 2.50 ms outside the target interval. Correct word/nucleus identity is
reported only by the stricter P2 score above.

P1 is deliberately simpler and earlier in the product contract: the beginning
of the first rendered animation for a word (`RenderStartSeconds`) must align
with MFA's acoustic word onset. The summary reports this independently as
`word_animation_onset`. A word may therefore satisfy P1 even when its later
viseme center cannot yet be proven to correspond to the correct MFA nucleus;
P2 measures that additional benefit.

Strong audio-only beat detection still does not imply reliable online
beat-to-transcript identity. A false positive, missed reduced vowel, or
disagreement between transcript and acoustic region counts can change phase.
The mode remains opt-in so correspondence work can be measured against MFA
without changing the ordinary runtime scheduler.
