# Phone-First Planning and Playback Spec

Status: proposed replacement direction after the refreshed MFA gold corpus.

## 1. Core conclusion

The refreshed gold corpus makes two things much clearer than before:

1. `Speech regions` are usually real acoustic islands with obvious nonspeech time between them.
2. `Words inside a speech region` are usually contiguous in MFA. They often do **not** have real silent gaps between them.

That changes the correct decomposition of the runtime problem:

- `speech occupancy` should own region start, region stop, pause, and resume
- `transcript + CMU phones` should own identity and ordering
- `within-region pacing` should be treated as a **continuous phone-progression problem**, not a word-gap detection problem

The old mental model:

```text
speech region -> word pauses -> viseme playback
```

should be replaced with:

```text
speech region gate -> dense CMU phone path -> visible viseme projection
```

## 2. What the refreshed gold means

The new corpus shows:

- speech regions are coarse acoustic structure
- word boundaries are usually lexical timing structure, not silence structure
- the red MFA row is best understood as a dense `phone` timeline
- visible mouth shapes are only a sparse projection of that dense phone timeline

Implications:

- we should stop expecting ordinary word boundaries to be recoverable from pauses
- we should stop inserting or preserving synthetic inter-word dead time inside active speech
- non-visible phones still matter because they consume time and separate neighboring visible phones
- the planner should become more explicitly `phone-first`, while playback remains `absolute-time viseme event` based

Before deleting any remaining word-gap logic wholesale, the repo should measure
the actual intra-region inter-word-gap distribution across the full approved
corpus:

- count inter-word gaps inside each speech region
- report mean / median / p90 / max gap duration
- bucket the distribution into near-zero, micro-gap, and hesitation-like tails

If the distribution is overwhelmingly near-zero with only a thin hesitation
tail, the phone-first continuous-pacing model is confirmed. If a meaningful
middle cluster exists, runtime logic must preserve a local hold mechanism for
those cases rather than assuming they do not exist.

## 3. Ownership model

This is the required ownership split.

### 3.1 Planner ownership

The planner owns:

- transcript normalization
- CMU pronunciation lookup
- the full ordered CMU phone chain
- phone weights / relative duration priors
- visible viseme projection from phones
- stable global phone indices and word-local phone indices

The planner does **not** own:

- speech region timing
- word pause timing
- final playback times

### 3.2 Speech detector ownership

The detector owns:

- speech onset
- speech cessation
- pause / resume
- whether audio is currently inside or outside active speech

The detector does **not** own:

- phoneme identity
- viseme identity
- word boundaries except where a real acoustic pause creates a speech-region boundary

### 3.3 Playback ownership

Playback owns:

- mapping observed active speech time onto the dense phone path
- converting phone progress into absolute visible event times
- bounded ahead-of-playback commitment
- monotonic append-only track emission

Playback must never:

- reorder phones
- invent phone identity from audio
- suppress planned visible visemes as a shortcut
- treat observed audio as the playback clock

Clarification:

```text
The playback clock remains the line playback clock.
The pacing numerator inside an open speech region should be raw elapsed real time
accumulated while occupancy is open, not a classifier-derived "mass" score.
```

This is one of the main benefits of the redesign. The dominant within-region
pacing signal becomes a robust scalar driven by elapsed time under open
occupancy, while phone-family and local acoustic evidence remain secondary
advisory signals only.

## 4. Planner revisions

### 4.1 Make the dense phone path explicitly canonical

`Plan.ExpectedPhones` must be the master timeline substrate.

Requirements:

- every CMU phone from the chosen transcript pronunciation must appear in `ExpectedPhones`
- preserve the original stressed phone in `Phone`
- preserve the stress-stripped base phone in `BasePhone`
- preserve both:
  - `PhoneIndex` = global phone index across the line
  - `WordPhoneIndex` = phone index inside the word
- preserve `WordIndex`, `PhraseIndex`, and sentence metadata
- preserve `WeightSeconds` as the planner's relative duration prior

This means the planner becomes the owner of the dense hidden timing path, not only the visible event list.

### 4.2 Treat visible visemes as a projection of the phone path

`Plan.Events` should remain the visible renderable events, but they must be interpreted as a projection of `ExpectedPhones`, not as the canonical timeline.

Requirements:

- each visible event must map to exactly one `SourcePhoneGlobalIndex`
- a visible event may only exist if its source CMU phone maps to an available MetaHuman pose
- if two adjacent phones map to the same pose, they still remain distinct in the dense phone path
- do not merge same-pose adjacent phones at planning time just to simplify playback
- do not add fake inter-word rest events

Important consequence:

```text
The dense phone chain is authoritative for timing continuity.
The visible event list is authoritative only for what may be rendered.
```

### 4.3 Remove synthetic visible fallback words

The current planner contains a fallback `cmu_no_visible_phone` event when a word has no visible CMU phone mapping.

That should be removed, but only after verifying that the performer can
gracefully hold or blend across longer invisible-phone spans without
introducing a frozen-mouth artifact.

Desired behavior:

- if a word has visible phones, emit those visible events
- if a word has no visible phones, emit no visible event for that word
- the word still consumes time through the dense phone chain

Reason:

- fake fallback visible events invent articulation that is not grounded in the phone plan
- the dense phone chain already carries the timing obligation
- the performer can naturally hold/blend adjacent real visible events through invisible-phone spans

Required verification before promotion:

- run the corpus with fallback removed
- inspect cases with long invisible-phone spans
- confirm performer output remains perceptually stable when no synthetic filler
  event exists between neighboring real visible events

### 4.4 Add fast lookup metadata from words to phone ranges

The runtime now needs region-local continuous pacing over the phone path. That is easier and less error-prone if the planner exports direct ranges.

Add planner-side metadata such as:

- `WordPhoneBeginIndices`
- `WordPhoneEndIndices`
- `WordVisibleEventBeginIndices`
- `WordVisibleEventEndIndices`

These arrays are metadata only. They do not carry timing ownership.

### 4.5 Keep punctuation as a prior, not a silence promise

Punctuation should remain planner metadata, but it must not imply a synthetic gap.

Meaning:

- hard punctuation may suggest a likely speech-region boundary
- soft punctuation may suggest a likely local slowdown or clause edge
- neither hard nor soft punctuation may directly create silent time inside the plan

The refreshed gold says silence is an acoustic fact, not a text fact.

## 5. Playback revisions

### 5.1 Replace word-gap thinking with region-gated continuous pacing

The runtime should stop thinking in terms of:

- "did a word boundary gap happen?"
- "should the next word wait for a pause?"

and instead think in terms of:

- "are we inside active speech?"
- "how far through the current region's phone chain should we be by now?"

So the main runtime object should become:

```text
speech-region gate + monotonic phone-progress cursor
```

not:

```text
event-by-event word pause seeker
```

### 5.2 Speech occupancy should only gate coarse structure

Speech occupancy should do exactly this:

- open a region when speech starts
- close a region when speech ends
- freeze all active-speech progression while closed
- reopen and continue at the next unconsumed phone/word prefix

Speech occupancy should not do this:

- decide phoneme identity
- stretch words to fill dead space that does not exist in gold
- create implied word boundaries where there is no real pause

### 5.3 Pace inside a region using the dense phone path

Inside an active speech region, progression should be continuous over phone weight mass.

The runtime should maintain a monotonic phone-progress scalar such as:

- cumulative region-local planned active seconds consumed
- cumulative expected phone weight consumed after mapping elapsed active time
  onto the region-local phone path

The numerator for this mapping should be:

```text
raw elapsed real time while occupancy is open inside the current speech region
```

not:

```text
a classifier-derived activity mass score
```

Advisory acoustic evidence may locally bias or freeze the cursor, but it should
not replace elapsed active time as the main progress signal.

Required behavior:

- progression freezes when speech is absent
- progression resumes when speech resumes
- progression is continuous inside a region
- words do not inject artificial dead time
- non-visible phones still advance the cursor

This is the biggest conceptual revision from the current behavior.

Important limitation:

- pure linear region-local time-warping cannot localize a real interior
  hesitation pause if the detector never closes the region
- if a real brief quiet occurs mid-region, linearly spreading the slowdown
  across the whole region will mistime the hold

Therefore the runtime should also support:

```text
sub-threshold local freeze
```

Meaning:

- short local acoustic dips that do not justify region close may still briefly
  freeze phone-progress advance
- this freeze does not create a new speech region
- this freeze does not choose identity
- this freeze only says "do not advance the cursor while this brief quiet is
  happening"

This reuses pause-family / local quiet evidence in a much safer role than word
boundary detection.

### 5.4 Use region-local timing, not whole-line stretching

The refreshed corpus suggests that real silence mostly lives between speech regions, not between ordinary words.

Therefore playback should reason region-locally:

- region `N` owns a contiguous transcript prefix/range
- the runtime maps the observed active duration of region `N` onto that region's phone-weight mass
- once region `N` closes, progress freezes
- region `N+1` starts from the next unconsumed phone prefix

This prevents silence between regions from polluting within-region pacing.

### 5.5 Promote the phone cursor to the main scheduling substrate

The runtime should schedule visible events from the phone cursor, not directly from visible-event order norms.

Required logic:

1. compute dense phone progress inside the current active region
2. derive the implied time span for each dense phone
3. place each visible event from its source phone span
4. apply pose-specific lead and span shaping afterward

This preserves the runtime contract:

- visible events are still absolute playback-time events
- the performer still samples only committed event windows
- commitment remains append-only and monotonic

### 5.6 Keep local acoustic evidence advisory only

The refreshed gold justifies stronger use of occupancy and weaker dependence on word-gap heuristics, but it does **not** authorize audio to choose content.

If `OnlinePhoneAligner` or family detectors remain in play, their role should be:

- sub-threshold local-freeze support
- local cursor nudging
- local speed-up / slow-down hints
- salient-phone timing support for already-known upcoming phones

They must not:

- choose which phone comes next
- skip across phones
- delete invisible phones from the path
- replace the planner's phone inventory

### 5.7 Keep bounded commitment in absolute playback time

The runtime contract from [Runtime_Contract_Spec.md](/C:/git/liplab/docs/Runtime_Contract_Spec.md) remains valid.

This spec does **not** change:

- append-only committed track semantics
- absolute playback-time `FinalRenderCenterSeconds`
- performer sampling behavior
- bounded lead ahead of playback

Only the source of event placement changes:

```text
from visible-event order norms
to dense-phone progress projected into visible events
```

## 6. Specific behaviors to delete or demote

These behaviors are no longer justified by the gold and should be removed or downgraded.

### 6.1 Treating word boundaries as expected silent gaps

Delete the assumption that ordinary word boundaries should create pause-like timing unless occupancy confirms real silence.

### 6.2 Planner-inserted fake visible filler for invisible-phone words

Remove `cmu_no_visible_phone`-style fallback visible events.

### 6.3 Word-gap seeking inside active speech

Any logic that waits for a pause before feeling allowed to advance to the next word should be removed.

### 6.4 Silence padding from punctuation alone

Punctuation may be a prior, but it must not inject silent time into the schedule.

### 6.5 Whole-line timing compensation that ignores region structure

Do not let long dead space between speech regions smear pacing across the whole line.

## 7. Data contract revisions

These fields should exist and be trusted.

### 7.1 `FOffgridAIExpectedPhone`

Required fields:

- `PhoneIndex`
- `WordPhoneIndex`
- `Phone`
- `BasePhone`
- `SourceWord`
- `WordIndex`
- `PhraseIndex`
- `SentenceIslandIndex`
- `WeightSeconds`
- `bIsVisibleViseme`
- `FirstVisibleEventIndex`

### 7.2 `FOffgridAITextVisemeEvent`

Required fields:

- `PoseID`
- `WordIndex`
- `PhraseIndex`
- `SentenceIslandIndex`
- `SourcePhoneIndex`
- `SourcePhoneGlobalIndex`
- `SourcePhone`
- `SourcePhoneBase`

Constraint:

```text
Every committed visible event must be traceable back to one dense source phone.
```

### 7.3 Runtime event

`FOffgridAIAlignedVisemeEvent` should continue to expose:

- absolute render times
- source phone global index
- source phone base
- word index
- commit diagnostics

No change is needed to performer-facing semantics.

## 8. Diagnostics we now need

The new architecture should log the dense substrate directly.

Add or preserve diagnostics for:

- active speech region index
- active region start / end
- active elapsed wall-clock time inside the region
- phone cursor position in global and word-local space
- observed active speech seconds
- planned active speech seconds
- current owned phone range for the region
- visible events derived from each phone
- whether a visible event came from a real CMU phone vs any fallback path

Important new debug questions:

- which phones are currently owned by the active region?
- how much of the current region's phone mass has been consumed?
- which visible events are pending only because their source phones have not yet been reached?
- are any word boundaries being treated like pauses even though occupancy stayed open?
- how often did sub-threshold local-freeze engage, for how long, and at what
  cursor positions?
- how does the corpus-wide intra-region inter-word-gap distribution actually look?

## 9. Acceptance criteria

The new direction is correct if it produces these qualitative behaviors:

- speech regions start and stop on real acoustic boundaries
- pauses between regions stop mouth progression instead of stretching words
- words inside a region feel continuously paced rather than stop-start
- visible peaks align better to salient phones because invisible phones no longer collapse out of timing
- the runtime no longer depends on ordinary word-gap detection to stay synchronized

Quantitatively, success should show up as:

- speech-region metrics staying strong or improving:
  - `speech_boundary_start_ms`
  - `speech_boundary_end_ms`
  - speech-region count mismatch
- word-head timing improving without requiring more predicted gaps:
  - `word_head_start_ms`
  - `word_head_start_median_ms`
- better phoneme coverage from dense-phone ownership:
  - `phoneme_coverage_rate`
- better intra-word timing because visible events are anchored to source phones rather than only to text order:
  - `intra_word_coverage_rate`
  - `intra_word_center_ms`
- continued safety on:
  - `order_fail_cases = 0`

The same before/after grading discipline used elsewhere in the repo should
remain the promotion gate here. This direction does not get a softer standard
just because the architectural thesis is stronger.

## 10. Recommended implementation order

### Phase 0: corpus confirmation

- measure intra-region inter-word-gap distribution over the approved gold corpus
- confirm that the main within-region pacing signal should indeed be continuous
  elapsed active time rather than ordinary word-gap ownership
- record the resulting numbers in docs alongside this spec

### Phase 1: planner cleanup

- make `ExpectedPhones` explicitly canonical
- remove fake no-visible fallback events
- export direct word-to-phone and word-to-visible ranges
- ensure every visible event has a valid `SourcePhoneGlobalIndex`

### Phase 2: runtime substrate change

- replace visible-order pacing with dense-phone pacing
- keep occupancy as the only coarse pause/resume gate
- make progression region-local
- define observed active progress explicitly as raw elapsed time under open occupancy
- add sub-threshold local-freeze support for short quiet dips that do not close
  the region

### Phase 3: visible projection cleanup

- derive absolute visible event centers from source phone spans
- keep pose lead/span shaping as a final render step only

### Phase 4: optional advisory acoustic bias

- use phone-family or detector evidence only to nudge the local dense-phone cursor
- do not promote any advisory evidence to identity ownership

## 11. Final design statement

The correct mental model after the refreshed corpus is:

```text
The transcript provides a dense CMU phone chain.
Speech occupancy tells us when articulation is happening.
Playback moves monotonically through that phone chain while speech is active.
Visible visemes are a sparse rendering projection of the underlying phone chain.
```

That is the planning and playback model the repo should now be revised toward.
