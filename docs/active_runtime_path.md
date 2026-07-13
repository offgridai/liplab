# Active Runtime Path

This repo's active lipsync path is intentionally narrow.

## Authoritative runtime flow

1. `OffgridAITextVisemePlanner` converts transcript text into authoritative CMU phones, visemes, duration priors, syllable nuclei, strong-phone markers, and punctuation fences.
2. `OffgridAIStreamingSpeechDetector` converts streamed PCM into causal occupancy, quiet/resume candidates, and frame features.
3. `OffgridAIAcousticEvidence` converts one feature frame into broad articulatory probabilities. It does not align or select phones.
4. The cursor in `OffgridAILipsyncRuntimeAdapter` resolves punctuation fences strictly and assigns syllable/strong-phone anchors softly, always left to right.
5. The scheduler rebases only the uncommitted suffix and commits transcript visemes monotonically.
6. `harness/main.cpp` streams corpus PCM and exports raw runtime diagnostics. Python owns aggregate grading and policy checks.

## Ownership

- Transcript planning owns identity and order.
- Audio evidence owns observed timing evidence only.
- The cursor owns accepted anchor progression.
- The scheduler owns suffix rebasing and irreversible commits.
- Diagnostics observe decisions and never alter them.

## Non-goals

The active path does not rely on:

- TTS hint streams
- token-progress ownership
- predicted word schedules
- external timing middleware
- acoustic selection of viseme identity
- overlapping fallback schedulers

The current conceptual model is:

- `speech regions`
- `speech occupancy`
- `transcript-derived viseme identity`
- `monotonic region-local playback`
- `strict punctuation close/resume anchors`
- `soft syllable and strong-phone suffix rebasing`
