# Word-state decoder shadow patch

Diagnostic-only patch. Runtime playback is unchanged.

This patch keeps the contract-safe shadow framework and replaces the previous shadow word span allocator with a global word-state segmentation pass:

- For each unambiguously mapped speech region/sentence, collect the expected transcript words.
- Build duration priors from the existing CMU-derived word weights.
- Generate candidate internal word boundaries from:
  - expected duration-prior boundary times,
  - local RMS valleys,
  - spectral flux peaks,
  - low-evidence frames,
  - boundary-confidence windows.
- Solve the full ordered boundary sequence with dynamic programming.
  - This avoids the earlier `boundary -> advance one word` failure mode.
  - Extra acoustic candidates can be ignored.
  - Missing acoustic candidates fall back to duration priors instead of stalling the current word.
- Emit the resulting word spans through the existing absolute-time, append-only shadow contract.
- Continue writing `word_assignment_shadow.csv`, `word_assignment_shadow_grade.json`, and `word_state_shadow.csv`.

No live committed track, performer, LineCoach, or playback timing behavior is changed.

The intended readout is whether `WORD_ASSIGNMENT_SHADOW` improves over the previous shadow path before any runtime promotion.
