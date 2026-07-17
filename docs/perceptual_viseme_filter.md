# Perceptual viseme render filter

The CMU-derived plan and committed timing track retain every phone. The
performer renders only phone classes with a defensible, externally visible
contrast. This is a presentation filter, not a second planner or scheduler.

## Evidence used

- Files et al. (2015) found excellent discrimination between established
  visual consonant classes, including reliable information within some
  traditional viseme classes. This argues against reducing animation to only
  a few generic mouth states.
  https://doi.org/10.3389/fpsyg.2015.00826
- Magnotti et al. (2019) measured strong visual categories for bilabials,
  labiodentals, interdentals, `/w/`, `/r/`, `/l/`, and postalveolar
  fricatives/affricates. Rear and alveolar articulations formed broader,
  less-specific groups.
  https://doi.org/10.1016/j.specom.2018.12.002
- Lidestam et al. (1998) found that masking tongue and teeth had little effect
  on consonant-viseme recognition. This supports avoiding weak, isolated
  internal-tongue animation when it does not produce a distinctive external
  configuration.
  https://doi.org/10.1044/jslhr.4103.564
- Audiovisual vowel experiments show useful visible information in jaw
  aperture, lip spreading, and especially lip rounding. Those vowel states
  remain intact.
  https://doi.org/10.1016/j.specom.2006.01.002
- Sharma et al. (2024) measured reliable lip-aperture and protrusion
  differences between `/s/` and `/sh/`. The filter therefore retains the
  postalveolar `/sh/` family rather than treating every sibilant alike.
  https://doi.org/10.21437/Interspeech.2024-2415

## Render policy

| Rendered | CMU phones | Primary visible information |
|---|---|---|
| Yes | all vowels | aperture, spreading, rounding |
| Yes | P B M | complete lip closure |
| Yes | F V | lower-lip/upper-teeth contact |
| Yes | TH DH | interdental contrast |
| Yes | W R | rounding/protrusion |
| Yes | L | visually distinct lingual category |
| Yes | SH ZH CH JH | postalveolar lip shaping |
| No | T D S Z K G HH Y N NG | primarily hidden or broad low-specificity articulation |

Unknown-word fallback poses remain visible. Filtered events remain in the
plan and committed track, preserving word ownership, monotonicity, timing,
and diagnostics. Envelope continuity is computed between the remaining
rendered events so hidden consonants do not create artificial mouth dropouts.
