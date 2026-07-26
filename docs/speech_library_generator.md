# Synthetic speech library generator

`scripts/generate_speech_library.py` is an offline data-factory prototype. It
does not participate in runtime lipsync and does not expose TTS timing to the
runtime scheduler.

The generator takes transcript, voice-clone JSON, and seed sets and synthesizes
their Cartesian product through one persistent Qwen process. It records exact
model/voice/executable provenance and SHA-256 hashes, builds an MFA dictionary,
and runs the project-local MFA installation. Generated artifacts are written to
`outputs/speech_library/<library-id>/`.

Run the built-in three-line, two-seed demonstration:

```bat
python scripts\generate_speech_library.py --library-id demo_v1
```

Run the same command again to compare every regenerated WAV with the hashes in
the previous manifest. `all_previous_wavs_matched` is true only when all WAVs
are byte-identical.

Custom lines may be supplied as either a JSON list or `{ "lines": [...] }`:

```json
{
  "lines": [
    {"id": "greeting", "text": "Hello, welcome back."},
    {"id": "list_pause", "text": "Bring paper, pencils, and blue maps."}
  ]
}
```

Useful options:

- `--lines-json <path>` selects a transcript manifest.
- `--seeds 41,42,43` controls deterministic sampling.
- Repeat `--voice-json [id=]<path>` to select one or more Qwen clone embeddings.
- `--qwen-root <path>` or `--qwen-exe <path>` selects the TTS checkout/build.
- `--skip-synthesis` reuses existing WAVs and reruns MFA packaging.
- `--skip-mfa` generates WAVs and provenance without alignment.
- `--recipe-json <path>` selects exact per-case transcript, voice, and seed assignments.
- `--validate-only` validates a recipe and stages its Qwen job list without synthesis.

For example, two transcripts, two voices, and three seeds produce twelve WAVs:

```bat
python scripts\generate_speech_library.py ^
  --lines-json inputs\library_lines.json ^
  --voice-json priestley=C:\git\qwen3-tts-cpp-streaming\reference\priestley_0.6b_f16.json ^
  --voice-json lana=C:\git\qwen3-tts-cpp-streaming\reference\lana_0.6b_f16.json ^
  --seeds 41,42,43
```

The output manifest is the reproducibility contract. It records the Qwen Git
revision, hashes of the executable and every voice JSON, explicit sampling
settings, the persistent batch command, WAV format/duration/hash, dictionary
provenance, and TextGrid hashes and interval counts.

## Existing corpus recipe

`inputs/speech_library/existing_corpus_v1.json` assigns one deterministic seed
and one voice to each of the 350 checked-in corpus transcripts. Assignments are
pseudorandom but reproducible: they are derived from a named SHA-256 namespace
and the original case ID. The recipe currently spreads cases across the Alfie,
Lana, Priestley, and Priestley 2 reference JSONs.

Validate that every embedded transcript still matches its canonical source and
that all Qwen voice JSONs resolve:

```bat
python scripts\build_corpus_speech_recipe.py --check
python scripts\generate_speech_library.py ^
  --recipe-json inputs\speech_library\existing_corpus_v1.json ^
  --library-id existing_corpus_v1 ^
  --validate-only
```

Remove `--validate-only` to synthesize and MFA-align the complete assigned
corpus. Regenerate the checked-in recipe after intentionally changing the
source corpus with `python scripts\build_corpus_speech_recipe.py`.
