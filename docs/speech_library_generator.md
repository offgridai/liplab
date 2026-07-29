# Synthetic speech library generator

`scripts/generate_speech_library.py` is the offline Data Factory. It
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
and one voice to each of the 441 recorded corpus transcripts. This is a
recreation recipe: its voice is not mistaken for the speaker in the existing
recording. Assignments are
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

## Neural scheduler expansion corpus

`inputs/speech_library/neural_scheduler_lines_v1.json` contains a balanced
suite of 100 short, long, multi-clause, multi-sentence, list, question,
instruction, contrast, and emphatic transcripts. The generated
`neural_scheduler_corpus_v1.json` recipe crosses every line once with each of
the four available 1.7B voice-clone references, for 400 fixed cases.

```bat
python scripts\build_neural_corpus_recipe.py
python scripts\generate_speech_library.py ^
  --recipe-json inputs\speech_library\neural_scheduler_corpus_v1.json ^
  --library-id neural_scheduler_corpus_v1 ^
  --model-identifier qwen3-tts-1.7b-base-f16
python scripts\import_speech_library.py outputs\speech_library\neural_scheduler_corpus_v1
```

The import copies WAVs into `inputs/wav`, preserves the generated canonical
transcripts, and stages the completed TextGrids under `outputs/mfa_align/latest`
for the normal gold-draft/export workflow. Text-group hashing keeps all four
voice renditions of one transcript in the same train, validation, or test
partition.

## Unified corpus

There is no separate old corpus and Data Factory corpus at grading or training
time. `inputs/corpus.csv` is the authoritative inventory for both:

- `origin=recorded` identifies the 441 Offgrid recordings. Their
  `existing_corpus_v1` seed/voice fields describe an optional recreation.
- `origin=data_factory` identifies the 400 imported Qwen 1.7B utterances. Their
  seed and voice describe the actual canonical WAV.

Every row points to its transcript, WAV, and approved gold package and records
the real `speaker_id` separately from the recipe voice. The runner, MFA/gold
tools, and timing split consume this manifest. `inputs/timing_split_v1.json` is
derived from it, so all renditions of one normalized transcript stay together
and held-out-speaker logic uses the actual speaker.

Rebuild or validate the inventory with:

```bat
python scripts\build_corpus_manifest.py
python scripts\build_corpus_manifest.py --check
python scripts\build_timing_dataset_split.py --check
```

The Offgrid-log and Data Factory importers rebuild the manifest and split after
copying assets. The recipe files remain under `inputs/speech_library` because
they are reproducibility inputs, not parallel corpus indexes.
