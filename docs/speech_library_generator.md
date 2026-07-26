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
