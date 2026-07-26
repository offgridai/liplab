# Synthetic speech library generator

`scripts/generate_speech_library.py` is an offline data-factory prototype. It
does not participate in runtime lipsync and does not expose TTS timing to the
runtime scheduler.

The generator combines a fixed transcript set with explicit Qwen seeds,
synthesizes callback-path WAVs using a stored voice embedding, records exact
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
- `--voice-json <path>` selects a Qwen clone embedding.
- `--qwen-root <path>` or `--qwen-exe <path>` selects the TTS checkout/build.
- `--skip-synthesis` reuses existing WAVs and reruns MFA packaging.
- `--skip-mfa` generates WAVs and provenance without alignment.

The output manifest is the reproducibility contract. It records the Qwen Git
revision, hashes of the executable and voice JSON, explicit sampling settings,
the command for every case, WAV format/duration/hash, dictionary provenance,
and TextGrid hashes and interval counts.
