# LipLab

LipLab is a standalone research harness for OffgridAI's streaming lipsync
runtime. It builds the shared C++/CUDA implementation, runs a speech corpus, and
compares the resulting animation with forced-alignment references.

The runtime keeps transcript-derived viseme identity and order authoritative.
Audio controls timing only; it cannot invent or reorder visemes.

## Requirements

- Windows 10 or 11
- Visual Studio 2022 with C++ tools
- CMake and Ninja (the Visual Studio copies are detected automatically)
- Python 3
- An NVIDIA GPU and CUDA toolkit compatible with the packaged model

LibTorch and Montreal Forced Aligner are needed only to train models or rebuild
reference data. They are not runtime dependencies.

## Build and verify

Run the complete check from a Visual Studio command prompt or a normal command
prompt:

```bat
scripts\verify.bat
```

This builds the runtime, validates the corpus and reference data, runs all 841
cases, writes results to `outputs/runs/latest`, and checks them against the
accepted baseline.

## Repository layout

- `offgrid_dropin/`: authoritative runtime code shared with OffgridAI
- `standalone_ue_shim/`: minimal Unreal-compatible types for standalone builds
- `harness/`: corpus runner and offline training tools
- `inputs/`: transcripts, audio, and approved timing references
- `scripts/`: validation, grading, and data-maintenance commands
- `docs/`: focused architecture, training, and regression notes

Start with [lipsync.md](lipsync.md) for the runtime design and
[docs/regression_policy.md](docs/regression_policy.md) for scoring. Model
training is described in [docs/neural_runtime.md](docs/neural_runtime.md).

## Data and model

The checked-in corpus contains 441 recorded utterances and 400 synthetic
utterances generated with Qwen voice-clone references. The packaged checkpoint
is evaluated against this unified corpus. The training workflow can fine-tune
from it, but the accepted weights may predate newly imported regression cases
when candidate checkpoints fail promotion. The Apache license grants copyright
and patent permissions; it does not grant rights to a person's voice or
likeness. Confirm speaker and voice-reference consent before redistributing
these assets.

## Contributing

Keep the runtime deterministic and preserve transcript ownership of viseme
identity and order. Before submitting a change, run `scripts\verify.bat` and
include any intentional baseline change in the same review.

## License

Except for the bundled CMU Pronouncing Dictionary, LipLab is licensed under the
[Apache License 2.0](LICENSE). CMUdict retains its original permissive license;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
