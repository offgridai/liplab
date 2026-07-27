# LibTorch-free Offgrid neural runtime

## Packaging boundary

LibTorch is a training dependency only. The deployable Offgrid component is
`liplab_neural_streamer_cuda`, a small static CUDA library whose public API is
`offgrid_dropin/Public/Lipsync/OffgridAINeuralStreamerRuntime.h`. It links only
the CUDA Runtime and the C++ standard library. It does not load LibTorch,
cuDNN, Python, ONNX Runtime, or a graph executor.

The accepted version-3 checkpoint lives beside the authoritative Offgrid source
at `offgrid_dropin/Private/Lipsync/Models/OffgridAINeuralStreamerV3.bin`. The
runtime accepts either a file path or an in-memory byte span. File loading is
convenient for development and hot-swapping; memory loading lets an Offgrid host
embed the model in its executable, DLL, Unreal pak, or platform resource.

The checkpoint header validates its magic, schema version, feature dimensions,
parameter count, and exact byte count before allocating GPU weights. The runtime
also reports a 64-bit content fingerprint. The checked-in model is 171,436 bytes
and has SHA-256
`D2D0D02CC598FB45BF5705E8328B5ABB5AE3957C8811A6B0A666A02FBEACDB7E`.

## Standalone proof

Run:

```bat
scripts\build_neural_runtime.bat
```

This configures a CUDA-only build with `LIPLAB_ENABLE_TORCH=OFF`, produces native
RTX 4090 `sm_89` code, embeds the accepted checkpoint into
`liplab_neural_runtime_smoke.exe`, executes one real GPU inference chunk, and
fails if the executable imports a LibTorch DLL. The smoke executable needs no
checkpoint file at runtime; deleting or moving the source model after the build
does not affect it.

## Offgrid integration

For an in-process Offgrid build, compile
`offgrid_dropin/Private/Lipsync/OffgridAINeuralStreamerRuntime.cu`, expose
`offgrid_dropin/Public`, and link the CUDA Runtime. Construct one `CudaRuntime`
per concurrent lipsync stream, load the model once, call `ResetTokens` once per
utterance, and submit up to four 10 ms feature frames per `PushChunk`. The class
owns a persistent least-priority nonblocking CUDA stream and retains its causal
feature history; no allocation occurs on the recurring inference path.

An Offgrid packager may choose either:

- a separate `.bin` model asset, enabling model replacement without relinking;
- an embedded model resource using `LoadCheckpoint(data, size)`, producing one
  self-contained DLL or executable like the supplied smoke target.

The embedded form is the recommended default release package. The separate
asset form is useful for research builds. An out-of-process service is not
required and would add IPC latency, buffering, and another lifecycle boundary.

Failure remains explicit: if CUDA initialization, checkpoint validation, token
setup, or inference fails, `LastError()` describes the reason and the Offgrid
host emits no lipsync track for that utterance. There is no non-neural runtime
scheduler.
