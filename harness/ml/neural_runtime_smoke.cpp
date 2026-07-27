#include "Lipsync/OffgridAINeuralStreamerModelFormat.h"
#include "Lipsync/OffgridAINeuralStreamerRuntime.h"
#include "Lipsync/OffgridAINeuralLipsyncBridge.h"
#include "OffgridAINeuralStreamerEmbeddedModel.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main()
{
    using namespace offgridai::neural_streamer;
    CudaRuntime runtime;
    if (runtime.LoadCheckpoint(
            offgridai::embedded_neural_streamer::ModelBytes,
            offgridai::embedded_neural_streamer::ModelSize - 1)
        || runtime.LastError().empty()) {
        std::cerr << "invalid embedded checkpoint was not rejected\n";
        return 1;
    }
    if (!runtime.LoadCheckpoint(
            offgridai::embedded_neural_streamer::ModelBytes,
            offgridai::embedded_neural_streamer::ModelSize)) {
        std::cerr << "embedded checkpoint load failed: " << runtime.LastError() << '\n';
        return 2;
    }

    const FOffgridAITextVisemePlan plan = FOffgridAITextVisemePlanner::BuildPlan(
        FText::FromString(FString(TEXT("Bring a pencil, a ruler, and clean paper."))));
    const FOffgridAINeuralTranscriptTensor transcript =
        FOffgridAINeuralLipsyncBridge::BuildTranscriptTensor(plan);
    if (transcript.NumTokens() <= kForwardTokenWindow
        || transcript.Features.Num() != transcript.NumTokens() * kTokenDimensions
        || !transcript.SilenceTokens[0]
        || !transcript.SilenceTokens[transcript.NumTokens() - 1]) {
        std::cerr << "Offgrid transcript bridge produced an invalid tensor\n";
        return 3;
    }
    if (!runtime.ResetTokens(transcript.Features.GetData(), transcript.NumTokens())) {
        std::cerr << "token setup failed: " << runtime.LastError() << '\n';
        return 4;
    }

    std::array<float, kRuntimeChunkFrames * kAudioFeatureCount> audio{};
    for (int frame = 0; frame < kRuntimeChunkFrames; ++frame) {
        FOffgridAIStreamingAudioFeatureFrame source;
        source.RMS = 0.02f + 0.005f * frame;
        source.RMSNorm = 0.35f + 0.08f * frame;
        source.DeltaRMS = 0.08f;
        source.Flux = 0.12f * frame;
        source.ZCR = 0.10f;
        source.LowBandNorm = source.RichLowBandNorm = 0.25f;
        source.MidBandNorm = source.RichMidBandNorm = 0.50f;
        source.HighBandNorm = source.RichHighBandNorm = 0.25f;
        source.SpectralCentroidNorm = source.RichSpectralCentroidNorm = 0.45f;
        source.Periodicity = source.RichPeriodicity = 0.60f;
        source.RichSpectralRolloffNorm = 0.65f;
        source.RichSpectralFlatness = 0.20f;
        source.RichSpectralFlux = source.Flux;
        FOffgridAINeuralLipsyncBridge::BuildAudioFeatureVector(
            source, audio.data() + frame * kAudioFeatureCount);
    }

    std::array<float, kRuntimeChunkFrames * kForwardTokenWindow> scores{};
    std::array<float, kRuntimeChunkFrames> region_logits{};
    int scored_tokens = 0;
    if (!runtime.PushChunk(audio.data(), kRuntimeChunkFrames, 0,
            scores.data(), scored_tokens, region_logits.data())) {
        std::cerr << "CUDA inference failed: " << runtime.LastError() << '\n';
        return 5;
    }
    if (scored_tokens != kForwardTokenWindow) {
        std::cerr << "unexpected forward token count: " << scored_tokens << '\n';
        return 6;
    }
    for (float value : scores)
        if (!std::isfinite(value)) return 7;
    for (float value : region_logits)
        if (!std::isfinite(value)) return 8;

    const RuntimeFootprint footprint = runtime.Footprint();
    if (footprint.CheckpointBytes != offgridai::embedded_neural_streamer::ModelSize
        || footprint.CheckpointFingerprint == 0)
        return 9;
    std::cout << "standalone_cuda_runtime=ok"
        << " model_bytes=" << footprint.CheckpointBytes
        << " weight_bytes=" << footprint.WeightBytes
        << " token_bytes=" << footprint.TokenBytes
        << " stream_priority=" << footprint.StreamPriority
        << " fingerprint=0x" << std::hex << footprint.CheckpointFingerprint
        << " sha256=" << offgridai::embedded_neural_streamer::ModelSha256
        << '\n';
    return 0;
}
