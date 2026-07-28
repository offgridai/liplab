#include "Lipsync/OffgridAINeuralStreamerModelFormat.h"
#include "Lipsync/OffgridAINeuralStreamerRuntime.h"
#include "Lipsync/OffgridAINeuralLipsyncBridge.h"
#include "OffgridAINeuralStreamerEmbeddedModel.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

std::string cuda_version(int encoded)
{
    if (encoded <= 0) return "unknown";
    return std::to_string(encoded / 1000) + "."
        + std::to_string((encoded % 1000) / 10);
}

}  // namespace

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

    std::vector<float> full_scores(
        static_cast<size_t>(kRuntimeChunkFrames) * transcript.NumTokens());
    constexpr int kFullLatticeCalls = 256;
    const auto full_begin = std::chrono::steady_clock::now();
    for (int call = 0; call < kFullLatticeCalls; ++call) {
        if (!runtime.PushChunkAllTokens(
                audio.data(), kRuntimeChunkFrames, full_scores.data(),
                scored_tokens, region_logits.data())) {
            std::cerr << "full-lattice CUDA inference failed: "
                << runtime.LastError() << '\n';
            return 9;
        }
    }
    const auto full_end = std::chrono::steady_clock::now();
    if (scored_tokens != transcript.NumTokens()) return 10;
    const double full_mean_us = std::chrono::duration<double, std::micro>(
        full_end - full_begin).count() / kFullLatticeCalls;

    const RuntimeFootprint footprint = runtime.Footprint();
    const RuntimeDeviceInfo device = runtime.DeviceInfo();
    if (footprint.CheckpointBytes != offgridai::embedded_neural_streamer::ModelSize
        || footprint.CheckpointFingerprint == 0)
        return 11;
    std::cout << "standalone_cuda_runtime=ok"
        << " gpu=\"" << device.Name << '"'
        << " device=" << device.DeviceIndex
        << " compute_capability=" << device.ComputeCapabilityMajor << '.'
        << device.ComputeCapabilityMinor
        << " cuda_driver=" << cuda_version(device.DriverVersion)
        << " cuda_runtime=" << cuda_version(device.RuntimeVersion)
        << " model_bytes=" << footprint.CheckpointBytes
        << " weight_bytes=" << footprint.WeightBytes
        << " token_bytes=" << footprint.TokenBytes
        << " full_lattice_mean_us=" << std::fixed << std::setprecision(1)
        << full_mean_us
        << " stream_priority=" << footprint.StreamPriority
        << " fingerprint=0x" << std::hex << footprint.CheckpointFingerprint
        << " sha256=" << offgridai::embedded_neural_streamer::ModelSha256
        << '\n';
    return 0;
}
