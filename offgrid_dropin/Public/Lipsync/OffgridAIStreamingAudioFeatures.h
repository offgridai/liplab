#pragma once

#include "CoreMinimal.h"

struct FOffgridAIStreamingAudioFeatureFrame
{
    float AudioBufferCenterSec = 0.0f;
    float AudioBufferStartSec = 0.0f;
    float AudioBufferEndSec = 0.0f;
    float RMS = 0.0f;
    float RMSNorm = 0.0f;
    float DeltaRMS = 0.0f;
    float Flux = 0.0f;
    float ZCR = 0.0f;
    float LowBandNorm = 0.0f;
    float MidBandNorm = 0.0f;
    float HighBandNorm = 0.0f;
    float SpectralCentroidNorm = 0.0f;
    float Periodicity = 0.0f;
    float RichLowBandNorm = 0.0f;
    float RichMidBandNorm = 0.0f;
    float RichHighBandNorm = 0.0f;
    float RichSpectralCentroidNorm = 0.0f;
    float RichSpectralRolloffNorm = 0.0f;
    float RichSpectralFlatness = 0.0f;
    float RichSpectralFlux = 0.0f;
    float RichPeriodicity = 0.0f;
};

// Causal 10 ms PCM feature extraction only. Speech regions and viseme timing
// are owned by the neural aligner; this class contains no scheduler or VAD.
class OFFGRIDAI_API FOffgridAIStreamingAudioFeatureExtractor
{
public:
    void Reset();
    void AppendPCM16(
        const TArray<uint8>& PCMChunk,
        int32 BytesToUse,
        int32 SampleRate,
        int32 NumChannels,
        int64 ChunkStartSample = -1);
    void Finalize(float FinalObservedAudioBufferEndSec = -1.0f);

    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetFeatureFrames() const
    {
        return FeatureFrames;
    }
    float GetObservedAudioBufferEndSec() const { return ObservedAudioBufferEndSec; }

private:
    void ProcessAnalysisFrame(
        float FrameStartSeconds,
        float FrameEndSeconds,
        float RMS,
        float ZCR,
        float LowBandNorm,
        float MidBandNorm,
        float HighBandNorm,
        float SpectralCentroidNorm,
        float Periodicity);

    TArray<FOffgridAIStreamingAudioFeatureFrame> FeatureFrames;
    TArray<float> PendingMonoSamples;
    TArray<float> RichAnalysisSamples;
    TArray<float> PreviousRichBandDistribution;
    int64 PendingSampleBase = 0;
    int32 ActiveSampleRate = 0;
    float SpeechPeakRMS = 0.0001f;
    float ObservedAudioBufferEndSec = 0.0f;
};
