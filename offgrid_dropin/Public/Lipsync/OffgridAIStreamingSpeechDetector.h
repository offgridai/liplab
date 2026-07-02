#pragma once

#include "CoreMinimal.h"

struct FOffgridAIStreamingSpeechIsland
{
    int32 IslandIndex = INDEX_NONE;
    float AudioBufferStartSec = 0.0f;
    float AudioBufferLastSpeechSec = 0.0f;
    float AudioBufferEndSec = 0.0f;
    bool bStarted = false;
    bool bEnded = false;

    // Endpoint diagnostics. ProvisionalEndSec is the first quiet frame that
    // eventually closed or nearly closed this island; EndDecisionSec is when
    // the detector had enough trailing evidence to commit the close. ReopenCount
    // increments when speech resumes inside the reopen window and the island is
    // merged instead of split.
    float ProvisionalEndSec = -1.0f;
    float EndDecisionSec = -1.0f;
    int32 ReopenCount = 0;
    FName EndReason = NAME_None;
};

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

    bool bLocalRMSPeak = false;
    bool bLocalRMSValley = false;
    bool bLocalFluxPeak = false;
};

class OFFGRIDAI_API FOffgridAIStreamingSpeechDetector
{
public:
    void Reset();
    void AppendPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample = -1);
    void Finalize(float FinalObservedAudioBufferEndSec = -1.0f);

    const TArray<FOffgridAIStreamingSpeechIsland>& GetIslands() const { return Islands; }
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetFeatureFrames() const { return FeatureFrames; }
    bool HasObservedFirstSpeechStart() const { return bHasObservedFirstSpeechStart; }
    float GetFirstSpeechAudioBufferStartSec() const { return FirstSpeechAudioBufferStartSec; }
    float GetObservedAudioBufferEndSec() const { return ObservedAudioBufferEndSec; }

private:
    void ProcessAnalysisFrame(float FrameStartSeconds, float FrameEndSeconds, float RMS, float ZCR, float LowBandNorm, float MidBandNorm, float HighBandNorm, float SpectralCentroidNorm, float Periodicity);
    void RefreshLocalFeatureFlags();

    TArray<FOffgridAIStreamingSpeechIsland> Islands;
    TArray<FOffgridAIStreamingAudioFeatureFrame> FeatureFrames;
    bool bInSpeech = false;
    bool bSpeechCandidateActive = false;
    float SpeechCandidateStartSeconds = 0.0f;
    float SpeechCandidateAccumSeconds = 0.0f;
    float SpeechCandidatePeakEvidence = 0.0f;
    float SilenceAccumSeconds = 0.0f;
    float SilenceStartSeconds = 0.0f;
    bool bEndpointCandidateActive = false;
    float EndpointCandidateStartSeconds = 0.0f;
    float ActiveIslandPeakRMS = 0.0001f;
    float ActiveIslandSpeechSeconds = 0.0f;
    float ActiveLowEnergyAccumSeconds = 0.0f;
    float ActiveLowEnergyStartSeconds = 0.0f;
    bool bHasObservedFirstSpeechStart = false;
    float FirstSpeechAudioBufferStartSec = 0.0f;
    float ObservedAudioBufferEndSec = 0.0f;

    TArray<float> PendingMonoSamples;
    int64 PendingSampleBase = 0;
    int32 ActiveSampleRate = 0;

    float SpeechPeakRMS = 0.0001f;
    float NoiseFloorRMS = 0.0001f;
};
