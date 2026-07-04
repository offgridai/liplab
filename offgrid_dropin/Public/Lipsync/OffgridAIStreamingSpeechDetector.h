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

struct FOffgridAIStreamingSpeechGapCandidate
{
    int32 GapIndex = INDEX_NONE;
    int32 PrevIslandIndex = INDEX_NONE;
    int32 NextIslandIndex = INDEX_NONE;
    float GapStartSec = 0.0f;
    float GapEndSec = 0.0f;
    float GapDurationSec = 0.0f;
    float PrevIslandDurationSec = 0.0f;
    float QuietEvidence = 0.0f;
    float QuietRMSNorm = 0.0f;
    int32 GapFrameCount = 0;
    int32 LowEvidenceFrameCount = 0;
    int32 StrongQuietFrameCount = 0;
    float GapEvidenceSum = 0.0f;
    float GapRMSNormSum = 0.0f;
    float GapPeriodicitySum = 0.0f;
    float GapFluxSum = 0.0f;
    float GapCentroidSum = 0.0f;
    float GapEvidenceMin = 1.0f;
    float GapRMSNormMin = 1.0f;
    float GapPeriodicityMax = 0.0f;
    float GapFluxMax = 0.0f;
    float ReopenEvidence = 0.0f;
    float ReopenFlux = 0.0f;
    bool bStrongQuietClose = false;
    bool bStrongOnsetReopen = false;
    bool bBridged = false;
    FName CloseReason = NAME_None;
    FName DecisionClass = NAME_None;
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

    // Detector-owned occupancy diagnostics captured at 10 ms frame cadence.
    float SpeechEvidence = 0.0f;
    float OpenThreshold = 0.0f;
    float CloseThreshold = 0.0f;
    float SilenceAccumSec = 0.0f;
    float EndpointCandidateStartSec = -1.0f;
    float ActiveIslandStartSec = -1.0f;
    float ActiveIslandEndSec = -1.0f;
    bool bInSpeechBeforeFrame = false;
    bool bInSpeechAfterFrame = false;
    bool bOpenCandidate = false;
    bool bKeepOpen = false;
    bool bStrongOnsetAnchor = false;
    bool bStrongQuiet = false;
    bool bLowEvidence = false;
    bool bEndpointCandidateActive = false;
    bool bFrameStartedIsland = false;
    bool bFrameClosedIsland = false;
    bool bFrameBridgedIsland = false;
    FName OccupancyDecision = NAME_None;

    bool bLocalRMSPeak = false;
    bool bLocalRMSValley = false;
    bool bLocalFluxPeak = false;

    // Detector-owned pause-family cue for nearby silence/gap structure.
    FName PauseFamily = NAME_None;
    float PauseFamilyConfidence = 0.0f;
    float PauseGapAgeSec = 0.0f;
};

struct FOffgridAIStreamingPauseCue
{
    FName Family = NAME_None;
    float Confidence = 0.0f;
    float GapAgeSec = 0.0f;
    bool bInSpeech = false;
    bool bEndpointCandidateActive = false;
};

class OFFGRIDAI_API FOffgridAIStreamingSpeechDetector
{
public:
    void Reset();
    void AppendPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample = -1);
    void Finalize(float FinalObservedAudioBufferEndSec = -1.0f);

    const TArray<FOffgridAIStreamingSpeechIsland>& GetIslands() const { return Islands; }
    const TArray<FOffgridAIStreamingSpeechGapCandidate>& GetGapCandidates() const { return GapCandidates; }
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetFeatureFrames() const { return FeatureFrames; }
    const FOffgridAIStreamingPauseCue& GetLatestPauseCue() const { return LatestPauseCue; }
    bool HasObservedFirstSpeechStart() const { return bHasObservedFirstSpeechStart; }
    float GetFirstSpeechAudioBufferStartSec() const { return FirstSpeechAudioBufferStartSec; }
    float GetObservedAudioBufferEndSec() const { return ObservedAudioBufferEndSec; }

private:
    void ProcessAnalysisFrame(float FrameStartSeconds, float FrameEndSeconds, float RMS, float ZCR, float LowBandNorm, float MidBandNorm, float HighBandNorm, float SpectralCentroidNorm, float Periodicity);
    void RefreshLocalFeatureFlags();

    TArray<FOffgridAIStreamingSpeechIsland> Islands;
    TArray<FOffgridAIStreamingSpeechGapCandidate> GapCandidates;
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
    float EndpointCandidateMinEvidence = 1.0f;
    int32 EndpointCandidateFrameCount = 0;
    int32 EndpointCandidateLowEvidenceFrameCount = 0;
    int32 EndpointCandidateStrongQuietFrameCount = 0;
    float EndpointCandidateEvidenceSum = 0.0f;
    float EndpointCandidateRMSNormSum = 0.0f;
    float EndpointCandidatePeriodicitySum = 0.0f;
    float EndpointCandidateFluxSum = 0.0f;
    float EndpointCandidateCentroidSum = 0.0f;
    float EndpointCandidateMinRMSNorm = 1.0f;
    float EndpointCandidateMaxPeriodicity = 0.0f;
    float EndpointCandidateMaxFlux = 0.0f;
    float ActiveIslandPeakRMS = 0.0001f;
    float ActiveIslandSpeechSeconds = 0.0f;
    float ActiveLowEnergyAccumSeconds = 0.0f;
    float ActiveLowEnergyStartSeconds = 0.0f;
    bool bHasObservedFirstSpeechStart = false;
    float FirstSpeechAudioBufferStartSec = 0.0f;
    float ObservedAudioBufferEndSec = 0.0f;
    bool bPendingGapCandidateActive = false;
    FOffgridAIStreamingSpeechGapCandidate PendingGapCandidate;

    TArray<float> PendingMonoSamples;
    int64 PendingSampleBase = 0;
    int32 ActiveSampleRate = 0;

    float SpeechPeakRMS = 0.0001f;
    float NoiseFloorRMS = 0.0001f;
    FOffgridAIStreamingPauseCue LatestPauseCue;
};
