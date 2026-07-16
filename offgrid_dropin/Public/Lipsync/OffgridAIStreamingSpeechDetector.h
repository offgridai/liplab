#pragma once

#include "CoreMinimal.h"

struct FOffgridAIStreamingSpeechRegion
{
    int32 SpeechRegionIndex = INDEX_NONE;
    float AudioBufferStartSec = 0.0f;
    float AudioBufferLastSpeechSec = 0.0f;
    float AudioBufferEndSec = 0.0f;
    bool bStarted = false;
    bool bEnded = false;

    // Endpoint diagnostics. ProvisionalEndSec is the first quiet frame that
    // eventually closed or nearly closed this speech region; EndDecisionSec is when
    // the detector had enough trailing evidence to commit the close. ReopenCount
    // increments when speech resumes inside the reopen window and the region is
    // merged instead of split.
    float ProvisionalEndSec = -1.0f;
    float EndDecisionSec = -1.0f;
    int32 ReopenCount = 0;
    FName EndReason = NAME_None;
};

struct FOffgridAIStreamingSpeechGapCandidate
{
    int32 GapIndex = INDEX_NONE;
    int32 PrevSpeechRegionIndex = INDEX_NONE;
    int32 NextSpeechRegionIndex = INDEX_NONE;
    float GapStartSec = 0.0f;
    float GapEndSec = 0.0f;
    float GapDurationSec = 0.0f;
    float PrevSpeechRegionDurationSec = 0.0f;
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

// Advisory-only low-energy interval observed inside an active speech region.
// Unlike a gap candidate, this never changes speech-region segmentation.
struct FOffgridAIStreamingSoftLullCandidate
{
    int32 LullIndex = INDEX_NONE;
    int32 SpeechRegionIndex = INDEX_NONE;
    float LullStartSec = 0.0f;
    float LullEndSec = 0.0f;
    float LullDurationSec = 0.0f;
    int32 FrameCount = 0;
    float RelativeRMSSum = 0.0f;
    float RelativeRMSMin = 1.0f;
    float EvidenceSum = 0.0f;
    float EvidenceMin = 1.0f;
    float ReopenEvidence = 0.0f;
    float ReopenFlux = 0.0f;
    bool bStrongOnsetReopen = false;
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

    // Parallel 30 ms analysis window, evaluated at the existing 10 ms cadence.
    // These richer features are advisory and do not affect speech occupancy.
    float RichLowBandNorm = 0.0f;
    float RichMidBandNorm = 0.0f;
    float RichHighBandNorm = 0.0f;
    float RichSpectralCentroidNorm = 0.0f;
    float RichSpectralRolloffNorm = 0.0f;
    float RichSpectralFlatness = 0.0f;
    float RichSpectralFlux = 0.0f;
    float RichPeriodicity = 0.0f;
    // Normalized energies at the fixed Goertzel frequencies used by the
    // lightweight vowel-nucleus classifier.
    TArray<float> RichBandDistribution;

    // Detector-owned occupancy diagnostics captured at 10 ms frame cadence.
    float SpeechEvidence = 0.0f;
    float OpenThreshold = 0.0f;
    float CloseThreshold = 0.0f;
    float SilenceAccumSec = 0.0f;
    float EndpointCandidateStartSec = -1.0f;
    float ActiveSpeechRegionStartSec = -1.0f;
    float ActiveSpeechRegionEndSec = -1.0f;
    bool bInSpeechBeforeFrame = false;
    bool bInSpeechAfterFrame = false;
    bool bOpenCandidate = false;
    bool bKeepOpen = false;
    bool bStrongOnsetAnchor = false;
    bool bStrongQuiet = false;
    bool bLowEvidence = false;
    bool bEndpointCandidateActive = false;
    bool bFrameStartedSpeechRegion = false;
    bool bFrameClosedSpeechRegion = false;
    bool bFrameBridgedSpeechRegion = false;
    FName OccupancyDecision = NAME_None;

    bool bLocalRMSPeak = false;
    bool bLocalRMSValley = false;
    bool bLocalFluxPeak = false;

    float LearnedSpeechProbability = 0.0f;
    bool bLearnedSpeech = false;
    // Transcript punctuation may temporarily request greater sensitivity to
    // acoustic gaps while traversing a list. This never supplies timing or
    // changes speech evidence; it only selects the quiet-run duration used by
    // the learned region decoder for this frame.
    bool bListGapSensitive = false;

};

class OFFGRIDAI_API FOffgridAIStreamingSpeechDetector
{
public:
    void Reset();
    void SetListGapSensitivity(bool bEnabled) { bListGapSensitive = bEnabled; }
    void AppendPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample = -1);
    void Finalize(float FinalObservedAudioBufferEndSec = -1.0f);

    // Causal occupancy state used by live playback.
    const TArray<FOffgridAIStreamingSpeechRegion>& GetSpeechRegions() const { return SpeechRegions; }
    const TArray<FOffgridAIStreamingSpeechGapCandidate>& GetGapCandidates() const { return GapCandidates; }
    // Postroll-refined estimates used for boundary analysis and future delayed decisions.
    const TArray<FOffgridAIStreamingSpeechRegion>& GetRefinedSpeechRegions() const { return LearnedSpeechRegions; }
    const TArray<FOffgridAIStreamingSpeechGapCandidate>& GetRefinedGapCandidates() const { return LearnedGapCandidates; }
    const TArray<FOffgridAIStreamingSoftLullCandidate>& GetSoftLullCandidates() const { return SoftLullCandidates; }
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetFeatureFrames() const { return FeatureFrames; }
    bool HasObservedFirstSpeechStart() const { return bHasObservedFirstSpeechStart; }
    float GetFirstSpeechAudioBufferStartSec() const { return FirstSpeechAudioBufferStartSec; }
    float GetObservedAudioBufferEndSec() const { return ObservedAudioBufferEndSec; }

private:
    void ProcessAnalysisFrame(float FrameStartSeconds, float FrameEndSeconds, float RMS, float ZCR, float LowBandNorm, float MidBandNorm, float HighBandNorm, float SpectralCentroidNorm, float Periodicity);
    void SuppressRecentMicroSpeechRegionIfNeeded();
    void RefreshLocalFeatureFlags();
    void CommitPendingSoftLull();
    void RefinePendingGapFromRecoveredContext(float GapEndSec);
    void ProcessLearnedRegionFrames(bool bFlush);
    float ComputeLearnedSpeechProbability(int32 FrameIndex) const;
    void DecodeLearnedSpeechFrame(int32 FrameIndex);

    TArray<FOffgridAIStreamingSpeechRegion> SpeechRegions;
    TArray<FOffgridAIStreamingSpeechGapCandidate> GapCandidates;
    TArray<FOffgridAIStreamingSoftLullCandidate> SoftLullCandidates;
    TArray<FOffgridAIStreamingAudioFeatureFrame> FeatureFrames;
    TArray<FOffgridAIStreamingSpeechRegion> LearnedSpeechRegions;
    TArray<FOffgridAIStreamingSpeechGapCandidate> LearnedGapCandidates;
    int32 LearnedNextFrameIndex = 0;
    int32 LearnedSpeechCandidateStartFrame = INDEX_NONE;
    int32 LearnedQuietCandidateStartFrame = INDEX_NONE;
    bool bLearnedInSpeech = false;
    bool bLearnedHasObservedFirstSpeechStart = false;
    float LearnedFirstSpeechAudioBufferStartSec = 0.0f;
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
    float ActiveSpeechRegionPeakRMS = 0.0001f;
    TArray<float> ActiveSpeechRegionRMSHistory;
    float ActiveSpeechRegionSeconds = 0.0f;
    float ActiveSoftCollapseAccumSeconds = 0.0f;
    float ActiveSoftCollapseStartSeconds = 0.0f;
    float ActiveLowEnergyAccumSeconds = 0.0f;
    float ActiveLowEnergyStartSeconds = 0.0f;
    float ActiveHardCollapseAccumSeconds = 0.0f;
    float ActiveHardCollapseStartSeconds = 0.0f;
    bool bHasObservedFirstSpeechStart = false;
    float FirstSpeechAudioBufferStartSec = 0.0f;
    float ObservedAudioBufferEndSec = 0.0f;
    bool bPendingGapCandidateActive = false;
    FOffgridAIStreamingSpeechGapCandidate PendingGapCandidate;
    bool bPendingSoftLullActive = false;
    FOffgridAIStreamingSoftLullCandidate PendingSoftLull;

    TArray<float> PendingMonoSamples;
    TArray<float> RichAnalysisSamples;
    TArray<float> PreviousRichBandDistribution;
    int64 PendingSampleBase = 0;
    int32 ActiveSampleRate = 0;

    float SpeechPeakRMS = 0.0001f;
    float NoiseFloorRMS = 0.0001f;
    bool bListGapSensitive = false;
};
