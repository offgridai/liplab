#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"

struct FOffgridAIStreamingSyllablePositionEstimate
{
    int32 SyllableIndex = INDEX_NONE;
    int32 NucleusPhoneIndex = INDEX_NONE;
    int32 WordIndex = INDEX_NONE;
    int32 SpeechRegionIndex = INDEX_NONE;
    float AudioCenterSec = 0.0f;
    float DecisionSec = 0.0f;
    float MatchScore = 0.0f;
    float Confidence = 0.0f;
};

struct FOffgridAIStreamingDensePositionEstimate
{
    float AudioCenterSec = 0.0f;
    int32 SyllableIndex = INDEX_NONE;
    int32 NucleusPhoneIndex = INDEX_NONE;
    int32 WordIndex = INDEX_NONE;
    float SyllablePhase = 0.0f;
    float LocalRate = 1.0f;
    float Confidence = 0.0f;
    float RunnerUpMargin = 0.0f;
    float LastAnchorAgeSec = 0.0f;
};

struct FOffgridAIStreamingSyllableCandidateSet
{
    float AudioCenterSec = 0.0f;
    float DecisionSec = 0.0f;
    TArray<int32> SyllableIndices;
    TArray<float> Scores;
};

// Advisory monotonic matcher. Transcript phones own identity; acoustic pulses
// and nearby phone-family evidence only estimate progress through that chain.
class OFFGRIDAI_API FOffgridAIStreamingSyllablePositionEstimator
{
public:
    // Causal bounded candidate generator. Each pulse considers only the
    // current cursor and a few following transcript syllables. This stage
    // preserves alternatives; it does not commit identity or move playback.
    static TArray<FOffgridAIStreamingSyllableCandidateSet> EstimateCandidateSets(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
        int32 LookaheadSyllables = 2,
        int32 BeamWidth = 12,
        int32 MaxCandidates = 6);

    static TArray<FOffgridAIStreamingSyllablePositionEstimate> Estimate(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates);

    // Resolves one completed speech region to exactly one acoustic pulse time
    // per planned syllable. Callers must supply the already-established text/
    // audio region pairing; this method chooses timing, never identity.
    static TArray<FOffgridAIStreamingSyllablePositionEstimate> ResolveRegionPulseTrack(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
        int32 TextSpeechRegionIndex,
        float AudioStartSec,
        float AudioEndSec);

    static TArray<FOffgridAIStreamingDensePositionEstimate> EstimateDense(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates);

    // Accepts a past pulse assignment only after later pulses leave that
    // assignment unchanged. These anchors are advisory and never revise an
    // already accepted transcript identity.
    static TArray<FOffgridAIStreamingSyllablePositionEstimate> EstimateHistoricalAnchors(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
        const TArray<FOffgridAIStreamingSpeechRegion>& ObservedSpeechRegions,
        int32 RequiredStableUpdates = 2,
        float MinMatchScore = 0.85f);

    // Simulates rebasing the remaining text prior from the newest historical
    // anchor that is available before each audio frame reaches playback.
    static TArray<FOffgridAIStreamingDensePositionEstimate> EstimateHistoricalRebase(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
        const TArray<FOffgridAIStreamingSpeechRegion>& ObservedSpeechRegions,
        int32 RequiredStableUpdates = 2,
        float MinMatchScore = 0.85f);
};
