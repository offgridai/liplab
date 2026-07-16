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

struct FOffgridAIStreamingSyllableCandidateSet
{
    float AudioCenterSec = 0.0f;
    float DecisionSec = 0.0f;
    TArray<int32> SyllableIndices;
    TArray<float> Scores;
};

// Monotonic matcher used by the runtime's bounded suffix correction. Transcript
// phones own identity; acoustic pulses only estimate progress through the plan.
class OFFGRIDAI_API FOffgridAIStreamingSyllablePositionEstimator
{
public:
    // Exposes nearby candidates for regression grading of the active matcher.
    static TArray<FOffgridAIStreamingSyllableCandidateSet> EstimateCandidateSets(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
        int32 LookaheadSyllables = 2,
        int32 BeamWidth = 12,
        int32 MaxCandidates = 6);

    static TArray<FOffgridAIStreamingSyllablePositionEstimate> Estimate(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates);

    // Accepts a past assignment only after later evidence leaves it unchanged.
    // The runtime can then move only the uncommitted suffix by a bounded amount.
    static TArray<FOffgridAIStreamingSyllablePositionEstimate> EstimateHistoricalAnchors(
        const FOffgridAITextVisemePlan& Plan,
        const TArray<FOffgridAIAudioLandmarkObservation>& EvidenceCandidates,
        const TArray<FOffgridAIStreamingSpeechRegion>& ObservedSpeechRegions,
        int32 RequiredStableUpdates = 2,
        float MinMatchScore = 0.85f);

};
