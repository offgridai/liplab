#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAIStreamingEvidenceSurface.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"

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

};
