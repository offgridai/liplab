#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"

// Phase 4. Owns envelope sampling only. It must not move centers, change pose ids,
// suppress authored events, or inspect text/audio evidence. Sample() may also
// return the reserved "JawOpen" carrier: hosts should apply it to mandibular
// aperture after resolving the ordinary transcript-derived pose samples.
class OFFGRIDAI_API FOffgridAIVisemePerformer
{
public:
    static TArray<FOffgridAISubmittedVisemeSample> Sample(const FOffgridAICommittedVisemeTrack& Track, float PlaybackSeconds, bool bGateBeforeSpeechStart = true);
    static TMap<FName, float> CollapseByPoseID(const TArray<FOffgridAISubmittedVisemeSample>& Samples);

    // Builds and advances the complete detailed-pose state. Every eligible
    // PoseID returned by Sample() is retained; no six-shape reduction occurs.
    static FOffgridAILipsyncPoseRuntimeState BuildPoseStateFromSamples(const TArray<FOffgridAISubmittedVisemeSample>& Samples);
    static void BuildPoseWeightMapFromState(TMap<FName, float>& OutMap, const FOffgridAILipsyncPoseRuntimeState& State);
    static FOffgridAILipsyncPoseRuntimeState StepDisplayedPose(
        const FOffgridAILipsyncPoseRuntimeState& Current,
        const FOffgridAILipsyncPoseRuntimeState& Target,
        float DeltaTimeSeconds,
        float RestClosedWeight,
        bool bLineEndingOrIdle);
};
