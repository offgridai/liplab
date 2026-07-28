#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"

// Phase 4. Owns envelope sampling only. It must not move centers, change pose ids,
// suppress authored events, or inspect text/audio evidence. Jaw articulation is
// supplied by the host's authored controls for each returned pose.
class OFFGRIDAI_API FOffgridAIVisemePerformer
{
public:
    static TArray<FOffgridAISubmittedVisemeSample> Sample(const FOffgridAICommittedVisemeTrack& Track, float PlaybackSeconds, bool bGateBeforeSpeechStart = true);
    static TMap<FName, float> CollapseByPoseID(const TArray<FOffgridAISubmittedVisemeSample>& Samples);

    // Builds and advances the complete detailed-pose state. Every eligible
    // library PoseID returned by Sample() is retained without shape reduction.
    static FOffgridAILipsyncPoseRuntimeState BuildPoseStateFromSamples(const TArray<FOffgridAISubmittedVisemeSample>& Samples);
    static void BuildPoseWeightMapFromState(TMap<FName, float>& OutMap, const FOffgridAILipsyncPoseRuntimeState& State);
    static FOffgridAILipsyncPoseRuntimeState StepDisplayedPose(
        const FOffgridAILipsyncPoseRuntimeState& Current,
        const FOffgridAILipsyncPoseRuntimeState& Target,
        float DeltaTimeSeconds,
        float RestClosedWeight,
        bool bLineEndingOrIdle);
};
