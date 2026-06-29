#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"

// Phase 4. Owns envelope sampling only. It must not move centers, change pose ids,
// suppress authored events, or inspect text/audio evidence.
class OFFGRIDAI_API FOffgridAIVisemePerformer
{
public:
    static TArray<FOffgridAISubmittedVisemeSample> Sample(const FOffgridAIAlignedVisemeTrack& Track, float PlaybackSeconds, bool bGateBeforeSpeechStart = true);
    static TMap<FName, float> CollapseByPoseID(const TArray<FOffgridAISubmittedVisemeSample>& Samples);
    static TArray<FOffgridAIPerformedVisemeFrame> BuildFrames(const FOffgridAIAlignedVisemeTrack& Track, float FPS);

    static FOffgridAILipsyncPoseRuntimeState BuildPoseStateFromPoseWeights(const TMap<FName, float>& PoseWeights);
    static void BuildPoseWeightMapFromState(TMap<FName, float>& OutMap, const FOffgridAILipsyncPoseRuntimeState& State);
    static FOffgridAILipsyncPoseRuntimeState StepDisplayedPose(
        const FOffgridAILipsyncPoseRuntimeState& Current,
        const FOffgridAILipsyncPoseRuntimeState& Target,
        float DeltaTimeSeconds,
        float RestClosedWeight,
        bool bLineEndingOrIdle);
};
