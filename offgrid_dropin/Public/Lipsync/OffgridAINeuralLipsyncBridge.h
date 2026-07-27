#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAINeuralStreamerModelFormat.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"

// Engine-facing tensor adapters for the standalone CUDA runtime. These are the
// only authoritative mappings from Offgrid transcript/audio types into model
// features; hosts should not reproduce the feature layout independently.
struct FOffgridAINeuralTranscriptTensor
{
    TArray<float> Features;
    TArray<int32> EventIndices;
    TArray<int32> WordIndices;
    TArray<int32> PhoneIndices;
    TArray<bool> SilenceTokens;
    TArray<bool> SentenceBoundaryTokens;
    // Silence inserted after punctuation that explicitly expects a pause.
    // The decoder uses this to wait for neural quiet-then-resume evidence.
    TArray<bool> PauseBoundaryTokens;

    int32 NumTokens() const { return EventIndices.Num(); }
};

class OFFGRIDAI_API FOffgridAINeuralLipsyncBridge
{
public:
    static FOffgridAINeuralTranscriptTensor BuildTranscriptTensor(
        const FOffgridAITextVisemePlan& Plan);
    static void BuildAudioFeatureVector(
        const FOffgridAIStreamingAudioFeatureFrame& Frame,
        float* OutFeatures);
};
