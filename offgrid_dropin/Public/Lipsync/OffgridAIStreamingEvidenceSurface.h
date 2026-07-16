#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

enum class EOffgridAIAudioLandmarkType : uint8
{
    SyllabicPulse,
    Lull,
    Resume,
    Bilabial,
    Labiodental,
    Glide,
    Sibilant,
    OpenVowel,
    FrontVowel,
    RoundedVowel
};

struct FOffgridAIStreamingEvidenceSurfaceConfig
{
    float PrerollSec = 0.350f;
    float PostrollSec = 1.500f;
    // Exposes lower-confidence phone-family candidates for offline grading.
    // Runtime uses the conservative default.
    bool bPermissivePhoneCandidates = false;
};

struct FOffgridAIAudioLandmarkObservation
{
    EOffgridAIAudioLandmarkType Type = EOffgridAIAudioLandmarkType::SyllabicPulse;
    float StartSec = 0.0f;
    float EndSec = 0.0f;
    float CenterSec = 0.0f;
    float DecisionSec = 0.0f;
    float Score = 0.0f;
};

// Extracts acoustic observations without choosing transcript identity or moving
// animation. Each decision is limited to the recent postroll and available
// preroll around the simulated audible playhead.
class OFFGRIDAI_API FOffgridAIStreamingEvidenceSurface
{
public:
    static TArray<FOffgridAIAudioLandmarkObservation> Analyze(
        const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
        const FOffgridAIStreamingEvidenceSurfaceConfig& Config);

    static FString LandmarkTypeToString(EOffgridAIAudioLandmarkType Type);
};
