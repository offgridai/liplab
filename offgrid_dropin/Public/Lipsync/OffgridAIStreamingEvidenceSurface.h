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
    // Default history used by offline diagnostics. The live runtime selects a
    // shorter bounded history explicitly when making scheduling decisions.
    float PostrollSec = 1.500f;
    // Optional confirmed acoustic regions. When supplied, nucleus candidates
    // on opposite sides of a confirmed pause belong to different temporal
    // epochs and cannot suppress or replace one another.
    const TArray<FOffgridAIStreamingSpeechRegion>* SpeechRegions = nullptr;
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

// Extracts the acoustic observations consumed by the scheduler without choosing
// transcript identity or moving animation.
class OFFGRIDAI_API FOffgridAIStreamingEvidenceSurface
{
public:
    static TArray<FOffgridAIAudioLandmarkObservation> Analyze(
        const TArray<FOffgridAIStreamingAudioFeatureFrame>& Frames,
        const FOffgridAIStreamingEvidenceSurfaceConfig& Config);

    static FString LandmarkTypeToString(EOffgridAIAudioLandmarkType Type);
};
