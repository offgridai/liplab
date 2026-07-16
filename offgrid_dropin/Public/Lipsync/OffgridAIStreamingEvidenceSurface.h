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
    // Exposes weaker phone-family candidates for a separate transcript
    // conditioning pass. It does not affect pulse or pause observations.
    bool bPermissivePhoneCandidates = false;
    // Enables the higher-recall pulse candidate experiment. Keep this false
    // for existing runtime consumers until sequence assignment is revalidated.
    bool bRefinedPulseCandidates = false;
    // Emits a dense set of short-scale sonority maxima for transcript-count-
    // conditioned pulse-track resolution. Do not consume these as independent
    // high-confidence pulse detections.
    bool bPermissivePulseCandidates = false;
    // Applies family-specific thresholds calibrated inside pulse-matched
    // syllable envelopes. Advisory until transcript assignment is validated.
    bool bRefinedIntraEnvelopePhoneCandidates = false;
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
