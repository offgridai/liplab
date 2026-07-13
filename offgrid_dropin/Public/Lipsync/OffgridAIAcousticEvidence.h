#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

// Broad acoustic classes used by the streaming forced aligner.  These are not
// identity decisions: transcript/CMU phones remain authoritative and these
// classes only score how plausible a frame is for the next expected phone.
enum class EOffgridAIPhoneClass : uint8
{
    Silence,
    VowelOpen,
    VowelFront,
    VowelRound,
    Bilabial,
    Labiodental,
    Dental,
    Sibilant,
    StopBurst,
    Liquid,
    Glide,
    Nasal,
    Unknown
};

struct FOffgridAIPhoneClassScores
{
    float Silence = 0.0f;
    float VowelOpen = 0.0f;
    float VowelFront = 0.0f;
    float VowelRound = 0.0f;
    float Bilabial = 0.0f;
    float Labiodental = 0.0f;
    float Dental = 0.0f;
    float Sibilant = 0.0f;
    float StopBurst = 0.0f;
    float Liquid = 0.0f;
    float Glide = 0.0f;
    float Nasal = 0.0f;
    float Unknown = 0.0f;
};

// One frame-level evidence field for all advisory acoustic reasoning.
//
// The runtime still treats the CMU transcript plan as authoritative; this field
// only describes how the current audio frame *sounds*.  Phone scoring, lexical
// transition hints, and progress-density diagnostics are all derived from this
// one probability field so the code does not contain competing acoustic
// detectors with slightly different meanings.
struct FOffgridAIArticulatoryProbabilityField
{
    float Speech = 0.0f;
    float Silence = 0.0f;
    float Voiced = 0.0f;
    float Vowel = 0.0f;
    float Fricative = 0.0f;
    float Closure = 0.0f;
    float Release = 0.0f;
    float Sonorant = 0.0f;
    float Transition = 0.0f;
    float RedHerring = 0.0f;

    float LowTilt = 0.0f;
    float HighTilt = 0.0f;
    float SpectralChange = 0.0f;
    float EnergyChange = 0.0f;

    FOffgridAIPhoneClassScores PhoneScores;
};

// Lightweight frame classifier used by transcript-conditioned runtime anchors.
// It never aligns, invents, or reorders phones; transcript identity remains
// authoritative.
class OFFGRIDAI_API FOffgridAIAcousticEvidence
{
public:
    static EOffgridAIPhoneClass ClassForPhoneBase(const FString& PhoneBase);
    static FOffgridAIArticulatoryProbabilityField BuildArticulatoryProbabilityField(const FOffgridAIStreamingAudioFeatureFrame& Frame);
    static FOffgridAIPhoneClassScores ScoreFramePhoneClasses(const FOffgridAIStreamingAudioFeatureFrame& Frame);
    static float ScoreForClass(const FOffgridAIPhoneClassScores& Scores, EOffgridAIPhoneClass PhoneClass);
    static FString PhoneClassToString(EOffgridAIPhoneClass PhoneClass);
};
