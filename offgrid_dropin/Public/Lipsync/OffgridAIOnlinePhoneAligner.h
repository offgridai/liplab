#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"
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

struct FOffgridAIOnlinePhoneAlignmentInput
{
    const FOffgridAITextVisemePlan* Plan = nullptr;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames = nullptr;
    const TArray<FOffgridAIStreamingSpeechIsland>* SpeechIslands = nullptr;
    float ObservedAudioEndSec = 0.0f;
    float PlaybackSec = 0.0f;
    float LookaheadSec = 0.350f;
    float CommitLagSec = 0.120f;
    int32 BeamWidth = 18;
    bool bFinal = false;
};

struct FOffgridAIOnlinePhoneAlignmentResult
{
    int32 HighestAlignedPhoneIndex = INDEX_NONE;
    int32 HighestAlignedWordIndex = INDEX_NONE;
    bool bHasSpeechEvidence = false;
    float VisibleSpeechSeconds = 0.0f;
    float VisibleExpectedSeconds = 0.0f;
    float SpeechRateScale = 1.0f;

    TArray<float> PhoneStartSeconds;
    TArray<float> PhoneCenterSeconds;
    TArray<float> PhoneEndSeconds;
    TArray<float> PhoneMatchScores;
    TArray<float> PhoneScoreGaps;
    TArray<float> PhoneObservedDurations;
    TArray<float> PhoneExpectedDurations;
    // Diagnostic-only: audio-only lexical-transition salience observed near the
    // transition into each aligned phone.  The aligner uses this as a weak
    // reward when the expected transcript crosses a word boundary, never as a
    // hard segmentation constraint.
    TArray<float> PhoneWordBoundarySalience;
    TArray<FName> PhoneAdvanceReasons;

    EOffgridAIPhoneClass CurrentExpectedClass = EOffgridAIPhoneClass::Unknown;
    float CurrentExpectedClassScore = 0.0f;
    FName AdvanceReason = FName(TEXT("none"));
};

// Streaming forced alignment over the expected CMU phone path.
//
// This replaces the previous occupancy-stretch aligner.  The algorithm is still
// lightweight enough for runtime use, but it is structurally MFA-inspired:
// a monotonic left-to-right phone sequence is segmented over observed speech
// frames using acoustic emission scores plus duration priors.  The aligner never
// invents phones and never reorders the transcript-derived phone path.
class OFFGRIDAI_API FOffgridAIOnlinePhoneAligner
{
public:
    static FOffgridAIOnlinePhoneAlignmentResult Compute(const FOffgridAIOnlinePhoneAlignmentInput& Input);
    static int32 FindPhoneForEvent(const FOffgridAITextVisemePlan& Plan, const FOffgridAITextVisemeEvent& Event);

    static EOffgridAIPhoneClass ClassForPhoneBase(const FString& PhoneBase);
    static FOffgridAIPhoneClassScores ScoreFramePhoneClasses(const FOffgridAIStreamingAudioFeatureFrame& Frame);
    static float ScoreForClass(const FOffgridAIPhoneClassScores& Scores, EOffgridAIPhoneClass PhoneClass);
    static FString PhoneClassToString(EOffgridAIPhoneClass PhoneClass);
};
