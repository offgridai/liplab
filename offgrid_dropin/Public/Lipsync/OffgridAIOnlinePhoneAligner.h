#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

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
    bool bFinal = false;
};

struct FOffgridAIOnlinePhoneAlignmentResult
{
    int32 HighestAlignedPhoneIndex = INDEX_NONE;
    int32 HighestAlignedWordIndex = INDEX_NONE;
    bool bHasSpeechEvidence = false;

    TArray<float> PhoneStartSeconds;
    TArray<float> PhoneCenterSeconds;
    TArray<float> PhoneEndSeconds;
    TArray<float> PhoneMatchScores;
    TArray<FName> PhoneAdvanceReasons;

    EOffgridAIPhoneClass CurrentExpectedClass = EOffgridAIPhoneClass::Unknown;
    float CurrentExpectedClassScore = 0.0f;
    FName AdvanceReason = FName(TEXT("none"));
};

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
