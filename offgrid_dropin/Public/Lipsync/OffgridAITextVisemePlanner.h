#pragma once

#include "CoreMinimal.h"

// Text-first lipsync viseme groups. These are articulation groups, not literal phonemes.
enum class EOffgridAITextViseme : uint8
{
    Rest,
    MBP, // M / B / P bilabial closure
    AAA, // Open vowel / jaw-open family
    EEE, // Wide vowel family
    OOO, // Rounded vowel family
    WUH, // W / rounded onset funnel
    FVS  // F / V / S teeth/fricative family
};

struct FOffgridAITextVisemeEvent
{
    float StartNorm = 0.0f;
    float EndNorm = 0.0f;
    EOffgridAITextViseme Viseme = EOffgridAITextViseme::Rest;

    // Direct MetaHuman viseme pose id from MetaHumanVisemeLibrary.json.
    FName PoseID = NAME_None;

    float Strength = 0.0f;
    FString SourceText;
    int32 WordIndex = INDEX_NONE;

    // Text sentence metadata for diagnostics only; runtime audio occupancy remains the timing authority.
    int32 SentenceIndex = 0;
    int32 SourcePhoneIndex = INDEX_NONE;
    int32 SourcePhoneGlobalIndex = INDEX_NONE;
    FString SourcePhone;
    FString SourcePhoneBase;
    float PhoneLocalNorm = 0.5f;
    bool bIsStrongVisibleEvent = false;
    FName Generator = NAME_None;
};


struct FOffgridAIExpectedPhone
{
    int32 PhoneIndex = INDEX_NONE;
    int32 WordPhoneIndex = INDEX_NONE;
    FString Phone;
    FString BasePhone;
    FString SourceWord;
    int32 WordIndex = INDEX_NONE;
    int32 SentenceIndex = 0;
    bool bIsVowel = false;
    bool bIsVisibleViseme = false;
    int32 FirstVisibleEventIndex = INDEX_NONE;
    TCHAR BoundaryAfterWord = TCHAR(0);
    float WeightSeconds = 0.075f;
};

struct FOffgridAITextVisemePlan
{
    TArray<FOffgridAITextVisemeEvent> Events;
    TArray<FOffgridAIExpectedPhone> ExpectedPhones;
    float EstimatedDurationSeconds = 0.0f;
    // Word metadata used for text planning and sentence diagnostics.
    TArray<int32> WordSentenceIndices;
    TArray<int32> WordSyllableCounts;
    TArray<int32> WordPhoneBeginIndices;
    TArray<int32> WordPhoneEndIndices;
    TArray<int32> WordVisibleEventBeginIndices;
    TArray<int32> WordVisibleEventEndIndices;

    // Boundary punctuation following each tokenized word. Zero means no boundary.
    TArray<TCHAR> WordBoundaryPunctuationAfter;
};




class OFFGRIDAI_API FOffgridAITextVisemePlanner
{
public:
    static FOffgridAITextVisemePlan BuildPlan(const FText& Dialogue, float CharactersPerSecond = 14.0f, float MinDurationSeconds = 0.45f);
    static const TCHAR* ToPoseKey(EOffgridAITextViseme Viseme);
    static FString ToDebugString(EOffgridAITextViseme Viseme);
};
