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
    int32 PhraseIndex = 0;

    // Text phrase/sentence metadata for diagnostics and weak boundary metadata only; runtime audio occupancy remains the timing authority.
    int32 SentenceIslandIndex = 0;
    int32 SourcePhoneIndex = INDEX_NONE;
    int32 SourcePhoneGlobalIndex = INDEX_NONE;
    FString SourcePhone;
    FString SourcePhoneBase;
    float PhoneLocalNorm = 0.5f;
    bool bIsStrongVisibleEvent = false;
    bool bIsDominant = false;
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
    int32 PhraseIndex = 0;
    int32 SentenceIslandIndex = 0;
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
    // Word metadata used by the online aligner for soft text boundary metadata and diagnostics.
    TArray<int32> WordSentenceIslandIndices;
    TArray<int32> WordPhraseIndices;
    TArray<int32> WordSyllableCounts;

    // Boundary punctuation following each tokenized word. Zero means no boundary. Runtime code treats this as a soft boundary metadata only.
    TArray<TCHAR> WordBoundaryPunctuationAfter;
};




class OFFGRIDAI_API FOffgridAITextVisemePlanner
{
public:
    static FOffgridAITextVisemePlan BuildPlan(const FText& Dialogue, float CharactersPerSecond = 14.0f, float MinDurationSeconds = 0.45f);
    static const TCHAR* ToPoseKey(EOffgridAITextViseme Viseme);
    static FString ToDebugString(EOffgridAITextViseme Viseme);
};
