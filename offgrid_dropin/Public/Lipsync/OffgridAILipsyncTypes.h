#pragma once

#include "CoreMinimal.h"

struct FOffgridAIAlignedVisemeEvent
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    float Strength = 0.0f;
    FString SourceWord;
    int32 WordIndex = INDEX_NONE;
    union
    {
        int32 SpeechRegionIndex = INDEX_NONE;
        // Deprecated compatibility alias for older Offgrid-side log/export code.
        int32 PhraseIndex;
    };
    // Text sentence metadata only. Runtime speech region ownership comes from
    // occupancy detection, not from this field.
    int32 SentenceIndex = INDEX_NONE;
    // Strong visible event metadata for diagnostics/render emphasis only. It is not a scheduling landmark.
    bool bIsStrongVisibleEvent = false;

    float TextCenterNorm = 0.0f;
    float TextDiagnosticCenterSeconds = 0.0f;

    float FinalRenderCenterSeconds = 0.0f;
    float RenderStartSeconds = 0.0f;
    float RenderEndSeconds = 0.0f;
    float PriorStartSeconds = 0.0f;
    float PriorCenterSeconds = 0.0f;
    float PriorEndSeconds = 0.0f;
    float LeadAdjustedCenterSeconds = 0.0f;
    float PlaybackOffsetSeconds = 0.0f;
    float TotalPausedSecondsAtCommit = 0.0f;
    float MinLiveLeadDelaySeconds = 0.0f;
    float InterEventFloorDelaySeconds = 0.0f;
    float TotalCenterDelaySeconds = 0.0f;

    int32 SourcePhoneIndex = INDEX_NONE;
    FString SourcePhoneBase;
    FName SourcePhoneClass = NAME_None;
    bool bMappedToObservedSpeech = false;

    float CommitPlaybackSeconds = 0.0f;
    float CommitLeadSeconds = 0.0f;
    FName CommitReason = FName(TEXT("unknown"));

    float RequiredActiveElapsedSeconds = 0.0f;
    float ObservedActiveElapsedSeconds = 0.0f;
    float ActiveProgressDeficitSeconds = 0.0f;
    float RequiredProgressNorm = 0.0f;
    float ObservedProgressNorm = 0.0f;
    float ActiveProgressRatio = 1.0f;

    float DetectedWordStartSeconds = 0.0f;
    bool bDetectedWordStartMappedToObservedSpeech = false;

    bool bLandmarkPacingEnabled = false;
    bool bLandmarkPacingSeeded = false;
    float LandmarkPacingRateAtCommit = 1.0f;
    float LandmarkPacingConfidenceAtCommit = 0.0f;
    float LandmarkPacingAnchorObservedSeconds = 0.0f;
    float LandmarkPacingAnchorPriorActiveSeconds = 0.0f;
    FName LandmarkPacingAnchorType = NAME_None;
};

struct FOffgridAIDroppedVisemeEvent
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    FString SourceWord;
    int32 WordIndex = INDEX_NONE;
    int32 SpeechRegionIndex = INDEX_NONE;
    // Text sentence metadata only. Runtime speech region ownership comes from
    // occupancy detection, not from this field.
    int32 SentenceIndex = INDEX_NONE;
    bool bIsStrongVisibleEvent = false;

    int32 SourcePhoneIndex = INDEX_NONE;
    FString SourcePhoneBase;
    FName SourcePhoneClass = NAME_None;

    float DropPlaybackSeconds = 0.0f;
    float RegionStartSeconds = 0.0f;
    float RegionEndSeconds = 0.0f;
    float RequiredActiveElapsedSeconds = 0.0f;
    float ObservedActiveElapsedSeconds = 0.0f;
    FName DropReason = FName(TEXT("unknown"));
};

struct FOffgridAIAlignedVisemeTrack
{
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float SpeechStartSeconds = 0.0f;
    float SpeechEndSeconds = 0.0f;
    TArray<FOffgridAIAlignedVisemeEvent> Events;
    TArray<FOffgridAIDroppedVisemeEvent> DroppedEvents;
};

struct FOffgridAIPerformedVisemeFrame
{
    float PlaybackSeconds = 0.0f;
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    TMap<FName, float> AbstractVisemeWeights;
};

struct FOffgridAISubmittedVisemeSample
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    FString SourceWord;
    float PlaybackSeconds = 0.0f;
    float CommittedRenderStartSeconds = 0.0f;
    float CommittedRenderCenterSeconds = 0.0f;
    float CommittedRenderEndSeconds = 0.0f;
    float SubmittedWeight = 0.0f;
    float SourceStrength = 0.0f;
};

struct FOffgridAILipsyncPoseFrame
{
    float TimeSeconds = 0.0f;
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    TMap<FName, float> MouthPoseWeights;
    bool bIsTransient = false;
};

struct FOffgridAILipsyncPoseRuntimeState
{
    float Open = 0.0f;
    float Closed = 0.0f;
    float Wide = 0.0f;
    float Round = 0.0f;
    float Funnel = 0.0f;
    float Teeth = 0.0f;
};
