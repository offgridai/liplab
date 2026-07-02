#pragma once

#include "CoreMinimal.h"

struct FOffgridAIAlignedVisemeEvent
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    float Strength = 0.0f;
    FString SourceWord;
    int32 WordIndex = INDEX_NONE;
    int32 PhraseIndex = 0;
    int32 SentenceIndex = INDEX_NONE;
    // Strong visible event metadata for diagnostics/render emphasis only. It is not a scheduling landmark.
    bool bIsStrongVisibleEvent = false;

    float TextCenterNorm = 0.0f;
    float TextDiagnosticCenterSeconds = 0.0f;

    float FinalRenderCenterSeconds = 0.0f;
    float RenderStartSeconds = 0.0f;
    float RenderEndSeconds = 0.0f;

    int32 SourcePhoneIndex = INDEX_NONE;
    FString SourcePhoneBase;
    FName SourcePhoneClass = NAME_None;
    float AlignedPhoneStartSeconds = 0.0f;
    float AlignedPhoneEndSeconds = 0.0f;
    float AlignmentConfidence = 0.0f;
    float AlignmentScoreGap = 0.0f;
    float AlignmentObservedDurationSeconds = 0.0f;
    float AlignmentExpectedDurationSeconds = 0.0f;
    FName AlignmentReason = NAME_None;
    bool bMappedToObservedSpeech = false;

    float CommitPlaybackSeconds = 0.0f;
    float CommitLeadSeconds = 0.0f;
    FName CommitReason = FName(TEXT("unknown"));
    bool bCommitStableByConfidence = false;
    bool bCommitStableByBoundary = false;
    bool bCommitStableByDuration = false;
    bool bCommitStableByLead = false;
    bool bCommitStableByLag = false;
    float CommitConfidenceThreshold = 0.0f;
    float CommitAlignableLagSeconds = 0.0f;

    float RequiredActiveElapsedSeconds = 0.0f;
    float ObservedActiveElapsedSeconds = 0.0f;
    float ActiveProgressDeficitSeconds = 0.0f;
    float RequiredProgressNorm = 0.0f;
    float ObservedProgressNorm = 0.0f;
    float ActiveProgressRatio = 1.0f;
};

struct FOffgridAIAlignedVisemeTrack
{
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float SpeechStartSeconds = 0.0f;
    float SpeechEndSeconds = 0.0f;
    TArray<FOffgridAIAlignedVisemeEvent> Events;
    // Runtime-only state: when each transcript phone first became alignable in
    // the observed stream. This lets the streaming harness keep a short
    // movable suffix before freezing a phone into the committed prefix.
    TArray<float> RuntimeFirstAlignedObservedEndSeconds;
    // Runtime phone occupancy diagnostics for the full transcript phone stream,
    // not just committed visible visemes.
    TArray<float> RuntimeObservedPhoneStartSeconds;
    TArray<float> RuntimeObservedPhoneEndSeconds;
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
