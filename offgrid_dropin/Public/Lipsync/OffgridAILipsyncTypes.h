#pragma once

#include "CoreMinimal.h"

struct FOffgridAIStreamingSpeechRegion
{
    int32 SpeechRegionIndex = INDEX_NONE;
    float AudioBufferStartSec = 0.0f;
    float AudioBufferLastSpeechSec = 0.0f;
    float AudioBufferEndSec = 0.0f;
    bool bStarted = false;
    bool bEnded = false;
    float ProvisionalEndSec = -1.0f;
    float EndDecisionSec = -1.0f;
    int32 ReopenCount = 0;
    FName EndReason = NAME_None;
};

struct FOffgridAICommittedVisemeEvent
{
    int32 EventIndex = INDEX_NONE;
    FName PoseID = NAME_None;
    // Authoritative presentation peak chosen by the text/phoneme plan.
    // Downstream rendering may shape or duck it, but must not reinterpret it
    // with another phoneme- or pose-specific magnitude policy.
    float Strength = 0.0f;
    // Vowel-height/stress target copied from the text plan. Consonants carry
    // no independent jaw target and inherit the continuous syllabic carrier.
    float JawOpenTarget = -1.0f;
    FString SourceWord;
    int32 WordIndex = INDEX_NONE;
    int32 SpeechRegionIndex = INDEX_NONE;
    // Text sentence metadata only. Runtime speech region ownership comes from
    // detected speech regions, not from this field.
    int32 SentenceIndex = INDEX_NONE;
    // Strong visible event metadata for diagnostics/render emphasis only. It is not a scheduling landmark.
    bool bIsStrongVisibleEvent = false;
    bool bIsRenderable = true;
    // The event remains committed for transcript completeness and diagnostics,
    // but a later acoustically anchored word has authoritatively ended this
    // word before the event could be displayed.
    bool bCanceledByWordHandoff = false;

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


    bool bUsedInitialSpeechAnchor = false;
    bool bUsedResumeAnchor = false;
    FName AcousticAnchorKind = NAME_None;
    float AcousticAnchorSeconds = -1.0f;
    float AcousticAnchorErrorSeconds = 0.0f;
    float ObservedPauseDecaySeconds = -1.0f;
    float ObservedResumeOnsetSeconds = -1.0f;
    float ObservedResumeEnergyAnchorSeconds = -1.0f;

    int32 BoundaryWordIndex = INDEX_NONE;
    FString BoundaryMark;
    FName BoundaryOutcome = NAME_None;


};

struct FOffgridAICommittedVisemeTrack
{
    struct FSpeechRegion
    {
        int32 SpeechRegionIndex = INDEX_NONE;
        float StartSeconds = 0.0f;
        float EndSeconds = 0.0f;
        bool bEnded = false;
    };

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float SpeechStartSeconds = 0.0f;
    float SpeechEndSeconds = 0.0f;
    // Neural region bounds used only to gate pose envelopes. Event
    // centers and identity remain immutable once committed.
    TArray<FSpeechRegion> SpeechRegions;
    TArray<FOffgridAICommittedVisemeEvent> Events;
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

struct FOffgridAILipsyncPoseRuntimeState
{
    float Open = 0.0f;
    float Closed = 0.0f;
    float Wide = 0.0f;
    float Round = 0.0f;
    float Funnel = 0.0f;
    float Teeth = 0.0f;
};
