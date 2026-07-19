#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

// Transcript identity and streamed acoustic evidence are deliberately separate.
// The transcript owns event order; audio owns region timing and optional pulse
// anchors for the uncommitted suffix.
struct FOffgridAILipsyncRuntimeUpdateInput
{
    const FOffgridAITextVisemePlan* TextPlan = nullptr;
    const TArray<FOffgridAIStreamingSpeechRegion>* SpeechRegions = nullptr;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames = nullptr;

    float CurrentPlaybackSec = 0.0f;
    float PrerollSec = 0.350f;
    float ObservedAudioBufferEndSec = 0.0f;
    bool bInputStreamClosed = false;
    bool bPlaybackFinalized = false;

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
};

struct FOffgridAIStreamTailDiagnosticRow
{
    FName LineID = NAME_None;
    int32 PCMChunkCount = 0;
    int64 PCMBytesReceived = 0;
    int64 PCMSamplesReceived = 0;
    int32 LastSampleRate = 0;
    int32 LastNumChannels = 0;
    int64 LastChunkStartSample = -1;
    int64 LastChunkEndSample = -1;
    float ObservedAudioBufferEndSec = 0.0f;
    float FirstSpeechAudioBufferStartSec = 0.0f;
    int32 SpeechRegionCount = 0;
    bool bInputStreamClosed = false;
    FName DiagnosticKind = NAME_None;
};

struct FOffgridAIRuntimeSpeechRegionDiagnosticRow
{
    FName LineID = NAME_None;
    int32 UpdateOrdinal = 0;
    bool bFinalReplay = false;
    float CurrentPlaybackSec = 0.0f;
    int32 RegionIndex = INDEX_NONE;
    float RegionOpenSec = 0.0f;
    float RegionCloseSec = 0.0f;
    float LastSpeechSec = 0.0f;
    float ProvisionalEndSec = 0.0f;
    float EndDecisionSec = 0.0f;
    int32 ReopenCount = 0;
    bool bStarted = false;
    bool bEnded = false;
    bool bContainsPlaybackSec = false;
    int32 CommittedEventCount = 0;
    FName CloseReason = NAME_None;
    FName DiagnosticKind = NAME_None;
};

struct FOffgridAIRuntimeSyllableAssignmentDiagnosticRow
{
    FName LineID = NAME_None;
    int32 UpdateOrdinal = 0;
    int32 AudioSpeechRegionIndex = INDEX_NONE;
    int32 TextSpeechRegionIndex = INDEX_NONE;
    int32 PhoneIndex = INDEX_NONE;
    float ObservedAudioSec = 0.0f;
    float Prominence = 0.0f;
    float Confidence = 0.0f;
    int32 SkipCount = 0;
    FName AnchorKind = NAME_None;
    float TimelineCorrectionSec = 0.0f;
    // Word-paced experiment diagnostics. A word's rate is immutable once it
    // starts; an observed interval only updates the rate for later words.
    int32 WordIndex = INDEX_NONE;
    float WordPriorRate = 1.0f;
    float ObservedWordIntervalSec = -1.0f;
    float PriorWordIntervalSec = -1.0f;
    int32 CanceledPriorWordEventCount = 0;
};

// Compact state trace for the single audio-primary scheduler.
struct FOffgridAIRuntimeBoundaryDiagnosticRow
{
    FName LineID = NAME_None;
    int32 UpdateOrdinal = 0;
    bool bFinalReplay = false;
    float CurrentPlaybackSec = 0.0f;
    bool bPlayheadStarted = false;
    bool bAudioSpeechActive = false;
    int32 ActiveSpeechRegionIndex = INDEX_NONE;
    int32 ActiveTextSpeechRegionIndex = INDEX_NONE;
    float ActiveRegionStartSec = -1.0f;
    float ActiveRegionEndSec = -1.0f;
    float TimelineRate = 1.0f;
    int32 LastMatchedSyllableIndex = INDEX_NONE;
    int32 LastMatchedSyllablePhoneIndex = INDEX_NONE;
    float LastMatchedSyllableAudioSec = -1.0f;
    float LastMatchedSyllableConfidence = 0.0f;
    int32 SchedulerNextEventIndex = INDEX_NONE;
    int32 SchedulerNextPhoneIndex = INDEX_NONE;
    float SchedulerCandidateCenterSec = -1.0f;
    float SchedulerCommitFrontierSec = -1.0f;
    float SchedulerCommitLeadSec = -1.0f;
    FString SchedulerBlockReason;
    int32 CommittedEventCount = 0;
    FName DiagnosticKind = NAME_None;
};

struct FOffgridAILipsyncRuntimeBeginInput
{
    FString DialogueText;
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float PrerollSec = 0.350f;
};

// Complete mutable state for the single monotonic scheduler.
struct FOffgridAILipsyncSchedulerState
{
    bool bPlayheadStarted = false;
    int32 NextTextEventIndex = 0;
    int32 ActiveSpeechRegionIndex = INDEX_NONE;
    int32 ActiveTextSpeechRegionIndex = INDEX_NONE;
    float TimelineRate = 1.0f;
    int32 LastMatchedSyllableIndex = INDEX_NONE;
    int32 LastMatchedSyllablePhoneIndex = INDEX_NONE;
    float LastMatchedSyllableAudioSec = -1.0f;
    float LastMatchedSyllableConfidence = 0.0f;
    float LastProcessedSyllablePulseSec = -1.0f;
    int32 LastCommittedWordIndex = INDEX_NONE;
    int32 LastWordAnchorAudioRegionIndex = INDEX_NONE;
    float LastWordAnchorAudioSec = -1.0f;
    float LastWordAnchorPriorSec = -1.0f;
    float AdaptiveWordPriorRate = 1.0f;
    TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow> PendingSyllableAssignments;

    int32 SchedulerNextEventIndex = INDEX_NONE;
    int32 SchedulerNextPhoneIndex = INDEX_NONE;
    float SchedulerCandidateCenterSec = -1.0f;
    float SchedulerCommitFrontierSec = -1.0f;
    float SchedulerCommitLeadSec = -1.0f;
    FName SchedulerBlockReason = NAME_None;
};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeSession
{
public:
    static const TCHAR* GetImplementationVersion();
    static int32 GetDiagnosticSchemaVersion();

    void Reset();
    void BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input);
    void PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample = -1);
    void CloseInputStream();
    void Update(float CurrentPlaybackSec);
    void Finalize(float FinalPlaybackSec);

    const FOffgridAITextVisemePlan& GetTextPlan() const { return TextPlan; }
    const FOffgridAIStreamingSpeechDetector& GetSpeechDetector() const { return Detector; }
    const TArray<FOffgridAIStreamingSpeechRegion>& GetSpeechRegions() const { return ResolvedSpeechRegions; }
    const FOffgridAICommittedVisemeTrack& GetCommittedTrack() const { return CommittedTrack; }
    const TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow>& GetRuntimeSpeechRegionDiagnosticRows() const { return RuntimeSpeechRegionDiagnosticRows; }
    const TArray<FOffgridAIRuntimeBoundaryDiagnosticRow>& GetRuntimeBoundaryDiagnosticRows() const { return RuntimeBoundaryDiagnosticRows; }
    const TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow>& GetRuntimeSyllableAssignmentDiagnosticRows() const { return RuntimeSyllableAssignmentDiagnosticRows; }
    const FOffgridAIStreamTailDiagnosticRow& GetStreamTailDiagnosticRow() const { return StreamTailDiagnosticRow; }
    FOffgridAICommittedVisemeTrack& GetMutableCommittedTrack() { return CommittedTrack; }
    bool IsCommittedTrackBuilt() const { return bCommittedTrackBuilt; }
    float GetPlaybackSeconds() const { return PlaybackSec; }

private:
    void RefreshResolvedSpeechRegions();
    void RecordRuntimeDiagnostics(float CurrentPlaybackSec, bool bFinalReplay);

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    FString DialogueText;
    float PrerollSec = 0.350f;
    float PlaybackSec = 0.0f;
    bool bBegun = false;
    bool bCommittedTrackBuilt = false;
    bool bInputStreamClosed = false;
    FOffgridAITextVisemePlan TextPlan;
    FOffgridAIStreamingSpeechDetector Detector;
    TArray<FOffgridAIStreamingSpeechRegion> ResolvedSpeechRegions;
    FOffgridAICommittedVisemeTrack CommittedTrack;
    TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow> RuntimeSpeechRegionDiagnosticRows;
    TArray<FOffgridAIRuntimeBoundaryDiagnosticRow> RuntimeBoundaryDiagnosticRows;
    TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow> RuntimeSyllableAssignmentDiagnosticRows;
    int32 DiagnosticUpdateOrdinal = 0;
    FOffgridAIStreamTailDiagnosticRow StreamTailDiagnosticRow;
    int32 PCMChunkCount = 0;
    int64 PCMBytesReceived = 0;
    int64 PCMSamplesReceived = 0;
    int32 LastPCMChunkSampleRate = 0;
    int32 LastPCMChunkChannels = 0;
    int64 LastPCMChunkStartSample = -1;
    int64 LastPCMChunkEndSample = -1;
    FOffgridAILipsyncSchedulerState PlaybackState;
};
