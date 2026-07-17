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
    // Retained for source compatibility and diagnostics. The scheduler uses
    // resolved speech regions rather than independently interpreting these.
    const TArray<FOffgridAIStreamingSpeechGapCandidate>* GapCandidates = nullptr;
    const TArray<FOffgridAIStreamingSoftLullCandidate>* SoftLullCandidates = nullptr;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames = nullptr;

    float CurrentPlaybackSec = 0.0f;
    float PrerollSec = 0.350f;
    float ObservedAudioBufferEndSec = 0.0f;
    bool bEnableFocusedWordStartAlignment = false;
    bool bInputStreamClosed = false;
    bool bPlaybackFinalized = false;

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
};

struct FOffgridAIRuntimeCommitDiagnosticRow
{
    FName LineID = NAME_None;
    int32 UpdateOrdinal = 0;
    bool bFinalReplay = false;
    float CurrentPlaybackSec = 0.0f;
    float PrerollSec = 0.0f;
    int32 SourceEventIndex = INDEX_NONE;
    FString Word;
    FName PoseID = NAME_None;
    float PlannedCenterSec = 0.0f;
    float CommittedCenterSec = 0.0f;
    float RenderStartSec = 0.0f;
    float RenderEndSec = 0.0f;
    FName CommitReason = NAME_None;
    FName PlaybackMode = NAME_None;
    bool bMappedToObservedSpeech = false;
    FName DiagnosticKind = NAME_None;
    bool bUsedInitialSpeechAnchor = false;
    bool bUsedResumeAnchor = false;
    FName AcousticAnchorKind = NAME_None;
    float AcousticAnchorSec = -1.0f;
    float AcousticAnchorErrorSec = 0.0f;
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
    float ActiveTextRegionAudioStartSec = -1.0f;
    float ActiveRegionStartSec = -1.0f;
    float ActiveRegionEndSec = -1.0f;
    float TimelinePriorAnchorSec = 0.0f;
    float TimelineAudioAnchorSec = 0.0f;
    float TimelineRate = 1.0f;
    int32 LastMatchedSyllableIndex = INDEX_NONE;
    int32 LastMatchedSyllablePhoneIndex = INDEX_NONE;
    float LastMatchedSyllableAudioSec = -1.0f;
    float LastMatchedSyllableConfidence = 0.0f;
    int32 BoundedSyllableRebaseCount = 0;
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
    bool bEnableFocusedWordStartAlignment = false;
};

// Complete mutable state for the single monotonic scheduler.
struct FOffgridAIBoundaryPlaybackState
{
    bool bPlayheadStarted = false;
    int32 NextTextEventIndex = 0;
    int32 ActiveSpeechRegionIndex = INDEX_NONE;
    int32 ActiveTextSpeechRegionIndex = INDEX_NONE;
    float ActiveTextRegionAudioStartSec = -1.0f;
    int32 LastAnchoredSpeechRegionIndex = INDEX_NONE;
    float TimelinePriorAnchorSec = 0.0f;
    float TimelineAudioAnchorSec = 0.0f;
    float TimelineRate = 1.0f;
    int32 LastMatchedSyllableIndex = INDEX_NONE;
    int32 LastMatchedSyllablePhoneIndex = INDEX_NONE;
    float LastMatchedSyllableAudioSec = -1.0f;
    float LastMatchedSyllableConfidence = 0.0f;
    int32 BoundedSyllableRebaseCount = 0;
    int32 LastAnalyzedFeatureFrameCount = 0;
    int32 PendingMatchedSyllableIndex = INDEX_NONE;
    float PendingMatchedSyllableAudioSec = -1.0f;
    int32 PendingMatchedStableUpdates = 0;
    TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow> PendingSyllableAssignments;
    float LastConsumedProsodicRestartSec = -1.0f;
    int32 LastListRestartWordIndex = INDEX_NONE;
    int32 LastResolvedListBoundaryWordIndex = INDEX_NONE;
    int32 PendingListBoundaryWordIndex = INDEX_NONE;
    float PendingListSearchStartSec = -1.0f;
    float PendingListQuietStartSec = -1.0f;
    int32 LastResolvedPauseBoundaryWordIndex = INDEX_NONE;

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
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetAudioFeatureFrames() const { return Detector.GetFeatureFrames(); }
    const FOffgridAICommittedVisemeTrack& GetCommittedTrack() const { return CommittedTrack; }
    const TArray<FOffgridAIRuntimeCommitDiagnosticRow>& GetRuntimeCommitDiagnosticRows() const { return RuntimeCommitDiagnosticRows; }
    const TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow>& GetRuntimeSpeechRegionDiagnosticRows() const { return RuntimeSpeechRegionDiagnosticRows; }
    const TArray<FOffgridAIRuntimeBoundaryDiagnosticRow>& GetRuntimeBoundaryDiagnosticRows() const { return RuntimeBoundaryDiagnosticRows; }
    const TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow>& GetRuntimeSyllableAssignmentDiagnosticRows() const { return RuntimeSyllableAssignmentDiagnosticRows; }
    const FOffgridAIStreamTailDiagnosticRow& GetStreamTailDiagnosticRow() const { return StreamTailDiagnosticRow; }
    FOffgridAICommittedVisemeTrack& GetMutableCommittedTrack() { return CommittedTrack; }
    bool IsCommittedTrackBuilt() const { return bCommittedTrackBuilt; }
    bool HasPlaybackStarted() const { return bPlaybackStarted; }
    float GetPlaybackSeconds() const { return PlaybackSec; }
    float GetPrerollSeconds() const { return PrerollSec; }

private:
    void UpdatePlaybackGate(float ObservedEndSec);
    void RefreshResolvedSpeechRegions();
    void RecordRuntimeDiagnostics(float CurrentPlaybackSec, bool bFinalReplay);

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    FString DialogueText;
    float PrerollSec = 0.350f;
    bool bEnableFocusedWordStartAlignment = false;
    float PlaybackSec = 0.0f;
    bool bBegun = false;
    bool bPlaybackStarted = false;
    bool bCommittedTrackBuilt = false;
    bool bInputStreamClosed = false;
    FOffgridAITextVisemePlan TextPlan;
    FOffgridAIStreamingSpeechDetector Detector;
    TArray<FOffgridAIStreamingSpeechRegion> ResolvedSpeechRegions;
    FOffgridAICommittedVisemeTrack CommittedTrack;
    TArray<FOffgridAIRuntimeCommitDiagnosticRow> RuntimeCommitDiagnosticRows;
    TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow> RuntimeSpeechRegionDiagnosticRows;
    TArray<FOffgridAIRuntimeBoundaryDiagnosticRow> RuntimeBoundaryDiagnosticRows;
    TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow> RuntimeSyllableAssignmentDiagnosticRows;
    int32 RuntimeCommitDiagnosticUpdateOrdinal = 0;
    FOffgridAIStreamTailDiagnosticRow StreamTailDiagnosticRow;
    int32 PCMChunkCount = 0;
    int64 PCMBytesReceived = 0;
    int64 PCMSamplesReceived = 0;
    int32 LastPCMChunkSampleRate = 0;
    int32 LastPCMChunkChannels = 0;
    int64 LastPCMChunkStartSample = -1;
    int64 LastPCMChunkEndSample = -1;
    FOffgridAIBoundaryPlaybackState PlaybackState;
};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeAdapter
{
public:
    static void UpdateCommittedTrack(
        const FOffgridAILipsyncRuntimeUpdateInput& Input,
        FOffgridAICommittedVisemeTrack& InOutTrack,
        FOffgridAIBoundaryPlaybackState& InOutState,
        bool& bInOutTrackBuilt);
};
