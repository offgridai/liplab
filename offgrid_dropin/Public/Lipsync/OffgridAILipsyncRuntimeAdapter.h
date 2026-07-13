#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

// Runtime input: transcript timing priors plus detector-owned speech/pause
// evidence. The live timing path has only two audio-driven corrections:
// 1. speech start / punctuation resume anchors, and
// 2. local syllable-envelope rebasing inside an already-open speech section.
struct FOffgridAILipsyncRuntimeUpdateInput
{
    const FOffgridAITextVisemePlan* TextPlan = nullptr;
    const TArray<FOffgridAIStreamingSpeechRegion>* SpeechRegions = nullptr;
    const TArray<FOffgridAIStreamingSpeechGapCandidate>* GapCandidates = nullptr;
    const TArray<FOffgridAIStreamingSoftLullCandidate>* SoftLullCandidates = nullptr;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames = nullptr;

    float CurrentPlaybackSec = 0.0f;
    float PrerollSec = 0.350f;
    float ObservedAudioBufferEndSec = 0.0f;
    bool bInputStreamClosed = false;
    bool bPlaybackFinalized = false;

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
};

// Per-commit diagnostics for monotonic transcript playback, optionally reset by
// detected speech start / punctuation resume anchors and locally rebased by
// syllable envelopes.
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
    float ObservedPauseDecaySec = -1.0f;
    float ObservedResumeOnsetSec = -1.0f;
    float ObservedResumeEnergyAnchorSec = -1.0f;
    int32 BoundaryWordIndex = INDEX_NONE;
    FString BoundaryMark;
    FName BoundaryOutcome = NAME_None;
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
    int32 PhoneIndex = INDEX_NONE;
    float ObservedAudioSec = 0.0f;
    float Prominence = 0.0f;
    float Confidence = 0.0f;
    float Margin = 0.0f;
    int32 SkipCount = 0;
};

struct FOffgridAIRuntimeBoundaryDiagnosticRow
{
    FName LineID = NAME_None;
    int32 UpdateOrdinal = 0;
    bool bFinalReplay = false;
    float CurrentPlaybackSec = 0.0f;

    bool bPlayheadStarted = false;
    bool bHoldActive = false;
    bool bObservedPauseLull = false;
    bool bSawConfirmedOutOfSpeechAfterBoundary = false;
    bool bResumeAnchorActive = false;
    bool bResumeAnchorFromInitialSpeech = false;

    int32 BoundaryWordIndex = INDEX_NONE;
    FString BoundaryMark;
    FString PauseClass;
    int32 HoldStartSpeechRegionIndex = INDEX_NONE;
    int32 ResumeAnchorEventIndex = INDEX_NONE;

    float PlaybackOriginSec = 0.0f;
    float LastPlaybackSec = 0.0f;
    float ActivePlayheadSec = 0.0f;
    float TotalPausedSec = 0.0f;
    float HoldStartPlaybackSec = 0.0f;
    float BoundarySearchStartPlaybackSec = 0.0f;
    float HoldDeadlinePlaybackSec = 0.0f;
    float HoldResumeTargetActiveSec = 0.0f;
    float HoldResumeTargetLeadSec = 0.0f;
    float ConfirmedQuietStartPlaybackSec = -1.0f;
    float QuietRMSNormAtDecay = 1.0f;
    float QuietEvidenceAtDecay = 1.0f;
    float QuietRawRMSAtDecay = 1.0f;
    float ObservedResumeOnsetPlaybackSec = -1.0f;
    float ObservedResumeEnergyAnchorSec = -1.0f;
    float ResumeAnchorActiveSec = 0.0f;
    float ResumeAnchorLeadSec = 0.0f;
    float ResumeAnchorFinalCenterSec = 0.0f;

    bool bSyllableRebaseActive = false;
    int32 SyllableAnchorPhoneIndex = INDEX_NONE;
    int32 NextExpectedSyllablePhoneIndex = INDEX_NONE;
    float SyllableAnchorActiveSec = 0.0f;
    float SyllableAnchorAudioSec = 0.0f;
    float SyllableRate = 1.0f;
    float SyllablePlaybackRate = 1.0f;
    float SyllableSectionStartAudioSec = 0.0f;
    float SyllableAssignmentConfidence = 0.0f;
    float SyllableAssignmentMargin = 0.0f;
    int32 SyllableAssignmentSkipCount = 0;
    float SyllableObservedAudioSec = 0.0f;
    int32 SyllablePulseCount = 0;
    int32 SyllableAssignmentCount = 0;
    int32 SyllableProgressCount = 0;
    int32 SyllableLastAssignedPhoneIndex = INDEX_NONE;
    int32 SyllableRejectLowProminenceCount = 0;
    int32 SyllableRejectBeforeSectionCount = 0;
    int32 SyllableLateAssignmentCount = 0;
    int32 SyllableRejectNoCandidateCount = 0;
    int32 SyllableAmbiguousAssignmentCount = 0;
    int32 SyllableDuplicatePulseCount = 0;
    bool bStrongPhoneRebaseActive = false;
    int32 StrongPhoneAnchorPhoneIndex = INDEX_NONE;
    float StrongPhoneAnchorActiveSec = 0.0f;
    float StrongPhoneAnchorAudioSec = 0.0f;
    float StrongPhoneAnchorConfidence = 0.0f;
    int32 StrongPhoneCandidateCount = 0;
    int32 StrongPhoneAssignmentCount = 0;
    int32 TranscriptAnchorCursorPhoneIndex = 0;
    int32 TranscriptAnchorFenceWordIndex = INDEX_NONE;
    int32 TranscriptAnchorFencePhoneIndex = INDEX_NONE;
    int32 TranscriptAnchorLastResolvedBoundaryWordIndex = INDEX_NONE;

    FString LastFenceEstimatorOutcome;
    FString LastResolvedBoundaryOutcome;
    int32 LastResolvedBoundaryWordIndex = INDEX_NONE;
    FString LastResolvedBoundaryMark;
    float LastResolvedBoundaryDecaySec = -1.0f;
    float LastResolvedBoundaryResumeOnsetSec = -1.0f;
    float LastResolvedBoundaryResumeEnergyAnchorSec = -1.0f;

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

struct FOffgridAIResolvedBoundaryDiagnostic
{
    FName Outcome = NAME_None;
    int32 WordIndex = INDEX_NONE;
    TCHAR Mark = TCHAR(0);
    float DecaySec = -1.0f;
    float ResumeOnsetSec = -1.0f;
    float ResumeEnergyAnchorSec = -1.0f;
};

struct FOffgridAIBoundaryPlaybackState
{
    bool bPlayheadStarted = false;
    bool bHoldActive = false;
    bool bObservedPauseLull = false;
    int32 BoundaryWordIndex = INDEX_NONE;
    EOffgridAIBoundaryPauseClass ActivePauseClass = EOffgridAIBoundaryPauseClass::None;
    float PlaybackOriginSec = 0.0f;
    float LastPlaybackSec = 0.0f;
    float ActivePlayheadSec = 0.0f;
    float TotalPausedSec = 0.0f;
    float HoldStartPlaybackSec = 0.0f;
    // Acoustic decay search may begin slightly before the runtime entered the hold,
    // because the waveform trough can precede the final pre-boundary viseme center.
    float BoundarySearchStartPlaybackSec = 0.0f;
    float HoldDeadlinePlaybackSec = 0.0f;
    float HoldResumeTargetActiveSec = 0.0f;
    float HoldResumeTargetLeadSec = 0.0f;
    // Punctuation is a freeze/wait/resume state, not a timer delay. Every
    // punctuation mark stops new commits, waits for decay/resume, then anchors
    // the first post-boundary viseme to resumed speech. Marks differ only by
    // patience window.
    int32 HoldStartSpeechRegionIndex = INDEX_NONE;
    bool bSawConfirmedOutOfSpeechAfterBoundary = false;
    float ConfirmedQuietStartPlaybackSec = -1.0f;
    float QuietRMSNormAtDecay = 1.0f;
    float QuietEvidenceAtDecay = 1.0f;
    float QuietRawRMSAtDecay = 1.0f;
    TCHAR ActiveBoundaryMark = TCHAR(0);
    bool bProvisionalListPause = false;
    float ProvisionalListPauseStartSec = -1.0f;
    float ProvisionalListPauseEndSec = -1.0f;
    float ProvisionalListPauseConfidence = 0.0f;
    float ProvisionalListPauseRMSNorm = 1.0f;
    float ProvisionalListPauseEvidence = 1.0f;
    float ObservedResumeOnsetPlaybackSec = -1.0f;
    float ObservedResumeEnergyAnchorSec = -1.0f;
    FOffgridAIResolvedBoundaryDiagnostic LastResolvedBoundary;
    FName LastFenceEstimatorOutcome = NAME_None;

    // Punctuation resume anchor. When a real pause/resume cycle is observed,
    // the first visible post-boundary phone is attached directly to the new
    // acoustic onset. Later phones are scheduled relative to this anchor so
    // the pause does not get absorbed as a long gap inside the first word.
    bool bInitialSpeechAnchorResolved = false;
    bool bResumeAnchorActive = false;
    bool bResumeAnchorFromInitialSpeech = false;
    int32 ResumeAnchorEventIndex = INDEX_NONE;
    float ResumeAnchorActiveSec = 0.0f;
    float ResumeAnchorLeadSec = 0.0f;
    float ResumeAnchorFinalCenterSec = 0.0f;

    // Syllable pulses rebase only the uncommitted suffix
    // inside the current punctuation-bounded speech section. Phone identity is
    // never inferred from audio; the transcript supplies the expected nucleus.
    bool bSyllableRebaseActive = false;
    int32 SyllableAnchorPhoneIndex = INDEX_NONE;
    int32 SyllableRebaseEndPhoneIndex = INDEX_NONE;
    int32 NextExpectedSyllablePhoneIndex = INDEX_NONE;
    int32 LastSyllableScanFrameIndex = INDEX_NONE;
    float SyllableAnchorActiveSec = 0.0f;
    float SyllableAnchorAudioSec = 0.0f;
    float SyllableRate = 1.0f;
    float SyllablePlaybackRate = 1.0f;
    float SyllableSectionStartAudioSec = 0.0f;
    float SyllableAssignmentConfidence = 0.0f;
    float SyllableAssignmentMargin = 0.0f;
    int32 SyllableAssignmentSkipCount = 0;
    float SyllableObservedAudioSec = 0.0f;
    int32 SyllablePulseCount = 0;
    int32 SyllableAssignmentCount = 0;
    int32 SyllableProgressCount = 0;
    int32 SyllableLastAssignedPhoneIndex = INDEX_NONE;
    int32 SyllableRejectLowProminenceCount = 0;
    int32 SyllableRejectBeforeSectionCount = 0;
    int32 SyllableLateAssignmentCount = 0;
    int32 SyllableRejectNoCandidateCount = 0;
    int32 SyllableAmbiguousAssignmentCount = 0;
    int32 SyllableDuplicatePulseCount = 0;
    float SyllableLastAssignedProminence = 0.0f;
    TArray<FOffgridAIRuntimeSyllableAssignmentDiagnosticRow> PendingSyllableAssignments;


    // A strong transcript-known phone may refine timing inside the accepted
    // syllable. It shares the syllable scheduler and expires at the next
    // syllable; audio never changes phone identity or order.
    bool bStrongPhoneRebaseActive = false;
    int32 StrongPhoneAnchorPhoneIndex = INDEX_NONE;
    float StrongPhoneAnchorActiveSec = 0.0f;
    float StrongPhoneAnchorAudioSec = 0.0f;
    float StrongPhoneAnchorConfidence = 0.0f;
    int32 StrongPhoneCandidateCount = 0;
    int32 StrongPhoneAssignmentCount = 0;

    // One monotonic transcript cursor owns syllable, strong-phone, and
    // punctuation anchor progression. Optional acoustic anchors may advance to
    // this fence, but never through unresolved punctuation.
    int32 TranscriptAnchorCursorPhoneIndex = 0;
    int32 TranscriptAnchorFenceWordIndex = INDEX_NONE;
    int32 TranscriptAnchorFencePhoneIndex = INDEX_NONE;
    int32 TranscriptAnchorLastResolvedBoundaryWordIndex = INDEX_NONE;

};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeSession
{
public:
    void Reset();
    // Starts a new lipsync line. This owns transcript planning only; no audio
    // timing exists until streamed PCM arrives.
    void BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input);
    // Appends newly arrived PCM. This advances observed-audio state only.
    void PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample = -1);
    // Signals that no more PCM will arrive. Playback may still be ongoing after
    // this; callers must continue calling Update() until audible playback ends.
    void CloseInputStream();
    // Advances the runtime against the caller-owned audible playback clock.
    // This must be called every playback tick while audio is audibly playing,
    // including after CloseInputStream() during buffered/drain playback.
    void Update(float CurrentPlaybackSec);
    // Final audible-playback drain. Call only when the line's audible playback
    // is actually complete.
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
    FOffgridAIBoundaryPlaybackState PunctuationHoldState;
};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeAdapter
{
public:
    static void UpdateCommittedTrack(
        const FOffgridAILipsyncRuntimeUpdateInput& Input,
        FOffgridAICommittedVisemeTrack& InOutTrack,
        FOffgridAIBoundaryPlaybackState& InOutHoldState,
        bool& bInOutTrackBuilt);
};
