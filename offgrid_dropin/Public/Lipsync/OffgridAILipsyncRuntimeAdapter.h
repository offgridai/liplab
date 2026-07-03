#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

struct FOffgridAILipsyncRuntimeUpdateInput
{
    const FOffgridAITextVisemePlan* TextPlan = nullptr;
    const TArray<FOffgridAIStreamingSpeechIsland>* SpeechIslands = nullptr;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames = nullptr;

    float CurrentPlaybackSec = 0.0f;
    float PrerollSec = 0.350f;
    float ObservedAudioBufferEndSec = 0.0f;
    bool bInputStreamClosed = false;
    bool bPlaybackFinalized = false;

    // v17 monotonic time-warp timing correction.  The known transcript/CMU
    // timeline remains the path; streaming audio contributes anchors that bend
    // a conservative canonical->observed warp for future visemes.
    float TimingPLLPlayRate = 1.0f;
    float TimingPLLConfidence = 0.0f;
    float TimingPLLTempoRate = 1.0f;
    float TimingPLLTempoConfidence = 0.0f;
    float TimingPLLPhaseRate = 1.0f;
    float TimingWarpRate = 1.0f;
    float TimingWarpConfidence = 0.0f;
    float TimingWarpAnchorCanonicalSec = 0.0f;
    float TimingWarpAnchorObservedSec = 0.0f;
    float TimingWarpErrorSec = 0.0f;

    // Optional temporal prior for diagnostics.
    // The runtime session supplies the previous tick's cursor so the
    // preroll matcher behaves like a navigator over a precomputed CMU
    // articulation stream rather than a set of competing progress/phone
    // estimators.
    bool bHasPreviousCursor = false;
    int32 PreviousCursorSentenceIndex = INDEX_NONE;
    float PreviousCursorProgress01 = 0.0f;
    float PreviousCursorPlaybackSec = 0.0f;

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
};

struct FOffgridAIAudioOccupancyDiagnosticRow
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
    float AlignedPhoneStartSec = 0.0f;
    float AlignedPhoneEndSec = 0.0f;
    float AudioActiveSec = 0.0f;
    float TextPlayheadSec = 0.0f;

    float RequiredActiveElapsedSec = 0.0f;
    float ObservedActiveElapsedSec = 0.0f;
    float ActiveProgressDeficitSec = 0.0f;
    float RequiredProgressNorm = 0.0f;
    float ObservedProgressNorm = 0.0f;
    float ActiveProgressRatio = 1.0f;
    bool bMappedToObservedSpeech = false;

    FName DiagnosticKind = NAME_None;
};


struct FOffgridAIAudioProgressMeasurementRow
{
    FName LineID = NAME_None;
    int32 UpdateOrdinal = 0;
    float CurrentPlaybackSec = 0.0f;
    float ObservedAudioEndSec = 0.0f;

    // Rolling preroll-window inference.  Every runtime tick re-evaluates the
    // configurable lookahead window [playback, playback + preroll] as a local
    // acoustic probability landscape, then matches that landscape against the
    // CMU phone plan for the active speech region.  These fields are advisory
    // diagnostics for now; they document whether the buffer is being used as
    // active lookahead rather than just latency padding.
    float PrerollWindowStartSec = 0.0f;
    float PrerollWindowEndSec = 0.0f;
    float PrerollWindowDurationSec = 0.0f;
    float PrerollWindowCoverage01 = 0.0f;
    float PrerollWindowRegionStartProgress01 = 0.0f;
    float PrerollWindowRegionEndProgress01 = 0.0f;
    float PrerollWindowPosteriorStartProgress01 = 0.0f;
    float PrerollWindowPosteriorEndProgress01 = 0.0f;
    float PrerollWindowBestPhoneProbability = 0.0f;
    float PrerollWindowMeanPhoneProbability = 0.0f;
    float PrerollWindowBestBoundaryProbability = 0.0f;
    float PrerollWindowMeanBoundaryProbability = 0.0f;
    float PrerollWindowMeanSpeechProbability = 0.0f;
    float PrerollWindowMatchConfidence = 0.0f;
    // Cursor-template match: slides the observed preroll window over the
    // normalized CMU articulation stream and reports the best local cursor
    // alignment.
    float PrerollWindowSequenceMatchScore = 0.0f;
    float PrerollWindowSequencePriorScore = 0.0f;
    float PrerollWindowSequenceScoreGap = 0.0f;
    float PrerollWindowSequenceStartProgress01 = 0.0f;
    float PrerollWindowSequenceEndProgress01 = 0.0f;
    float PrerollWindowSequenceOffsetMs = 0.0f;

    // Speech-region cursor tracker.  The hidden state is a cursor over the
    // normalized CMU articulation stream for the active speech region.  These
    // compatibility field names still say "posterior" because older harness
    // CSV consumers read them, but semantically they are cursor-distribution
    // diagnostics.
    float PrerollWindowPosteriorMeanProgress01 = 0.0f;
    float PrerollWindowPosteriorStdDevProgress01 = 0.0f;
    float PrerollWindowPosteriorPeakProgress01 = 0.0f;
    float PrerollWindowPosteriorPeakProbability = 0.0f;
    float PrerollWindowPosteriorEntropy01 = 0.0f;
    float PrerollWindowTrackedPriorProgress01 = 0.0f;
    float PrerollWindowExpectedStreamScore = 0.0f;
    int32 PrerollWindowPosteriorPeakPhoneIndex = INDEX_NONE;
    float PrerollWindowCommitSafeEndSec = 0.0f;

    int32 CurrentWordIndex = INDEX_NONE;
    FString CurrentWord;
    int32 NextWordIndex = INDEX_NONE;
    FString NextWord;
    float WordStartSec = 0.0f;
    float WordEndSec = 0.0f;
    float ExpectedDurationSec = 0.0f;
    float ElapsedInWordSec = 0.0f;
    float DurationRatio = 0.0f;
    float BoundaryTimeSec = 0.0f;
    float BoundaryConfidence = 0.0f;
    float FilteredBoundaryConfidence = 0.0f;
    float BoundaryRedHerringProbability = 0.0f;
    float DurationAdvancePrior = 0.0f;
    float TimePriorProgress01 = 0.0f;
    float AudioDensityProgress01 = 0.0f;
    float BoundaryEvidenceProgress01 = 0.0f;
    float PhoneExpectationProgress01 = 0.0f;
    float EstimatedRegionProgress01 = 0.0f;
    float EstimatorDisagreement01 = 0.0f;
    float EffectiveTimeWeight = 0.0f;
    float EffectiveDensityWeight = 0.0f;
    float EffectiveBoundaryWeight = 0.0f;
    float EffectivePhoneWeight = 0.0f;
    float TranscriptProgress = 0.0f;
    float PriorTranscriptProgress = 0.0f;
    float ProgressWordFloat = 0.0f;
    float RegionStartSec = 0.0f;
    float RegionEndSec = 0.0f;
    float RegionProgress01 = 0.0f;
    float RegionPriorProgress01 = 0.0f;
    float AnimationProgress01 = 0.0f;
    float ProgressError01 = 0.0f;
    float EstimatedVelocity01PerSec = 0.0f;
    float SuggestedPlayRate = 1.0f;
    float ProgressConfidence = 0.0f;

    // v14 timing-pivot diagnostics.  The transcript/CMU plan is treated as the
    // correct path; these fields ask whether smaller acoustic pauses can split
    // that path into better local timing regions, and whether retrospective
    // drift should bend future play rate rather than replacing timing with the
    // cursor tracker.
    int32 MicroPauseCount50ms = 0;
    int32 MicroPauseCount75ms = 0;
    int32 MicroPauseCount120ms = 0;
    float NearestMicroPauseTimeSec = 0.0f;
    float NearestMicroPauseDurationSec = 0.0f;
    float NearestMicroPauseProgress01 = 0.0f;
    float RetrospectiveDrift01 = 0.0f;
    float RetrospectiveDriftSec = 0.0f;
    float RetrospectiveDriftConfidence = 0.0f;
    float DriftSuggestedPlayRate = 1.0f;
    float FilteredPLLPlayRate = 1.0f;
    float FilteredPLLConfidence = 0.0f;
    float PLLTempoRate = 1.0f;
    float PLLTempoConfidence = 0.0f;
    float PLLPhaseRate = 1.0f;
    float TimingWarpRate = 1.0f;
    float TimingWarpConfidence = 0.0f;
    float TimingWarpAnchorCanonicalSec = 0.0f;
    float TimingWarpAnchorObservedSec = 0.0f;
    float TimingWarpErrorSec = 0.0f;

    float StillCurrentProbability = 1.0f;
    float NextWordProbability = 0.0f;
    bool bWouldAdvance = false;
    FName AdvanceReason = NAME_None;
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
    int32 SpeechIslandCount = 0;
    bool bInputStreamClosed = false;
    FName DiagnosticKind = NAME_None;
};

struct FOffgridAILipsyncRuntimeBeginInput
{
    FString DialogueText;
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float PrerollSec = 0.350f;
};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeSession
{
public:
    void Reset();
    void BeginLine(const FOffgridAILipsyncRuntimeBeginInput& Input);
    void PushAudioPCM16(const TArray<uint8>& PCMChunk, int32 BytesToUse, int32 SampleRate, int32 NumChannels, int64 ChunkStartSample = -1);
    void CloseInputStream();
    void Update(float CurrentPlaybackSec);
    void Finalize(float FinalPlaybackSec);

    const FOffgridAITextVisemePlan& GetTextPlan() const { return TextPlan; }
    const FOffgridAIStreamingSpeechDetector& GetSpeechDetector() const { return Detector; }
    const TArray<FOffgridAIStreamingSpeechIsland>& GetSpeechIslands() const { return Detector.GetIslands(); }
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetAudioFeatureFrames() const { return Detector.GetFeatureFrames(); }
    const FOffgridAIAlignedVisemeTrack& GetCommittedTrack() const { return CommittedTrack; }
    const TArray<FOffgridAIAudioOccupancyDiagnosticRow>& GetAudioOccupancyDiagnosticRows() const { return AudioOccupancyDiagnosticRows; }
    const FOffgridAIStreamTailDiagnosticRow& GetStreamTailDiagnosticRow() const { return StreamTailDiagnosticRow; }
    const TArray<FOffgridAIAudioProgressMeasurementRow>& GetAudioProgressMeasurementRows() const { return AudioProgressMeasurementRows; }
    FOffgridAIAlignedVisemeTrack& GetMutableCommittedTrack() { return CommittedTrack; }
    bool IsCommittedTrackBuilt() const { return bCommittedTrackBuilt; }
    bool HasPlaybackStarted() const { return bPlaybackStarted; }
    float GetPlaybackSeconds() const { return PlaybackSec; }
    float GetPrerollSeconds() const { return PrerollSec; }

private:
    void UpdatePlaybackGate(float ObservedEndSec);
    void RecordAlignmentDiagnostics(float CurrentPlaybackSec, bool bFinalReplay);
    void RecordAudioProgressMeasurements(float CurrentPlaybackSec);

    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    FString DialogueText;
    float PrerollSec = 0.350f;
    float PlaybackSec = 0.0f;
    float TimingPLLPlayRate = 1.0f;
    float TimingPLLConfidence = 0.0f;
    float TimingPLLTempoRate = 1.0f;
    float TimingPLLTempoConfidence = 0.0f;
    float TimingPLLPhaseRate = 1.0f;
    float TimingWarpRate = 1.0f;
    float TimingWarpConfidence = 0.0f;
    float TimingWarpAnchorCanonicalSec = 0.0f;
    float TimingWarpAnchorObservedSec = 0.0f;
    float TimingWarpErrorSec = 0.0f;
    bool bBegun = false;
    bool bPlaybackStarted = false;
    bool bCommittedTrackBuilt = false;
    bool bInputStreamClosed = false;

    FOffgridAITextVisemePlan TextPlan;
    FOffgridAIStreamingSpeechDetector Detector;
    FOffgridAIAlignedVisemeTrack CommittedTrack;
    TArray<FOffgridAIAudioOccupancyDiagnosticRow> AudioOccupancyDiagnosticRows;
    int32 AudioOccupancyDiagnosticUpdateOrdinal = 0;
    TArray<FOffgridAIAudioProgressMeasurementRow> AudioProgressMeasurementRows;
    FOffgridAIStreamTailDiagnosticRow StreamTailDiagnosticRow;

    bool bHaveLastCursor = false;
    int32 LastCursorSentenceIndex = INDEX_NONE;
    float LastCursorProgress01 = 0.0f;
    float LastCursorPlaybackSec = 0.0f;

    int32 PCMChunkCount = 0;
    int64 PCMBytesReceived = 0;
    int64 PCMSamplesReceived = 0;
    int32 LastPCMChunkSampleRate = 0;
    int32 LastPCMChunkChannels = 0;
    int64 LastPCMChunkStartSample = -1;
    int64 LastPCMChunkEndSample = -1;
};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeAdapter
{
public:
    static void UpdateCommittedTrack(
        const FOffgridAILipsyncRuntimeUpdateInput& Input,
        FOffgridAIAlignedVisemeTrack& InOutTrack,
        bool& bInOutTrackBuilt);
};
