#pragma once

#include "CoreMinimal.h"
#include "Lipsync/OffgridAILipsyncTypes.h"
#include "Lipsync/OffgridAITextVisemePlanner.h"
#include "Lipsync/OffgridAIStreamingSpeechDetector.h"

// Simplified runtime input: text plan + speech occupancy only.
// There is intentionally no phone/Viterbi/acoustic-class alignment path here.
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
    int32 DroppedEventCount = 0;
    FName CloseReason = NAME_None;
    FName DiagnosticKind = NAME_None;
};

struct FOffgridAILipsyncRuntimeBeginInput
{
    FString DialogueText;
    FName NPCID = NAME_None;
    FName LineID = NAME_None;
    float PrerollSec = 0.350f;
};

struct FOffgridAIPunctuationHoldState
{
    bool bPlayheadStarted = false;
    bool bHoldActive = false;
    bool bWaitingForSpeechResume = false;
    int32 BoundaryWordIndex = INDEX_NONE;
    int32 ResumeRegionIndex = INDEX_NONE;
    int32 HoldRegionIndex = INDEX_NONE;
    float PlaybackOriginSec = 0.0f;
    float LastPlaybackSec = 0.0f;
    float ActivePlayheadSec = 0.0f;
    float TotalPausedSec = 0.0f;
    float HoldStartPlaybackSec = 0.0f;
    float HoldDeadlinePlaybackSec = 0.0f;
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
    FOffgridAITextVisemePlan& GetMutableTextPlan() { return TextPlan; }
    const FOffgridAIStreamingSpeechDetector& GetSpeechDetector() const { return Detector; }
    const TArray<FOffgridAIStreamingSpeechRegion>& GetSpeechRegions() const { return ResolvedSpeechRegions; }
    const TArray<FOffgridAIStreamingSpeechRegion>& GetSpeechIslands() const { return ResolvedSpeechRegions; }
    const TArray<FOffgridAIStreamingAudioFeatureFrame>& GetAudioFeatureFrames() const { return Detector.GetFeatureFrames(); }
    const FOffgridAIAlignedVisemeTrack& GetCommittedTrack() const { return CommittedTrack; }
    const TArray<FOffgridAIAudioOccupancyDiagnosticRow>& GetAudioOccupancyDiagnosticRows() const { return AudioOccupancyDiagnosticRows; }
    const TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow>& GetRuntimeSpeechRegionDiagnosticRows() const { return RuntimeSpeechRegionDiagnosticRows; }
    const FOffgridAIStreamTailDiagnosticRow& GetStreamTailDiagnosticRow() const { return StreamTailDiagnosticRow; }
    FOffgridAIAlignedVisemeTrack& GetMutableCommittedTrack() { return CommittedTrack; }
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
    FOffgridAIAlignedVisemeTrack CommittedTrack;
    TArray<FOffgridAIAudioOccupancyDiagnosticRow> AudioOccupancyDiagnosticRows;
    TArray<FOffgridAIRuntimeSpeechRegionDiagnosticRow> RuntimeSpeechRegionDiagnosticRows;
    int32 AudioOccupancyDiagnosticUpdateOrdinal = 0;
    FOffgridAIStreamTailDiagnosticRow StreamTailDiagnosticRow;

    int32 PCMChunkCount = 0;
    int64 PCMBytesReceived = 0;
    int64 PCMSamplesReceived = 0;
    int32 LastPCMChunkSampleRate = 0;
    int32 LastPCMChunkChannels = 0;
    int64 LastPCMChunkStartSample = -1;
    int64 LastPCMChunkEndSample = -1;
    FOffgridAIPunctuationHoldState PunctuationHoldState;
};

class OFFGRIDAI_API FOffgridAILipsyncRuntimeAdapter
{
public:
    static void UpdateCommittedTrack(
        const FOffgridAILipsyncRuntimeUpdateInput& Input,
        FOffgridAIAlignedVisemeTrack& InOutTrack,
        FOffgridAIPunctuationHoldState& InOutHoldState,
        bool& bInOutTrackBuilt);
};
